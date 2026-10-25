import logging
import time

from langgraph.graph import END

from job_finder.core.jf_database import get_unalerted, is_new, mark_alerted, save_job
from job_finder.core.jf_discord import notify_heartbeat, notify_job, notify_rejected
from job_finder.core.jf_job_structure import Job
from job_finder.core.jf_judge import gate, judge_batch
from job_finder.core.jf_keyword_filter import keyword_filter
from job_finder.core.jf_paths import RESUME_PDF
from job_finder.core.jf_profile_builder import ensure_profile

from job_finder.core.sources.jf_remote_ok import fetch_remoteok
from job_finder.core.sources.jf_himalayas import fetch_himalayas
from job_finder.core.sources.jf_working_nomads import fetch_working_nomads
from job_finder.core.sources.jf_we_work_remotely import fetch_we_work_remotely


from job_finder.jf_state import AgentState

logger = logging.getLogger(__name__)

SOURCES = [fetch_remoteok, fetch_himalayas, fetch_working_nomads, fetch_we_work_remotely]      # add Remotive, WorkingNomads, ... here later
DISCORD_PAUSE = 0.5             # seconds between posts; webhooks rate-limit bursts


def _stats(state: AgentState, **new) -> dict:
    return {**state.get("stats", {}), **new}


def ensure_profile_node(state: AgentState) -> dict:
    try:
        rebuilt = ensure_profile(str(RESUME_PDF))
        logger.info("profile %s", "rebuilt" if rebuilt else "unchanged")
        return {}
    except Exception as exc:
        logger.error("profile build failed: %s", exc)
        notify_heartbeat(f"JobRadar stopped: profile build failed: {exc}")
        return {"abort": True}


def route_after_profile(state: AgentState) -> str:
    return END if state.get("abort") else "fetch_jobs"


def fetch_jobs_node(state: AgentState) -> dict:
    jobs, failed = [], []
    for source in SOURCES:
        try:
            jobs.extend(source())
        except Exception as exc:
            logger.warning("source %s failed: %s", source.__name__, exc)
            failed.append(source.__name__)
    return {"jobs": jobs,
            "stats": _stats(state, fetched=len(jobs), failed_sources=failed)}


def dedup_node(state: AgentState) -> dict:
    """Drop ids already in the DB (accepted or rejected) and duplicate ids
    inside this fetch, which would otherwise break save_job."""
    seen, new = set(), []
    for job in state.get("jobs", []):
        if job.id in seen or not is_new(job.id):
            continue
        seen.add(job.id)
        new.append(job)
    return {"new_jobs": new, "stats": _stats(state, new=len(new))}


def keyword_filter_node(state: AgentState) -> dict:
    survivors, rejected = keyword_filter(state.get("new_jobs", []))
    return {"survivors": survivors, "rejected": rejected,
            "stats": _stats(state, rule_rejected=len(rejected))}


def route_after_keyword_filter(state: AgentState) -> str:
    return "judge" if state.get("survivors") else "persist_alert"


def judge_node(state: AgentState) -> dict:
    survivors = state.get("survivors", [])
    judged = judge_batch(survivors)
    return {"judged": judged,
            "stats": _stats(state, judged=len(judged),
                            unjudged=len(survivors) - len(judged))}


def gate_node(state: AgentState) -> dict:
    accepted, rejected = gate(state.get("judged", []))
    return {"accepted": accepted, "rejected": rejected,
            "stats": _stats(state, accepted=len(accepted),
                            gate_rejected=len(rejected))}


def _deliver(job: Job) -> bool:
    """Post one stored job to the right channel."""
    ok = notify_rejected(job) if job.decision == "rejected" else notify_job(job)
    time.sleep(DISCORD_PAUSE)
    return ok


def persist_alert_node(state: AgentState) -> dict:
    accepted = state.get("accepted", [])
    rejected = state.get("rejected", [])

    if state.get("dry_run"):
        for job in accepted + rejected:
            print(f"{job.decision:8} score={job.fit_score} | {job.title} at "
                  f"{job.company} | {job.salary_text} | {job.reason}")
        return {"stats": _stats(state, retried=0, post_failures=0)}

    retried = failures = 0

    # 1. retry posts that failed on earlier cycles (before saving new jobs,
    #    so a job saved in this cycle is never posted twice)
    for job in get_unalerted():
        if _deliver(job):
            mark_alerted(job.id)
            retried += 1
        else:
            failures += 1

    # 2. save and post this cycle's jobs; alerted only if the post worked
    for job in accepted + rejected:
        save_job(job)
        if _deliver(job):
            mark_alerted(job.id)
        else:
            failures += 1

    return {"stats": _stats(state, retried=retried, post_failures=failures)}


def heartbeat_node(state: AgentState) -> dict:
    s = state.get("stats", {})
    message = (
        f"JobRadar cycle: fetched {s.get('fetched', 0)}, new {s.get('new', 0)}, "
        f"rule-rejected {s.get('rule_rejected', 0)}, judged {s.get('judged', 0)}, "
        f"accepted {s.get('accepted', 0)}, gate-rejected {s.get('gate_rejected', 0)}, "
        f"unjudged {s.get('unjudged', 0)}, retried {s.get('retried', 0)}, "
        f"post failures {s.get('post_failures', 0)}"
    )
    if s.get("failed_sources"):
        message += f", failed sources: {', '.join(s['failed_sources'])}"
    if state.get("dry_run"):
        print(message)
    else:
        notify_heartbeat(message)
    return {}