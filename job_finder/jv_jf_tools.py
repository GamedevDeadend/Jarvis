"""
jv_jf_tools.py

Everything in JobRadar that touches the outside world, in one file:
  - STORE     SQLite (jobs.db): dedup, save, mark alerted
  - NOTIFIER  Discord webhooks: primary, secondary, heartbeat, rejected
  - SOURCES   job fetchers (RemoteOK for now)

Later sections (LLM, PROFILE, RULES, JUDGE, GATE) and Phase B @tool
functions are added below. This file imports nothing from the project.

All sqlite3 usage stays inside the STORE section on purpose: if the store
is swapped for something else later, nothing outside that section changes.

Webhook URLs come from job_finder/.env (gitignored):
    JF_WEBHOOK_PRIMARY, JF_WEBHOOK_SECONDARY,
    JF_WEBHOOK_HEARTBEAT, JF_WEBHOOK_REJECTED
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, time, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader

logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
load_dotenv(BASE / ".env")


# ============================================================================
# STORE
# ============================================================================

DB_PATH = BASE / "jobs.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    # row_factory lets us read rows by name (row["title"]) instead of by
    # position (row[1]), and convert with dict(row).
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the jobs table if it doesn't already exist. Safe to call
    on every startup."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id                TEXT PRIMARY KEY,
                source            TEXT NOT NULL,
                title             TEXT NOT NULL,
                company           TEXT NOT NULL,
                url               TEXT NOT NULL,
                salary_text       TEXT,
                salary_estimated  INTEGER NOT NULL DEFAULT 0,
                priority          TEXT NOT NULL DEFAULT 'secondary',
                decision          TEXT NOT NULL DEFAULT 'accepted',
                fit_score         INTEGER,
                reason            TEXT,
                first_seen        TEXT NOT NULL,
                alerted           INTEGER NOT NULL DEFAULT 0
            )
            """
        )
    logger.info("jobs table ready at %s", DB_PATH)


def is_new(job_id: str) -> bool:
    """Return True if this job_id has never been stored before
    (accepted or rejected)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    return row is None


def save_job(job: dict) -> None:
    """Insert a new job row. Expects a dict with keys:
    id, source, title, company, url, and optionally salary_text,
    salary_estimated, priority, decision, fit_score, reason.

    Does NOT mark it as alerted -- call mark_alerted() only after the
    Discord post actually succeeds. Saving first and alerting second means
    a failed post can be retried on the next run, since the job is already
    stored but alerted=0.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO jobs
                (id, source, title, company, url,
                 salary_text, salary_estimated, priority,
                 decision, fit_score, reason,
                 first_seen, alerted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                job["id"],
                job["source"],
                job["title"],
                job["company"],
                job["url"],
                job.get("salary_text"),
                int(job.get("salary_estimated", 0)),
                job.get("priority", "secondary"),
                job.get("decision", "accepted"),
                job.get("fit_score"),
                job.get("reason"),
                now,
            ),
        )
    logger.info("saved job %s (%s at %s)", job["id"], job["title"], job["company"])


def mark_alerted(job_id: str) -> None:
    """Flag a job as alerted, after the Discord post for it succeeds."""
    with _connect() as conn:
        conn.execute("UPDATE jobs SET alerted = 1 WHERE id = ?", (job_id,))
    logger.info("marked alerted: %s", job_id)


def get_unalerted() -> list[dict]:
    """Return all stored jobs that haven't been alerted yet, e.g. to
    retry a Discord post that failed on a previous run."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM jobs WHERE alerted = 0").fetchall()
    return [dict(row) for row in rows]


# ============================================================================
# NOTIFIER
# ============================================================================

WEBHOOKS = {
    "primary": os.environ.get("JF_WEBHOOK_PRIMARY"),
    "secondary": os.environ.get("JF_WEBHOOK_SECONDARY"),
    "heartbeat": os.environ.get("JF_WEBHOOK_HEARTBEAT"),
    "rejected": os.environ.get("JF_WEBHOOK_REJECTED"),
}


def _post(channel: str, content: str) -> bool:
    """POST a message to the given channel's webhook. Returns True on
    success, False on failure (never raises -- a Discord outage should
    not crash the pipeline)."""
    url = WEBHOOKS.get(channel)
    if not url:
        logger.error("no webhook URL configured for channel %r", channel)
        return False

    try:
        resp = requests.post(url, json={"content": content}, timeout=10)
        # Discord webhooks return 204 No Content on success.
        if resp.status_code == 204:
            return True
        logger.error(
            "discord webhook for %s failed: HTTP %s %s",
            channel, resp.status_code, resp.text[:200],
        )
        return False
    except requests.RequestException as exc:
        logger.error("discord webhook for %s raised: %s", channel, exc)
        return False


def notify_job(job: dict) -> bool:
    """Format and send a single job alert to the channel matching its
    priority ('primary' or 'secondary'). Returns whether the post
    succeeded, so the caller can decide whether to call mark_alerted()."""
    channel = job.get("priority", "secondary")
    if channel not in ("primary", "secondary"):
        channel = "secondary"

    salary = job.get("salary_text") or "salary not listed"
    if job.get("salary_estimated"):
        salary += " [est]"

    message = (
        f"**{job['title']}** at {job['company']}\n"
        f"{salary}\n"
        f"{job['url']}"
    )
    return _post(channel, message)


def notify_rejected(job: dict) -> bool:
    """One compact line per rejected job, to #rejected. The <url> form
    stops Discord from expanding a big preview card."""
    score = job.get("fit_score")
    tag = f"{score}/10" if score is not None else "rule"
    reason = (job.get("reason") or "")[:150]
    message = (
        f"Rejected {tag} · {job['title']} at {job['company']} · {reason}\n"
        f"<{job['url']}>"
    )
    return _post("rejected", message)


def notify_heartbeat(message: str) -> bool:
    """Send a plain status message to #heartbeat, e.g. 'N matched,
    A accepted, R rejected' after a run."""
    return _post("heartbeat", message)


# ============================================================================
# SOURCES
# ============================================================================

API_URL = "https://remoteok.com/api"
HEADERS = {"User-Agent": "JobRadar/0.1 (personal job alert)"}

# Role fit: match against title + tags (loose on purpose; rules + LLM tighten it later)
ROLE_KEYWORDS = ["ai", "llm", "langchain", "langgraph", "mcp", "rag",
                 "machine learning", "python", "fastapi", "agent", "genai"]

# ₹2L/month is roughly $24K/year
SALARY_FLOOR_USD_YEAR = 24_000

ATS_DOMAINS = ["greenhouse.io", "lever.co", "ashbyhq.com", "workable.com"]


def _clean(html: str, limit: int = 2500) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _location_ok(location: str) -> bool:
    loc = (location or "").strip().lower()
    return loc == "" or any(k in loc for k in ("worldwide", "anywhere", "india", "global"))


def _role_ok(raw: dict) -> bool:
    text = (raw.get("position", "") + " " + " ".join(raw.get("tags", []))).lower()
    return any(k in text for k in ROLE_KEYWORDS)


def _salary_ok(raw: dict) -> bool:
    top = raw.get("salary_max") or 0
    if top == 0:
        return True  # unstated: keep, estimation comes in the judge step
    return top >= SALARY_FLOOR_USD_YEAR


def _priority(apply_url: str) -> str:
    return "secondary" if any(d in (apply_url or "") for d in ATS_DOMAINS) else "primary"


def fetch_remoteok() -> list[dict]:
    resp = requests.get(API_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    items = resp.json()[1:]  # first element is a legal notice, not a job

    jobs = []
    for raw in items:
        if not (_location_ok(raw.get("location")) and _role_ok(raw) and _salary_ok(raw)):
            continue
        smin, smax = raw.get("salary_min") or 0, raw.get("salary_max") or 0
        stated = smax > 0
        url = raw.get("apply_url") or raw.get("url", "")
        jobs.append({
            "id": f"remoteok:{raw['id']}",
            "source": "remoteok",
            "title": raw.get("position", ""),
            "company": raw.get("company", ""),
            "url": url,
            "salary_text": f"${smin:,}-${smax:,}/yr" if stated else "not stated",
            "salary_estimated": 0,
            "priority": _priority(url),
            # used by the judge only; save_job ignores it (no column)
            "description": _clean(raw.get("description")),
        })
    logger.info("remoteok: %d jobs passed filters", len(jobs))
    return jobs


def rules(jobs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Free keyword filter. Returns (survivors, rejected). Rejected jobs
    carry decision='rejected', fit_score=None and a reason, ready for
    save_job() and notify_rejected()."""
    cfg = json.loads(RULES_JSON.read_text(encoding="utf-8"))
    survivors, rejected = [], []

    for job in jobs:
        
        title = job["title"].lower()
        text = f"{title} {job.get('description', '')}".lower()

        asked = max([int(n) for n in re.findall(
      r"(\d{1,2})(?:\s*-\s*\d{1,2})?\s*\+?\s*(?:years?|yrs?)", text)], default=0)
        
        skill_hits = sum(bool(re.search(rf"\b{re.escape(s)}\b", text))
                         for s in cfg["skills"])

        if asked > cfg["years_of_experience"]:
            reason = f"rule: asks {asked}+ years, you have {cfg['years_of_experience']}"
            
        elif any(re.search(rf"\b{re.escape(r)}\b", title) for r in cfg["target_roles"]):
            reason = ""   # a target role is in the title
            
        elif skill_hits >= 2:
            reason = ""   # no role match, but 2+ of your skills appear
        else:
            reason = "rule: no matching role or skills"

        if reason:
            rejected.append({**job, "decision": "rejected",
                             "fit_score": None, "reason": reason})
        else:
            survivors.append(job)

    logger.info("rules: %d passed, %d rejected", len(survivors), len(rejected))
    return survivors, rejected


# ============================================================================
# LLM
# ============================================================================

MODEL_ID = "openai/gpt-oss-120b"   # one-time call, so the stronger model


def get_llm(model: str, **kwargs) -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"GROQ_API_KEY not set. Add it to {BASE / '.env'} "
            "as GROQ_API_KEY=gsk_... (no quotes, no spaces)."
        )
    return ChatGroq(model=model, temperature=0, max_retries=3,
                    api_key=api_key, **kwargs)


# ============================================================================
# PROFILE
# ============================================================================

RES_DIR = BASE / "resources"
PROFILE_MD = RES_DIR / "jv_resume_profile.md"
RULES_JSON = RES_DIR / "jv_rules.json"
HASH_FILE = RES_DIR / ".resume_hash"

MAX_TRIES = 3


class Profile(BaseModel):
    summary: str
    years: int
    target_roles: list[str]
    skills: list[str]

PROFILE_SYSTEM = """Extract a job-search profile from a resume.
Field guide:
- summary: two sentences.

- years: whole years of professional work. If the resume states it (for
      example "4+ years"), use that number. Otherwise count from the earliest
      job start to the latest job end, counting overlapping jobs once. Never
      add up each job's duration, and ignore education.
      
- target_roles: roles the candidate is aiming for.
- skills : skills of candidate from resume.
All titles and terms are lowercase."""


def _resume_text(path: str) -> str:
    return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)


def _write_profile_files(p: Profile) -> None:
    RES_DIR.mkdir(exist_ok=True)
    bullets = lambda xs: "\n".join(f"- {x}" for x in xs)
    PROFILE_MD.write_text(
        f"""# Candidate profile
{p.summary}

Experience: {p.years} yrs professional work.

## Target roles
{bullets(p.target_roles)}

## All skills
{bullets(p.skills)}
""",
        encoding="utf-8",
    )
    
    RULES_JSON.write_text(
        json.dumps(
            {
                "skills": p.skills,
                "target_roles": p.target_roles,
                "years_of_experience": p.years,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_profile(resume_path: str) -> None:
    """One ChatGroq call with structured output, then semantic validation.
    On any failure the error is fed back and the call is retried.
    Raises RuntimeError if all tries fail; nothing is written then."""
    resume = _resume_text(resume_path)
    llm = get_llm(MODEL_ID).with_structured_output(Profile)
    error = ""
    for attempt in range(1, MAX_TRIES + 1):
        user = resume
        if error:
            user += f"\n\nYour previous answer was rejected: {error}\nFix it."
            
        try:
            profile = llm.invoke([("system", PROFILE_SYSTEM), ("user", user)])
            
            if not profile.skills or not profile.target_roles:
                error = "skills and target_roles must be non-empty"
            else:
                _write_profile_files(profile)
                logger.info("profile built on attempt %d", attempt)
                return
                
        except Exception as exc:
            profile, error = None, f"{type(exc).__name__}: {str(exc)[:500]}"
        
        logger.warning("profile attempt %d rejected: %s", attempt, error)
    raise RuntimeError(f"profile build failed after {MAX_TRIES} tries: {error}")


def ensure_profile(resume_path: str) -> bool:
    """Rebuild only if the resume changed. Returns True if rebuilt."""
    digest = hashlib.sha256(Path(resume_path).read_bytes()).hexdigest()
    
    if (HASH_FILE.exists() and HASH_FILE.read_text() == digest
            and PROFILE_MD.exists() and RULES_JSON.exists()):
        return False
    
    build_profile(resume_path)
    HASH_FILE.write_text(digest)
    return True


# ============================================================================
# RULES / JUDGE / GATE  (added in the next steps)
# ============================================================================

JOB_DESCRIPTION_CHARS = 1500
PAUSE_BETWEEN_JUDGE_CALLS = 45 #(avoid rate limits; 30-60s is typical for Groq)
MAX_JUDGE_CALLS  = 10
JUDGE_BATCH_SIZE = 6

JUDGE_SYSTEM_PROMPT = """""
You screen remote jobs for one candidate.
 
Candidate profile:
{profile}

Additional requirements:
{requirements}
 
For EACH job return:
- id: copy it exactly
- fit_score: integer 0-10 (7 or more means worth applying)
- reason: at most 25 words
- salary_est_inr_month: monthly pay in INR if the text states or implies it
  (convert currencies, divide yearly pay by 12), otherwise null
"""

class Verdict(BaseModel):
    id: str
    fit_score: int = Field(ge=0, le=10)
    reason: str = Field(max_length=300)
    salary_est_inr_month: int | None = Field(default=None, ge=0)
    
class Verdicts(BaseModel):
    verdicts: list[Verdict]
    
    
_KEY_SENTENCE = re.compile(
    r"years?|yrs?|salary|compensation|\$|€|£|₹|usd|aud|eur|gbp|inr|"
    r"located|based in|resident|citizen|time ?zone|visa|must", re.I)
 
 
def _snippet(text: str, limit: int = JOB_DESCRIPTION_CHARS) -> str:
    """Start of the description plus later sentences that mention years,
    pay, location or 'must'. Requirements usually sit near the end."""
    head, rest = text[:900], text[900:]
    sentences = re.split(r"(?<=[.!?])\s+", rest)
    extra = " ".join(s for s in sentences if _KEY_SENTENCE.search(s))
    return f"{head} ... {extra}"[:limit] if extra else head


def judge_batch(jobs: list[dict], max_calls: int = MAX_JUDGE_CALLS) -> list[dict]:
    """Scores jobs with Groq, JUDGE_BATCH_SIZE per call. Returns the judged
    jobs with fit_score, reason and salary_est_inr_month added.
 
    A job that was not judged (call cap reached, API error, or missing from
    the answer) is simply left out of the result. It is not saved either, so
    dedup does not drop it and the next cycle tries it again."""
    if not jobs:
        return []
 
    system = JUDGE_SYSTEM_PROMPT.format(profile=PROFILE_MD.read_text(encoding="utf-8"))
    llm = get_llm(MODEL_ID).with_structured_output(Verdicts)
 
    batches = [jobs[i: (i + JUDGE_BATCH_SIZE)]
               for i in range(0, len(jobs), JUDGE_BATCH_SIZE)][:max_calls]
    judged = []
 
    for n, batch in enumerate(batches, start=1):
        if n > 1:
            time.sleep(PAUSE_BETWEEN_JUDGE_CALLS)  # avoid rate limits
 
        payload = "\n\n".join(
            f"id: {j['id']}\ntitle: {j['title']}\ncompany: {j['company']}\n"
            f"salary field: {j['salary_text']}\n"
            f"description: {_snippet(j.get('description', ''))}"
            for j in batch
        )
        try:
            result = llm.invoke([("system", system), ("user", payload)])
        except Exception as exc:
            logger.warning("judge call %d/%d failed: %s", n, len(batches), exc)
            continue
 
        by_id = {v.id: v for v in result.verdicts}
        for job in batch:
            v = by_id.get(job["id"])
            if v is None:
                logger.warning("judge gave no verdict for %s", job["id"])
                continue
            judged.append({
                **job,
                "fit_score": max(0, min(10, v.fit_score)),
                "reason": v.reason,
                "salary_est_inr_month": v.salary_est_inr_month,
            })
 
    logger.info("judge: %d of %d jobs judged in %d calls",
                len(judged), len(jobs), len(batches))
    return judged
    
        
    



if __name__ == "__main__":
    # Build the profile by hand:
    #   python -m job_finder.jv_jf_tools job_finder/resources/resume.pdf
    import sys
    logging.basicConfig(level=logging.INFO)
    rebuilt = ensure_profile(sys.argv[1])
    print("rebuilt, now REVIEW both files in resources/" if rebuilt
          else "resume unchanged, nothing to do")