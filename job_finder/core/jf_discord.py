import os
import logging
import requests

from pathlib import Path
from dotenv import load_dotenv
from job_finder.core.jf_job_structure import Job


logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
load_dotenv(BASE / ".env")

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


def notify_job(job: Job) -> bool:
    """Format and send a single accepted job alert to the channel matching
    its priority ('primary' or 'secondary'). Returns whether the post
    succeeded, so the caller can decide whether to call mark_alerted()."""
    if job.priority == "rejected":
        logger.error("notify_job called with a rejected job: %s", job.id)
        return False

    salary = job.salary_text or "salary not listed"
    if job.is_salary_estimated:
        salary += " [estimated]"

    message = (
        f"**{job.title}** at {job.company}\n"
        f"{salary}\n"
        f"{job.url}"
    )
    return _post(job.priority, message)


def notify_rejected(job: Job) -> bool:
    """One compact line per rejected job, to #rejected. The <url> form
    stops Discord from expanding a big preview card."""
    tag = f"{job.fit_score}/10" if job.fit_score is not None else "rule"
    reason = (job.reason or "")[:150]
    message = (
        f"Rejected {tag} · {job.title} at {job.company} · {reason}\n"
        f"<{job.url}>"
    )
    return _post("rejected", message)


def notify_heartbeat(message: str) -> bool:
    """Send a plain status message to #heartbeat, e.g. 'N matched,
    A accepted, R rejected' after a run."""
    return _post("heartbeat", message)
