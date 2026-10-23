import json
import logging
import re

from pathlib import Path
from job_finder.core.jf_job_structure import Job
from job_finder.core.jf_paths import RULES_JSON

logger = logging.getLogger(__name__)


def keyword_filter(jobs: list[Job]) -> tuple[list[Job], list[Job]]:
    """Free keyword filter (stage 1, no LLM). Returns (survivors, rejected).
    Rejected jobs carry decision='rejected', priority='rejected',
    fit_score=None and a reason, ready for save_job() and notify_rejected()."""
    cfg = json.loads(RULES_JSON.read_text(encoding="utf-8"))
    survivors, rejected = [], []

    for job in jobs:
        title = job.title.lower()
        text = f"{title} {job.description}".lower()

        asked = max(
            [int(n) for n in re.findall(
                r"(\d{1,2})(?:\s*-\s*\d{1,2})?\s*\+?\s*(?:years?|yrs?)", text)],
            default=0,
        )

        skill_hits = sum(
            bool(re.search(rf"(?<!\w){re.escape(s)}(?!\w)", text)) for s in cfg["skills"]
        )

        if asked > cfg["years_of_experience"]:
            reason = f"rule: asks {asked}+ years, you have {cfg['years_of_experience']}"
        elif any(re.search(rf"(?<!\w){re.escape(r)}(?!\w)", title) for r in cfg["target_roles"]):
            reason = ""   # a target role is in the title
        elif skill_hits >= 2:
            reason = ""   # no role match, but 2+ of your skills appear
        else:
            reason = "rule: no matching role or skills"

        if reason:
            rejected.append(job.model_copy(update={
                "decision": "rejected",
                "priority": "rejected",
                "fit_score": None,
                "reason": reason,
            }))
        else:
            survivors.append(job)

    logger.info("keyword_filter: %d passed, %d rejected", len(survivors), len(rejected))
    return survivors, rejected