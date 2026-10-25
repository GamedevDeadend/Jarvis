import logging
import requests
import re

from job_finder.core.jf_job_structure import Job
from job_finder.core.jf_paths import BASE
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger(__name__)

load_dotenv(BASE / ".env")

API_URL = "https://remoteok.com/api"
HEADERS = {"User-Agent": "JobRadar/0.1 (personal job alert)"}

ATS_DOMAINS = ["greenhouse.io", "lever.co", "ashbyhq.com", "workable.com"]


def _clean(html: str, limit: int = 10000) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _priority(apply_url: str) -> str:
    return "secondary" if any(d in (apply_url or "") for d in ATS_DOMAINS) else "primary"


def fetch_remoteok() -> list[Job]:
    resp = requests.get(API_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    items = resp.json()[1:]  # first element is a legal notice, not a job

    jobs = []
    for raw in items:
        smin, smax = raw.get("salary_min") or 0, raw.get("salary_max") or 0
        stated = smax > 0
        url = raw.get("apply_url") or raw.get("url", "")
        jobs.append(Job(
            id=f"remoteok:{raw['id']}",
            source="remoteok",
            title=raw.get("position", ""),
            company=raw.get("company", ""),
            url=url,
            salary_text=f"${smin:,}-${smax:,}/yr" if stated else "not stated",
            priority=_priority(url),
            description=_clean(raw.get("description")),
            location=raw.get("location") or "",
        ))
    logger.info("remoteok: %d jobs fetched", len(jobs))
    return jobs