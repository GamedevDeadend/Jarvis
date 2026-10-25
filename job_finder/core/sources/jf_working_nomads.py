import logging
import re
import html, re

import requests

from job_finder.core.jf_job_structure import Job
from job_finder.core.jf_paths import RULES_JSON

logger = logging.getLogger(__name__)

API_URL = "https://www.workingnomads.com/api/exposed_jobs/"
HEADERS = {"User-Agent": "JobRadar/0.1 (personal job alert)"}
_ID_REGEX = re.compile(r"/job/go/(\d+)")

def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</h\d>", "\n", raw)  # keep line breaks
    raw = re.sub(r"<[^>]+>", "", raw)                            # drop all tags
    raw = html.unescape(raw)                                     # &amp; -> &
    raw = re.sub(r"[ \t\u00a0]+", " ", raw)
    raw = re.sub(r"\n\s*\n+", "\n", raw)
    return raw.strip()


def fetch_working_nomads() -> list[Job]:
    resp = requests.get(API_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    items = resp.json()

    jobs = []
    for raw in items:
        url = raw.get("url", "")
        id = _ID_REGEX.search(url).group(1)
        company  = (raw.get("company_name") or "").strip()
        
        if not id:
            skipped += 1
            continue
        
        
        jobs.append(Job(
            id=f"workingnomads:{id}",
            source="workingnomads",
            title=raw.get("position", ""),
            company=company,
            url=url,
            salary_text=f"not mentioned",
            priority="secondary",
            description=html_to_text(raw.get("description", "")),
            location=raw.get("location") or "",
        ))
        
    logger.info("workingnomads: %d jobs fetched, %d skipped", len(jobs), skipped)
    return jobs