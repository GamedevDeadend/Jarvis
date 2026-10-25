import json
import logging
import re
import time

import requests

from job_finder.core.jf_job_structure import Job
from job_finder.core.jf_paths import RULES_JSON

logger = logging.getLogger(__name__)

SEARCH_URL = "https://himalayas.app/jobs/api/search"
HEADERS = {"User-Agent": "JobRadar/0.1 (personal job alert)"}

MAX_QUERIES = 6                # cap on search terms taken from target_roles
PAGES_PER_QUERY = 2            # 20 jobs per page
PAUSE_BETWEEN_REQUESTS = 1.0   # seconds; the public allowance is shared


def _queries() -> list[str]:
    """Search terms from the resume's target roles (jv_rules.json)."""
    cfg = json.loads(RULES_JSON.read_text(encoding="utf-8"))
    seen, queries = set(), []
    for role in cfg.get("target_roles", []):
        role = role.strip().lower()
        if role and role not in seen:
            seen.add(role)
            queries.append(role)
    if not queries:
        raise RuntimeError("himalayas: no target_roles in jv_rules.json")
    return queries[:MAX_QUERIES]

def _clean(html: str, limit: int = 10000) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _salary_text(raw: dict) -> str:
    """e.g. 'USD 153,000-186,000 annual' or 'USD 60-120 hourly'.
    'not stated' when there is no amount or no currency."""
    lo, hi = raw.get("minSalary"), raw.get("maxSalary")
    currency = raw.get("currency")
    period = raw.get("salaryPeriod") or "annual"
    if not currency or not (lo or hi):
        return "not stated"
    lo, hi = lo or hi, hi or lo
    amount = f"{lo:,}" if lo == hi else f"{lo:,}-{hi:,}"
    return f"{currency} {amount} {period}"


def _to_job(raw: dict) -> Job:
    header = []
    if raw.get("employmentType"):
        header.append(f"Employment type: {raw['employmentType']}.")
    if raw.get("seniority"):
        header.append(f"Seniority: {', '.join(raw['seniority'])}.")
    description = " ".join(header + [_clean(raw.get("description"))])

    return Job(
        id=f"himalayas:{raw['guid']}",
        source="himalayas",
        title=raw.get("title", ""),
        company=raw.get("companyName", ""),
        url=raw["applicationLink"],          # unchanged, as the API terms require
        salary_text=_salary_text(raw),
        priority="secondary",                # apply links go via Himalayas to an ATS
        location=", ".join(raw.get("locationRestrictions") or []),
        description=description,
    )


def fetch_himalayas() -> list[Job]:
    now = time.time()
    seen: set[str] = set()
    jobs: list[Job] = []
    ok_requests = 0
    first = True

    for query in _queries():
        for page in range(1, PAGES_PER_QUERY + 1):
            if not first:
                time.sleep(PAUSE_BETWEEN_REQUESTS)
            first = False

            try:
                resp = requests.get(
                    SEARCH_URL,
                    params={"q": query,
                            "sort": "recent", "page": page},
                    headers=HEADERS,
                    timeout=20,
                )
                resp.raise_for_status()
                results = resp.json().get("jobs", [])
                ok_requests += 1
            except Exception as exc:
                logger.warning("himalayas query %r page %d failed: %s", query, page, exc)
                break   # skip the rest of this query's pages

            if not results:
                break

            for raw in results:
                guid = raw.get("guid")
                if not guid or guid in seen or not raw.get("applicationLink"):
                    continue
                if (raw.get("expiryDate") or now + 1) < now:
                    continue   # already expired
                seen.add(guid)
                jobs.append(_to_job(raw))

    if ok_requests == 0:
        raise RuntimeError("himalayas: every request failed")

    logger.info("himalayas: %d jobs fetched", len(jobs))
    return jobs