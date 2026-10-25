import logging
import html
import re
import requests
import xml.etree.ElementTree as ET

from job_finder.core.jf_job_structure import Job


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</h\d>", "\n", raw)
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t\u00a0]+", " ", raw)
    raw = re.sub(r"\n\s*\n+", "\n", raw)
    return raw.strip()

logger = logging.getLogger(__name__)

FEEDS = [
    "https://weworkremotely.com/categories/remote-programming-jobs.rss",
]
HEADERS = {"User-Agent": "JobRadar/0.1 (personal job alert)"}


def fetch_we_work_remotely() -> list[Job]:
    jobs: list[Job] = []
    seen: set[str] = set()
    skipped = 0
    failed_feeds = 0

    for feed_url in FEEDS:
        try:
            resp = requests.get(feed_url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as e:
            logger.warning("wwr: feed failed %s (%s)", feed_url, e)
            failed_feeds += 1
            continue

        for item in root.iter("item"):
            link = (item.findtext("link") or "").strip()
            slug = link.rstrip("/").rsplit("/", 1)[-1]
            if "/remote-jobs/" not in link or not slug or slug in seen:
                skipped += 1
                continue
            seen.add(slug)

            raw_title = (item.findtext("title") or "").strip()
            company, sep, title = raw_title.partition(": ")
            if not sep:
                company, title = "Unknown", raw_title

            category = (item.findtext("category") or "").strip()
            region = (item.findtext("region") or "").strip()
            description = (
                f"Category: {category}\n"
                f"Region: {region}\n"
                + html_to_text(item.findtext("description") or "")
            )
            jobs.append(Job(
                id=f"wwr:{slug}",
                source="weworkremotely",
                title=title.strip(),
                company=company.strip(),
                url=link,
                priority="secondary",
                description=description,
                location=region,
            ))

    if failed_feeds == len(FEEDS):
        raise RuntimeError("wwr: every feed failed")

    logger.info("wwr: %d jobs fetched, %d skipped", len(jobs), skipped)
    return jobs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for job in fetch_we_work_remotely()[:10]:
        print(job.id, "|", job.title, "|", job.company, "|", job.location)