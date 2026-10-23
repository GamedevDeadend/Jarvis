import sqlite3
import logging

from job_finder.core.jf_job_structure import Job
from job_finder.core.jf_paths import DB_PATH
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

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
                id                  TEXT PRIMARY KEY,
                source              TEXT NOT NULL,
                title               TEXT NOT NULL,
                company             TEXT NOT NULL,
                url                 TEXT NOT NULL,
                salary_text         TEXT,
                is_salary_estimated INTEGER NOT NULL DEFAULT 0,
                priority            TEXT NOT NULL DEFAULT 'secondary',
                decision            TEXT NOT NULL DEFAULT 'accepted',
                fit_score           INTEGER,
                reason              TEXT,
                first_seen          TEXT NOT NULL,
                alerted             INTEGER NOT NULL DEFAULT 0
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


def save_job(job: Job) -> None:
    """Insert a new job row from a Job (accepted or rejected).

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
                 salary_text, is_salary_estimated, priority,
                 decision, fit_score, reason,
                 first_seen, alerted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                job.id,
                job.source,
                job.title,
                job.company,
                job.url,
                job.salary_text,
                job.is_salary_estimated,
                job.priority,
                job.decision,
                job.fit_score,
                job.reason,
                now,
            ),
        )
    logger.info("saved job %s (%s at %s)", job.id, job.title, job.company)


def mark_alerted(job_id: str) -> None:
    """Flag a job as alerted, after the Discord post for it succeeds."""
    with _connect() as conn:
        conn.execute("UPDATE jobs SET alerted = 1 WHERE id = ?", (job_id,))
    logger.info("marked alerted: %s", job_id)


def get_unalerted() -> list[Job]:
    """Return all stored jobs (accepted and rejected) whose Discord post
    has not succeeded yet, e.g. to retry after a failure. Route each one
    by job.decision: notify_job if accepted, notify_rejected if rejected.

    Fields that are not stored (description, exact_salary, location) come
    back at their defaults."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM jobs WHERE alerted = 0").fetchall()
    return [Job(**dict(row)) for row in rows]