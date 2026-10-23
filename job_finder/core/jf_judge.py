import logging
import re
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from job_finder.core.jf_job_structure import Job
from job_finder.core.jf_paths import BASE, PROFILE_MD, REQUIREMENTS_TXT

logger = logging.getLogger(__name__)
load_dotenv(BASE / ".env")

MODEL_ID = "openai/gpt-oss-120b"

JOB_DESCRIPTION_CHARS = 1500
PAUSE_BETWEEN_JUDGE_CALLS = 45   # avoid rate limits; 30-60s is typical for Groq
MAX_JUDGE_CALLS = 10
JUDGE_BATCH_SIZE = 6
FIT_SCORE_THRESHOLD = 7

SALARY_FLOOR_INR_MONTH = 200000

JUDGE_SYSTEM_PROMPT = """
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
  (convert currencies, divide yearly pay by 12; for a range use the top of
  the range), otherwise null
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


def judge_batch(jobs: list[Job], max_calls: int = MAX_JUDGE_CALLS) -> list[Job]:
    """Scores jobs with Groq, JUDGE_BATCH_SIZE per call. Returns copies of
    the judged jobs with fit_score, reason and exact_salary filled in.

    A job that was not judged (call cap reached, API error, or missing from
    the answer) is simply left out of the result. It is not saved either, so
    dedup does not drop it and the next cycle tries it again."""
    if not jobs:
        return []

    system = JUDGE_SYSTEM_PROMPT.format(
        profile=PROFILE_MD.read_text(encoding="utf-8"),
        requirements=REQUIREMENTS_TXT.read_text(encoding="utf-8"),
    )
    llm = ChatGroq(model=MODEL_ID, temperature=0, max_retries=3).with_structured_output(Verdicts)

    batches = [jobs[i: (i + JUDGE_BATCH_SIZE)]
               for i in range(0, len(jobs), JUDGE_BATCH_SIZE)][:max_calls]
    judged: list[Job] = []

    for n, batch in enumerate(batches, start=1):
        if n > 1:
            time.sleep(PAUSE_BETWEEN_JUDGE_CALLS)  # avoid rate limits

        payload = "\n\n".join(
            f"id: {j.id}\ntitle: {j.title}\ncompany: {j.company}\n"
            f"salary field: {j.salary_text}\n"
            f"location: {j.location or 'not stated'}\n"
            f"description: {_snippet(j.description)}"
            for j in batch
        )
        try:
            result = llm.invoke([("system", system), ("user", payload)])
        except Exception as exc:
            logger.warning("judge call %d/%d failed: %s", n, len(batches), exc)
            continue

        by_id = {v.id: v for v in result.verdicts}
        for job in batch:
            v = by_id.get(job.id)
            if v is None:
                logger.warning("judge gave no verdict for %s", job.id)
                continue
            judged.append(job.model_copy(update={
                "fit_score": v.fit_score,
                "reason": v.reason,
                "exact_salary": v.salary_est_inr_month,
            }))

    logger.info("judge: %d of %d jobs judged in %d calls",
                len(judged), len(jobs), len(batches))
    return judged


def _reject(job: Job, reason: str) -> Job:
    return job.model_copy(update={
        "decision": "rejected",
        "priority": "rejected",
        "reason": reason,
    })


def gate(judged: list[Job]) -> tuple[list[Job], list[Job]]:
    """Applies the score threshold and the salary floor. Returns
    (accepted, rejected).

    - score below the threshold: rejected.
    - salary 'not stated' and no estimate: rejected (salary unknown).
    - any job whose exact_salary is below the floor: rejected.
    - salary stated but the judge gave no number: accepted (cannot check).
    - accepted with 'not stated' salary: the estimate becomes salary_text
      and is_salary_estimated=1 (notify_job adds the [estimated] tag)."""
    accepted, rejected = [], []
    for job in judged:
        stated = job.salary_text != "not stated"
        salary = job.exact_salary

        if (job.fit_score or 0) < FIT_SCORE_THRESHOLD:
            rejected.append(_reject(job, job.reason or "low fit score"))
        elif salary is None and not stated:
            rejected.append(_reject(job, f"gate: salary unknown. {job.reason}"))
        elif salary is not None and salary < SALARY_FLOOR_INR_MONTH:
            kind = "stated" if stated else "est"
            rejected.append(_reject(
                job, f"gate: {kind} ~₹{salary:,}/mo is below the floor. {job.reason}"))
        else:
            if not stated:
                job = job.model_copy(update={
                    "salary_text": f"~₹{salary:,}/mo",
                    "is_salary_estimated": 1,
                })
            accepted.append(job.model_copy(update={"decision": "accepted"}))

    logger.info("gate: %d accepted, %d rejected", len(accepted), len(rejected))
    return accepted, rejected