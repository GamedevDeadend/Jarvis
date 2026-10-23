import hashlib
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from pypdf import PdfReader

from job_finder.core.jf_paths import BASE, RES_DIR, PROFILE_MD, RULES_JSON, HASH_FILE

logger = logging.getLogger(__name__)

load_dotenv(BASE / ".env")

MAX_TRIES = 3
API_KEY = os.environ.get("GROQ_API_KEY")


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


MODEL_ID = "openai/gpt-oss-120b"   # one-time call, so the stronger model


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
    llm = ChatGroq(model=MODEL_ID, api_key=API_KEY, temperature=0, max_retries=3).with_structured_output(Profile)
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