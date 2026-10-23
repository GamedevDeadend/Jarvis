import operator
from typing import Annotated, TypedDict

from job_finder.core.jf_job_structure import Job


class AgentState(TypedDict, total=False):
    """Shared state for one JobRadar cycle."""

    jobs: list[Job]          # everything fetched
    new_jobs: list[Job]      # after dedup
    survivors: list[Job]     # passed keyword_filter
    judged: list[Job]        # scored by the LLM
    accepted: list[Job]      # passed the gate
    # keyword and gate rejects are appended together
    rejected: Annotated[list[Job], operator.add]

    stats: dict              # counts for the heartbeat
    dry_run: bool
    abort: bool              # set by ensure_profile to end the cycle