"""
State schema for the Browser Agent's perceive-act-verify LangGraph loop.

"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict, Sequence

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages

import logging
logger = logging.getLogger(__name__)


class BrowserAgentState(TypedDict):
    """Full shared state for one Browser agent task run.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages]

    goal: str
    goal_steps: str
    max_steps: int
    _start_url: str

    last_snapshot: str
    last_snapshot_taken: bool

    pending_action: dict | None
    ref_validation_error: str | None

    last_action_result: dict | None
    step_count: int

    end_reason: Literal["completed", "gave_up", "max_steps_reached"] | None
    final_answer: str | None
    ref_validation_count: int
    is_goal_reached : bool


def initial_state(
    goal: str,
    start_url: str = "https://www.wikipedia.org",
    max_steps: int = 12,
) -> BrowserAgentState:
    """Build a fresh state dict for a new Browser agent task."""

    logger.info(f"Building Intial State for agent to complete this goal \n\n {goal} \n\n")

    return BrowserAgentState(
        messages=[],

        _start_url=start_url,
        goal=goal,
        goal_steps="",

        step_count=0,
        max_steps=max_steps,
        
        last_snapshot="",
        last_snapshot_taken=False,

        last_action_result=None,
        pending_action=None,

        ref_validation_error=None,
        ref_validation_count = 0,
        
        end_reason=None,
        final_answer=None,
        is_goal_reached = False
    )