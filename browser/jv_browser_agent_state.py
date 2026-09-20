"""
State schema for the Browser Agent's perceive-act-verify LangGraph loop.

"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict, Sequence

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages

import logging
logger = logging.getLogger(__name__)


class BrowserAgentState(TypedDict):
    """Full shared state for one Browser agent task run.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages]

    _start_url: str

    goal: str
    goal_steps: str
    is_goal_reached : bool

    last_snapshot: str
    last_snapshot_taken: bool

    ref_validation_count: int
    ref_validation_error: str | None
    pending_action: dict | None

    last_action_result: dict | None

    step_count: int
    max_steps: int

    human_feedback_query : str | None
    human_feedback_caller : str | None
    human_feedback : str | None


    end_reason: Literal["completed", "gave_up", "max_steps_reached"] | None
    final_answer: str | None


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
        goal_status_steps={},
        is_goal_reached = False,

        last_snapshot="",
        last_snapshot_taken=False,

        ref_validation_count = 0,
        ref_validation_error=None,
        pending_action=None,

        last_action_result=None,

        step_count=0,
        max_steps=max_steps,

        human_feedback_query = "",
        human_feedback_caller = "",
        human_feedback = "",
        
        end_reason=None,
        final_answer=None,
    )