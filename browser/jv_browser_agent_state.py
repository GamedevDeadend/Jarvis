"""
State schema for the Browser Agent's perceive-act-verify LangGraph loop.

"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages


class BrowserAgentState(TypedDict):
    """Full shared state for one Browser agent task run.
    """

    messages: Annotated[list[AnyMessage], add_messages]

    goal: str
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


def initial_state(
    goal: str,
    start_url: str = "https://www.wikipedia.org",
    max_steps: int = 12,
) -> BrowserAgentState:
    """Build a fresh state dict for a new Browser agent task."""
    return BrowserAgentState(
        messages=[],
        goal=goal,
        max_steps=max_steps,
        _start_url=start_url,
        last_snapshot="",
        last_snapshot_taken=False,
        pending_action=None,
        ref_validation_error=None,
        last_action_result=None,
        step_count=0,
        end_reason=None,
        final_answer=None,
        ref_validation_count = 0
    )