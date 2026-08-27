"""
StateGraph assembly for the Browser Agent's perceive-act-verify loop.
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

from langchain.messages import HumanMessage, SystemMessage
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama

from jv_browser_agent_state import BrowserAgentState, initial_state
from jv_browser_agent_nodes import (
    initial_query_parse_node,
    initial_navigate_node,
    observer_node,
    decider_node,
    ref_validator_node,
    end_message_node,

    MODEL_ID
)
from jv_browser_agent_tools import BROWSER_TOOLS, close_session


# ---- Routing / conditional-edge functions ----
# These decide WHICH already-registered node to go to next -- they are
# never registered as nodes themselves via add_node(). Kept here, in the
# graph file, since routing is part of graph topology, not node logic.

def route_after_validation(state: BrowserAgentState) -> str:
    """Conditional edge: did the ref pass validation?"""

    print(f"\n\n Route after validation called with state: {state.get('pending_action')}")

    if state.get("ref_validation_error"):
        return "decider_node"  # bounce back with feedback, no LLM cost wasted on execution
    
    return "browser_tool_node"

GOAL_CHECK_PROMPT = """Goal: {goal}

Current page snapshot:
{snapshot}

Has the goal already been fully achieved based on what's visible in this snapshot?

Respond in EXACTLY this format, nothing else:
ACHIEVED: yes or no
ANSWER: <if yes, the direct answer to the goal. If no, leave blank>"""

class GoalCheckSchema(BaseModel):
    isGoalCompleted : bool = Field(description = "Boolean which shows if goal is achieved or not")
    goal_chck_msg : str = Field(description = "String msg to show if goal is achieved or not")


async def should_continue(state: BrowserAgentState) -> str:
    """Conditional edge: should the agent continue executing steps?
       Based on lastsnapshot and goal, no. of steps take and max_steps.
    """

    print(f"\n\n Should continue router called with state: {state.get('pending_action')}, "
          f"{state.get('last_action_result')}, {state.get('step_count')}, {state.get('max_steps')}")


    messages = [ 
                    SystemMessage(
                        content = GOAL_CHECK_PROMPT.format(
                            goal = state["goal"],
                            snapshot = state.get("last_snapshot", "")
                        )
                    ),

                    HumanMessage(content = "Check now")   
                ]
    
    llm = ChatOllama(model = MODEL_ID, temperature = 0, max_tokens = 200)
    structured_llm = llm.with_structured_output(GoalCheckSchema)
    response : GoalCheckSchema = await structured_llm.ainvoke(messages)

    is_goal_completed = response.isGoalCompleted

    print(f"\n\n Response at should continue {response} \n\n")

    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 12)

    if step_count >= max_steps:
        return "end_message_node"  # max steps reached, end the run


    if is_goal_completed:
        return "end_message_node"

    return "decider_node"


# ---- Graph assembly ----

def build_browser_agent_graph():

    tool_node = ToolNode(BROWSER_TOOLS)

    graph = StateGraph(BrowserAgentState)

    graph.add_node("initial_query_parse", initial_query_parse_node)
    graph.add_node("initial_navigate", initial_navigate_node)
    graph.add_node("observer_node", observer_node)
    graph.add_node("decider_node", decider_node)
    graph.add_node("ref_validator_node", ref_validator_node)
    graph.add_node("browser_tool_node", tool_node)
    graph.add_node("end_message_node", end_message_node)

    graph.add_edge(START, "initial_query_parse")
    graph.add_edge("initial_query_parse", "initial_navigate")
    graph.add_edge("initial_navigate", "observer_node")
    graph.add_edge("decider_node", "ref_validator_node")

    # route_after_validation returns a real node name directly:
    # "decider_node", "browser_tool_node", or "should_continue_router"
    graph.add_conditional_edges("ref_validator_node", route_after_validation)

    # should_continue returns a real node name directly:
    # "decider_node", "end_message_node"
    graph.add_conditional_edges("observer_node", should_continue)

    graph.add_edge("browser_tool_node", "observer_node")

    return graph.compile()


async def main():
    app = build_browser_agent_graph()

    goal = "Go to duck duck go find the tallest mountain in the world, then search for that mountain's country on google and tell me its capital city."
    state = initial_state(goal=goal, max_steps=12)  # start_url now auto-parsed

    # try:
    # finally:

    result = await app.ainvoke(state)
    print("--- Final result ---")
    print(f"end_reason: {result.get('end_reason')}")
    print(f"final_answer: {result.get('final_answer')}")
    print(f"step_count: {result.get('step_count')}")
    input("Press Enter to close the browser and exit...")
    await close_session()  # ensure browser session is closed even if error occurs


if __name__ == "__main__":
    asyncio.run(main())