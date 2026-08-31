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

from browser.jv_browser_agent_state import BrowserAgentState, initial_state
from browser.jv_browser_agent_nodes import (
    initial_query_parse_node,
    initial_navigate_node,
    observer_node,
    decider_node,
    ref_validator_node,
    end_message_node,

    MODEL_ID
)
from browser.jv_browser_agent_tools import BROWSER_TOOLS, close_session

import logging
logger = logging.getLogger(__name__)


# ---- Routing / conditional-edge functions ----
# These decide WHICH already-registered node to go to next -- they are
# never registered as nodes themselves via add_node(). Kept here, in the
# graph file, since routing is part of graph topology, not node logic.

def route_after_validation(state: BrowserAgentState) -> str:
    """Conditional edge: did the ref pass validation?"""

    logger.info(f"Route after validation called with pending action: {state.get('pending_action')} \n\n")

    if state.get("ref_validation_error"):
        logger.info(f"Going to decider node with ref validation error : {state.get('ref_validation_error')} \n\n")
        return "decider_node"  

    logger.info("Going to tool node to execute tool \n\n")
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


    logger.info(f"Should continue router called with pending action: {state.get('pending_action')} \n\n")

    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 12)

    if step_count >= max_steps:
        logger.info("Max steps tried : directing to end_message_node \n\n")
        return "end_message_node"  # max steps reached, end the run

    if state.get("pending_action") is None and state.get("final_answer"):
        logger.info("Decider self-reported done : directing to end_message_node \n\n")
        return "end_message_node"


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

    if is_goal_completed:
        logger.info("Goal is achieved : directing to end_message_node \n\n")
        return "end_message_node"

    logger.info("Goal not achieved yet : directing to decider_node \n\n")
    return "decider_node"


# ---- Graph assembly ----

def build_browser_agent_graph():

    logger.info("Intializing agent Graph \n\n")

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

    goal = "Play Nandemoiya song of artist named Radwimp on youtube.com"
    state = initial_state(goal=goal, max_steps=12)

    result = await app.ainvoke(state)
    print("--- Final result ---")
    print(f"end_reason: {result.get('end_reason')}")
    print(f"final_answer: {result.get('final_answer')}")
    print(f"step_count: {result.get('step_count')}")
    input("Press Enter to close the browser and exit...")
    await close_session()
    return "Press Enter to close Session" # ensure browser session is closed even if error occurs



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("jarvis.log", encoding="utf-8"),
        ],
    )
    asyncio.run(main())
