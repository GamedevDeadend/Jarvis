"""
State based Graph assembly for the Browser Agent's perceive-act-verify loop.
"""

from __future__ import annotations

import asyncio
import base64
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
from browser.jv_browser_agent_tools import BROWSER_TOOLS, close_session, _get_session
from browsegrab import BrowseSession

import logging
logger = logging.getLogger(__name__)

VL_MODEL_ID = "qwen3.5:0.8b"





def route_after_validation(state: BrowserAgentState) -> str:
    """
    # ---- Routing / conditional-edge functions ----
    # These decide WHICH already-registered node to go to next -- they are
    # never registered as nodes themselves via add_node(). Kept here, in the
    # graph file, since routing is part of graph topology, not node logic.
    """

    logger.info(f"Route after validation called with pending action: {state.get('pending_action')} \n\n")

    if state.get("ref_validation_error"):
        logger.info(f"Going to Observer node with ref validation error : {state.get('ref_validation_error')} \n\n")
        return "observer_node"

    logger.info("Going to tool node to execute tool \n\n")
    return "browser_tool_node"



GOAL_CHECK_PROMPT = """Goal: {goal}

You are checking whether a goal has been achieved based on a screenshot.
Has the goal already been achieved based on what's visible in this screenshot?

For example : 
1) If goal is to PLAY OR OPEN a SONG OR VIDEO on YouTube. If you see on youtube video page is open and playback controls are visible then goal is achieved.
2) If goal is to SEARCH for a song on YouTube. If you see search results page with relevant results then goal is achieved.
3) If goal is to SEARCH an article for a topic on Wikipedia. If you see search results page with relevant results or if any article is opened then goal is achieved.
4) If goal is to ORDER a product on ecommerce site. If you see product page with product added to cart then goal is achieved.(Payment page is not allowed)

Respond in EXACTLY this format, nothing else:
ACHIEVED: yes or no
ANSWER: <if yes, the direct answer to the goal. If no, leave blank>"""



class GoalCheckSchema(BaseModel):
    """
    Schema representing the result of a goal completion check.

    Args:
        isGoalCompleted (bool): Indicates whether the goal has been successfully achieved. 
        goal_chck_msg (str): Message describing whether the goal was achieved or not.
    """
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
        return "end_message_node"

    if state.get("pending_action") is None and state.get("final_answer"):
        logger.info("Decider self-reported done : directing to end_message_node \n\n")
        return "end_message_node"

    if state.get("ref_validation_error"):
        logger.info("Recovery cycle (validation failed) : skipping goal-check, directing to decider_node \n\n")
        return "decider_node"

    try : 
        session  = await _get_session()
    except Exception as e:
        logger.error(f"Error while getting session : {e} \n\n")
        return "decider_node"

    session_page = await session._ensure_page()
    page_screenshot_bytes = await session_page.screenshot()

    image_b64 = base64.b64encode(page_screenshot_bytes).decode("utf-8")

    logger.info(f"Screenshot is being taken for goal check : \n\n")


    messages = [ 
                    SystemMessage(
                        content = GOAL_CHECK_PROMPT.format(
                            goal = state.get("goal", "No goal found in state"),
                        )
                    ),

                    HumanMessage(content = [{"type": "text", "text": f"Goal: {state.get('goal', 'No goal found')}\n\nHas this goal been achieved based on the screenshot? Respond with your judgment."},
                                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                                            ])
                ]
    
    llm = ChatOllama(model = VL_MODEL_ID, temperature = 0, max_tokens = 200)
    structured_llm = llm.with_structured_output(GoalCheckSchema)
    response : GoalCheckSchema = await structured_llm.ainvoke(messages)

    logger.info(f"Screenshot Goal check response : {response.goal_chck_msg} \n\n")

    is_goal_completed = response.isGoalCompleted

    if is_goal_completed:
        logger.info("Goal is achieved : directing to end_message_node \n\n")
        return "end_message_node"

    logger.info("Goal not achieved yet : directing to decider_node \n\n")
    return "decider_node"



def build_browser_agent_graph():
    """Initialization of graph through nodes creation and connecting edges

    Returns:
       CompiledStateGraph: The compiled `StateGraph`.
    """

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

    graph.add_conditional_edges("ref_validator_node", route_after_validation)


    graph.add_conditional_edges("observer_node", should_continue)

    graph.add_edge("browser_tool_node", "observer_node")

    return graph.compile()


async def main(): 
    
    app = build_browser_agent_graph()

    goal = "Play Nandemonaiya song(official video) by artist named Radwimps on youtube"
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
