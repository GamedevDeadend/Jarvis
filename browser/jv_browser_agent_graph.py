"""
State based Graph assembly for the Browser Agent's perceive-act-verify loop.
"""

from __future__ import annotations

import asyncio
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode

from browser.jv_browser_agent_state import BrowserAgentState, initial_state
from browser.jv_browser_agent_nodes import (
    goal_check_node,
    initial_query_parse_node,
    initial_navigate_node,
    observer_node,
    snapshot_optimizer_node,
    plan_reviewer_node,
    decider_node,
    ref_validator_node,
    end_message_node,
    human_feedback_node,
)
from browser.jv_browser_agent_tools import BROWSER_TOOLS, close_browser


from dotenv import load_dotenv
load_dotenv()

from langfuse.langchain import CallbackHandler 
langfuse_handler = CallbackHandler()

import logging
logger = logging.getLogger(__name__)



async def process_hf_interrupts(app, config, result):
    """ 
    Function to process interrupts raised for Human feed back by Browser Agent
    """
    while result.get("__interrupt__"):
        agent_query = result["__interrupt__"][0].value
        answer = input( f"{agent_query}\n\n" )
        result = await app.ainvoke(Command(resume=answer), config)
    return result


def route_after_validation(state: BrowserAgentState) -> str:
    """
    Conditional edge: route after validation based on ref_validation_error
    1. If ref_validation_error is present, route to observer_node for recovery.
    2. If no ref_validation_error, route to browser_tool_node to execute the tool.
    """

    logger.info(f"Route after validation called with pending action: {state.get('pending_action')} \n\n")

    if state.get("ref_validation_error"):
        logger.info(f"Going to Observer node with ref validation error : {state.get('ref_validation_error')} \n\n")
        return "observer_node"

    logger.info("Going to tool node to execute tool \n\n")
    return "browser_tool_node"


async def should_continue(state: BrowserAgentState) -> str:
    """
    Conditional edge: should the agent continue executing steps?
    Based on lastsnapshot and goal, no. of steps take, ref_validation_error and max_steps.
    """

    logger.info(f"Should continue router called with pending action: {state.get('pending_action')} \n\n")

    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 12)

    if step_count >= max_steps:
        logger.info("Max steps tried : directing to end_message_node \n\n")
        return "end_message_node"

    if state.get("ref_validation_error"):
        logger.info("Recovery cycle (validation failed) : skipping goal-check, directing to decider_node \n\n")
        return "decider_node"
    
    if state.get("is_goal_reached"):
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

    graph.add_node("initial_query_parse_node", initial_query_parse_node)
    graph.add_node("initial_navigate_node", initial_navigate_node)
    graph.add_node("observer_node", observer_node)
    graph.add_node("snapshot_optimizer_node", snapshot_optimizer_node)
    graph.add_node("plan_reviewer_node", plan_reviewer_node)
    graph.add_node("goal_check_node", goal_check_node)
    graph.add_node("decider_node", decider_node)
    graph.add_node("ref_validator_node", ref_validator_node)
    graph.add_node("browser_tool_node", tool_node)
    graph.add_node("end_message_node", end_message_node)
    graph.add_node("human_feedback_node", human_feedback_node)

    graph.add_edge(START, "initial_query_parse_node")
    graph.add_edge("initial_query_parse_node", "initial_navigate_node")
    graph.add_edge("initial_navigate_node", "observer_node")
    graph.add_edge( "observer_node", "snapshot_optimizer_node",)
    graph.add_edge( "snapshot_optimizer_node","plan_reviewer_node")
    graph.add_edge("plan_reviewer_node", "goal_check_node")
    graph.add_edge("decider_node", "ref_validator_node")
    graph.add_edge("browser_tool_node", "observer_node")

    graph.add_conditional_edges("ref_validator_node", route_after_validation)
    graph.add_conditional_edges("goal_check_node", should_continue)


    return graph.compile(checkpointer=InMemorySaver())


async def main(): 
    
    app = build_browser_agent_graph()

    # goal = "Order Zoro poster on Amazon"
    goal = "Play Mexican Coke song by artist CHIEF on youtube"
    state = initial_state(goal=goal, max_steps=12)

    config = {
            "configurable": {"thread_id": {str(uuid.uuid4())}},
            "callbacks": [langfuse_handler],
            "run_name" : f"goal: {goal}",
        }

    result = await app.ainvoke(state, config)

    print(f"\n\n {result} \n\n")

    result = await process_hf_interrupts(app, config, result)

    print("--- Final result ---")
    print(f"end_reason: {result.get('end_reason')}")
    print(f"final_answer: {result.get('final_answer')}")
    print(f"step_count: {result.get('step_count')}")

    print("\n\n--- Conversational History ---\n\n")
    print (result.get('messages'))

    input("Press Enter to close the browser and exit...")
    await close_browser()


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
