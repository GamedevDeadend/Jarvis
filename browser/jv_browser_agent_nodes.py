"""
LangGraph nodes for the Browser Agent's perceive-act-verify loop.

Implements the diagram:
    Query -> Initial navigate -> Observer_Node (snapshot)->Snapshot Optimiser->Plan Reviewer
          -> Decider_Node -> Ref_Validator_Node -> (loop back to Observer_Node | Tools_node)
          -> Observer_Node -> Should_Continue -> (loop back to Decider | END)

  - Decider_Node :  picks exactly ONE next tool call. It does not execute
    anything and does not judge success/failure — single narrow job.

  - Ref_Validator_Node :  It checks the proposed
    ref's role (from the last snapshot) against what the action needs.

  - Should_Continue / Outcome_Router is one function (LangGraph
    conditional edge). it reads last_action_result + step_count + max_steps and routes on code logic
    alone.
"""


from __future__ import annotations

import asyncio
import base64
import os
import re

from typing import Literal
from langgraph.types import interrupt, Command

from langchain.messages import SystemMessage, HumanMessage
from langchain_typesafe import Score, TypeSafeClassifier

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

from browser.jv_browser_agent_state import BrowserAgentState
from browser.jv_browser_agent_tools import BROWSER_TOOLS, REF_LOOKUP, _get_page, optimised_snapshot, _wait_for_load, _get_locator

from pydantic import BaseModel, Field

import logging
logger = logging.getLogger(__name__)


GROQ_MODEL_ID = "openai/gpt-oss-20b"
api_key = os.getenv("GROQ_API_KEY")

MODEL_ID = "qwen2.5:3b"
VL_MODEL_ID = "qwen3.5:0.8b"



class InitialPlanningSchema(BaseModel) :
    """ 
        Schema representing the result of the initial planning of the goal.
        Args:
            goal_steps (str): The steps to reach the goal in the query.
    """
    goal_steps : str = Field(description="""Steps to reach goal in query. Break Goal in steps. STEPS:
                                1. <first step>
                                2. <second step>
                                3. <third step>
                                ...""")


INITIAL_PLANNING_SYSTEM_PROMPT = """
You are a Browser-agent goal planner.

Convert the user's goal into a minimal ordered list of actions using ONLY these browser tools:
- browser_navigate: navigate to a particular URL
- browser_snapshot: take a snapshot of the current webpage
- browser_click: click a particular button, link, etc.
- browser_type: type text in a textbox, searchbox, combobox, etc.
- browser_go_back: go to the previous page
- browser_scroll: scroll the web page
- browser_extract_content: extract information such as an article, paragraph, etc.

RULES:
1. The first step is ALWAYS fixed: go to google.com, search for a query related to the goal, and click one of the most relevant results.
2. Keep the plan minimal. Do not add unnecessary steps.
3. Use browser_snapshot after a page loads or changes, before clicking or typing on it.
4. Use browser_extract_content only when the goal requires reading information from a page.
5. "Order" or "buy" in a goal means ONLY add the item to the cart. Never proceed to checkout, payment, or place the order.
6. Output only the numbered plan, in the format: "N. tool_name: short description".

EXAMPLES:

Goal: Play Re-zero latest episode on crunchyroll
Plan:
1. browser_navigate: Go to google.com
2. browser_type: Type "Re:Zero crunchyroll" in the search box
3. browser_click: Click the Crunchyroll Re:Zero series result
4. browser_snapshot: Take a snapshot of the series page
5. browser_scroll: Scroll down to the episodes list
6. browser_snapshot: Take a snapshot to identify the latest episode
7. browser_click: Click the latest episode
8. browser_snapshot: Take a snapshot of the episode page
9. browser_click: Click the play button

Goal: Order me dietcoke from Swiggy Instamart
Plan:
1. browser_navigate: Go to google.com
2. browser_type: Type "Diet coke Swiggy Instamart" in the search box
3. browser_click: Click the most relevant Swiggy Instamart result
4. browser_snapshot: Take a snapshot of the results page
5. browser_click: Click the most relevant diet coke product
7. browser_snapshot: Take a snapshot of the product page
8. browser_click: Click "Add" on the Diet Coke
9. browser_snapshot: Confirm the item is in the cart (do not checkout)

Goal: Order Zoro poster on Amazon
Plan:
1. browser_navigate: Go to google.com
2. browser_type: Type "Zoro poster Amazon" in the search box
3. browser_click: Click the most relevant Amazon result
4. browser_snapshot: Take a snapshot of the results page
5. browser_click: Click the most relevant Zoro poster product
6. browser_snapshot: Take a snapshot of the product page
7. browser_click: Click "Add to Cart"
8. browser_snapshot: Confirm the item is in the cart (do not checkout)

Goal: Play Mexican Coke song by artist CHIEF on youtube
Plan:
1. browser_navigate: Go to google.com
2. browser_type: Type "Mexican Coke CHIEF youtube" in the search box
3. browser_click: Click the YouTube result for the song
4. browser_snapshot: Take a snapshot of the video page
5. browser_click: Click the play button if the video is not already playing

Goal: Find the opening hours of the Louvre Museum and the ticket price for adults.
Plan:
1. browser_navigate: Go to google.com
2. browser_type: Type "Louvre Museum official website opening hours tickets" in the search box
3. browser_click: Click the official Louvre website result
4. browser_snapshot: Take a snapshot of the homepage
5. browser_click: Click the "Practical information" or "Tickets" link
6. browser_snapshot: Take a snapshot of the new page
7. browser_extract_content: Extract opening hours and adult ticket price

Now plan for the user's goal below.
"""

async def initial_planning_node(state: BrowserAgentState) -> dict:
    """
    Runs once to generate the initial plan to reach the goal.
    """
    model = ChatOllama(model=MODEL_ID, temperature=0)
    
    messages = [
        SystemMessage(content=INITIAL_PLANNING_SYSTEM_PROMPT),
        HumanMessage(content=f'Goal: "{state["goal"]}"'),
    ]

    # structured_llm = model.with_structured_output(InitialPlanningSchema)
    response : InitialPlanningSchema = await model.ainvoke(messages)
    
    goal_lines = "\n" + response.content
    logger.info(f"Steps for {state.get('goal')} : \n{response}\n\n")
    logger.info(f"Steps for {state.get('goal')} : \n{goal_lines}\n\n")

    return {"goal_steps" : goal_lines, "human_feedback" : None}

        
class SearchQuerySchema(BaseModel):
    """
    Schema representing the result of the search query parsing.

    Args:
        search_query (str): The best search query to reach the goal.
    """
    search_query: str = Field(
        description=(
            "Best search query to reach goal. "
            "Query: Order Zoro poster on Amazon = Result: Zoro poster on amazon "
            "Query: Play one piece latest episode on crunchyroll -> Result: One piece latest episode on crunchyroll "
            "Query: Order DietCoke for me from Blinkit -> Result: DietCoke on Blinkit"
        )
    )
    
async def generate_search_query(state):
    model = ChatOllama(model=MODEL_ID, temperature=0)
    model_with_structured_output = model.with_structured_output(SearchQuerySchema)
    response = await model_with_structured_output.ainvoke([
        SystemMessage(content="""You are a search engine query parser. You are part of browser automation system. You have to parse user query and give best search engine query to reach goal. \n\n
                      Examples: 
                      Query : Order Zoro poster on Amazon -> Result: Zoro poster on amazon \n
                      Query : Play one piece latest episode on crunchyroll -> Result: One piece latest episode on crunchyroll \n
                      Query : Order DietCoke for me from Blinkit -> Result: DietCoke on Blinkit \n"""),
        HumanMessage(content=f'Query: "{state["goal"]}"')
    ])
    
    search_query = response.search_query
    return search_query
    
async def initial_search_node(state: BrowserAgentState) -> dict:
    """
    Runs once, After User query is provided. Uses a cheap LLM call to extract best search engine query to reach goal.
    """
    
    search_query = await generate_search_query(state)
    
    logger.info(f"Search query parsed for {state.get('goal')} : \n{search_query}\n\n")

    page = await _get_page()
    snapshot = await optimised_snapshot()
    
    logger.info(f"Snapshot taken of len:{len(snapshot)}\n\n" + snapshot[:2000] + "\n\n")
    
    await page.goto("https://www.google.com")
    
    snapshot = await optimised_snapshot()
    snapshot = snapshot[:5000]
    
    search_box_ref_id = None
    lines  = snapshot.splitlines()
    
    for line in lines:
        combox_string = re.search(r'combobox\s*"Search"', line)
        
        if combox_string is not None:
            logger.info(f"Search box found in snapshot : {line} \n\n")
            search_box_ref_id = re.search(r'\[ref=(.*?)\]', line).group(1)
            break
        
        
    if search_box_ref_id is None:
        logger.error(f"Search box not found in snapshot : {snapshot[:5000]} \n\n")
        return {}
        
    search_box_locator = await _get_locator(search_box_ref_id)
    await search_box_locator.press_sequentially(search_query, delay=2.0)
    await search_box_locator.press("Enter")
    
    return {}
    
    
async def observer_node(state: BrowserAgentState) -> dict:
    """Takes a snapshot only, via the real browser_snapshot @tool through
    a one-off ToolNode. Same reasoning as initial_navigate_node above --
    separated from decision-making so this node's sole job is perception,
    never judgment.
    """
    
    await _wait_for_load()

    logger.info(f"Observer node called {state['pending_action']} \n\n")
    
    snap = await optimised_snapshot()
    
    logger.info(f"Snapshot taken of len:{len(snap)}\n\n" + snap[:2000] + "\n\n")

    return {
        "last_snapshot": snap,
        "last_snapshot_taken": True,
    }
    
    
async def snapshot_optimizer_node(state:BrowserAgentState)->dict:
    """
    Orignal snapshot of many sites is heavy. Which not suitable for local llms. 
    This Node first filter ss to remove unecessary things and then divide it into chunks.
    Afterwards retrieves most relevant chunk. 
    """
    
    #Filter SS
    classifier = TypeSafeClassifier(model="tev1:0.8b")
    snapshot = state.get("last_snapshot") or ""
    goal = state.get("goal") or ""
    
    snapshot = re.sub( r"\s*\[cursor=[^\]]*\]", "",snapshot) #Removing [cursor=...]
    snapshot = re.sub(r"(?m)^[\t\s]*-[\s]*\/url.*\n?\r?", "", snapshot) #Removing [urls]
    snapshot = re.sub(r"(?m)^[\t\s]*-\s*(generic|listitem).+\n?\r?", "", snapshot) #Removing [generic, listitem]
    
    
    current_ss_length = len(snapshot)
    ideal_char_length = 3000
    total_chunks = current_ss_length // ideal_char_length
    rem = current_ss_length % ideal_char_length
    
    if rem > 0 :
            total_chunks  = total_chunks+1
    
    start_index = 0
    end_index = 0
    
    snapshot_lookup = {}
    max_score = -1.0

    for i in range(total_chunks):
        
        if start_index >= current_ss_length:
            break
        
        end_index = start_index + ideal_char_length
        end_index = min(end_index, current_ss_length)
        
        while end_index < current_ss_length and snapshot[end_index] != "\n":
            end_index += 1        
        
        current_chunk = snapshot[start_index : end_index]
        
        payload = {
            "state" : f" This chunk of snapshot of current webpage  : {current_chunk}",
            "questions" : {
                f"Relevancy_Score_{i+1}" : Score(
                    instructions = f"""How relevant / close this snapshot chunk is to the goal? 
                    The goal is : {goal}.""",
                    criteria = ["Not relevant at all",  "Relevant", "Extremely relevant"]
                )
            }
        }
        
        result  = await classifier.ainvoke(payload)
        score = result.scores[f"Relevancy_Score_{i+1}"].score
        
        if score > max_score:
            max_score = score
        
        snapshot_lookup[ score ] = current_chunk
        start_index = end_index
        
     
    for  score in snapshot_lookup.keys():
        logging.info(f"score : {score} \n\n Snapshot : {snapshot_lookup[score][:200]}\n\n")
        
    best_chunk = snapshot_lookup.get(max_score)
    
    if(best_chunk is None):
        logging.info(f"Best Chunk is None. Using first chunk as best chunk \n\n")
        best_chunk = snapshot_lookup.get(list(snapshot_lookup.keys())[0])
    
    logging.info(f" Length of new SS : {len(best_chunk)}\n\nSS : {best_chunk[:5000]}\n\n")
    
    return {
        "last_snapshot_taken": True,
        "last_snapshot" : best_chunk
    }


PLAN_REVIEWER_PROMPT = """
You are plan reviewer agent.Your goal is to review and edit our intial plans if needed. Update the plan only if required, it is not compulsion.

Browser automation agent has following tools available:
- browser_navigate: navigate to a particular URL
- browser_snapshot: take a snapshot of the current webpage
- browser_click: click a particular button, link, etc.
- browser_type: type text in a textbox, searchbox, combobox, etc.
- browser_go_back: go to the previous page
- browser_scroll: scroll the web page
- browser_extract_content: extract information such as an article, paragraph, etc.

RULES:
1. The first step is ALWAYS fixed: go to google.com, search for a query related to the goal, and click one of the most relevant results.
3. Use browser_snapshot after a page loads or changes, before clicking or typing on it.
4. Use browser_extract_content only when the goal requires reading information from a page.
5. "Order" or "buy" in a goal means ONLY add the item to the cart. Never proceed to checkout, payment, or place the order.
6. Output only the numbered plan, in the format: "N. tool_name: short description".

Example : 

1. browser_navigate: Go to google.com
2. browser_type: Type "Zoro poster Amazon" in the search box
3. browser_click: Click the most relevant Amazon result
4. browser_snapshot: Take a snapshot of the results page
5. browser_click: Click the most relevant Zoro poster product
6. browser_snapshot: Take a snapshot of the product page
7. browser_click: Click "Add to Cart"
8. browser_snapshot: Confirm the item is in the cart (do not checkout)


Goal Steps : 
{goal_steps}

Last action result:
{last_action_result}


NOTE : Snapshot is being used in chunks for better relevancy.

latest_snapshot : 

{latest_snapshot}

"""

async def plan_reviewer_node(state:BrowserAgentState)->dict:
    """
    Plan Reviewer Node : Its function is to check latest developments in loop and update plan if needed
    """
    
    goal_steps = state.get("goal_steps") or ""
    goal = state.get("goal") or ""
    snapshot = state.get("last_snapshot") or ""
    last_result = state.get("last_action_result") or ""
    
    model = ChatOllama(model=MODEL_ID, temperature=0,  num_predict=2048)
    response = await model.ainvoke( [SystemMessage(content = PLAN_REVIEWER_PROMPT.format(goal_steps=goal_steps, last_action_result=last_result, latest_snapshot=snapshot))])
    
    new_goal_steps = response.content
    
    logging.info(f"New Steps :\n\n{new_goal_steps}\n\n")
    
    # return{}
    return {"goal_steps" : new_goal_steps}


GOAL_CHECK_PROMPT = """
You are a goal-checker. Compare the GOAL to the SCREENSHOT_DESCRIPTION and decide if the goal was achieved.

RULES:
1. If the SCREENSHOT_DESCRIPTION indicates an advertisement is currently playing or displayed (sponsored content, promotional material,
    or any content visibly distinct from and unrelated to the GOAL),do not judge the goal by the ad itself.
    Instead, look for evidence of the underlying content the user actually requested
   (its title, name, or identifying details).

1. PLAY/OPEN a song or video → Achieved if the video/song is shown loaded or playing in a main player (not just listed in search results or a queue/sidebar).
2. SEARCH a topic → Achieved if relevant search results are shown OR an article/page is opened.
3. ORDER a product → Achieved if product is added to cart. Payment/checkout pages do NOT count.
4. If none of the above patterns fit, use your best judgment based on whether the description shows the end-state of the goal.

Only use the SCREENSHOT_DESCRIPTION as evidence. Do not assume steps happened that aren't described.

Goal:
\"\"\" 
{goal}
\"\"\"

Screenshot Description:
\"\"\"
 {screenshot_description}.
\"\"\"

"""

class GoalCheckSchema(BaseModel):
    """
    Schema representing the result of a goal completion check.

    Args:
        isGoalCompleted (bool): Indicates whether the goal has been successfully achieved. 
        goal_chck_msg (str): Message describing whether the goal was achieved or not.
    """
    is_goal_completed : bool = Field(description = "Boolean which shows if goal is achieved or not")
    goal_chck_msg : str = Field(description = "String msg to show if goal is achieved or not")
    
async def goal_check_node(state: BrowserAgentState) -> str:
    """
    Checks if the goal has been achieved based on the Screenshot.
    """

    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 12)

    if step_count >= max_steps:
        logger.info("Max steps tried : No need to take Screenshot \n\n")
        return {}

    if state.get("ref_validation_error"):
        logger.info("Recovery cycle (validation failed) : skipping Screenshot \n\n")
        return {}



    try : 
        page  = await _get_page()

    except Exception as e:

        logger.error(f"Error while getting page : {e} \n\n")
        return {}
    
    # page_screenshot_bytes = await page.screenshot()
    
    # image_b64 = base64.b64encode(page_screenshot_bytes).decode("utf-8")
    logger.info(f"Screenshot is being taken for goal check : \n\n")
    
    # messages_ss = [
    #     SystemMessage(
    #         content = (
    #             "You are a screenshot describer. You provide a detailed visual "
    #             "description of the given image, so that it can be used to check "
    #             "whether the user's goal has been achieved or not."
    #         )
    #     ),

    #     HumanMessage(
    #         content= [
    #             {"type": "text", "text": "Describe the screenshot in detail. Focus on the visual elements, layout, and any text present."},

    #             {
    #                 "type": "image_url",
    #                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
    #             }
    #         ]
    #     )
    # ]

    
    vllm = ChatOllama(model = VL_MODEL_ID, temperature = 0.3, num_predict = 2048)
    
    response_ss = "Vision analysis disabled during testing."
    
    # try:
    #     response_ss = await vllm.ainvoke(messages_ss)
    # except Exception as e:
    #     logger.warning(f"Vision check failed: {e}")
    #     response_ss = await vllm.ainvoke(messages_ss)


    prompt_with_context = GOAL_CHECK_PROMPT.format(
            goal=state.get("goal", "No goal found in state"),
            screenshot_description= (response_ss)
        )

    
        
    llm = ChatGroq(model=GROQ_MODEL_ID, temperature=0, max_tokens=2048, api_key=api_key)
    # llm = ChatOllama(model=MODEL_ID, temperature=0, num_predict=1024)
    structured_llm = llm.with_structured_output(GoalCheckSchema)
    response : GoalCheckSchema = await structured_llm.ainvoke(prompt_with_context)

    if(response.is_goal_completed):

        logger.info(f"Goal is achieved based on screenshot : {response.goal_chck_msg} \n\n")
        return {
            "is_goal_reached" : response.is_goal_completed,
            "final_answer" : response.goal_chck_msg
        }
    
    else:
        logger.info(f"Goal is NOT achieved based on screenshot : {response.goal_chck_msg} \n\n")
        return {
            "is_goal_reached": False,
            "final_answer": None
        }

DECIDER_SYSTEM_PROMPT = """You are the Decider node in a browser automation system.

The browser automation agent has the following tools available:
- browser_navigate: navigate to a particular URL
- browser_snapshot: take a snapshot of the current webpage
- browser_click: click a particular button, link, etc.
- browser_type: type text in a textbox, searchbox, combobox, etc.
- browser_go_back: go to the previous page
- browser_scroll: scroll the web page
- browser_extract_content: extract information such as an article, paragraph, etc.

Based on a partial page snapshot and other details, you will decide the next tool call to perform.
Propose exactly ONE tool call per turn.

RULE: Even if the snapshot is chunked (partial), you must still make tool calls based on it.
RULE: Only use ref tokens (like "e11") copied exactly from the snapshot — the value inside [ref=...]. A ref token is NEVER a role name like "link", "button", or "searchbox". Never guess a ref.
RULE: Never provide a summary or description of anything. If the goal isn't met yet, call a tool now — don't just plan it.

RULE: Check the ROLE of the ref before picking a tool:
  - role is "searchbox", "textbox", "combobox", "spinbutton", or "slider" -> call browser_type directly.
  - role is "button", "link", "checkbox", "radio", "tab", "menuitem", "option", or similar interactive-but-not-typable roles -> call browser_click.
"""

async def decider_node(state: BrowserAgentState) -> dict:
    """Pick exactly one next tool call, given goal + last snapshot + history.

    Does NOT execute anything. Does NOT judge completion. If a prior
    ref_validation_error exists in state, it's surfaced to the model as
    feedback so it doesn't just repeat the same mistake blindly.

    """
    
    # model = ChatGroq(model=GROQ_MODEL_ID, temperature=0, max_tokens=1024, api_key=api_key)
    model = ChatOllama(model=MODEL_ID, temperature=0, num_ctx=4096, num_predict=2048)
    model_with_tools = model.bind_tools(BROWSER_TOOLS)

    context_parts = [f"GOAL: {state['goal']}\n\n"]
    # context_parts.append(f"GOAL STEPS: {state.get('goal_steps', '')}")
    
    # if state.get("suggested_step"):
    #     context_parts.append(f"Suggested next step : \n{state['suggested_step']}\n\n")

    if state.get("last_snapshot"):
        context_parts.append(f"CHUNKED SNAPSHOT (This is not whole Snapshot) :\n{state['last_snapshot']}\n\n")

    if state.get("ref_validation_error"):
        context_parts.append(
            f"YOUR LAST PROPOSED ACTION WAS REJECTED: {state['ref_validation_error']}\n\n"
            f"Pick a DIFFERENT ref that actually matches the required role."
        )

    if state.get("last_action_result") and not state["last_action_result"].get("success"):
        context_parts.append(
            f"YOUR LAST ACTION FAILED: {state['last_action_result'].get('error')}\n"
            f"Re-check the snapshot before trying again — do not just retry blindly.\n\n"
        )

    logger.info(f"Last action result: {state.get('last_action_result')}\n\n")

        # SystemMessage(content = "Conversational History : \n\n"),
        # *state.get("messages", []),
    messages = [
        SystemMessage(content=DECIDER_SYSTEM_PROMPT),
        HumanMessage(content="\n\n".join(context_parts)),
    ]

    response = await model_with_tools.ainvoke(messages)


    if not response.tool_calls:
        
        # Model didn't call a tool — treat as "done" signal via final_answer.
        logger.info(f"[decider] step={state.get('step_count')} -> NO TOOL CALL (treating as done). content={response.content[:100]!r} \n\n")
        
        return {
            "messages": [response],
            "pending_action": None,
            "final_answer": response.content,
        }

    call = response.tool_calls[0]
    response.tool_calls = [call]

    logger.info(f"[decider] step={state.get('step_count')} -> {call['name']}({call['args']})"
          + (f" | prior_error={state['ref_validation_error'][:80]!r}" if state.get('ref_validation_error') else "") + "\n\n")


    pending: dict = {
        "tool": call["name"],
        "ref": call["args"].get("ref"),
        "args": call["args"],
    }

    return {
        "ref_validation_error" : None,
        "messages": [response],
        "pending_action": pending,
    }
    
    
EXPECTED_ROLES = {

    # Expected roles for each tool.
    # These are the ARIA roles that the tool expects the ref to have in the snapshot.

    "browser_click": {
        "button", "link", "checkbox", "radio", "switch",
        "tab", "menuitem", "menuitemcheckbox", "menuitemradio",
        "option", "treeitem", "gridcell", "cell", "columnheader", "rowheader",
        "listitem",
    },
    
    "browser_type": {
        "searchbox", "search", "textbox", "combobox", "spinbutton", "slider",
    },
    
}

async def ref_validator_node(state: BrowserAgentState) -> dict:
    """Checks the proposed ref's role against
    what the action actually needs, using the last real snapshot as
    ground truth. Also caps repeated validation failures so a stuck
    model can't loop here forever.
    """
    validation_count = state.get("ref_validation_count", 0)

    pending = state.get("pending_action")

    if not pending or not pending.get("ref"):
        logger.info("Not pending tasks\n\n")
        return {"ref_validation_error": None, "ref_validation_count": 0}

    tool_name = pending["tool"]
    ref = pending["ref"]
    expected_roles = EXPECTED_ROLES.get(tool_name)

    if not expected_roles:
        logger.info(f"Tool not need ref {tool_name}\n\n")
        return {"ref_validation_error": None, "ref_validation_count": 0}
    
    actual_role = None
    logger.info(f"Checking ref {ref}\n\n")
    
    try : 
        actual_role = REF_LOOKUP[ref].get('role')
    except Exception as e:
        logger.info(f"actual_role not found due to {e} \n\n")
        snapshot = state.get("last_snapshot")
        pattern = rf"(?m)^[ \t]*-[ \t]+'?([a-z]+)\b.*\[[^\]]*\bref={re.escape(ref)}[\],]"
        m = re.search(pattern, snapshot)
        actual_role = m.group(1) if m else None
        logger.info(f"Role name : {actual_role}")
        
        
    if actual_role is None:
        error = (
            f"Ref '{ref}' was not found in the most recent snapshot. "
            f"It may be stale or invented — re-read the snapshot and pick a real ref."
        )
    elif actual_role not in expected_roles:
        error = (
            f"Ref '{ref}' has role '{actual_role}', but {tool_name} requires "
            f"one of {sorted(expected_roles)}. Re-check the snapshot and match "
            f"by role AND label, not proximity."
        )
    else:
        logger.info(f"Ref check passed no errors found\n\n")
        return {"ref_validation_error": None, "ref_validation_count": 0}

    new_count = validation_count + 1

    if new_count >= 3:
        logger.info(f"Limit of ref validation count exceeded \n\n")
        return {
            "ref_validation_error": "stale ref issue. New Snapshot is captured",
            "ref_validation_count": 0,
            "pending_action": None,
            "last_action_result": {"success": True, "error": None, "url": None},
            "step_count": state.get("step_count", 0) + 1
        }
    else :

        logger.info(f"Current ref validation error {error} \n\n")
        return {
            "ref_validation_error": error,
            "ref_validation_count": new_count,
            "pending_action" : None
        }
        
        
def end_message_node(state: BrowserAgentState) -> dict:
    """Sets the final end_reason and final_answer once the loop is done.

    Runs right before END. Kept separate from should_continue so that
    should_continue only decides ROUTING (loop again vs stop), while this
    node owns writing the final result fields. Reads step_count/max_steps
    the same way should_continue does, so the reason stays consistent
    with whatever decision was just made to stop.
    """
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 12)

    if step_count >= max_steps:
        reason = "max_steps_reached"
    else:
        reason = "completed"

    logger.info(f"[end_message_node] reason={reason} final_answer={state.get('final_answer')!r}\n\n")

    return {
        "end_reason": reason,
        "final_answer": state.get("final_answer") or "Task ended without a final answer.",
    }
    

def human_feedback_node(state : BrowserAgentState) -> Command :
    """
    General Human feedback node this can be called by any node
    to take human input to reduce or avoid hallucinating.
    It doesn't require any edge connection it is called using Command()
    """

    caller_node = state.get("human_feedback_caller")
    feedback_query = state.get("human_feedback_query")

    logger.info(f"Calling now HIL Flow from {caller_node}")

    human_response = interrupt(
        f"{caller_node} node need your input on this :\n{feedback_query}"
    ) 

    return Command(
        goto = caller_node,
        update={"human_feedback_query" : None, "human_feedback_caller" : None, "human_feedback" : human_response}
    )