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

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

from browser.jv_browser_agent_state import BrowserAgentState
from browser.jv_browser_agent_tools import BROWSER_TOOLS, REF_LOOKUP, _get_page, optimised_snapshot, _wait_for_load

from pydantic import BaseModel, Field

import logging
logger = logging.getLogger(__name__)


GROQ_MODEL_ID = "openai/gpt-oss-20b"
api_key = os.getenv("GROQ_API_KEY")

MODEL_ID = "qwen2.5:3b"
VL_MODEL_ID = "qwen3.5:0.8b"


class IntialParsingSchema(BaseModel) :
    """ 
        Schema representing the result of the initial parsing of the goal.
        Args:
            intial_url (str): The initial URL to navigate to.
            goal_steps (str): The steps to reach the goal in the query.
    """
    initial_url : str  = Field(description="Initial url which agent should navigate to start the process . Example https://www.youtube.com, https://www.wikipedia.org ")
    goal_steps : str = Field(description="""Steps to reach goal in query. Break Goal in steps. STEPS:
                                1. <first step>
                                2. <second step>
                                3. <third step>
                                ...""")


INITIAL_PARSING_SYSTEM_PROMPT = """

You are a Browser-agent goal planner.

Convert the user's goal into:
1. A starting URL.(Note : Region is India)
2. A minimal ordered list of actions using the available browser tools

SITE:
- Explicit URL → use it.
- Named/implied website → use that website.
- Otherwise → https://www.google.com

    
EXAMPLES:

Goal: "Play Sparkle from Your Name movie on YouTube."
URL: https://www.youtube.com
STEPS:
1. browser_type — enter "Sparkle Your Name" in the search field and submit
2. browser_click — open the relevant video
3. browser_snapshot — verify the video is playing
4. browser_extract_content — extract playback information

Goal: "Go to Wikipedia and search for 'Mongols'. Tell me the first sentence."
URL: https://www.wikipedia.org
STEPS:
1. browser_type — enter "Mongols" in the search field and submit
2. browser_click — open the "Mongols" article
3. browser_snapshot — read the first sentence
4. browser_extract_content — extract the first sentence

Goal: "Go to https://duckduckgo.com and search for 'browsegrab'. Tell me the title of the first result."
URL: https://duckduckgo.com
STEPS:
1. browser_type — enter "browsegrab" in the search field and submit
2. browser_snapshot — read the first result's title
3. browser_extract_content — extract the title


OUTPUT:
URL: <starting URL>
STEPS:
1. <tool_name> — <action>
2. <tool_name> — <action>
"""


async def initial_query_parse_node(state: BrowserAgentState) -> dict:
    """
    Runs once, before initial_navigate_node. Uses a cheap LLM call to
    extract which site the goal implies AND break the goal into ordered
    steps, instead of requiring the caller to pass _start_url explicitly
    or leaving the Decider to infer the whole multi-step plan on its own.

    """
    model = ChatOllama(model=MODEL_ID, temperature=0)

    messages = [
        SystemMessage(content=INITIAL_PARSING_SYSTEM_PROMPT),
        HumanMessage(content=f'Goal: "{state["goal"]}"'),
    ]

    structured_llm = model.with_structured_output(IntialParsingSchema)
    response : IntialParsingSchema = await structured_llm.ainvoke(messages)

    url = response.initial_url
    goal_lines = "\n" + response.goal_steps

    logger.info(f" Initial url : {url} \n\n")
    logger.info(f"Steps for {state.get('goal')} : \n{goal_lines}\n\n")

    return {"_start_url" : url, "goal_steps" : goal_lines, "human_feedback" : None}


async def initial_navigate_node(state: BrowserAgentState, config=None, *, store=None) -> dict:
    """
        Query -> Initial navigate. Hardcoded, not Decider-driven: step zero
        always opens the target site, so there's no need to spend an LLM call
        deciding to do the obvious first action.
    """

    logger.info(f"Initial navigation node called with url = {state.get('_start_url')} \n\n")

    url = state.get("_start_url", "https://www.wikipedia.org")

    page = await _get_page()
    result = await page.goto(url)
    success = getattr(result, "ok", True)
    error = getattr(result, "status_text", None)

    logger.info(f"Navigation results :\n {result} \n\n")

    return {
        "last_snapshot_taken": False,
        "last_action_result": {"success": success, "error": error, "url": url if success else None},
        "step_count": state.get("step_count", 0) + 1,
    }
    
    
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
    

CHUNK_OBSERVER_PROPMPT = """You are Chunk Observer Agent. You are part of browser automation system which has tools like click, type, goback, navigate.
On each step you are provided with series of snapshot.
As a agent you have to observe one snapshot at a time. And Give best candidate ref id among it which can help reach our goal.

For example :

Best_Candidate : Button with ref e10 because this is product page.
Best_Candidate : Searchbox with ref e20 because this can be used to search product.

Anything along those lines.

NOTE : As There will be multiple snapshot you can also give your observation as PASS.

Recent Snapshot : 

{snapshot}
"""

async def make_observation(snapshot, goal, i:int, j:int):

    model = ChatOllama(model=MODEL_ID, temperature=0,  num_predict=1024)

    messages = [SystemMessage(content=CHUNK_OBSERVER_PROPMPT.format(snapshot=snapshot[i:j])),
                HumanMessage(content = f"Goal : {goal}")]

    result = await model.ainvoke(messages)

    logging.info(f"Observation : {result.content}\n\n")
    return result

CHUNK_SELECTOR_PROMPT = """These are all the observations of various web snapshot.
You have to decide which observation is best direction to achieve Goal.
Keyword PASS means in that observation there was nothing relevant to goal

RULE: Base your choice only on what's stated in the observations below — do not assume content not mentioned.
RULE: If multiple observations look equally relevant, prefer the one whose candidate ref most directly matches an action the goal requires (e.g. a search box for a search goal, a specific product/video link for an open/play goal).

Make sure to return Non-Zero Index 

\n\n All observations : {obs}
"""

class Best_Observation(BaseModel):
    index : int = Field (description="Give Index of best observation from all the observation")
    reason : str = Field (description="Reason to select this observation")
    
async def snapshot_optimizer_node(state:BrowserAgentState)->dict:
    """
    Orignal snapshot of many sites is heavy. Which not suitable for local llms. 
    This Node first filter ss to remove unecessary things and then divide it into chunks.
    Afterwards retrieves most relevant chunk. 
    """
    
    #Filter SS
    snapshot = state.get("last_snapshot") or ""
    goal = state.get("goal") or ""
    
    snapshot = re.sub( r"\s*\[cursor=[^\]]*\]", "",snapshot) #Removing [cursor=...]
    snapshot = re.sub(r"(?m)^[\t\s]*-[\s]*\/url.*\n?\r?", "", snapshot) #Removing [urls]
    snapshot = re.sub(r"(?m)^[\t\s]*-\s*(generic|listitem).+\n?\r?", "", snapshot) #Removing [generic, listitem]
    
    
    current_ss_length = len(snapshot)
    ideal_char_length = 5000
    total_chunks = current_ss_length // ideal_char_length
    rem = current_ss_length % ideal_char_length
    
    if rem > 0 :
            total_chunks  = total_chunks+1
    
    observations = []
    snapshot_chunks = {}
    
    start_index = 0

    for i in range(total_chunks):
        
        if start_index >= len(snapshot):
            break
        
        end_index = start_index + ideal_char_length
        end_index = min(end_index, len(snapshot))
        
        while end_index < len(snapshot) and snapshot[end_index] != "\n":
            end_index += 1

        result = await make_observation(snapshot, goal, start_index, end_index)
        snapshot_chunks[i+1] = snapshot[start_index : end_index]
        observations.append( (f"{i+1} : {result.content}\n\n"))
        
        start_index = end_index
        
    
    model = ChatOllama(model=MODEL_ID, temperature=0,  num_predict=1024)
    model_with_structured_output = model.with_structured_output(Best_Observation)
    
    combined_obs = " ".join(observations)
    
    messages = [SystemMessage(content=CHUNK_SELECTOR_PROMPT.format(obs=combined_obs)),
                    HumanMessage(content = f"This is user query :{goal}")]
    
    result = await model_with_structured_output.ainvoke(messages)
    logging.info(f"Final Obs Index {result.index}\n Reason : {result.reason}\n\n")
    
    if result.index == 0:
        result.index = 1
    
    best_chunk = snapshot_chunks.get(result.index)
    
    logging.info(f" Length of new SS : {len(best_chunk)}\n\nSS : {best_chunk[:5000]}\n\n")
    
    return {
        "last_snapshot_taken": True,
        "last_snapshot" : best_chunk,
        "suggested_step" : observations[(result.index-1)]
    }


PLAN_REVIEWER_PROMPT = """
You are plan review agent working in browser automation system. Your goal is to review and edit our intial plans if needed.
Based on relevant snapshot chunk, our goal and last action result. You will update the plan. 
Update the plan only if required, it is not compulsion.

CAUTION : Make sure revised steps are in similar format as of old ones. DON'T change the way steps were written.

Goal or Query : 
{goal}

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
    
    new_goal_steps = state.get("goal_steps") or ""
    goal = state.get("goal") or ""
    snapshot = state.get("last_snap_shot") or ""
    last_result = state.get("last_action_result") or ""
    
    model = ChatOllama(model=MODEL_ID, temperature=0,  num_predict=2048)
    response = await model.ainvoke( [SystemMessage(content = PLAN_REVIEWER_PROMPT.format(goal=goal, last_action_result=last_result, latest_snapshot=snapshot))])
    
    new_goal_steps = response.content
    
    logging.info(f"New Steps :\n\n{new_goal_steps}\n\n")
    
    return{}
    
    # return {"goal_steps" : new_goal_steps}


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
    
    page_screenshot_bytes = await page.screenshot()
    
    image_b64 = base64.b64encode(page_screenshot_bytes).decode("utf-8")
    logger.info(f"Screenshot is being taken for goal check : \n\n")
    
    messages_ss = [
        SystemMessage(
            content = (
                "You are a screenshot describer. You provide a detailed visual "
                "description of the given image, so that it can be used to check "
                "whether the user's goal has been achieved or not."
            )
        ),

        HumanMessage(
            content= [
                {"type": "text", "text": "Describe the screenshot in detail. Focus on the visual elements, layout, and any text present."},

                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                }
            ]
        )
    ]

    
    vllm = ChatOllama(model = VL_MODEL_ID, temperature = 0, num_predict = 1024)
    # response_ss = await vllm.ainvoke(messages_ss)

    # logger.info(f"Screenshot description : {response_ss.content} \n\n")

    
    prompt_with_context = GOAL_CHECK_PROMPT.format(
            goal=state.get("goal", "No goal found in state"),
            screenshot_description= (f"Not providing Screenshot for now use snapshot : \n\n{state.get('last_snapshot')}")
        )

    
        
    # llm = ChatGroq(model=GROQ_MODEL_ID, temperature=0, max_tokens=2048, api_key=api_key)
    llm = ChatOllama(model=MODEL_ID, temperature=0, num_predict=1024)
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
        

DECIDER_SYSTEM_PROMPT = """You are the Decider node in a browser automation system. You are given access to several browser tools. Based on a partial page snapshot and other details, you will decide the next tool call to perform.

Propose exactly ONE tool call per turn.

RULE: Even if the snapshot is chunked (partial), you must still make tool calls based on it.

RULE: Only use ref tokens (like "e11") copied exactly from the snapshot — the value inside [ref=...]. A ref token is NEVER a role name like "link", "button", or "searchbox". Never guess a ref.

RULE: Never provide a summary or description of anything. If the goal isn't met yet, call a tool now — don't just plan it.

RULE: Check the ROLE of the ref before picking a tool:
  - role is "searchbox", "textbox", "combobox", "spinbutton", or "slider" -> call browser_type directly. Do NOT click it first — clicking is not required and wastes a step.
  - role is "button", "link", "checkbox", "radio", "tab", "menuitem", "option", or similar interactive-but-not-typable roles -> call browser_click.

EXAMPLES (study the ref-to-tool mapping carefully):

Snapshot excerpt:
  - searchbox "Search Amazon.in" [ref=e23]
Correct action: browser_type(ref="e23", text="wireless mouse")
Wrong action: browser_click(ref="e23")  # WRONG — searchbox should be typed into directly, never clicked first

Snapshot excerpt:
  - combobox "Select category" [ref=e8]
Correct action: browser_type(ref="e8", text="Electronics")
Wrong action: browser_click(ref="e8")  # WRONG — combobox follows the same rule as searchbox

Snapshot excerpt:
  - link "Zoro" [ref=e41]
Correct action: browser_click(ref="e41")
Wrong action: browser_click(ref="link")  # WRONG — "link" is the ROLE, not the ref. The ref is "e41".

Snapshot excerpt:
  - button "Add to Cart" [ref=e17]
Correct action: browser_click(ref="e17")

If your last action was rejected, read the error and pick a different action. Do not repeat the same mistake."""

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
    context_parts.append(f"GOAL STEPS: {state.get('goal_steps', '')}")
    
    # if state.get("suggested_step"):
    #     context_parts.append(f"Latest Observation : \n{state['suggested_step']}\n\n")

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
    
    actual_role = REF_LOOKUP[ref].get('role')

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