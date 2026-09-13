"""
LangGraph nodes for the Browser Agent's perceive-act-verify loop.

Implements the diagram:
    Query -> Initial navigate -> Observer_Node (snapshot)
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
import base64
import os

from typing import Literal

from langchain.messages import SystemMessage, HumanMessage

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

api_key = os.getenv("GROQ_API_KEY")

from browser.jv_browser_agent_state import BrowserAgentState
from browser.jv_browser_agent_tools import BROWSER_TOOLS, _get_session

from pydantic import BaseModel, Field 

import logging
logger = logging.getLogger(__name__)


GROQ_MODEL_ID = "openai/gpt-oss-20b"
MODEL_ID = "qwen2.5:3b"
VL_MODEL_ID = "qwen3.5:0.8b"

# Which snapshot "role" each action tool requires. Used by Ref_Validator_Node
# to catch mismatches like typing into a combobox instead of a searchbox.
EXPECTED_ROLES = {
    """
    Expected roles for each tool.
    These are the ARIA roles that the tool expects the ref to have in the snapshot.
    """
    "browser_click": {
        "button", "link", "checkbox", "radio", "switch",
        "tab", "menuitem", "menuitemcheckbox", "menuitemradio",
        "option", "treeitem", "gridcell", "cell", "columnheader", "rowheader",
        "listitem",
    },
    "browser_type": {
        "searchbox", "textbox", "combobox", "spinbutton", "slider",
    },
}


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
You are a browser-agent goal planner.

Convert the user's goal into:
1. A starting URL.
2. A minimal ordered list of actions using the available browser tools.

SITE:
- Explicit URL → use it.
- Named/implied website → use that website.
- Otherwise → https://www.google.com

EXAMPLES:

Goal: "Go to Wikipedia and search for 'Mongols'. Tell me the first sentence."

URL: https://www.wikipedia.org
STEPS:
1. browser_type — enter "Mongols" in the search field and submit
2. browser_click — open the "Mongols" article
3. browser_snapshot — read the first sentence
4. browser_extract_content — extract the first sentence

Goal: "Play Sparkle from Your Name movie on YouTube."

URL: https://www.youtube.com
STEPS:
1. browser_type — enter "Sparkle Your Name" in the search field and submit
2. browser_click — open the relevant video
3. browser_snapshot — verify the video is playing
4. browser_extract_content — extract playback information

Goal: "Go to https://duckduckgo.com and search for 'browsegrab'. Tell me the title of the first result."

URL: https://duckduckgo.com
STEPS:
1. browser_type — enter "browsegrab" in the search field and submit
2. browser_snapshot — read the first result's title
3. browser_extract_content — extract the title

Goal: "Go to Wikipedia."

URL: https://www.wikipedia.org
STEPS:
1. browser_navigate — open Wikipedia

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


    return {"_start_url" : url, "goal_steps" : goal_lines}



async def initial_navigate_node(state: BrowserAgentState, config=None, *, store=None) -> dict:
    """
        Query -> Initial navigate. Hardcoded, not Decider-driven: step zero
        always opens the target site, so there's no need to spend an LLM call
        deciding to do the obvious first action.
    """

    logger.info(f"Initial navigation node called with url = {state.get('_start_url')} \n\n")

    url = state.get("_start_url", "https://www.wikipedia.org")

    session = await _get_session()
    result = await session.navigate(url)
    success = getattr(result, "success", True)
    error = getattr(result, "error", None)

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

    logger.info(f"Observer node called {state['pending_action']} \n\n")

    await asyncio.sleep(3.0)
    session = await _get_session()
    snap = await session.snapshot()


    logger.info("Snapshot taken \n\n" + snap.tree_text[:2000] + "\n\n")

    return {
        "last_snapshot": snap.tree_text,
        "last_snapshot_taken": True,
    }


DECIDER_SYSTEM_PROMPT = """You control a browser. Propose exactly ONE tool call per turn.

RULE: Only use ref tokens (like "e11") copied exactly from the snapshot. Never guess.

RULE: Never describe what you will do next. If the goal isn't met yet, call a tool now — don't just plan it.

RULE : In combobox or searchbox, Use type and submit, click is not required.

RULE : Conversation history is just for overall context never pick ref number from it. Old ref are for old web pages ignore them. Use new ones from snapshot

If your last action was rejected, read the error and pick a different action. Do not repeat the same mistake."""

async def decider_node(state: BrowserAgentState) -> dict:
    """Pick exactly one next tool call, given goal + last snapshot + history.

    Does NOT execute anything. Does NOT judge completion. If a prior
    ref_validation_error exists in state, it's surfaced to the model as
    feedback so it doesn't just repeat the same mistake blindly.

    """
    

    model = ChatOllama(model=MODEL_ID, temperature=0)
    model_with_tools = model.bind_tools(BROWSER_TOOLS)

    context_parts = [f"GOAL: {state['goal']}"]
    context_parts.append(f"GOAL STEPS: {state.get('goal_steps', '')}")

    if state.get("last_snapshot"):
        context_parts.append(f"MOST RECENT SNAPSHOT:\n{state['last_snapshot']}")

    if state.get("ref_validation_error"):
        context_parts.append(
            f"YOUR LAST PROPOSED ACTION WAS REJECTED: {state['ref_validation_error']}\n"
            f"Pick a DIFFERENT ref that actually matches the required role."
        )

    if state.get("last_action_result") and not state["last_action_result"].get("success"):
        context_parts.append(
            f"YOUR LAST ACTION FAILED: {state['last_action_result'].get('error')}\n"
            f"Re-check the snapshot before trying again — do not just retry blindly."
        )

    logger.info(f"Last action result: {state.get('last_action_result')}\n\n")

    messages = [
        SystemMessage(content=DECIDER_SYSTEM_PROMPT),
        SystemMessage(content = "Conversational History : \n\n"),
        *state.get("messages", []),
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


class RefRoleSchema(BaseModel):
    """
    Schema representing the result of a ref role check.
    """
    role: Literal["alert", "alertdialog", "application", "article", "banner",
    "blockquote", "button", "caption", "cell", "checkbox",
    "code", "columnheader", "combobox", "complementary",
    "contentinfo", "definition", "deletion", "dialog", "directory",
    "document", "emphasis", "feed", "figure", "form", "generic",
    "grid", "gridcell", "group", "heading", "img", "insertion",
    "link", "list", "listbox", "listitem", "log", "main",
    "marquee", "math", "meter", "menu", "menubar", "menuitem",
    "menuitemcheckbox", "menuitemradio", "navigation", "none",
    "note", "option", "paragraph", "presentation", "progressbar",
    "radio", "radiogroup", "region", "row", "rowgroup", "rowheader",
    "scrollbar", "search", "searchbox", "separator", "slider",
    "spinbutton", "status", "strong", "subscript", "superscript",
    "switch", "tab", "table", "tablist", "tabpanel", "term",
    "textbox", "time", "timer", "toolbar", "tooltip", "tree",
    "treegrid", "treeitem"] 


ROLE_CHECK_PROMPT = """You are a strict accessibility-tree parser.
Snapshot:
{snapshot}

Find the EXACT occurrence of [ref={ref}].

Return the role explicitly written for that exact ref.

Rules:
1. Do NOT infer the role from surrounding elements.
2. Do NOT infer the role from the element's label.
3. Do NOT infer the role from what the element is likely to do.
4. Copy the role exactly as represented in the snapshot.
5. If [ref={ref}] does not occur anywhere in the snapshot, return NOT_FOUND.
"""

async def _parse_ref_roles(snapshot_text: str, ref : str) -> str | None:
    """Ask the LLM what role a specific ref has in the snapshot.

    Returns the role string (e.g. "link", "searchbox"), or None if the
    ref wasn't found. Replaces the old regex-based _parse_ref_roles,
    since real testing showed the snapshot format varies too much
    across sites (unlabeled elements, wrapped multi-line labels) for a
    single regex to reliably cover.
    """

    logger.info(f"Ref Parser function called with snapshot {snapshot_text[:200]} and ref {ref}\n\n")

    messages = [
        SystemMessage(content=ROLE_CHECK_PROMPT.format(snapshot=snapshot_text, ref=ref)),
        HumanMessage(content=f"ref={ref}"),
    ]

    model = ChatOllama(model=MODEL_ID, temperature=0)

    structured_llm = model.with_structured_output(RefRoleSchema)
    response : RefRoleSchema = await structured_llm.ainvoke(messages)

    logger.info(f"[ref_role_check] ref={ref!r} -> {response.role!r}\n\n")

    if response.role == "NOT_FOUND" or not response.role:
        return None
    
    return response.role


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

    actual_role = await _parse_ref_roles(state.get("last_snapshot", ""), state.get("pending_action", {}).get("ref"))

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
        session  = await _get_session()

    except Exception as e:

        logger.error(f"Error while getting session : {e} \n\n")
        return {}

    session_page = await session._ensure_page()
    page_screenshot_bytes = await session_page.screenshot()
    
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

    
    vllm = ChatOllama(model = VL_MODEL_ID, temperature = 0, max_tokens = 200)
    response_ss = await vllm.ainvoke(messages_ss)

    logger.info(f"Screenshot description : {response_ss.content} \n\n")

    
    prompt_with_context = GOAL_CHECK_PROMPT.format(
            goal=state.get("goal", "No goal found in state"),
            screenshot_description= response_ss.content
        )

    
        
    llm = ChatGroq(model=GROQ_MODEL_ID, temperature=0, max_tokens=1024, api_key=api_key)
    # llm = ChatOllama(model=MODEL_ID, temperature=0, max_tokens=1024)
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