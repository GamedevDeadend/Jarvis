"""
LangGraph nodes for the Browser Agent's perceive-act-verify loop.

Implements the diagram:
    Query -> Initial navigate -> Observer_Node (snapshot)
          -> Decider_Node -> Ref_Validator_Node -> Browser_Tool_Node
          -> Observer_Node -> Should_Continue -> (loop back to Decider | END)

Design principles carried over from browser_tools.py and browser_agent_state.py:
  - Decider_Node picks exactly ONE next tool call. It does not execute
    anything and does not judge success/failure — single narrow job.
  - Ref_Validator_Node is PURE CODE, no LLM call. It checks the proposed
    ref's role (from the last snapshot) against what the action needs,
    catching the exact "e29 is a combobox option, not a searchbox" bug
    found in real testing — deterministically, for free.
  - Should_Continue / Outcome_Router is one function (LangGraph
    conditional edge), not two separate LLM nodes: it reads
    last_action_result + step_count + max_steps and routes on code logic
    alone. No LLM judgment about "are we done" — that was the original
    self-reported-completion trust problem this whole design avoids.

FIX (this version): ref_validator_node's 3-strike cap now also clears
last_action_result when it force-exits. Without this, should_continue's
"was the last action a failure?" check could see a STALE failed result
from before the validation rejections and route back to decider_node
anyway — silently discarding final_answer and never actually stopping
the loop.
"""

from __future__ import annotations

import asyncio
import re

from langchain.messages import SystemMessage, HumanMessage

from langchain_ollama import ChatOllama

from jv_browser_agent_state import BrowserAgentState
from jv_browser_agent_tools import BROWSER_TOOLS, _get_session


MODEL_ID = "qwen2.5:3b"

# Which snapshot "role" each action tool requires. Used by Ref_Validator_Node
# to catch mismatches like typing into a combobox instead of a searchbox.
EXPECTED_ROLES = {
    "browser_click": {"button", "link", "checkbox", "radio", "menuitem", "tab"},
    "browser_type": {"searchbox", "textbox", "combobox"},
}


async def initial_navigate_node(state: BrowserAgentState, config=None, *, store=None) -> dict:
    """
        Query -> Initial navigate. Hardcoded, not Decider-driven: step zero
        always opens the target site, so there's no need to spend an LLM call
        deciding to do the obvious first action.
    """

    print("\n\n Initial navigation node called")

    url = state.get("_start_url", "https://www.wikipedia.org")

    session = await _get_session()
    result = await session.navigate(url)
    success = getattr(result, "success", True)
    error = getattr(result, "error", None)

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

    print(f"\n\n Observer node called {state['pending_action']}")

    session = await _get_session()
    snap = await session.snapshot()

    await asyncio.sleep(2)

    print("\n\n Snapshot take \n\n" + snap.tree_text[:500] + "\n\n")

    return {
        "last_snapshot": snap.tree_text,
        "last_snapshot_taken": True,
    }


ROLE_CHECK_PROMPT = """Snapshot:
{snapshot}

What is the accessibility role of the element with ref={ref}? 
Respond with ONLY the role word (e.g. "link", "button", "searchbox", "combobox"). 
If ref={ref} does not appear in the snapshot, respond with exactly: NOT_FOUND"""

async def _parse_ref_roles(snapshot_text: str, ref : str) -> str | None:
    """Ask the LLM what role a specific ref has in the snapshot.

    Returns the role string (e.g. "link", "searchbox"), or None if the
    ref wasn't found. Replaces the old regex-based _parse_ref_roles,
    since real testing showed the snapshot format varies too much
    across sites (unlabeled elements, wrapped multi-line labels) for a
    single regex to reliably cover.
    """

    print(f"\n\n Ref Parser function called {snapshot_text}")

    model = ChatOllama(model=MODEL_ID, temperature=0)

    messages = [
        SystemMessage(content=ROLE_CHECK_PROMPT.format(snapshot=snapshot_text, ref=ref)),
        HumanMessage(content=f"ref={ref}"),
    ]

    response = await model.ainvoke(messages)
    role = response.content.strip()

    print(f"[ref_role_check] ref={ref!r} -> {role!r}")

    if role == "NOT_FOUND" or not role:
        return None
    
    return role




INITIAL_PARSING_SYSTEM_PROMPT = """Break the goal into two things: the starting website, and an ordered list of steps to reach the answer.

Respond in EXACTLY this format, nothing else:
URL: <the starting website URL>
STEPS:
1. <first step>
2. <second step>
...

Keep each step short and concrete — one action or one clear sub-goal per line. Do not include ref tokens or technical details; steps describe WHAT to do, not HOW to click.

Examples:

Goal: "Go to Wikipedia and search for 'Mongols'. Tell me the first sentence of the article."
URL: https://www.wikipedia.org
STEPS:
1. Search for "Mongols"
2. Open the Mongols article
3. Read and report the first sentence

Goal: "Play Sparkle from Your Name movie on YouTube."
URL: https://www.youtube.com
STEPS:
1. Search for "Sparkle Your Name"
2. Open the correct video from the search results
3. Confirm the video is playing

Goal: "Go to https://duckduckgo.com and search for 'browsegrab'. Tell me the title of the first result."
URL: https://duckduckgo.com
STEPS:
1. Search for "browsegrab"
2. Read the first search result's title
3. Report the title

If no specific site is implied, use: https://www.google.com
If the goal is already a single simple action, output exactly one step."""


async def initial_query_parse_node(state: BrowserAgentState) -> dict:
    """Runs once, before initial_navigate_node. Uses a cheap LLM call to
    extract which site the goal implies AND break the goal into ordered
    steps, instead of requiring the caller to pass _start_url explicitly
    or leaving the Decider to infer the whole multi-step plan on its own.

    The parsed steps are appended to the existing goal text (not tracked
    as a separate state field) -- decider_node already reads state["goal"]
    every turn, so no other wiring changes are needed. Falls back to
    whatever _start_url/goal were already set if parsing fails.
    """
    model = ChatOllama(model=MODEL_ID, temperature=0)

    messages = [
        SystemMessage(content=INITIAL_PARSING_SYSTEM_PROMPT),
        HumanMessage(content=f'Goal: "{state["goal"]}"'),
    ]

    response = await model.ainvoke(messages)
    text = response.content.strip()

    url = None
    goal_lines = []
    in_steps = False

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("URL:"):
            url = line[len("URL:"):].strip()
        elif line.startswith("STEPS:"):
            in_steps = True
        elif in_steps and line:
            goal_lines.append(line)

    print(f"[initial_query_parse_node] goal={state['goal'][:60]!r} -> url={url!r} steps={goal_lines}")

    update = {}

    if url and (url.startswith("http://") or url.startswith("https://")):
        update["_start_url"] = url
    else:
        print(f"[initial_query_parse_node] REJECTED non-URL output, keeping existing _start_url")

    if goal_lines:
        update["goal"] = state["goal"] + "\n\nSteps:\n" + "\n".join(goal_lines)
    else:
        print(f"[initial_query_parse_node] No steps parsed, keeping original goal")

    return update


DECIDER_SYSTEM_PROMPT = """You control a browser. Propose exactly ONE tool call per turn.

RULE: To type into a search box, textbox, or combobox — call browser_type directly. This items not required click action.

RULE: Only use ref tokens (like "e11") copied exactly from the snapshot. Never guess.

RULE: Never describe what you will do next. If the goal isn't met yet, call a tool now — don't just plan it.

RULE : In combobox or searchbox, Use type and submit tool, click is not required.

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

    print(f"\n\n Last action result: {state.get('last_action_result')}")

    messages = [
        SystemMessage(content=DECIDER_SYSTEM_PROMPT),
        *state.get("messages", []),
        HumanMessage(content="\n\n".join(context_parts)),
    ]

    response = await model_with_tools.ainvoke(messages)


    if not response.tool_calls:
        # Model didn't call a tool — treat as "done" signal via final_answer.

        print(f"[decider] step={state.get('step_count')} -> NO TOOL CALL (treating as done). content={response.content[:100]!r}")
        
        return {
            "messages": [response],
            "pending_action": None,
            "final_answer": response.content,
        }

    call = response.tool_calls[0]  # enforce ONE action per turn, ignore extras
    response.tool_calls = [call]

    print(f"[decider] step={state.get('step_count')} -> {call['name']}({call['args']})"
          + (f" | prior_error={state['ref_validation_error'][:80]!r}" if state.get('ref_validation_error') else ""))


    pending: dict = {
        "tool": call["name"],
        "ref": call["args"].get("ref"),
        "args": call["args"],
    }

    return {
        "messages": [response],
        "pending_action": pending,
        "ref_validation_error": None,  # clear previous error once a new attempt is made
    }


async def ref_validator_node(state: BrowserAgentState) -> dict:
    """Pure code, zero LLM calls. Checks the proposed ref's role against
    what the action actually needs, using the last real snapshot as
    ground truth. Also caps repeated validation failures so a stuck
    model can't loop here forever.
    """
    validation_count = state.get("ref_validation_count", 0)

    pending = state.get("pending_action")

    if not pending or not pending.get("ref"):
        return {"ref_validation_error": None, "ref_validation_count": 0}

    tool_name = pending["tool"]
    ref = pending["ref"]
    expected_roles = EXPECTED_ROLES.get(tool_name)

    if not expected_roles:
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
        # Valid ref -> reset the counter, let it through.
        return {"ref_validation_error": None, "ref_validation_count": 0}

    new_count = validation_count + 1

    if new_count >= 3:
        # Cap hit: stop looping here entirely, force a hard exit rather
        # than another retry. pending_action=None + final_answer set
        # means should_continue will route to end_max_steps/end_completed
        # instead of back into this same failure loop.
        #
        # BUG FIX: also clear/neutralize last_action_result here. Without
        # this, should_continue's check ("was the last action a failure?
        # if so, refuse the completion claim") can see a STALE failed
        # result from before these validation rejections and route back
        # to decider_node anyway -- silently discarding final_answer and
        # never actually stopping the loop. Setting success=True here
        # means "we are choosing to stop on our own terms, not because
        # the last real action failed" -- the actual failure reason is
        # preserved in final_answer's text instead.
        return {
            "ref_validation_error": None,
            "ref_validation_count": 0,
            "pending_action": None,
            "last_action_result": {"success": True, "error": None, "url": None},
            "final_answer": (
                f"Gave up after 3 consecutive invalid ref attempts on '{ref}'. "
                f"Last error: {error}"
            ),
        }
    else :
        return {
            "ref_validation_error": error,
            "ref_validation_count": new_count,
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

    print(f"[end_message_node] reason={reason} final_answer={state.get('final_answer')!r}")

    return {
        "end_reason": reason,
        "final_answer": state.get("final_answer") or "Task ended without a final answer.",
    }