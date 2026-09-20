"""
Raw browser tools for the Jarvis Browser Agent.

It is mix of custom tools(Using Playwright API) and some optimization functions
to optimise snapshot.

This is an ASYNC API. Tools below use LangChain's async `@tool` support
(coroutine functions) accordingly — not sync wrappers.
"""

from __future__ import annotations

import os
import re
import asyncio


from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command

from playwright.async_api import BrowserContext, Response, Page, Locator, async_playwright, Playwright


from dotenv import load_dotenv
load_dotenv()

import logging
logger = logging.getLogger(__name__)

browser_profile_path = os.getenv("BROWSER_PROFILE_PATH")
browser_executable_path = os.getenv("BROWSER_EXECUTABLE_PATH")


# config = BrowseGrabConfig()
# config.snapshot.filter_interactive_only = True
# # config.snapshot.max_snapshot_length = 3000
# # config.snapshot.max_content_length = 1000
# config.browser.headless = False

browser = None
playwright = None

REF_LOOKUP = {}

VALID_ROLES = {
    "alert",
    "alertdialog",
    "application",
    "article",
    "banner",
    "blockquote",
    "button",
    "caption",
    "cell",
    "checkbox",
    "code",
    "columnheader",
    "combobox",
    "complementary",
    "contentinfo",
    "definition",
    "deletion",
    "dialog",
    "directory",
    "document",
    "emphasis",
    "feed",
    "figure",
    "form",
    "generic",
    "grid",
    "gridcell",
    "group",
    "heading",
    "img",
    "insertion",
    "link",
    "list",
    "listbox",
    "listitem",
    "log",
    "main",
    "marquee",
    "math",
    "meter",
    "menu",
    "menubar",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "navigation",
    "none",
    "note",
    "option",
    "paragraph",
    "presentation",
    "progressbar",
    "radio",
    "radiogroup",
    "region",
    "row",
    "rowgroup",
    "rowheader",
    "scrollbar",
    "search",
    "searchbox",
    "separator",
    "slider",
    "spinbutton",
    "status",
    "strong",
    "subscript",
    "superscript",
    "switch",
    "tab",
    "table",
    "tablist",
    "tabpanel",
    "term",
    "textbox",
    "time",
    "timer",
    "toolbar",
    "tooltip",
    "tree",
    "treegrid",
    "treeitem",
}


async def _get_browser()->BrowserContext:
    """ Lazily load and return browser"""

    global playwright,browser

    if browser is None:
        
        playwright = await async_playwright().start()
        chromium =  playwright.chromium
        browser = await chromium.launch_persistent_context(user_data_dir=browser_profile_path, executable_path=browser_executable_path, headless=False, args=["--profile-directory=Profile 2"] )
        page = await browser.new_page()
            
    return browser


async def _get_page()->Page:
    """
    Get Current page of browser
    """

    browser = await _get_browser()

    if browser.pages:
        return browser.pages[-1]

    logger.info("No page is currently open")
    return None

async def _wait_for_load():
    """
    Helper function to wait for page loading
    """
    page  = await _get_page()
    await page.wait_for_load_state("load")


async def _get_locator(ref_id : str):
    """
    Get locator using ref id from Ref_Lookup
    """
    
    if not REF_LOOKUP:
        
        logging.info("Ref lookup invalid \n\n")
        return None
    
    role = REF_LOOKUP[ref_id]["role"]
    name = REF_LOOKUP[ref_id]["name"]

    page = await _get_page()
    locator = page.get_by_role(role=role, name=name)

    return locator

    
def ref_map_builder(snapshot : str):
    """
    Function to build Reference Lookup Dict
    """

    lines = snapshot.splitlines()

    for line in lines:

        # Search refs 
        ref_match = re.search(r"\[ref=([^\]]+)\]", line)

        if not ref_match:
            continue

        ref_id = ref_match.group(1)

        # Pattern to lookup role of refs
        REF_LOOKUP[ref_id] = {"role" : "", "name" : ""}

        role_match_group = re.search(r"^\s*-\s*([a-z]+)", line)

        if not role_match_group:
            del REF_LOOKUP[ref_id]
            continue

        role = role_match_group.group(1)

        if role not in VALID_ROLES:
            del REF_LOOKUP[ref_id]
            continue

        # name / heading of refs
        name_match = re.search(r'"(.+)"', line)

        if not name_match:
            del REF_LOOKUP[ref_id]
            continue

        name = name_match.group(1)
        
        REF_LOOKUP[ref_id] = {"role" : role, "name" : name}


async def optimised_snapshot():

    REF_LOOKUP.clear()

    snapshot = ""

    page = await _get_page()

    if page is None :
        return "No Page Found"
        
    snapshot = await page.locator("body").aria_snapshot(mode="ai")
            

    snapshot = re.sub( r"\s*\[cursor=[^\]]*\]", "",snapshot) #Removing [cursor=...]
    snapshot = re.sub(r"(?m)^[\t\s]*-[\s]*\/url.*\n?\r?", "", snapshot) #Removing [urls]
    snapshot = re.sub(r"(?m)^[\t\s]*-\s*(generic|listitem).+\n?\r?", "", snapshot) #Removing [generic, listitem]

    ref_map_builder(snapshot)

    return snapshot


async def close_browser() -> None:
    """Explicitly close the browser. Call this when the Browser
    agent's task graph finishes (success, failure, or max-steps cutoff) —
    otherwise the browser process is left running.
    """
    browser = _get_browser()

    if browser is not None:
        await browser.close()
        browser = None
        
    if playwright is not None:
        await playwright.stop()
        playwright = None


@tool
async def browser_navigate(runtime: ToolRuntime, url: str) -> Command:
    """Navigate the browser to a given URL.

    Args:
        url: Full URL to navigate to, including https://
    """

    response = None
    page = _get_page()

    try :

        response = await page.goto(url)
        await page.wait_for_load_state("load")
        success = True

    except Exception as e:

        return Command(update={
            "last_snapshot_taken": False,
            "last_action_result": {"success": getattr(response, "ok", False), "error": str(e), "url": None},
            "step_count": step_count,
            "messages": [
                ToolMessage(
                    content=f"Navigation to {url} FAILED: {error or 'unknown error'}",
                    tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
                       )
    
    success = getattr(response, "ok", True)
    error  = None
    step_count = runtime.state.get("step_count", 0) + 1
    
    return Command(
        update={
            "last_snapshot_taken": False,
            "last_action_result": {"success": success, "error": None, "url": url},
            "step_count": step_count,
            "messages": [
                ToolMessage(
                    content=f"Navigated to {url}. Call browser_snapshot to see the page state.",
                    tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )


@tool
async def browser_snapshot(runtime: ToolRuntime) -> Command:
    """Take a snapshot of the current page's accessibility tree.

    Also creates a compact, ref-based (e1, e2, ...) representation of visible
    interactive elements and content, optimized for small local LLMs.

    This is the ONLY source of truth for what is actually on the page.
    Always call this after an action to verify what happened — never
    assume an action succeeded without checking.
    """

    snapshot_text = await optimised_snapshot()

    step_count = runtime.state.get("step_count", 0) + 1

    return Command(
        update={
            "last_snapshot": snapshot_text,
            "last_snapshot_taken": True,
            "step_count": step_count,
            "messages": [
                ToolMessage(
                    content=snapshot_text,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
async def browser_click(ref: str, runtime: ToolRuntime) -> Command:
    """Click an element on the page by its ref.

    Args:
        ref: The ref TOKEN from the most recent browser_snapshot output
    """
    
    locator = await _get_locator(ref)
    result=success=error=url= None

    if locator is None:
        return Command(
            update={
                "last_snapshot_taken": False,
                "last_action_result": {"success": False, "error": "Locator not found, ref can't be resolved", "url": None},
                "step_count": runtime.state.get("step_count", 0) + 1,
                "messages": [ToolMessage(
                    content="Click FAILED: ref not found or not clickable. Call browser_snapshot to see current valid refs.",
                    tool_call_id=runtime.tool_call_id,)],
            }
        )


    try :
        result = await locator.click(timeout=10_000)
        success = True
    except Exception as e:
        success = False
        error = str(e)

    print(f"\n\nClick result {result} \n")

    page = await _get_page()
    url = page.url if success else None
    step_count = runtime.state.get("step_count", 0) + 1

    if not success:
        return Command(
            update={
                "last_snapshot_taken": False,
                "last_action_result": {"success": False, "error": error, "url": url},
                "step_count": step_count,
                "messages": [
                    ToolMessage(
                        content=(
                            f"Click on ref '{ref}' FAILED: "
                            f"{error or 'ref not found or not clickable'}. "
                            f"Call browser_snapshot to see current valid refs before trying again."
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return Command(
        update={
            "last_snapshot_taken": False,
            "last_action_result": {"success": True, "error": None, "url": url},
            "step_count": step_count,
            "messages": [
                ToolMessage(
                    content=f"Clicked {ref}. Now at {url}. Call browser_snapshot to verify the result.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
async def browser_type(ref: str, text: str, submit: bool = True, clear: bool = True, runtime: ToolRuntime = None) -> Command:
    """ Type text into an input element on the page by its ref.
        To perform a search: ALWAYS set submit=True in the SAME call that types
        the search text. Do NOT type first and click a separate button afterward
        that is unreliable and unnecessary. This tool submits the form/search
        directly via submit=True.
    Args:
        ref: The ref TOKEN from the most recent browser_snapshot output
             text: Text to type into the element

        clear : boolean to clear existing text
             
        submit: Set True to submit immediately after typing (e.g. pressesEnter).
                Use this for ALL searches — do not click a search
                button separately.
    """
    
    locator = _get_locator(ref)
    result=success=error=url= None

    if locator is None:

        return Command(
            update={
                "last_snapshot_taken": False,
                "last_action_result": {"success": False, "error": "Locator not found, ref can't be resolved", "url": url},
                "step_count": step_count,
                "messages": [
                    ToolMessage(
                        content=(
                            f"Typing into ref '{ref}' FAILED: "
                            f"{error or 'ref not found or not typeable'}. "
                            f"Call browser_snapshot to see current valid refs before trying again."
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    try:
        if clear:
            result = await locator.fill(text, timeout=10_000)
        else:
            result = await locator.press_sequentially(text, delay=50, timeout=10_000)
        
        if submit:
            result = await locator.press("Enter")
            await asyncio.sleep(5.0)

        success = True

    except Exception as e:
        success = False
        error = str(e)

    page = await _get_page()
    url = page.url if success else None
    step_count = runtime.state.get("step_count", 0) + 1

    if not success:
        return Command(
            update={
                "last_snapshot_taken": False,
                "last_action_result": {"success": False, "error": error, "url": url},
                "step_count": step_count,
                "messages": [
                    ToolMessage(
                        content=(
                            f"Typing into ref '{ref}' FAILED: "
                            f"{error or 'ref not found or not typeable'}. "
                            f"Call browser_snapshot to see current valid refs before trying again."
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return Command(
        update={
            "last_snapshot_taken": False,
            "last_action_result": {"success": True, "error": None, "url": url},
            "step_count": step_count,
            "messages": [
                ToolMessage(
                    content=f"Typed '{text}' into {ref}"
                    + (" and submitted" if submit else "")
                    + ". Call browser_snapshot to verify the result.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
async def browser_go_back(runtime: ToolRuntime) -> Command:
    """Navigate back to the previous page in browser history."""
    page = await _get_page()
    result = page.go_back()
    success = getattr(result, "success", True)
    error = getattr(result, "error", None)
    url = getattr(result, "url", None)

    step_count = runtime.state.get("step_count", 0) + 1

    if not success:
        return Command(
            update={
                "last_snapshot_taken": False,
                "last_action_result": {"success": False, "error": error, "url": url},
                "step_count": step_count,
                "messages": [
                    ToolMessage(
                        content=f"Go back FAILED: {error or 'unknown error'}. "
                        f"Call browser_snapshot to see current page state.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return Command(
        update={
            "last_snapshot_taken": False,
            "last_action_result": {"success": True, "error": None, "url": url},
            "step_count": step_count,
            "messages": [
                ToolMessage(
                    content="Went back. Call browser_snapshot to verify the result.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
async def browser_scroll(
    direction: str = "down", amount: int = 500, ref: str | None = None, runtime: ToolRuntime = None
) -> Command:
    """Scroll the page up or down.

    Use this when the content you need isn't visible in the current
    snapshot — e.g. the article text is below the fold. Always follow
    with browser_snapshot to see what's now visible.

    Args:
        direction: "up" or "down"
        amount: Pixels to scroll
        ref: Optional ref TOKEN to scroll within a specific
             scrollable element. Omit to scroll the whole page.
    """
    
    page = await _get_page()
    
    delta_x = 0
    delta_y = amount if direction == "down" else -amount if direction == "up" else 0
    if direction == "right":
        delta_x = amount
    elif direction == "left":
        delta_x = -amount
        
    result = await page.mouse.wheel(delta_x=delta_x, delta_y=delta_y)
    
    success = getattr(result, "success", True)
    error = getattr(result, "error", None)
    url = getattr(result, "url", None)
    

    step_count = runtime.state.get("step_count", 0) + 1

    if not success:
        return Command(
            update={
                "last_snapshot_taken": False,
                "last_action_result": {"success": False, "error": error, "url": url},
                "step_count": step_count,
                "messages": [
                    ToolMessage(
                        content=f"Scroll FAILED: {error or 'unknown error'}. "
                        f"Call browser_snapshot to see current page state.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return Command(
        update={
            "last_snapshot_taken": False,
            "last_action_result": {"success": True, "error": None, "url": url},
            "step_count": step_count,
            "messages": [
                ToolMessage(
                    content=f"Scrolled {direction} by {amount}px. Call browser_snapshot to verify the result.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
async def browser_extract_content(
    max_length: int | None = None,
    scope: str | None = None,
    runtime: ToolRuntime = None,
) -> Command:
    """Extract readable content from the current page.

    Use this when you need to READ the actual page content after the
    browser agent has completed its task — e.g. an article, search
    results, product details, or other page information.

    If `scope` is provided, it is treated as a CSS selector and only
    that section is extracted.

    If `scope` is not provided, the tool automatically looks for the
    most likely main content region using semantic HTML:
        article → main → [role="main"] → body

    Args:
        max_length: Optional maximum number of characters to return.
        scope: Optional CSS selector to restrict extraction to one section.
               Example: "article" or "#content".
    """

    page = await _get_page()

    try:
        
        if scope:
            locator = page.locator(scope).first

            if await locator.count() > 0:
                content = (await locator.inner_text()).strip()
            else:
                content = ""
                
        else:    
            selectors = [
                "article",
                "main",
                '[role="main"]',
            ]

            best_content = ""

            for selector in selectors:
                locator = page.locator(selector)

                count = await locator.count()

                for i in range(count):
                    candidate = locator.nth(i)

                    try:
                        if not await candidate.is_visible():
                            continue

                        text = (await candidate.inner_text()).strip()

                        if len(text) > len(best_content):
                            best_content = text

                    except Exception:
                        continue


            if len(best_content) >= 300:
                content = best_content
                
            else:
                content = (
                    await page.locator("body").inner_text()
                ).strip()
                
                
        if max_length is not None:
            content = content[:max_length]

        return Command(
            update={
                "last_snapshot": content,
                "last_snapshot_taken": True,
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    except Exception as e:
        return Command(
            update={
                "last_snapshot_taken": False,
                "messages": [
                    ToolMessage(
                        content=f"Content extraction failed: {e}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )


# The full toolset to bind to the agent for the perceive-act-verify loop.
BROWSER_TOOLS = [
    browser_navigate,
    browser_snapshot,
    browser_click,
    browser_type,
    browser_go_back,
    browser_scroll,
    browser_extract_content,
]

# NOTE: call `await close_browser()` when the Browser agent's task ends
# (success, failure, or max-steps cutoff) to release the browser process.
# This is not a tool the model calls — it's invoked by your loop controller.