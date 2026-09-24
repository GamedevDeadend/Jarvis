"""
Raw browser tools for the Jarvis Browser Agent.

Wraps browsegrab's Python API directly (not via its MCP server) so we get
full code-level control — specifically, the ability to enforce the
"never trust the model's self-reported done" rule from outside the LLM.

I deliberately use ONLY this manual-control API, never session.browse(...).
session.browse() is browsegrab's own built-in agentic loop — it has its own
LLM call and its own "done" self-termination, which is exactly the
unverified-completion problem we're trying to solve.

This is an ASYNC API. Tools below use LangChain's async `@tool` support
(coroutine functions) accordingly — not sync wrappers.
"""

from __future__ import annotations

from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command

from browsegrab import BrowseSession
from browsegrab.browser.manager import BrowserManager
from browsegrab.config import BrowseGrabConfig

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

import os
import re

from dotenv import load_dotenv
load_dotenv()

browser_profile_path = os.getenv("BROWSER_PROFILE_PATH")
browser_executable_path = os.getenv("BROWSER_EXECUTABLE_PATH")


config = BrowseGrabConfig()
config.snapshot.filter_interactive_only = True
# config.snapshot.max_snapshot_length = 3000
# config.snapshot.max_content_length = 1000
config.browser.headless = False

session = None

async def optimised_snapshot():
    session = await _get_session()
    page = await session._ensure_page()
    snapshot = await page.locator("body").aria_snapshot(depth=6, mode="ai")

    snapshot = re.sub( r"\s*\[cursor=[^\]]*\]", "",snapshot) #Removing [cursor=...]
    snapshot = re.sub(r"(?m)^[\t\s]*-[\s]*\/url.*\n?\r?", "", snapshot) #Removing [urls]
    snapshot = re.sub(r"(?m)^[\t\s]*-\s*(generic|listitem).+\n?\r?", "", snapshot) #Removing [generic, listitem]


    return snapshot



def overriding_browser_manager_class() : 
    """
    Overriding default methods of BrowserMangerClass
    to make it compatible with Lauch_Persistent_Context (Browser launch with Saved profiels)
    """


    async def _ensure_browser(self)->Browser : 
        """Lazily launch browser on first use."""

        if self._browser is None:
            if self._playwright is None:
                self._playwright = await async_playwright().start()

            self._browser = await self._playwright.chromium.launch_persistent_context(user_data_dir=browser_profile_path, executable_path=browser_executable_path, headless=config.browser.headless, args=["--profile-directory=Profile 2"] )

        return self._browser

    async def new_context(self, **kw):
        return await self._ensure_browser()

    async def new_page(self, **kw):
        context = await self.new_context()
        page = await context.new_page()
        page.set_default_timeout(self.config.timeout_ms)

        return page

    async def close(self):
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    BrowserManager._ensure_browser = _ensure_browser
    BrowserManager.new_context = new_context
    BrowserManager.new_page = new_page
    BrowserManager.close = close


overriding_browser_manager_class()


async def _get_session() -> BrowseSession:
    """Lazily open the browsegrab session. One per agent run."""
    global session
    if session is None:
        session = BrowseSession(config=config)
        await session.__aenter__()

    return session


async def close_session() -> None:
    """Explicitly close the shared session. Call this when the Browser
    agent's task graph finishes (success, failure, or max-steps cutoff) —
    otherwise the browser process is left running.
    """
    global session
    if session is not None:
        await session.close()
        session = None


@tool
async def browser_navigate(runtime: ToolRuntime, url: str) -> Command:
    """Navigate the browser to a given URL.

    Args:
        url: Full URL to navigate to, including https://
    """
    session = await _get_session()
    result = await session.navigate(url)
    success = getattr(result, "success", True)
    error = getattr(result, "error", None)

    step_count = runtime.state.get("step_count", 0) + 1

    if not success:
        return Command(
            update={
                "last_snapshot_taken": False,
                "last_action_result": {"success": False, "error": error, "url": None},
                "step_count": step_count,
                "messages": [
                    ToolMessage(
                        content=f"Navigation to {url} FAILED: {error or 'unknown error'}",
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
                    content=f"Navigated to {url}. Call browser_snapshot to see the page state.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
async def browser_snapshot(runtime: ToolRuntime) -> Command:
    """Take a snapshot of the current page's accessibility tree.

    Returns a compact, ref-based (e1, e2, ...) representation of visible
    interactive elements and content, optimized for small local LLMs.
    Each element is shown with its ref token in brackets, e.g.:
        searchbox "Search Wikipedia" [ref=e11]
        button "Search" [ref=e90]

    This is the ONLY source of truth for what is actually on the page.
    Always call this after an action to verify what happened — never
    assume an action succeeded without checking.
    """
    session = await _get_session()
    snap = await session.snapshot()
    snapshot_text = snap.tree_text

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
    session = await _get_session()
    result = await session.click(ref)
    success = getattr(result, "success", True)
    error = getattr(result, "error", None)
    url = getattr(result, "url", None)

    step_count = runtime.state.get("step_count", 0) + 1

    if not success:
        # Honest failure report — do NOT claim the click happened.
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
async def browser_type(ref: str, text: str, submit: bool = True, runtime: ToolRuntime = None) -> Command:
    """ Type text into an input element on the page by its ref.
        To perform a search: ALWAYS set submit=True in the SAME call that types
        the search text. Do NOT type first and click a separate button afterward
        that is unreliable and unnecessary. This tool submits the form/search
        directly via submit=True.
    Args:
        ref: The ref TOKEN from the most recent browser_snapshot output
             text: Text to type into the element
             
        submit: Set True to submit immediately after typing (e.g. pressesEnter).
                Use this for ALL searches — do not click a search
                button separately.
    """
    session = await _get_session()
    result = await session.type(ref, text, submit=submit)
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
    session = await _get_session()
    result = await session.go_back()
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
    """Scroll the page (or a specific scrollable element) up or down.

    Use this when the content you need isn't visible in the current
    snapshot — e.g. the article text is below the fold. Always follow
    with browser_snapshot to see what's now visible.

    Args:
        direction: "up" or "down"
        amount: Pixels to scroll
        ref: Optional ref TOKEN to scroll within a specific
             scrollable element. Omit to scroll the whole page.
    """
    session = await _get_session()
    result = await session.scroll(direction=direction, amount=amount, ref=ref)
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
    max_length: int | None = None, scope: str | None = None, runtime: ToolRuntime = None
) -> Command:
    """Extract the FULL text content of the current page (compressed DOM +
    markdown), not just the visible accessibility tree.

    Use this when you need to actually READ the content of a page — e.g.
    an article's body text, a search result's details — rather than just
    see what elements are clickable. browser_snapshot shows structure and
    interactive elements; this shows the actual readable content.

    This is the tool to use before answering any question that asks you
    to summarize, quote, or report information FROM the page. Never
    answer from browser_snapshot output alone if the question is about
    page content, not page structure.

    Args:
        max_length: Optional cap on returned content length
        scope: Optional CSS selector to limit extraction to one section
               (e.g. the main article body, skipping nav/footer)
    """
    session = await _get_session()
    content = await session.extract_content(max_length=max_length, scope=scope)

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

# NOTE: call `await close_session()` when the Browser agent's task ends
# (success, failure, or max-steps cutoff) to release the browser process.
# This is not a tool the model calls — it's invoked by your loop controller.