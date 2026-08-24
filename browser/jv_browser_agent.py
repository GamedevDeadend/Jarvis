# jv_browser_agent.py

import asyncio
import logging
import time
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq


# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jv_browser_agent")


EXCLUDED_TOOLS = {"browser_run_code_unsafe"}


def get_browser_llm(is_simple_query: bool):
    if is_simple_query:
        logger.info("Using local Ollama model (ibm/granite4.1:8b) — simple query")
        return ChatOllama(
            model="ibm/granite4.1:8b",
            temperature=0,
            num_predict=512,
            streaming=False,
        )
    else:
        logger.info("Using Groq cloud model (openai/gpt-oss-120b) — complex query")
        return ChatOllama(
            model="ibm/granite4.1:8b",
            temperature=0,
            num_predict=512,
            streaming=False,
        )


def _truncate(text: str, limit: int = 300) -> str:
    """Shorten long tool results for log readability."""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text)} chars total]"


async def browser_agent_loop(goal: str, is_simple_query: bool = False, max_steps: int = 10):
    logger.info(f"Starting browser_agent_loop | goal='{goal}' | max_steps={max_steps}")
    start_time = time.monotonic()

    client = MultiServerMCPClient({
        "playwright": {
            "transport": "stdio",
            "command": "npx",
            "args": ["@playwright/mcp@latest", "--browser=chrome"],
        }
    })

    try:
        async with client.session("playwright") as session:
            logger.info("MCP session established")

            tools = await load_mcp_tools(session)
            tools = [t for t in tools if t.name not in EXCLUDED_TOOLS]
            logger.info(f"Loaded {len(tools)} tools (excluded: {EXCLUDED_TOOLS})")

            llm = get_browser_llm(is_simple_query)
            llm_with_tools = llm.bind_tools(tools)

            messages = [
                {"role": "system", "content": (
                    "You are a browser automation agent. You will be given a goal. "
                    "Use the available tools to accomplish it step by step. "
                    "Always call browser_navigate first if no page is loaded yet, "
                    "and browser_snapshot after navigating or acting, to see the "
                    "current state of the page before deciding your next action. "
                    "When the goal is complete, respond with no tool calls and a "
                    "final text summary of what you did."
                )},
                {"role": "user", "content": goal}
            ]

            for step in range(max_steps):
                step_start = time.monotonic()
                logger.info(f"--- Step {step} ---")

                response = await llm_with_tools.ainvoke(messages)
                messages.append(response)

                llm_elapsed = time.monotonic() - step_start
                logger.info(f"LLM responded in {llm_elapsed:.2f}s")

                if not response.tool_calls:
                    total_elapsed = time.monotonic() - start_time
                    logger.info(f"No tool calls — goal complete. Total time: {total_elapsed:.2f}s")
                    logger.info(f"Final response: {_truncate(response.content)}")
                    return response.content

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    logger.info(f"Tool call: {tool_name} | args={tool_args}")

                    tool = next((t for t in tools if t.name == tool_name), None)
                    if tool is None:
                        logger.error(f"LLM requested unknown tool '{tool_name}' — skipping")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": f"Error: tool '{tool_name}' not found"
                        })
                        continue

                    tool_start = time.monotonic()
                    try:
                        result = await tool.ainvoke(tool_args)
                        tool_elapsed = time.monotonic() - tool_start
                        logger.info(f"Tool '{tool_name}' completed in {tool_elapsed:.2f}s | result: {_truncate(result)}")
                    except Exception as e:
                        tool_elapsed = time.monotonic() - tool_start
                        logger.error(f"Tool '{tool_name}' failed after {tool_elapsed:.2f}s: {e}")
                        result = f"Error executing {tool_name}: {e}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(result)
                    })

            total_elapsed = time.monotonic() - start_time
            logger.warning(f"Reached max_steps ({max_steps}) without completing. Total time: {total_elapsed:.2f}s")
            return "Reached max steps without completing the goal."

    except Exception as e:
        logger.exception(f"browser_agent_loop failed with unhandled exception: {e}")
        raise


if __name__ == "__main__":
    result = asyncio.run(browser_agent_loop(
        goal="Search on Wikipedia about Mongols",
        is_simple_query=True,
        max_steps=6
    ))
    print("\nFINAL RESULT:", result)