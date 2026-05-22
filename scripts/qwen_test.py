"""
Qwen3 (via Ollama) + web-search-mcp agent loop.

Spawns the MCP server as a subprocess, exposes its tools to Qwen3 via
Ollama's OpenAI-compatible tool-calling, and runs the conversation loop.
"""

import asyncio
import json
from contextlib import AsyncExitStack

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- Config ---------------------------------------------------------------

MCP_SERVER_PATH = "/Users/connorott/PycharmProjects/rag/web-search-mcp/dist/index.js"
OLLAMA_MODEL = "qwen3:4b"
MAX_TURNS = 10  # hard cap on tool-call rounds per user query

SYSTEM_PROMPT = """You are a helpful research assistant with access to web search tools.

When the user asks about current events, recent information, or anything you're \
uncertain about, use the search tools. Prefer `get-web-search-summaries` for quick \
lookups and `full-web-search` when you need full page contents. Use \
`get-single-web-page-content` when you already have a specific URL.

After gathering information, synthesize a clear answer with sources."""


# --- MCP <-> OpenAI schema translation ------------------------------------

def mcp_tool_to_openai(tool) -> dict:
    """Convert an MCP Tool object to OpenAI/Ollama tool-calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


def extract_text(mcp_result) -> str:
    """Flatten an MCP CallToolResult into a string Qwen3 can consume."""
    chunks = []
    for block in mcp_result.content:
        if hasattr(block, "text"):
            chunks.append(block.text)
        else:
            chunks.append(str(block))
    return "\n".join(chunks) if chunks else "(no content)"


# --- Agent loop -----------------------------------------------------------

async def run_agent(user_query: str):
    server_params = StdioServerParameters(
        command="node",
        args=[MCP_SERVER_PATH],
        env={
            "MAX_CONTENT_LENGTH": "10000",   # keep results context-friendly
            "BROWSER_HEADLESS": "true",
            "MAX_BROWSERS": "2",
            "DEFAULT_TIMEOUT": "8000",
        },
    )

    async with AsyncExitStack() as stack:
        # 1. Start MCP server subprocess + open session
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        # 2. Discover tools and translate schemas
        tool_list = await session.list_tools()
        ollama_tools = [mcp_tool_to_openai(t) for t in tool_list.tools]
        print(f"[mcp] Loaded {len(ollama_tools)} tools: "
              f"{[t['function']['name'] for t in ollama_tools]}\n")

        # 3. Conversation state
        client = ollama.AsyncClient()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]

        for turn in range(MAX_TURNS):
            response = await client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                tools=ollama_tools,
                options={"temperature": 0.3},
                # Qwen3 thinks a lot; disable for cleaner tool calls. Remove
                # this line if you want the reasoning traces.
                think=False,
            )

            msg = response["message"]
            messages.append(msg)

            tool_calls = msg.get("tool_calls") or []

            # No tool calls? Model gave the final answer.
            if not tool_calls:
                print(f"\n=== Final answer ===\n{msg.get('content', '')}\n")
                return msg.get("content", "")

            # Dispatch every requested tool call (Qwen3 supports parallel calls)
            for call in tool_calls:
                fn_name = call["function"]["name"]
                fn_args = call["function"]["arguments"]
                if isinstance(fn_args, str):
                    fn_args = json.loads(fn_args)

                print(f"[turn {turn}] → {fn_name}({json.dumps(fn_args)[:120]}…)")

                try:
                    result = await session.call_tool(fn_name, fn_args)
                    content = extract_text(result)
                except Exception as e:
                    content = f"ERROR calling {fn_name}: {e!r}"
                    print(f"[turn {turn}] ✗ {content}")

                # Feed result back as a tool message
                messages.append({
                    "role": "tool",
                    "content": content[:15000],  # belt-and-braces truncation
                    "name": fn_name,
                })

        print("\n[!] Hit MAX_TURNS without a final answer.")
        return None


# --- Entry point ----------------------------------------------------------

if __name__ == "__main__":
    query = "What were the major AI model releases in the last month? Give me sources."
    asyncio.run(run_agent(query))