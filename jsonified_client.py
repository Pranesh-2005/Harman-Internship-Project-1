import asyncio
import json
import os
import sys
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

from openai import OpenAI
from dotenv import load_dotenv

# ✅ NEW
import sqlite3
from datetime import datetime

load_dotenv()


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        self.openai_client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY")
        )

        # ✅ DB INIT
        self.db = sqlite3.connect("logs.db", check_same_thread=False)
        self.cursor = self.db.cursor()

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_query TEXT,
            response TEXT,
            tools_used TEXT
        )
        """)
        self.db.commit()

    # ✅ LOG FUNCTION
    def log_to_db(self, query, response, tools_used):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "response": response,
            "tools_used": tools_used
        }

        print("\n[LOG]")
        print(json.dumps(log_entry, indent=2))

        self.cursor.execute("""
            INSERT INTO query_logs (timestamp, user_query, response, tools_used)
            VALUES (?, ?, ?, ?)
        """, (
            log_entry["timestamp"],
            log_entry["query"],
            log_entry["response"],
            json.dumps(log_entry["tools_used"])
        ))

        self.db.commit()

    async def connect_to_sse_server(self, server_url: str):
        """Connect to MCP SSE server"""

        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        await self.session.initialize()

        print("Initialized SSE client...")
        response = await self.session.list_tools()
        tools = response.tools

        print("\nConnected tools:", [tool.name for tool in tools])

    async def cleanup(self):
        if hasattr(self, '_session_context'):
            await self._session_context.__aexit__(None, None, None)
        if hasattr(self, '_streams_context'):
            await self._streams_context.__aexit__(None, None, None)

    async def process_query(self, query: str) -> str:
        """Process query with tool-calling loop"""

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can use tools to query databases."
            },
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()

        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool in response.tools
        ]

        final_text = []
        tools_used = []  # ✅ TRACK TOOLS

        max_rounds = 20

        for round_num in range(max_rounds):

            print(f"\n--- Round {round_num + 1} ---")

            try:
                response = self.openai_client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=messages,
                    tools=available_tools,
                    tool_choice="auto"
                )
            except Exception as e:
                print("LLM error:", e)
                break

            msg = response.choices[0].message

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": msg.tool_calls
            })

            if msg.content:
                final_text.append(msg.content)
                print("Assistant:", msg.content)

            if msg.tool_calls:

                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    tools_used.append(tool_name)  # ✅ TRACK

                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except:
                        continue

                    print(f"\nCalling tool: {tool_name} | Args: {tool_args}")

                    try:
                        result = await self.session.call_tool(tool_name, tool_args)

                        if isinstance(result.content, list):
                            result_content = "\n".join(str(x) for x in result.content)
                        else:
                            result_content = str(result.content)

                    except Exception as e:
                        result_content = f"Tool error: {e}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_content
                    })

                    print("Tool result:", result_content[:200])

                continue

            break

        final_response = "\n".join(final_text) if final_text else "No response"

        # ✅ LOG EVERYTHING
        self.log_to_db(query, final_response, tools_used)

        return final_response

    async def chat_loop(self):
        print("\nMCP Client Started!")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() in ["quit", "exit"]:
                    break

                if not query:
                    continue

                # ✅ INPUT JSON
                print("\n[INPUT JSON]")
                print(json.dumps({"query": query}, indent=2))

                response = await self.process_query(query)

                # ✅ OUTPUT JSON
                print("\n[OUTPUT JSON]")
                print(json.dumps({"response": response}, indent=2))

            except KeyboardInterrupt:
                break

            except Exception as e:
                print("Error:", e)


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <SSE URL>")
        sys.exit(1)

    client = MCPClient()

    try:
        await client.connect_to_sse_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())