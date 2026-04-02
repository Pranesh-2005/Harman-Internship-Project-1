import asyncio
import json
import os
import sys
import sqlite3
from datetime import datetime
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        self.openai_client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=os.getenv("NEBIUS_API_KEY")
        )

        # Initialize SQLite memory
        self.db = sqlite3.connect("memory.db")
        self.cursor = self.db.cursor()

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            intent TEXT,
            query TEXT,
            sql TEXT,
            tools TEXT,
            created_at TEXT
        )
        """)
        self.db.commit()

    async def connect_to_sse_server(self, server_url: str):
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session = await self._session_context.__aenter__()

        await self.session.initialize()

        response = await self.session.list_tools()
        print("Connected tools:", [tool.name for tool in response.tools])

    async def cleanup(self):
        if hasattr(self, '_session_context'):
            await self._session_context.__aexit__(None, None, None)
        if hasattr(self, '_streams_context'):
            await self._streams_context.__aexit__(None, None, None)

    # ---------------- MEMORY ---------------- #

    def extract_intent(self, query: str) -> str:
        q = query.lower()
        return "general"

    def store_memory(self, query, tools_used, messages):
        sql_query = None

        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tool in msg["tool_calls"]:
                    if tool.function.name in ["execute_query", "plot_query_results"]:
                        try:
                            args = json.loads(tool.function.arguments)
                            sql_query = args.get("query")
                        except:
                            pass

        if not sql_query:
            return

        intent = self.extract_intent(query)

        self.cursor.execute("""
        INSERT INTO memory (intent, query, sql, tools, created_at)
        VALUES (?, ?, ?, ?, ?)
        """, (
            intent,
            query,
            sql_query,
            json.dumps(tools_used),
            datetime.utcnow().isoformat()
        ))

        self.db.commit()

    def retrieve_memory(self, query: str) -> str:
        intent = self.extract_intent(query)

        self.cursor.execute("""
        SELECT sql FROM memory
        WHERE intent = ?
        ORDER BY id DESC
        LIMIT 2
        """, (intent,))

        rows = self.cursor.fetchall()

        if not rows:
            return ""

        return "\n\n".join([r[0] for r in rows])

    # ---------------- MAIN LOGIC ---------------- #

    async def process_query(self, query: str) -> str:

        print("\n[INPUT JSON]")
        print(json.dumps({"query": query}, indent=2))

        memory_context = self.retrieve_memory(query)

        messages = [
            {
                "role": "system",
                "content": f"""
                    You are a SQL Server expert.

                    Use tools to answer queries.

                    If similar queries exist, reuse SQL patterns.

                    Previous queries:
                    {memory_context}

                    Avoid unnecessary exploration if pattern exists.
                    """
            },
            {"role": "user", "content": query}
        ]

        response = await self.session.list_tools()

        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        final_text = []
        tools_used = []

        for round_num in range(10):
            print(f"\n--- Round {round_num + 1} ---")

            llm_response = self.openai_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                tools=available_tools,
                tool_choice="auto"
            )

            assistant_message = llm_response.choices[0].message

            msg_dict = {
                "role": "assistant",
                "content": assistant_message.content
            }

            if assistant_message.tool_calls:
                msg_dict["tool_calls"] = assistant_message.tool_calls

            messages.append(msg_dict)

            if assistant_message.content:
                print("Assistant:", assistant_message.content)
                final_text.append(assistant_message.content)

            if not assistant_message.tool_calls:
                break

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                tools_used.append(tool_name)

                print(f"\nCalling tool: {tool_name} | Args: {tool_args}")

                result = await self.session.call_tool(tool_name, tool_args)

                if isinstance(result.content, list):
                    result_content = "\n".join(str(x) for x in result.content)
                else:
                    result_content = str(result.content)

                print("Tool result:", result_content[:200])

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_content
                })

        # Store memory after execution
        self.store_memory(query, tools_used, messages)

        final_output = "\n".join(final_text)

        print("\n[OUTPUT JSON]")
        print(json.dumps({"response": final_output}, indent=2))

        return final_output


# ---------------- CLI ---------------- #

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <SSE_URL>")
        sys.exit(1)

    client = MCPClient()

    try:
        await client.connect_to_sse_server(sys.argv[1])

        print("\nMCP Client Started")

        while True:
            query = input("\nQuery: ")

            if query.lower() in ["exit", "quit"]:
                break

            response = await client.process_query(query)
            print("\nResponse:", response)

    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())