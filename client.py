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

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai_client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY")
        )

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        # Store the context managers so they stay alive
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # Initialize
        await self.session.initialize()

        # List available tools to verify connection
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if hasattr(self, '_session_context') and self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if hasattr(self, '_streams_context') and self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI with multiple rounds of tool calling"""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can use various tools to answer questions about databases and data. You can call multiple tools in sequence to complete complex tasks. Don't use reasoning - just execute the tools needed."
            },
            {
                "role": "user",
                "content": query
            }
        ]

        # Get available tools from MCP server
        response = await self.session.list_tools()
        available_tools = []
        
        for tool in response.tools:
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            available_tools.append(tool_def)

        final_text = []
        max_rounds = 10  # Prevent infinite loops
        round_count = 0

        while round_count < max_rounds:
            round_count += 1
            print(f"\n--- Round {round_count} ---")
            
            # OpenAI API call
            try:
                response = self.openai_client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=messages,
                    tools=available_tools if available_tools else None,
                    tool_choice="auto" if available_tools else None
                )
            except Exception as e:
                print(f"OpenAI API error: {e}")
                break

            assistant_message = response.choices[0].message
            
            # Add assistant's response to messages
            message_to_add = {
                "role": "assistant",
                "content": assistant_message.content
            }
            
            if assistant_message.tool_calls:
                message_to_add["tool_calls"] = assistant_message.tool_calls
            
            messages.append(message_to_add)

            if assistant_message.content:
                final_text.append(assistant_message.content)
                print(f"Assistant: {assistant_message.content}")

            # Handle tool calls if any
            if assistant_message.tool_calls:
                tool_calls_made = False
                
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        print(f"Invalid JSON in tool arguments: {tool_call.function.arguments}")
                        continue
                    
                    print(f"\n[Calling tool {tool_name} with args {tool_args}]")
                    
                    # Execute tool call via MCP
                    try:
                        result = await self.session.call_tool(tool_name, tool_args)
                        
                        # Handle different result content types
                        if hasattr(result, 'content'):
                            if isinstance(result.content, list):
                                tool_result_content = "\n".join(str(item) for item in result.content)
                            else:
                                tool_result_content = str(result.content)
                        else:
                            tool_result_content = str(result)
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "content": tool_result_content,
                            "tool_call_id": tool_call.id
                        })
                        
                        print(f"Tool result: {tool_result_content[:200]}...")
                        tool_calls_made = True
                        
                    except Exception as e:
                        error_msg = f"Error executing tool {tool_name}: {str(e)}"
                        print(f"Tool error: {error_msg}")
                        messages.append({
                            "role": "tool",
                            "content": error_msg,
                            "tool_call_id": tool_call.id
                        })
                        tool_calls_made = True
                
                # If tool calls were made, continue the loop for potential additional tool calls
                if tool_calls_made:
                    continue
            
            # No tool calls made in this round, we're done
            break

        return "\n".join(final_text) if final_text else "No response generated."

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        print("Example queries:")
        print("- 'First say hello, then list all databases'")
        print("- 'Connect and show me all available tables'")
        print("- 'List tables, then show schema for Customer table'")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not query:
                    continue
                    
                response = await self.process_query(query)
                print(f"\nResponse: {response}")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <URL of SSE MCP server>")
        print("Example: python client.py http://localhost:8080/sse")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_sse_server(server_url=sys.argv[1])
        await client.chat_loop()
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())