import asyncio
import json
import os

import litellm
from dotenv import load_dotenv
from mcp import Client

load_dotenv()

# Any LiteLLM model string works, e.g. "groq/llama-3.3-70b-versatile", "gpt-4o-mini"
MODEL = os.getenv("MODEL", "groq/llama-3.3-70b-versatile")
MCP_URL = os.getenv("MCP_URL", "http://localhost:8000/mcp")


async def ask(question: str) -> str:
    # A fresh connection per question: the server keeps no state between them.
    async with Client(MCP_URL) as client:
        # Convert MCP tools to the OpenAI-style format LiteLLM expects
        tools = (await client.list_tools()).tools
        llm_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

        messages = [{"role": "user", "content": question}]
        response = litellm.completion(model=MODEL, messages=messages, tools=llm_tools)
        msg = response.choices[0].message

        # Let the model call MCP tools until it has a final answer
        while msg.tool_calls:
            messages.append(msg)
            for call in msg.tool_calls:
                result = await client.call_tool(
                    call.function.name, json.loads(call.function.arguments)
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result.content[0].text,
                    }
                )
            response = litellm.completion(
                model=MODEL, messages=messages, tools=llm_tools
            )
            msg = response.choices[0].message

        return msg.content


async def main():
    print(await ask("Where is order 1001?"))
    print(await ask("Is order 1002 delayed? When will it arrive?"))


if __name__ == "__main__":
    asyncio.run(main())
