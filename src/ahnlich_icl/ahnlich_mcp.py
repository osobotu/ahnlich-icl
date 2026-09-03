import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from ahnlich_icl.spider import Example

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ToolResult = dict[str, Any] | list[dict[str, Any]]


@asynccontextmanager
async def open_ahnlich_session() -> AsyncGenerator[ClientSession, None]:
    server = StdioServerParameters(
        command="ahnlich-mcp",
        args=["--profile", "ai"],
    )

    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            yield session

async def call_ahnlich_tool(
    name: str,
    arguments: dict[str, Any],
) -> ToolResult:
    async with open_ahnlich_session() as session:
        response = await session.call_tool(name, arguments)

    if response.isError:
        raise RuntimeError(_response_text(response.content))

    if response.structuredContent is not None:
        return response.structuredContent

    return json.loads(_response_text(response.content))


async def ping_ahnlich() -> ToolResult:
    return await call_ahnlich_tool("ping", {})

async def create_store(store_name: str) -> ToolResult:
    return await call_ahnlich_tool(
        "create_store",
        {
            "store_name": store_name,
            "error_if_exists": False,
        },
    )

async def store_examples(
    store_name: str,
    examples: list[Example],
) -> ToolResult:
    return await call_ahnlich_tool(
        "store_entries",
        {
            "store_name": store_name,
            "entries": [_text_entry(example) for example in examples],
        },
    )

async def similarity_search(
    store_name: str,
    question: str,
    k: int = 5,
) -> list[dict[str, Any]]:
    result = await call_ahnlich_tool(
        "similarity_search",
        {
            "store_name": store_name,
            "query": question,
            "top_k": k,
            "algorithm": "cosine",
        },
    )

    if isinstance(result, dict):
        result = result.get("result")

    if not isinstance(result, list):
        raise RuntimeError("Ahnlich MCP returned unexpected search results")

    return result

async def drop_store(store_name: str) -> ToolResult:
    return await call_ahnlich_tool(
        "drop_store",
        {
            "store_name": store_name,
            "error_if_not_exists": False,
        },
    )

def _response_text(content: list[Any]) -> str:
    text = [item.text for item in content if hasattr(item, "text")]
    if not text:
        raise RuntimeError("Ahnlich MCP returned no text content")
    return "\n".join(text)

def _text_entry(example: Example) -> dict[str, Any]:
    metadata = {
        "example_id": example.example_id,
        "db_id": example.db_id,
        "sql": example.sql,
    }

    if example.difficulty is not None:
        metadata["difficulty"] = example.difficulty

    return {
        "content": example.question,
        "metadata": metadata,
    }
