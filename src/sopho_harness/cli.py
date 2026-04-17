import asyncio
import os
from pathlib import Path

from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

MINIMAX_BASE_URL = "https://api.minimax.chat/v1"
MINIMAX_MODEL = "MiniMax-M2.7-highspeed"

minimax_api_key = os.getenv("MINIMAX_API_KEY")
if not minimax_api_key:
    raise RuntimeError("MINIMAX_API_KEY is not set in .env or environment variables")

set_default_openai_client(
    AsyncOpenAI(
        api_key=minimax_api_key,
        base_url=MINIMAX_BASE_URL,
    ),
    use_for_tracing=False,
)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)


@function_tool
def read_file(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return f"File not found: {file_path}"
    return file_path.read_text(encoding="utf-8")


@function_tool
def write_patch(path: str, content: str) -> str:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"Wrote file: {file_path}"


agent = Agent(
    name="Coding agent",
    instructions="You are a helpful coding agent. Solve programming tasks clearly, accurately, and concisely.",
    model=MINIMAX_MODEL,
    tools=[read_file, write_patch],
)


async def run() -> None:
    result = await Runner.run(
        starting_agent=agent,
        input="Write a Python function that returns the factorial of a non-negative integer in a file test.py.",
        max_turns=10,
    )
    print(result.final_output)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
