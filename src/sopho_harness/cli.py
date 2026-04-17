
import asyncio
import os

from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
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

agent = Agent(
    name="History tutor",
    instructions="You answer history questions clearly and concisely.",
    model=MINIMAX_MODEL,
)


async def run() -> None:
    result = await Runner.run(agent, "When did the Roman Empire fall?")
    print(result.final_output)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
