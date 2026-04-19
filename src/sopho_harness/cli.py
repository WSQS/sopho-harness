import argparse
import asyncio
import os
from typing import Any

from agents import (
    Agent,
    ModelResponse,
    RunContextWrapper,
    RunHooks,
    Runner,
    Tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.items import TResponseInputItem
from agents.run_context import AgentHookContext
from dotenv import load_dotenv
from openai import AsyncOpenAI

from sopho_harness.config import SophoHarnessConfig, load_config
from sopho_harness.context import build_instructions
from sopho_harness.tools import build_tools

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


def _shorten(value: Any, limit: int = 200) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


class LoggingRunHooks(RunHooks):
    async def on_agent_start(
        self, context: AgentHookContext[Any], agent: Agent[Any]
    ) -> None:
        print(f"[hook] agent_start: {agent.name}")

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        print(
            f"[hook] llm_start: {agent.name} | input_items={len(input_items)} | "
            f"system_prompt={_shorten(system_prompt)} | input={_shorten(input_items)}"
        )

    async def on_llm_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        response: ModelResponse,
    ) -> None:
        print(
            f"[hook] llm_end: {agent.name} | response_id={response.response_id} | "
            f"output={_shorten(response.output)}"
        )

    async def on_tool_start(
        self, context: RunContextWrapper[Any], agent: Agent[Any], tool: Tool
    ) -> None:
        tool_arguments = getattr(context, "tool_arguments", None)
        print(
            f"[hook] tool_start: {agent.name} -> {tool.name} | "
            f"args={_shorten(tool_arguments)}"
        )

    async def on_tool_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        tool: Tool,
        result: str,
    ) -> None:
        tool_arguments = getattr(context, "tool_arguments", None)
        print(
            f"[hook] tool_end: {agent.name} -> {tool.name} | "
            f"args={_shorten(tool_arguments)} | result={_shorten(result)}"
        )

    async def on_handoff(
        self,
        context: RunContextWrapper[Any],
        from_agent: Agent[Any],
        to_agent: Agent[Any],
    ) -> None:
        print(f"[hook] handoff: {from_agent.name} -> {to_agent.name}")

    async def on_agent_end(
        self, context: AgentHookContext[Any], agent: Agent[Any], output: Any
    ) -> None:
        print(f"[hook] agent_end: {agent.name} | output={_shorten(output)}")


async def run(task_input: str) -> None:
    config = load_config()
    if config is None:
        print("No config found. Please create a .sopho-harness/config.toml file.")
        config = SophoHarnessConfig()
    agent = Agent(
        name="Coding agent",
        instructions=build_instructions(config),
        model=MINIMAX_MODEL,
        tools=build_tools(config),
    )
    result = await Runner.run(
        starting_agent=agent,
        input=task_input,
        max_turns=100,
        hooks=LoggingRunHooks(),
    )
    print(result.final_output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task_input",
        nargs="*",
        help="Task input for the coding agent.",
    )
    args = parser.parse_args()
    task_input = " ".join(args.task_input).strip() or "Code review this python project."
    asyncio.run(run(task_input))


if __name__ == "__main__":
    main()
