import argparse
import asyncio
import os
from pathlib import Path
from typing import Any, Literal

from agents import (
    Agent,
    ModelResponse,
    RunContextWrapper,
    RunHooks,
    Runner,
    SQLiteSession,
    Tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.items import TResponseInputItem
from agents.run_context import AgentHookContext
from dotenv import load_dotenv
from openai import AsyncOpenAI

from sopho_harness.agent.clarify import get_clarify_agent
from sopho_harness.agent.context import get_context_agent
from sopho_harness.agent.implement import get_implement_agent
from sopho_harness.agent.plan import get_plan_agent
from sopho_harness.agent.profile import get_profile_agent
from sopho_harness.agent.review import get_review_agent
from sopho_harness.agent.verify import get_verify_agent
from sopho_harness.config import SophoHarnessConfig, load_config
from sopho_harness.context import build_instructions
from sopho_harness.tools import build_tools

load_dotenv()

openai_model = os.getenv("OPENAI_MODEL")
if not openai_model:
    raise RuntimeError("OPENAI_MODEL is not set in .env or environment variables")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in .env or environment variables")

openai_base_url = os.getenv("OPENAI_BASE_URL")
openai_api_mode_raw = os.getenv("OPENAI_API_MODE", "responses")
if openai_api_mode_raw == "chat_completions":
    openai_api_mode: Literal["chat_completions", "responses"] = "chat_completions"
elif openai_api_mode_raw == "responses":
    openai_api_mode = "responses"
else:
    raise RuntimeError(
        "OPENAI_API_MODE must be either 'chat_completions' or 'responses'"
    )

set_default_openai_client(
    AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_base_url,
    )
)
set_default_openai_api(openai_api_mode)
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
    session_dir = Path(".sopho-harness")
    session_dir.mkdir(parents=True, exist_ok=True)
    session = SQLiteSession(
        session_id="default",
        db_path=session_dir / "session.db",
    )
    profile_agent, profile_input = get_profile_agent()
    profile_agent.model = openai_model
    result = await Runner.run(
        starting_agent=profile_agent,
        input=profile_input,
        max_turns=100,
        hooks=LoggingRunHooks(),
        # session=session,
    )
    profile = result.final_output.to_human()
    print("Profile Agent Result:\n", profile)
    clarify_agent, clarify_input = get_clarify_agent()
    clarify_agent.model = openai_model
    result = await Runner.run(
        starting_agent=clarify_agent,
        input=clarify_input + profile + task_input,
        max_turns=100,
        hooks=LoggingRunHooks(),
        # session=session,
    )
    clarify = result.final_output.to_human()
    print("Clarify Agent Result:\n", clarify)
    context_agent, context_input = get_context_agent()
    context_agent.model = openai_model
    result = await Runner.run(
        starting_agent=context_agent,
        input=clarify_input + profile + clarify + context_input,
        max_turns=100,
        hooks=LoggingRunHooks(),
        # session=session,
    )
    context = result.final_output.to_human()
    print("Context Agent Result:\n", context)
    plan_agent, plan_input = get_plan_agent()
    plan_agent.model = openai_model
    result = await Runner.run(
        starting_agent=plan_agent,
        input=plan_input + profile + clarify + context,
        max_turns=100,
        hooks=LoggingRunHooks(),
        # session=session,
    )
    plan = result.final_output.to_human()
    print("Plan Agent Result:\n", plan)
    implement_agent, implement_input = get_implement_agent()
    implement_agent.model = openai_model
    result = await Runner.run(
        starting_agent=implement_agent,
        input=implement_input + clarify + context + plan,
        max_turns=100,
        hooks=LoggingRunHooks(),
        # session=session,
    )
    implementation = result.final_output.to_human()
    print("Implement Agent Result:\n", implementation)
    verify_agent, verify_input = get_verify_agent()
    verify_agent.model = openai_model
    result = await Runner.run(
        starting_agent=verify_agent,
        input=verify_input + plan + implementation,
        max_turns=100,
        hooks=LoggingRunHooks(),
        # session=session,
    )
    verification = result.final_output.to_human()
    print("Verify Agent Result:\n", verification)
    review_agent, review_input = get_review_agent()
    review_agent.model = openai_model
    result = await Runner.run(
        starting_agent=review_agent,
        input=review_input + profile + clarify + context + plan + implementation + verification,
        max_turns=100,
        hooks=LoggingRunHooks(),
        # session=session,
    )
    review = result.final_output.to_human()
    print("Review Agent Result:\n", review)


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
