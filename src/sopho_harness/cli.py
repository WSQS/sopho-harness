import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
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

from sopho_harness.agent.clarify import (
    ClarifiedTask,
    build_clarify_input,
    get_clarify_agent,
)
from sopho_harness.agent.context import build_context_input, get_context_agent
from sopho_harness.agent.implement import build_implement_input, get_implement_agent
from sopho_harness.agent.plan import build_plan_input, get_plan_agent
from sopho_harness.agent.profile import build_profile_input, get_profile_agent
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


class RunRecorder:
    def __init__(self, run_id: str, run_dir: Path) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def record(self, stage: str, payload: dict[str, Any]) -> None:
        record_payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "stage": stage,
            **payload,
        }
        file_path = self.run_dir / f"{stage}.jsonl"
        with file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record_payload, ensure_ascii=False) + "\n")


class LoggingRunHooks(RunHooks):
    def __init__(self, recorder: RunRecorder, stage: str) -> None:
        self.recorder = recorder
        self.stage = stage

    def _record(self, payload: dict[str, Any]) -> None:
        self.recorder.record(self.stage, payload)

    async def on_agent_start(
        self, context: AgentHookContext[Any], agent: Agent[Any]
    ) -> None:
        print(f"[hook] agent_start: {agent.name}")
        self._record({"agent": agent.name, "event": "agent_start"})

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
        self._record(
            {
                "agent": agent.name,
                "event": "llm_start",
                "system_prompt": system_prompt,
                "input_items": str(input_items),
            }
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
        self._record(
            {
                "agent": agent.name,
                "event": "llm_end",
                "response_id": response.response_id,
                "output": str(response.output),
            }
        )

    async def on_tool_start(
        self, context: RunContextWrapper[Any], agent: Agent[Any], tool: Tool
    ) -> None:
        tool_arguments = getattr(context, "tool_arguments", None)
        print(
            f"[hook] tool_start: {agent.name} -> {tool.name} | "
            f"args={_shorten(tool_arguments)}"
        )
        self._record(
            {
                "agent": agent.name,
                "event": "tool_start",
                "tool": tool.name,
                "arguments": tool_arguments,
            }
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
        self._record(
            {
                "agent": agent.name,
                "event": "tool_end",
                "tool": tool.name,
                "arguments": tool_arguments,
                "result": result,
            }
        )

    async def on_handoff(
        self,
        context: RunContextWrapper[Any],
        from_agent: Agent[Any],
        to_agent: Agent[Any],
    ) -> None:
        print(f"[hook] handoff: {from_agent.name} -> {to_agent.name}")
        self._record(
            {
                "agent": from_agent.name,
                "event": "handoff",
                "to_agent": to_agent.name,
            }
        )

    async def on_agent_end(
        self, context: AgentHookContext[Any], agent: Agent[Any], output: Any
    ) -> None:
        print(f"[hook] agent_end: {agent.name} | output={_shorten(output)}")
        self._record({"agent": agent.name, "event": "agent_end", "output": str(output)})


async def run(task_input: str) -> None:
    config = load_config()
    if config is None:
        print("No config found. Please create a .sopho-harness/config.toml file.")
        config = SophoHarnessConfig()
    session_dir = Path(".sopho-harness")
    session_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = session_dir / "runs" / run_id
    recorder = RunRecorder(run_id=run_id, run_dir=run_dir)
    session = SQLiteSession(
        session_id="default",
        db_path=session_dir / "session.db",
    )
    profile_agent = get_profile_agent()
    profile_agent.model = openai_model
    result = await Runner.run(
        starting_agent=profile_agent,
        input=build_profile_input(),
        max_turns=100,
        hooks=LoggingRunHooks(recorder=recorder, stage="profile"),
        # session=session,
    )
    profile_output = result.final_output
    profile = result.final_output.to_human()
    print("Profile Agent Result:\n", profile)
    clarify_agent = get_clarify_agent()
    clarify_agent.model = openai_model
    clarify_session = SQLiteSession(
        session_id="default",
    )
    result = await Runner.run(
        starting_agent=clarify_agent,
        input=build_clarify_input(task_input, profile_output),
        max_turns=100,
        hooks=LoggingRunHooks(recorder=recorder, stage="clarify"),
        session=clarify_session,
    )
    clarified_task = result.final_output_as(ClarifiedTask)
    clarify = result.final_output.to_human()
    print("Clarify Agent Result:\n", clarify)
    while not clarified_task.ready_for_planning:
        result = await Runner.run(
            starting_agent=clarify_agent,
            input="The ready for planning is false, keep asking user.",
            max_turns=100,
            hooks=LoggingRunHooks(recorder=recorder, stage="clarify"),
            session=clarify_session,
        )
        clarified_task = result.final_output_as(ClarifiedTask)
        clarify = result.final_output.to_human()
        print("Clarify Agent Result:\n", clarify)
    context_agent = get_context_agent()
    context_agent.model = openai_model
    result = await Runner.run(
        starting_agent=context_agent,
        input=build_context_input(
            profile=profile_output,
            clarified_task=clarified_task,
        ),
        max_turns=100,
        hooks=LoggingRunHooks(recorder=recorder, stage="context"),
        # session=session,
    )
    context_output = result.final_output
    context = result.final_output.to_human()
    print("Context Agent Result:\n", context)
    plan_agent = get_plan_agent()
    plan_agent.model = openai_model
    result = await Runner.run(
        starting_agent=plan_agent,
        input=build_plan_input(
            profile=profile_output,
            clarified_task=clarified_task,
            context=context_output,
        ),
        max_turns=100,
        hooks=LoggingRunHooks(recorder=recorder, stage="plan"),
        # session=session,
    )
    plan_output = result.final_output
    plan = result.final_output.to_human()
    print("Plan Agent Result:\n", plan)
    implement_agent = get_implement_agent()
    implement_agent.model = openai_model
    result = await Runner.run(
        starting_agent=implement_agent,
        input=build_implement_input(
            clarified_task=clarified_task,
            context=context_output,
            plan=plan_output,
        ),
        max_turns=100,
        hooks=LoggingRunHooks(recorder=recorder, stage="implement"),
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
        hooks=LoggingRunHooks(recorder=recorder, stage="verify"),
        # session=session,
    )
    verification = result.final_output.to_human()
    print("Verify Agent Result:\n", verification)
    review_agent, review_input = get_review_agent()
    review_agent.model = openai_model
    result = await Runner.run(
        starting_agent=review_agent,
        input=review_input
        + profile
        + clarify
        + context
        + plan
        + implementation
        + verification,
        max_turns=100,
        hooks=LoggingRunHooks(recorder=recorder, stage="review"),
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
