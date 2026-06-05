import asyncio
from dataclasses import dataclass
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

from sopho_harness.config import SophoHarnessConfig, load_config
from sopho_harness.context import build_instructions
from sopho_harness.tools import build_tools
from imgui_bundle import hello_imgui, imgui

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


@dataclass
class GuiState:
    pass


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
    agent = Agent(
        name="Coding agent",
        instructions=build_instructions(config),
        model=openai_model,
        tools=build_tools(config),
    )
    result = await Runner.run(
        starting_agent=agent,
        input=task_input,
        max_turns=100,
        hooks=LoggingRunHooks(),
        session=session,
    )
    print(result.final_output)


def gui(state: GuiState) -> None:
    viewport = imgui.get_main_viewport()
    imgui.set_next_window_pos(viewport.work_pos)
    imgui.set_next_window_size(viewport.work_size)

    flags = (
        imgui.WindowFlags_.no_decoration
        | imgui.WindowFlags_.no_move
        | imgui.WindowFlags_.no_saved_settings
        | imgui.WindowFlags_.no_scrollbar
        | imgui.WindowFlags_.no_scroll_with_mouse
    )

    imgui.begin("RootWindow", None, flags)
    imgui.text("Hello from async ImGui")
    imgui.separator()
    imgui.text("This window fills the main application window.")
    imgui.input_text("Input", "Type something here...")
    imgui.end()


def main() -> None:
    assets_dir = Path(__file__).parent / "assets"
    hello_imgui.set_assets_folder(str(assets_dir))
    state = GuiState()

    def load_fonts() -> None:
        try:
            font_path = "fonts/NotoSansSC-Regular.ttf"
            hello_imgui.load_font(font_path, 18.0)
        except Exception as e:
            print(f"Failed to load any CJK font: {e}")

    runner_params = hello_imgui.RunnerParams()
    runner_params.callbacks.load_additional_fonts = load_fonts
    runner_params.callbacks.show_gui = lambda: gui(state)
    asyncio.run(hello_imgui.run_async(runner_params))


if __name__ == "__main__":
    main()
