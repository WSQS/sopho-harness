import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Literal, cast
from agents import (
    Agent,
    ModelResponse,
    RunContextWrapper,
    RunHooks,
    Runner,
    SessionABC,
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


def _item_to_message(item: TResponseInputItem) -> tuple[str, str] | None:
    role = item.get("role")
    v_type = item.get("type")
    content = item.get("content")

    def handle_data(data: Any) -> str:
        if data is None:
            return ""

        if isinstance(data, str):
            return data

        if isinstance(data, Mapping):
            mapping = cast(Mapping[str, Any], data)
            item_type = mapping.get("type")
            if item_type == "output_text":
                text = mapping.get("text")
                if isinstance(text, str):
                    return text
            if item_type == "function_call":
                return (
                    f"function name: {mapping.get('name')}, "
                    f"arguments: {mapping.get('arguments')}"
                )
            if item_type == "function_call_output":
                output = mapping.get("output")
                return f"output length: {len(str(output))}"

            nested_content = mapping.get("content")
            if nested_content is not None:
                nested_text = handle_data(nested_content)
                if nested_text:
                    return nested_text

            return _shorten(mapping)

        if isinstance(data, Sequence) and not isinstance(data, str | bytes | bytearray):
            sequence = cast(Sequence[Any], data)
            if len(sequence) == 1:
                return handle_data(sequence[0])

            parts = [text for part in sequence if (text := handle_data(part))]
            if parts:
                return "\n".join(parts)

        return _shorten(data)

    if isinstance(role, str):
        return (role, handle_data(content))
    if isinstance(v_type, str):
        return (v_type, handle_data(item))
    return None


class UiSQLiteSession(SessionABC):
    """Session backed by SQLite with an in-memory item cache for the GUI."""

    def __init__(self, session_id: str, db_path: str | Path) -> None:
        self._backend = SQLiteSession(session_id=session_id, db_path=db_path)
        self._items: list[TResponseInputItem] = []
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._load())
        except RuntimeError:
            asyncio.run(self._load())

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        await self._backend.add_items(items)
        self._items.extend(items)

    async def clear_session(self) -> None:
        await self._backend.clear_session()
        self._items.clear()

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        return await self._backend.get_items(limit=limit)

    async def pop_item(self) -> TResponseInputItem | None:
        result = await self._backend.pop_item()
        if result is not None and self._items:
            self._items.pop()
        return result

    async def _load(self) -> None:
        self._items = await self._backend.get_items()

    @property
    def items(self) -> Sequence[TResponseInputItem]:
        return self._items


@dataclass
class GuiState:
    session: UiSQLiteSession
    agent: Agent[Any]
    input_text: str = ""
    status_text: str = "Idle"
    pending_task: asyncio.Task[None] | None = None

    async def send_message(self, message: str) -> None:
        self.status_text = "Sending"
        try:
            await Runner.run(
                starting_agent=self.agent,
                input=message,
                max_turns=100,
                hooks=LoggingRunHooks(),
                session=self.session,
            )
            self.status_text = "Idle"
        except Exception as exc:
            self.status_text = f"Error: {_shorten(exc)}"

    def start_send(self, message: str) -> None:
        if self.pending_task is not None and not self.pending_task.done():
            self.status_text = "Busy"
            return
        self.pending_task = asyncio.create_task(self.send_message(message))


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
    imgui.text("sopho-harness")
    imgui.same_line()
    imgui.text_disabled(f"Status: {state.status_text}")
    imgui.separator()

    footer_height = 170
    messages_height = max(0.0, imgui.get_content_region_avail().y - footer_height)
    child_flags = imgui.WindowFlags_.horizontal_scrollbar

    imgui.begin_child("Messages", imgui.ImVec2(0, messages_height), True, child_flags)
    for item in state.session.items:
        r = _item_to_message(item)
        if r is None:
            continue
        role, content = r
        imgui.text_colored(
            (0.4, 0.7, 1.0, 1.0) if role == "assistant" else (0.7, 1.0, 0.4, 1.0),
            role,
        )
        imgui.same_line()
        imgui.text_wrapped(content)
        imgui.spacing()
    if imgui.get_scroll_y() >= imgui.get_scroll_max_y() - 4:
        imgui.set_scroll_here_y(1.0)
    imgui.end_child()

    imgui.separator()
    _, state.input_text = imgui.input_text_multiline(
        "##chat_input",
        state.input_text,
        imgui.ImVec2(-1, 110),
    )

    if imgui.button("Send") and state.input_text.strip():
        message = state.input_text.strip()
        state.input_text = ""
        state.start_send(message)

    imgui.same_line()
    if imgui.button("Clear"):
        state.status_text = "Cleared"
        state.input_text = ""

    imgui.end()


def main() -> None:
    config = load_config()
    if config is None:
        print("No config found. Please create a .sopho-harness/config.toml file.")
        config = SophoHarnessConfig()

    assets_dir = Path(__file__).parent / "assets"
    hello_imgui.set_assets_folder(str(assets_dir))
    session_dir = Path(".sopho-harness")
    session_dir.mkdir(parents=True, exist_ok=True)
    session = UiSQLiteSession(
        session_id="default",
        db_path=session_dir / "session.db",
    )
    agent = Agent(
        name="Coding agent",
        instructions=build_instructions(config),
        model=openai_model,
        tools=build_tools(config),
    )
    state = GuiState(session=session, agent=agent)

    def load_fonts() -> None:
        try:
            font_path = "fonts/NotoSansSC-Regular.ttf"
            hello_imgui.load_font(font_path, 18.0)
        except Exception as e:
            print(f"Failed to load any CJK font: {e}")

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Sopho Harness"
    runner_params.callbacks.load_additional_fonts = load_fonts
    runner_params.callbacks.show_gui = lambda: gui(state)
    asyncio.run(hello_imgui.run_async(runner_params))


if __name__ == "__main__":
    main()
