import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Literal
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


def is_multiline(text: str) -> bool:
    return "\n" in text


class UiSQLiteSession(SessionABC):
    """Session backed by SQLite with an in-memory item cache for the GUI."""

    def __init__(self, session_id: str, db_path: str | Path) -> None:
        self._backend = SQLiteSession(session_id=session_id, db_path=db_path)
        self._items: list[tuple[int, TResponseInputItem]] = []
        self._index = 0

        async def load() -> None:
            items = await self._backend.get_items()
            self._items = list(enumerate(items, start=self._index))
            if self._items:
                self._index = self._items[-1][0] + 1

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(load())
        except RuntimeError:
            asyncio.run(load())

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        await self._backend.add_items(items)
        self._items.extend(list(enumerate(items, start=self._index)))
        if self._items:
            self._index = self._items[-1][0] + 1

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

    @property
    def items(self) -> Sequence[tuple[int, TResponseInputItem]]:
        return self._items


@dataclass
class GuiState:
    session: UiSQLiteSession
    agent: Agent[Any]
    input_text: str = ""
    status_text: str = "Idle"
    pending_task: asyncio.Task[None] | None = None
    collapse: dict[int, bool] = field(default_factory=dict[int, bool])

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
    items = state.session.items
    for index, item in items:
        match item:
            case {"content": content, "role": role} if (
                isinstance(content, str) and len(item) == 2
            ):
                imgui.text_colored((0.4, 0.7, 1.0, 1.0), role)
                imgui.same_line()
                imgui.text_wrapped(content)
            case {
                "type": "message",
                "role": role,
                "content": [{"type": "output_text", "text": text}],
            }:
                if is_multiline(text):
                    collapse = state.collapse.get(index, True)
                    if collapse:
                        if imgui.button(f"expand##{index}"):
                            state.collapse[index] = not collapse
                        imgui.same_line()
                        imgui.text_colored((0.7, 1.0, 0.4, 1.0), role)
                        imgui.same_line()
                        imgui.text(text.split("\n")[0] + "...")
                    else:
                        if imgui.button(f"collapse##{index}"):
                            state.collapse[index] = not collapse
                        imgui.same_line()
                        imgui.text_colored((0.7, 1.0, 0.4, 1.0), role)
                        imgui.same_line()
                        imgui.text_wrapped(text)
                else:
                    imgui.text_colored((0.7, 1.0, 0.4, 1.0), role)
                    imgui.same_line()
                    imgui.text(text)
            case {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": text}],
            }:
                imgui.text_colored((0.7, 1.0, 0.4, 1.0), "reasoning")
                imgui.same_line()
                imgui.text_wrapped(text)
            case {
                "type": "reasoning",
                "content": [],
                "encrypted_content": encrypted_content,
            } if isinstance(encrypted_content, str):
                # we should not display encrypted content reasoning
                continue
            case {"type": "function_call", "name": name, "arguments": arguments}:
                imgui.text_colored((0.7, 1.0, 0.4, 1.0), "function_call")
                imgui.same_line()
                imgui.text_wrapped(f"{name}({arguments})")
            case {"type": "function_call_output", "output": output} if isinstance(
                output, str
            ):
                if is_multiline(output):
                    collapse = state.collapse.get(index, True)
                    if collapse:
                        if imgui.button(f"expand##{index}"):
                            state.collapse[index] = not collapse
                        imgui.same_line()
                        imgui.text_colored((0.7, 1.0, 0.4, 1.0), "function_call_output")
                        imgui.same_line()
                        imgui.text(output.split("\n")[0] + "...")
                    else:
                        if imgui.button(f"collapse##{index}"):
                            state.collapse[index] = not collapse
                        imgui.same_line()
                        imgui.text_colored((0.7, 1.0, 0.4, 1.0), "function_call_output")
                        imgui.same_line()
                        imgui.text_wrapped(output)
                else:
                    imgui.text_colored((0.7, 1.0, 0.4, 1.0), "function_call_output")
                    imgui.same_line()
                    imgui.text_wrapped(f"output: {output}")
            case _:
                imgui.text_wrapped(str(item))
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
