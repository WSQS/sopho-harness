from pathlib import Path

from agents import Tool, function_tool

from .config import SophoHarnessConfig


@function_tool
def read_file(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return f"File not found: {file_path}"
    if file_path.is_dir():
        entries = sorted(
            [
                f"{item.name}/" if item.is_dir() else item.name
                for item in file_path.iterdir()
            ]
        )
        return "\n".join(entries) if entries else f"Directory is empty: {file_path}"
    return file_path.read_text(encoding="utf-8")


@function_tool
def write_patch(path: str, content: str) -> str:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"Wrote file: {file_path}"


def build_tools(config: SophoHarnessConfig) -> list[Tool]:
    """Build the enabled tool list from config."""
    return [read_file, write_patch]
