import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

from agents import Tool, function_tool

from .config import PythonConfig, SophoHarnessConfig


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
def write_patch(content: str) -> str:
    """Apply unified diff content to the current git working tree."""
    is_supported_patch = (
        content.startswith("--- ") and "\n+++ " in content and "\n@@" in content
    )
    if not is_supported_patch:
        return (
            "Unsupported patch format. write_patch only accepts unified diff content "
            "that starts with '--- ', includes '+++ ', and contains at least one '@@' hunk."
        )

    with NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".patch") as tmp:
        tmp.write(content)
        patch_file = Path(tmp.name)

    try:
        check_result = subprocess.run(
            [
                "git",
                "apply",
                "--check",
                "--whitespace=nowarn",
                "--recount",
                str(patch_file),
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        if check_result.returncode != 0:
            output = (check_result.stderr or check_result.stdout).strip()
            return f"Failed to validate patch: {output}"

        apply_result = subprocess.run(
            [
                "git",
                "apply",
                "--whitespace=nowarn",
                "--recount",
                str(patch_file),
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
        if apply_result.returncode != 0:
            output = (apply_result.stderr or apply_result.stdout).strip()
            return f"Failed to apply patch: {output}"
        return "Applied patch successfully."
    finally:
        patch_file.unlink(missing_ok=True)


@function_tool
def query_user(question: str, options: list[str] | None = None) -> str:
    print(question)
    if options:
        print("Options:")
        for idx, opt in enumerate(options, 1):
            print(f"{idx}. {opt}")
    while True:
        answer = input("Your answer: ").strip()
        if options:
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return options[int(answer) - 1]
            else:
                return answer
        else:
            return answer


def build_python_tools(config: PythonConfig) -> list[Tool]:
    """Build Python-specific tools from config."""
    if not config.entrypoint:
        return []

    entrypoint = config.entrypoint
    tools: list[Tool] = []

    @function_tool
    def run_python_tests() -> str:
        command = [entrypoint, "run", "pytest"] if entrypoint == "uv" else [entrypoint, "pytest"]
        result = subprocess.run(command, capture_output=True, text=True, cwd=Path.cwd())
        output = result.stdout or result.stderr
        return output.strip() or f"Tests finished with exit code {result.returncode}"

    tools.append(run_python_tests)

    ruff_check_command = [entrypoint, "run", "ruff", "--version"] if entrypoint == "uv" else [entrypoint, "ruff", "--version"]
    ruff_check_result = subprocess.run(
        ruff_check_command,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )
    if ruff_check_result.returncode != 0:
        print("[tools] ruff is not available; skipping run_python_lint tool. Please install ruff manually.")
        print(f"[tools] ruff check return code: {ruff_check_result.returncode}")
        print(f"[tools] ruff check stdout: {ruff_check_result.stdout}")
        print(f"[tools] ruff check stderr: {ruff_check_result.stderr}")
        return tools

    @function_tool
    def run_python_lint() -> str:
        command = [entrypoint, "run", "ruff", "check", "."] if entrypoint == "uv" else [entrypoint, "ruff", "check", "."]
        result = subprocess.run(command, capture_output=True, text=True, cwd=Path.cwd())
        output = result.stdout or result.stderr
        return output.strip() or f"Lint finished with exit code {result.returncode}"

    tools.append(run_python_lint)
    return tools


def build_tools(config: SophoHarnessConfig) -> list[Tool]:
    """Build the enabled tool list from config."""
    tools: list[Tool] = [read_file, write_patch]

    if config.language == "python" and config.python is not None:
        tools.extend(build_python_tools(config.python))

    return tools
