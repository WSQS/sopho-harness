import tomllib
from dataclasses import dataclass


@dataclass
class PythonConfig:
    entrypoint: str | None = None


@dataclass
class SophoHarnessConfig:
    language: str | None = None
    python: PythonConfig | None = None


def load_config(path: str = ".sopho-harness/config.toml") -> SophoHarnessConfig | None:
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return None
    python_data = data.get("python", {})
    return SophoHarnessConfig(
        language=data.get("language"),
        python=PythonConfig(
            entrypoint=python_data.get("entrypoint"),
        ),
    )
