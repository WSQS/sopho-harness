from .config import SophoHarnessConfig


def build_instructions(config: SophoHarnessConfig) -> str:
    """Build instructions string from config."""
    if config.language:
        return f"You are a helpful coding agent for {config.language} projects."
    return "You are a helpful coding agent."
