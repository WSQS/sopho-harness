from agents import Agent
from pydantic import BaseModel, Field

from sopho_harness.agent.clarify import ClarifiedTask
from sopho_harness.agent.profile import ProjectProfile
from sopho_harness.tools import read_file


class TaskContext(BaseModel):
    relevant_files: list[str] = Field(description="The files most relevant to the current task.")
    relevant_symbols: list[str] = Field(description="The functions, classes, methods, or other symbols most relevant to the current task.")
    existing_patterns: list[str] = Field(description="Existing implementation patterns, conventions, or abstractions that the task should follow.")
    possible_change_points: list[str] = Field(description="The most likely files, functions, or modules where changes may be needed.")
    local_constraints: list[str] = Field(description="Task-specific constraints or invariants that should be preserved in the affected area of the codebase.")
    open_questions: list[str] = Field(description="Important local uncertainties that may affect planning or implementation.")
    extracts: str = Field(description="Short supporting excerpts or evidence from relevant files that justify the local context summary.")

    def to_human(self) -> str:
        def format_list(title: str, items: list[str]) -> str:
            if not items:
                return f"## {title}\n\n- None"
            lines = "\n".join(f"- {item}" for item in items)
            return f"## {title}\n\n{lines}"

        sections = [
            "# Task context",
            format_list("Relevant files", self.relevant_files),
            format_list("Relevant symbols", self.relevant_symbols),
            format_list("Existing patterns", self.existing_patterns),
            format_list("Possible change points", self.possible_change_points),
            format_list("Local constraints", self.local_constraints),
            format_list("Open questions", self.open_questions),
            f"## Extracts\n\n{self.extracts}",
        ]
        return "\n\n".join(sections)


def get_context_agent():
    agent = Agent(
        name="Context Agent",
        instructions="Build local task context for the current request. Identify the most relevant files, symbols, implementation patterns, likely change points, and local constraints. Base conclusions on files you read and record unresolved local uncertainties without inventing details.",
        tools=[read_file],
        output_type=TaskContext,
    )
    return agent


def build_context_input(profile: ProjectProfile, clarified_task: ClarifiedTask) -> str:
    goals = "\n".join(f"- {goal}" for goal in clarified_task.goals) or "- None"
    non_goals = (
        "\n".join(f"- {non_goal}" for non_goal in clarified_task.non_goals) or "- None"
    )
    constraints = (
        "\n".join(f"- {constraint}" for constraint in clarified_task.constraints)
        or "- None"
    )
    open_questions = (
        "\n".join(f"- {question}" for question in clarified_task.open_questions)
        or "- None"
    )

    return f"""Current step:
Build the local code context for the current task.

Context mission:
- Read only the code most relevant to the clarified task.
- Identify the local files, symbols, patterns, change points, and constraints that downstream planning must understand.
- Focus on the local area of the codebase rather than summarizing the whole repository.
- Do not create an implementation plan in this step.

Task summary:
{clarified_task.task_summary}

Goals:
{goals}

Non-goals:
{non_goals}

Constraints:
{constraints}

Non-blocking open questions from clarify:
{open_questions}

Project profile summary:
- Project name: {profile.project_name}
- Purpose: {profile.purpose}
- Entrypoints: {", ".join(profile.entrypoints) if profile.entrypoints else "None"}
- Key modules: {", ".join(profile.key_modules) if profile.key_modules else "None"}
- Project constraints: {"; ".join(profile.constraints) if profile.constraints else "None"}

Output focus:
- Prioritize the most relevant files and symbols.
- Highlight the most likely change points.
- Record local constraints that planning should preserve.
- Record unresolved local uncertainties only if they are grounded in the code you read.
- Base conclusions on file evidence and do not invent details.
"""
