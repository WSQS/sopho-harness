from agents import Agent
from pydantic import BaseModel, Field

from sopho_harness.agent.clarify import ClarifiedTask
from sopho_harness.agent.context import TaskContext
from sopho_harness.agent.profile import ProjectProfile


class PlanStep(BaseModel):
    name: str
    description: str
    target_files: list[str]
    uncertainties: list[str]


class ChangePlan(BaseModel):
    plan_summary: str = Field(description="A concise summary of the intended change.")
    steps: list[PlanStep]

    def to_human(self) -> str:
        def format_list(title: str, items: list[str]) -> str:
            if not items:
                return f"## {title}\n\n- None"
            lines = "\n".join(f"- {item}" for item in items)
            return f"## {title}\n\n{lines}"

        def format_steps(steps: list[PlanStep]) -> str:
            if not steps:
                return "## Steps\n\n- None"

            rendered_steps: list[str] = []
            for index, step in enumerate(steps, 1):
                sections = [
                    f"### Step {index}: {step.name}",
                    step.description,
                    format_list("Target files", step.target_files),
                    format_list("Uncertainties", step.uncertainties),
                ]
                rendered_steps.append("\n\n".join(sections))

            return "## Steps\n\n" + "\n\n".join(rendered_steps)

        sections = [
            f"# {self.plan_summary}",
            format_steps(self.steps),
        ]
        return "\n\n".join(sections)


def get_plan_agent():
    agent = Agent(
        name="Plan Agent",
        instructions="Create a minimal change plan for the current task. Use the clarified task and local context to identify target files, ordered implementation steps, explicit non-goals, and major risks. Keep scope tight and do not invent unsupported requirements.",
        output_type=ChangePlan,
    )
    return agent


def build_plan_input(
    profile: ProjectProfile,
    clarified_task: ClarifiedTask,
    context: TaskContext,
) -> str:
    goals = "\n".join(f"- {goal}" for goal in clarified_task.goals) or "- None"
    non_goals = (
        "\n".join(f"- {non_goal}" for non_goal in clarified_task.non_goals)
        or "- None"
    )
    constraints = (
        "\n".join(f"- {constraint}" for constraint in clarified_task.constraints)
        or "- None"
    )
    relevant_files = (
        "\n".join(f"- {file}" for file in context.relevant_files) or "- None"
    )
    possible_change_points = (
        "\n".join(f"- {item}" for item in context.possible_change_points)
        or "- None"
    )
    local_constraints = (
        "\n".join(f"- {item}" for item in context.local_constraints)
        or "- None"
    )
    open_questions = (
        "\n".join(f"- {item}" for item in context.open_questions) or "- None"
    )

    return f"""Current step:
Create the minimal change plan for the current task.

Planning mission:
- Produce a minimal, bounded plan grounded in the clarified task and local code context.
- Prefer the smallest credible set of steps and target files.
- Do not create implementation details in this step.
- Do not invent unsupported requirements.

Task summary:
{clarified_task.task_summary}

Goals:
{goals}

Non-goals:
{non_goals}

Constraints:
{constraints}

Project profile summary:
- Project name: {profile.project_name}
- Purpose: {profile.purpose}

Relevant files from context:
{relevant_files}

Possible change points:
{possible_change_points}

Local constraints:
{local_constraints}

Local open questions:
{open_questions}
"""
