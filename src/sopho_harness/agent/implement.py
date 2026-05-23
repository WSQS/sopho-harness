from agents import Agent
from pydantic import BaseModel, Field

from sopho_harness.agent.clarify import ClarifiedTask
from sopho_harness.agent.context import TaskContext
from sopho_harness.agent.plan import ChangePlan
from sopho_harness.tools import read_file, write_patch


class ImplementationDraft(BaseModel):
    implementation_summary: str = Field(description="A concise summary of the proposed implementation.")
    planned_edits: list[str] = Field(description="The concrete file-level or symbol-level edits that should be applied.")
    expected_effects: list[str] = Field(description="The intended observable effects of the implementation.")
    constraints_followed: list[str] = Field(description="The key constraints or boundaries that the implementation is intended to respect.")
    open_questions: list[str] = Field(description="Remaining implementation uncertainties or blockers.")

    def to_human(self) -> str:
        def format_list(title: str, items: list[str]) -> str:
            if not items:
                return f"## {title}\n\n- None"
            lines = "\n".join(f"- {item}" for item in items)
            return f"## {title}\n\n{lines}"

        sections = [
            f"# {self.implementation_summary}",
            format_list("Planned edits", self.planned_edits),
            format_list("Expected effects", self.expected_effects),
            format_list("Constraints followed", self.constraints_followed),
            format_list("Open questions", self.open_questions),
        ]
        return "\n\n".join(sections)


def get_implement_agent():
    agent = Agent(
        name="Implement Agent",
        instructions="""
Execute the approved change plan by applying concrete code edits.

Primary objective:
- Apply the plan to the codebase in the smallest possible way.
- Stay strictly within the approved scope and target files justified by the plan.

Execution rules:
- Read files as needed before modifying them.
- Use write_patch to apply concrete edits when implementation is clear.
- Prefer minimal deltas over broad rewrites or unrelated refactors.
- Do not expand the task beyond the approved plan.
- Do not invent unsupported requirements or new goals.

Output discipline:
- Report what changes were actually applied.
- Summarize the concrete edits that were made.
- Summarize the expected effects of those applied edits.
- Record which constraints were followed.
- Record any remaining implementation uncertainties or blockers.

Do not do the following:
- Do not re-plan the task.
- Do not produce verification conclusions.
- Do not produce final review judgments.
- Do not claim an edit was applied if you did not actually apply it.
""",
        output_type=ImplementationDraft,
        tools=[read_file, write_patch],
    )
    return agent


def build_implement_input(
    clarified_task: ClarifiedTask,
    context: TaskContext,
    plan: ChangePlan,
) -> str:
    constraints = (
        "\n".join(f"- {constraint}" for constraint in clarified_task.constraints)
        or "- None"
    )
    context_constraints = (
        "\n".join(f"- {item}" for item in context.local_constraints) or "- None"
    )

    step_lines: list[str] = []
    for index, step in enumerate(plan.steps, 1):
        target_files = ", ".join(step.target_files) if step.target_files else "None"
        uncertainties = (
            "; ".join(step.uncertainties) if step.uncertainties else "None"
        )
        step_lines.append(
            f"{index}. {step.name}\n"
            f"   Description: {step.description}\n"
            f"   Target files: {target_files}\n"
            f"   Uncertainties: {uncertainties}"
        )
    plan_steps = "\n".join(step_lines) or "- None"

    return f"""Current step:
Execute the approved change plan by applying concrete code edits.

Implementation mission:
- Apply the approved plan to the codebase in the smallest possible way.
- Stay within the plan and do not expand scope.
- Modify only files justified by the approved plan.
- Report applied edits and expected effects.
- Do not produce verification or review conclusions in this step.

Task summary:
{clarified_task.task_summary}

Constraints from clarify:
{constraints}

Local constraints from context:
{context_constraints}

Approved plan summary:
{plan.plan_summary}

Approved plan steps:
{plan_steps}
"""
