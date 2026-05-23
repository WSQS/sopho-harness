from agents import Agent
from pydantic import BaseModel, Field


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
    plan_input = "Create the minimal change plan for the current task."
    return agent, plan_input
