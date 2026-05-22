from agents import Agent
from pydantic import BaseModel, Field


class ChangePlan(BaseModel):
    plan_summary: str = Field(description="A concise summary of the intended change.")
    goals: list[str] = Field(description="The specific outcomes this change plan should achieve.")
    target_files: list[str] = Field(description="The files most likely to require modification for this change.")
    steps: list[str] = Field(description="A minimal ordered implementation plan for the change.")
    non_goals: list[str] = Field(description="Explicitly out-of-scope work that should not be included in this change.")
    risks: list[str] = Field(description="The main implementation risks, uncertainties, or invariants to watch.")
    open_questions: list[str] = Field(description="Remaining planning questions that may affect implementation quality or scope.")

    def to_human(self) -> str:
        def format_list(title: str, items: list[str]) -> str:
            if not items:
                return f"## {title}\n\n- None"
            lines = "\n".join(f"- {item}" for item in items)
            return f"## {title}\n\n{lines}"

        sections = [
            f"# {self.plan_summary}",
            format_list("Goals", self.goals),
            format_list("Target files", self.target_files),
            format_list("Steps", self.steps),
            format_list("Non-goals", self.non_goals),
            format_list("Risks", self.risks),
            format_list("Open questions", self.open_questions),
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
