from agents import Agent
from pydantic import BaseModel, Field


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
        instructions="Describe the candidate implementation for the approved change plan. Stay within the declared scope, focus on concrete edits, and do not expand the task beyond the plan.",
        output_type=ImplementationDraft,
    )
    implement_input = "Describe the candidate implementation for the current change plan."
    return agent, implement_input
