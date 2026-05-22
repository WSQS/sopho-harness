from agents import Agent
from pydantic import BaseModel, Field


class ReviewReport(BaseModel):
    review_summary: str = Field(description="A concise summary of the overall review judgment.")
    strengths: list[str] = Field(description="The strongest aspects of the candidate change.")
    concerns: list[str] = Field(description="The main review concerns, correctness issues, or design risks.")
    style_alignment: list[str] = Field(description="Ways the change does or does not align with existing project patterns and style.")
    recommended_focus: list[str] = Field(description="The most important areas for a human reviewer to inspect closely.")
    open_questions: list[str] = Field(description="Unresolved review questions that still need confirmation.")

    def to_human(self) -> str:
        def format_list(title: str, items: list[str]) -> str:
            if not items:
                return f"## {title}\n\n- None"
            lines = "\n".join(f"- {item}" for item in items)
            return f"## {title}\n\n{lines}"

        sections = [
            f"# {self.review_summary}",
            format_list("Strengths", self.strengths),
            format_list("Concerns", self.concerns),
            format_list("Style alignment", self.style_alignment),
            format_list("Recommended focus", self.recommended_focus),
            format_list("Open questions", self.open_questions),
        ]
        return "\n\n".join(sections)


def get_review_agent():
    agent = Agent(
        name="Review Agent",
        instructions="Review the current candidate change as a human-facing copilot reviewer. Summarize strengths, concerns, style alignment, recommended review focus, and unresolved questions without claiming certainty you do not have.",
        output_type=ReviewReport,
    )
    review_input = "Review the current candidate change for a human maintainer."
    return agent, review_input
