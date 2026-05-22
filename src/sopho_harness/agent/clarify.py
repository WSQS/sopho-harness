from agents import Agent
from pydantic import BaseModel, Field

from sopho_harness.tools import query_user


class ClarifiedTask(BaseModel):
    task_summary: str = Field(description="A concise restatement of the user's requested task.")
    goals: list[str] = Field(description="The concrete goals that the implementation should achieve.")
    non_goals: list[str] = Field(description="What should explicitly remain out of scope for this task.")
    constraints: list[str] = Field(description="Relevant constraints, requirements, or boundaries stated or implied by the user.")
    open_questions: list[str] = Field(description="Important ambiguities or missing details that may require follow-up before planning or implementation.")

    def to_human(self) -> str:
        def format_list(title: str, items: list[str]) -> str:
            if not items:
                return f"## {title}\n\n- None"
            lines = "\n".join(f"- {item}" for item in items)
            return f"## {title}\n\n{lines}"

        sections = [
            f"# {self.task_summary}",
            format_list("Goals", self.goals),
            format_list("Non-goals", self.non_goals),
            format_list("Constraints", self.constraints),
            format_list("Open questions", self.open_questions),
        ]
        return "\n\n".join(sections)


def get_clarify_agent():
    agent = Agent(
        name="Clarify Agent",
        instructions=(
            "Clarify the user's task before planning. "
            "Your job is to transform the user's raw request into a task description that downstream planning can reliably use. "
            "Restate the task, separate goals from non-goals, preserve explicit constraints, and record unresolved ambiguities without inventing requirements. "
            "Use the query_user tool only when important missing information would materially affect planning, scope, or success criteria. "
            "When you ask the user a question, ask as few questions as possible, prefer specific questions with clear options, and allow freeform clarification when needed. "
            "Do not ask the user for information that can be inferred from the provided context. "
            "Do not turn vague preferences into hard constraints unless the user explicitly confirms them. "
            "If the task is already sufficiently clear, do not ask follow-up questions and produce the clarified output directly."
        ),
        tools=[query_user],
        output_type=ClarifiedTask,
    )
    clarify_input = "Clarify the user's task for the next planning step."
    return agent, clarify_input
