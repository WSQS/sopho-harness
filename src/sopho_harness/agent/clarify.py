from agents import Agent
from pydantic import BaseModel, Field

from sopho_harness.agent.profile import ProjectProfile
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
            "Use the query_user tool when important missing information would materially affect planning, scope, or success criteria. "
            "If a request is broad enough that it could reasonably refer to multiple improvement targets, multiple workflow stages, or both design and implementation work, prefer asking the user to narrow scope before producing the final clarified task. "
            "If you do not know whether the user wants changes to one agent, several agents, or the overall workflow, ask. "
            "If you do not know whether the user wants discussion only or direct code changes, ask. "
            "When you ask the user a question, ask as few questions as possible, prefer specific questions with clear options, and allow freeform clarification when needed. "
            "Do not ask the user for information that can be inferred from the provided context. "
            "Do not turn vague preferences into hard constraints unless the user explicitly confirms them. "
            "If the task is already sufficiently clear, do not ask follow-up questions and produce the clarified output directly."
        ),
        tools=[query_user],
        output_type=ClarifiedTask,
    )
    return agent

def build_clarify_input(user_request: str, profile:ProjectProfile) -> str:
    profile_summary = f"""Project name: {profile.project_name}
Purpose: {profile.purpose}
Entrypoints: {", ".join(profile.entrypoints) if profile.entrypoints else "None"}
Key modules: {", ".join(profile.key_modules) if profile.key_modules else "None"}
Project constraints: {"; ".join(profile.constraints) if profile.constraints else "None"}"""

    return f"""Current step:
Clarify the user's task for the next planning step.
First decide whether the request is already specific enough for downstream planning.
If the request is broad and could reasonably refer to multiple improvement targets, prefer asking the user to narrow scope before producing the final clarified task.
If important missing information would materially affect scope, planning direction, or success criteria, ask the user a small number of specific follow-up questions.
If the task is already clear enough, do not ask follow-up questions.
Do not create an implementation plan in this step.

User request:
{user_request}

Project profile summary:
{profile_summary}
"""