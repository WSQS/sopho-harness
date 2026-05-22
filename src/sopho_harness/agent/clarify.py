from agents import Agent
from pydantic import BaseModel, Field

from sopho_harness.agent.profile import ProjectProfile
from sopho_harness.tools import query_user


class ClarifiedTask(BaseModel):
    ready_for_planning: bool = Field(description="Whether the task is now sufficiently clear for downstream planning to proceed without asking the user for more essential information.")
    task_summary: str = Field(description="A concise restatement of the user's requested task.")
    goals: list[str] = Field(description="The concrete goals that the implementation should achieve.")
    non_goals: list[str] = Field(description="What should explicitly remain out of scope for this task.")
    constraints: list[str] = Field(description="Relevant constraints, requirements, or boundaries stated or implied by the user.")
    open_questions: list[str] = Field(description="Minor non-blocking follow-up questions or implementation-level details that can be left for later planning or execution once the main task definition is already clear.")

    def to_human(self) -> str:
        def format_list(title: str, items: list[str]) -> str:
            if not items:
                return f"## {title}\n\n- None"
            lines = "\n".join(f"- {item}" for item in items)
            return f"## {title}\n\n{lines}"

        sections = [
            f"# {self.task_summary}",
            f"## Ready for planning\n\n- {'Yes' if self.ready_for_planning else 'No'}",
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
            "Your first responsibility is to decide whether downstream planning can proceed safely without more user input. "
            "Your job is to transform the user's raw request into a task description that downstream planning can reliably use. "
            "Restate the task, separate goals from non-goals, preserve explicit constraints, and record unresolved ambiguities without inventing requirements. "
            "If important missing information would materially affect planning, scope, success criteria, or the choice between multiple plausible directions, you should ask the user before producing the final clarified task. "
            "If a request is broad enough that it could reasonably refer to multiple improvement targets, multiple workflow stages, or both design and implementation work, ask the user to narrow scope before producing the final clarified task. "
            "If you do not know whether the user wants changes to one agent, several agents, or the overall workflow, ask the user. "
            "If you do not know whether the user wants discussion only or direct code changes, ask the user. "
            "If you do not know what success would look like, and that uncertainty would change downstream planning, ask the user. "
            "In these cases, asking the user is preferred over making a conservative guess. "
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
First decide whether downstream planning can proceed safely without more user input.
If the request is broad and could reasonably refer to multiple improvement targets, ask the user to narrow scope before producing the final clarified task.
If important missing information would materially affect scope, planning direction, success criteria, or the choice between multiple plausible directions, ask the user a small number of specific follow-up questions before finalizing the clarified task.
If the task is already clear enough, do not ask follow-up questions.
Do not create an implementation plan in this step.

User request:
{user_request}

Project profile summary:
{profile_summary}
"""