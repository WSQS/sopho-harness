from agents import Agent
from pydantic import BaseModel, Field

from sopho_harness.agent.profile import ProjectProfile
from sopho_harness.tools import query_user


class ClarifiedTask(BaseModel):
    ready_for_planning: bool = Field(description="Whether downstream planning can proceed without asking the user any additional blocking questions. Set this to true only when the task target, intended kind of work, scope boundaries, and success-critical constraints are clear enough that the planner does not need to guess among materially different directions.")
    task_summary: str = Field(description="A concise restatement of the user's requested task.")
    goals: list[str] = Field(description="The concrete goals that the implementation should achieve.")
    non_goals: list[str] = Field(description="What should explicitly remain out of scope for this task.")
    constraints: list[str] = Field(description="Relevant constraints, requirements, or boundaries stated or implied by the user.")
    open_questions: list[str] = Field(description="Only minor non-blocking follow-up questions or later implementation details. Do not put any question here if its answer would materially change the planning direction, target files, task scope, success criteria, or whether the task is discussion-only versus code changes.")

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
        instructions="""
Clarify the user's task before planning.

Primary objective:
- Transform the user's raw request into a clarified task that downstream planning can use reliably.
- Your first responsibility is to decide whether planning can proceed safely without more user input.

Required output discipline:
- Restate the task clearly.
- Separate goals from non-goals.
- Preserve explicit constraints.
- Record only non-blocking unresolved questions.
- Do not invent requirements.
- Do not create an implementation plan in this step.

Strict readiness rule:
- Set `ready_for_planning=true` only when the remaining unknowns are genuinely non-blocking.
- A missing detail is blocking if its answer could materially change any of the following:
  - the task target
  - whether the work is discussion versus code changes
  - which agents or workflow stages are in scope
  - the main planning direction
  - the scope boundary
  - the success criteria
- If any blocking ambiguity remains, ask the user before producing the final clarified task.
- Do not mark the task ready merely because you can write a plausible summary.
- Mark it ready only when the planner would not need to guess among materially different interpretations.

When you must ask the user:
- Ask if the request is broad enough to reasonably refer to multiple improvement targets.
- Ask if the request might cover multiple workflow stages or both design discussion and implementation work.
- Ask if it is unclear whether the user wants changes to one agent, several agents, or the overall workflow.
- Ask if it is unclear whether the user wants discussion only or direct code changes.
- Ask if success criteria are unclear in a way that would change downstream planning.
- When in doubt about whether a question is blocking, treat it as blocking and ask.

Question quality rubric:
- Ask as few questions as possible.
- Combine related blocking uncertainties into one compact question when practical.
- Prefer specific questions with clear options.
- Allow freeform clarification when needed.
- Do not ask for information that can be inferred from the provided context.
- Do not turn vague preferences into hard constraints unless the user explicitly confirms them.

Open questions rule:
- `open_questions` must contain only non-blocking items.
- If answering a question would materially change the plan, it is not an open question.
- In that case, keep `ready_for_planning=false` and ask the user first.

If the task is sufficiently clear under this strict rule, do not ask follow-up questions and produce the clarified output directly.
""",
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
Use a strict readiness rule: planning is ready only if the remaining unknowns are non-blocking and would not materially change task target, task type, scope, success criteria, or planning direction.
If the request is broad and could reasonably refer to multiple improvement targets, ask the user to narrow scope before producing the final clarified task.
If important missing information would materially affect scope, planning direction, success criteria, or the choice between multiple plausible directions, ask the user a small number of specific follow-up questions before finalizing the clarified task.
If you are unsure whether a question is blocking, treat it as blocking and ask.
Do not put blocking questions into open_questions.
If the task is already clear enough, do not ask follow-up questions.
Do not create an implementation plan in this step.

User request:
{user_request}

Project profile summary:
{profile_summary}
"""