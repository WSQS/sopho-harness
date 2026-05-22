from agents import Agent
from pydantic import BaseModel, Field

from sopho_harness.tools import read_file


class ProjectProfile(BaseModel):
    project_name: str = Field(description="The name of the repository, package, or tool.")
    purpose: str = Field(description="A short summary of the project's main goal and what it is used for.")
    entrypoints: list[str] = Field(description="The main entry files, commands, or modules used to start or interact with the project.")
    key_modules: list[str] = Field(description="The most important modules or files for understanding the project's core behavior.")
    constraints: list[str] = Field(description="Known limitations, assumptions, runtime requirements, or design constraints found in the repository.")
    open_questions: list[str] = Field(description="Important uncertainties or missing information that could not be confirmed from the files that were read.")
    extracts: str = Field(description="Short supporting excerpts or evidence from repository files that justify the profile summary.")

    def to_human(self) -> str:
        def format_list(title: str, items: list[str]) -> str:
            if not items:
                return f"## {title}\n\n- None"
            lines = "\n".join(f"- {item}" for item in items)
            return f"## {title}\n\n{lines}"

        sections = [
            f"# {self.project_name}",
            f"## Purpose\n\n{self.purpose}",
            format_list("Entrypoints", self.entrypoints),
            format_list("Key modules", self.key_modules),
            format_list("Constraints", self.constraints),
            format_list("Open questions", self.open_questions),
            f"## Extracts\n\n{self.extracts}",
        ]
        return "\n\n".join(sections)


def get_profile_agent():
    agent = Agent(
        name="Profile Agent",
        instructions="Build up project profile, you are running in the root directory of the project.",
        tools=[read_file],
        output_type=ProjectProfile,
    )
    return agent

def build_profile_input():
    return "Build the project profile for this repository."
