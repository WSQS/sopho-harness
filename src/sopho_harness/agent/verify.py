from agents import Agent
from pydantic import BaseModel, Field


class VerificationReport(BaseModel):
    verification_summary: str = Field(description="A concise summary of the verification status.")
    checks_run: list[str] = Field(description="The checks, validations, or inspections that were run or should be run.")
    findings: list[str] = Field(description="The important results, failures, or observations from verification.")
    remaining_risks: list[str] = Field(description="Risks that remain after verification or could not yet be checked.")
    follow_up_actions: list[str] = Field(description="Recommended next verification or remediation actions.")

    def to_human(self) -> str:
        def format_list(title: str, items: list[str]) -> str:
            if not items:
                return f"## {title}\n\n- None"
            lines = "\n".join(f"- {item}" for item in items)
            return f"## {title}\n\n{lines}"

        sections = [
            f"# {self.verification_summary}",
            format_list("Checks run", self.checks_run),
            format_list("Findings", self.findings),
            format_list("Remaining risks", self.remaining_risks),
            format_list("Follow-up actions", self.follow_up_actions),
        ]
        return "\n\n".join(sections)


def get_verify_agent():
    agent = Agent(
        name="Verify Agent",
        instructions="Summarize verification for the current candidate change. Capture what checks were run or should be run, the findings, remaining risks, and concrete follow-up actions.",
        output_type=VerificationReport,
    )
    verify_input = "Summarize verification for the current candidate change."
    return agent, verify_input
