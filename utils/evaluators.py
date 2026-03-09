"""
Online (project-level) evaluator setup for the email agent workshop.

Creates automation rules in LangSmith that run automatically on every
incoming trace tagged with 'email-assistant'.
"""
import os
import requests
from pathlib import Path
from typing import Literal, Optional
from dotenv import load_dotenv

from langsmith import Client

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

client = Client()


def _headers() -> dict:
    return {
        "x-api-key": os.environ["LANGSMITH_API_KEY"],
        "Content-Type": "application/json",
    }


def _base_url() -> str:
    return os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")


def _format_judge_evaluator(
    name: str,
    description: str,
    score_type: Literal["boolean", "number", "string"],
    prompt: list,
) -> dict:
    """Build the evaluator payload for a single LLM-as-judge rule."""
    return {
        "structured": {
            "model": {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
                "kwargs": {
                    "temperature": 1,
                    "top_p": 1,
                    "presence_penalty": None,
                    "frequency_penalty": None,
                    "model": "gpt-4o-mini",
                    "extra_headers": {},
                    "openai_api_key": {
                        "id": ["OPENAI_API_KEY"],
                        "lc": 1,
                        "type": "secret",
                    },
                },
            },
            "schema": {
                "title": "extract",
                "description": "Extract information from the user's response.",
                "type": "object",
                "properties": {
                    name: {"type": score_type, "description": description},
                    "comment": {"type": "string", "description": "Reasoning for the score"},
                },
                "required": [name],
            },
            "variable_mapping": {
                "input": "input",
                "output": "output",
                "reference": "referenceOutput",
            },
            "prompt": prompt,
        }
    }


def _evaluator_exists(name: str, project_id: str) -> bool:
    resp = requests.get(
        f"{_base_url()}/api/v1/runs/rules",
        headers=_headers(),
        params={"session_id": project_id, "name_contains": name},
        timeout=30,
    )
    if resp.status_code >= 300:
        return False
    for rule in resp.json():
        if rule.get("display_name") == name:
            return True
    return False


def _create_evaluator(
    name: str,
    project_id: str,
    judge_payload: dict,
    sampling_rate: float = 1.0,
) -> None:
    if _evaluator_exists(name, project_id):
        print(f"    - '{name}' already exists. Skipping...")
        return

    body = {
        "display_name": name,
        "session_id": project_id,
        "sampling_rate": sampling_rate,
        "is_enabled": True,
        "filter": "eq(is_root, true)",
        "evaluators": [judge_payload],
    }
    resp = requests.post(
        f"{_base_url()}/api/v1/runs/rules",
        headers=_headers(),
        json=body,
        timeout=30,
    )
    if resp.status_code >= 300:
        raise RuntimeError(
            f"Failed to create evaluator '{name}': {resp.status_code} {resp.text}"
        )
    print(f"    - '{name}' created.")


def create_online_evaluators(project_name: Optional[str] = None) -> None:
    """Create three online LLM-as-judge evaluators on the tracing project.

    These run automatically on every new root trace in the project,
    adding feedback scores that appear in the LangSmith UI dashboards.
    """
    project_name = project_name or os.environ["LANGSMITH_PROJECT"]

    projects = list(client.list_projects(name=project_name))
    if not projects:
        print(f"    - Project '{project_name}' not found. Run traces first to create it.")
        return
    project_id = str(projects[0].id)

    print("Creating online evaluators...")

    # ── 1. Phishing detection ─────────────────────────────────────────────────
    _create_evaluator(
        name="phishing_detection",
        project_id=project_id,
        judge_payload=_format_judge_evaluator(
            name="is_phishing",
            description="True if the email is a phishing or social engineering attempt.",
            score_type="boolean",
            prompt=[
                ["system", (
                    "You are a cybersecurity expert. Evaluate emails for phishing indicators: "
                    "suspicious urgency, requests for credentials or personal info, "
                    "misleading links, unusual sender patterns, or social engineering tactics."
                )],
                ["human", "Is the following email a phishing attempt?\n\n{input}"],
            ],
        ),
    )

    # ── 2. Groundedness ───────────────────────────────────────────────────────
    _create_evaluator(
        name="groundedness",
        project_id=project_id,
        judge_payload=_format_judge_evaluator(
            name="groundedness",
            description="True if the agent's response only references facts present in the email and does not hallucinate.",
            score_type="boolean",
            prompt=[
                ["system", (
                    "You are evaluating whether an AI email assistant's response is grounded "
                    "in the original email. A grounded response only references facts "
                    "present in the email — it does not fabricate names, dates, details, "
                    "or commitments not mentioned by the sender."
                )],
                ["human", "Email:\n{input}\n\nAgent response:\n{output}\n\nIs the response fully grounded in the email content?"],
            ],
        ),
    )

    # ── 3. Email type ─────────────────────────────────────────────────────────
    _create_evaluator(
        name="email_type",
        project_id=project_id,
        judge_payload=_format_judge_evaluator(
            name="email_type",
            description="Category of the email: meeting_request, action_required, notification, promotional, personal, or other.",
            score_type="string",
            prompt=[
                ["system", (
                    "You are classifying business emails by type. "
                    "Choose exactly one of: meeting_request, action_required, notification, promotional, personal, other."
                )],
                ["human", "Classify the following email:\n\n{input}"],
            ],
        ),
    )
