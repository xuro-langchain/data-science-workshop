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


def _get_evaluator_id(name: str, project_id: str) -> Optional[str]:
    """Return the rule ID for an evaluator with this display name, or None."""
    resp = requests.get(
        f"{_base_url()}/api/v1/runs/rules",
        headers=_headers(),
        params={"session_id": project_id},
        timeout=30,
    )
    if resp.status_code >= 300:
        return None
    for rule in resp.json():
        if rule.get("display_name") == name:
            return rule.get("id")
    return None


def _delete_evaluator(name: str, project_id: str) -> None:
    rule_id = _get_evaluator_id(name, project_id)
    if rule_id:
        requests.delete(
            f"{_base_url()}/api/v1/runs/rules/{rule_id}",
            headers=_headers(),
            timeout=30,
        )


def _create_evaluator(
    name: str,
    project_id: str,
    judge_payload: dict,
    sampling_rate: float = 1.0,
    force: bool = False,
) -> None:
    existing_id = _get_evaluator_id(name, project_id)
    if existing_id:
        if not force:
            print(f"    - '{name}' already exists. Skipping...")
            return
        _delete_evaluator(name, project_id)

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
    print(f"    - '{name}' {'recreated' if existing_id else 'created'}.")


def create_online_evaluators(project_name: Optional[str] = None, force: bool = False) -> None:
    """Create online LLM-as-judge evaluators on the tracing project.

    These run automatically on every new root trace, adding feedback scores
    that appear in the LangSmith UI dashboards.

    Args:
        force: If True, delete and recreate any evaluators that already exist.
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
        force=force,
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
        force=force,
        judge_payload=_format_judge_evaluator(
            name="groundedness",
            description="True if the agent's response does not misrepresent or fabricate facts from the email.",
            score_type="boolean",
            prompt=[
                ["system", (
                    "You are evaluating whether an AI email assistant's response is factually "
                    "accurate relative to the email it received.\n\n"
                    "A response IS grounded if it:\n"
                    "- Correctly identifies what the email is about\n"
                    "- Adds reasonable professional context (e.g. 'I'll look into this', offering a timeline)\n"
                    "- Proposes actions not explicitly requested but appropriate to the situation\n\n"
                    "A response is NOT grounded only if it:\n"
                    "- Fabricates specific facts stated in the email (wrong names, wrong deadlines, wrong details)\n"
                    "- Misrepresents what the sender asked for\n"
                    "- Invents information that contradicts the email\n\n"
                    "Be generous: most reasonable professional responses should be grounded."
                )],
                ["human", "Email received:\n{input}\n\nAgent response:\n{output}\n\nIs the response factually grounded?"],
            ],
        ),
    )

    # ── 3. Email type ─────────────────────────────────────────────────────────
    _create_evaluator(
        name="email_type",
        project_id=project_id,
        force=force,
        judge_payload=_format_judge_evaluator(
            name="email_type",
            description="Category: meeting_request, action_required, notification, promotional, or personal.",
            score_type="string",
            prompt=[
                ["system", (
                    "Classify the following email into exactly one of these categories:\n\n"
                    "- meeting_request: asks to schedule, confirm, or discuss timing for a meeting or call\n"
                    "- action_required: asks the recipient to do something — review a document, "
                    "submit a report, answer a question, or investigate an issue\n"
                    "- notification: FYI only — alerts, status updates, reminders, GitHub notifications, "
                    "system alerts, subscription renewals — no direct action needed\n"
                    "- promotional: marketing emails, newsletters, conference invitations, product announcements\n"
                    "- personal: personal life matters such as medical appointments or family activities\n\n"
                    "Reply with only the category name."
                )],
                ["human", "{input}"],
            ],
        ),
    )

    # ── 4. Professionalism ────────────────────────────────────────────────────
    _create_evaluator(
        name="professionalism",
        project_id=project_id,
        force=force,
        judge_payload=_format_judge_evaluator(
            name="professionalism",
            description="True if the agent's response meets business communication standards.",
            score_type="boolean",
            prompt=[
                ["system", (
                    "You are evaluating the professional quality of an AI email assistant's response.\n\n"
                    "A professional response:\n"
                    "- Uses clear, polite, and respectful business language\n"
                    "- Is appropriately concise without being curt\n"
                    "- Addresses the email's purpose directly\n"
                    "- Avoids slang, inappropriate informality, or rudeness\n\n"
                    "For emails that are ignored or only notified (no response drafted), "
                    "score as True — the decision itself can be professional even with no written reply.\n\n"
                    "Be generous: most competent responses should pass."
                )],
                ["human", "Email received:\n{input}\n\nAgent response:\n{output}\n\nIs this a professional response?"],
            ],
        ),
    )
