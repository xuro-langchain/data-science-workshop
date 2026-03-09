import sys
from pathlib import Path

# Allow running directly as a script from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from utils.agent import email_assistant
from utils.datasets import email_inputs, email_names


def create_traces(num_examples: int = 16):
    """Invoke the email assistant on dataset examples to generate LangSmith traces."""
    for idx, (email_input, name) in enumerate(zip(email_inputs, email_names)):
        if idx >= num_examples:
            break

        print(f"\n--- Running example {idx + 1}: {name} ---")
        result = email_assistant.invoke(
            {"email_input": email_input},
            config={
                "run_name": f"email_assistant:{name}",
                "tags": ["email-assistant", "trace", name],
                "metadata": {"dataset": "email_inputs", "example_name": name, "example_index": idx},
            },
        )

        decision = result.get("classification_decision", "unknown")
        print(f"classification_decision: {decision}")

        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            content = last.get("content") if isinstance(last, dict) else getattr(last, "content", None)
            if content:
                preview = str(content)[:300] + ("..." if len(str(content)) > 300 else "")
                print(f"last_message_preview: {preview}")


if __name__ == "__main__":
    create_traces()
