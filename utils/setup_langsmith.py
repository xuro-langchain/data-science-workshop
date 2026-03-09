"""
Run this script to pre-load LangSmith assets for the workshop.
Safe to re-run — all steps are idempotent.

  - Agent and eval prompts  (skipped if already exist)
  - Labeled email datasets  (skipped if already exist)
  - Online evaluators       (skipped if already exist)
  - All 16 traces           (skipped if project already has traces)

Usage:
    python utils/setup_langsmith.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from langsmith import Client
from utils.prompts import load_all_prompts
from utils.datasets import load_datasets
from utils.traces import create_traces
from utils.evaluators import create_online_evaluators


def _project_has_traces(client: Client, project_name: str) -> bool:
    """Return True if the project exists and already has at least one trace."""
    try:
        runs = list(client.list_runs(project_name=project_name, is_root=True, limit=2))
        return len(runs) > 1
    except Exception:
        return False


def main(force_traces: bool = False):
    client = Client()
    project_name = os.environ["LANGSMITH_PROJECT"]

    load_all_prompts()
    print()
    load_datasets()
    print()

    has_traces = _project_has_traces(client, project_name)  # check before touching anything

    if force_traces or not has_traces:
        create_traces(num_examples=1)   # bootstrap — creates the project
        create_online_evaluators()      # attach rules to the now-existing project
        time.sleep(3)                   # brief pause for rules to register
        create_traces()                 # all 16 traces — evaluators fire on each one
    else:
        print("Traces already exist — skipping. Use --force-traces to generate more.")
        create_online_evaluators()      # still idempotent — ensure rules exist


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-traces",
        action="store_true",
        help="Generate a new batch of traces even if the project already has traces.",
    )
    args = parser.parse_args()
    main(force_traces=args.force_traces)
