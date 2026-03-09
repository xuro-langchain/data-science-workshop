import sqlite3
import requests
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


# ── Email helpers ──────────────────────────────────────────────────────────────

def parse_email(email_input: dict) -> tuple:
    """Return (author, to, subject, email_thread) from an email input dict."""
    return (
        email_input["author"],
        email_input["to"],
        email_input["subject"],
        email_input["email_thread"],
    )

def format_email_markdown(subject, author, to, email_thread, email_id=None) -> str:
    """Format email fields into a markdown string for display."""
    id_section = f"\n**ID**: {email_id}" if email_id else ""
    return f"\n\n**Subject**: {subject}\n**From**: {author}\n**To**: {to}{id_section}\n\n{email_thread}\n\n---\n"

def triage_eval(run, example):
    """Simple code evaluator for triage classification (for use with LangSmith evaluate)."""
    correctness = (
        example["outputs"]["classification"].lower()
        == run["outputs"]["output"]["content"].split("\n\n")[0].lower()
    )
    return {"correctness": correctness}


# ── Graph display ──────────────────────────────────────────────────────────────

def show_graph(graph, xray=False):
    """Display a LangGraph mermaid diagram with ASCII fallback.
    
    Args:
        graph: The LangGraph object that has a get_graph() method
        xray: Whether to show the internal structure of the graph
    """
    from IPython.display import Image
    try:
        return Image(graph.get_graph(xray=xray).draw_mermaid_png())
    except Exception as e:
        print(f"⚠️  Image rendering failed: {e}")
        print("\n📊 Showing ASCII diagram instead:\n")
        ascii_diagram = graph.get_graph(xray=xray).draw_ascii()
        print(ascii_diagram)
        return None

def get_engine_for_chinook_db():
    """Pull sql file, populate in-memory database, and create engine."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )