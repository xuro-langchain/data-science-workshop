from datetime import datetime
from typing import Literal
from pydantic import BaseModel
from langchain_core.tools import tool

@tool
def schedule_meeting(
    attendees: list[str], subject: str, duration_minutes: int, preferred_day: datetime, start_time: int
) -> str:
    """Schedule a calendar meeting."""
    date_str = preferred_day.strftime("%A, %B %d, %Y")
    return f"Meeting '{subject}' scheduled on {date_str} at {start_time} for {duration_minutes} minutes with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    return f"Email sent to {to} with subject '{subject}' and content: {content}"

@tool
class Done(BaseModel):
    """E-mail has been sent."""
    done: bool

def get_tools():
    return [schedule_meeting, check_calendar_availability, write_email, Done]

def get_tools_by_name(tools):
    return {tool.name: tool for tool in tools}
