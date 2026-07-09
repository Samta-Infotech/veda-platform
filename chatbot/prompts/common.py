"""chatbot.prompts.common — helpers shared across every prompt file."""
from __future__ import annotations

from datetime import date


def today_str() -> str:
    """Today's date, computed fresh on every call (never hardcoded) — used so
    prompts can reason about "today"/"this week"/relative dates correctly
    whenever the process happens to run."""
    return date.today().strftime("%Y-%m-%d (%A)")
