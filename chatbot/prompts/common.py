"""chatbot.prompts.common — helpers shared across every prompt file."""
from __future__ import annotations

import re

from datetime import date


def today_str() -> str:
    """Today's date, computed fresh on every call (never hardcoded) — used so
    prompts can reason about "today"/"this week"/relative dates correctly
    whenever the process happens to run."""
    return date.today().strftime("%Y-%m-%d (%A)")

_RUN_OF_SPACES = re.compile(r"[ \t]{2,}")


def tidy(prompt: str) -> str:
    """Collapse the runs of spaces that `\\` line continuations leave inside a
    triple-quoted prompt.

    The source aligns its bullets for readability; Python keeps that alignment as
    LITERAL spaces, so the rendered prompt reads
    `"drill_down" — narrows into a MORE SPECIFIC value of a dimension already<18 spaces>in play`.
    Measured 2026-09-17: 7-9% of every prompt was these runs. Harmless to the model but
    paid for on every call, and it blurs the structure the alignment was meant to show.
    Only spaces and tabs are touched — newlines, and therefore the layout, are kept.
    """
    return "\n".join(_RUN_OF_SPACES.sub(" ", line).rstrip() for line in prompt.split("\n"))
