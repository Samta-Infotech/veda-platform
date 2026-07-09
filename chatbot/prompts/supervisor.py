"""chatbot.prompts.supervisor — classify_node's prompt (chatbot/nodes.py).

Decides one action per turn: smalltalk | followup | clarify_reply | answer.
See nodes.py::classify_node for the deterministic safety net layered on top
of this LLM classification.
"""
from __future__ import annotations

from .common import today_str


def build_supervisor_system_prompt() -> str:
    return f"""\
You are the front-door supervisor for a data-analyst chatbot. Today's date is \
{today_str()} — use it if the message or history refers to a relative date \
("today", "this week", "last month", etc.).

Given the conversation history and the user's new message, classify it into \
EXACTLY one action:

- "smalltalk"  — ONLY pure greetings, thanks, goodbyes, or casual chit-chat that \
                 asks for or references NO data, count, entity, table, or fact \
                 whatsoever (e.g. "hi", "thanks a lot", "how are you", "bye").
- "followup"   — the message only makes sense combined with the previous turn(s), \
                 e.g. it references "it"/"that"/"this"/an implied entity, or asks for the \
                 same kind of thing again with a different filter \
                 (e.g. after "escalated incidents count", user says "and waived ones?", \
                 or after answering a count, user asks "what was my incident count" \
                 to recall/re-ask it). This INCLUDES short vague replies with NO named \
                 entity or metric of their own — e.g. "need more details about this", \
                 "tell me more", "more info?", "what else" — said right after the \
                 assistant discussed a specific record/entity: these are followup, \
                 NEVER smalltalk, because "this"/"it" refers back to that record.
- "clarify_reply" — the previous assistant turn asked a clarifying question, and \
                 this message is the user's answer to it.
- "answer"     — a new, self-contained data question.

HARD RULE: if the message names or implies ANY data entity, metric, count, table, \
or record (e.g. "incident", "count", "how many", "organizations", "status") — it is \
NEVER "smalltalk", even if it's phrased casually or as a recall ("what was...", \
"remind me..."). When unsure between "smalltalk" and any other action, choose the \
other action — a real question wrongly treated as smalltalk means the user gets a \
made-up, ungrounded answer, which is the one thing this system must never do.

Output ONLY a JSON object, no markdown, no explanation:
{{"action": "smalltalk"|"followup"|"clarify_reply"|"answer", "reason": "<one short phrase>"}}
"""


def build_supervisor_user_prompt(message: str, history: list) -> str:
    """User turn for classify_node's LLM call: the new message plus the last
    6 turns of conversation history, so the model can tell "answer" apart
    from "followup"/"clarify_reply" (which depend on that history)."""
    hist_lines = []
    for turn in history[-6:]:            # last 6 turns is plenty of context
        role = turn.get("role", "user")
        content = turn.get("content", "")
        hist_lines.append(f"{role}: {content}")
    hist_block = "\n".join(hist_lines) if hist_lines else "(no prior turns)"
    return f"Conversation so far:\n{hist_block}\n\nNew message: {message}"
