"""chatbot.prompts.followup — resolve_followup_node's prompt (chatbot/nodes.py).

Rewrites a context-dependent message ("and waived ones?", a clarification
reply, etc.) into one fully self-contained question the engine can answer
with no conversation memory of its own.
"""
from __future__ import annotations

FOLLOWUP_SYSTEM_PROMPT = (
    "Rewrite the user's new message as a single, fully self-contained data "
    "question, using the conversation history for missing context (e.g. "
    "entity/table/filters implied by earlier turns). Output ONLY the "
    "rewritten question, no explanation, no quotes."
)


def build_followup_user_prompt(message: str, history: list) -> str:
    hist_lines = [f"{t.get('role', 'user')}: {t.get('content', '')}" for t in history[-6:]]
    hist_block = "\n".join(hist_lines) if hist_lines else "(no prior turns)"
    return f"Conversation so far:\n{hist_block}\n\nNew message: {message}"
