"""chatbot.prompts.followup — resolve_followup_node's prompt (chatbot/nodes.py).

Rewrites a context-dependent message ("and waived ones?", a clarification
reply, etc.) into one fully self-contained question the engine can answer
with no conversation memory of its own.
"""
from __future__ import annotations

FOLLOWUP_SYSTEM_PROMPT = (
    "If the new message is ALREADY a complete, self-contained data question on its "
    "own (it doesn't rely on 'it'/'that'/an implied entity/a repeated-with-different-"
    "filter pattern from earlier turns), output it VERBATIM — copy it exactly, do not "
    "reword, paraphrase, or 'clean up' the phrasing in any way, even if you think "
    "another wording is clearer. Only if the message truly cannot be understood on "
    "its own, rewrite it as a single, fully self-contained data question using the "
    "conversation history for the missing context (e.g. entity/table/filters implied "
    "by earlier turns) — and even then, reuse the ORIGINAL wording/words for anything "
    "not being filled in, changing only what's necessary to resolve the missing "
    "reference. Output ONLY the question, no explanation, no quotes."
)


def build_followup_user_prompt(message: str, history: list) -> str:
    hist_lines = [f"{t.get('role', 'user')}: {t.get('content', '')}" for t in history[-6:]]
    hist_block = "\n".join(hist_lines) if hist_lines else "(no prior turns)"
    return f"Conversation so far:\n{hist_block}\n\nNew message: {message}"
