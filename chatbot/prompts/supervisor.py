"""chatbot.prompts.supervisor — classify_node's prompt (chatbot/nodes.py).

Decides one action per turn: smalltalk | followup | clarify_reply | answer.
See nodes.py::classify_node for the deterministic safety net layered on top
of this LLM classification.

Latency fix: when a structured QueryFrame already exists (`frame` passed in
below), this SAME call ALSO asks for the memory delta classification
(chatbot/memory/classify.py's job) — new_topic|refine|drill_down|drill_up|
compare|ambiguous, plus grounded slot_candidates — instead of
context_resolve_node making a SECOND, separate SLM round-trip afterward.
This was the original design intent (merge into one call, net latency win)
that the first cut of chatbot/memory/ shipped as a second call instead
(flagged in the memory-system audit's Performance Assessment) — fixed here.
When `frame` is None/empty (no prior successful query this session), the
prompt is IDENTICAL to before this change — zero behavior change for the
common "first turn" / "no memory yet" case.
"""
from __future__ import annotations

import json

from .common import today_str

_DELTA_BLOCK = """

The user also has a CURRENT ANALYTICAL FRAME — what they were just looking \
at, already computed and executed, not a guess:
{frame_json}

If action is "followup" or "clarify_reply" (or the message continues the \
SAME topic as the frame above), ALSO classify which ONE of these the new \
message is, and include it as "delta_type":

- "new_topic"   — asks about a different entity/subject than the frame.
- "refine"      — adds or changes a filter/grouping on the SAME entity.
- "drill_down"  — narrows into a MORE SPECIFIC value of a dimension already \
                   in play (e.g. after "by region", user says "North America").
- "drill_up"    — asks to go back / zoom out / remove the most specific filter.
- "compare"     — asks to compare the frame against another time period or \
                   another value of the same dimension.
- "ambiguous"   — references something ("it", "that", "inactive ones") that \
                   is NOT clearly resolvable from the frame with high \
                   confidence. When unsure, choose this — never guess.

CRITICAL: never invent a column, table, or filter value that isn't the \
frame's own remembered fact or a word the user just typed in the NEW \
message. Also include "slot_candidates": a list of words copied VERBATIM \
from the NEW message that name a filter value, dimension, or time period \
(empty list if none — do not invent one).

If action is "smalltalk" or "answer" (a genuinely new, self-contained \
question unrelated to continuing the frame), set "delta_type" to \
"new_topic" and "slot_candidates" to []."""


def build_supervisor_system_prompt(frame: dict | None = None) -> str:
    delta_addendum = ""
    action_schema = '"action": "smalltalk"|"followup"|"clarify_reply"|"answer", "reason": "<one short phrase>"'
    if frame and frame.get("entity"):
        frame_view = {
            "entity": frame.get("entity_display") or frame.get("entity"),
            "understanding": frame.get("understanding"),
            "filters": [f"{f.get('field')} {f.get('operator')} {f.get('value')}"
                        for f in (frame.get("filters") or [])],
            "group_by": frame.get("group_by") or [],
            "drill_path": frame.get("drill_path") or [],
        }
        delta_addendum = _DELTA_BLOCK.format(frame_json=json.dumps(frame_view, default=str))
        action_schema += (', "delta_type": "new_topic"|"refine"|"drill_down"|"drill_up"|'
                          '"compare"|"ambiguous", "slot_candidates": [<verbatim words from '
                          'the NEW message, or empty list>]')

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
{delta_addendum}

Output ONLY a JSON object, no markdown, no explanation:
{{{action_schema}}}
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
