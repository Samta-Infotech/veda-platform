"""chatbot.prompts.delta_classify — chatbot/memory/classify.py's ONE prompt.

Replaces followup.py's free-text rewrite (FOLLOWUP_SYSTEM_PROMPT) for any
turn where a structured QueryFrame already exists (see docs/MEMORY_ARCHITECTURE.md
§8). The model is deliberately NEVER asked to invent a column name, table
name, or filter value — it only classifies which of a closed set of
"continuation types" the new message is, and (optionally) points at words
that already appear in the user's OWN message as candidate slot values.
chatbot/memory/frame.py's render_frame_as_query() does the actual merging,
using only the frame's own previously-PROVEN facts (harvested from an
executed, already-validated query) plus the user's verbatim new words —
never anything the model supplies from thin air.

Falls back to followup.py's free rewrite (unchanged) when this classifier
itself says "ambiguous", or when there is no frame yet — see
chatbot/nodes.py::context_resolve_node.
"""
from __future__ import annotations

import json

from .common import today_str

DELTA_TYPES = ("new_topic", "refine", "drill_down", "drill_up", "compare", "ambiguous")


def build_delta_classify_system_prompt() -> str:
    return f"""\
You are a strict continuation classifier for an enterprise analytics assistant. \
Today's date is {today_str()}.

You will see the CURRENT ANALYTICAL FRAME (what the user was just looking at — \
already computed and executed, not a guess) and the user's NEW message. Decide \
which ONE of these the new message is:

- "new_topic"   — asks about a different entity/subject than the current frame \
                   (e.g. frame is about revenue, message asks about compliance \
                   incidents).
- "refine"      — adds or changes a filter/grouping on the SAME entity in the \
                   current frame (e.g. "only Finance", "group by department").
- "drill_down"  — narrows into a MORE SPECIFIC value of a dimension already in \
                   play (e.g. after "by region", user says "North America").
- "drill_up"    — asks to go back / zoom out / remove the most specific filter \
                   ("go back", "zoom out", "remove that filter", "show all again").
- "compare"     — asks to compare the current frame against another time period \
                   or another value of the same dimension ("compare with last \
                   month", "compare with Sales").
- "ambiguous"   — the message references something ("it", "that", "the other \
                   one", "inactive ones") that is NOT clearly resolvable from the \
                   current frame with high confidence. When unsure, choose this \
                   — it is always safer to ask than to guess.

CRITICAL RULES:
1. NEVER invent a column, table, or filter value that isn't the frame's own \
   remembered fact or a word the user just typed in the NEW message.
2. If the new message names a field/value with NO relationship to the current \
   frame's entity, and could reasonably stand alone, prefer "new_topic".
3. If you are not at least reasonably confident, output "ambiguous" — never guess.

Output ONLY a JSON object, no markdown, no explanation:
{{"delta_type": "new_topic"|"refine"|"drill_down"|"drill_up"|"compare"|"ambiguous", \
"slot_candidates": [<words copied VERBATIM from the NEW message that name a filter \
value, dimension, or time period — empty list if none>], "reason": "<one short phrase>"}}
"""


def build_delta_classify_user_prompt(frame: dict, message: str, episodic: list | None = None) -> str:
    """`episodic` (audit fix H1 — was previously computed/stored but never
    actually passed to this prompt at all): the short, Redis-capped
    [user, assistant] buffer (chatbot/memory/store.py, at most
    _EPISODIC_MAX turns), for reference resolution ONLY (e.g. "tell me
    more", "what about that one") — never a source of new filter/entity
    facts (those come exclusively from `frame`, which is itself
    evidence-only). Rendered compactly; assistant turns are already
    one-line templated gists (chatbot/nodes.py::_templated_gist), never the
    full reply, so this stays small regardless of how verbose an actual
    answer was.
    """
    frame_view = {
        "entity": frame.get("entity_display") or frame.get("entity"),
        "understanding": frame.get("understanding"),
        "filters": [f"{f.get('field')} {f.get('operator')} {f.get('value')}"
                    for f in (frame.get("filters") or [])],
        "group_by": frame.get("group_by") or [],
        "drill_path": frame.get("drill_path") or [],
    }
    recent = ""
    if episodic:
        lines = [f"{t.get('role', 'user')}: {t.get('content', '')}" for t in episodic]
        recent = "\n\nMost recent turns (for resolving references like 'it'/'that' ONLY " \
                 "— do not pull new filter/entity facts from here, only from the frame " \
                 "above):\n" + "\n".join(lines)
    return (f"Current frame:\n{json.dumps(frame_view, default=str)}"
            f"{recent}\n\nNew message: {message}")
