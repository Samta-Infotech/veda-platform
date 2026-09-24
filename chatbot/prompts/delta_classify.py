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

from .common import tidy, today_str

# Re-exported from the canonical source (chatbot/prompts/delta_types.py) so the closed
# set and the field rule exist ONCE. Every existing importer keeps working.
#
# The per-type PROSE below is deliberately NOT shared with the supervisor's compact
# rendering. This is the focused prompt — the one measured classifying shape operations
# 4/4 where the merged supervisor scored 0/4 — and the extra detail it carries about
# limit/group_by/order_by IS that difference. Sharing the wording to save duplication
# would trade a measured accuracy win for tidiness.
from .delta_types import DELTA_TYPES, DELTA_TYPES_WITH_FIELD  # noqa: F401


def build_delta_classify_system_prompt() -> str:
    return tidy(f"""\
You are a strict continuation classifier for an enterprise analytics assistant. \
Today's date is {today_str()}.

You will see the CURRENT ANALYTICAL FRAME (what the user was just looking at — \
already computed and executed, not a guess) and the user's NEW message. Decide \
which ONE of these the new message is:

- "new_topic"   — asks about a different entity/subject than the current frame \
                   (e.g. frame is about revenue, message asks about compliance \
                   incidents).
- "refine"      — ADDS a filter/grouping on the SAME entity, keeping everything \
                   already in the frame (e.g. "only active ones", "group by \
                   department").
- "replace"     — swaps something the frame ALREADY HAS for a new one. Either a \
                   FILTER VALUE (frame has Year 2025, user says "what about 2024?") \
                   or the SHAPE of the question: how many rows ("make it top 10" -> \
                   delta_field "limit"), what it is broken down by ("by month \
                   instead" -> delta_field "group_by"), or what it is sorted by \
                   ("sort by amount instead" -> delta_field "order_by"). Also \
                   include "delta_field": the filter name, or the exact word \
                   "limit", "group_by", "order_by" or "measures".
- "remove"      — drops something the frame ALREADY HAS ("remove India", "without \
                   the date filter"), including a shape slot: "don't sort by \
                   amount" -> delta_field "order_by", "show all of them" -> \
                   delta_field "limit". Also include "delta_field", same rule as \
                   "replace".
- "drill_down"  — narrows into a MORE SPECIFIC value of a dimension already in \
                   play (e.g. after "by region", user says "North America").
- "drill_up"    — asks to go back / zoom out / remove the most specific filter \
                   ("go back", "zoom out", "remove that filter", "show all again").
- "compare"     — asks to see two things SIDE BY SIDE, using an explicit \
                   comparison word: "compare", "versus", "vs", "against", "both", \
                   "difference between" ("compare 2025 with 2024"). If the user is \
                   just SWITCHING to a different value and expects only the new one \
                   ("what about 2024?", "and 2023?", "what about the US?"), that is \
                   "replace", NOT "compare".
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

EXAMPLES (frame shown, then message -> output):
frame filters ["Year equals 2025"]        "what about 2024?"
  {{"delta_type": "replace", "delta_field": "Year", "slot_candidates": ["2024"]}}
frame filters ["Year equals 2025"]        "and 2023?"
  {{"delta_type": "replace", "delta_field": "Year", "slot_candidates": ["2023"]}}
frame filters ["Year equals 2025"]        "compare 2025 with 2024"
  {{"delta_type": "compare", "slot_candidates": ["2024"]}}
frame filters ["Country equals India"]    "what about the US?"
  {{"delta_type": "replace", "delta_field": "Country", "slot_candidates": ["US"]}}
frame filters ["Country equals India"]    "remove India"
  {{"delta_type": "remove", "delta_field": "Country", "slot_candidates": ["India"]}}
frame filters ["Status equals Active"]    "only the ones in Mumbai"
  {{"delta_type": "refine", "slot_candidates": ["Mumbai"]}}
frame limit 100                           "make it top 10"
  {{"delta_type": "replace", "delta_field": "limit", "slot_candidates": ["10"]}}
frame group_by ["year"]                   "show it by month instead"
  {{"delta_type": "replace", "delta_field": "group_by", "slot_candidates": ["month"]}}
frame ranked_by ["amount (highest first)"] "don't sort by amount"
  {{"delta_type": "remove", "delta_field": "order_by", "slot_candidates": ["amount"]}}

Output ONLY a JSON object, no markdown, no explanation:
{{"delta_type": "new_topic"|"refine"|"replace"|"remove"|"drill_down"|"drill_up"|\
"compare"|"ambiguous", "delta_field": "<the frame filter, or one of \
"limit"/"group_by"/"order_by"/"measures", being replaced/removed; omit otherwise>", "slot_candidates": [<words copied VERBATIM from the NEW message \
that name a filter value, dimension, or time period — empty list if none>]}}
""")


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
        # Shown because delta_field may now name one of these (see the system prompt):
        # a model cannot be asked to replace a limit or a ranking it cannot see.
        "measures": frame.get("measures") or [],
        "ranked_by": [f"{o.get('field')} ({'highest' if o.get('desc') else 'lowest'} first)"
                      for o in (frame.get("order_by") or []) if o.get("field")],
        "limit": frame.get("limit"),
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
