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

from .common import tidy, today_str

_DELTA_BLOCK = """

The user also has a CURRENT ANALYTICAL FRAME — what they were just looking \
at, already computed and executed, not a guess:
{frame_json}

If action is "followup" or "clarify_reply" (or the message continues the \
SAME topic as the frame above), ALSO classify which ONE of these the new \
message is, and include it as "delta_type":

- "new_topic"   — asks about a different entity/subject than the frame.
- "refine"      — ADDS a filter/grouping, keeping everything already in the frame.
- "replace"     — swaps the value of a filter the frame ALREADY HAS (frame has \
                   Year 2025, user says "what about 2024?"). Also give "delta_field": \
                   the frame filter being swapped.
- "remove"      — drops a filter the frame ALREADY HAS ("remove India", "exclude \
                   enterprise"). Also give "delta_field": the filter being dropped.
- "drill_down"  — narrows into a MORE SPECIFIC value of a dimension already \
                   in play (e.g. after "by region", user says "North America").
- "drill_up"    — asks to go back / zoom out / remove the most specific filter.
- "compare"     — asks to see two things SIDE BY SIDE, with an explicit \
                   comparison word ("compare", "versus", "vs", "against"). Just \
                   SWITCHING to a different value ("what about 2024?", "and 2023?", \
                   "what about the US?") is "replace", NOT "compare".
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
"new_topic" and "slot_candidates" to [].

EXAMPLES (frame filters shown, then message -> the two fields that matter):
["Year equals 2025"]      "what about 2024?"    -> replace, delta_field "Year"
["Year equals 2025"]      "and 2023?"           -> replace, delta_field "Year"
["Year equals 2025"]      "compare with 2024"   -> compare, no delta_field
["Country equals India"]  "what about the US?"  -> replace, delta_field "Country"
["Country equals India"]  "remove India"        -> remove,  delta_field "Country"
["Status equals Active"]  "only the ones in Mumbai" -> refine, no delta_field

"""


# The HARD RULE below was rewritten on 2026-09-22, after measuring that a real
# question could be refused as chit-chat. It used to enumerate RELATIONAL nouns
# ("incident", "count", "organizations", "status"), and a question naming none of them
# fell off that list: with conversation history present, "what is the dress code" and
# "how long is the probation period" were classified smalltalk 3/3 and answered with
# "I'm here for questions about your data" — the user's question never reached the
# engine. Measured as history-triggered, NOT document- or frame-triggered: the same
# questions asked as a first turn were classified correctly, and swapping the frame
# between none, a document frame and a relational frame changed nothing.
#
# Adding document nouns to the list would only move the cliff, so the rule now states
# the invariant instead: asking for information is never smalltalk; smalltalk requests
# nothing. A/B against the previous wording, history present, both directions probed
# (10 real questions, 20 greetings/thanks/acknowledgements): 28/30 -> 29/29, with no
# greeting or acknowledgement regression, "how's it going" included.
#
# This file's prompt has regressed three times (see the delta-block note below). Any
# further edit goes through evaluation/conversation/run_eval.py --mode full, and is
# reverted on anything below 54/54 scenarios.
def build_supervisor_system_prompt(frame: dict | None = None) -> str:
    delta_addendum = ""
    action_schema = '"action": "smalltalk"|"followup"|"clarify_reply"|"answer", "reason": "<one short phrase>"'
    if frame and frame.get("entity"):
        # measures/order_by/limit are shown because the frame now REMEMBERS them
        # (chatbot/memory/frame.py, 2026-09-17) and the classifier's job depends on
        # them: "the most expensive instead" is a re-ranking of the same question, and
        # a model that cannot see what the ranking WAS has nothing to classify the
        # change against. They were harvested and carried into the resolved query, but
        # never shown here — the one place the decision is actually made.
        frame_view = {
            "entity": frame.get("entity_display") or frame.get("entity"),
            "understanding": frame.get("understanding"),
            "filters": [f"{f.get('field')} {f.get('operator')} {f.get('value')}"
                        for f in (frame.get("filters") or [])],
            "group_by": frame.get("group_by") or [],
            "measures": frame.get("measures") or [],
            "ranked_by": [f"{o.get('field')} ({'highest' if o.get('desc') else 'lowest'} first)"
                          for o in (frame.get("order_by") or []) if o.get("field")],
            "limit": frame.get("limit"),
            "drill_path": frame.get("drill_path") or [],
        }
        # This addendum is paid on every turn that has a frame, and it is the most
        # expensive static text in the package (671 tok). Compressing it was ATTEMPTED
        # and REVERTED on 2026-09-21: rendering it from a shared compact source took the
        # harness 54/54 -> 53/55 with two consistent failures, and restoring the examples
        # to that compact form made it 52/55. The examples and the per-type prose are
        # load-bearing here in a way that does not survive paraphrase. Do not retry
        # without the harness in --mode full, and revert on anything below 54/54.
        delta_addendum = _DELTA_BLOCK.format(frame_json=json.dumps(frame_view, default=str))
        action_schema += (', "delta_type": "new_topic"|"refine"|"replace"|"remove"|'
                          '"drill_down"|"drill_up"|"compare"|"ambiguous", '
                          '"delta_field": "<the frame filter being replaced/removed, '
                          'omit otherwise>", "slot_candidates": [<verbatim words from '
                          'the NEW message, or empty list>], '
                          # STEP 7 (VEDA_MEMORY_LAYER_PLAN.md): a presentation-only request
                          # the whole-message regexes miss ("can i see that as a chart",
                          # "draw it", "show me a graph of that") as a TYPED output. Held
                          # to the harness bar in the header note; validated in code.
                          '"render": "none"|"table"|"chart"|"pie"|"bar"|"line"|"csv" '
                          '(not "none" ONLY when the message asks to see the PREVIOUS '
                          'answer in a different form and asks for nothing new)')

    return tidy(f"""\
You are the front-door supervisor for a data-analyst chatbot. Today's date is \
{today_str()}.

Given the conversation history and the user's new message, classify it into \
EXACTLY one action:

- "smalltalk"  — pure greetings, thanks, goodbyes, casual chit-chat, or a bare \
                 ACKNOWLEDGEMENT of the answer just given that asks for nothing \
                 ("hi", "thanks", "bye", "okay", "got it", "hmm"). Nothing here \
                 references any data, count, entity, table or fact.
- "followup"   — the message only makes sense combined with the previous turn(s), \
                 e.g. it references "it"/"that"/"this"/an implied entity, or asks for the \
                 same kind of thing again with a different filter \
                 (e.g. after "escalated incidents count", user says "and waived ones?", \
                 or after answering a count, user asks "what was my incident count" \
                 to recall/re-ask it). This INCLUDES short vague replies with NO named \
                 entity or metric of their own — e.g. "need more details about this", \
                 "tell me more", "more info?", "what else" — said right after the \
                 assistant discussed a specific record/entity: these are followup, \
                 NEVER smalltalk.
- "clarify_reply" — the previous assistant turn asked a clarifying question, and \
                 this message is the user's answer to it.
- "answer"     — a new, self-contained data question.

HARD RULE: if the message ASKS FOR INFORMATION of any kind — a number, a fact, a \
rule, a definition, a list, or what some document or record says — it is NEVER \
"smalltalk", even when phrased casually, as a recall ("what was...", "remind \
me..."), or about something this chatbot may not hold. "smalltalk" is only for \
messages that REQUEST NOTHING: a greeting, thanks, a farewell, or a bare \
acknowledgement. When unsure between "smalltalk" and any other action, choose the \
other action.
{delta_addendum}

Output ONLY a JSON object, no markdown, no explanation:
{{{action_schema}}}
""")


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
