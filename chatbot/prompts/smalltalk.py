"""chatbot.prompts.smalltalk — smalltalk_node's prompt (chatbot/nodes.py).

Only ever invoked after classify_node has already confirmed the message is
pure chit-chat with no data question — this prompt must never invent data
facts, even so.

OFF BY DEFAULT since 2026-09-21 (chatbot/nodes.py::smalltalk_node, flag
CHATBOT_SMALLTALK_LLM_REPLY). Kept correct rather than deleted because the flag
restores it, and because the fixed replies below are held to the same rules:
English only, and no claim about the user's data.
"""
from __future__ import annotations

from .common import today_str


def build_smalltalk_system_prompt() -> str:
    return (
        f"You are a data-analytics chatbot. Today's date is {today_str()}. "
        "Reply to this casual message (a greeting/thanks/goodbye — it has "
        "already been confirmed to contain no data question) in ONE short, "
        "warm sentence. If it's a greeting or the start of a conversation, "
        "briefly introduce yourself as someone who can help with data "
        "analytics questions (counts, trends, records, etc.) and invite them "
        "to ask one. NEVER state or imply any number, count, status, or fact "
        "about the user's actual data — you have no access to it here and "
        "must not invent one, even to sound helpful."
    )


FALLBACK_REPLY = "Hi! I'm your data-analytics assistant — ask me anything about your data."

# For smalltalk the canned patterns above did NOT match — a typo ("helo"), an
# acknowledgement ("okay"), another language ("hola"). Deliberately tone-NEUTRAL, unlike
# FALLBACK_REPLY, which is a greeting and reads wrong as an answer to "ok" or "thanks".
#
# This replaces a second model call. Measured 2026-09-21: that call cost 5-20s on its own
# (a typo'd greeting took 12.3s end to end, of which ~half was this), and it produced
# "¡Hola! ... preguntas sobre数据分析" for "hola" — a mixed Spanish/Chinese reply. The
# classify call has ALREADY decided the turn is smalltalk; asking a second time for the
# words is latency and a drift surface, not accuracy.
SMALLTALK_FALLBACK_REPLY = (
    "I'm here for questions about your data — ask me anything whenever you're ready."
)

# A message made of nothing but punctuation — "???", "...", "!!!". It carries no
# request, so there is nothing to answer and nothing to look up, and unlike every other
# reply here it should say so rather than sound like an invitation. Measured
# 2026-09-21: "???" was classified as a data question and spent a full engine
# round-trip before this existed.
REPHRASE_REPLY = "I'm not sure I understood that. Could you rephrase your question?"

# Deterministic answers for the two questions every new user actually opens with.
# These need no model and no engine: the answer is a property of the PRODUCT, not
# of the user's data, so there is nothing to look up and nothing to invent. Before
# these existed, "who are you" / "what can you do" fell through the greeting
# patterns, and classify_node — which defaults to "answer" whenever the classifier
# is unavailable — sent them into the SQL engine, which has no such table and
# refused. Neither reply states a single fact about the user's data.
IDENTITY_REPLY = (
    "I'm VEDA, your data-analytics assistant. You ask questions about your data in "
    "plain English and I work out which tables answer them, run the query, and show "
    "you the result."
)

CAPABILITY_REPLY = (
    "Ask me about your data in plain English — counts, totals, averages, rankings, "
    "trends over time, or a straight listing of records. I find the right tables, "
    "run the query, and reply with a table, a chart where one helps, and an "
    "explanation of how I got there. You can follow up in the same breath — "
    "\"only the ones in Mumbai\", \"break that down by month\", \"go back\" — and I "
    "keep the earlier question's context. If I can't answer something from your "
    "data, I'll say so rather than guess."
)
