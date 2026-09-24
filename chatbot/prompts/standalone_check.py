"""chatbot.prompts.standalone_check — classify_node's second-opinion prompt
(chatbot/nodes.py::_depends_on_history).

Generic (non-keyword) sanity check run only when the supervisor prompt
(supervisor.py) already said "smalltalk" and prior turns exist: asks the LLM
directly, instead of pattern-matching specific phrasings — no fixed word list
generalizes to real production traffic.

WHAT IT ASKS MATTERS. The first version asked "does understanding this message
depend on the earlier turns?" — and a 'dependent' answer was treated as "this is
a follow-up question". Those are not the same thing, and the gap was measured on
2026-09-19: for "got it" the model correctly answered 'dependent' (the phrase
means nothing without the previous turn), the node read that as a data question,
and the literal words "got it" were sent to the SQL engine. The model was right;
the question was wrong. It now asks the thing the caller actually needs to know —
is the user REQUESTING something — which an acknowledgement answers 'no' to while
still being entirely context-dependent.
"""
from __future__ import annotations

STANDALONE_CHECK_SYSTEM = (
    "A chatbot classifier just labeled a user's message as pure smalltalk "
    "(no data question at all). Before trusting that, sanity-check ONE thing: "
    "is the user ASKING FOR SOMETHING in this message — more data, a different "
    "cut of it, a change to what they were just shown — where the earlier turns "
    "are needed to know what they mean? "
    "Answer 'dependent' ONLY for a request of that kind, however it is phrased "
    "(\"what about the other one\", \"tell me more\", \"and last year?\", "
    "\"aur bata\"). "
    "Answer 'standalone' for everything else — including a message that is "
    "context-dependent but asks for NOTHING: acknowledging or reacting to the "
    "answer they just read (\"okay\", \"got it\", \"hmm\", \"makes sense\", "
    "\"acha\"), or a remark that stands on its own. "
    "Output EXACTLY one word: dependent or standalone."
)


def build_standalone_check_user_prompt(message: str, history: list) -> str:
    hist_lines = [f"{t.get('role', 'user')}: {t.get('content', '')}" for t in history[-6:]]
    hist_block = "\n".join(hist_lines)
    return f"Conversation before this message:\n{hist_block}\n\nMessage: {message}"
