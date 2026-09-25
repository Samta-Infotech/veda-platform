"""chatbot.prompts.carryover_check — classify_node's back-reference second opinion
(chatbot/nodes.py::_carries_over_subject).

WHY A THIRD PROMPT EXISTS. The merged supervisor call (supervisor.py) decides
`action` and `delta_type` together, and it was measured on 2026-09-24 to be
CONFIDENTLY WRONG — not unsure — on plain back-referential follow-ups. With a live
frame on `assets_salelisting` and four turns of history, 2 runs each:

    "What are their prices?"                             -> answer / new_topic  (2/2)
    "How much are they?"                                 -> answer / new_topic  (2/2)
    "What do these cost?"                                -> answer / new_topic  (2/2)
    "Which one is the cheapest?"                         -> answer / new_topic  (2/2)
    "Which one has the largest area?"                    -> answer / new_topic  (2/2)
    "Which of these has the lowest price per square foot?" -> answer / new_topic (2/2)

`answer` + `new_topic` is the one combination that carries NO context
(nodes.py's `referential` is False, so ConversationContext.from_frame gets
carry_state=False) AND wipes the drill stack (memory_write_node's
`reset = delta_type == "new_topic"`). So the conversation silently restarted on
every one of these, and the drill stack could never leave zero — which is exactly
what the 25-turn run in evaluation/drilldown_l7/ measured.

WHY NOT REUSE standalone_check.py. It was tried first, on the same 24 messages.
It asks its question inside a "a classifier just labeled this pure smalltalk"
framing, and the framing does not transfer: 14/24 back-references came back
'dependent' (58%) and, worse, 2/8 genuinely self-contained CONTROLS did too
("What is the average rent across all lease listings?", 2/2) — a false positive
there pollutes a brand-new question with the previous subject's filters.

This prompt asks the mechanical question instead — does the message NAME the
records it is about, or stand in for them with a pronoun/demonstrative — and was
measured on the same 24 messages, 2 runs each, fully reproducible (every case 2/2):

    pronoun        carryover 8/8      demonstrative  carryover 6/8
    entity_drill   carryover 6/8      control_new    carryover 0/8

20/24 back-references caught, ZERO of 8 controls misfired. The 4 misses
("Which one has the largest area?", "Tell me more about the cheapest property.")
both name a descriptive noun phrase of their own; they fail closed to the
pre-existing behaviour, which is why this gate can only ever ADD context to a
turn that is today getting none.
"""
from __future__ import annotations

CARRYOVER_CHECK_SYSTEM = (
    "You decide ONE thing about a user's newest message in a data chat: does it "
    "CARRY OVER the subject of the previous turn, or does it name its own subject?\n"
    "Answer 'carryover' when the message uses a pronoun or a demonstrative "
    "(\"they\", \"them\", \"their\", \"it\", \"its\", \"these\", \"those\", "
    "\"this one\", \"which one\", \"of these\") in place of naming the records, so "
    "the only way to know WHICH records it is about is the previous turn.\n"
    "Answer 'own_subject' when the message names the records it is about, even if "
    "it happens to be the same kind of thing as before, and even if it is short.\n"
    "Judge ONLY the newest message. Output EXACTLY one word: carryover or own_subject."
)


def build_carryover_check_user_prompt(frame: dict | None, message: str,
                                      history: list) -> str:
    """The frame's own words for what is on screen, the last few turns, and the
    new message. Nothing here is invented: `understanding`/`entity_display` are
    facts the memory layer already harvested from an EXECUTED query, and the
    message is passed verbatim."""
    frame = frame or {}
    subject = frame.get("entity_display") or frame.get("entity") or ""
    understanding = frame.get("understanding") or ""
    hist_lines = [f"{t.get('role', 'user')}: {t.get('content', '')}"
                  for t in (history or [])[-4:]]
    hist_block = "\n".join(hist_lines) if hist_lines else "(no prior turns)"
    shown = f"They were just shown: {understanding} (records: {subject}).\n" if subject else ""
    return (f"{shown}Conversation before this message:\n{hist_block}\n\n"
            f"Newest message: {message}")
