"""apps.chat.turn_events — accumulation of one assistant turn's event stream.

Pulled out of ``views.py`` for the same reason ``table_rendering.py`` and
``visualization.py`` were pulled out of ``services.py``: ``views.py`` imports
``services``, which imports ``chatbot.run`` → langgraph → redis. That heavy chain
makes anything defined alongside it effectively untestable in a plain unit-test
environment. This module has ZERO Django, DRF, and chatbot dependencies, so the
turn-folding rules can be exercised directly.

No behaviour lives here that did not already live in the two view methods — this
is the shared half of what were previously two byte-identical if/elif ladders.
"""
from __future__ import annotations


class TurnEventAccumulator:
    """Collects the ordered events of one assistant turn into the pieces both
    response paths persist and return.

    The JSON and SSE endpoints consume the SAME event stream and previously built
    the SAME state from it with two byte-identical if/elif ladders — so a new
    event type had to be handled twice, and ``insights`` in fact only ever got
    handled in one of them. The ladder now lives here once (DRY).

    Only the ACCUMULATION is shared, never the control flow: the SSE path must
    forward each event as it arrives and stop at ``error``, while the JSON path
    buffers everything and answers 502 — those differences stay in the views.
    ``error`` is deliberately not accumulated here for the same reason: an
    errored turn is never persisted, so there is no state to fold.

    Last-write-wins for the scalar fields (``thinking``, ``usage``,
    ``explainability``, ``insights``) — a turn emits each at most once, except
    ``thinking``, where the LAST progress message is the one shown as the turn's
    final "what it did" line.
    """

    #: Event kinds that append to the ordered response[] array (§history).
    CONTENT_KINDS = ("content", "visualization")

    def __init__(self) -> None:
        self.content_blocks: list = []
        self.explainability: dict | None = None
        self.thinking_text: str = ""
        self.summary_text: str = ""
        self.usage: dict = {}
        self.insights: dict | None = None

    def consume(self, kind: str, payload: dict) -> None:
        """Fold one ``{event, data}`` pair into the accumulated turn state.

        Unknown event kinds are ignored on purpose: a newly-added upstream event
        must never break an existing turn, it simply isn't accumulated until this
        module learns about it.
        """
        if kind == "thinking":
            self.thinking_text = payload.get("message", "")
        elif kind in self.CONTENT_KINDS:
            self.content_blocks.append(payload)
            if payload.get("is_summary"):
                self.summary_text = payload.get("content", "")
        elif kind == "explainability":
            self.explainability = payload
        elif kind == "usage":
            self.usage = payload
        elif kind == "insights":
            self.insights = payload

    def metadata(self) -> dict:
        """The persisted/returned ``metadata`` block for this turn."""
        return {"thinking": self.thinking_text, "explainability": self.explainability,
                "usage": self.usage}
