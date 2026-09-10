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
        #: Ordered, user-safe lifecycle events for this turn (traceability Phase 1).
        #: Accumulated from the SAME `thinking` events the SSE path streams, so the
        #: persisted timeline and what the user watched can never disagree. Only
        #: STRUCTURED thinking events (those carrying a `phase` + `status`) are
        #: collected — a legacy free-text progress message is streamed but not
        #: folded in, since it has no phase to file it under.
        self.timeline: list = []

    def consume(self, kind: str, payload: dict) -> None:
        """Fold one ``{event, data}`` pair into the accumulated turn state.

        Unknown event kinds are ignored on purpose: a newly-added upstream event
        must never break an existing turn, it simply isn't accumulated until this
        module learns about it.
        """
        if kind == "thinking":
            self.thinking_text = payload.get("message", "")
            if payload.get("phase") and payload.get("status"):
                self.timeline.append({
                    "phase": payload.get("phase"),
                    "status": payload.get("status"),
                    "title": payload.get("title", ""),
                    "message": payload.get("message", ""),
                })
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
        """The persisted/returned ``metadata`` block for this turn.

        ``trace_id`` is read back out of the explainability payload
        (build_explain's ``support.trace_id``) rather than threaded separately —
        one source of truth, and it is simply absent on a turn that produced no
        explain block. It is the support reference a user can quote and an
        operator can grep the engine trace by.
        """
        md = {"thinking": self.thinking_text, "explainability": self.explainability,
              "usage": self.usage}
        if self.timeline:
            md["timeline"] = self.timeline
        trace_id = ((self.explainability or {}).get("support") or {}).get("trace_id")
        if trace_id:
            md["trace_id"] = trace_id
        return md
