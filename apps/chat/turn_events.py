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
        #: The normalized four-step model, as it stood at the LAST frame that
        #: carried one — i.e. the terminal frame. Only the terminal frame is worth
        #: keeping: the intermediate ones are the same model part-way through.
        self.steps: dict | None = None
        self.summary_text: str = ""
        self.usage: dict = {}
        self.insights: dict | None = None
        #: What the conversation layer carried into this turn — the remembered entity
        #: and filters, the operation applied, and the text actually sent to the engine
        #: (chatbot/nodes.py::_context_used). Absent on a turn that used no context,
        #: which is every first question and every route that never reaches the engine.
        self.context: dict | None = None
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
            # Last-write-wins, but ONLY over a frame that actually carries the
            # model. A trailing legacy-only frame (phase + message, no `steps`)
            # must not wipe the terminal one — and a turn that bypassed the engine
            # carries no model at all, which is correctly stored as absent.
            _st = payload.get("steps")
            if isinstance(_st, dict) and _st.get("steps"):
                self.steps = _st
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
        elif kind == "context":
            self.context = payload

    def metadata(self, *, action: str | None = None) -> dict:
        """The persisted/returned ``metadata`` block for this turn.

        ``trace_id`` is read back out of the explainability payload
        (build_explain's ``support.trace_id``) rather than threaded separately —
        one source of truth, and it is simply absent on a turn that produced no
        explain block. It is the support reference a user can quote and an
        operator can grep the engine trace by.
        """
        # Small talk is deliberately a terminal conversational reply, not an
        # analytical turn. Keep usage for telemetry, but omit the empty progress
        # and explainability slots so JSON/history clients cannot render phantom
        # panels. `action` comes from the conversation layer, which is the single
        # authoritative source for whether the engine was bypassed.
        md = {"usage": self.usage}
        if action != "smalltalk":
            md.update({"thinking": self.thinking_text,
                       "explainability": self.explainability})
        # The four-step model the user actually watched. Without it, reopening a
        # conversation could not rebuild the progress panel: `thinking` is a single
        # legacy line ("Finalizing the results...") and `timeline` is the RAW
        # backend phase list, which is audit-level content and not what was shown.
        # Absent, not null, for a turn that produced no progress.
        if action != "smalltalk" and self.steps:
            md["steps"] = self.steps
        if action != "smalltalk" and self.timeline:
            md["timeline"] = self.timeline
        # Absent, not null (the envelope convention above): a turn that carried no
        # context has nothing to show, and an empty object would render as a panel
        # claiming an understanding that was never applied.
        if action != "smalltalk" and self.context:
            md["context"] = self.context
        trace_id = (((self.explainability or {}).get("support") or {}).get("trace_id")
                    if action != "smalltalk" else None)
        if trace_id:
            md["trace_id"] = trace_id
        return md
