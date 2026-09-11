from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Iterator

from chatbot.run import run_chat_turn

from apps.core.messages import MESSAGES

from .models import ChatMessage, ChatSession, MessageType
from .table_rendering import (
    project_display_columns as _project_display_columns,
    rows_to_markdown_table as _rows_to_markdown_table,
)
from .thinking_messages import business_friendly_message
from .thinking_context import ThinkingContext
from . import thinking_steps as ts_mod
from .thinking_steps import ThinkingStepTracker
from .visualization import VisualizationRecommender

logger = logging.getLogger(__name__)

DEFAULT_CONVERSATION_TITLE = "New Chat"

# User-facing error copy for the `error` SSE event / 502 body. Raw exception text
# and tracebacks are NEVER sent to the client (they leak internals like connection
# strings / import errors and read as gibberish to a user) — the detail is logged
# server-side via logger.exception, and one of these safe messages is shown instead.
# Two DISTINCT codes so the frontend can react differently:
#   • LLM_UNAVAILABLE — the inference/LLM tier is unreachable or down. Transient:
#     the UI should say the assistant is temporarily unavailable and offer a retry.
#   • MODEL_ERROR — an unexpected fault while generating the answer (not a known
#     outage). Also retryable from the user's side, but not a clean "service down".
#
# Public (no leading underscore): views.py renders the same copy on its own
# stream-level failure path, so this is a deliberate cross-module contract rather
# than a private detail being reached into.
CODE_LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
MSG_LLM_UNAVAILABLE = MESSAGES["chat"]["llm_unavailable"]
CODE_MODEL_ERROR = "MODEL_ERROR"
MSG_MODEL_ERROR = MESSAGES["chat"]["model_error"]
# Error code for a fault raised while the SSE response is already streaming
# (views.py) — distinct from MODEL_ERROR so the two are separable in the client
# and in logs, even though they share the same user-facing copy.
CODE_STREAM_ERROR = "STREAM_ERROR"

# Bounds how long run_turn waits on the worker thread after it has signalled
# completion. It has already finished by then (the "done" sentinel is enqueued in
# the worker's `finally`), so this only caps a pathological worst case.
_WORKER_JOIN_TIMEOUT_S = 5

# Sentinel the verified-query cache reports instead of a real table name. Imported
# from apps.query.audit — the shared definition both front doors read — rather than
# re-typed here, so the two can never disagree.
from apps.query.audit import CACHED_TABLE_SENTINEL  # noqa: E402

def _thinking_steps_enabled() -> bool:
    """Whether to fold the internal phase stream into the four user-facing steps.

    Default OFF: with it off, every `thinking` event is byte-identical to before, so
    an existing client is untouched. With it on the SAME events are emitted, each
    additionally carrying a `steps` block — the legacy `phase`/`message` fields never
    change, so old and new clients can both read the same stream.
    """
    from django.conf import settings
    return bool(getattr(settings, "VEDA_THINKING_STEPS", False))


_visualization_recommender = VisualizationRecommender()

# Fixed-shape fallback when the engine has no "explain" for this turn (smalltalk,
# clarify, refusal — no SQL ran) — lets the frontend render one schema unconditionally
# instead of null-checking every field.
_NO_EXPLAIN = {
    "version": "1.0",
    "understanding": {"summary": None},
    "data_used": {"datasets": [], "fields": []},
    "operations": [],
    "filters": {"applied": [], "summary": "No filters applied."},
    "validation": {"passed": None, "checks": []},
    # `enabled: False`, not True. This fallback ships precisely when NO SQL ran (the
    # comment above says so), so advertising the SQL block as enabled was wrong on its
    # own terms. This stays False regardless of EXPLAIN_EXPOSE_SQL (default ON again
    # since 2026-09-11): that flag decides whether SQL that EXISTS may be shown, and
    # here none exists to show.
    # Observed on a document answer: the payload said `enabled: true, query: null`,
    # which reads as "SQL is available and we are withholding it" rather than "this
    # question was not answered with SQL at all".
    "sql": {"enabled": False, "query": None},
}


class ChatNotFound(Exception):
    """Raised when chat_id is provided but no matching, owned ChatSession exists."""


def _positional_rows(cols: list, rows: list) -> list:
    # The plain SQL path returns positional rows (list/tuple aligned with cols by
    # index) but the federated executor returns column-keyed dicts
    # (dict(zip(cols, r))) — normalize to positional so the table/chart builders
    # can index by column position.
    return [
        [row.get(c) for c in cols] if isinstance(row, dict) else row
        for row in rows
    ]


def _spec_from_suggestion(cols: list, rows: list, suggestion: dict | None):
    """Turn the query tier's validated {type,x_axis,y_axis,...} column-name
    suggestion into a real VisualizationSpec, reusing the existing
    recommender's own chart-data builders (never re-implemented here) — the
    suggestion only names WHICH columns to chart, the recommender still owns
    HOW the chart_data is built."""
    if not suggestion or not isinstance(suggestion, dict):
        return None
    vtype = suggestion.get("type")
    x_name, y_name = suggestion.get("x_axis"), suggestion.get("y_axis")
    if x_name not in cols or y_name not in cols:
        return None
    x_idx, y_idx = cols.index(x_name), cols.index(y_name)
    if vtype == "line":
        return _visualization_recommender.build_line_spec(cols, rows, x_idx, y_idx)
    if vtype in ("bar", "pie"):
        # build_category_specs returns a LIST (it may offer pie + bar for the same data).
        # This function's contract is ONE spec (the callers do `spec.to_dict()`), so
        # return the spec matching the requested type — or the first available — never
        # the raw list (a list has no .to_dict(), which crashed _build_visualizations
        # on any bar/pie candidate/suggestion that reached this fallback). None when the
        # data can't be charted (e.g. a single category).
        specs = _visualization_recommender.build_category_specs(cols, rows, x_idx, y_idx)
        if not specs:
            return None
        return next((s for s in specs if s.type.value == vtype), specs[0])
    return None


class ConversationQueryService:
    """One assistant turn: resolve chat -> run the chatbot supervisor -> persist."""

    def __init__(self, user, source_id: int | None = None, tenant: str = "default",
                 source_ids: list[int] | None = None, data_scope: dict | None = None,
                 source_profiles: dict | None = None,
                 access_denied: bool = False,
                 access_check_ms: float | None = None):
        self.user = user
        self.source_id = source_id
        # Validated query SCOPE (P5) — ready source ids, primary first, resolved
        # server-side by the view (apps.query.scope.resolve_query_scope). Forwarded to
        # inference so multi-source scopes retrieve/federate exactly like /api/v1/query.
        self.source_ids = list(source_ids) if source_ids else ([source_id] if source_id else None)
        self.tenant = tenant
        # Gate 1 (User Story 3, Task 15): the view's precomputed RBAC data scope
        # (apps.access_management.services.serialize_data_scope), forwarded verbatim
        # to the chatbot turn. None = no restriction, same as every existing caller.
        self.data_scope = data_scope
        # Per-source routing metadata (apps.query.scope.source_profiles_for), forwarded exactly
        # as /api/v1/query does. It is what tells the engine a source is datalake/document rather
        # than relational; without it veda_hybrid._is_datalake_source is False, the
        # datalake-isolated semantic model is never loaded, and the SQL planner receives the
        # PRIMARY source's 178-table homzhub schema — which does not contain the datalake tables
        # at all, so a datalake question could only choose among irrelevant tables and ended in an
        # "ambiguous subject" clarify. /api/v1/query answered the same question correctly, which is
        # why this only ever reproduced through chat. None = same behaviour as before.
        self.source_profiles = source_profiles
        # Set when the view already knows (permitted_source_ids returned empty)
        # that this caller has zero granted sources — run_turn then skips the
        # engine entirely and yields a synthetic denial turn instead. Previously
        # this case short-circuited as a raw HTTP 403 before a chat/message row
        # ever existed: no history, and a shape the frontend had to special-case
        # separately from every other kind of refusal. Routing it through the
        # SAME run_turn/persist path means it costs no engine compute (the one
        # thing the 403 was protecting) while looking, streaming, and saving
        # exactly like any other turn — user's call.
        self.access_denied = access_denied
        #: MEASURED duration of the RBAC resolution the view performed before this
        #: service was constructed (traceability: real timing, never inferred).
        #: Surfaced as the "Checking access permissions" sub-check's duration.
        self.access_check_ms = access_check_ms
        #: Set by _build_reply_events once the turn has produced its events.
        #: {} until then, so a view that reads it after an errored turn gets an
        #: empty dict rather than an AttributeError.
        self.last_audit: dict = {}

    def create_conversation(self, title: str = "") -> ChatSession:
        name = (title or "").strip() or DEFAULT_CONVERSATION_TITLE
        return ChatSession.objects.create(user=self.user, name=name[:255])

    def resolve_chat(self, chat_id: int | None, name_hint: str = "") -> ChatSession:
        if chat_id is not None:
            chat = ChatSession.objects.filter(
                pk=chat_id, user=self.user, is_deleted=False
            ).first()
            if chat is None:
                raise ChatNotFound(f"chat_id={chat_id} not found")
            return chat
        return self.create_conversation(name_hint)

    def list_conversations(self):
        """Owned, non-deleted conversations, most-recently-updated first."""
        return (
            ChatSession.objects.filter(user=self.user, is_deleted=False)
            .only("id", "name", "created_at", "updated_at")
            .order_by("-updated_at")
        )

    def get_conversation_history(self, chat_id: int):
        """Ownership-checked chat plus its messages in chronological order."""
        chat = self.resolve_chat(chat_id)
        messages = chat.messages.filter(is_deleted=False).order_by("created_at")
        return chat, messages

    def save_user_message(self, chat: ChatSession, message: str) -> ChatMessage:
        return ChatMessage.objects.create(
            session=chat, type=MessageType.USER, content=message,
        )

    def save_assistant_message(
        self, chat: ChatSession, content_blocks: list, metadata: dict
    ) -> ChatMessage:
        # Stored verbatim as JSON so history can return it exactly as persisted,
        # without regenerating or lossily flattening the structured blocks.
        return ChatMessage.objects.create(
            session=chat, type=MessageType.ASSISTANT,
            content=json.dumps(content_blocks), metadata=metadata,
        )

    def run_turn(
        self, chat: ChatSession, message: str, request_id: str = "", stream: bool = False,
    ) -> Iterator[dict]:
        """Yields: thinking* (stream only, zero for an instant fast-path answer)
        -> content* -> visualization? -> explainability -> error?.

        `chat.pk` doubles as the chatbot/LangGraph session_id (thread_id) — the
        graph's own Redis checkpointer accumulates conversation history per
        session automatically, so no history is threaded through manually here.

        stream=True sources "thinking" from chatbot's own on_event callback,
        which itself forwards the inference tier's real SSE progress events
        (classify / decompose / route / answer) live as the pipeline advances,
        via a background thread bridged through a queue (see _run_streamed).
        No placeholder "thinking" event fires up front — an instant fast-path
        answer (smalltalk, runtime context) never emits one at all, and a real
        question's first genuine thinking event (classify_node's "Understanding
        your message...") arrives moments later on its own."""
        if self.access_denied:
            # No thinking/engine events at all — this never reaches the pipeline,
            # so there is nothing to progress-report. Same event contract
            # (content -> explainability -> usage) as every other terminal turn.
            yield from self._access_denied_events()
            return
        session_id = str(chat.pk)
        kwargs = dict(tenant=self.tenant, source_id=self.source_id,
                      source_ids=self.source_ids, request_id=request_id,
                      data_scope=self.data_scope,
                      source_profiles=self.source_profiles)
        # End-to-end wall clock for THIS turn — so latency_ms is ALWAYS reportable,
        # even when the engine result carries none (a refusal/clarify that never
        # reached _done(), or a path that returned no latency). Used as the fallback
        # in _build_reply_events' usage event.
        _turn_t0 = time.monotonic()

        if stream:
            response = yield from self._run_streamed(message, session_id, kwargs, chat)
            if response is None:
                return   # _run_streamed already yielded the error event
        else:
            try:
                response = run_chat_turn(message, session_id, **kwargs)
            except Exception:
                # Raw exception logged (with traceback) — NOT sent to the client.
                logger.exception("conversation query pipeline failed chat_id=%s", chat.pk)
                yield {"event": "error",
                       "data": {"code": CODE_MODEL_ERROR, "message": MSG_MODEL_ERROR}}
                return

        if response.get("engine_unavailable"):
            # The inference/LLM tier is down/unreachable (call_engine_node mapped an
            # InferenceUnavailable or a mid-stream error to this flag) — a transient
            # outage, surfaced with its own code so the UI can prompt a retry rather
            # than a generic failure.
            #
            # Deliberately do NOT surface response["reply_text"] here: on the
            # unavailable path chatbot/nodes.py sets it to the clarify fallback
            # ("Could you clarify what you're asking about?"), which would MISLEAD
            # the user into thinking their question was bad when in fact the AI
            # service is down. Always show the outage copy.
            logger.warning("conversation query pipeline unavailable chat_id=%s", chat.pk)
            # Resolve the four steps BEFORE the error. This path returns without
            # reaching _build_reply_events, which is the only other place the
            # terminal step frame is emitted — so the steps were left exactly as the
            # last progress frame had them. Observed in a real stream: the turn died
            # on LLM_UNAVAILABLE with "Analyzing the information" still `active`, so
            # the UI kept spinning on a step that would never finish, next to an
            # error telling the user the assistant was down.
            yield from self._terminal_step_frame(
                failed=True, error_code=CODE_LLM_UNAVAILABLE, retryable=True)
            yield {"event": "error",
                   "data": {"code": CODE_LLM_UNAVAILABLE,
                            "message": MSG_LLM_UNAVAILABLE}}
            return

        if isinstance(response, dict):
            response["_turn_latency_ms"] = round((time.monotonic() - _turn_t0) * 1000, 2)
        yield from self._build_reply_events(response)

    def _run_streamed(self, message: str, session_id: str, kwargs: dict, chat: ChatSession):
        """Bridges run_chat_turn's synchronous on_event callback (fired from
        inside a blocking graph.invoke() call) into this generator's yield
        contract, via a background thread + thread-safe queue — there's no
        asyncio available here (unlike inference/routes/hybrid.py's SSE route),
        so a plain thread+queue is the equivalent for a sync Django view.

        Ordering is guaranteed: on_event(...) calls happen strictly BEFORE
        run_chat_turn returns (invoked synchronously, nested inside
        call_engine_node's iteration of the inference SSE stream), so every
        "thinking" item is enqueued before the terminal "result"/"error" item.

        Returns the final response dict via a StopIteration value (consumed
        by `response = yield from self._run_streamed(...)` in run_turn), or
        None if an error event was already yielded here.
        """
        q: "queue.Queue[tuple[str, object]]" = queue.Queue()
        # Four-step progress model for THIS turn. Lives here, in the api tier,
        # because folding internal phases into user-facing steps is a display
        # concern and this is already the boundary where internal phase names stop
        # (see thinking_messages.py). The pipeline is not touched.
        steps = ThinkingStepTracker() if _thinking_steps_enabled() else None
        step_ctx = ThinkingContext() if steps is not None else None
        self._steps, self._step_ctx = steps, step_ctx
        if steps is not None and self.access_check_ms is not None:
            from .thinking_steps import STEP_FINDING
            steps.set_sub_check_duration(STEP_FINDING, "access", self.access_check_ms)

        def on_event(phase: str, evt_message: str, extra: dict | None = None) -> None:
            # `extra` carries the inference tier's per-phase structured fields
            # (route's intent=, sub_query's index=/total=/sub_query=, ...) —
            # merged in so the SSE "thinking" payload isn't just flattened
            # phase/message text. phase/message win on key collision (unlikely,
            # but they're the guaranteed-present fields).
            # `phase` itself is forwarded verbatim (never renamed — logs/tracing
            # upstream are unaffected); only the displayed `message` is swapped
            # for a business-friendly one (thinking_messages.py, UX Phase 1).
            payload = {**(extra or {}), "phase": phase,
                       "message": business_friendly_message(phase, evt_message)}
            # `narration` is the async SLM narrator's delivery (veda/narrator.py).
            # It is NOT a pipeline phase: it carries no progress of its own, it only
            # replaces the contextual sentence of a step. Swallowed here rather than
            # forwarded, so no client ever sees a phase that did not happen.
            if phase == "narration":
                if steps is not None:
                    steps.set_context((extra or {}).get("step") or "analyzing",
                                      evt_message, from_narrator=True)
                return
            if steps is not None:
                step_ctx.absorb(payload)
                if steps.consume(payload):
                    for _k in steps.steps:
                        steps.set_context(_k, step_ctx.sentence(_k))
                        steps.set_details(_k, step_ctx.details(_k))
                    if steps.has_progress():
                        payload["steps"] = steps.as_payload()
                # The LEGACY `message` line, kept for clients that predate the step
                # model. Most phases have no user-facing copy of their own, so it
                # went out EMPTY on 9 of the 12 frames a normal turn emits — a client
                # rendering it showed a status line that appeared, blanked, and
                # reappeared several times per turn. It now falls back to the running
                # step's own summary: the same sentence the step model already shows,
                # so the two can no longer disagree, and nothing new is claimed.
                if not payload.get("message"):
                    _cur = steps.current_step()
                    if _cur:
                        _st = steps.steps.get(_cur)
                        _sum = getattr(_st, "summary", None) if _st else None
                        if _sum:
                            payload["message"] = _sum
            q.put(("thinking", payload))

        def target() -> None:
            try:
                result = run_chat_turn(message, session_id, on_event=on_event, **kwargs)
                q.put(("result", result))
            except Exception as exc:
                logger.exception("conversation query pipeline failed (thread) chat_id=%s", chat.pk)
                q.put(("error", str(exc)))
            finally:
                # ALWAYS enqueued, even on exception above — this is what
                # prevents the consumer loop below from blocking forever.
                q.put(("done", None))

        thread = threading.Thread(target=target, daemon=True, name=f"chatbot-turn-{chat.pk}")
        thread.start()

        result, error_message = None, None
        while True:
            kind, payload = q.get()
            if kind == "done":
                break
            elif kind == "result":
                result = payload
            elif kind == "error":
                error_message = payload
            else:
                yield {"event": kind, "data": payload}
        thread.join(timeout=_WORKER_JOIN_TIMEOUT_S)   # already finished by the time "done" was enqueued

        if error_message is not None:
            # error_message is the raw str(exc) from the worker thread — already
            # logged with its traceback in target()'s except. Show the safe copy.
            yield {"event": "error",
                   "data": {"code": CODE_MODEL_ERROR, "message": MSG_MODEL_ERROR}}
            return None
        if result is None:
            logger.error("conversation query pipeline returned no result chat_id=%s", chat.pk)
            yield {"event": "error",
                   "data": {"code": CODE_MODEL_ERROR, "message": MSG_MODEL_ERROR}}
            return None
        return result

    def _access_denied_events(self):
        """The synthetic terminal turn for ``self.access_denied`` — same wording
        as veda_core's own access-denied refusal (``veda.feedback.
        ACCESS_DENIED_WHY``/``_WHAT``) for consistency with a partial-access
        denial (some sources granted, the specific one asked about isn't) that
        reaches that path INSIDE the engine. Duplicated here rather than
        imported: the api tier never imports veda_core directly (see
        InferenceClient's own docstring) — this is the source-level denial,
        decided entirely from RBAC grants before the engine is ever called, so
        there is no shared module to import from without crossing that
        boundary. No suggestions, same reasoning as the engine-side branch:
        naming other resources here would itself be a leak."""
        why = MESSAGES["chat"]["access_denied_why"]
        what = MESSAGES["chat"]["access_denied_what"]
        yield {"event": "content",
              "data": {"type": "markdown", "content": f"{why} {what}", "is_summary": True}}
        yield {"event": "explainability", "data": {
            "version": "1.0",
            "understanding": {"summary": why},
            "why": why,
            "what_would_help": what,
            "suggestions": [],
        }}
        yield {"event": "usage", "data": {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "latency_ms": 0}}

    def _terminal_step_frame(self, *, failed: bool, error_code: str | None = None,
                             retryable: bool | None = None):
        """Freeze and emit the four steps once, at the real end of the turn.

        Shared by the normal reply path and the outage path so the two cannot drift:
        every way out of a turn must leave the steps resolved. A no-op when the
        feature is off or the frame was already emitted.
        """
        _steps = getattr(self, "_steps", None)
        if _steps is None or _steps.finished:
            return
        _ctx = getattr(self, "_step_ctx", None)
        if _ctx is not None:
            if failed:
                _ctx.no_answer = True
            for _k in _steps.steps:
                _steps.set_context(_k, _ctx.sentence(_k))
                _steps.set_details(_k, _ctx.details(_k), terminal=True)
        # `answered_without_result` = the turn produced no answer (refusal/clarify).
        # finish() uses it to decide whether a validation FAILURE still describes the
        # answer: on a delivered answer it does not, because the failing attempt was
        # superseded; on a refusal it is the whole story.
        _no_answer = bool(getattr(_ctx, "no_answer", False)) if _ctx is not None else False
        _steps.finish(failed=failed, error_code=error_code, retryable=retryable,
                      answered_without_result=_no_answer)
        _frame = {"phase": "completed",
                  "status": "failed" if failed else "completed",
                  "message": business_friendly_message("output", "Done")}
        # A turn that bypassed the engine (a canned greeting, answered in ~100 ms)
        # has no progress to report. Emitting the four steps anyway put four empty
        # circles above a finished answer under a `completed` status — nothing had
        # completed. The legacy phase/message still goes out, so a client that only
        # reads those is unaffected.
        if _steps.has_progress():
            _frame["steps"] = _steps.as_payload()
        elif not failed:
            # NOTHING HAPPENED, so say nothing. A canned greeting never reaches the
            # engine, and this still shipped `phase: completed, message: "Finalizing
            # the results..."` — there were no results to finalize. The four steps
            # were already suppressed here for the same reason; the legacy line was
            # left behind and kept narrating work that did not occur.
            #
            # A FAILED turn still emits: the frame is how the error code and the
            # failed step reach the client, and the outage path has no progress
            # either.
            return
        yield {"event": "thinking", "data": _frame}

    def _build_reply_events(self, response: dict):
        res0 = response.get("engine_result") or {}
        # TERMINAL step frame: freeze every open step at the real end of the turn and
        # emit the completed four-step model once, as the answer lands. Emitted BEFORE
        # the content so a client can collapse the live progress section at the same
        # moment it renders the answer. Absorbs the final explainability payload first
        # so the "Preparing" step can say what was actually produced (chart / table /
        # summary) rather than guessing from the live stream alone.
        _vizzes = None
        _steps = getattr(self, "_steps", None)
        if _steps is not None and not _steps.finished:
            _ctx = getattr(self, "_step_ctx", None)
            # Was this an error, or a turn that ran to completion and explained
            # why it cannot answer? An ALLOW-LIST of engine statuses answered that
            # question before, and it named `clarify` but not `qualifier_dropped` —
            # so identical clarifications reported different terminal statuses. The
            # decision now rests on what the turn PRODUCED. See terminal_outcome().
            _explain0 = res0.get("explain") or {}
            _has_refusal = bool(_explain0.get("why")
                                or _explain0.get("what_would_help")
                                or _explain0.get("suggestions"))
            _failed, _err_code = ts_mod.terminal_outcome(
                ok=bool(res0.get("ok")), engine_status=res0.get("status"),
                has_refusal_explanation=_has_refusal,
                access_denied=_steps.access_denied())
            if _ctx is not None:
                _ctx.absorb_explain(res0.get("explain") or {})
                # The turn's OUTCOME, known only here. Without it the Preparing step
                # said "Putting your answer together." on a refusal — this sentence
                # overwrites the phase's own honest message at the terminal frame,
                # so the honest text never reached the user.
                # `_no_results` is the engine's own decision, made once at its front
                # door (veda_hybrid._mark_empty_results) from the rows/passages the
                # turn actually produced. Reading it here rather than re-deriving is
                # the point: `ok`/`status` describe whether the pipeline RAN, and
                # reading them as "did it find anything" is what put four green ticks
                # and "Checks passed" above a reply reading "No results found."
                _ctx.found_nothing = bool(res0.get("_no_results"))
                _ctx.no_answer = bool(res0.get("_no_results")) or (
                    res0.get("status") in ("refused", "clarify")) or (
                    not res0.get("ok") and res0.get("status") not in ("answered", None))
                if res0.get("_from_cache"):
                    _ctx.from_cache = True
                # Counted evidence, from what the turn ACTUALLY returned. Never a
                # score: "20 rows" is a fact the reader can weigh, a confidence
                # percentage the backend never defined is not (§7).
                _rows0 = res0.get("rows")
                if isinstance(_rows0, list):
                    _steps.set_evidence(rows=len(_rows0))
                if _ctx.passages is not None:
                    _steps.set_evidence(passages=_ctx.passages)
                if _ctx.source_names:
                    _steps.set_evidence(sources=len(_ctx.source_names))
                elif _ctx.source_count:
                    _steps.set_evidence(sources=_ctx.source_count)
                if _steps.execution_type != ts_mod.EXEC_UNKNOWN:
                    _ctx.execution_type = _steps.execution_type
                elif _ctx.execution_type != ts_mod.EXEC_UNKNOWN:
                    _steps.execution_type = _ctx.execution_type
                else:
                    # Still unknown: the shape is normally read from an event's
                    # `intent`, and the CROSS-SOURCE lane emits none — a federated
                    # answer therefore shipped `execution: {"type": "unknown"}` even
                    # though the payload named the sources it combined (observed on
                    # the csv_lake/parquet questions). Derive it from what the turn
                    # DEMONSTRABLY produced instead of leaving it blank. Every branch
                    # rests on a fact already established elsewhere in this payload;
                    # none of them guesses, and no branch fires without one.
                    _shape = ts_mod.execution_shape_from_evidence(
                        source_count=_ctx.source_count,
                        source_names=_ctx.source_names,
                        passages=_ctx.passages,
                        has_rows=isinstance(_rows0, list))
                    if _shape:
                        _steps.execution_type = _shape
                        _ctx.execution_type = _shape
                # `multi_source` is a claim about how many sources contributed, so
                # it is checked against how many did. The hybrid head reports
                # `hybrid` for "database first, then documents" — one source, two
                # attempts — and that was being read as several sources.
                _fixed = ts_mod.correct_multi_source_claim(
                    _steps.execution_type,
                    source_count=_ctx.source_count,
                    source_names=_ctx.source_names,
                    passages=_ctx.passages,
                    has_rows=bool(_rows0) if isinstance(_rows0, list) else False)
                if _fixed != _steps.execution_type:
                    _steps.execution_type = _fixed
                    _ctx.execution_type = _fixed
                # Context and details are applied BEFORE finish(), not after.
                # finish() decides what to do with a step that never started, and
                # that decision depends on whether the step has content: content
                # means the work happened and was simply never reported as a phase
                # (the document/RAG head emits nothing mapping to "Analyzing"),
                # whereas no content means it genuinely did not run. Setting details
                # afterwards hid that distinction and left the step `pending`
                # between two completed ones.
                # Computed BEFORE the terminal frame, deliberately. It used to run
                # after, and announced itself with its own `thinking` event — which
                # arrived AFTER the frame that had already said `status: completed`,
                # telling the client the turn was over and then sending it more
                # progress (measured on every charted turn). The fact is real, so it
                # belongs IN the model rather than after it.
                _vizzes = self._build_visualizations(res0)
                if _vizzes:  # noqa: SIM102 — the chart fact belongs in the model
                    _ctx.output = ("chart+summary" if _ctx.output == "summary"
                                   else "chart")
                for _k in _steps.steps:
                    _steps.set_context(_k, _ctx.sentence(_k))
                    _steps.set_details(_k, _ctx.details(_k), terminal=True)
            # Emit through the SHARED terminal emitter, not a second copy of it.
            # There are two ways out of a turn — this one and the outage path — and
            # keeping two copies is how the no-progress guard came to be applied to
            # only one of them: a canned greeting still shipped four empty circles.
            # `_terminal_step_frame` re-applies context/details itself, so the loop
            # above is now only about the evidence the api tier contributes.
            yield from self._terminal_step_frame(
                failed=_failed, error_code=_err_code,
                retryable=(False if _err_code == ts_mod.ERROR_ACCESS_DENIED else None))
        # Audit facts for this turn (traceability Part 19), stashed on the
        # per-request service instance rather than emitted as an event: they are
        # for the QueryLog row only and must never cross the wire. The view reads
        # `service.last_audit` after draining run_turn. Populated from what the
        # turn ALREADY produced — nothing re-derived.
        self.last_audit = {
            "route": res0.get("_route") or "",
            "status": res0.get("status") or ("answered" if res0.get("ok") else ""),
            # `_from_cache` is set by the engine AT the lane. The sentinel
            # comparison is kept as a fallback for callers that still emit it,
            # but it alone recorded False on every hit since the sentinel was
            # removed from the engine (last true row: 2026-07-09).
            "cache_hit": bool(res0.get("_from_cache")) or (
                res0.get("table") == CACHED_TABLE_SENTINEL),
            "latency_ms": response.get("_turn_latency_ms"),
            "usage": res0.get("usage") or {},
            "explain": res0.get("explain"),
        }
        # Computed (fast, synchronous, no LLM — same call as before) BEFORE any
        # content streams, so the thinking message below completes the
        # "thinking" sequence rather than interleaving mid-answer.
        # Reused, never recomputed: the terminal block above already built these in
        # order to report the chart as part of the step model. It is None only when
        # that block did not run (no tracker, or an already-finished one).
        vizzes = _vizzes if _vizzes is not None else self._build_visualizations(res0)
        # NO `thinking` event here. The turn has already reported its one terminal
        # state, and a progress frame after that contradicts it.
        for block in self._build_content_blocks(response, res0):
            yield {"event": "content", "data": block}
        if vizzes:
            # ONE event carrying every recommended chart (2026-07, multi-viz):
            # was previously one "visualization" event PER spec. This IS a
            # wire-contract change (unlike the earlier multi-spec support in
            # VisualizationRecommender itself, which only changed cardinality
            # of an already-list-shaped return) — any existing frontend
            # reading `data.type`/`data.chart_data` directly off a
            # "visualization" event must move to `data.visualizations[i].type`
            # etc. instead. Order is preserved — vizzes[0] is still today's
            # single-chart choice (see visualization.py's own docstring).
            yield {"event": "visualization", "data": {"visualizations": vizzes}}
        # veda_core (veda/business_explain.py) builds this deterministically from the
        # final validated SQL + semantic model — never from retrieval/routing internals
        # (those live only in res0["trace"], for our own debugging, never sent over SSE).
        # Confidence lives INSIDE explain (build_explain()'s "confidence" key) —
        # one canonical place, not duplicated as a second top-level field. It's a
        # deterministic weakest-link value from anchor/join gating signals
        # (veda/pipeline.py's _done(), query/result_explainer.py's
        # synthesize_confidence) — never an LLM self-report — always present for
        # an answered Tier-1 query, regardless of INSIGHT_ENGINE_ENABLED.
        # `_NO_EXPLAIN` is the fixed-shape fallback for a turn the engine answered
        # without building a payload. It is NOT for a turn the engine never saw: a
        # canned greeting shipped every block empty — `sql.enabled: false`,
        # `validation.passed: null`, "No filters applied." — which a client renders
        # as a "how this answer was generated" panel explaining nothing. There is
        # no explanation because there was no query.
        _explain0 = res0.get("explain")
        _bypassed = not (getattr(self, "_steps", None)
                         and self._steps.has_progress()) and not _explain0
        if not _bypassed:
            yield {"event": "explainability", "data": _explain0 or _NO_EXPLAIN}
        # Token usage (veda_core/slm/_call_slm.py's usage accumulator, surfaced
        # via veda/pipeline.py's _done() / veda_hybrid.py's Tier-2 dispatch).
        # Always a 3-key dict — {0,0,0} for deterministic fast paths that never
        # call an SLM — so the UI never special-cases absence. No cost figure:
        # self-hosted SLMs have no real per-token billing.
        #
        # latency_ms is the TOTAL end-to-end response time for this turn (engine +
        # the chatbot supervisor graph + serialization/streaming overhead) — the
        # turn wall clock measured from run_turn's _turn_t0. ALWAYS present, on
        # success and on a failed/refused turn alike, so it is never null. (This is
        # server-side turn time, NOT the browser HTTP round-trip — a client wanting
        # true wall-clock still measures its own.) The engine-only/inference slice
        # is intentionally NOT surfaced — the product only needs the total.
        yield {"event": "usage", "data": {
            **(res0.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            "latency_ms": response.get("_turn_latency_ms"),
        }}
        # Insight Engine (additive event type): only present when
        # INSIGHT_ENGINE_ENABLED produced these keys server-side.
        if "insights" in res0 or "follow_up_questions" in res0:
            yield {"event": "insights", "data": {
                "insights": res0.get("insights") or [],
                "follow_up_questions": res0.get("follow_up_questions") or [],
            }}

    @staticmethod
    def _build_content_blocks(response: dict, res0: dict) -> list:
        blocks = []
        reply_text = response.get("reply_text")   # covers answer + smalltalk + clarify uniformly
        if reply_text:
            # is_summary marks this as the primary answer (vs. supporting content like
            # the table below) so callers can surface it distinctly without a second
            # LLM call or re-deriving which block "is" the summary.
            summary = str(reply_text)
            # Fold the Insight-Engine observations into the SAME summary block
            # instead of a separate "insights" event (2026-07-17): one block, not
            # two. res0["insights"] is List[str] (0-3 factual observations); only
            # present on answered turns with INSIGHT_ENGINE_ENABLED — never on
            # smalltalk/clarify, so this is a no-op there. The deterministic
            # "Analysis:" patterns already live inside reply_text (pipeline.py).
            insights = [str(i).strip() for i in (res0.get("insights") or []) if str(i).strip()]
            if insights:
                summary = summary.rstrip() + "\n\n" + "\n".join(f"- {i}" for i in insights)
            blocks.append({"type": "markdown", "content": summary, "is_summary": True})
        cols, rows = res0.get("cols"), res0.get("rows")
        if cols and rows:
            rows = _positional_rows(cols, rows)
            # Drop non-business columns (e.g. join-only ids) from the rendered
            # table — the engine's own display_columns already excludes
            # identifier-role columns (see project_display_columns's docstring).
            # Fails safe to the original cols/rows when analytics is absent.
            display_cols = (res0.get("analytics") or {}).get("display_columns")
            table_cols, table_rows = _project_display_columns(cols, rows, display_cols)
            blocks.append({"type": "markdown",
                           "content": _rows_to_markdown_table(table_cols, table_rows)})
        if not blocks:
            blocks.append({"type": "markdown", "content": "No response could be generated."})
        return blocks

    @staticmethod
    def _build_visualizations(res0: dict) -> list:
        cols, rows = res0.get("cols"), res0.get("rows")
        if not cols or not rows:
            return []
        rows = _positional_rows(cols, rows)
        # res0["analytics"]: the engine's one deterministic post-execution
        # analysis (result_analyzer.analytics_summary) — column kinds/roles
        # computed once server-side, preferred over this tier's own structural
        # heuristics (which remain the fallback, e.g. for federated results).
        specs = _visualization_recommender.recommend(cols, rows, analytics=res0.get("analytics"))
        if specs:
            return [spec.to_dict() for spec in specs]
        # Deterministic rules found nothing confident — fall back to the query
        # tier's Insight Engine suggestion (already validated server-side:
        # column existence + type compatibility — see
        # query/result_explainer.py's validate_visualization). Still built into
        # the SAME chart_data shape via the existing recommender's own builders,
        # never served as a bare column-name suggestion.
        spec = _spec_from_suggestion(cols, rows, res0.get("visualization"))
        if spec:
            return [spec.to_dict()]
        # Second deterministic fallback (2026-07-17): the engine ALSO computes
        # analytics["chart_candidates"] every turn (result_analyzer.
        # compute_chart_candidates — shape-driven canonical chart + confidence)
        # but until now nothing ever read it — pure wasted computation/wire
        # bytes. Same {type, x_axis, y_axis} shape the suggestion path above
        # already consumes, so no new plumbing: just try its first (highest-
        # confidence) candidate before giving up on a chart entirely.
        candidates = (res0.get("analytics") or {}).get("chart_candidates") or []
        spec = _spec_from_suggestion(cols, rows, candidates[0]) if candidates else None
        return [spec.to_dict()] if spec else []

