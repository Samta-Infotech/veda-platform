from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Iterator

from chatbot.run import run_chat_turn

from .models import ChatMessage, ChatSession, MessageType
from .table_rendering import (
    project_display_columns as _project_display_columns,
    rows_to_markdown_table as _rows_to_markdown_table,
)
from .thinking_messages import business_friendly_message
from .visualization import VisualizationRecommender
from apps.query.inference_client import InferenceClient, InferenceUnavailable

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
_CODE_LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
_MSG_LLM_UNAVAILABLE = ("The AI assistant is temporarily unavailable. "
                        "Please try again in a moment.")
_CODE_MODEL_ERROR = "MODEL_ERROR"
_MSG_MODEL_ERROR = ("Something went wrong while generating a response. "
                    "Please try again.")

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
    "sql": {"enabled": True, "query": None},
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
        return _visualization_recommender._line(cols, rows, x_idx, y_idx)
    if vtype in ("bar", "pie"):
        # _category_numeric returns a LIST (it may offer pie + bar for the same data).
        # This function's contract is ONE spec (the callers do `spec.to_dict()`), so
        # return the spec matching the requested type — or the first available — never
        # the raw list (a list has no .to_dict(), which crashed _build_visualizations
        # on any bar/pie candidate/suggestion that reached this fallback). None when the
        # data can't be charted (e.g. a single category).
        specs = _visualization_recommender._category_numeric(cols, rows, x_idx, y_idx)
        if not specs:
            return None
        return next((s for s in specs if s.type.value == vtype), specs[0])
    return None


class ConversationQueryService:
    """One assistant turn: resolve chat -> run the chatbot supervisor -> persist."""

    def __init__(self, user, source_id=None, tenant: str = "default", source_ids=None):
        self.user = user
        self.source_id = source_id
        # Validated query SCOPE (P5) — ready source ids, primary first, resolved
        # server-side by the view (QueryView._resolve_scope). Forwarded to inference
        # so multi-source scopes retrieve/federate exactly like /api/v1/query.
        self.source_ids = list(source_ids) if source_ids else ([source_id] if source_id else None)
        self.tenant = tenant

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
        session_id = str(chat.pk)
        kwargs = dict(tenant=self.tenant, source_id=self.source_id,
                      source_ids=self.source_ids, request_id=request_id)
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
                       "data": {"code": _CODE_MODEL_ERROR, "message": _MSG_MODEL_ERROR}}
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
            yield {"event": "error",
                   "data": {"code": _CODE_LLM_UNAVAILABLE,
                            "message": _MSG_LLM_UNAVAILABLE}}
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

        def on_event(phase: str, evt_message: str, extra: dict | None = None) -> None:
            # `extra` carries the inference tier's per-phase structured fields
            # (route's intent=, sub_query's index=/total=/sub_query=, ...) —
            # merged in so the SSE "thinking" payload isn't just flattened
            # phase/message text. phase/message win on key collision (unlikely,
            # but they're the guaranteed-present fields).
            # `phase` itself is forwarded verbatim (never renamed — logs/tracing
            # upstream are unaffected); only the displayed `message` is swapped
            # for a business-friendly one (thinking_messages.py, UX Phase 1).
            q.put(("thinking", {**(extra or {}), "phase": phase,
                               "message": business_friendly_message(phase, evt_message)}))

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
        thread.join(timeout=5)   # already finished by the time "done" was enqueued; bounds worst case

        if error_message is not None:
            # error_message is the raw str(exc) from the worker thread — already
            # logged with its traceback in target()'s except. Show the safe copy.
            yield {"event": "error",
                   "data": {"code": _CODE_MODEL_ERROR, "message": _MSG_MODEL_ERROR}}
            return None
        if result is None:
            logger.error("conversation query pipeline returned no result chat_id=%s", chat.pk)
            yield {"event": "error",
                   "data": {"code": _CODE_MODEL_ERROR, "message": _MSG_MODEL_ERROR}}
            return None
        return result

    def _build_reply_events(self, response: dict):
        res0 = response.get("engine_result") or {}
        # Computed (fast, synchronous, no LLM — same call as before) BEFORE any
        # content streams, so the thinking message below completes the
        # "thinking" sequence rather than interleaving mid-answer.
        vizzes = self._build_visualizations(res0)
        if vizzes:
            # Only emitted when a chart is actually about to be shown — a
            # text/table-only answer never yields this, so it's never a
            # "thinking" message describing work that isn't happening.
            yield {"event": "thinking",
                  "data": {"phase": "visualization_prep",
                           "message": business_friendly_message("visualization_prep", "")}}
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
        yield {"event": "explainability", "data": res0.get("explain") or _NO_EXPLAIN}
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

