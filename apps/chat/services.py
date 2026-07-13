from __future__ import annotations

import json
import logging
import queue
import threading
from typing import Iterator

from chatbot.run import run_chat_turn

from .models import ChatMessage, ChatSession, MessageType
from .visualization import VisualizationRecommender

logger = logging.getLogger(__name__)

DEFAULT_CONVERSATION_TITLE = "New Chat"

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
        return _visualization_recommender._category_numeric(cols, rows, x_idx, y_idx)
    return None


def _rows_to_markdown_table(cols: list, rows: list, limit: int = 20) -> str:
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body_lines = [
        "| " + " | ".join(str(v) for v in row[:len(cols)]) + " |" for row in rows[:limit]
    ]
    return "\n".join([header, sep, *body_lines])


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

        if stream:
            response = yield from self._run_streamed(message, session_id, kwargs, chat)
            if response is None:
                return   # _run_streamed already yielded the error event
        else:
            try:
                response = run_chat_turn(message, session_id, **kwargs)
            except Exception as exc:
                logger.exception("conversation query pipeline failed chat_id=%s", chat.pk)
                yield {"event": "error", "data": {"code": "MODEL_ERROR", "message": str(exc)}}
                return

        if response.get("engine_unavailable"):
            logger.warning("conversation query pipeline unavailable chat_id=%s", chat.pk)
            yield {"event": "error",
                   "data": {"code": "MODEL_ERROR",
                            "message": response.get("reply_text") or "Inference tier unavailable."}}
            return

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

        def on_event(phase: str, evt_message: str) -> None:
            q.put(("thinking", {"phase": phase, "message": evt_message}))

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
            yield {"event": "error", "data": {"code": "MODEL_ERROR", "message": error_message}}
            return None
        if result is None:
            yield {"event": "error", "data": {"code": "MODEL_ERROR", "message": "No result from chatbot pipeline."}}
            return None
        return result

    def _build_reply_events(self, response: dict):
        res0 = response.get("engine_result") or {}
        for block in self._build_content_blocks(response, res0):
            yield {"event": "content", "data": block}
        for viz in self._build_visualizations(res0):
            yield {"event": "visualization", "data": viz}
        # veda_core (veda/business_explain.py) builds this deterministically from the
        # final validated SQL + semantic model — never from retrieval/routing internals
        # (those live only in res0["trace"], for our own debugging, never sent over SSE).
        yield {"event": "explainability", "data": res0.get("explain") or _NO_EXPLAIN}
        # Insight Engine (additive, new event type): only present when
        # INSIGHT_ENGINE_ENABLED produced these keys server-side (veda/pipeline.py's
        # _done()) — absent entirely when the flag is off, so old clients that only
        # listen for content/visualization/explainability/error see nothing new.
        if "insights" in res0 or "follow_up_questions" in res0 or "confidence" in res0:
            yield {"event": "insights", "data": {
                "insights": res0.get("insights") or [],
                "follow_up_questions": res0.get("follow_up_questions") or [],
                "confidence": res0.get("confidence"),
            }}

    @staticmethod
    def _build_content_blocks(response: dict, res0: dict) -> list:
        blocks = []
        reply_text = response.get("reply_text")   # covers answer + smalltalk + clarify uniformly
        if reply_text:
            # is_summary marks this as the primary answer (vs. supporting content like
            # the table below) so callers can surface it distinctly without a second
            # LLM call or re-deriving which block "is" the summary.
            blocks.append({"type": "markdown", "content": str(reply_text), "is_summary": True})
        cols, rows = res0.get("cols"), res0.get("rows")
        if cols and rows:
            rows = _positional_rows(cols, rows)
            blocks.append({"type": "markdown", "content": _rows_to_markdown_table(cols, rows)})
        if not blocks:
            blocks.append({"type": "markdown", "content": "No response could be generated."})
        return blocks

    @staticmethod
    def _build_visualizations(res0: dict) -> list:
        cols, rows = res0.get("cols"), res0.get("rows")
        if not cols or not rows:
            return []
        rows = _positional_rows(cols, rows)
        specs = _visualization_recommender.recommend(cols, rows)
        if specs:
            return [spec.to_dict() for spec in specs]
        # Deterministic rules found nothing confident — fall back to the query
        # tier's Insight Engine suggestion (already validated server-side:
        # column existence + type compatibility — see
        # query/result_explainer.py's validate_visualization). Still built into
        # the SAME chart_data shape via the existing recommender's own builders,
        # never served as a bare column-name suggestion.
        spec = _spec_from_suggestion(cols, rows, res0.get("visualization"))
        return [spec.to_dict()] if spec else []

