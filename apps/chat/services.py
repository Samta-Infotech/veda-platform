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


def _rows_to_markdown_table(cols: list, rows: list, limit: int = 20) -> str:
    # rows are positional (each row is a list/tuple aligned with cols by index) —
    # this is what the engine actually returns (JSON-serialized SQL tuples), NOT
    # column-keyed dicts.
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

        stream=True sources "thinking" from the inference tier's real SSE progress
        events (classify / decompose / route / answer) as the pipeline actually
        advances, instead of one blocking call that only reports "done" at the end."""
        yield {"event": "thinking",
               "data": {"phase": "reasoning", "message": "Analyzing request..."}}

        client = InferenceClient()
        payload = None
        try:
            if stream:
                for kind, data in client.stream_hybrid_query(
                    message, source_id=self.source_id, tenant=self.tenant, request_id=request_id,
                    source_ids=self.source_ids,
                ):
                    if kind == "progress":
                        yield {"event": "thinking",
                               "data": {"phase": data.get("phase", "progress"),
                                        "message": data.get("message", "")}}
                    elif kind == "error":
                        logger.warning("conversation query pipeline error chat_id=%s: %s",
                                       chat.pk, data)
                        yield {"event": "error",
                               "data": {"code": "MODEL_ERROR",
                                        "message": data.get("message", "inference error")}}
                        return
                    elif kind == "result":
                        payload = data
            else:
                payload = client.run_hybrid_query(
                    message, source_id=self.source_id, tenant=self.tenant, request_id=request_id,
                    source_ids=self.source_ids,
                )
        except InferenceUnavailable as exc:
            logger.warning("conversation query pipeline unavailable chat_id=%s: %s", chat.pk, exc)
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
            blocks.append({"type": "markdown", "content": _rows_to_markdown_table(cols, rows)})
        if not blocks:
            blocks.append({"type": "markdown", "content": "No response could be generated."})
        return blocks

    @staticmethod
    def _build_visualizations(res0: dict) -> list:
        cols, rows = res0.get("cols"), res0.get("rows")
        if not cols or not rows:
            return []
        return [spec.to_dict() for spec in _visualization_recommender.recommend(cols, rows)]

