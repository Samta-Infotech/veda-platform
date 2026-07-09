from __future__ import annotations

import json
import logging
from typing import Iterator

from apps.query.inference_client import InferenceClient, InferenceUnavailable

from .models import ChatMessage, ChatSession, MessageType
from .visualization import VisualizationRecommender

logger = logging.getLogger(__name__)

DEFAULT_CONVERSATION_TITLE = "New Chat"

_visualization_recommender = VisualizationRecommender()


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
    """One assistant turn: resolve chat -> run the existing pipeline -> persist."""

    def __init__(self, user, source_id=None, tenant: str = "default"):
        self.user = user
        self.source_id = source_id
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
        """Yields: thinking (-> thinking*, when stream) -> content* -> visualization?
        -> explainability -> error?.

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
                )
        except InferenceUnavailable as exc:
            logger.warning("conversation query pipeline unavailable chat_id=%s: %s", chat.pk, exc)
            yield {"event": "error", "data": {"code": "MODEL_ERROR", "message": str(exc)}}
            return

        if payload is None:
            logger.warning("conversation query pipeline returned no result chat_id=%s", chat.pk)
            yield {"event": "error",
                   "data": {"code": "MODEL_ERROR", "message": "No result from inference tier."}}
            return

        res0, trace = self._first_item_result(payload)

        for block in self._build_content_blocks(res0):
            yield {"event": "content", "data": block}

        for viz in self._build_visualizations(res0):
            yield {"event": "visualization", "data": viz}

        yield {"event": "explainability",
               "data": {"steps": self._build_explainability_steps(trace)}}

    @staticmethod
    def _first_item_result(payload: dict) -> tuple[dict, dict]:
        result = (payload or {}).get("result") or {}
        items = result.get("items") or []
        item0 = items[0] if items and isinstance(items[0], dict) else {}
        res0 = item0.get("result") or {}
        trace = res0.get("trace") or {}
        return res0, trace

    @staticmethod
    def _build_content_blocks(res0: dict) -> list:
        blocks = []
        answer = res0.get("answer")
        if answer:
            # is_summary marks this as the primary answer (vs. supporting content like
            # the table below) so callers can surface it distinctly without a second
            # LLM call or re-deriving which block "is" the summary.
            blocks.append({"type": "markdown", "content": str(answer), "is_summary": True})
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

    @staticmethod
    def _build_explainability_steps(trace: dict) -> list:
        sections = trace.get("sections") or {}
        return [
            {"step": name, **(data if isinstance(data, dict) else {"detail": data})}
            for name, data in sections.items()
        ]
