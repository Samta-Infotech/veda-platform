from __future__ import annotations

import json
import logging

from django.http import StreamingHttpResponse
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from apps.access_management.services import (
            compute_data_scope, resolve_effective_permissions, serialize_data_scope,
        )
from apps.core import api
from apps.core.messages import MESSAGES

from .models import MessageType
from .serializers import (
    ConversationHistorySerializer,
    ConversationQuerySerializer,
    CreateConversationSerializer,
)
from .services import (
    ChatNotFound,
    CODE_MODEL_ERROR,
    CODE_STREAM_ERROR,
    ConversationQueryService,
    MSG_MODEL_ERROR,
)
from .turn_events import TurnEventAccumulator
from apps.query.scope import (
    NoReadySource,
    SourceAccessDenied,
    permitted_source_ids,
    query_execute_allowed,
    resolve_query_scope,
)

logger = logging.getLogger(__name__)

# Tenant used by the chat entry point until tenant-from-principal lands (§6.2),
# matching apps.query.views.DEFAULT_TENANT.
DEFAULT_TENANT = "default"

_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _resolve_user(request):
    """The authenticated principal, or None.

    User Story 3 audit (2026-08-08) finding, fixed here: this used to fall back
    to the seeded dummy ``admin`` user (chat migration 0002) whenever the request
    carried no authenticated principal — meaning an UNAUTHENTICATED caller was
    silently treated as a real, persistent identity rather than rejected. Every
    caller of this function already checks ``if user is None: return
    _unauthenticated_response()`` — that check simply never fired before this
    fix. No dev convenience is lost: real auth (JWT/session/token) has existed
    in this app for a while now: a caller with no credentials gets 401, exactly
    like /api/v1/query's own AllowAny + real-request.user pattern.
    """
    user = getattr(request, "user", None)
    if user is not None and user.is_authenticated:
        return user
    return None


def _unauthenticated_response():
    return api.error(MESSAGES["chat"]["auth_required"], status.HTTP_401_UNAUTHORIZED)


def _forbidden_response():
    return api.error(MESSAGES["chat"]["access_denied"], status.HTTP_403_FORBIDDEN)


def _sse_format(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _iso_z(dt) -> str | None:
    # Delegates so the platform has one timestamp format; kept as a local alias
    # because it is used three times below and in _serialize_history_message.
    return api.iso_z(dt)


class ConversationQueryView(APIView):
    """POST /api/v1/conversations/query {message, chat_id?, stream?}."""

    permission_classes = [AllowAny]

    def post(self, request):
        rid = getattr(request, "request_id", "")
        logger.info("conversation query received request_id=%s", rid)

        serializer = ConversationQuerySerializer(data=request.data)
        if not serializer.is_valid():
            logger.warning("conversation query rejected: invalid payload — %s", serializer.errors)
            return api.invalid_payload(serializer.errors)
        data = serializer.validated_data
        logger.info("conversation query validated request_id=%s", rid)

        user = _resolve_user(request)
        if user is None:
            return _unauthenticated_response()

        # Gate 1 (User Story 3, Task 17): resolve RBAC permissions ONCE per
        # request — reused below by both the source-level check and the
        # table/column payload. Lazy import: this module must stay importable
        # without apps.access_management in INSTALLED_APPS for a caller that
        # never touches RBAC at all.

        effective = resolve_effective_permissions(user)

        # The coarse "may this caller use the query feature at all" gate,
        # checked BEFORE the source-level one below — orthogonal to data.read,
        # see query_execute_allowed's own docstring for why this previously had
        # zero effect (defined, seeded, grantable — never actually checked).
        #
        # Both this and the permitted-sources check below used to return a raw
        # HTTP 403 here — no chat/message row ever created, nothing in history,
        # a shape the frontend had to special-case apart from every other kind
        # of refusal. User's call: route these through the SAME turn/persist
        # path as a real answer instead (see ConversationQueryService.
        # access_denied) — still zero engine compute (that's what the early
        # return protects), but now it streams, and it saves to history like
        # any other turn.
        if not query_execute_allowed(user, effective):
            logger.warning("conversation query denied: user_id=%s lacks query.execute",
                           user.pk)
            return self._denied_turn_response(user, data, rid)

        # Authenticated + RBAC active but permitted NOTHING -> fail closed
        # BEFORE any scope resolution or engine call, never a leaked
        # resource/table/column name. `permitted is None` means "no narrowing at
        # all" (RBAC off, or staff) — not this branch.
        permitted = permitted_source_ids(user, effective)
        if permitted is not None and not permitted:
            logger.warning("conversation query denied: user_id=%s has no permitted sources",
                           user.pk)
            return self._denied_turn_response(user, data, rid)

        # Resolve the query SCOPE server-side exactly like /api/v1/query (§6.2, P5):
        # all READY sources by default, an optional request pin intersected with the
        # ready registry. VEDA_DEFAULT_SOURCE_ID is the last-resort fallback ONLY
        # when RBAC was never narrowing anything (no user / RBAC off) — once RBAC
        # HAS narrowed the set, resolve_query_scope raises rather than silently
        # substituting a source the caller didn't ask for and might not even be
        # connected (found live: a denied pin silently answered from a different
        # source with no indication of the swap; a temporarily-not-ready permitted
        # source fell back to a never-connected, empty-host row and surfaced as a
        # confusing "LLM_UNAVAILABLE").
        try:
            source_ids = resolve_query_scope(request.data, tenant=DEFAULT_TENANT, user=user,
                                             effective=effective)
        except SourceAccessDenied:
            logger.warning("conversation query denied: user_id=%s pinned a source "
                           "outside their RBAC grants", user.pk)
            return _forbidden_response()
        except NoReadySource:
            logger.warning("conversation query: user_id=%s has no ready permitted source",
                           user.pk)
            return api.error(MESSAGES["chat"]["no_ready_source"], status.HTTP_503_SERVICE_UNAVAILABLE)
        # Gate 1 (User Story 3, Task 15): same allow-payload as /api/v1/query,
        # threaded through the chat turn to call_engine_node (chatbot/nodes.py) —
        # None (RBAC off / staff) means no narrowing, exactly as before this change.
        data_scope = serialize_data_scope(compute_data_scope(user, source_ids, effective=effective))
        service = ConversationQueryService(
            user=user, source_id=source_ids[0], source_ids=source_ids,
            data_scope=data_scope)
        try:
            chat = service.resolve_chat(data["chat_id"], name_hint=data["message"])
        except ChatNotFound:
            logger.warning("conversation query: chat_id=%s not found", data["chat_id"])
            return api.error(MESSAGES["chat"]["not_found"], status.HTTP_404_NOT_FOUND)
        logger.info("conversation query chat loaded/created chat_id=%s request_id=%s", chat.pk, rid)

        service.save_user_message(chat, data["message"])

        if data["stream"]:
            return self._stream_response(service, chat, data["message"], rid)
        return self._json_response(service, chat, data["message"], rid)

    def _denied_turn_response(self, user, data, rid):
        """A source-level RBAC denial (missing query.execute, or zero permitted
        sources), rendered as a real turn instead of a raw HTTP error — see
        ConversationQueryService.access_denied's own docstring for why. No
        source_ids/data_scope: access_denied short-circuits run_turn before
        either would ever be read.
        """
        service = ConversationQueryService(user=user, access_denied=True)
        try:
            chat = service.resolve_chat(data["chat_id"], name_hint=data["message"])
        except ChatNotFound:
            logger.warning("conversation query: chat_id=%s not found", data["chat_id"])
            return api.error(MESSAGES["chat"]["not_found"], status.HTTP_404_NOT_FOUND)
        service.save_user_message(chat, data["message"])

        if data["stream"]:
            return self._stream_response(service, chat, data["message"], rid)
        return self._json_response(service, chat, data["message"], rid)

    def _json_response(self, service, chat, message, rid):
        logger.info("conversation query AI processing started chat_id=%s", chat.pk)
        turn = TurnEventAccumulator()
        # turn_error, not error: this holds the *engine's* error payload, which is a
        # different thing from the HTTP error rendered below from it.
        turn_error = None
        for evt in service.run_turn(chat, message, request_id=rid):
            kind, payload = evt["event"], evt["data"]
            if kind == "error":
                turn_error = payload
            else:
                turn.consume(kind, payload)
        logger.info("conversation query AI processing completed chat_id=%s", chat.pk)

        if turn_error is not None:
            return api.error(
                turn_error.get("message", "Unable to generate response."),
                status.HTTP_502_BAD_GATEWAY,
                data={"chat_id": chat.pk,
                      "code": turn_error.get("code", CODE_MODEL_ERROR)},
            )

        metadata = turn.metadata()
        assistant_msg = service.save_assistant_message(chat, turn.content_blocks, metadata)
        logger.info("conversation query persistence completed chat_id=%s message_id=%s",
                    chat.pk, assistant_msg.pk)

        response_data = {
            "chat_id": chat.pk,
            "message_id": assistant_msg.pk,
            "summary": turn.summary_text,
            "response": turn.content_blocks,
            "metadata": metadata,
        }
        if turn.insights is not None:
            response_data["insights"] = turn.insights["insights"]
            response_data["follow_up_questions"] = turn.insights["follow_up_questions"]

        return api.success(MESSAGES["conversation"]["query_processed"], response_data)

    def _stream_response(self, service, chat, message, rid):
        response = StreamingHttpResponse(
            self._sse_generator(service, chat, message, rid),
            content_type="text/event-stream",
        )
        response["Cache-Control"] = "no-cache"
        response["Connection"] = "keep-alive"
        response["X-Accel-Buffering"] = "no"
        return response

    def _sse_generator(self, service, chat, message, rid):
        """ Generator for streaming SSE responses from the conversation query service."""
        
        logger.info("conversation query streaming started chat_id=%s", chat.pk)
        turn = TurnEventAccumulator()
        try:
            for evt in service.run_turn(chat, message, request_id=rid, stream=True):
                kind, payload = evt["event"], evt["data"]
                if kind == "error":
                    # Terminal: the turn produced no answer, so nothing is
                    # persisted and no "completed" frame follows.
                    yield _sse_format("error", payload)
                    logger.warning("conversation query streaming error chat_id=%s: %s",
                                   chat.pk, payload)
                    return
                turn.consume(kind, payload)
                yield _sse_format(kind, payload)
        except Exception:  # never break the connection mid-stream
            # Raw exception logged (with traceback) — NOT sent to the client; show
            # the safe copy, same as the other error paths (services.py).
            logger.exception("conversation query streaming failed chat_id=%s", chat.pk)
            yield _sse_format("error",
                              {"code": CODE_STREAM_ERROR, "message": MSG_MODEL_ERROR})
            return

        metadata = turn.metadata()
        assistant_msg = service.save_assistant_message(chat, turn.content_blocks, metadata)
        logger.info("conversation query persistence completed chat_id=%s message_id=%s",
                    chat.pk, assistant_msg.pk)
        yield _sse_format("completed",
                          {"chat_id": chat.pk, "message_id": assistant_msg.pk,
                           "summary": turn.summary_text, "is_complete": True})
        logger.info("conversation query streaming completed chat_id=%s", chat.pk)


class CreateConversationView(APIView):
    """POST /api/v1/conversations/create {conversation_title?}."""

    permission_classes = [AllowAny]

    def post(self, request):
        """Create a new conversation (chat)."""

        logger.info("conversation creation requested")

        serializer = CreateConversationSerializer(data=request.data)
        if not serializer.is_valid():
            logger.warning("conversation creation rejected: invalid payload — %s", serializer.errors)
            return api.invalid_payload(serializer.errors)

        user = _resolve_user(request)
        if user is None:
            return _unauthenticated_response()

        service = ConversationQueryService(user=user)
        chat = service.create_conversation(serializer.validated_data["conversation_title"] or "")
        logger.info("conversation created chat_id=%s user_id=%s", chat.pk, user.pk)

        return api.success(MESSAGES["conversation"]["created"], {
            "chat_id": chat.pk,
            "conversation_title": chat.name,
            "created_at": _iso_z(chat.created_at),
            "created_by": user.pk,
        }, status_code=status.HTTP_201_CREATED)


class ListConversationsView(APIView):
    """GET /api/v1/conversations/list — owned, non-deleted conversations.

    Read-only, no params needed — GET only, cacheable/bookmarkable/
    safely-retryable, unlike POST.
    """

    permission_classes = [AllowAny]

    def get(self, request):
        """Get a list of conversations."""

        logger.info("conversation list requested")

        user = _resolve_user(request)
        if user is None:
            return _unauthenticated_response()

        service = ConversationQueryService(user=user)
        conversations = [
            {
                "chat_id": chat.pk,
                "conversation_title": chat.name,
                "created_at": _iso_z(chat.created_at),
                "updated_at": _iso_z(chat.updated_at),
            }
            for chat in service.list_conversations()
        ]
        logger.info("conversation list returned count=%s user_id=%s", len(conversations), user.pk)

        return api.success(MESSAGES["conversation"]["list"], {"conversations": conversations})


class ConversationHistoryView(APIView):
    """GET /api/v1/conversations/history?chat_id=

    Read-only, so GET only — its parameter travels as a query param.
    """

    permission_classes = [AllowAny]

    def get(self, request):
        logger.info("conversation history requested")

        serializer = ConversationHistorySerializer(data=request.query_params)
        if not serializer.is_valid():
            logger.warning("conversation history rejected: invalid payload — %s", serializer.errors)
            return api.invalid_payload(serializer.errors)

        user = _resolve_user(request)
        if user is None:
            return _unauthenticated_response()

        service = ConversationQueryService(user=user)
        chat_id = serializer.validated_data["chat_id"]
        try:
            chat, messages = service.get_conversation_history(chat_id)
        except ChatNotFound:
            logger.warning("conversation history: chat_id=%s not found", chat_id)
            return api.error(MESSAGES["conversation"]["not_found"], status.HTTP_404_NOT_FOUND)

        logger.info("conversation history returned chat_id=%s message_count=%s", chat.pk, len(messages))
        return api.success(MESSAGES["conversation"]["retrieved"], {
            "chat_id": chat.pk,
            "conversation_title": chat.name,
            "created_at": _iso_z(chat.created_at),
            "messages": [_serialize_history_message(m) for m in messages],
        })


def _serialize_history_message(msg) -> dict:
    """Serialize a Message model instance for the conversation history API response."""
    if msg.type == MessageType.ASSISTANT:
        try:
            response = json.loads(msg.content)
        except (TypeError, ValueError):
            response = [{"type": "markdown", "content": msg.content}]
        meta = msg.metadata or {}
        content = {
            "response": response,
            "metadata": {
                "thinking": meta.get("thinking", ""),
                "explainability": meta.get("explainability"),
                # dict() copy: the shared constant must never be handed out by
                # reference into a mutable response payload.
                "usage": meta.get("usage") or dict(_ZERO_USAGE),
            },
        }
    else:
        content = msg.content
    return {
        "message_id": msg.pk,
        "role": msg.type.upper(),
        "content": content,
        "created_at": _iso_z(msg.created_at),
    }
