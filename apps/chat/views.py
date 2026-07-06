from __future__ import annotations

import json
import logging
import os

from django.contrib.auth import authenticate, get_user_model
from django.http import StreamingHttpResponse
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import MessageType
from .serializers import (
    ConversationHistorySerializer,
    ConversationQuerySerializer,
    CreateConversationSerializer,
    LoginRequestSerializer,
)
from .services import ChatNotFound, ConversationQueryService

logger = logging.getLogger(__name__)


def _resolve_user(request):
    """Real authenticated user if present; dev fallback to the seeded admin
    user otherwise (mirrors QueryView._resolve_tenant's dev-default pattern)."""
    user = getattr(request, "user", None)
    if user is not None and user.is_authenticated:
        return user
    return get_user_model().objects.filter(username="admin").first()


def _unauthenticated_response():
    return Response(
        {"status_code": status.HTTP_401_UNAUTHORIZED, "message": "Authentication required."},
        status=status.HTTP_401_UNAUTHORIZED,
    )


def _sse_format(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _iso_z(dt) -> str | None:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None


def _authenticate_login(request, username: str, password: str) -> dict | None:
    user = authenticate(request, username=username, password=password)
    if user is None or not user.is_active:
        return None
    return {
        "user_id": user.pk,
        "username": user.username,
        "display_name": user.first_name or user.username,
    }


class LoginView(APIView):
    """POST /api/v1/auth/login {username, password} — dummy dev login."""

    permission_classes = [AllowAny]

    def post(self, request):
        serializer = LoginRequestSerializer(data=request.data)
        if not serializer.is_valid():
            logger.warning("login rejected: invalid payload — %s", serializer.errors)
            return Response(
                {"status_code": status.HTTP_400_BAD_REQUEST, "message": "Invalid request data.",
                 "errors": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        creds = serializer.validated_data
        user = _authenticate_login(request, creds["username"], creds["password"])
        if user is None:
            logger.warning("login failed for username=%s", creds["username"])
            return Response(
                {"status_code": status.HTTP_401_UNAUTHORIZED,
                 "message": "Invalid username or password."},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        logger.info("login successful for username=%s", user["username"])
        return Response(
            {
                "status_code": status.HTTP_200_OK,
                "message": "Login successful.",
                "data": {
                    **user,
                    "access_token": "dummy_access_token",
                    "token_type": "Bearer",
                },
            },
            status=status.HTTP_200_OK,
        )


class ConversationQueryView(APIView):
    """POST /api/v1/conversations/query {message, chat_id?, stream?}."""

    permission_classes = [AllowAny]

    def post(self, request):
        rid = getattr(request, "request_id", "")
        logger.info("conversation query received request_id=%s", rid)

        serializer = ConversationQuerySerializer(data=request.data)
        if not serializer.is_valid():
            logger.warning("conversation query rejected: invalid payload — %s", serializer.errors)
            return Response(
                {"status_code": status.HTTP_400_BAD_REQUEST, "message": "Invalid request data.",
                 "errors": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )
        data = serializer.validated_data
        logger.info("conversation query validated request_id=%s", rid)

        user = _resolve_user(request)
        if user is None:
            return _unauthenticated_response()

        source_id = int(os.environ.get("VEDA_DEFAULT_SOURCE_ID", "1"))
        service = ConversationQueryService(user=user, source_id=source_id)
        try:
            chat = service.resolve_chat(data["chat_id"], name_hint=data["message"])
        except ChatNotFound:
            logger.warning("conversation query: chat_id=%s not found", data["chat_id"])
            return Response(
                {"status_code": status.HTTP_404_NOT_FOUND, "message": "Chat not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        logger.info("conversation query chat loaded/created chat_id=%s request_id=%s", chat.pk, rid)

        service.save_user_message(chat, data["message"])

        if data["stream"]:
            return self._stream_response(service, chat, data["message"], rid)
        return self._json_response(service, chat, data["message"], rid)

    def _json_response(self, service, chat, message, rid):
        logger.info("conversation query AI processing started chat_id=%s", chat.pk)
        content_blocks, explainability, thinking_text, error = [], [], "", None
        for evt in service.run_turn(chat, message, request_id=rid):
            kind, payload = evt["event"], evt["data"]
            if kind == "thinking":
                thinking_text = payload.get("message", "")
            elif kind in ("content", "visualization"):
                content_blocks.append(payload)  # one ordered response[] array (§history)
            elif kind == "explainability":
                explainability = payload.get("steps", [])
            elif kind == "error":
                error = payload
        logger.info("conversation query AI processing completed chat_id=%s", chat.pk)

        if error is not None:
            return Response(
                {"status_code": status.HTTP_502_BAD_GATEWAY,
                 "message": error.get("message", "Unable to generate response."),
                 "data": {"chat_id": chat.pk, "code": error.get("code", "MODEL_ERROR")}},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        metadata = {"thinking": thinking_text, "explainability": explainability}
        assistant_msg = service.save_assistant_message(chat, content_blocks, metadata)
        logger.info("conversation query persistence completed chat_id=%s message_id=%s",
                    chat.pk, assistant_msg.pk)

        return Response({
            "status_code": status.HTTP_200_OK,
            "message": "Query processed successfully.",
            "data": {
                "chat_id": chat.pk,
                "message_id": assistant_msg.pk,
                "response": content_blocks,
                "metadata": metadata,
            },
        }, status=status.HTTP_200_OK)

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
        logger.info("conversation query streaming started chat_id=%s", chat.pk)
        content_blocks, explainability, thinking_text = [], [], ""
        try:
            for evt in service.run_turn(chat, message, request_id=rid):
                kind, payload = evt["event"], evt["data"]
                if kind == "thinking":
                    thinking_text = payload.get("message", "")
                elif kind in ("content", "visualization"):
                    content_blocks.append(payload)  # one ordered response[] array (§history)
                elif kind == "explainability":
                    explainability = payload.get("steps", [])
                elif kind == "error":
                    yield _sse_format("error", payload)
                    logger.warning("conversation query streaming error chat_id=%s: %s",
                                   chat.pk, payload)
                    return
                yield _sse_format(kind, payload)
        except Exception as exc:  # never break the connection mid-stream
            logger.exception("conversation query streaming failed chat_id=%s", chat.pk)
            yield _sse_format("error", {"code": "STREAM_ERROR", "message": str(exc)})
            return

        metadata = {"thinking": thinking_text, "explainability": explainability}
        assistant_msg = service.save_assistant_message(chat, content_blocks, metadata)
        logger.info("conversation query persistence completed chat_id=%s message_id=%s",
                    chat.pk, assistant_msg.pk)
        yield _sse_format("completed",
                          {"chat_id": chat.pk, "message_id": assistant_msg.pk, "is_complete": True})
        logger.info("conversation query streaming completed chat_id=%s", chat.pk)


class CreateConversationView(APIView):
    """POST /api/v1/conversations/create {conversation_title?}."""

    permission_classes = [AllowAny]

    def post(self, request):
        logger.info("conversation creation requested")

        serializer = CreateConversationSerializer(data=request.data)
        if not serializer.is_valid():
            logger.warning("conversation creation rejected: invalid payload — %s", serializer.errors)
            return Response(
                {"status_code": status.HTTP_400_BAD_REQUEST, "message": "Invalid request data.",
                 "errors": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        user = _resolve_user(request)
        if user is None:
            return _unauthenticated_response()

        service = ConversationQueryService(user=user)
        chat = service.create_conversation(serializer.validated_data["conversation_title"] or "")
        logger.info("conversation created chat_id=%s user_id=%s", chat.pk, user.pk)

        return Response({
            "status_code": status.HTTP_201_CREATED,
            "message": "Conversation created successfully.",
            "data": {
                "chat_id": chat.pk,
                "conversation_title": chat.name,
                "created_at": _iso_z(chat.created_at),
                "created_by": user.pk,
            },
        }, status=status.HTTP_201_CREATED)


class ListConversationsView(APIView):
    """POST /api/v1/conversations/list {} — owned, non-deleted conversations."""

    permission_classes = [AllowAny]

    def post(self, request):
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

        return Response({
            "status_code": status.HTTP_200_OK,
            "message": "Conversations retrieved successfully.",
            "data": {"conversations": conversations},
        }, status=status.HTTP_200_OK)


class ConversationHistoryView(APIView):
    """POST /api/v1/conversations/history {chat_id}."""

    permission_classes = [AllowAny]

    def post(self, request):
        logger.info("conversation history requested")

        serializer = ConversationHistorySerializer(data=request.data)
        if not serializer.is_valid():
            logger.warning("conversation history rejected: invalid payload — %s", serializer.errors)
            return Response(
                {"status_code": status.HTTP_400_BAD_REQUEST, "message": "Invalid request data.",
                 "errors": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        user = _resolve_user(request)
        if user is None:
            return _unauthenticated_response()

        service = ConversationQueryService(user=user)
        chat_id = serializer.validated_data["chat_id"]
        try:
            chat, messages = service.get_conversation_history(chat_id)
        except ChatNotFound:
            logger.warning("conversation history: chat_id=%s not found", chat_id)
            return Response(
                {"status_code": status.HTTP_404_NOT_FOUND, "message": "Conversation not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        logger.info("conversation history returned chat_id=%s message_count=%s", chat.pk, len(messages))
        return Response({
            "status_code": status.HTTP_200_OK,
            "message": "Conversation retrieved successfully.",
            "data": {
                "chat_id": chat.pk,
                "conversation_title": chat.name,
                "created_at": _iso_z(chat.created_at),
                "messages": [_serialize_history_message(m) for m in messages],
            },
        }, status=status.HTTP_200_OK)


def _serialize_history_message(msg) -> dict:
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
                "explainability": meta.get("explainability", []),
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
