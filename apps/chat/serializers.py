"""apps.chat.serializers — request-payload validation for the chat endpoints.

These are INPUT-only serializers: every one is constructed as
``Serializer(data=request.data)`` and read via ``.validated_data``; none is ever
used to render a response. Response shaping is done explicitly in ``views.py``
(and, for a persisted assistant turn, by ``_serialize_history_message``), because
the reply is a structured content-block array assembled by the service layer
rather than a flat model projection. That is also why no field needs
``write_only`` — nothing here is ever serialized outward.

Validation is deliberately SHALLOW — presence, type, and blank-ness only. It is
the first gate (a malformed body becomes a 400 before any work starts), not the
authorization or ownership check: ownership of a ``chat_id`` is enforced later,
against the requesting user, in ``ConversationQueryService.resolve_chat`` (which
raises ``ChatNotFound`` → 404). A serializer that merely accepts an integer must
never be mistaken for one that proves the caller owns that conversation.

Fields that the views read unconditionally (e.g. ``data["chat_id"]``) all declare
an explicit ``default``, so ``validated_data`` always carries the key and the
views never need ``.get()`` with a second fallback.
"""
from rest_framework import serializers


class LoginRequestSerializer(serializers.Serializer):
    """Credentials for ``POST /api/v1/auth/login``.

    Both fields are mandatory and non-blank so an empty submission is rejected
    here rather than reaching ``django.contrib.auth.authenticate`` — which treats
    a blank password as a normal failed attempt and would waste a password-hash
    comparison on it.

    No ``max_length``/format constraint is imposed: the credential is checked
    against the configured auth backend, and a stricter rule here would only
    reveal which inputs are *shaped* like real usernames.
    """

    username = serializers.CharField(required=True, allow_blank=False)
    password = serializers.CharField(required=True, allow_blank=False)


class ConversationQuerySerializer(serializers.Serializer):
    """Body of ``POST /api/v1/conversations/query`` — one assistant turn.

    Fields:
        message: The user's question. ``allow_blank=False`` because a blank turn
            has nothing to answer and would still cost a full pipeline run.
        chat_id: Existing conversation to continue. ``None`` (the default, and an
            explicitly permitted value) means "start a new conversation" — the
            service creates one, titled from ``message``. Kept nullable rather
            than merely optional so a client can send ``{"chat_id": null}``
            explicitly, which is what a fresh UI session does.
        stream: Whether to answer as an SSE stream. Defaults to **True** — the
            streaming path is the primary UX (it surfaces live "thinking"
            progress); ``false`` opts into the buffered JSON response instead.

    Note: ``message`` is intentionally uncapped here — it is persisted to
    ``ChatMessage.content``, a ``TextField`` with no length limit. Any bound on
    request size belongs at the gateway (nginx ``client_max_body_size``) /
    ``DATA_UPLOAD_MAX_MEMORY_SIZE``, one place for every endpoint, not
    re-litigated per serializer.

    The scope of the query (which data sources it may read) is NOT taken from
    this body — it is resolved server-side by ``apps.query.scope`` and any
    client-supplied source pin is intersected with the ready-source registry
    (§6.2). Nothing here grants data access.
    """

    message = serializers.CharField(required=True, allow_blank=False)
    chat_id = serializers.IntegerField(required=False, allow_null=True, default=None)
    stream = serializers.BooleanField(required=False, default=True)


class CreateConversationSerializer(serializers.Serializer):
    """Body of ``POST /api/v1/conversations/create``.

    ``conversation_title`` is fully optional: absent, blank, and ``null`` are all
    accepted and all mean "no title supplied". The view normalizes the ``None``
    case (``... or ""``) and ``ConversationQueryService.create_conversation``
    then applies the business rule — fall back to ``DEFAULT_CONVERSATION_TITLE``
    and truncate to the ``ChatSession.name`` column width.

    That truncation is deliberately NOT expressed as a ``max_length`` here: an
    over-long title is trimmed to fit rather than rejected, so naming a chat
    after a long first message can never fail the request.
    """

    conversation_title = serializers.CharField(
        required=False, allow_blank=True, allow_null=True, default=None
    )


class ConversationHistorySerializer(serializers.Serializer):
    """Body of ``POST /api/v1/conversations/history``.

    ``chat_id`` is required and non-null here — unlike the query endpoint, there
    is no "start a new one" fallback: a history request without a target is a
    client error (400), not an empty transcript.

    Ownership is enforced downstream, not here — see the module docstring.
    """

    chat_id = serializers.IntegerField(required=True)
