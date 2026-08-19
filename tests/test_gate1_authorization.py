"""Coverage for Gate 1 (User Story 3) Task 17 — the view-level 403 that closes the
loop on Tasks 13-16: a user permitted NOTHING must never reach the inference tier
at all, and the response must never leak a resource/table/column/internal-RBAC name.

  apps/query/views.py::QueryView.post
  apps/chat/views.py::ConversationQueryView.post

Both real entry points, driven over HTTP (Django ``Client``, real routes, real DB)
rather than calling the view method directly — the routing/serializer/permission
wiring is part of what's being verified. ``InferenceClient``/``run_chat_turn`` are
mocked: this suite is about the AUTHORIZATION decision, not the query engine, and
must not require a live inference/engine tier to run.

Cases (per the brief's own list): authorized, unauthorized (no permission),
multi-role, full access, partial access, RBAC-disabled, admin bypass.

Run from repo root: ``pytest tests/test_gate1_authorization.py``
"""
import json
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _setup_django():
    import config  # noqa: F401

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
    import django

    django.setup()


_setup_django()

from django.contrib.auth import get_user_model  # noqa: E402
from django.test import Client, override_settings  # noqa: E402

from apps.access_management import resource_path as rp  # noqa: E402
from apps.access_management.gate import MODE_ENFORCE, MODE_OFF  # noqa: E402
from apps.access_management.models import Effect, Permission, Role, RolePermission, UserRole  # noqa: E402
from apps.sources.models import Dialect, Source  # noqa: E402
from apps.chat.models import ChatMessage, MessageType

ADMIN_PASSWORD = "admin-correct-horse-staple"
QUERY_URL = "/api/v1/query"
CONVERSATION_QUERY_URL = "/api/v1/conversations/query"

_TEST_OVERRIDES = dict(
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
    PASSWORD_HASHERS=["django.contrib.auth.hashers.MD5PasswordHasher"],
    VEDA_RBAC_MODE=MODE_ENFORCE,
)


@pytest.fixture(scope="module", autouse=True)
def _database():
    from django.db import connection
    from django.test.utils import setup_test_environment, teardown_test_environment

    try:
        setup_test_environment()
        owns_environment = True
    except RuntimeError:
        owns_environment = False

    with override_settings(MIGRATION_MODULES={"substrate": None}):
        old_config = connection.creation.create_test_db(verbosity=0, serialize=False)
    try:
        yield
    finally:
        connection.creation.destroy_test_db(old_config, verbosity=0)
        if owns_environment:
            teardown_test_environment()


@pytest.fixture(autouse=True)
def _isolated(_database):
    from django.core.cache import cache
    from django.db import transaction as db_transaction

    with override_settings(**_TEST_OVERRIDES):
        cache.clear()
        atomic = db_transaction.atomic()
        atomic.__enter__()
        try:
            yield
        finally:
            db_transaction.set_rollback(True)
            atomic.__exit__(None, None, None)


@pytest.fixture
def member():
    return get_user_model().objects.create_user(username="alice", password=ADMIN_PASSWORD)


@pytest.fixture
def staff_member():
    return get_user_model().objects.create_user(
        username="root", password=ADMIN_PASSWORD, is_staff=True)


@pytest.fixture
def read_perm():
    return Permission.objects.get(code="data.read")


def _client_as(user):
    client = Client()
    client.force_login(user)
    return client


def _source(name, ready=True):
    return Source.objects.create(
        name=name, dialect=Dialect.POSTGRES, connector_type="sql", ready=ready)


def _grant(role, permission, path="", effect=Effect.ALLOW):
    return RolePermission.objects.create(
        role=role, permission=permission, resource_path=path, effect=effect)


def _role_for(user, name="Analyst"):
    role = Role.objects.create(name=name, is_active=True)
    UserRole.objects.create(user=user, role=role)
    return role


def _source_path(source_name):
    # SOURCE-level path (2 segments). Strict hierarchy (2026-08): only a
    # source-level allow opens a source, so a grant meant to open one must be here.
    return rp.build("db", source_name)


def _post_json(client, url, body):
    return client.post(url, data=json.dumps(body), content_type="application/json")


_FAKE_INFERENCE_PAYLOAD = {"status": "answered", "result": {"items": []}}


def _fake_run_chat_turn(*args, **kwargs):
    return {"session_id": kwargs.get("session_id", "1"), "answer_text": "ok",
            "status": "answered", "engine_result": {}, "engine_unavailable": False}


# ---------------------------------------------------------------------------
# /api/v1/query
# ---------------------------------------------------------------------------

@mock.patch("apps.query.inference_client.InferenceClient.run_hybrid_query",
           return_value=_FAKE_INFERENCE_PAYLOAD)
def test_query_no_permissions_is_403_and_never_reaches_inference(mock_run, member):
    _source("crm")
    client = _client_as(member)

    resp = _post_json(client, QUERY_URL, {"query": "how many rows"})

    assert resp.status_code == 403
    mock_run.assert_not_called()


def test_query_403_body_leaks_no_resource_names(member):
    _source("crm", ready=True)
    client = _client_as(member)

    resp = _post_json(client, QUERY_URL, {"query": "show me crm employee salary"})

    assert resp.status_code == 403
    body_text = resp.content.decode().lower()
    # No resource/table/column names, and no internal RBAC vocabulary (grant,
    # role, rbac, resource_path) — a generic denial only.
    for leaked in ("crm", "employee", "salary", "grant", "role", "rbac", "resource_path"):
        assert leaked not in body_text, f"{leaked!r} leaked into the 403 body: {body_text}"


@mock.patch("apps.query.inference_client.InferenceClient.run_hybrid_query",
           return_value=_FAKE_INFERENCE_PAYLOAD)
def test_query_full_access_is_authorized(mock_run, member, read_perm):
    crm = _source("crm")
    role = _role_for(member)
    _grant(role, read_perm, _source_path("crm"))
    client = _client_as(member)

    resp = _post_json(client, QUERY_URL, {"query": "how many rows"})

    assert resp.status_code == 200
    mock_run.assert_called_once()


@mock.patch("apps.query.inference_client.InferenceClient.run_hybrid_query",
           return_value=_FAKE_INFERENCE_PAYLOAD)
def test_query_partial_access_is_authorized_and_scoped_to_the_granted_source(mock_run, member, read_perm):
    crm = _source("crm")
    _source("billing")  # not granted
    role = _role_for(member)
    _grant(role, read_perm, _source_path("crm"))
    client = _client_as(member)

    resp = _post_json(client, QUERY_URL, {"query": "how many rows"})

    assert resp.status_code == 200
    called_kwargs = mock_run.call_args.kwargs
    assert called_kwargs["source_ids"] == [crm.id]


@mock.patch("apps.query.inference_client.InferenceClient.run_hybrid_query",
           return_value=_FAKE_INFERENCE_PAYLOAD)
def test_query_multi_role_union_is_authorized(mock_run, member, read_perm):
    crm = _source("crm")
    billing = _source("billing")
    role_a = Role.objects.create(name="A", is_active=True)
    role_b = Role.objects.create(name="B", is_active=True)
    UserRole.objects.create(user=member, role=role_a)
    UserRole.objects.create(user=member, role=role_b)
    _grant(role_a, read_perm, _source_path("crm"))
    _grant(role_b, read_perm, _source_path("billing"))
    client = _client_as(member)

    resp = _post_json(client, QUERY_URL, {"query": "how many rows"})

    assert resp.status_code == 200
    assert set(mock_run.call_args.kwargs["source_ids"]) == {crm.id, billing.id}


@mock.patch("apps.query.inference_client.InferenceClient.run_hybrid_query",
           return_value=_FAKE_INFERENCE_PAYLOAD)
def test_query_admin_bypass_is_authorized_with_zero_grants(mock_run, staff_member):
    _source("crm")
    client = _client_as(staff_member)

    resp = _post_json(client, QUERY_URL, {"query": "how many rows"})

    assert resp.status_code == 200
    mock_run.assert_called_once()


@override_settings(VEDA_RBAC_MODE=MODE_OFF)
@mock.patch("apps.query.inference_client.InferenceClient.run_hybrid_query",
           return_value=_FAKE_INFERENCE_PAYLOAD)
def test_query_rbac_disabled_is_authorized_with_zero_grants(mock_run, member):
    _source("crm")
    client = _client_as(member)

    resp = _post_json(client, QUERY_URL, {"query": "how many rows"})

    assert resp.status_code == 200
    mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# /api/v1/conversations/query
# ---------------------------------------------------------------------------

@mock.patch("apps.chat.services.run_chat_turn", side_effect=_fake_run_chat_turn)
def test_conversation_query_no_permissions_is_a_persisted_denial_turn_not_a_raw_403(mock_run, member):
    """User's call: a source-level denial (zero permitted sources) is rendered
    as a real turn — persisted, same 200 + response shape as any other answer
    — rather than a raw HTTP error with no chat/message row and nothing in
    history. The engine must still never be called; that's the actual thing
    being protected, not the status code."""
    _source("crm")
    client = _client_as(member)

    resp = _post_json(client, CONVERSATION_QUERY_URL,
                      {"message": "how many rows", "stream": False})

    assert resp.status_code == 200
    mock_run.assert_not_called()
    body = resp.json()["data"]
    assert body["chat_id"] and body["message_id"]
    assert "permission" in body["summary"].lower()

    # And it really did land in history — not just in the response body.
    messages = ChatMessage.objects.filter(session_id=body["chat_id"]).order_by("created_at")
    assert [m.type for m in messages] == [MessageType.USER, MessageType.ASSISTANT]
    assert "permission" in messages[1].content.lower()


@mock.patch("apps.chat.services.run_chat_turn", side_effect=_fake_run_chat_turn)
def test_conversation_query_full_access_is_authorized(mock_run, member, read_perm):
    _source("crm")
    role = _role_for(member)
    _grant(role, read_perm, _source_path("crm"))
    client = _client_as(member)

    resp = _post_json(client, CONVERSATION_QUERY_URL,
                      {"message": "how many rows", "stream": False})

    assert resp.status_code == 200
    mock_run.assert_called_once()


@mock.patch("apps.chat.services.run_chat_turn", side_effect=_fake_run_chat_turn)
def test_conversation_query_admin_bypass_is_authorized_with_zero_grants(mock_run, staff_member):
    _source("crm")
    client = _client_as(staff_member)

    resp = _post_json(client, CONVERSATION_QUERY_URL,
                      {"message": "how many rows", "stream": False})

    assert resp.status_code == 200
    mock_run.assert_called_once()


@override_settings(VEDA_RBAC_MODE=MODE_OFF)
@mock.patch("apps.chat.services.run_chat_turn", side_effect=_fake_run_chat_turn)
def test_conversation_query_rbac_disabled_is_authorized_with_zero_grants(mock_run, member):
    _source("crm")
    client = _client_as(member)

    resp = _post_json(client, CONVERSATION_QUERY_URL,
                      {"message": "how many rows", "stream": False})

    assert resp.status_code == 200
    mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Authentication (2026-08-08 audit finding): an unauthenticated chat request
# must be REJECTED, never silently treated as a real, persistent identity.
# ---------------------------------------------------------------------------

@mock.patch("apps.chat.services.run_chat_turn", side_effect=_fake_run_chat_turn)
def test_conversation_query_unauthenticated_is_401_not_a_dummy_identity(mock_run):
    _source("crm")
    client = Client()  # no force_login — no credentials at all

    resp = _post_json(client, CONVERSATION_QUERY_URL,
                      {"message": "how many rows", "stream": False})

    assert resp.status_code == 401
    mock_run.assert_not_called()
