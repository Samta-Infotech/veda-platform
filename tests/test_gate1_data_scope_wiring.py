"""Coverage for Gate 1 (User Story 3) Task 15 — the RBAC data-scope payload
crossing the Django-to-inference HTTP boundary.

  apps/access_management/services/data_scope.py :: serialize_data_scope
  apps/query/inference_client.py                :: X-Veda-Data-Scope header
  veda_core/context.py                          :: RequestContext.allowed_resources
                                                     / parse_allowed_resources
  inference/main.py                             :: the ASGI middleware that parses it

Each of the three legs is tested in isolation (serialize -> header -> parse) plus
the middleware's fail-closed behaviour on a malformed header — nothing here needs
a database, so it runs Django-free like ``test_apps_layer_refactor.py``.

Run from repo root: ``pytest tests/test_gate1_data_scope_wiring.py``
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _setup_django():
    import config
    from django.conf import settings

    if not settings.configured:
        settings.configure(
            INSTALLED_APPS=[
                "django.contrib.contenttypes", "django.contrib.auth",
                "apps.core", "apps.sources", "apps.substrate",
                "apps.access_management", "apps.query", "apps.chat",
            ],
            DATABASES={},
            USE_TZ=True,
            DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
        )
        import django
        django.setup()


_setup_django()

from apps.access_management.services.data_scope import (  # noqa: E402
    SourceDataScope,
    TableScope,
    serialize_data_scope,
)
from apps.query.inference_client import InferenceClient  # noqa: E402
from veda_core.context import RequestContext, parse_allowed_resources  # noqa: E402


# ---------------------------------------------------------------------------
# 1. serialize_data_scope — Django dataclasses -> JSON-safe wire dict
# ---------------------------------------------------------------------------

def test_serialize_none_stays_none():
    assert serialize_data_scope(None) is None


def test_serialize_open_source_has_an_empty_tables_object():
    scope = {1: SourceDataScope(open=True)}
    assert serialize_data_scope(scope) == {"1": {"open": True, "tables": {}}}


def test_serialize_restricted_source_lists_tables_and_columns():
    scope = {
        7: SourceDataScope(open=False, tables=(
            TableScope(name="Employee", columns=("id", "name")),
            TableScope(name="Department", columns=None),
        )),
    }
    assert serialize_data_scope(scope) == {
        "7": {"open": False, "tables": {"Employee": ["id", "name"], "Department": None}},
    }


def test_serialize_is_json_encodable():
    scope = {1: SourceDataScope(open=False, tables=(TableScope("t", ("c",)),))}
    json.dumps(serialize_data_scope(scope))  # must not raise


# ---------------------------------------------------------------------------
# 2. InferenceClient — the header is sent only when there is something to send
# ---------------------------------------------------------------------------

def _client():
    from apps.query.inference_client import InferenceClientConfig
    return InferenceClient(InferenceClientConfig(base_url="http://inference.test"))


def test_no_header_when_data_scope_is_none():
    request = _client()._request("/v1/x", {}, source_id=1, tenant="default")
    assert "X-Veda-Data-Scope" not in request.headers


def test_header_carries_the_json_encoded_payload():
    payload = {"1": {"open": True, "tables": {}}}
    request = _client()._request(
        "/v1/x", {}, source_id=1, tenant="default", data_scope=payload)
    assert json.loads(request.headers["X-veda-data-scope"]) == payload


def test_run_hybrid_query_and_stream_hybrid_query_both_accept_data_scope():
    """Signature-level regression: both real call sites (QueryView,
    call_engine_node) pass ``data_scope=`` — a future refactor dropping the
    parameter from either would break at the call site, not silently no-op."""
    import inspect
    assert "data_scope" in inspect.signature(InferenceClient.run_hybrid_query).parameters
    assert "data_scope" in inspect.signature(InferenceClient.stream_hybrid_query).parameters


# ---------------------------------------------------------------------------
# 3. parse_allowed_resources — the wire dict -> a hashable RequestContext field
# ---------------------------------------------------------------------------

def test_parse_none_and_empty_string_both_mean_no_restriction():
    assert parse_allowed_resources(None) is None
    assert parse_allowed_resources("") is None


def test_parse_round_trips_serialize_data_scopes_own_output():
    scope = {
        3: SourceDataScope(open=False, tables=(
            TableScope(name="Employee", columns=("id", "salary")),
        )),
        4: SourceDataScope(open=True),
    }
    wire = serialize_data_scope(scope)
    parsed = parse_allowed_resources(json.dumps(wire))
    parsed_by_id = dict(parsed)
    assert parsed_by_id[4] == (True, ())
    assert parsed_by_id[3] == (False, (("Employee", ("id", "salary")),))


def test_parsed_context_stays_hashable():
    wire = serialize_data_scope({1: SourceDataScope(open=True)})
    ctx = RequestContext(source_id=1, tenant="default",
                         allowed_resources=parse_allowed_resources(json.dumps(wire)))
    hash(ctx)  # must not raise


def test_malformed_json_raises_rather_than_silently_permitting_everything():
    """The parser itself fails loudly (not-JSON) so the caller — the inference
    middleware — is forced to decide the fail-closed behaviour explicitly,
    rather than this pure function silently returning "no restriction" for
    input it could not understand."""
    with pytest.raises(Exception):
        parse_allowed_resources("not json at all")


def test_a_context_with_no_allowed_resources_field_defaults_to_none():
    """Every pre-Task-15 construction (ingestion, evaluation, CLI tooling) never
    passes this kwarg at all — must stay byte-identical."""
    ctx = RequestContext(source_id=1, tenant="default")
    assert ctx.allowed_resources is None


# ---------------------------------------------------------------------------
# 4. inference/main.py middleware — parses the header, fails closed on garbage
# ---------------------------------------------------------------------------

def _app_with_captured_context(monkeypatch):
    from starlette.testclient import TestClient

    import inference.main as main_mod

    captured = []
    monkeypatch.setattr(main_mod, "set_context", lambda ctx: captured.append(ctx))
    app = main_mod.create_app()
    return TestClient(app), captured


def test_middleware_leaves_allowed_resources_none_when_header_absent(monkeypatch):
    client, captured = _app_with_captured_context(monkeypatch)
    client.get("/healthz", headers={
        "x-veda-source-id": "1", "x-veda-tenant": "default"})
    assert captured[-1].allowed_resources is None


def test_middleware_parses_a_valid_header(monkeypatch):
    client, captured = _app_with_captured_context(monkeypatch)
    wire = json.dumps({"1": {"open": True, "tables": {}}})
    client.get("/healthz", headers={
        "x-veda-source-id": "1", "x-veda-tenant": "default",
        "x-veda-data-scope": wire})
    assert captured[-1].allowed_resources == ((1, (True, ())),)


def test_middleware_fails_closed_on_a_malformed_header(monkeypatch):
    """A future sender bug in the header must deny everything, never silently
    fall through to the RBAC-off "no restriction" behaviour."""
    client, captured = _app_with_captured_context(monkeypatch)
    client.get("/healthz", headers={
        "x-veda-source-id": "1", "x-veda-tenant": "default",
        "x-veda-data-scope": "{not valid json"})
    assert captured[-1].allowed_resources == ()
