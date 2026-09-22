"""Coverage for the governance layer (traceability Parts 19/20/21).

  apps/query/audit.py                        :: record_query, audit_fields_from_explain
  apps/query/models.py                      :: QueryLog.user + partial/warnings/sources
  apps/access_management/models/audit.py    :: AuthorizationDecision
  apps/access_management/gate.py            :: RequiresPermission._audit

Uses the REAL models and the REAL gate — no mocked resolver — so a change to how
denials are classified is caught here.

Run from repo root: ``pytest tests/test_query_governance.py``
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _setup_django():
    import config  # noqa: F401

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
    import django

    django.setup()


_setup_django()

from django.contrib.auth import get_user_model  # noqa: E402
from django.test import override_settings  # noqa: E402

from apps.access_management.gate import (  # noqa: E402
    MODE_ENFORCE, MODE_OFF, MODE_SHADOW, RequiresPermission,
)
from apps.access_management.models import (  # noqa: E402
    AuthorizationDecision, Decision, DenialReason, Effect, Permission, Role,
    RolePermission, UserRole,
)
from apps.query.audit import (  # noqa: E402
    CACHED_TABLE_SENTINEL, audit_fields_from_explain, normalize_status, record_query,
)
from apps.query.models import QueryLog, TerminalStatus  # noqa: E402

# ── DB harness ───────────────────────────────────────────────────────────────
# Hand-rolled rather than pytest-django's `db` fixture: the plugin is installed but
# NOT loaded in this repo's pytest configuration, so `pytest.mark.django_db` is an
# unknown mark here. tests/test_data_scope.py established this pattern — a
# module-scoped real test database plus per-test transaction rollback — and this
# follows it so both suites behave identically under the same runner.
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
    """Every test runs in a transaction that is rolled back, so the real audit
    tables are never left dirty."""
    from django.db import transaction as db_transaction

    atomic = db_transaction.atomic()
    atomic.__enter__()
    try:
        yield
    finally:
        db_transaction.set_rollback(True)
        atomic.__exit__(None, None, None)


# ── fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def user():
    return get_user_model().objects.create_user(
        username="gov_tester", email="gov@example.com", password="x", is_active=True)


class _Req:
    """Minimal request stand-in: the gate reads only .user and .request_id."""

    def __init__(self, user, request_id="rid-gov-1", path="/api/v1/users/list"):
        self.user = user
        self.request_id = request_id
        self.path = path


class _View:
    def __init__(self, code):
        self.required_permission = code


# ── Part 20: user FK on QueryLog ─────────────────────────────────────────────
def test_querylog_records_the_user(user):
    record_query(query="how many x", tenant="gov_tester", user=user, status="answered",
                 request_id="rid-1")
    row = QueryLog.objects.get(request_id="rid-1")
    assert row.user_id == user.pk
    assert row.status == TerminalStatus.ANSWERED


def test_querylog_survives_user_deletion(user):
    """SET_NULL, never CASCADE: deleting a user must not delete audit history."""
    record_query(query="q", tenant="t", user=user, status="answered", request_id="rid-2")
    uid = user.pk
    user.delete()
    row = QueryLog.objects.get(request_id="rid-2")
    assert row.user_id is None
    assert row.query_text == "q"
    assert not get_user_model().objects.filter(pk=uid).exists()


def test_anonymous_user_is_recorded_as_none_not_dropped():
    """An unsaved/anonymous principal must not cost us the whole audit row."""
    class Anon:
        pk = None
        is_authenticated = False

    record_query(query="q", tenant="t", user=Anon(), status="answered", request_id="rid-3")
    row = QueryLog.objects.get(request_id="rid-3")
    assert row.user_id is None


# ── Part 19: the shared writer + derived fields ──────────────────────────────
def test_status_normalization_covers_engine_vocabularies():
    # SubResult / MultiResult statuses are NOT TerminalStatus members
    assert normalize_status("ok") == "answered"
    assert normalize_status("refused") == "refuse"
    assert normalize_status("error") == "exec_error"
    # real TerminalStatus values pass through
    assert normalize_status("qualifier_dropped") == "qualifier_dropped"
    assert normalize_status("ANSWERED") == "answered"
    # an unmapped status must not invent a new bucket
    assert normalize_status("some_new_engine_status") == "refuse"
    assert normalize_status("") == "exec_error"
    assert normalize_status(None) == "exec_error"


def test_normalized_status_is_always_a_model_choice():
    valid = {c[0] for c in TerminalStatus.choices}
    for probe in ("ok", "refused", "error", "access_denied", "no_match", "unavailable",
                  "", None, "garbage", "answered"):
        assert normalize_status(probe) in valid


def test_audit_fields_from_v2_explain():
    explain = {
        "version": "2.0",
        "sql": {"enabled": True, "query": "SELECT COUNT(*) FROM t WHERE x = %s"},
        "result": {"row_count": 5, "truncated": False, "partial": True},
        "warnings": [{"code": "partial_source_failure", "severity": "warning",
                      "message": "..."}],
        "execution": {"sources": [{"id": "2", "name": "homzhub"},
                                  {"id": "3", "name": "docs"}]},
        "support": {"trace_id": "abc123"},
    }
    got = audit_fields_from_explain(explain)
    assert got["sql"].startswith("SELECT COUNT(*)")
    assert got["partial"] is True
    assert got["warning_codes"] == ["partial_source_failure"]
    assert got["participating_sources"] == ["2", "3"]
    assert got["request_id"] == "abc123"


def test_audit_fields_tolerates_v1_and_refusal_payloads():
    v1 = {"version": "1.0", "sql": {"enabled": True, "query": "SELECT 1"}}
    assert audit_fields_from_explain(v1) == {"sql": "SELECT 1"}

    refusal = {"version": "2.0", "status": "ungrounded", "why": "no such value",
               "warnings": [], "limitations": []}
    got = audit_fields_from_explain(refusal)
    assert got["status"] == "ungrounded"
    assert got["refusal"] == "no such value"
    assert "sql" not in got

    assert audit_fields_from_explain(None) == {}
    assert audit_fields_from_explain("not a dict") == {}


def test_record_query_persists_the_derived_governance_fields(user):
    record_query(query="q", tenant="t", user=user, status="answered", request_id="rid-4",
                 participating_sources=["2", 3, "junk"], partial=True,
                 warning_codes=["result_truncated", None, ""],
                 usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                 latency_ms=1234.7, cache_hit=True, route="deterministic",
                 sql="SELECT %s", refusal="")
    row = QueryLog.objects.get(request_id="rid-4")
    assert row.participating_sources == [2, 3]        # unparseable id dropped, not stored
    assert row.partial is True
    assert row.warning_codes == ["result_truncated"]  # blanks filtered
    assert (row.prompt_tokens, row.completion_tokens, row.total_tokens) == (10, 2, 12)
    assert row.latency_ms == 1235                     # rounded to the int column
    assert row.cache_hit is True
    assert row.route == "deterministic"


def test_record_query_never_raises_on_a_bad_row(user):
    """Audit is best-effort by contract: it must not turn an answered query into a 500."""
    before = QueryLog.objects.count()
    # `route` is max_length=16; an over-long value is truncated, not raised
    record_query(query="q", tenant="t", user=user, status="answered",
                 route="x" * 200, request_id="rid-5")
    assert QueryLog.objects.filter(request_id="rid-5").exists()
    # and a genuinely unpersistable value is swallowed rather than propagated
    record_query(query="q", tenant="t", user=object(), status="answered",
                 request_id="rid-6")
    assert QueryLog.objects.count() >= before


def test_cache_sentinel_is_shared_not_retyped():
    """Both front doors must agree on this string or one silently records
    cache_hit=False forever."""
    from apps.query.views import _CACHED_TABLE_SENTINEL
    from apps.chat.services import CACHED_TABLE_SENTINEL as chat_sentinel
    assert _CACHED_TABLE_SENTINEL is CACHED_TABLE_SENTINEL
    assert chat_sentinel is CACHED_TABLE_SENTINEL


# ── Part 21: authorization decision audit ────────────────────────────────────
@pytest.fixture
def data_read():
    return Permission.objects.get_or_create(
        code="data.read", defaults={"name": "Read data", "is_active": True})[0]


def _grant(user, permission, resource_path, effect):
    role = Role.objects.create(name=f"gov_role_{effect}_{resource_path or 'global'}")
    UserRole.objects.create(user=user, role=role)
    RolePermission.objects.create(role=role, permission=permission,
                                  resource_path=resource_path, effect=effect)
    return role


@override_settings(VEDA_RBAC_MODE=MODE_ENFORCE, VEDA_AUTHZ_AUDIT=True)
def test_denial_with_no_grant_is_audited_as_no_grant(user, data_read):
    gate = RequiresPermission()
    view = _View("data.read")
    req = _Req(user)
    view.get_required_resource = lambda r: "db:crm"

    assert gate.has_permission(req, view) is False
    row = AuthorizationDecision.objects.get(request_id="rid-gov-1")
    assert row.user_id == user.pk
    assert row.decision == Decision.DENY
    assert row.reason_code == DenialReason.NO_GRANT
    assert row.action == "data.read"
    assert row.mode == MODE_ENFORCE


@override_settings(VEDA_RBAC_MODE=MODE_ENFORCE, VEDA_AUTHZ_AUDIT=True)
def test_explicit_deny_is_distinguished_from_no_grant(user, data_read):
    """The whole point of Part 21: "never granted" and "explicitly forbidden" need
    different fixes, and a log line cannot tell them apart."""
    _grant(user, data_read, "db:crm", Effect.ALLOW)      # source-level allow
    _grant(user, data_read, "db:crm:salary", Effect.DENY)  # explicit carve-out

    gate = RequiresPermission()
    view = _View("data.read")
    view.get_required_resource = lambda r: "db:crm:salary"
    req = _Req(user, request_id="rid-gov-deny")

    assert gate.has_permission(req, view) is False
    row = AuthorizationDecision.objects.get(request_id="rid-gov-deny")
    assert row.reason_code == DenialReason.EXPLICIT_DENY


@override_settings(VEDA_RBAC_MODE=MODE_ENFORCE, VEDA_AUTHZ_AUDIT=True)
def test_allow_is_not_audited(user, data_read):
    """Only denials are recorded — a row per permitted request would be enormous
    and adds nothing (the denominator comes from QueryLog)."""
    _grant(user, data_read, "db:crm", Effect.ALLOW)
    gate = RequiresPermission()
    view = _View("data.read")
    view.get_required_resource = lambda r: "db:crm"

    assert gate.has_permission(_Req(user, request_id="rid-allow"), view) is True
    assert not AuthorizationDecision.objects.filter(request_id="rid-allow").exists()


@override_settings(VEDA_RBAC_MODE=MODE_SHADOW, VEDA_AUTHZ_AUDIT=True)
def test_shadow_would_deny_is_audited_and_marked_shadow(user, data_read):
    gate = RequiresPermission()
    view = _View("data.read")
    view.get_required_resource = lambda r: "db:crm"
    req = _Req(user, request_id="rid-shadow")

    # shadow NEVER blocks...
    assert gate.has_permission(req, view) is True
    # ...but the would-have-denied IS recorded, flagged so a rollout is not read
    # as an outage
    row = AuthorizationDecision.objects.get(request_id="rid-shadow")
    assert row.decision == Decision.DENY
    assert row.mode == MODE_SHADOW


@override_settings(VEDA_RBAC_MODE=MODE_ENFORCE, VEDA_AUTHZ_AUDIT=True)
def test_undeclared_permission_is_audited_and_fails_closed(user):
    class Bare:
        pass  # opted into the gate but declared no permission

    gate = RequiresPermission()
    req = _Req(user, request_id="rid-undeclared")
    assert gate.has_permission(req, Bare()) is False
    row = AuthorizationDecision.objects.get(request_id="rid-undeclared")
    assert row.reason_code == DenialReason.UNDECLARED
    assert row.action == ""


@override_settings(VEDA_RBAC_MODE=MODE_ENFORCE, VEDA_AUTHZ_AUDIT=True)
def test_resource_path_is_never_stored(user, data_read):
    """The path names the schema/table/column of something the caller was just
    told they may not see. Only the coarse KIND may be durable."""
    gate = RequiresPermission()
    view = _View("data.read")
    view.get_required_resource = lambda r: "db:crm_postgres:employee:salary"
    gate.has_permission(_Req(user, request_id="rid-leak"), view)

    row = AuthorizationDecision.objects.get(request_id="rid-leak")
    assert row.resource_kind == "db"
    blob = " ".join(str(getattr(row, f.name)) for f in AuthorizationDecision._meta.fields)
    for leak in ("crm_postgres", "employee", "salary", "db:crm"):
        assert leak not in blob, f"{leak!r} must not be persisted"


@override_settings(VEDA_RBAC_MODE=MODE_ENFORCE, VEDA_AUTHZ_AUDIT=False)
def test_no_audit_row_when_the_flag_is_off(user, data_read):
    gate = RequiresPermission()
    view = _View("data.read")
    view.get_required_resource = lambda r: "db:crm"
    assert gate.has_permission(_Req(user, request_id="rid-off"), view) is False
    assert not AuthorizationDecision.objects.filter(request_id="rid-off").exists()


@override_settings(VEDA_RBAC_MODE=MODE_OFF, VEDA_AUTHZ_AUDIT=True)
def test_gate_off_allows_and_audits_nothing(user, data_read):
    gate = RequiresPermission()
    view = _View("data.read")
    view.get_required_resource = lambda r: "db:crm"
    assert gate.has_permission(_Req(user, request_id="rid-modeoff"), view) is True
    assert not AuthorizationDecision.objects.filter(request_id="rid-modeoff").exists()


@override_settings(VEDA_RBAC_MODE=MODE_ENFORCE, VEDA_AUTHZ_AUDIT=True)
def test_audit_failure_never_changes_the_decision(user, data_read, monkeypatch):
    """An audit write must never be able to turn a deny into a 500."""
    from apps.access_management.models import audit as audit_models

    class Boom:
        @staticmethod
        def create(**kw):
            raise RuntimeError("audit table is on fire")

    monkeypatch.setattr(audit_models.AuthorizationDecision, "objects", Boom())
    gate = RequiresPermission()
    view = _View("data.read")
    view.get_required_resource = lambda r: "db:crm"
    # still a clean, correct DENY
    assert gate.has_permission(_Req(user, request_id="rid-boom"), view) is False


def test_authorization_decision_correlates_to_querylog(user, data_read):
    """One id end to end: the same request_id joins the audit row, the QueryLog row
    and the engine trace."""
    rid = "rid-correlate"
    with override_settings(VEDA_RBAC_MODE=MODE_ENFORCE, VEDA_AUTHZ_AUDIT=True):
        gate = RequiresPermission()
        view = _View("data.read")
        view.get_required_resource = lambda r: "db:crm"
        gate.has_permission(_Req(user, request_id=rid), view)
    record_query(query="q", tenant="t", user=user, status="refuse", request_id=rid)

    assert AuthorizationDecision.objects.filter(request_id=rid).count() == 1
    assert QueryLog.objects.filter(request_id=rid).count() == 1


# ── the governance questions Part 21 names, answered by query ────────────────
@override_settings(VEDA_RBAC_MODE=MODE_ENFORCE, VEDA_AUTHZ_AUDIT=True)
def test_the_three_operator_questions_are_answerable(user, data_read):
    other = get_user_model().objects.create_user(username="gov_other", password="x")
    gate = RequiresPermission()
    view = _View("data.read")
    view.get_required_resource = lambda r: "db:crm"
    gate.has_permission(_Req(user, request_id="q1"), view)
    gate.has_permission(_Req(other, request_id="q2"), view)

    _grant(other, data_read, "db:hr", Effect.ALLOW)
    _grant(other, data_read, "db:hr:pay", Effect.DENY)
    view.get_required_resource = lambda r: "db:hr:pay"
    gate.has_permission(_Req(other, request_id="q3"), view)

    # "How many queries were denied?"
    assert AuthorizationDecision.objects.filter(decision=Decision.DENY).count() == 3
    # "Which users experienced access denials?"
    assert set(AuthorizationDecision.objects.values_list("user__username", flat=True)) == {
        "gov_tester", "gov_other"}
    # "Missing permission, or policy?"
    by_reason = dict(
        (r, AuthorizationDecision.objects.filter(reason_code=r).count())
        for r in (DenialReason.NO_GRANT, DenialReason.EXPLICIT_DENY))
    assert by_reason[DenialReason.NO_GRANT] == 2
    assert by_reason[DenialReason.EXPLICIT_DENY] == 1


def test_participating_sources_prefers_what_executed_over_the_scope(user):
    """Both front doors must agree on this column's meaning. Measured live: the
    query path recorded the whole permitted scope while chat recorded only the
    sources that ran."""
    explain = {"version": "2.0",
               "execution": {"sources": [{"id": "2", "name": "homzhub"}]}}
    derived = audit_fields_from_explain(explain)
    assert derived["participating_sources"] == ["2"]

    # the shared writer stores what it is given, normalized to ints
    record_query(query="q", tenant="t", user=user, status="answered",
                 request_id="rid-participating",
                 participating_sources=derived["participating_sources"])
    assert QueryLog.objects.get(request_id="rid-participating").participating_sources == [2]


def test_participating_sources_falls_back_to_scope_when_unknown(user):
    """A v1 payload / a refusal has no execution block — the scope is then the only
    honest answer available, not an empty list."""
    assert "participating_sources" not in audit_fields_from_explain(
        {"version": "1.0", "sql": {"query": "SELECT 1"}})
    record_query(query="q", tenant="t", user=user, status="refuse",
                 request_id="rid-fallback", participating_sources=[2, 3])
    assert QueryLog.objects.get(request_id="rid-fallback").participating_sources == [2, 3]
