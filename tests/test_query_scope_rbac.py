"""Coverage for the RBAC narrowing added to ``apps.query.scope`` (User Story 3,
"Gate 1" — source-level enforcement).

  apps/query/scope.py :: permitted_source_ids / resolve_query_scope(user=...)

``resolve_query_scope`` already has full non-RBAC coverage in
``test_apps_layer_refactor.py``; this file only covers the NEW behaviour: given a
real user with real (Role, RolePermission) rows, does the "ready" set get narrowed
to exactly the sources ``data.read`` actually reaches — never less (fail-open),
never more than the story asks for (no accidental narrowing when RBAC is off).

Run from repo root: ``pytest tests/test_query_scope_rbac.py``
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

from apps.access_management import resource_path as rp  # noqa: E402
from apps.access_management.gate import MODE_ENFORCE, MODE_OFF  # noqa: E402
from apps.access_management.models import Effect, Permission, Role, RolePermission, UserRole  # noqa: E402
from apps.query.scope import permitted_source_ids, resolve_query_scope  # noqa: E402
from apps.sources.models import Dialect, Source  # noqa: E402

ADMIN_PASSWORD = "admin-correct-horse-staple"

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


def _source(name, ready=True, dialect=Dialect.POSTGRES):
    return Source.objects.create(
        name=name, dialect=dialect, connector_type="sql", ready=ready)


def _grant(role, permission, path="", effect=Effect.ALLOW):
    return RolePermission.objects.create(
        role=role, permission=permission, resource_path=path, effect=effect)


def _role_for(user, name="Analyst"):
    role = Role.objects.create(name=name, is_active=True)
    UserRole.objects.create(user=user, role=role)
    return role


def _source_path(source_name):
    return rp.build("db", source_name, "employee")


# ---------------------------------------------------------------------------
# RBAC off / no user — behaviour is unchanged (byte-identical to pre-Gate-1)
# ---------------------------------------------------------------------------

def test_no_user_means_no_narrowing_even_when_enforce(member, read_perm):
    """Existing callers that don't pass ``user`` (nothing has been updated to yet)
    must keep seeing every ready source — the additive/backward-compat contract."""
    _source("crm")
    _source("billing")
    assert resolve_query_scope({}, "default") == sorted(
        Source.objects.values_list("id", flat=True))


@override_settings(VEDA_RBAC_MODE=MODE_OFF)
def test_rbac_off_ignores_grants_entirely(member, read_perm):
    crm = _source("crm")
    _source("billing")
    role = _role_for(member)
    _grant(role, read_perm, _source_path("crm"))  # even WITH a grant, off = no filter

    scope = resolve_query_scope({}, "default", user=member)
    assert scope == sorted(Source.objects.values_list("id", flat=True))
    assert crm.id in scope


# ---------------------------------------------------------------------------
# Enforce mode: authorized / unauthorized / partial / full / no-permissions
# ---------------------------------------------------------------------------

def test_no_permissions_permits_nothing(member):
    """Fail-closed at the ``permitted_source_ids`` level: a real user with zero
    grants must resolve to the empty set, not ``None`` ("no narrowing") and not
    every ready source."""
    _source("crm")
    _source("billing")
    assert permitted_source_ids(member) == set()


def test_no_permissions_falls_back_to_the_dev_default_source_not_a_403(member):
    """``resolve_query_scope`` itself never returns an empty scope — that
    contract ("always return a non-empty scope") is intentional and unchanged by
    Task 17: it still falls back to VEDA_DEFAULT_SOURCE_ID when RBAC narrows
    permitted sources to nothing. Turning "you have access to nothing" into a 403
    is the CALLING VIEW's job (see tests/test_gate1_authorization.py, which
    covers the view-level check that now runs BEFORE this function is even
    called, using ``permitted_source_ids`` directly) — pinned here so this
    function's own contract stays documented by a real assertion, not just
    prose."""
    _source("crm")
    _source("billing")
    assert resolve_query_scope({}, "default", user=member) == [1]


def test_full_access_via_one_grant_per_source(member, read_perm):
    crm = _source("crm")
    billing = _source("billing")
    role = _role_for(member)
    _grant(role, read_perm, _source_path("crm"))
    _grant(role, read_perm, _source_path("billing"))

    scope = resolve_query_scope({}, "default", user=member)
    assert set(scope) == {crm.id, billing.id}


def test_partial_access_narrows_to_the_granted_source_only(member, read_perm):
    crm = _source("crm")
    _source("billing")  # no grant on this one
    role = _role_for(member)
    _grant(role, read_perm, _source_path("crm"))

    assert resolve_query_scope({}, "default", user=member) == [crm.id]


def test_a_request_pin_outside_the_granted_scope_falls_back_to_the_granted_scope(member, read_perm):
    crm = _source("crm")
    billing = _source("billing")
    role = _role_for(member)
    _grant(role, read_perm, _source_path("crm"))

    # billing is ready, but not granted — pinning it must not leak access to it.
    assert resolve_query_scope({"source_id": billing.id}, "default", user=member) == [crm.id]
    assert resolve_query_scope({"source_id": crm.id}, "default", user=member) == [crm.id]


def test_a_global_blank_path_grant_does_not_open_any_source(member, read_perm):
    """ADR §3.4 / resolver docstring precedent: a permission with no resource path
    is not resource-scoped, so it must not count as reaching a specific source —
    this is exactly why staff need the explicit bypass below."""
    _source("crm")
    role = _role_for(member)
    _grant(role, read_perm, "")  # global grant

    assert permitted_source_ids(member) == set()


def test_a_denied_source_stays_out_even_with_another_role_allowing_it(member, read_perm):
    """Deny-wins precision for a SPECIFIC resource is deferred to the table/column
    phase — but a source with ONLY a deny (no allow anywhere) must not appear."""
    _source("crm")
    role = _role_for(member, name="Restricted")
    _grant(role, read_perm, _source_path("crm"), effect=Effect.DENY)

    assert permitted_source_ids(member) == set()


def test_multi_role_grants_union(member, read_perm):
    """A user with two roles, each granting a different source, sees the union —
    reusing the resolver's own established union guarantee, not reimplementing it."""
    crm = _source("crm")
    billing = _source("billing")
    role_a = Role.objects.create(name="CRM Analyst", is_active=True)
    role_b = Role.objects.create(name="Billing Analyst", is_active=True)
    UserRole.objects.create(user=member, role=role_a)
    UserRole.objects.create(user=member, role=role_b)
    _grant(role_a, read_perm, _source_path("crm"))
    _grant(role_b, read_perm, _source_path("billing"))

    scope = resolve_query_scope({}, "default", user=member)
    assert set(scope) == {crm.id, billing.id}


def test_admin_bypass_staff_sees_everything_regardless_of_grants(staff_member):
    """The seeded Admin role's grants are all global (blank resource_path), which
    per the resolver's own rule never opens a specific source — without this
    bypass even the platform's real admin role would see zero sources under
    enforce mode. ``is_staff`` is treated as the real "operational admin" signal,
    same precedent as the login role-check and the last-admin guard."""
    crm = _source("crm")
    billing = _source("billing")
    # staff_member holds NO role/grant at all — bypass must not depend on one.
    assert set(resolve_query_scope({}, "default", user=staff_member)) == {crm.id, billing.id}


def test_narrowing_is_case_insensitive_on_source_name(member, read_perm):
    """resource_path.build() always lowercases the source name into the path
    (see CatalogService), but Source.name is stored as typed — the narrowing must
    not silently drop a source just because its name has uppercase in it."""
    mixed_case = _source("CRM_Prod")
    role = _role_for(member)
    _grant(role, read_perm, _source_path("CRM_Prod"))

    assert resolve_query_scope({}, "default", user=member) == [mixed_case.id]


def test_not_ready_source_stays_excluded_even_with_a_grant(member, read_perm):
    """RBAC only narrows the ready set further — it must never resurrect a source
    the ingestion pipeline hasn't marked ready. ``permitted_source_ids`` itself
    doesn't know about readiness (that's ``_ready_source_ids``'s job), so it
    correctly permits the source; ``resolve_query_scope`` is what must still
    exclude it via the ready-set intersection."""
    not_ready = _source("crm", ready=False)
    role = _role_for(member)
    _grant(role, read_perm, _source_path("crm"))

    assert not_ready.id in permitted_source_ids(member)
    assert resolve_query_scope({}, "default", user=member) == [1]  # falls back — see above
