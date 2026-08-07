"""Coverage for the permission resolver.

  apps/access_management/services/resolver.py
  POST /api/v1/users/permissions/effective

The resolver is the first component that produces an *authorization answer*, so the
tests are weighted toward the ways an answer can be wrong in a dangerous direction:

  * DENY must be unpierceable at any depth
  * absence of a grant must be denial, never a default allow
  * an inactive user / role / permission must contribute nothing
  * a global (blank-path) grant must NOT cover concrete resources
  * prefix matching must be segment-wise, so `db:crm` never covers `db:crm_replica`

It still enforces nothing — ``test_resolution_does_not_enforce_anything`` pins that.

Run from repo root: ``pytest tests/test_permission_resolver.py``
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
from django.test import Client, override_settings  # noqa: E402

from apps.access_management.models import (  # noqa: E402
    Effect,
    Permission,
    Role,
    RolePermission,
    UserRole,
)
from apps.access_management.services import (  # noqa: E402
    NO_PERMISSIONS,
    PermissionResolver,
)

URL = "/api/v1/users/permissions/effective"
ADMIN_PASSWORD = "admin-correct-horse-staple"

_TEST_OVERRIDES = dict(
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
    PASSWORD_HASHERS=["django.contrib.auth.hashers.MD5PasswordHasher"],
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
def admin_client():
    user = get_user_model().objects.create_user(
        username="root", password=ADMIN_PASSWORD, is_staff=True)
    client = Client()
    client.force_login(user)
    return client


@pytest.fixture
def read():
    return Permission.objects.get(code="data.read")


@pytest.fixture
def manage():
    return Permission.objects.get(code="user.manage")


def _grant(role, permission, path="", effect=Effect.ALLOW):
    return RolePermission.objects.create(
        role=role, permission=permission, resource_path=path, effect=effect)


def _role_for(member, name="Analyst", active=True):
    role = Role.objects.create(name=name, is_active=active)
    UserRole.objects.create(user=member, role=role)
    return role


def _resolve(member):
    return PermissionResolver().resolve(member)


# ---------------------------------------------------------------------------
# Default deny
# ---------------------------------------------------------------------------


def test_a_user_with_nothing_is_denied_everything(member):
    effective = _resolve(member)

    assert effective.grants == ()
    assert effective.allows("data.read", "db:crm:employee") is False
    assert effective.allows("user.manage") is False


def test_absence_of_a_grant_is_denial_not_a_default_allow(member, read):
    """The single most important property. A user with SOME permissions must still be
    denied the ones nobody granted."""
    role = _role_for(member)
    _grant(role, read, "db:crm:employee")

    effective = _resolve(member)

    assert effective.allows("data.read", "db:crm:employee") is True
    assert effective.allows("data.read", "db:other:table") is False
    assert effective.allows("user.manage") is False


def test_an_anonymous_or_missing_user_resolves_to_nothing():
    assert PermissionResolver().resolve(None) is NO_PERMISSIONS
    assert NO_PERMISSIONS.allows("data.read", "db:crm:employee") is False


# ---------------------------------------------------------------------------
# Prefix inheritance
# ---------------------------------------------------------------------------


def test_a_grant_on_a_source_covers_its_tables_and_columns(member, read):
    role = _role_for(member)
    _grant(role, read, "db:crm")

    effective = _resolve(member)

    assert effective.allows("data.read", "db:crm") is True
    assert effective.allows("data.read", "db:crm:employee") is True
    assert effective.allows("data.read", "db:crm:employee:salary") is True


def test_a_grant_on_a_table_does_not_cover_a_sibling(member, read):
    role = _role_for(member)
    _grant(role, read, "db:crm:employee")

    effective = _resolve(member)

    assert effective.allows("data.read", "db:crm:invoice") is False
    assert effective.allows("data.read", "db:crm") is False   # not upward either


def test_prefix_matching_is_segment_wise(member, read):
    """`db:crm` must never cover `db:crm_replica`. A string `startswith` would grant
    an entirely unrelated source."""
    role = _role_for(member)
    _grant(role, read, "db:crm")

    effective = _resolve(member)

    assert effective.allows("data.read", "db:crm_replica:employee") is False
    assert effective.allows("data.read", "db:crmx") is False


# ---------------------------------------------------------------------------
# DENY precedence
# ---------------------------------------------------------------------------


def test_deny_beats_allow_at_the_same_level(member, read):
    role = _role_for(member)
    _grant(role, read, "db:crm:employee", Effect.ALLOW)
    other = _role_for(member, "Restricted")
    _grant(other, read, "db:crm:employee", Effect.DENY)

    assert _resolve(member).allows("data.read", "db:crm:employee") is False


def test_a_broad_deny_cannot_be_pierced_by_a_deeper_allow(member, read):
    """ADR §3.5's accepted trade-off, pinned. "Deny the source except one table" is
    NOT expressible — the deny wins."""
    role = _role_for(member)
    _grant(role, read, "db:crm", Effect.DENY)
    _grant(role, read, "db:crm:employee", Effect.ALLOW)

    effective = _resolve(member)

    assert effective.allows("data.read", "db:crm:employee") is False
    assert effective.denies("data.read", "db:crm:employee") is True


def test_a_deny_elsewhere_does_not_leak(member, read):
    role = _role_for(member)
    _grant(role, read, "db:crm", Effect.ALLOW)
    _grant(role, read, "db:other", Effect.DENY)

    assert _resolve(member).allows("data.read", "db:crm:employee") is True


def test_denies_is_distinct_from_not_allowed(member, read):
    """An explicit DENY and "never granted" are different things to an operator."""
    role = _role_for(member)
    _grant(role, read, "db:crm", Effect.DENY)

    effective = _resolve(member)

    assert effective.denies("data.read", "db:crm:employee") is True
    assert effective.denies("data.read", "db:untouched:table") is False
    assert effective.allows("data.read", "db:untouched:table") is False


# ---------------------------------------------------------------------------
# Global grants
# ---------------------------------------------------------------------------


def test_a_global_grant_satisfies_an_unscoped_check(member, manage):
    role = _role_for(member)
    _grant(role, manage, "")

    assert _resolve(member).allows("user.manage") is True


def test_a_global_grant_does_not_cover_concrete_resources(member, read):
    """Fail-closed reading: granting `data.read` with no resource must NOT silently
    open every table."""
    role = _role_for(member)
    _grant(role, read, "")

    effective = _resolve(member)

    assert effective.allows("data.read") is True                    # the unscoped question
    assert effective.allows("data.read", "db:crm:employee") is False


def test_a_resource_grant_does_not_satisfy_an_unscoped_check(member, read):
    role = _role_for(member)
    _grant(role, read, "db:crm")

    assert _resolve(member).allows("data.read") is False


# ---------------------------------------------------------------------------
# Anything inactive grants nothing
# ---------------------------------------------------------------------------


def test_an_inactive_user_resolves_to_nothing(member, read):
    role = _role_for(member)
    _grant(role, read, "db:crm")
    member.is_active = False
    member.save(update_fields=["is_active"])

    assert _resolve(member).grants == ()


def test_a_retired_role_grants_nothing(member, read):
    role = _role_for(member)
    _grant(role, read, "db:crm")
    role.is_active = False
    role.save(update_fields=["is_active", "updated_at"])

    assert _resolve(member).allows("data.read", "db:crm:employee") is False


def test_a_disabled_permission_grants_nothing(member, read):
    role = _role_for(member)
    _grant(role, read, "db:crm")
    Permission.objects.filter(pk=read.pk).update(is_active=False)

    assert _resolve(member).allows("data.read", "db:crm:employee") is False


def test_revoking_the_assignment_removes_everything(member, read):
    role = _role_for(member)
    _grant(role, read, "db:crm")
    UserRole.objects.filter(user=member, role=role).delete()

    assert _resolve(member).grants == ()


# ---------------------------------------------------------------------------
# Multiple roles
# ---------------------------------------------------------------------------


def test_grants_from_several_roles_are_unioned(member, read, manage):
    analyst = _role_for(member, "Analyst")
    _grant(analyst, read, "db:crm")
    # Not named "Admin": migration 0007 seeds a real role by that name, and this
    # fixture role is an unrelated, unseeded namesake it would collide with.
    manager = _role_for(member, "Manager")
    _grant(manager, manage, "")

    effective = _resolve(member)

    assert effective.allows("data.read", "db:crm:employee") is True
    assert effective.allows("user.manage") is True
    assert set(effective.permission_codes) == {"data.read", "user.manage"}


def test_a_deny_in_one_role_overrides_an_allow_in_another(member, read):
    """Roles are additive for ALLOW but not for DENY — a restricted role cannot be
    escaped by also holding a permissive one."""
    _grant(_role_for(member, "Permissive"), read, "db:crm", Effect.ALLOW)
    _grant(_role_for(member, "Restricted"), read, "db:crm", Effect.DENY)

    assert _resolve(member).allows("data.read", "db:crm:employee") is False


# ---------------------------------------------------------------------------
# The result object
# ---------------------------------------------------------------------------


def test_the_result_is_immutable(member, read):
    """It is an authorization answer that will be cached and shared — a caller must
    not be able to edit its own permissions after the fact."""
    import dataclasses

    role = _role_for(member)
    _grant(role, read, "db:crm")
    effective = _resolve(member)

    with pytest.raises(dataclasses.FrozenInstanceError):
        effective.user_id = 999
    # a tuple has no append at all, and the mapping proxy refuses assignment
    with pytest.raises(AttributeError):
        effective.grants.append("nope")
    with pytest.raises(TypeError):
        effective._by_code["data.read"] = ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        effective.grants[0].effect = "allow"      # the Grant itself is frozen too


def test_resources_for_lists_only_allowed_paths(member, read):
    role = _role_for(member)
    _grant(role, read, "db:crm")
    _grant(role, read, "db:other", Effect.DENY)

    assert _resolve(member).resources_for("data.read") == ("db:crm",)


def test_an_unaddressable_path_matches_nothing(member, read):
    """A malformed path names no resource, so it can be granted nothing."""
    role = _role_for(member)
    _grant(role, read, "db:crm")

    assert _resolve(member).allows("data.read", "db::bad") is False


def test_resolution_is_one_query(member, read):
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    role = _role_for(member)
    for i in range(10):
        _grant(role, read, f"db:crm:table{i}")

    with CaptureQueriesContext(connection) as ctx:
        PermissionResolver().resolve_for_user_id(member.pk)

    assert len(ctx.captured_queries) == 1


def test_checks_cost_no_queries(member, read):
    """Resolution is one query; every subsequent decision is in-memory. This is what
    makes the object cacheable."""
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    role = _role_for(member)
    _grant(role, read, "db:crm")
    effective = _resolve(member)

    with CaptureQueriesContext(connection) as ctx:
        for _ in range(50):
            effective.allows("data.read", "db:crm:employee:salary")

    assert len(ctx.captured_queries) == 0


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------


def test_endpoint_returns_the_effective_set(admin_client, member, read):
    role = _role_for(member)
    _grant(role, read, "db:crm")

    response = admin_client.post(URL, {"user_id": member.pk},
                                 content_type="application/json")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["user_id"] == member.pk
    assert data["username"] == "alice"
    assert data["permission_codes"] == ["data.read"]
    assert data["permissions"] == [
        {"permission_code": "data.read", "resource_path": "db:crm", "effect": "allow"}]


def test_endpoint_answers_a_specific_question(admin_client, member, read):
    """So a client never re-implements prefix inheritance and drifts from the server."""
    role = _role_for(member)
    _grant(role, read, "db:crm")

    response = admin_client.post(
        URL, {"user_id": member.pk, "permission_code": "data.read",
              "resource_path": "db:crm:employee:salary"},
        content_type="application/json")

    decision = response.json()["data"]["decision"]
    assert decision["allowed"] is True
    assert decision["explicitly_denied"] is False
    assert decision["granted_on"] == ["db:crm"]


def test_endpoint_reports_an_explicit_deny_distinctly(admin_client, member, read):
    role = _role_for(member)
    _grant(role, read, "db:crm", Effect.DENY)

    decision = admin_client.post(
        URL, {"user_id": member.pk, "permission_code": "data.read",
              "resource_path": "db:crm:employee"},
        content_type="application/json").json()["data"]["decision"]

    assert decision["allowed"] is False
    assert decision["explicitly_denied"] is True


def test_endpoint_404s_for_an_unknown_user(admin_client):
    """"No permissions" and "no such user" are very different answers."""
    response = admin_client.post(URL, {"user_id": 999_999},
                                 content_type="application/json")

    assert response.status_code == 404


def test_endpoint_canonicalises_the_resource_path(admin_client, member, read):
    role = _role_for(member)
    _grant(role, read, "db:crm")

    decision = admin_client.post(
        URL, {"user_id": member.pk, "permission_code": "data.read",
              "resource_path": "DB:CRM:Employee"},
        content_type="application/json").json()["data"]["decision"]

    assert decision["allowed"] is True


def test_endpoint_rejects_a_resource_without_a_permission(admin_client, member):
    """A resource alone is not a question the resolver can answer."""
    response = admin_client.post(
        URL, {"user_id": member.pk, "resource_path": "db:crm"},
        content_type="application/json")

    assert response.status_code == 400
    assert "permission_code" in response.json()["errors"]


def test_endpoint_rejects_a_malformed_resource_path(admin_client, member):
    response = admin_client.post(
        URL, {"user_id": member.pk, "permission_code": "data.read",
              "resource_path": "db::bad"},
        content_type="application/json")

    assert response.status_code == 400


def test_endpoint_requires_staff(member):
    anonymous = Client().post(URL, {"user_id": member.pk},
                              content_type="application/json")
    plain = Client()
    plain.force_login(member)

    assert anonymous.status_code == 401
    assert plain.post(URL, {"user_id": member.pk},
                      content_type="application/json").status_code == 403


def test_endpoint_is_post_only(admin_client):
    assert admin_client.get(URL).status_code == 405
    assert admin_client.put(URL).status_code == 405


def test_route_is_wired():
    from django.urls import resolve

    from apps.access_management.views import EffectivePermissionsView

    assert resolve(URL).func.view_class is EffectivePermissionsView


def test_resolution_does_not_enforce_anything(member, read):
    """The resolver answers; it does not act. `/api/v1/query` is unchanged, and this
    test should only be deleted when Gate 2 deliberately changes that."""
    from django.urls import resolve as url_resolve

    from rest_framework.permissions import AllowAny

    role = _role_for(member)
    _grant(role, read, "db:crm", Effect.DENY)

    assert AllowAny in url_resolve("/api/v1/query").func.view_class.permission_classes
