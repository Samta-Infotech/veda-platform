"""Coverage for apps/access_management — role assignment and permission grants.

  POST /api/v1/users/roles/{assign,revoke,list}
  POST /api/v1/roles/permissions/{grant,revoke,list}

The two behaviours worth the most attention:

  * **Idempotency.** Assign/grant describe a desired state, so repeating them is
    success (201 new / 200 already), and revoke never 404s.
  * **One decision per triple.** Re-granting with the opposite effect must UPDATE the
    row, never add a contradicting second one — two rows disagreeing about the same
    (role, permission, resource) would make the outcome depend on row order.

Run from repo root: ``pytest tests/test_grants.py``
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
from django.db import IntegrityError, transaction  # noqa: E402
from django.test import Client, override_settings  # noqa: E402

from apps.access_management.models import (  # noqa: E402
    CatalogResource,
    Effect,
    Permission,
    Role,
    RolePermission,
    UserRole,
)
from apps.access_management.services import (  # noqa: E402
    CODE_PERMISSION_INACTIVE,
    CODE_ROLE_INACTIVE,
    RoleInactive,
    RolePermissionService,
    UserRoleService,
)

ASSIGN_URL = "/api/v1/users/roles/assign"
REVOKE_URL = "/api/v1/users/roles/revoke"
ASSIGN_LIST_URL = "/api/v1/users/roles/list"
GRANT_URL = "/api/v1/roles/permissions/grant"
GRANT_REVOKE_URL = "/api/v1/roles/permissions/revoke"
GRANT_LIST_URL = "/api/v1/roles/permissions/list"

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
def admin_user():
    return get_user_model().objects.create_user(
        username="root", password=ADMIN_PASSWORD, is_staff=True)


@pytest.fixture
def admin_client(admin_user):
    client = Client()
    client.force_login(admin_user)
    return client


@pytest.fixture
def member():
    return get_user_model().objects.create_user(username="alice", password=ADMIN_PASSWORD)


@pytest.fixture
def role():
    return Role.objects.create(name="Data Analyst")


@pytest.fixture
def permission():
    return Permission.objects.get(code="data.read")


@pytest.fixture
def resource():
    from apps.sources.models import Source

    source = Source.objects.create(
        name="crm_postgres", dialect="postgres", connector_type="relational")
    CatalogResource.objects.create(
        path="db:crm_postgres:employee", kind="db",
        parent_path="db:crm_postgres", source=source)
    return "db:crm_postgres:employee"


def _post(client, url, **body):
    return client.post(url, body, content_type="application/json")


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------


def test_assign_creates_the_edge(admin_client, member, role):
    response = _post(admin_client, ASSIGN_URL, user_id=member.pk, role_id=role.pk)

    assert response.status_code == 201
    data = response.json()["data"]
    assert data["user_id"] == member.pk
    assert data["role_id"] == role.pk
    assert UserRole.objects.filter(user=member, role=role).exists()


def test_assign_records_who_granted_it(admin_client, admin_user, member, role):
    """The only durable record of who conferred authority until a real audit trail
    exists — the first question asked in any access incident."""
    _post(admin_client, ASSIGN_URL, user_id=member.pk, role_id=role.pk)

    assert UserRole.objects.get(user=member, role=role).granted_by_id == admin_user.pk


def test_assign_is_idempotent(admin_client, member, role):
    """"Make sure alice is an analyst" must be runnable twice. 201 new, 200 already —
    both success, unlike create endpoints which 409 on a duplicate."""
    first = _post(admin_client, ASSIGN_URL, user_id=member.pk, role_id=role.pk)
    second = _post(admin_client, ASSIGN_URL, user_id=member.pk, role_id=role.pk)

    assert first.status_code == 201
    assert second.status_code == 200
    assert UserRole.objects.filter(user=member, role=role).count() == 1


def test_duplicate_assignment_is_blocked_by_the_database(member, role):
    """The service relies on the constraint rather than a preceding SELECT, so two
    concurrent assignments cannot both insert."""
    UserRole.objects.create(user=member, role=role)

    with pytest.raises(IntegrityError):
        with transaction.atomic():
            UserRole.objects.create(user=member, role=role)


def test_assign_rejects_a_retired_role(admin_client, member, role):
    """Assigning an inactive role would confer authority that is switched off."""
    role.is_active = False
    role.save(update_fields=["is_active", "updated_at"])

    response = _post(admin_client, ASSIGN_URL, user_id=member.pk, role_id=role.pk)

    assert response.status_code == 409
    assert response.json()["code"] == CODE_ROLE_INACTIVE
    assert not UserRole.objects.exists()


def test_assign_allows_an_inactive_user(admin_client, member, role):
    """Pre-provisioning is legitimate: the assignment grants nothing until the account
    is active anyway."""
    member.is_active = False
    member.save(update_fields=["is_active"])

    assert _post(admin_client, ASSIGN_URL,
                 user_id=member.pk, role_id=role.pk).status_code == 201


@pytest.mark.parametrize("body", [
    {"user_id": 999999, "role_id": 1},
    {"user_id": 1, "role_id": 999999},
])
def test_assign_404s_on_an_unknown_target(admin_client, member, role, body):
    body = {**body}
    if body["user_id"] == 1:
        body["user_id"] = member.pk
    if body["role_id"] == 1:
        body["role_id"] = role.pk

    assert _post(admin_client, ASSIGN_URL, **body).status_code == 404


def test_revoke_removes_the_edge(admin_client, member, role):
    _post(admin_client, ASSIGN_URL, user_id=member.pk, role_id=role.pk)

    response = _post(admin_client, REVOKE_URL, user_id=member.pk, role_id=role.pk)

    assert response.status_code == 200
    assert response.json()["data"]["removed"] is True
    assert not UserRole.objects.exists()


def test_revoke_is_idempotent_and_does_not_404(admin_client, member, role):
    """The desired end state — "does not hold this role" — is already true for a
    target that was never assigned, or that does not exist. A revoke script must not
    fail on exactly the rows it has nothing to do."""
    never = _post(admin_client, REVOKE_URL, user_id=member.pk, role_id=role.pk)
    missing = _post(admin_client, REVOKE_URL, user_id=999_999, role_id=999_999)

    assert never.status_code == 200
    assert never.json()["data"]["removed"] is False
    assert missing.status_code == 200


def test_deleting_a_user_removes_their_assignments(admin_client, member, role):
    """CASCADE: an assignment without its user is meaningless."""
    _post(admin_client, ASSIGN_URL, user_id=member.pk, role_id=role.pk)

    member.delete()

    assert not UserRole.objects.exists()


def test_a_held_role_cannot_be_deleted(member, role):
    """PROTECT: deleting a role that is still held must be blocked, not cascade."""
    from django.db.models import ProtectedError

    UserRole.objects.create(user=member, role=role)

    with pytest.raises(ProtectedError):
        role.delete()


def test_assignment_list_filters_both_ways(admin_client, member, role):
    """"What does this user hold" and "who holds this role" are the same endpoint."""
    other = get_user_model().objects.create_user(username="bob", password=ADMIN_PASSWORD)
    other_role = Role.objects.create(name="Auditor")
    UserRole.objects.create(user=member, role=role)
    UserRole.objects.create(user=other, role=role)
    UserRole.objects.create(user=member, role=other_role)

    by_user = _post(admin_client, ASSIGN_LIST_URL, user_id=member.pk).json()["data"]
    by_role = _post(admin_client, ASSIGN_LIST_URL, role_id=role.pk).json()["data"]
    everything = _post(admin_client, ASSIGN_LIST_URL).json()["data"]

    assert by_user["pagination"]["total"] == 2
    assert by_role["pagination"]["total"] == 2
    assert everything["pagination"]["total"] == 3


# ---------------------------------------------------------------------------
# Permission grants
# ---------------------------------------------------------------------------


def test_grant_creates_a_decision(admin_client, role, permission, resource):
    response = _post(admin_client, GRANT_URL, role_id=role.pk,
                     permission_id=permission.pk, resource_path=resource)

    assert response.status_code == 201
    data = response.json()["data"]
    assert data["resource_path"] == resource
    assert data["effect"] == Effect.ALLOW
    assert data["resource_exists"] is True


def test_grant_defaults_to_allow(admin_client, role, permission, resource):
    """An explicit DENY should be a deliberate act, not something a typo produces."""
    _post(admin_client, GRANT_URL, role_id=role.pk,
          permission_id=permission.pk, resource_path=resource)

    assert RolePermission.objects.get().effect == Effect.ALLOW


def test_regranting_the_opposite_effect_updates_rather_than_duplicates(
        admin_client, role, permission, resource):
    """THE correctness test for the grant table.

    Two rows disagreeing about one (role, permission, resource) would make the
    authorization outcome depend on row order.
    """
    _post(admin_client, GRANT_URL, role_id=role.pk, permission_id=permission.pk,
          resource_path=resource, effect=Effect.ALLOW)

    response = _post(admin_client, GRANT_URL, role_id=role.pk,
                     permission_id=permission.pk, resource_path=resource,
                     effect=Effect.DENY)

    assert response.status_code == 200          # updated, not created
    assert RolePermission.objects.count() == 1
    assert RolePermission.objects.get().effect == Effect.DENY


def test_the_unique_key_excludes_effect(role, permission, resource):
    """Enforced by the database, so no code path can create the contradiction."""
    RolePermission.objects.create(role=role, permission=permission,
                                  resource_path=resource, effect=Effect.ALLOW)

    with pytest.raises(IntegrityError):
        with transaction.atomic():
            RolePermission.objects.create(role=role, permission=permission,
                                          resource_path=resource, effect=Effect.DENY)


def test_a_global_grant_uses_an_empty_path(admin_client, role):
    """``user.manage`` applies to the platform, not to a table."""
    manage = Permission.objects.get(code="user.manage")

    response = _post(admin_client, GRANT_URL, role_id=role.pk, permission_id=manage.pk)

    assert response.status_code == 201
    data = response.json()["data"]
    assert data["resource_path"] == ""
    assert data["resource_exists"] is True      # a global grant targets nothing


def test_grant_canonicalises_the_resource_path(admin_client, role, permission, resource):
    """Otherwise one resource is grantable under two strings, only one of which the
    resolver would ever match."""
    response = _post(admin_client, GRANT_URL, role_id=role.pk,
                     permission_id=permission.pk,
                     resource_path="DB:CRM_Postgres:Employee")

    assert response.json()["data"]["resource_path"] == "db:crm_postgres:employee"
    assert RolePermission.objects.count() == 1


def test_grant_rejects_a_malformed_resource_path(admin_client, role, permission):
    response = _post(admin_client, GRANT_URL, role_id=role.pk,
                     permission_id=permission.pk, resource_path="db::bad")

    assert response.status_code == 400
    assert "resource_path" in response.json()["errors"]


def test_grant_allows_a_path_the_catalog_does_not_know_yet(
        admin_client, role, permission):
    """Pre-provisioning a source that is still ingesting is legitimate — but the
    response flags it so a typo does not hide until someone wonders why access never
    worked."""
    response = _post(admin_client, GRANT_URL, role_id=role.pk,
                     permission_id=permission.pk,
                     resource_path="db:not_discovered_yet:employee")

    assert response.status_code == 201
    assert response.json()["data"]["resource_exists"] is False


def test_grant_rejects_a_retired_role_or_disabled_permission(
        admin_client, role, permission, resource):
    role.is_active = False
    role.save(update_fields=["is_active", "updated_at"])
    assert _post(admin_client, GRANT_URL, role_id=role.pk,
                 permission_id=permission.pk, resource_path=resource).status_code == 409

    role.is_active = True
    role.save(update_fields=["is_active", "updated_at"])
    Permission.objects.filter(pk=permission.pk).update(is_active=False)

    response = _post(admin_client, GRANT_URL, role_id=role.pk,
                     permission_id=permission.pk, resource_path=resource)
    assert response.status_code == 409
    assert response.json()["code"] == CODE_PERMISSION_INACTIVE


def test_grant_revoke_is_idempotent(admin_client, role, permission, resource):
    _post(admin_client, GRANT_URL, role_id=role.pk,
          permission_id=permission.pk, resource_path=resource)

    first = _post(admin_client, GRANT_REVOKE_URL, role_id=role.pk,
                  permission_id=permission.pk, resource_path=resource)
    second = _post(admin_client, GRANT_REVOKE_URL, role_id=role.pk,
                   permission_id=permission.pk, resource_path=resource)

    assert first.json()["data"]["removed"] is True
    assert second.status_code == 200
    assert second.json()["data"]["removed"] is False


def test_revoking_a_deny_does_not_create_an_allow(admin_client, role, permission, resource):
    """With nothing matching, ADR §3.5 default-deny applies."""
    _post(admin_client, GRANT_URL, role_id=role.pk, permission_id=permission.pk,
          resource_path=resource, effect=Effect.DENY)

    _post(admin_client, GRANT_REVOKE_URL, role_id=role.pk,
          permission_id=permission.pk, resource_path=resource)

    assert not RolePermission.objects.exists()


def test_deleting_a_role_removes_its_grants(role, permission, resource, member):
    """CASCADE: a role's grants are part of the role."""
    RolePermission.objects.create(role=role, permission=permission,
                                  resource_path=resource)

    role.delete()

    assert not RolePermission.objects.exists()


def test_a_granted_permission_cannot_be_deleted(role, permission, resource):
    """PROTECT: the catalogue is seeded and never deleted; an attempt must fail loudly
    rather than silently revoking every grant of that capability."""
    from django.db.models import ProtectedError

    RolePermission.objects.create(role=role, permission=permission,
                                  resource_path=resource)

    with pytest.raises(ProtectedError):
        permission.delete()


def test_grant_list_filters_and_reports_resource_existence(
        admin_client, role, permission, resource):
    _post(admin_client, GRANT_URL, role_id=role.pk,
          permission_id=permission.pk, resource_path=resource)
    _post(admin_client, GRANT_URL, role_id=role.pk,
          permission_id=permission.pk, resource_path="db:ghost:table")

    body = _post(admin_client, GRANT_LIST_URL, role_id=role.pk).json()["data"]

    assert body["pagination"]["total"] == 2
    by_path = {g["resource_path"]: g["resource_exists"] for g in body["grants"]}
    assert by_path[resource] is True
    assert by_path["db:ghost:table"] is False


def test_grant_list_costs_a_bounded_number_of_queries(admin_client, role, permission):
    """``resource_exists`` must be one query for the whole page, not one per row."""
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    for i in range(10):
        RolePermission.objects.create(role=role, permission=permission,
                                      resource_path=f"db:src:table{i}")

    with CaptureQueriesContext(connection) as ctx:
        service = RolePermissionService()
        grants, _ = service.list_grants(page=1, page_size=50)
        service.known_resource_paths([g.resource_path for g in grants])

    # COUNT + page + one existence lookup. Not 10 + n.
    assert len(ctx.captured_queries) == 3


# ---------------------------------------------------------------------------
# Service layer, access control, wiring
# ---------------------------------------------------------------------------


def test_service_raises_typed_errors(member, role):
    role.is_active = False
    role.save(update_fields=["is_active", "updated_at"])

    with pytest.raises(RoleInactive):
        UserRoleService().assign(user_id=member.pk, role_id=role.pk)


def test_service_requires_keyword_arguments(member, role):
    with pytest.raises(TypeError):
        UserRoleService().assign(member.pk, role.pk)


ALL_URLS = [ASSIGN_URL, REVOKE_URL, ASSIGN_LIST_URL,
            GRANT_URL, GRANT_REVOKE_URL, GRANT_LIST_URL]


@pytest.mark.parametrize("url", ALL_URLS)
def test_anonymous_and_non_staff_are_rejected(url, member):
    anonymous = Client().post(url, {}, content_type="application/json")
    plain = Client()
    plain.force_login(member)

    assert anonymous.status_code == 401
    assert plain.post(url, {}, content_type="application/json").status_code == 403


@pytest.mark.parametrize("url", ALL_URLS)
def test_endpoints_are_post_only(admin_client, url):
    assert admin_client.get(url).status_code == 405
    assert admin_client.delete(url).status_code == 405


def test_routes_are_wired():
    from django.urls import resolve

    from apps.access_management.views import (
        RolePermissionGrantView,
        RolePermissionListView,
        RolePermissionRevokeView,
        UserRoleAssignView,
        UserRoleListView,
        UserRoleRevokeView,
    )

    assert resolve(ASSIGN_URL).func.view_class is UserRoleAssignView
    assert resolve(REVOKE_URL).func.view_class is UserRoleRevokeView
    assert resolve(ASSIGN_LIST_URL).func.view_class is UserRoleListView
    assert resolve(GRANT_URL).func.view_class is RolePermissionGrantView
    assert resolve(GRANT_REVOKE_URL).func.view_class is RolePermissionRevokeView
    assert resolve(GRANT_LIST_URL).func.view_class is RolePermissionListView


def test_nothing_is_enforced_by_these_rows_yet(admin_client, member, role, permission):
    """States the honest limit of this phase, so nobody mistakes a populated grant
    table for working authorization: the query endpoint is unaffected."""
    from django.urls import resolve

    UserRole.objects.create(user=member, role=role)
    RolePermission.objects.create(role=role, permission=permission,
                                  resource_path="db:crm_postgres:employee")

    from rest_framework.permissions import AllowAny

    query_view = resolve("/api/v1/query").func.view_class
    assert AllowAny in query_view.permission_classes
