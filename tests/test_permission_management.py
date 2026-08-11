"""Coverage for apps/access_management — the permission catalogue.

  POST /api/v1/permissions/list
  POST /api/v1/permissions/detail

Read-only by design: permissions are seeded by migration 0004 because only code can
enforce one. A large share of these tests exist to hold that line — the absence of a
write path is the feature, and a future "just add a create endpoint" should fail here.

Self-contained in the same style as the other access-management suites; the throwaway
sqlite database runs this app's real migrations, so the seeded rows under test are the
ones a deployment gets.

Run from repo root: ``pytest tests/test_permission_management.py``
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _setup_django():
    import config  # noqa: F401  — ensures config/ wins over veda_core/config.py

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
    import django

    django.setup()


_setup_django()

from django.contrib.auth import get_user_model  # noqa: E402
from django.db import IntegrityError, transaction  # noqa: E402
from django.test import Client, override_settings  # noqa: E402

from apps.access_management.models import Permission  # noqa: E402
from apps.access_management.services import (  # noqa: E402
    CODE_PERMISSION_NOT_FOUND,
    PermissionNotFound,
    PermissionService,
)

LIST_URL = "/api/v1/permissions/list"
DETAIL_URL = "/api/v1/permissions/detail"

ADMIN_PASSWORD = "admin-correct-horse-staple"

PUBLIC_FIELDS = {"permission_id", "code", "name", "description", "is_active",
                 "created_at", "updated_at"}

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
def admin_client():
    user = get_user_model().objects.create_user(
        username="root", password=ADMIN_PASSWORD, email="root@example.com",
        is_staff=True)
    client = Client()
    client.force_login(user)
    return client


@pytest.fixture
def plain_client():
    user = get_user_model().objects.create_user(
        username="nobody", password=ADMIN_PASSWORD, email="nobody@example.com")
    client = Client()
    client.force_login(user)
    return client


def _list(client, **body):
    return client.get(LIST_URL, body)


def _detail(client, **body):
    return client.get(DETAIL_URL, body)


# ---------------------------------------------------------------------------
# The seed — what a deployment actually gets
# ---------------------------------------------------------------------------


def test_migration_seeds_the_catalogue():
    """A deployment must come up with a usable vocabulary, not an empty table."""
    assert Permission.objects.exists()


def test_every_seeded_permission_is_grounded_in_a_real_code_path():
    """Pins the catalogue as an exact set.

    It cannot prove a code path exists — no test can — but it makes adding a
    permission a deliberate, reviewed act rather than a quiet append, which is the
    enforceable half of "nothing speculative is seeded". The new entry should arrive
    with the gate that checks it.
    """
    expected = {
        "query.execute", "data.read", "source.manage", "ingestion.run",
        "evaluation.run", "user.manage", "role.manage", "permission.read",
    }

    assert set(Permission.objects.values_list("code", flat=True)) == expected


def test_seeded_permissions_are_active_and_described():
    """A dotted code alone is unreadable on a role screen; every row needs its prose."""
    for permission in Permission.objects.all():
        assert permission.is_active is True
        assert permission.name.strip(), f"{permission.code} has no name"
        assert permission.description.strip(), f"{permission.code} has no description"


def test_the_seed_is_idempotent():
    """``update_or_create`` keyed on code — re-running must not duplicate rows, so a
    replayed or repeated migration is safe."""
    from importlib import import_module

    from django.apps import apps as django_apps

    migration = import_module(
        "apps.access_management.migrations.0004_seed_permissions")
    before = Permission.objects.count()

    migration.seed_permissions(django_apps, None)

    assert Permission.objects.count() == before


def test_the_seed_does_not_reactivate_a_disabled_permission():
    """``is_active`` is deliberately absent from the seed defaults: an operator who
    switched a capability off must not have that undone by the next deploy."""
    from importlib import import_module

    from django.apps import apps as django_apps

    migration = import_module(
        "apps.access_management.migrations.0004_seed_permissions")
    Permission.objects.filter(code="data.read").update(is_active=False)

    migration.seed_permissions(django_apps, None)

    assert Permission.objects.get(code="data.read").is_active is False


def test_permission_codes_are_unique_case_insensitively():
    """Enforced by the database, so a concurrent seed cannot produce two rows for the
    same capability — and "Data.Read" cannot shadow "data.read"."""
    with pytest.raises(IntegrityError):
        with transaction.atomic():
            Permission.objects.create(code="DATA.READ", name="Shadow")


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


def test_list_returns_the_catalogue(admin_client):
    response = _list(admin_client, page_size=100)

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Permissions retrieved successfully."
    data = body["data"]
    assert data["pagination"]["total"] == Permission.objects.count()
    assert set(data["permissions"][0]) == PUBLIC_FIELDS


def test_list_is_paginated(admin_client):
    body = _list(admin_client, page=1, page_size=3).json()["data"]

    assert len(body["permissions"]) == 3
    assert body["pagination"]["has_next"] is True


def test_list_search_matches_code_or_name(admin_client):
    by_code = _list(admin_client, search="ingestion").json()["data"]
    assert by_code["pagination"]["total"] == 1
    assert by_code["permissions"][0]["code"] == "ingestion.run"

    by_name = _list(admin_client, search="Manage users").json()["data"]
    assert by_name["pagination"]["total"] == 1
    assert by_name["permissions"][0]["code"] == "user.manage"


def test_list_search_ignores_description(admin_client):
    """Description is prose — matching it would make a search for a common word
    return most of the catalogue."""
    only_in_description = _list(admin_client, search="conversational").json()["data"]

    assert only_in_description["pagination"]["total"] == 0


def test_list_filters_by_is_active(admin_client):
    Permission.objects.filter(code="data.read").update(is_active=False)

    disabled = _list(admin_client, is_active=False).json()["data"]

    assert disabled["pagination"]["total"] == 1
    assert disabled["permissions"][0]["code"] == "data.read"


def test_list_is_active_omitted_means_no_filter(admin_client):
    Permission.objects.filter(code="data.read").update(is_active=False)

    assert _list(admin_client).json()["data"]["pagination"]["total"] == \
        Permission.objects.count()


def test_list_ordering_is_honoured(admin_client):
    ascending = [p["code"] for p in
                 _list(admin_client, ordering="code", page_size=100).json()["data"]["permissions"]]
    descending = [p["code"] for p in
                  _list(admin_client, ordering="-code", page_size=100).json()["data"]["permissions"]]

    assert ascending == sorted(ascending)
    assert descending == list(reversed(ascending))


def test_list_rejects_an_unknown_ordering_field(admin_client):
    response = _list(admin_client, ordering="description")

    assert response.status_code == 400
    assert "ordering" in response.json()["errors"]


def test_list_caps_page_size(admin_client):
    assert _list(admin_client, page_size=10_000).status_code == 400


def test_list_costs_two_queries(admin_client):
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    with CaptureQueriesContext(connection) as ctx:
        PermissionService().list_permissions(page=1, page_size=100)

    assert len(ctx.captured_queries) == 2


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------


def test_detail_returns_one_permission(admin_client):
    seeded = Permission.objects.get(code="role.manage")

    response = _detail(admin_client, permission_id=seeded.pk)

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["code"] == "role.manage"
    assert set(data) == PUBLIC_FIELDS


def test_detail_matches_the_list_projection(admin_client):
    """One representation — opening a row must return what the row showed."""
    listed = _list(admin_client, search="role.manage").json()["data"]["permissions"][0]

    detail = _detail(admin_client, permission_id=listed["permission_id"]).json()["data"]

    assert detail == listed


def test_detail_of_a_missing_permission_is_404(admin_client):
    response = _detail(admin_client, permission_id=999_999)

    assert response.status_code == 404
    assert response.json()["code"] == CODE_PERMISSION_NOT_FOUND


@pytest.mark.parametrize("body", [{}, {"permission_id": 0}, {"permission_id": -1},
                                 {"permission_id": "abc"}, {"id": 1}])
def test_detail_rejects_a_malformed_body(admin_client, body):
    assert _detail(admin_client, **body).status_code == 400


# ---------------------------------------------------------------------------
# Read-only — the absence of writes is the feature
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [
    "/api/v1/permissions/create",
    "/api/v1/permissions/update",
    "/api/v1/permissions/delete",
])
def test_there_are_no_write_endpoints(path):
    """Only code can enforce a permission, so the catalogue is seeded by migration.
    A write endpoint would let an administrator create authority nothing checks —
    this test is what makes adding one a deliberate, visible decision.
    """
    from django.urls import Resolver404, resolve

    with pytest.raises(Resolver404):
        resolve(path)


def test_the_service_exposes_no_write_methods():
    """Same guard one layer down: a write path must not appear on the service either,
    where a future view could quietly reach it."""
    forbidden = {"create_permission", "update_permission", "delete_permission",
                 "create", "update", "delete"}

    assert not forbidden.intersection(dir(PermissionService))


@pytest.mark.parametrize("url", [LIST_URL, DETAIL_URL])
def test_endpoints_are_get_only(admin_client, url):
    """Both permission endpoints are read-only and now GET-only (2026-08-09) —
    cacheable/bookmarkable/safely-retryable, unlike POST."""
    assert admin_client.get(url).status_code != 405
    assert admin_client.post(url, {}, content_type="application/json").status_code == 405
    assert admin_client.put(url).status_code == 405
    assert admin_client.patch(url).status_code == 405
    assert admin_client.delete(url).status_code == 405


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url,body", [(LIST_URL, {}), (DETAIL_URL, {"permission_id": 1})])
def test_anonymous_requests_are_rejected(url, body):
    assert Client().post(url, body, content_type="application/json").status_code == 401


@pytest.mark.parametrize("url,body", [(LIST_URL, {}), (DETAIL_URL, {"permission_id": 1})])
def test_non_staff_requests_are_forbidden(plain_client, url, body):
    assert plain_client.post(url, body, content_type="application/json").status_code == 403


# ---------------------------------------------------------------------------
# Service layer / wiring
# ---------------------------------------------------------------------------


def test_service_raises_a_typed_not_found():
    with pytest.raises(PermissionNotFound):
        PermissionService().get_permission(999_999)


def test_routes_are_wired():
    from django.urls import resolve

    from apps.access_management.views import PermissionDetailView, PermissionListView

    assert resolve(LIST_URL).func.view_class is PermissionListView
    assert resolve(DETAIL_URL).func.view_class is PermissionDetailView


def test_permission_model_is_distinct_from_djangos_own():
    """``django.contrib.auth.models.Permission`` is model-level and unrelated.
    Confusing the two would be an authorization bug, so the separation is pinned."""
    from django.contrib.auth.models import Permission as DjangoPermission

    assert Permission is not DjangoPermission
    assert Permission._meta.db_table == "access_management_permission"
    assert Permission._meta.app_label == "access_management"


def test_permission_reuses_the_shared_timestamp_base():
    from apps.core.models import TimeStampedModel

    assert issubclass(Permission, TimeStampedModel)


# ---------------------------------------------------------------------------
# User Role Assignment Integration
# ---------------------------------------------------------------------------


def test_user_creation_and_update_with_role_ids():
    from apps.access_management.models import Role, UserRole
    from apps.access_management.services import UserService

    role1 = Role.objects.create(name="Role 1")
    role2 = Role.objects.create(name="Role 2")

    service = UserService()
    user = service.create_user(
        username="roleuser", email="roleuser@example.com", password="Password123!",
        role_ids=[role1.id]
    )

    assert set(UserRole.objects.filter(user=user).values_list("role_id", flat=True)) == {role1.id}

    service.update_user(user.pk, role_ids=[role1.id, role2.id])
    assert set(UserRole.objects.filter(user=user).values_list("role_id", flat=True)) == {role1.id, role2.id}
