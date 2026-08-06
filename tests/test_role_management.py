"""Coverage for apps/access_management — role administration.

  POST /api/v1/roles/create
  POST /api/v1/roles/detail
  POST /api/v1/roles/list
  POST /api/v1/roles/update

Self-contained in the same style as ``tests/test_user_management.py``: Django is
configured in-process and a throwaway sqlite test database is built here, rather than
adding a repo-wide ``pytest.ini`` that would change how every existing test module
bootstraps. The developer's ``db.sqlite3`` is never touched.

These tests need a real database: role-name uniqueness is enforced by a
``UniqueConstraint(Lower("name"))``, so a test that mocked the ORM away would verify
nothing about the property that matters.

Run from repo root: ``pytest tests/test_role_management.py``
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

from apps.access_management.models import Role  # noqa: E402
from apps.access_management.services import (  # noqa: E402
    CODE_ROLE_NAME_TAKEN,
    CODE_ROLE_NOT_FOUND,
    RoleNameTaken,
    RoleNotFound,
    RoleService,
)

CREATE_URL = "/api/v1/roles/create"
DETAIL_URL = "/api/v1/roles/detail"
LIST_URL = "/api/v1/roles/list"
UPDATE_URL = "/api/v1/roles/update"

ADMIN_PASSWORD = "admin-correct-horse-staple"

#: The one role representation every endpoint returns. Asserted as an exact set so a
#: future field addition is a deliberate contract change, not an accidental leak.
PUBLIC_FIELDS = {"role_id", "name", "description", "is_active",
                 "created_at", "updated_at"}

# Production hashers cost ~310ms per password by design; these tests only need an
# admin to authenticate as. See tests/test_user_management.py for the full rationale.
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

    # apps.substrate's 0002 migration is raw Postgres/pgvector DDL that sqlite cannot
    # parse; its models are managed=False mirrors. access_management's own migrations
    # DO run — the Role table and its constraint are what is under test.
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
        username="root", password=ADMIN_PASSWORD, email="root@example.com",
        is_staff=True)


@pytest.fixture
def plain_user():
    return get_user_model().objects.create_user(
        username="nobody", password=ADMIN_PASSWORD, email="nobody@example.com")


@pytest.fixture
def admin_client(admin_user):
    client = Client()
    client.force_login(admin_user)
    return client


def _create(client, **body):
    payload = {"name": "Data Analyst", "description": "Reads dashboards.", **body}
    return client.post(CREATE_URL, payload, content_type="application/json")


def _detail(client, **body):
    return client.post(DETAIL_URL, body, content_type="application/json")


def _list(client, **body):
    return client.post(LIST_URL, body, content_type="application/json")


def _update(client, **body):
    return client.post(UPDATE_URL, body, content_type="application/json")


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def test_admin_creates_a_role(admin_client):
    response = _create(admin_client)

    assert response.status_code == 201
    body = response.json()
    assert body["message"] == "Role created successfully."
    data = body["data"]
    assert data["name"] == "Data Analyst"
    assert data["description"] == "Reads dashboards."
    assert data["is_active"] is True
    assert isinstance(data["role_id"], int)
    assert set(data) == PUBLIC_FIELDS


def test_created_role_is_persisted(admin_client):
    role_id = _create(admin_client).json()["data"]["role_id"]

    role = Role.objects.get(pk=role_id)
    assert role.name == "Data Analyst"
    assert role.is_active is True
    assert role.created_at is not None


def test_description_is_optional(admin_client):
    response = admin_client.post(CREATE_URL, {"name": "Minimal"},
                                 content_type="application/json")

    assert response.status_code == 201
    assert response.json()["data"]["description"] == ""


def test_role_name_is_trimmed(admin_client):
    """Otherwise "  Admin  " and "Admin" would be two roles that look identical to
    every human reading the list."""
    response = _create(admin_client, name="  Spaced Out  ")

    assert response.json()["data"]["name"] == "Spaced Out"


# ---------------------------------------------------------------------------
# Create — uniqueness
# ---------------------------------------------------------------------------


def test_duplicate_role_name_is_a_conflict(admin_client):
    _create(admin_client)

    response = _create(admin_client)

    assert response.status_code == 409
    assert response.json()["code"] == CODE_ROLE_NAME_TAKEN
    assert Role.objects.filter(name="Data Analyst").count() == 1


def test_duplicate_name_differing_only_in_case_is_a_conflict(admin_client):
    """"Admin" and "admin" must be the same role — otherwise grants become ambiguous
    to every human who reads them."""
    _create(admin_client, name="Admin")

    response = _create(admin_client, name="aDMIN")

    assert response.status_code == 409
    assert response.json()["code"] == CODE_ROLE_NAME_TAKEN


def test_uniqueness_is_enforced_by_the_database(admin_client):
    """Enforcement must be a constraint, not application logic: a direct ORM write
    bypassing the service must still be refused. This is what makes it race-proof —
    two concurrent creates cannot both win."""
    Role.objects.create(name="Auditor")

    with pytest.raises(IntegrityError):
        with transaction.atomic():
            Role.objects.create(name="AUDITOR")


def test_a_retired_role_still_holds_its_name(admin_client):
    """Deactivation is not deletion: the name stays taken, so a new role cannot
    silently shadow a retired one in the audit history."""
    role_id = _create(admin_client).json()["data"]["role_id"]
    _update(admin_client, role_id=role_id, is_active=False)

    response = _create(admin_client)

    assert response.status_code == 409


# ---------------------------------------------------------------------------
# Create — validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body", [{}, {"description": "no name"}])
def test_create_requires_a_name(admin_client, body):
    response = admin_client.post(CREATE_URL, body, content_type="application/json")

    assert response.status_code == 400
    assert "name" in response.json()["errors"]


@pytest.mark.parametrize("name", ["", "   ", "\t\n"])
def test_create_rejects_a_blank_name(admin_client, name):
    """Whitespace-only would survive ``allow_blank=False`` and then be trimmed to an
    empty string that no constraint forbids."""
    assert _create(admin_client, name=name).status_code == 400


def test_create_rejects_an_over_long_name(admin_client):
    assert _create(admin_client, name="r" * 151).status_code == 400


@pytest.mark.parametrize("field", ["id", "role_id", "created_at", "updated_at"])
def test_create_rejects_server_owned_fields(admin_client, field):
    """Rejected, not ignored — a client that thinks it set an id must be told."""
    response = _create(admin_client, **{field: 1})

    assert response.status_code == 400
    assert field in response.json()["errors"]


def test_create_cannot_set_is_active(admin_client):
    """"Create a role that is already retired" is not a thing an administrator means;
    ``is_active`` is not on the create allowlist, so submitting it is an error."""
    response = _create(admin_client, is_active=False)

    assert response.status_code == 400


def test_create_rejects_a_non_object_body(admin_client):
    response = admin_client.post(CREATE_URL, [1, 2, 3], content_type="application/json")

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------


def test_detail_returns_one_role(admin_client):
    created = _create(admin_client).json()["data"]

    response = _detail(admin_client, role_id=created["role_id"])

    assert response.status_code == 200
    assert response.json()["message"] == "Role retrieved successfully."
    assert response.json()["data"] == created


def test_detail_of_a_missing_role_is_404(admin_client):
    response = _detail(admin_client, role_id=999_999)

    assert response.status_code == 404
    assert response.json()["code"] == CODE_ROLE_NOT_FOUND


@pytest.mark.parametrize("body", [{}, {"role_id": 0}, {"role_id": -1},
                                 {"role_id": "abc"}])
def test_detail_rejects_a_malformed_body(admin_client, body):
    """A nonsensical id is a 400 (client bug), not a 404 — the two stay diagnosable
    apart."""
    assert _detail(admin_client, **body).status_code == 400


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


@pytest.fixture
def population():
    """Nine roles with predictable names; every third one retired."""
    for i in range(9):
        Role.objects.create(
            name=f"role{i:02d}", description=f"description for {i:02d}",
            is_active=(i % 3 != 0))
    return Role.objects.count()


def test_list_returns_a_page_with_totals(admin_client, population):
    response = _list(admin_client, page=1, page_size=4)

    assert response.status_code == 200
    assert response.json()["message"] == "Roles retrieved successfully."
    body = response.json()["data"]
    assert len(body["roles"]) == 4
    pagination = body["pagination"]
    assert pagination["total"] == population
    assert pagination["total_pages"] == (population + 3) // 4
    assert pagination["has_next"] is True
    assert pagination["has_previous"] is False


def test_list_uses_the_same_projection_as_create(admin_client, population):
    row = _list(admin_client, page_size=1).json()["data"]["roles"][0]

    assert set(row) == PUBLIC_FIELDS


def test_list_paginates_without_repeating_or_dropping_rows(admin_client, population):
    """Deterministic ordering: paging through must visit every role exactly once."""
    seen = []
    page = 1
    while True:
        body = _list(admin_client, page=page, page_size=3).json()["data"]
        seen.extend(r["role_id"] for r in body["roles"])
        if not body["pagination"]["has_next"]:
            break
        page += 1

    assert len(seen) == population
    assert len(set(seen)) == population


def test_list_search_matches_name_or_description(admin_client, population):
    by_name = _list(admin_client, search="role01").json()["data"]
    assert by_name["pagination"]["total"] == 1

    by_description = _list(admin_client, search="description for 02").json()["data"]
    assert by_description["pagination"]["total"] == 1
    assert by_description["roles"][0]["name"] == "role02"


def test_list_search_is_case_insensitive(admin_client, population):
    assert _list(admin_client, search="ROLE03").json()["data"]["pagination"]["total"] == 1


def test_list_filters_by_is_active(admin_client, population):
    retired = _list(admin_client, is_active=False).json()["data"]

    assert retired["pagination"]["total"] == 3        # every third role
    assert all(r["is_active"] is False for r in retired["roles"])


def test_list_is_active_omitted_means_no_filter(admin_client, population):
    """Tri-state: absent must mean "all", not "False"."""
    assert _list(admin_client).json()["data"]["pagination"]["total"] == population


def test_list_ordering_is_honoured(admin_client, population):
    ascending = [r["name"] for r in
                 _list(admin_client, ordering="name", page_size=100).json()["data"]["roles"]]
    descending = [r["name"] for r in
                  _list(admin_client, ordering="-name", page_size=100).json()["data"]["roles"]]

    assert ascending == sorted(ascending)
    assert descending == list(reversed(ascending))


def test_list_rejects_an_unknown_ordering_field(admin_client):
    """order_by() with an arbitrary string can traverse relations or 500."""
    response = _list(admin_client, ordering="description")

    assert response.status_code == 400
    assert "ordering" in response.json()["errors"]


def test_list_caps_page_size(admin_client):
    response = _list(admin_client, page_size=10_000)

    assert response.status_code == 400
    assert "page_size" in response.json()["errors"]


def test_list_page_beyond_the_end_is_empty_not_an_error(admin_client, population):
    body = _list(admin_client, page=999).json()["data"]

    assert body["roles"] == []
    assert body["pagination"]["has_next"] is False


def test_list_costs_two_queries_regardless_of_page_size(admin_client, population):
    """One COUNT plus one page fetch — no N+1 as the page grows."""
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    with CaptureQueriesContext(connection) as ctx:
        RoleService().list_roles(page=1, page_size=100)

    assert len(ctx.captured_queries) == 2


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


@pytest.fixture
def role_id(admin_client):
    return _create(admin_client).json()["data"]["role_id"]


def test_update_changes_fields(admin_client, role_id):
    response = _update(admin_client, role_id=role_id, name="Senior Analyst",
                       description="Reads and exports.")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["name"] == "Senior Analyst"
    assert data["description"] == "Reads and exports."


def test_update_is_partial(admin_client, role_id):
    """Only what is sent changes — an omitted field must not be blanked."""
    _update(admin_client, role_id=role_id, description="Only this.")

    role = Role.objects.get(pk=role_id)
    assert role.description == "Only this."
    assert role.name == "Data Analyst"        # untouched


def test_update_refreshes_updated_at(admin_client, role_id):
    """``auto_now`` fields do NOT update when ``update_fields`` is passed unless they
    are named in it. Without this the timestamp would silently stop tracking edits."""
    before = Role.objects.get(pk=role_id).updated_at

    _update(admin_client, role_id=role_id, description="Changed.")

    assert Role.objects.get(pk=role_id).updated_at > before


def test_update_retires_a_role(admin_client, role_id):
    """This is the delete: there is no delete endpoint."""
    response = _update(admin_client, role_id=role_id, is_active=False)

    assert response.status_code == 200
    assert response.json()["data"]["is_active"] is False
    assert Role.objects.get(pk=role_id).is_active is False


def test_a_retired_role_can_be_reactivated(admin_client, role_id):
    _update(admin_client, role_id=role_id, is_active=False)

    assert _update(admin_client, role_id=role_id, is_active=True).status_code == 200
    assert Role.objects.get(pk=role_id).is_active is True


def test_update_of_a_missing_role_is_404(admin_client):
    response = _update(admin_client, role_id=999_999, name="Ghost")

    assert response.status_code == 404
    assert response.json()["code"] == CODE_ROLE_NOT_FOUND


def test_update_to_a_taken_name_is_409(admin_client, role_id):
    Role.objects.create(name="Occupant")

    response = _update(admin_client, role_id=role_id, name="Occupant")

    assert response.status_code == 409
    assert response.json()["code"] == CODE_ROLE_NAME_TAKEN


def test_update_to_a_taken_name_differing_only_in_case_is_409(admin_client, role_id):
    Role.objects.create(name="Occupant")

    assert _update(admin_client, role_id=role_id, name="OCCUPANT").status_code == 409


def test_update_can_resubmit_the_roles_own_name(admin_client, role_id):
    """The row must not be treated as conflicting with itself — that would make any
    form that submits all fields unusable."""
    response = _update(admin_client, role_id=role_id, name="Data Analyst",
                       description="Same name, new description.")

    assert response.status_code == 200


def test_update_with_no_changes_is_rejected(admin_client, role_id):
    assert _update(admin_client, role_id=role_id).status_code == 400


def test_update_requires_a_role_id(admin_client):
    response = _update(admin_client, name="Nameless")

    assert response.status_code == 400
    assert "role_id" in response.json()["errors"]


@pytest.mark.parametrize("field", ["id", "created_at", "updated_at", "unknown_field"])
def test_update_rejects_unknown_and_server_owned_fields(admin_client, role_id, field):
    response = _update(admin_client, role_id=role_id, **{field: 1})

    assert response.status_code == 400
    assert field in response.json()["errors"]


@pytest.mark.parametrize("name", ["", "   ", "r" * 151])
def test_update_validates_the_new_name(admin_client, role_id, name):
    assert _update(admin_client, role_id=role_id, name=name).status_code == 400


def test_update_writes_only_the_submitted_columns(admin_client, role_id):
    """``update_fields`` is what stops a partial update clobbering a column another
    request changed concurrently. ``updated_at`` rides along by design."""
    from unittest import mock

    real_save = Role.save
    captured = {}

    def capture(self, *args, **kwargs):
        captured["update_fields"] = kwargs.get("update_fields")
        return real_save(self, *args, **kwargs)

    with mock.patch.object(Role, "save", capture):
        _update(admin_client, role_id=role_id, description="Changed.")

    assert captured["update_fields"] == ["description", "updated_at"]


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url,body", [
    (CREATE_URL, {"name": "Sneaky"}),
    (DETAIL_URL, {"role_id": 1}),
    (LIST_URL, {}),
    (UPDATE_URL, {"role_id": 1, "name": "Sneaky"}),
])
def test_anonymous_requests_are_rejected(url, body):
    response = Client().post(url, body, content_type="application/json")

    assert response.status_code == 401
    assert not Role.objects.filter(name="Sneaky").exists()


@pytest.mark.parametrize("url,body", [
    (CREATE_URL, {"name": "Sneaky"}),
    (DETAIL_URL, {"role_id": 1}),
    (LIST_URL, {}),
    (UPDATE_URL, {"role_id": 1, "name": "Sneaky"}),
])
def test_non_staff_requests_are_forbidden(plain_user, url, body):
    client = Client()
    client.force_login(plain_user)

    response = client.post(url, body, content_type="application/json")

    assert response.status_code == 403
    assert not Role.objects.filter(name="Sneaky").exists()


# ---------------------------------------------------------------------------
# Service layer
# ---------------------------------------------------------------------------


def test_service_raises_typed_errors():
    service = RoleService()
    service.create_role(name="Taken")

    with pytest.raises(RoleNameTaken):
        service.create_role(name="TAKEN")
    with pytest.raises(RoleNotFound):
        service.get_role(999_999)
    with pytest.raises(RoleNotFound):
        service.update_role(999_999, name="Ghost")


def test_service_requires_keyword_arguments():
    """Two same-typed strings: a positional call that transposed name and description
    would otherwise be accepted silently."""
    with pytest.raises(TypeError):
        RoleService().create_role("Name", "Description")


def test_service_update_requires_at_least_one_field(role_id):
    with pytest.raises(ValueError):
        RoleService().update_role(role_id)


def test_an_unattributable_integrity_error_is_not_reported_as_a_conflict():
    """Reporting an unexplained constraint failure as "name already taken" would send
    an administrator chasing a role that does not exist."""
    from unittest import mock

    with mock.patch.object(Role.objects, "create",
                           side_effect=IntegrityError("CHECK constraint failed: other")):
        with pytest.raises(IntegrityError):
            RoleService().create_role(name="Whatever")


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_routes_are_wired():
    from django.urls import resolve

    from apps.access_management.views import (
        RoleCreateView,
        RoleDetailView,
        RoleListView,
        RoleUpdateView,
    )

    assert resolve(CREATE_URL).func.view_class is RoleCreateView
    assert resolve(DETAIL_URL).func.view_class is RoleDetailView
    assert resolve(LIST_URL).func.view_class is RoleListView
    assert resolve(UPDATE_URL).func.view_class is RoleUpdateView


def test_there_is_no_delete_endpoint():
    """Retirement is ``update {is_active: false}``. A delete route would need a
    "refuse if any user holds this role" guard that cannot exist until role
    assignment does."""
    from django.urls import Resolver404, resolve

    with pytest.raises(Resolver404):
        resolve("/api/v1/roles/delete")


@pytest.mark.parametrize("url", [CREATE_URL, DETAIL_URL, LIST_URL, UPDATE_URL])
def test_endpoints_are_post_only(admin_client, url):
    assert admin_client.get(url).status_code == 405
    assert admin_client.put(url).status_code == 405
    assert admin_client.patch(url).status_code == 405
    assert admin_client.delete(url).status_code == 405


def test_role_model_is_reachable_from_the_app_models_namespace():
    """Django requires it, and the package split must not have broken it."""
    from django.apps import apps as django_apps

    assert django_apps.get_model("access_management", "Role") is Role
    assert Role._meta.db_table == "access_management_role"


def test_role_reuses_the_shared_timestamp_base():
    """created_at/updated_at come from apps.core.models.TimeStampedModel rather than
    being re-declared — the abstraction already existed."""
    from apps.core.models import TimeStampedModel

    assert issubclass(Role, TimeStampedModel)
