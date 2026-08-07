"""Coverage for apps/access_management — user management endpoints.

  POST /api/v1/users/create
  POST /api/v1/users/detail
  POST /api/v1/users/list
  POST /api/v1/users/update

Self-contained in the same style as ``tests/test_authentication.py``: Django is
configured in-process and a throwaway sqlite test database is built here, rather
than adding a repo-wide ``pytest.ini`` that would change how every existing test
module bootstraps. The developer's ``db.sqlite3`` is never touched.

These tests need a real database: uniqueness is enforced by database constraints
(``username`` by Django's own, ``email`` by migration 0001's partial index), so a
test that mocked the ORM away would verify nothing about the property that matters.

Run from repo root: ``pytest tests/test_user_management.py``
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

from apps.access_management.models import Role, UserRole  # noqa: E402
from apps.access_management.services import (  # noqa: E402
    CODE_EMAIL_TAKEN,
    CODE_USERNAME_TAKEN,
    EmailTaken,
    UsernameTaken,
    UserService,
)

CREATE_URL = "/api/v1/users/create"
DETAIL_URL = "/api/v1/users/detail"
LIST_URL = "/api/v1/users/list"
UPDATE_URL = "/api/v1/users/update"
DELETE_URL = "/api/v1/users/delete"

ADMIN_PASSWORD = "admin-correct-horse-staple"
# Satisfies PasswordComplexityValidator (upper/lower/digit/special) as well as the
# four Django stock validators — this goes through UserCreateSerializer, unlike
# ADMIN_PASSWORD above which is set directly via create_user() and never validated.
NEW_PASSWORD = "Fresh-Tapir-97!"

# The one user representation every endpoint in this app returns. Asserted as an
# exact set so a future field addition is a deliberate contract change, not a leak.
PUBLIC_FIELDS = {"user_id", "username", "email", "display_name", "is_active",
                 "is_staff", "date_joined", "last_login"}

#: Only ``users/list`` carries these — a per-row admin-table need (roles, and the
#: display-date fields the frontend's table renders directly), not part of the
#: shared create/detail/update representation. See UserListView.
LIST_EXTRA_FIELDS = {"roles", "created_at", "updated_at", "deleted_at"}

VALID_PAYLOAD = {
    "username": "jdoe",
    "email": "j.doe@example.com",
    "password": NEW_PASSWORD,
    "first_name": "Jane",
    "last_name": "Doe",
}

# Production hashers cost ~310ms per password by design (PBKDF2, 720k iterations).
# Fixtures here create ten users apiece, so the real hasher put test setup at 7-9s
# each and the module at ~3 minutes. A fast hasher is the standard Django answer —
# these tests are about API behaviour, not about the KDF's work factor.
#
# The two tests that DO assert hashing behaviour re-enable the real hashers
# explicitly (see PRODUCTION_HASHERS below), so the fast default cannot make them
# pass vacuously.
FAST_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
PRODUCTION_HASHERS = ["django.contrib.auth.hashers.PBKDF2PasswordHasher"]

_TEST_OVERRIDES = dict(
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
    PASSWORD_HASHERS=FAST_HASHERS,
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
    # parse; its models are managed=False mirrors. Bypassing that app's migrations
    # lets the throwaway database build. access_management's own migration DOES run —
    # the email index is the thing under test.
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
    """Session-authenticated staff client.

    Session auth (rather than a JWT) because ``VEDA_JWT_AUTH`` is default-off, so the
    JWT authentication class is not installed — and this endpoint must work under the
    authenticators the deployment actually has today, whichever they are.
    """
    client = Client()
    client.force_login(admin_user)
    return client


def _create(client, **overrides):
    payload = {**VALID_PAYLOAD, **overrides}
    return client.post(CREATE_URL, payload, content_type="application/json")


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


def test_admin_creates_a_user(admin_client):
    response = _create(admin_client)

    assert response.status_code == 201
    body = response.json()
    assert body["status_code"] == 201
    assert body["message"] == "User created successfully."
    data = body["data"]
    assert data["username"] == "jdoe"
    assert data["email"] == "j.doe@example.com"
    assert data["display_name"] == "Jane"
    assert data["is_active"] is True
    assert isinstance(data["user_id"], int)


@override_settings(PASSWORD_HASHERS=PRODUCTION_HASHERS)
def test_created_user_is_persisted_with_a_usable_password(admin_client):
    """The password must be hashed with the project's real hasher and actually work.

    Runs under PRODUCTION_HASHERS deliberately: under the module's fast test hasher
    the ``pbkdf2_`` assertion would be vacuous, and this is the one place that
    verifies ``create_user`` is not storing something unusable.
    """
    _create(admin_client)

    user = get_user_model().objects.get(username="jdoe")
    assert user.check_password(NEW_PASSWORD)
    assert user.password != NEW_PASSWORD          # stored hashed, not plaintext
    assert user.password.startswith("pbkdf2_")


def test_created_user_is_unprivileged(admin_client):
    """New users get no privileges. Granting them is role assignment — a later
    phase — so a fresh account must never arrive with staff rights."""
    _create(admin_client)

    user = get_user_model().objects.get(username="jdoe")
    assert user.is_staff is False
    assert user.is_superuser is False
    assert user.is_active is True


def test_response_never_exposes_the_password_or_flags(admin_client):
    body = _create(admin_client).json()

    serialized = str(body)
    assert NEW_PASSWORD not in serialized
    assert "pbkdf2" not in serialized
    assert set(body["data"]) == PUBLIC_FIELDS


def test_optional_names_may_be_omitted(admin_client):
    payload = {"username": "minimal", "email": "min@example.com", "password": NEW_PASSWORD}
    response = admin_client.post(CREATE_URL, payload, content_type="application/json")

    assert response.status_code == 201
    # display_name falls back to the username, matching what login returns.
    assert response.json()["data"]["display_name"] == "minimal"


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------


def test_anonymous_request_is_rejected():
    response = _create(Client())

    assert response.status_code == 401
    assert not get_user_model().objects.filter(username="jdoe").exists()


def test_authenticated_non_staff_request_is_forbidden(plain_user):
    client = Client()
    client.force_login(plain_user)

    response = _create(client)

    assert response.status_code == 403
    assert not get_user_model().objects.filter(username="jdoe").exists()


def test_privilege_escalation_via_payload_is_rejected(admin_client):
    """A caller must not be able to mint an admin. Rejected loudly rather than
    silently ignored, so a client cannot believe it created a superuser."""
    response = _create(admin_client, is_staff=True, is_superuser=True)

    assert response.status_code == 400
    errors = response.json()["errors"]
    assert "is_staff" in errors and "is_superuser" in errors
    assert not get_user_model().objects.filter(username="jdoe").exists()


@pytest.mark.parametrize("field", ["is_active", "groups", "user_permissions", "id"])
def test_other_privileged_fields_are_rejected(admin_client, field):
    response = _create(admin_client, **{field: 1})

    assert response.status_code == 400
    assert field in response.json()["errors"]


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------


def test_duplicate_username_is_a_conflict(admin_client):
    _create(admin_client)

    response = _create(admin_client, email="different@example.com")

    assert response.status_code == 409
    assert response.json()["code"] == CODE_USERNAME_TAKEN
    assert get_user_model().objects.filter(username="jdoe").count() == 1


def test_duplicate_email_is_a_conflict(admin_client):
    _create(admin_client)

    response = _create(admin_client, username="someone-else")

    assert response.status_code == 409
    assert response.json()["code"] == CODE_EMAIL_TAKEN
    assert not get_user_model().objects.filter(username="someone-else").exists()


def test_duplicate_email_differing_only_in_case_is_a_conflict(admin_client):
    """The index is on LOWER(email), so 'J.Doe@EXAMPLE.com' must collide with
    'j.doe@example.com' — otherwise 'unique email' is trivially bypassable."""
    _create(admin_client)

    response = _create(admin_client, username="other", email="J.Doe@EXAMPLE.com")

    assert response.status_code == 409
    assert response.json()["code"] == CODE_EMAIL_TAKEN


def test_blank_emails_do_not_collide():
    """The index is partial (WHERE email <> ''), so accounts without an email must
    still be creatable and coexist.

    This is not hypothetical: chat migration 0002 seeds an ``admin`` account with a
    blank email, so a non-partial unique index would have made the migration
    unrunnable the moment a second email-less account appeared.
    """
    User = get_user_model()
    before = User.objects.filter(email="").count()
    assert before >= 1, "the seeded admin should already have a blank email"

    User.objects.create_user(username="blank-one", password=ADMIN_PASSWORD, email="")
    User.objects.create_user(username="blank-two", password=ADMIN_PASSWORD, email="")

    assert User.objects.filter(email="").count() == before + 2


def test_the_email_index_exists_and_is_enforced_by_the_database():
    """Enforcement must be a constraint, not application logic: a direct ORM write
    bypassing the service must still be refused. This is what makes the uniqueness
    race-proof."""
    User = get_user_model()
    User.objects.create_user(username="first", password=ADMIN_PASSWORD, email="dup@example.com")

    with pytest.raises(IntegrityError):
        with transaction.atomic():
            User.objects.create_user(
                username="second", password=ADMIN_PASSWORD, email="DUP@example.com")


def test_migration_refuses_to_run_over_pre_existing_duplicate_emails():
    """The migration's safety net, which is the one path a real deployment might hit.

    A production database may already hold two accounts sharing an address. The
    migration must then fail with a message naming them, rather than an opaque
    "UNIQUE constraint failed" from the index build — and it must not try to guess
    which account should keep the address.

    The index is dropped for the duration so the offending rows can be created at
    all; the surrounding fixture rolls everything back.
    """
    from importlib import import_module

    from django.apps import apps as django_apps
    from django.db import connection

    # import_module, not `from ... import`: the module name starts with a digit.
    migration = import_module(
        "apps.access_management.migrations.0001_user_email_unique_index")

    User = get_user_model()
    with connection.cursor() as cursor:
        cursor.execute(f'DROP INDEX IF EXISTS "{migration.INDEX_NAME}"')
    User.objects.create_user(username="dup-a", password=ADMIN_PASSWORD, email="Shared@example.com")
    User.objects.create_user(username="dup-b", password=ADMIN_PASSWORD, email="shared@EXAMPLE.com")

    with pytest.raises(RuntimeError) as raised:
        migration.assert_no_duplicate_emails(django_apps, connection.schema_editor())

    message = str(raised.value)
    assert "shared@example.com" in message      # normalized, and named for the operator
    assert "2 accounts" in message


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", [
    {},
    {"username": "u"},
    {"email": "a@b.com", "password": NEW_PASSWORD},
    {"username": "u", "password": NEW_PASSWORD},
    {"username": "u", "email": "a@b.com"},
])
def test_missing_required_fields_are_rejected(admin_client, payload):
    response = admin_client.post(CREATE_URL, payload, content_type="application/json")

    assert response.status_code == 400
    assert response.json()["message"] == "Invalid request data."
    assert response.json()["errors"]


@pytest.mark.parametrize("email", ["not-an-email", "a@", "@b.com", "a b@c.com", ""])
def test_malformed_email_is_rejected(admin_client, email):
    assert _create(admin_client, email=email).status_code == 400


@pytest.mark.parametrize("username", ["has space", "sym#bol", "sla/sh", ""])
def test_malformed_username_is_rejected(admin_client, username):
    """Charset comes from the model's own UnicodeUsernameValidator, so the API and
    the database cannot disagree about what a username may contain."""
    assert _create(admin_client, username=username).status_code == 400


def test_over_long_fields_are_rejected(admin_client):
    assert _create(admin_client, username="u" * 151).status_code == 400
    assert _create(admin_client, first_name="f" * 151).status_code == 400


@pytest.mark.parametrize("payload", [
    {"username": 12345, "email": "a@b.com", "password": NEW_PASSWORD},
    {"username": ["list"], "email": "a@b.com", "password": NEW_PASSWORD},
    {"username": "ok", "email": {"a": 1}, "password": NEW_PASSWORD},
])
def test_invalid_field_types_are_rejected(admin_client, payload):
    response = admin_client.post(CREATE_URL, payload, content_type="application/json")

    assert response.status_code == 400


def test_non_object_body_is_rejected(admin_client):
    response = admin_client.post(CREATE_URL, [1, 2, 3], content_type="application/json")

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Password policy — the project's configured AUTH_PASSWORD_VALIDATORS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("password,reason", [
    ("abc", "too short (MinimumLengthValidator)"),
    ("password", "too common (CommonPasswordValidator)"),
    ("31415926535", "entirely numeric (NumericPasswordValidator)"),
])
def test_weak_passwords_are_rejected(admin_client, password, reason):
    response = _create(admin_client, password=password)

    assert response.status_code == 400, reason
    assert "password" in response.json()["errors"]
    assert not get_user_model().objects.filter(username="jdoe").exists()


def test_password_too_similar_to_the_username_is_rejected(admin_client):
    """UserAttributeSimilarityValidator only works if it is handed the other field
    values; without that it silently passes everything. This is the test that would
    fail if the unsaved-user object stopped being passed to validate_password."""
    response = _create(admin_client, username="alexandra", password="alexandra1")

    assert response.status_code == 400
    assert "password" in response.json()["errors"]


def test_password_is_never_logged(admin_client, caplog):
    """A policy failure echoes field errors to the caller, but the submitted
    password must never reach the log — errors are logged by field NAME only.

    Asserts on a distinctive password value, so this actually fails if the view
    starts logging ``serializer.errors`` (which carries the rejected password's
    validator messages) or the payload.
    """
    import logging

    distinctive = "31415926535"          # numeric-only -> fails the policy
    with caplog.at_level(logging.DEBUG):
        response = _create(admin_client, password=distinctive)

    assert response.status_code == 400
    assert distinctive not in caplog.text
    # The field NAME is expected in the log line; the value is not.
    assert "password" in caplog.text


def test_a_non_uniqueness_integrity_error_is_not_reported_as_a_conflict(admin_client):
    """An unattributable IntegrityError must surface as a server error, not a 409.

    Telling an admin "that user already exists" when no such user exists sends them
    hunting for a row that is not there. Only real, verifiable conflicts get 409.
    """
    from unittest import mock

    from apps.access_management.services import UserService as Svc

    with mock.patch.object(type(get_user_model().objects), "create_user",
                           side_effect=IntegrityError("CHECK constraint failed: something_else")):
        with pytest.raises(IntegrityError):
            Svc().create_user(username="whoever", email="who@example.com",
                              password=NEW_PASSWORD)


# ---------------------------------------------------------------------------
# Transactions / service layer
# ---------------------------------------------------------------------------


def test_no_user_is_left_behind_when_creation_fails():
    """Rollback: a failure *inside* the creation transaction must leave no row.

    The failure has to be injected inside the ``atomic`` block to mean anything — an
    exception raised after it (e.g. from the logging call) would prove nothing,
    because the INSERT has already committed by then. So the row is created and then
    the same call raises, which is exactly the shape of the next phase's work
    (role assignment joining this same transaction).
    """
    from unittest import mock

    User = get_user_model()
    before = User.objects.count()
    real_create_user = User.objects.create_user

    def create_then_fail(**kwargs):
        real_create_user(**kwargs)                       # row now exists in the tx
        raise RuntimeError("boom inside the transaction")

    with mock.patch.object(type(User.objects), "create_user", side_effect=create_then_fail):
        with pytest.raises(RuntimeError):
            UserService().create_user(
                username="ghost", email="ghost@example.com", password=NEW_PASSWORD)

    assert User.objects.count() == before
    assert not User.objects.filter(username="ghost").exists()


def test_service_raises_typed_conflicts():
    service = UserService()
    service.create_user(username="taken", email="taken@example.com", password=NEW_PASSWORD)

    with pytest.raises(UsernameTaken):
        service.create_user(username="taken", email="other@example.com", password=NEW_PASSWORD)
    with pytest.raises(EmailTaken):
        service.create_user(username="other", email="TAKEN@example.com", password=NEW_PASSWORD)


def test_service_requires_keyword_arguments():
    """Five same-typed strings: a positional call that transposed username and
    email would otherwise be accepted silently."""
    with pytest.raises(TypeError):
        UserService().create_user("uname", "e@example.com", NEW_PASSWORD)


# ---------------------------------------------------------------------------
# List — POST /api/v1/users/list
# ---------------------------------------------------------------------------


def _list(client, **body):
    return client.post(LIST_URL, body, content_type="application/json")


@pytest.fixture
def population(admin_user):
    """Nine extra users with predictable names, plus the admin fixture."""
    User = get_user_model()
    for i in range(9):
        User.objects.create_user(
            username=f"user{i:02d}", password=ADMIN_PASSWORD,
            email=f"user{i:02d}@example.com", is_active=(i % 2 == 0))
    return User.objects.count()


def test_list_returns_a_page_with_totals(admin_client, population):
    response = _list(admin_client, page=1, page_size=4)

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Users retrieved successfully."
    assert len(body["data"]["users"]) == 4
    pagination = body["data"]["pagination"]
    assert pagination["page"] == 1
    assert pagination["page_size"] == 4
    assert pagination["total"] == population
    assert pagination["total_pages"] == (population + 3) // 4
    assert pagination["has_next"] is True
    assert pagination["has_previous"] is False


def test_list_uses_the_same_projection_as_create(admin_client, population):
    """One user representation across the API — a list row must carry every field
    the shared representation has, plus its own admin-table extras."""
    row = _list(admin_client, page_size=1).json()["data"]["users"][0]

    assert set(row) == PUBLIC_FIELDS | LIST_EXTRA_FIELDS


def test_list_never_exposes_password_hashes(admin_client, population):
    """Hasher-agnostic: asserts the actual stored hash strings are absent, rather
    than grepping for "pbkdf2" — which would silently stop testing anything the
    moment the hasher changes (as it does under this module's fast test hasher)."""
    stored_hashes = list(get_user_model().objects.values_list("password", flat=True))
    assert stored_hashes, "fixture should have created users with passwords"

    body = str(_list(admin_client, page_size=100).json())

    for stored in stored_hashes:
        assert stored not in body
    assert "password" not in body


def test_list_paginates_without_repeating_or_dropping_rows(admin_client, population):
    """Deterministic ordering: paging through must visit every user exactly once.

    Without the secondary sort on id, rows tying on the sort key can reappear on the
    next page or be skipped entirely.
    """
    seen = []
    page = 1
    while True:
        body = _list(admin_client, page=page, page_size=3).json()["data"]
        seen.extend(u["user_id"] for u in body["users"])
        if not body["pagination"]["has_next"]:
            break
        page += 1

    assert len(seen) == population
    assert len(set(seen)) == population        # no duplicates across pages


def test_list_page_beyond_the_end_is_empty_not_an_error(admin_client, population):
    body = _list(admin_client, page=999, page_size=10).json()

    assert body["status_code"] == 200
    assert body["data"]["users"] == []
    assert body["data"]["pagination"]["has_next"] is False


def test_list_search_matches_username_or_email(admin_client, population):
    by_username = _list(admin_client, search="user0").json()["data"]
    assert by_username["pagination"]["total"] >= 1
    assert all("user0" in u["username"] for u in by_username["users"])

    by_email = _list(admin_client, search="root@example").json()["data"]
    assert by_email["pagination"]["total"] == 1
    assert by_email["users"][0]["username"] == "root"


def test_list_search_is_case_insensitive(admin_client, population):
    assert _list(admin_client, search="USER00").json()["data"]["pagination"]["total"] == 1


def test_list_filters_by_is_active(admin_client, population):
    inactive = _list(admin_client, is_active=False).json()["data"]

    assert inactive["pagination"]["total"] >= 1
    assert all(u["is_active"] is False for u in inactive["users"])


def test_list_is_active_omitted_means_no_filter(admin_client, population):
    """Tri-state: absent must mean "all", not "False"."""
    everyone = _list(admin_client).json()["data"]["pagination"]["total"]

    assert everyone == population


def test_list_ordering_is_honoured(admin_client, population):
    ascending = [u["username"] for u in
                 _list(admin_client, ordering="username", page_size=100).json()["data"]["users"]]
    descending = [u["username"] for u in
                  _list(admin_client, ordering="-username", page_size=100).json()["data"]["users"]]

    assert ascending == sorted(ascending)
    assert descending == list(reversed(ascending))


def test_list_rejects_an_unknown_ordering_field(admin_client):
    """order_by() with an arbitrary string can traverse relations or 500 — the
    allowlist must reject anything not named."""
    response = _list(admin_client, ordering="password")

    assert response.status_code == 400
    assert "ordering" in response.json()["errors"]


def test_list_caps_page_size(admin_client):
    """One caller must not be able to ask for the whole table."""
    response = _list(admin_client, page_size=10_000)

    assert response.status_code == 400
    assert "page_size" in response.json()["errors"]


@pytest.mark.parametrize("body", [{"page": 0}, {"page": -1}, {"page_size": 0},
                                 {"page": "abc"}, {"is_active": "maybe"}])
def test_list_rejects_invalid_paging_params(admin_client, body):
    assert _list(admin_client, **body).status_code == 400


def test_list_costs_two_queries_regardless_of_page_size(admin_client, population):
    """One COUNT plus one page fetch — no N+1 as the page grows."""
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    from apps.access_management.services import UserService

    with CaptureQueriesContext(connection) as ctx:
        UserService().list_users(page=1, page_size=100)

    assert len(ctx.captured_queries) == 2


def test_list_roles_reflects_actual_assignments(admin_client):
    created = _create(admin_client).json()["data"]
    role_a = Role.objects.create(name="Data Analyst")
    role_b = Role.objects.create(name="Reviewer")
    UserRole.objects.create(user_id=created["user_id"], role=role_a)
    UserRole.objects.create(user_id=created["user_id"], role=role_b)

    row = next(u for u in _list(admin_client, page_size=100).json()["data"]["users"]
              if u["user_id"] == created["user_id"])

    assert {r["id"] for r in row["roles"]} == {role_a.pk, role_b.pk}
    assert {r["name"] for r in row["roles"]} == {"Data Analyst", "Reviewer"}


def test_list_roles_is_empty_for_an_unassigned_user(admin_client):
    created = _create(admin_client).json()["data"]

    row = next(u for u in _list(admin_client, page_size=100).json()["data"]["users"]
              if u["user_id"] == created["user_id"])

    assert row["roles"] == []


def test_deactivating_a_user_sets_deleted_at(admin_client):
    created = _create(admin_client).json()["data"]

    _update(admin_client, user_id=created["user_id"], is_active=False)

    row = next(u for u in _list(admin_client, page_size=100).json()["data"]["users"]
              if u["user_id"] == created["user_id"])
    assert row["deleted_at"] is not None


def test_reactivating_a_user_clears_deleted_at(admin_client):
    created = _create(admin_client).json()["data"]
    _update(admin_client, user_id=created["user_id"], is_active=False)

    _update(admin_client, user_id=created["user_id"], is_active=True)

    row = next(u for u in _list(admin_client, page_size=100).json()["data"]["users"]
              if u["user_id"] == created["user_id"])
    assert row["deleted_at"] is None


def test_a_new_user_has_no_deleted_at_and_a_real_updated_at(admin_client):
    """create_user() creates the profile row in the same unit of work — no user
    ever has to wait for its first deactivation before ``updated_at`` is real."""
    created = _create(admin_client).json()["data"]

    row = next(u for u in _list(admin_client, page_size=100).json()["data"]["users"]
              if u["user_id"] == created["user_id"])

    assert row["deleted_at"] is None
    assert row["updated_at"] != ""


def test_a_user_predating_the_profile_table_shows_placeholders(admin_client):
    """Backfill is lazy, not a data migration — a user created directly (bypassing
    UserService, as any pre-migration-0008 row effectively was) has no profile row
    until something touches it, and the list must not error on that gap."""
    user = get_user_model().objects.create_user(username="legacy", password=ADMIN_PASSWORD)

    row = next(u for u in _list(admin_client, page_size=100).json()["data"]["users"]
              if u["user_id"] == user.pk)

    assert row["updated_at"] == ""
    assert row["deleted_at"] is None


def test_list_requires_staff(plain_user):
    anonymous = _list(Client())
    client = Client()
    client.force_login(plain_user)

    assert anonymous.status_code == 401
    assert _list(client).status_code == 403


# ---------------------------------------------------------------------------
# Detail — POST /api/v1/users/detail
# ---------------------------------------------------------------------------


def _detail(client, **body):
    return client.post(DETAIL_URL, body, content_type="application/json")


def test_detail_returns_one_user(admin_client):
    created = _create(admin_client).json()["data"]

    response = _detail(admin_client, user_id=created["user_id"])

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "User retrieved successfully."
    assert body["data"]["username"] == "jdoe"
    assert body["data"]["email"] == "j.doe@example.com"


def test_detail_uses_the_same_projection_as_create_and_list(admin_client):
    """One user representation across the API — opening a row must not return a
    different shape from the row that was clicked. ``list`` carries a few extra
    admin-table fields on top (``LIST_EXTRA_FIELDS``) but must not disagree with
    ``detail``/``create`` about any field the three share."""
    created = _create(admin_client).json()["data"]

    detail = _detail(admin_client, user_id=created["user_id"]).json()["data"]
    listed = next(u for u in _list(admin_client, page_size=100).json()["data"]["users"]
                  if u["user_id"] == created["user_id"])

    assert set(detail) == PUBLIC_FIELDS
    assert set(listed) == PUBLIC_FIELDS | LIST_EXTRA_FIELDS
    assert detail == created
    assert {k: v for k, v in listed.items() if k in PUBLIC_FIELDS} == detail


def test_detail_never_exposes_the_password_hash(admin_client):
    created = _create(admin_client).json()["data"]
    stored = get_user_model().objects.get(pk=created["user_id"]).password

    body = str(_detail(admin_client, user_id=created["user_id"]).json())

    assert stored not in body
    assert "password" not in body


def test_detail_of_a_missing_user_is_404(admin_client):
    from apps.access_management.services import CODE_USER_NOT_FOUND

    response = _detail(admin_client, user_id=999_999)

    assert response.status_code == 404
    assert response.json()["code"] == CODE_USER_NOT_FOUND


@pytest.mark.parametrize("body", [{}, {"user_id": 0}, {"user_id": -1},
                                 {"user_id": "abc"}, {"id": 1}])
def test_detail_rejects_a_malformed_body(admin_client, body):
    """A nonsensical id is a 400 (client bug), not a 404 — so the two stay
    diagnosable apart."""
    response = _detail(admin_client, **body)

    assert response.status_code == 400
    assert response.json()["errors"]


def test_detail_reflects_a_prior_update(admin_client):
    """Reads the committed row, not a cache."""
    user_id = _create(admin_client).json()["data"]["user_id"]
    _update(admin_client, user_id=user_id, first_name="Janet")

    assert _detail(admin_client, user_id=user_id).json()["data"]["display_name"] == "Janet"


def test_detail_costs_one_query(admin_client):
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    from apps.access_management.services import UserService

    user_id = _create(admin_client).json()["data"]["user_id"]

    with CaptureQueriesContext(connection) as ctx:
        UserService().get_user(user_id)

    assert len(ctx.captured_queries) == 1


def test_detail_requires_staff(plain_user, admin_client):
    user_id = _create(admin_client).json()["data"]["user_id"]
    client = Client()
    client.force_login(plain_user)

    assert _detail(Client(), user_id=user_id).status_code == 401
    assert _detail(client, user_id=user_id).status_code == 403


# ---------------------------------------------------------------------------
# Update — POST /api/v1/users/update
# ---------------------------------------------------------------------------


def _update(client, **body):
    return client.post(UPDATE_URL, body, content_type="application/json")


def _delete(client, **body):
    return client.post(DELETE_URL, body, content_type="application/json")


@pytest.fixture
def target(admin_client):
    """A user created through the API, to be updated."""
    return _create(admin_client).json()["data"]["user_id"]


def test_update_changes_profile_fields(admin_client, target):
    response = _update(admin_client, user_id=target, first_name="Janet",
                       last_name="Doherty", email="janet@example.com")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["display_name"] == "Janet"
    assert data["email"] == "janet@example.com"

    user = get_user_model().objects.get(pk=target)
    assert (user.first_name, user.last_name, user.email) == (
        "Janet", "Doherty", "janet@example.com")


def test_update_is_partial(admin_client, target):
    """Only what is sent changes — an omitted field must not be blanked."""
    _update(admin_client, user_id=target, first_name="Janet")

    user = get_user_model().objects.get(pk=target)
    assert user.first_name == "Janet"
    assert user.last_name == "Doe"                    # untouched
    assert user.email == "j.doe@example.com"          # untouched


def test_update_returns_the_shared_projection(admin_client, target):
    body = _update(admin_client, user_id=target, first_name="Janet").json()

    assert set(body["data"]) == PUBLIC_FIELDS
    assert "pbkdf2" not in str(body)


def test_update_of_a_missing_user_is_404(admin_client):
    from apps.access_management.services import CODE_USER_NOT_FOUND

    response = _update(admin_client, user_id=999_999, first_name="Nobody")

    assert response.status_code == 404
    assert response.json()["code"] == CODE_USER_NOT_FOUND


def test_update_to_a_taken_email_is_409(admin_client, target):
    get_user_model().objects.create_user(
        username="occupant", password=ADMIN_PASSWORD, email="occupied@example.com")

    response = _update(admin_client, user_id=target, email="occupied@example.com")

    assert response.status_code == 409
    assert response.json()["code"] == CODE_EMAIL_TAKEN


def test_update_to_a_taken_email_differing_only_in_case_is_409(admin_client, target):
    get_user_model().objects.create_user(
        username="occupant", password=ADMIN_PASSWORD, email="occupied@example.com")

    response = _update(admin_client, user_id=target, email="OCCUPIED@Example.com")

    assert response.status_code == 409


def test_update_can_resubmit_the_users_own_email(admin_client, target):
    """The row must not be treated as conflicting with itself — that would make any
    form that submits all fields unusable."""
    response = _update(admin_client, user_id=target, email="j.doe@example.com",
                       first_name="Janet")

    assert response.status_code == 200


def test_update_rejects_privileged_fields(admin_client, target):
    """Privilege escalation via update, the same guard as on create.

    ``is_active`` is deliberately NOT in this list — see
    ``tests/test_admin_bootstrap.py`` for its own (non-privileged) update coverage,
    including the last-admin guard it carries.
    """
    for field in ("is_staff", "is_superuser"):
        response = _update(admin_client, user_id=target, **{field: True})
        assert response.status_code == 400, field
        assert field in response.json()["errors"]

    user = get_user_model().objects.get(pk=target)
    assert user.is_staff is False and user.is_superuser is False


@pytest.mark.parametrize("field,value", [("username", "renamed"),
                                        ("password", "new-password-here-42")])
def test_update_refuses_username_and_password(admin_client, target, field, value):
    """Both belong elsewhere: renaming an identity is its own operation, and setting a
    credential belongs to apps.authentication (where it also revokes tokens)."""
    response = _update(admin_client, user_id=target, **{field: value})

    assert response.status_code == 400
    assert field in response.json()["errors"]


def test_update_with_no_changes_is_rejected(admin_client, target):
    """A body carrying only user_id is almost certainly a client bug, so it is a 400
    rather than a silent success."""
    response = _update(admin_client, user_id=target)

    assert response.status_code == 400


def test_update_requires_a_user_id(admin_client):
    response = _update(admin_client, first_name="Nobody")

    assert response.status_code == 400
    assert "user_id" in response.json()["errors"]


@pytest.mark.parametrize("email", ["not-an-email", "a@", ""])
def test_update_validates_the_new_email(admin_client, target, email):
    assert _update(admin_client, user_id=target, email=email).status_code == 400


def test_update_requires_staff(plain_user, target):
    anonymous = _update(Client(), user_id=target, first_name="X")
    client = Client()
    client.force_login(plain_user)

    assert anonymous.status_code == 401
    assert _update(client, user_id=target, first_name="X").status_code == 403


def test_update_writes_only_the_submitted_columns(admin_client, target):
    """``update_fields`` is what stops a partial update clobbering a column another
    request changed concurrently."""
    from unittest import mock

    User = get_user_model()
    real_save = User.save
    captured = {}

    def capture(self, *args, **kwargs):
        captured["update_fields"] = kwargs.get("update_fields")
        return real_save(self, *args, **kwargs)

    with mock.patch.object(User, "save", capture):
        _update(admin_client, user_id=target, first_name="Janet")

    assert captured["update_fields"] == ["first_name"]


def test_service_update_requires_at_least_one_field(target):
    from apps.access_management.services import UserService as Svc

    with pytest.raises(ValueError):
        Svc().update_user(target)


# ---------------------------------------------------------------------------
# Regression — the shared envelope refactor (apps/core/api.py)
# ---------------------------------------------------------------------------


def test_core_api_omits_absent_envelope_keys():
    """``data``/``code`` must be ABSENT, not null, when not supplied — clients check
    for key presence, and logout deliberately answers with no ``data``."""
    from apps.core.api import error, invalid_payload, success

    assert success("ok").data == {"status_code": 200, "message": "ok"}
    assert success("ok", {"a": 1}).data == {"status_code": 200, "message": "ok",
                                            "data": {"a": 1}}
    assert error("nope", 404).data == {"status_code": 404, "message": "nope"}
    assert error("nope", 409, code="X").data == {"status_code": 409, "message": "nope",
                                                 "code": "X"}
    assert error("nope", 502, data={"k": 1}).data == {"status_code": 502,
                                                      "message": "nope", "data": {"k": 1}}
    assert invalid_payload({"f": ["bad"]}).data == {
        "status_code": 400, "message": "Invalid request data.", "errors": {"f": ["bad"]}}


def test_chat_endpoints_keep_their_documented_envelope(admin_client):
    """The chat views were refactored onto the shared helpers; their wire format is
    a documented frontend contract (CHAT_API_CONTRACT.md) and must not have moved.

    Only the paths that need no LLM are exercised here — a 400 and a success — which
    is enough to prove the envelope shape is unchanged.
    """
    bad = admin_client.post("/api/v1/conversations/history", {},
                            content_type="application/json")
    assert bad.status_code == 400
    assert bad.json() == {"status_code": 400, "message": "Invalid request data.",
                          "errors": {"chat_id": ["This field is required."]}}

    listed = admin_client.post("/api/v1/conversations/list", {},
                               content_type="application/json")
    assert listed.status_code == 200
    assert listed.json() == {"status_code": 200,
                             "message": "Conversations retrieved successfully.",
                             "data": {"conversations": []}}


def test_auth_endpoints_keep_their_documented_envelope():
    """Same guarantee for apps/authentication, whose contract is AUTH_API_CONTRACT.md."""
    response = Client().post("/api/v1/auth/login", {"username": "nope"},
                             content_type="application/json")

    assert response.status_code == 400
    assert response.json() == {"status_code": 400, "message": "Invalid request data.",
                               "errors": {"password": ["This field is required."]}}


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_routes_are_wired():
    from django.urls import resolve

    from apps.access_management.views import (
        UserCreateView,
        UserDetailView,
        UserListView,
        UserUpdateView,
    )

    assert resolve(CREATE_URL).func.view_class is UserCreateView
    assert resolve(DETAIL_URL).func.view_class is UserDetailView
    assert resolve(LIST_URL).func.view_class is UserListView
    assert resolve(UPDATE_URL).func.view_class is UserUpdateView


def test_package_layout_re_exports_its_public_names():
    """The app is split into serializers/services/views packages so roles and
    permissions can land beside users. Callers must keep importing from the PACKAGE,
    so a name can move between modules without breaking them."""
    from apps.access_management import serializers, services, views

    assert services.UserService is not None
    assert services.AccessManagementError is not None
    assert serializers.UserCreateSerializer is not None
    assert serializers.UserDetailSerializer is not None
    assert views.UserDetailView is not None
    assert views.public_fields is not None
    # The shared base really is shared, not duplicated per domain.
    from apps.access_management.services.base import AccessManagementError

    assert services.AccessManagementError is AccessManagementError


@pytest.mark.parametrize("url", [CREATE_URL, DETAIL_URL, LIST_URL, UPDATE_URL, DELETE_URL])
def test_endpoints_are_post_only(admin_client, url):
    """The platform convention is POST <resource>/<action> (as apps/chat already
    does), so no other verb should be reachable on any of them."""
    assert admin_client.get(url).status_code == 405
    assert admin_client.put(url).status_code == 405
    assert admin_client.patch(url).status_code == 405
    assert admin_client.delete(url).status_code == 405


# ---------------------------------------------------------------------------
# Delete — a named convenience over update {is_active: false}, not a new path
# ---------------------------------------------------------------------------


def test_delete_is_a_soft_delete(admin_client, target):
    response = _delete(admin_client, user_id=target)

    assert response.status_code == 200
    user = get_user_model().objects.get(pk=target)
    assert user.is_active is False
    # The row still exists — this is not django.contrib.auth.models.User.delete().
    assert get_user_model().objects.filter(pk=target).exists()


def test_delete_sets_deleted_at(admin_client, target):
    _delete(admin_client, user_id=target)

    row = next(u for u in _list(admin_client, page_size=100).json()["data"]["users"]
              if u["user_id"] == target)
    assert row["deleted_at"] is not None


def test_delete_is_idempotent(admin_client, target):
    first = _delete(admin_client, user_id=target)
    second = _delete(admin_client, user_id=target)

    assert first.status_code == 200
    assert second.status_code == 200


def test_delete_the_last_admin_is_refused(admin_client, admin_user):
    response = _delete(admin_client, user_id=admin_user.pk)

    assert response.status_code == 409
    admin_user.refresh_from_db()
    assert admin_user.is_active is True


def test_delete_of_a_nonexistent_user_is_404(admin_client):
    response = _delete(admin_client, user_id=999_999)
    assert response.status_code == 404


def test_delete_requires_staff(plain_user):
    client = Client()
    client.force_login(plain_user)

    response = _delete(client, user_id=plain_user.pk)

    assert response.status_code == 403
