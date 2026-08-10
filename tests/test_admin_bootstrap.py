"""Coverage for User Story 1's remaining gaps: admin bootstrap, last-admin
protection, activate/deactivate, password complexity, password change, and the
login-time role/staff requirement.

Self-contained, same convention as ``tests/test_grants.py`` / ``tests/test_
authentication.py``: an in-process throwaway sqlite database, no repo-wide
pytest.ini.

Run from repo root: ``pytest tests/test_admin_bootstrap.py``
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
from django.core.exceptions import ValidationError as DjangoValidationError  # noqa: E402
from django.test import Client, override_settings  # noqa: E402

from apps.access_management.models import Role, UserRole  # noqa: E402
from apps.access_management.services import (  # noqa: E402
    ADMIN_ROLE_NAME,
    CODE_LAST_ADMIN_PROTECTED,
    CODE_LAST_ADMIN_ROLE_PROTECTED,
    AdminBootstrapService,
    AlreadyBootstrapped,
    LastAdminProtected,
    LastAdminRoleProtected,
    UserRoleService,
    UserService,
    is_last_active_admin,
)
from apps.authentication.password_validators import PasswordComplexityValidator  # noqa: E402
from apps.authentication.services import (  # noqa: E402
    AuthService,
    CurrentPasswordIncorrect,
    InvalidCredentials,
)

LOGIN_URL = "/api/v1/auth/login"
REFRESH_URL = "/api/v1/auth/refresh"
PASSWORD_CHANGE_URL = "/api/v1/auth/password/change"
USER_UPDATE_URL = "/api/v1/users/update"

# Satisfies every AUTH_PASSWORD_VALIDATORS rule (length, similarity, common,
# numeric, and the new complexity validator) so fixtures don't trip on policy
# incidentally to what each test actually means to exercise.
GOOD_PASSWORD = "Correct-Horse-97!"
GOOD_PASSWORD_2 = "Another-Fortress-42?"

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
def client():
    return Client()


def _make_admin(username="adminuser", password=GOOD_PASSWORD):
    """A second, independently-bootstrapped admin — for tests that need TWO admins
    so the "last admin" guard has room to say yes to one of them."""
    user = get_user_model().objects.create_user(
        username=username, password=password, is_staff=True)
    UserRole.objects.get_or_create(
        user=user, role=Role.objects.get(name=ADMIN_ROLE_NAME))
    return user


def _login(client, username, password):
    return client.post(LOGIN_URL, {"username": username, "password": password},
                       content_type="application/json")


def _clear_all_users():
    """Delete every user, INCLUDING the dev-fallback dummy ``apps/chat`` migration
    0002 seeds into every database (``username=admin``, not staff, not part of
    this feature). Bootstrap's precondition is "no users exist" — the dummy row
    means that is never literally true in a freshly-migrated database, real or
    test, so tests that mean to exercise a genuinely empty table clear it first
    rather than silently relying on (or being broken by) an unrelated fixture."""
    get_user_model().objects.all().delete()


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_creates_first_admin_with_both_is_staff_and_admin_role():
    _clear_all_users()

    user = AdminBootstrapService().bootstrap(
        username="root", email="root@example.com", password=GOOD_PASSWORD)

    user.refresh_from_db()
    assert user.is_staff is True
    assert UserRole.objects.filter(
        user=user, role__name=ADMIN_ROLE_NAME).exists()


def test_bootstrap_refuses_when_a_user_already_exists():
    _clear_all_users()
    get_user_model().objects.create_user(username="somebody", password=GOOD_PASSWORD)

    with pytest.raises(AlreadyBootstrapped):
        AdminBootstrapService().bootstrap(
            username="root", email="root@example.com", password=GOOD_PASSWORD)

    # Refused BEFORE any write: no second user, no accidental promotion.
    assert get_user_model().objects.count() == 1


def test_bootstrapped_admin_can_log_in_immediately(client):
    _clear_all_users()
    AdminBootstrapService().bootstrap(
        username="root", email="root@example.com", password=GOOD_PASSWORD)

    response = _login(client, "root", GOOD_PASSWORD)

    assert response.status_code == 200


def test_concurrent_bootstrap_yields_exactly_one_winner():
    """The race two ``bootstrap_admin`` invocations run on a fresh install.

    Reproduced deterministically rather than with real threads — this codebase's
    established convention for exactly this reason (see
    ``test_concurrent_refresh_of_one_token_yields_exactly_one_winner`` in
    ``tests/test_authentication.py``): a second sqlite connection on a different
    thread does not share this test's transaction, so a real race would either
    deadlock the shared file or silently see stale state, proving nothing.

    ``select_for_update`` on the Admin role row is the actual arbiter in
    production (Postgres); what matters for THIS test is the outcome it
    guarantees — call two, back to back, and the second must always see the
    first's committed user and refuse, never create a second admin.
    """
    _clear_all_users()

    winner = AdminBootstrapService().bootstrap(
        username="root0", email="root0@example.com", password=GOOD_PASSWORD)

    with pytest.raises(AlreadyBootstrapped):
        AdminBootstrapService().bootstrap(
            username="root1", email="root1@example.com", password=GOOD_PASSWORD)

    assert get_user_model().objects.filter(is_staff=True).count() == 1
    assert get_user_model().objects.filter(is_staff=True).get() == winner


# ---------------------------------------------------------------------------
# Last-admin protection
# ---------------------------------------------------------------------------


def test_is_last_active_admin_true_for_the_only_staff_user():
    admin = _make_admin()
    assert is_last_active_admin(admin) is True


def test_is_last_active_admin_false_when_another_admin_exists():
    first = _make_admin("first")
    _make_admin("second")
    assert is_last_active_admin(first) is False


def test_is_last_active_admin_false_for_a_non_staff_user():
    plain = get_user_model().objects.create_user(username="plain", password=GOOD_PASSWORD)
    assert is_last_active_admin(plain) is False


def test_deactivating_the_last_admin_is_refused():
    admin = _make_admin()

    with pytest.raises(LastAdminProtected):
        UserService().update_user(admin.pk, is_active=False)

    admin.refresh_from_db()
    assert admin.is_active is True


def test_deactivating_an_admin_when_another_exists_succeeds():
    first = _make_admin("first")
    _make_admin("second")

    user = UserService().update_user(first.pk, is_active=False)

    assert user.is_active is False


def test_deactivate_via_api_returns_409_for_the_last_admin():
    admin = _make_admin()
    client = Client()
    client.force_login(admin)

    response = client.post(
        USER_UPDATE_URL, {"user_id": admin.pk, "is_active": False},
        content_type="application/json")

    assert response.status_code == 409
    assert response.json()["code"] == CODE_LAST_ADMIN_PROTECTED


def test_removing_admin_role_from_the_last_admin_is_refused():
    admin = _make_admin()
    admin_role = Role.objects.get(name=ADMIN_ROLE_NAME)

    with pytest.raises(LastAdminRoleProtected) as exc_info:
        UserRoleService().revoke(user_id=admin.pk, role_id=admin_role.pk)

    assert exc_info.value.code == CODE_LAST_ADMIN_ROLE_PROTECTED
    assert UserRole.objects.filter(user=admin, role=admin_role).exists()


def test_removing_admin_role_when_another_admin_exists_succeeds():
    first = _make_admin("first")
    _make_admin("second")
    admin_role = Role.objects.get(name=ADMIN_ROLE_NAME)

    removed = UserRoleService().revoke(user_id=first.pk, role_id=admin_role.pk)

    assert removed is True


def test_removing_a_non_admin_role_is_never_guarded():
    """The guard is scoped to the Admin role specifically — an ordinary role can
    always be freely revoked, last-holder or not."""
    admin = _make_admin()
    other_role = Role.objects.create(name="Analyst")
    UserRole.objects.create(user=admin, role=other_role)

    removed = UserRoleService().revoke(user_id=admin.pk, role_id=other_role.pk)

    assert removed is True


@override_settings(VEDA_JWT_AUTH=True)
def test_deactivating_a_user_revokes_their_refresh_tokens(client):
    first = _make_admin("first")
    _make_admin("second")
    login_body = _login(client, "first", GOOD_PASSWORD).json()["data"]

    UserService().update_user(first.pk, is_active=False)

    response = client.post(
        REFRESH_URL, {"refresh_token": login_body["refresh_token"]},
        content_type="application/json")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Deactivate/activate via update — idempotency and access control
# ---------------------------------------------------------------------------


def test_deactivate_via_update_is_idempotent():
    first = _make_admin("first")
    second = _make_admin("second")
    client = Client()
    # Acting AS `second`, not `first` — Django's own session backend refuses an
    # inactive user on the very next request (ModelBackend.user_can_authenticate),
    # so the target of a deactivation can never be the one whose session calls it.
    client.force_login(second)

    UserService().update_user(first.pk, is_active=False)  # first pass, via service
    response = client.post(
        USER_UPDATE_URL, {"user_id": first.pk, "is_active": False},
        content_type="application/json")

    assert response.status_code == 200
    first.refresh_from_db()
    assert first.is_active is False


def test_activate_via_update_reactivates_an_inactive_user():
    first = _make_admin("first")
    _make_admin("second")
    UserService().update_user(first.pk, is_active=False)

    user = UserService().update_user(first.pk, is_active=True)

    assert user.is_active is True


def test_update_requires_staff():
    admin = _make_admin()
    plain = get_user_model().objects.create_user(username="plain", password=GOOD_PASSWORD)
    client = Client()
    client.force_login(plain)

    response = client.post(
        USER_UPDATE_URL, {"user_id": admin.pk, "is_active": False},
        content_type="application/json")

    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Password complexity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("password", [
    "alllowercase1!",   # no uppercase
    "ALLUPPERCASE1!",   # no lowercase
    "NoDigitsHere!!",   # no digit
    "NoSpecial123ab",   # no special character
])
def test_complexity_validator_rejects_a_missing_category(password):
    with pytest.raises(DjangoValidationError):
        PasswordComplexityValidator().validate(password)


def test_complexity_validator_accepts_a_compliant_password():
    PasswordComplexityValidator().validate(GOOD_PASSWORD)  # raises nothing


def test_complexity_validator_reports_every_failing_rule_at_once():
    with pytest.raises(DjangoValidationError) as exc_info:
        PasswordComplexityValidator().validate("alllowercase")

    assert len(exc_info.value.messages) == 3  # missing upper, digit, special


def test_complexity_thresholds_are_configurable_not_hardcoded():
    """A deployment can disable a category via OPTIONS — see AUTH_PASSWORD_
    VALIDATORS in config/settings/base.py — without touching this class."""
    lenient = PasswordComplexityValidator(min_special=0)
    lenient.validate("NoSpecialCharsHere1")  # raises nothing despite no special char


def test_user_create_via_api_rejects_a_password_missing_a_special_character():
    admin = _make_admin()
    client = Client()
    client.force_login(admin)

    response = client.post(
        "/api/v1/users/create",
        {"username": "newperson", "email": "newperson@example.com",
         "password": "NoSpecialChars123"},
        content_type="application/json")

    assert response.status_code == 400
    assert "password" in response.json()["errors"]


# ---------------------------------------------------------------------------
# Password change
# ---------------------------------------------------------------------------


def test_password_change_succeeds_with_correct_current_password(client):
    admin = _make_admin()
    client.force_login(admin)

    response = client.post(
        PASSWORD_CHANGE_URL,
        {"current_password": GOOD_PASSWORD, "new_password": GOOD_PASSWORD_2},
        content_type="application/json")

    assert response.status_code == 200
    admin.refresh_from_db()
    assert admin.check_password(GOOD_PASSWORD_2) is True


def test_password_change_rejects_the_wrong_current_password(client):
    admin = _make_admin()
    client.force_login(admin)

    response = client.post(
        PASSWORD_CHANGE_URL,
        {"current_password": "totally-wrong", "new_password": GOOD_PASSWORD_2},
        content_type="application/json")

    assert response.status_code == 401
    assert response.json()["code"] == "CURRENT_PASSWORD_INCORRECT"
    admin.refresh_from_db()
    assert admin.check_password(GOOD_PASSWORD) is True  # unchanged


def test_password_change_enforces_the_complexity_policy(client):
    admin = _make_admin()
    client.force_login(admin)

    response = client.post(
        PASSWORD_CHANGE_URL,
        {"current_password": GOOD_PASSWORD, "new_password": "nocapsnospecial1"},
        content_type="application/json")

    assert response.status_code == 400
    assert "new_password" in response.json()["errors"]


def test_change_password_service_unit_rejects_wrong_current_password():
    """Direct service-layer coverage, no HTTP — the same guarantee the API test
    above proves end to end."""
    admin = _make_admin()

    with pytest.raises(CurrentPasswordIncorrect):
        AuthService().change_password(admin, "totally-wrong", GOOD_PASSWORD_2)


def test_login_service_unit_raises_invalid_credentials_for_no_role_user():
    get_user_model().objects.create_user(username="nobody", password=GOOD_PASSWORD)

    with pytest.raises(InvalidCredentials):
        AuthService().login("nobody", GOOD_PASSWORD)


def test_password_change_requires_authentication():
    response = Client().post(
        PASSWORD_CHANGE_URL,
        {"current_password": GOOD_PASSWORD, "new_password": GOOD_PASSWORD_2},
        content_type="application/json")

    assert response.status_code == 401


@override_settings(VEDA_JWT_AUTH=True)
def test_password_change_revokes_existing_refresh_tokens(client):
    admin = _make_admin()
    login_body = _login(client, "adminuser", GOOD_PASSWORD).json()["data"]

    client.force_login(admin)
    client.post(
        PASSWORD_CHANGE_URL,
        {"current_password": GOOD_PASSWORD, "new_password": GOOD_PASSWORD_2},
        content_type="application/json")

    response = client.post(
        REFRESH_URL, {"refresh_token": login_body["refresh_token"]},
        content_type="application/json")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Login-time role / staff requirement
# ---------------------------------------------------------------------------


def test_login_is_refused_for_a_user_with_no_role_and_not_staff(client):
    get_user_model().objects.create_user(username="nobody", password=GOOD_PASSWORD)

    response = _login(client, "nobody", GOOD_PASSWORD)

    assert response.status_code == 401
    # Same generic code a wrong password gets — no enumeration signal that the
    # credentials were actually correct.
    assert response.json()["code"] == "INVALID_CREDENTIALS"


def test_login_succeeds_once_a_role_is_assigned(client):
    user = get_user_model().objects.create_user(username="nobody", password=GOOD_PASSWORD)
    UserRoleService().assign(
        user_id=user.pk, role_id=Role.objects.create(name="Analyst").pk)

    response = _login(client, "nobody", GOOD_PASSWORD)

    assert response.status_code == 200


def test_login_succeeds_for_a_staff_user_with_no_role_assignment(client):
    get_user_model().objects.create_user(
        username="staffer", password=GOOD_PASSWORD, is_staff=True)

    response = _login(client, "staffer", GOOD_PASSWORD)

    assert response.status_code == 200


def test_no_role_refusal_does_not_poison_the_lockout_counter(client):
    """Repeatedly hitting the no-role wall with the CORRECT password must never
    lock the account out — that counter only tracks wrong passwords."""
    get_user_model().objects.create_user(username="nobody", password=GOOD_PASSWORD)

    for _ in range(5):
        assert _login(client, "nobody", GOOD_PASSWORD).status_code == 401

    UserRoleService().assign(
        user_id=get_user_model().objects.get(username="nobody").pk,
        role_id=Role.objects.create(name="Analyst").pk)

    assert _login(client, "nobody", GOOD_PASSWORD).status_code == 200


@override_settings(VEDA_JWT_AUTH=True)
def test_login_response_includes_roles_and_permission_codes_when_jwt_is_on(client):
    admin = _make_admin()

    body = _login(client, "adminuser", GOOD_PASSWORD).json()["data"]

    assert body["roles"] == [ADMIN_ROLE_NAME]
    assert "user.manage" in body["permission_codes"]


def test_login_response_omits_authorization_context_when_jwt_is_off(client):
    """Byte-identical legacy contract while the rollout flag is off — the same
    invariant ``test_login_with_flag_off_returns_the_legacy_payload`` protects in
    tests/test_authentication.py."""
    _make_admin()

    body = _login(client, "adminuser", GOOD_PASSWORD).json()["data"]

    assert "roles" not in body
    assert "permission_codes" not in body
