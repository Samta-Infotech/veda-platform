"""Coverage for apps/authentication — POST /api/v1/auth/{login,refresh,logout}.

Self-contained on purpose, matching the convention the rest of this suite already
uses (see ``tests/test_apps_layer_refactor.py`` / ``tests/test_fk_roundtrip.py``):
Django is configured in-process and the test database is built here, rather than
adding a repo-wide ``pytest.ini``/``DJANGO_SETTINGS_MODULE`` that would change how
every existing test module bootstraps itself.

Unlike most of this suite these tests DO need a database — refresh-token rotation
and revocation are enforced by ``OutstandingToken``/``BlacklistedToken`` rows, and
a test that mocked those away would be testing nothing. A throwaway sqlite test
database is created once per module and destroyed after; the developer's
``db.sqlite3`` is never touched.

Run from repo root: ``pytest tests/test_authentication.py``
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _setup_django():
    """Minimal in-process Django config.

    ``config`` is imported first so the ``config/`` package wins over
    ``veda_core/config.py`` — the same name collision the other Django-touching
    test modules work around.
    """
    import config  # noqa: F401

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
    import django

    django.setup()


_setup_django()

from django.contrib.auth import get_user_model  # noqa: E402
from django.test import Client, override_settings  # noqa: E402
from django.urls import path  # noqa: E402
from rest_framework.permissions import IsAuthenticated  # noqa: E402
from rest_framework.response import Response as DRFResponse  # noqa: E402
from rest_framework.views import APIView  # noqa: E402
from rest_framework_simplejwt.authentication import JWTAuthentication  # noqa: E402
from rest_framework_simplejwt.settings import api_settings as jwt_settings  # noqa: E402

from config.urls import urlpatterns as root_urlpatterns  # noqa: E402
from apps.authentication.services import (  # noqa: E402
    AccountInactive,
    AccountLocked,
    AuthService,
    CODE_ACCOUNT_INACTIVE,
    CODE_ACCOUNT_LOCKED,
    CODE_INVALID_CREDENTIALS,
    CODE_INVALID_TOKEN,
    CODE_NO_ROLE_ASSIGNED,
    InvalidCredentials,
    InvalidRefreshToken,
    LEGACY_ACCESS_TOKEN,
    NoRoleAssigned,
    _RotatableRefreshToken,
)
from rest_framework_simplejwt.token_blacklist.models import OutstandingToken
from unittest import mock
from apps.authentication.views import _error_response
from rest_framework_simplejwt.token_blacklist.models import BlacklistedToken
import jwt as pyjwt
from datetime import timedelta
from rest_framework_simplejwt.tokens import AccessToken
import apps.chat.serializers as chat_serializers
import apps.chat.views as chat_views

LOGIN_URL = "/api/v1/auth/login"
REFRESH_URL = "/api/v1/auth/refresh"
LOGOUT_URL = "/api/v1/auth/logout"

PASSWORD = "correct-horse-battery-staple"

# Distinct source addresses, because the blocking lockout counter is keyed per
# (account, client IP) — that is what stops one caller locking out another.
ATTACKER_IP = "203.0.113.7"
VICTIM_IP = "198.51.100.22"

# An in-process cache for the whole module. The dev settings point the cache at a
# Redis that may not be running locally, and ``AuthService`` deliberately fails
# OPEN on cache errors — so against a dead Redis every lockout assertion would
# pass vacuously. locmem makes the counters real and deterministic.
#
# The real REST_FRAMEWORK config is deliberately left in place (throttling
# included): the per-test ``cache.clear()`` below also clears DRF's throttle
# history, so no test inherits another's request count, and no test here submits
# enough requests to reach the 10/min ``login`` scope on its own. Overriding
# REST_FRAMEWORK to disable throttling would mean these tests no longer exercise
# the real throttle wiring — and a missing scope rate is a 500 on every login.
_TEST_OVERRIDES = dict(
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
)


@pytest.fixture(scope="module", autouse=True)
def _database():
    from django.db import connection
    from django.test.utils import setup_test_environment, teardown_test_environment

    # Tolerate a plugin (e.g. pytest-django, if installed) having already put the
    # process into test mode: calling this twice is an error, and tearing down an
    # environment we did not set up would break every other test module.
    try:
        setup_test_environment()
        owns_environment = True
    except RuntimeError:
        owns_environment = False

    # apps.substrate's 0002 migration is raw Postgres/pgvector DDL (``vector(N)``
    # columns, ``USING hnsw``) that sqlite cannot parse. Its ORM models are
    # ``managed=False`` mirrors of those tables, so bypassing that app's
    # migrations lets the throwaway sqlite database be built without touching
    # anything the auth tests exercise. Nothing here reads the substrate.
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
    """Roll every test back and start it with an empty cache, so ordering cannot
    leak users, blacklisted tokens or failure counters between tests."""
    from django.core.cache import cache
    from django.db import transaction

    with override_settings(**_TEST_OVERRIDES):
        cache.clear()
        atomic = transaction.atomic()
        atomic.__enter__()
        try:
            yield
        finally:
            transaction.set_rollback(True)
            atomic.__exit__(None, None, None)


@pytest.fixture
def user():
    return get_user_model().objects.create_user(
        username="alice", password=PASSWORD, first_name="Alice", is_staff=True)


@pytest.fixture
def client():
    return Client()


def _login(client, username, password, ip=None, **extra):
    """POST the login endpoint. ``ip`` sets REMOTE_ADDR — the lockout is keyed per
    (account, source), so tests that exercise it must control the source address."""
    if ip:
        extra["REMOTE_ADDR"] = ip
    return client.post(LOGIN_URL, {"username": username, "password": password},
                       content_type="application/json", **extra)


def _refresh(client, refresh_token):
    return client.post(REFRESH_URL, {"refresh_token": refresh_token},
                       content_type="application/json")


def _logout(client, refresh_token):
    return client.post(LOGOUT_URL, {"refresh_token": refresh_token},
                       content_type="application/json")


def _tokens(client, username=None, password=PASSWORD):
    """Log in and return the issued token pair."""
    data = _login(client, username or "alice", password).json()["data"]
    return data["access_token"], data["refresh_token"]


# ---------------------------------------------------------------------------
# Login — happy path
# ---------------------------------------------------------------------------


@override_settings(VEDA_JWT_AUTH=True)
def test_login_issues_access_and_refresh_tokens(client, user):
    response = _login(client, "alice", PASSWORD)

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Login successful."
    data = body["data"]
    assert data["user_id"] == user.pk
    assert data["username"] == "alice"
    assert data["display_name"] == "Alice"
    assert data["token_type"] == "Bearer"
    assert data["expires_in"] > 0
    # Two distinct, real JWTs (three dot-separated segments), not placeholders.
    assert data["access_token"].count(".") == 2
    assert data["refresh_token"].count(".") == 2
    assert data["access_token"] != data["refresh_token"]


@override_settings(VEDA_JWT_AUTH=True)
def test_login_display_name_falls_back_to_username(client):
    get_user_model().objects.create_user(username="bob", password=PASSWORD, is_staff=True)

    data = _login(client, "bob", PASSWORD).json()["data"]

    assert data["display_name"] == "bob"


@override_settings(VEDA_JWT_AUTH=True)
def test_login_records_the_refresh_token_as_outstanding(client, user):
    """Rotation and revocation both key off OutstandingToken, so a login that
    did not record its jti would mint an unrevocable refresh token."""

    _login(client, "alice", PASSWORD)

    assert OutstandingToken.objects.filter(user=user).count() == 1


@override_settings(VEDA_JWT_AUTH=True)
def test_two_logins_issue_independent_tokens(client, user):
    """Logging in twice must not invalidate the first session — a user with two
    devices is normal, and rotation detection must not fire on it."""
    first = _login(client, "alice", PASSWORD).json()["data"]["refresh_token"]
    second = _login(client, "alice", PASSWORD).json()["data"]["refresh_token"]

    assert first != second


# ---------------------------------------------------------------------------
# Login — rollout flag off (existing behaviour must be byte-identical)
# ---------------------------------------------------------------------------


@override_settings(VEDA_JWT_AUTH=False)
def test_login_with_flag_off_returns_the_legacy_payload(client, user):
    """The pre-JWT contract, unchanged: placeholder access token, no refresh
    token, no expiry — so switching apps is invisible until the flag flips."""
    body = _login(client, "alice", PASSWORD).json()

    assert body == {
        "status_code": 200,
        "message": "Login successful.",
        "data": {
            "user_id": user.pk,
            "username": "alice",
            "display_name": "Alice",
            "access_token": LEGACY_ACCESS_TOKEN,
            "token_type": "Bearer",
        },
    }


@override_settings(VEDA_JWT_AUTH=False)
def test_login_with_flag_off_creates_no_token_rows(client, user):
    _login(client, "alice", PASSWORD)

    assert not OutstandingToken.objects.exists()


# ---------------------------------------------------------------------------
# Login — negative
# ---------------------------------------------------------------------------


@override_settings(VEDA_JWT_AUTH=True)
@pytest.mark.parametrize("payload", [
    {},
    {"username": "alice"},
    {"password": PASSWORD},
    {"username": "", "password": PASSWORD},
    {"username": "alice", "password": ""},
])
def test_login_rejects_malformed_payloads(client, user, payload):
    response = client.post(LOGIN_URL, payload, content_type="application/json")

    assert response.status_code == 400
    body = response.json()
    assert body["message"] == "Invalid request data."
    assert body["errors"]


@override_settings(VEDA_JWT_AUTH=True)
def test_login_rejects_wrong_password(client, user):
    response = _login(client, "alice", "wrong-password")

    assert response.status_code == 401
    assert response.json()["code"] == CODE_INVALID_CREDENTIALS


@override_settings(VEDA_JWT_AUTH=True)
def test_login_rejects_inactive_user(client):
    """A correct password against a deactivated account gets its own clear
    error (AccountInactive), not the generic invalid-credentials bucket —
    user's call: VEDA is an internal admin tool, so a real deactivated user
    deserves a clear reason rather than a confusing "invalid credentials"."""
    inactive = get_user_model().objects.create_user(
        username="carol", password=PASSWORD, is_staff=True)
    inactive.is_active = False
    inactive.save(update_fields=["is_active"])

    response = _login(client, "carol", PASSWORD)

    assert response.status_code == 401
    assert response.json()["code"] == CODE_ACCOUNT_INACTIVE

    # A WRONG password against the same deactivated account still can't tell
    # you the account exists — only a password that actually matches does.
    wrong_password = _login(client, "carol", "not-the-password")
    assert wrong_password.json()["code"] == CODE_INVALID_CREDENTIALS


# ---------------------------------------------------------------------------
# Login — security
# ---------------------------------------------------------------------------


@override_settings(VEDA_JWT_AUTH=True)
def test_login_does_not_leak_whether_an_account_exists(client, user):
    """Unknown username and a wrong password for a real one must still be one
    indistinguishable response — that pairing is the classic account oracle.

    A deactivated account is deliberately EXCLUDED from that bucket (user's
    call, see AccountInactive) — it gets its own response, checked separately
    in test_login_rejects_inactive_user."""
    unknown = _login(client, "nobody-here", PASSWORD)
    wrong_password = _login(client, "alice", "wrong-password")

    assert unknown.status_code == wrong_password.status_code == 401
    assert unknown.json() == wrong_password.json()


@override_settings(VEDA_JWT_AUTH=True)
def test_login_failure_response_leaks_nothing(client, user):
    body = _login(client, "alice", "wrong-password").json()

    assert set(body) == {"status_code", "message", "code"}
    assert body["message"] == "Invalid username or password."
    # No field-level detail, no stack trace, no hint about which half was wrong.
    assert "errors" not in body
    assert "password" not in body["message"].lower().replace("password.", "")


@override_settings(VEDA_JWT_AUTH=True)
def test_repeated_failures_from_one_source_are_locked_out(client, user):
    """The blocking counter: one source hammering one account is cut off before
    any further password hash is computed."""
    max_failures = 3
    with override_settings(VEDA_AUTH_LOGIN_MAX_FAILURES=max_failures):
        for _ in range(max_failures):
            assert _login(client, "alice", "wrong-password",
                          ip=ATTACKER_IP).status_code == 401

        locked = _login(client, "alice", "wrong-password", ip=ATTACKER_IP)

    assert locked.status_code == 429
    assert locked.json()["code"] == CODE_ACCOUNT_LOCKED


@override_settings(VEDA_JWT_AUTH=True)
def test_an_attacker_cannot_lock_out_a_legitimate_user(client, user):
    """C1 regression — this is the defect that made the first review fail.

    An anonymous caller who knows a username must NOT be able to deny service to
    its owner. The blocking counter is keyed per (account, source), so failures
    piled up from the attacker's address leave the real user — arriving from their
    own address, with the right password — completely unaffected.
    """
    with override_settings(VEDA_AUTH_LOGIN_MAX_FAILURES=3):
        for _ in range(10):
            _login(client, "alice", "guess", ip=ATTACKER_IP)

        attacker = _login(client, "alice", "guess", ip=ATTACKER_IP)
        victim = _login(client, "alice", PASSWORD, ip=VICTIM_IP)

    assert attacker.status_code == 429           # attacker is cut off...
    assert victim.status_code == 200             # ...and the owner still gets in
    assert victim.json()["data"]["refresh_token"]


@override_settings(VEDA_JWT_AUTH=True)
def test_account_wide_flood_never_refuses_the_correct_password(client, user):
    """The soft counter, from many addresses: wrong guesses become 429 (so clients
    back off and the flood is visible), but the account holder is still let in.

    An account-wide counter that refused correct passwords would recreate C1 with
    extra steps — any attacker with a handful of IPs could lock anyone out.
    """
    with override_settings(VEDA_AUTH_LOGIN_MAX_FAILURES=100,
                           VEDA_AUTH_LOGIN_ACCOUNT_MAX_FAILURES=4):
        for octet in range(4):                      # four failures, four addresses
            _login(client, "alice", "guess", ip=f"10.9.9.{octet}")

        flooded = _login(client, "alice", "guess", ip="10.9.9.50")
        owner = _login(client, "alice", PASSWORD, ip=VICTIM_IP)

    assert flooded.status_code == 429
    assert flooded.json()["code"] == CODE_ACCOUNT_LOCKED
    assert owner.status_code == 200


@override_settings(VEDA_JWT_AUTH=True)
def test_spoofed_forwarded_for_cannot_mint_fresh_lockout_quota(client, user):
    """The lockout ident must come from the proxy-validated entry, not the whole
    header. nginx appends the true peer LAST, so a caller pre-seeding
    X-Forwarded-For with fake hops must still land in its own bucket."""
    with override_settings(VEDA_AUTH_LOGIN_MAX_FAILURES=3):
        for i in range(3):
            # Same real peer each time, a different fake prefix every time.
            _login(client, "alice", "guess", ip=ATTACKER_IP,
                   HTTP_X_FORWARDED_FOR=f"1.2.3.{i}, {ATTACKER_IP}")

        still_locked = _login(client, "alice", "guess", ip=ATTACKER_IP,
                              HTTP_X_FORWARDED_FOR=f"9.9.9.9, {ATTACKER_IP}")

    assert still_locked.status_code == 429


@override_settings(VEDA_JWT_AUTH=True)
def test_lockout_of_an_unknown_username_is_not_an_oracle(client):
    """Hammering a username that does not exist must lock out identically, or
    the 401-vs-429 boundary itself would reveal which accounts are real."""
    with override_settings(VEDA_AUTH_LOGIN_MAX_FAILURES=2):
        for _ in range(2):
            _login(client, "ghost", "guess", ip=ATTACKER_IP)
        response = _login(client, "ghost", "guess", ip=ATTACKER_IP)

    assert response.status_code == 429


@override_settings(VEDA_JWT_AUTH=True)
def test_lockout_counter_ignores_username_case(client, user):
    """Alternating capitalisation must not reset the counter."""
    with override_settings(VEDA_AUTH_LOGIN_MAX_FAILURES=2):
        _login(client, "alice", "wrong-password", ip=ATTACKER_IP)
        _login(client, "ALICE", "wrong-password", ip=ATTACKER_IP)
        response = _login(client, "AlIcE", "wrong-password", ip=ATTACKER_IP)

    assert response.status_code == 429


@override_settings(VEDA_JWT_AUTH=True)
def test_successful_login_clears_the_failure_counter(client, user):
    """A user who mistypes, succeeds, then mistypes again must not be locked out
    by the earlier failures."""
    with override_settings(VEDA_AUTH_LOGIN_MAX_FAILURES=3):
        _login(client, "alice", "wrong-password")
        _login(client, "alice", "wrong-password")
        assert _login(client, "alice", PASSWORD).status_code == 200

        assert _login(client, "alice", "wrong-password").status_code == 401
        assert _login(client, "alice", "wrong-password").status_code == 401


def test_login_survives_a_cache_blip_between_incr_and_set(user):
    """C2 regression — the second defect that failed the first review.

    ``cache.incr`` raising ValueError (no counter yet) falls through to
    ``cache.set``. If that second call is not guarded independently, a Redis blip
    in the microseconds between them escapes as a 500 from the login path, because
    an exception raised inside an ``except`` clause is not caught by a later clause
    on the same ``try``. A cache fault must degrade to "no lockout", never to a
    server error.
    """

    with mock.patch("apps.authentication.services.cache") as broken:
        broken.get.return_value = 0
        broken.incr.side_effect = ValueError("key not found")      # counter missing
        broken.set.side_effect = ConnectionError("redis died now")  # blip right after

        with pytest.raises(InvalidCredentials):   # a 401, NOT a ConnectionError
            AuthService().login("alice", "wrong-password")


def test_lockout_fails_open_when_the_cache_is_unreachable(user):
    """A redis-cache outage must degrade to "no lockout", never to "nobody can
    log in" — the per-IP throttles still apply in that state."""

    with mock.patch("apps.authentication.services.cache") as broken:
        broken.get.side_effect = ConnectionError("redis down")
        broken.incr.side_effect = ConnectionError("redis down")
        broken.set.side_effect = ConnectionError("redis down")
        broken.delete.side_effect = ConnectionError("redis down")

        result = AuthService().login("alice", PASSWORD)

    assert result["user_id"] == user.pk


# ---------------------------------------------------------------------------
# Login — service-level unit tests (no HTTP)
# ---------------------------------------------------------------------------


def test_service_login_raises_invalid_credentials(user):
    with pytest.raises(InvalidCredentials):
        AuthService().login("alice", "wrong-password")


def test_service_login_raises_account_locked(user):
    with override_settings(VEDA_AUTH_LOGIN_MAX_FAILURES=1):
        with pytest.raises(InvalidCredentials):
            AuthService().login("alice", "wrong-password")
        with pytest.raises(AccountLocked):
            AuthService().login("alice", PASSWORD)


@override_settings(VEDA_JWT_AUTH=True)
def test_anonymous_login_is_not_blocked_by_csrf(user):
    """Login must work for a caller with no session and no CSRF token — that is
    every first-time browser client. Asserted against a CSRF-enforcing client,
    because the default test client silently skips the check."""
    strict = Client(enforce_csrf_checks=True)

    response = _login(strict, "alice", PASSWORD)

    assert response.status_code == 200


def test_auth_error_detail_never_reaches_the_client():
    """``AuthError`` renders its curated class-level message, never the exception
    argument — so a future ``raise InvalidCredentials(f"no user {username}")``
    debugging aid cannot leak onto the wire."""

    response = _error_response(InvalidCredentials("user 'alice' has no usable password"))

    assert response.data["message"] == "Invalid username or password."
    assert "alice" not in str(response.data)


def test_error_messages_never_name_the_missing_half():
    """Guard the copy itself: a future edit that "helpfully" says "no such user"
    would reintroduce enumeration without failing any other assertion."""
    for message in (InvalidCredentials.message, AccountLocked.message):
        lowered = message.lower()
        assert "not found" not in lowered
        assert "does not exist" not in lowered
        assert "no such" not in lowered
        assert "inactive" not in lowered
        assert "disabled" not in lowered


# ---------------------------------------------------------------------------
# Refresh — happy path
# ---------------------------------------------------------------------------


@override_settings(VEDA_JWT_AUTH=True)
def test_refresh_returns_a_new_token_pair(client, user):
    access, refresh_token = _tokens(client)

    response = _refresh(client, refresh_token)

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Token refreshed successfully."
    data = body["data"]
    assert data["user_id"] == user.pk
    assert data["username"] == "alice"
    assert data["access_token"] != access
    assert data["refresh_token"] != refresh_token
    assert data["token_type"] == "Bearer"
    assert data["expires_in"] > 0


@override_settings(VEDA_JWT_AUTH=True)
def test_refresh_can_be_chained(client, user):
    """Each rotation must hand out a token that is itself rotatable, or a session
    dies after one refresh."""
    _, refresh_token = _tokens(client)

    for _ in range(3):
        response = _refresh(client, refresh_token)
        assert response.status_code == 200
        refresh_token = response.json()["data"]["refresh_token"]


@override_settings(VEDA_JWT_AUTH=True)
def test_refresh_invalidates_the_old_token(client, user):
    """Single-use: the rotated token must be dead even though its ``exp`` is
    still days away."""
    _, refresh_token = _tokens(client)
    _refresh(client, refresh_token)

    replay = _refresh(client, refresh_token)

    assert replay.status_code == 401
    assert replay.json()["code"] == CODE_INVALID_TOKEN


@override_settings(VEDA_JWT_AUTH=True)
def test_refresh_blacklists_the_spent_token(client, user):
    _, refresh_token = _tokens(client)
    spent_jti = _RotatableRefreshToken(refresh_token)["jti"]

    _refresh(client, refresh_token)

    assert BlacklistedToken.objects.filter(token__jti=spent_jti).exists()


# ---------------------------------------------------------------------------
# Refresh — replay / reuse
# ---------------------------------------------------------------------------


@override_settings(VEDA_JWT_AUTH=True)
def test_replay_revokes_every_session_of_the_account(client, user):
    """A token presented twice means it was captured (we cannot tell which
    presenter is the thief), so every refresh token the account holds dies —
    including the one just handed out, and any from another device."""
    _, stolen = _tokens(client)
    _, other_device = _tokens(client)
    rotated = _refresh(client, stolen).json()["data"]["refresh_token"]

    replay = _refresh(client, stolen)

    assert replay.status_code == 401
    assert _refresh(client, rotated).status_code == 401
    assert _refresh(client, other_device).status_code == 401


@override_settings(VEDA_JWT_AUTH=True)
def test_concurrent_refresh_of_one_token_yields_exactly_one_winner(client, user):
    """The race the naive implementation loses.

    Both callers verify the *same* token before either has blacklisted it — the
    exact interleaving simplejwt's own rotation permits, reproduced here
    deterministically rather than with threads (whose scheduling would make this
    test flaky and whose sqlite connections would not share a transaction).

    The blacklist INSERT, not a SELECT, must decide: exactly one caller may spend
    the jti, and the loser must be treated as a replay.
    """
    _, refresh_token = _tokens(client)
    service = AuthService()

    first = _RotatableRefreshToken(refresh_token)
    second = _RotatableRefreshToken(refresh_token)

    assert service._spend(first) is True
    assert service._spend(second) is False


@override_settings(VEDA_JWT_AUTH=True)
def test_concurrent_refresh_second_caller_is_rejected_end_to_end(client, user):
    """The same race through the public API: the loser gets a 401, never a second
    valid token family."""

    _, refresh_token = _tokens(client)
    service = AuthService()

    # Force the loser's path: the token verified cleanly, but another caller
    # spent this jti in the window before the blacklist insert.
    with mock.patch.object(AuthService, "_spend", return_value=False):
        with pytest.raises(InvalidRefreshToken):
            service.refresh(refresh_token)


# ---------------------------------------------------------------------------
# Refresh — negative / security
# ---------------------------------------------------------------------------


@override_settings(VEDA_JWT_AUTH=True)
@pytest.mark.parametrize("payload", [{}, {"refresh_token": ""}, {"refresh": "x"}])
def test_refresh_rejects_malformed_payloads(client, user, payload):
    response = client.post(REFRESH_URL, payload, content_type="application/json")

    assert response.status_code == 400
    assert response.json()["message"] == "Invalid request data."


@override_settings(VEDA_JWT_AUTH=True)
def test_refresh_rejects_garbage(client, user):
    assert _refresh(client, "not-a-jwt").status_code == 401


@override_settings(VEDA_JWT_AUTH=True)
def test_refresh_rejects_a_forged_signature(client, user):
    """JWT forgery: a token whose payload was re-signed with the wrong key must be
    refused — and must NOT be able to trigger a revocation of the real user's
    sessions (verification precedes any state change)."""

    _, real = _tokens(client)
    claims = pyjwt.decode(real, options={"verify_signature": False})
    forged = pyjwt.encode(claims, "attacker-key-padded-to-32-bytes-min", algorithm="HS256")

    response = _refresh(client, forged)

    assert response.status_code == 401
    # The genuine token still works: a forgery cannot be used to lock a user out.
    assert _refresh(client, real).status_code == 200


@override_settings(VEDA_JWT_AUTH=True)
def test_refresh_rejects_an_expired_token(client, user):
    _, refresh_token = _tokens(client)
    token = _RotatableRefreshToken(refresh_token)
    token.set_exp(from_time=token.current_time - timedelta(days=30), lifetime=timedelta(days=1))

    assert _refresh(client, str(token)).status_code == 401


@override_settings(VEDA_JWT_AUTH=True)
def test_refresh_rejects_an_access_token(client, user):
    """Type confusion: an access token is signed with the same key, so only the
    ``token_type`` claim stops it being spent as a refresh token."""
    access, _ = _tokens(client)

    assert _refresh(client, access).status_code == 401


@override_settings(VEDA_JWT_AUTH=True)
def test_refresh_rejects_a_deactivated_user(client, user):
    """Deactivating an account must end it at the next refresh."""
    _, refresh_token = _tokens(client)
    user.is_active = False
    user.save(update_fields=["is_active"])

    assert _refresh(client, refresh_token).status_code == 401


@override_settings(VEDA_JWT_AUTH=True)
def test_refresh_rejects_a_deleted_user(client, user):
    _, refresh_token = _tokens(client)
    user.delete()

    assert _refresh(client, refresh_token).status_code == 401


@override_settings(VEDA_JWT_AUTH=True)
def test_rejecting_an_inactive_user_does_not_spend_the_token(client, user):
    """No pointless write on the reject path — and re-enabling the account
    restores the session rather than silently having burnt its token."""

    _, refresh_token = _tokens(client)
    user.is_active = False
    user.save(update_fields=["is_active"])
    _refresh(client, refresh_token)

    assert not BlacklistedToken.objects.exists()


@override_settings(VEDA_JWT_AUTH=True)
def test_refresh_failures_are_indistinguishable(client, user):
    """Garbage, forged, expired, wrong-type and already-spent must all look the
    same, so a captured token cannot be probed for its status."""

    access, spent = _tokens(client)
    _refresh(client, spent)
    _, live = _tokens(client)
    forged = pyjwt.encode(
        pyjwt.decode(live, options={"verify_signature": False}), "attacker-key-padded-to-32-bytes-min", algorithm="HS256")

    bodies = [_refresh(client, candidate).json()
              for candidate in ("not-a-jwt", forged, access, spent)]

    assert all(body == bodies[0] for body in bodies)
    assert set(bodies[0]) == {"status_code", "message", "code"}


@override_settings(VEDA_JWT_AUTH=False)
def test_refresh_is_inert_while_the_flag_is_off(user):
    """A token minted while the flag was on must not be spendable after it is
    turned off — least of all in exchange for a placeholder access token."""

    with override_settings(VEDA_JWT_AUTH=True):
        refresh_token = AuthService().login("alice", PASSWORD)["refresh_token"]

    with pytest.raises(InvalidRefreshToken):
        AuthService().refresh(refresh_token)
    assert not BlacklistedToken.objects.exists()


# ---------------------------------------------------------------------------
# Refresh — password-change revocation (C3)
# ---------------------------------------------------------------------------


@override_settings(VEDA_JWT_AUTH=True)
def test_password_change_revokes_existing_refresh_tokens(client, user):
    """C3 regression — the third defect that failed the first review.

    A user who changes their password because they suspect compromise must not
    leave the attacker a working refresh token. simplejwt enforces its
    CHECK_REVOKE_TOKEN claim for access tokens only, so rotation has to check it
    explicitly — without that, the stolen token stays renewable indefinitely.
    """
    _, refresh_token = _tokens(client)
    user.set_password("a-brand-new-password")
    user.save(update_fields=["password"])

    response = _refresh(client, refresh_token)

    assert response.status_code == 401
    assert response.json()["code"] == CODE_INVALID_TOKEN


@override_settings(VEDA_JWT_AUTH=True)
def test_password_change_does_not_spend_the_token(client, user):
    """Reject without a write, like the inactive-user path — the token is already
    worthless, so there is nothing to burn."""

    _, refresh_token = _tokens(client)
    user.set_password("a-brand-new-password")
    user.save(update_fields=["password"])
    _refresh(client, refresh_token)

    assert not BlacklistedToken.objects.exists()


@override_settings(VEDA_JWT_AUTH=True)
def test_a_token_carrying_no_revoke_claim_is_refused(client, user):
    """Fail closed: a token minted before CHECK_REVOKE_TOKEN was enabled cannot
    prove it is current, so it forces one re-login rather than being grandfathered
    in forever."""
    _, refresh_token = _tokens(client)
    token = _RotatableRefreshToken(refresh_token)
    del token.payload[jwt_settings.REVOKE_TOKEN_CLAIM]

    assert _refresh(client, str(token)).status_code == 401


@override_settings(VEDA_JWT_AUTH=True)
def test_unrelated_profile_edits_do_not_revoke_tokens(client, user):
    """The claim tracks the password, not the row: changing a display name must not
    sign the user out."""
    _, refresh_token = _tokens(client)
    user.first_name = "Alice Renamed"
    user.save(update_fields=["first_name"])

    assert _refresh(client, refresh_token).status_code == 200


# ---------------------------------------------------------------------------
# The access token actually authenticates (H1)
#
# The first review shipped 59 green tests without ever proving an issued access
# token was usable: override_settings(VEDA_JWT_AUTH=…) cannot re-install
# DEFAULT_AUTHENTICATION_CLASSES, which base.py computes at import. These tests
# mount a genuinely protected endpoint — JWTAuthentication + IsAuthenticated, the
# exact pair base.py installs when the flag is on — and drive it over HTTP.
# ---------------------------------------------------------------------------


class _ProtectedView(APIView):
    """Stands in for any endpoint that will require auth once RBAC lands."""

    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    throttle_classes = []

    def get(self, request):
        return DRFResponse({"user_id": request.user.pk, "username": request.user.username})


# The REAL root urlconf plus one protected route, so the auth endpoints keep their
# production paths (/api/v1/auth/...) and are exercised through the same routing
# the deployment uses — not a parallel test-only mount that could diverge from it.
urlpatterns = [
    *root_urlpatterns,
    path("protected", _ProtectedView.as_view()),
]

PROTECTED_URL = "/protected"


def _protected(client, token=None):
    headers = {"HTTP_AUTHORIZATION": f"Bearer {token}"} if token else {}
    return client.get(PROTECTED_URL, **headers)


# ROOT_URLCONF points at this module, so LOGIN_URL/REFRESH_URL/LOGOUT_URL keep
# resolving alongside the protected route.
_JWT_STACK = dict(VEDA_JWT_AUTH=True, ROOT_URLCONF=__name__)


@override_settings(**_JWT_STACK)
def test_an_issued_access_token_authenticates_a_protected_endpoint(client, user):
    access, _ = _tokens(client)

    response = _protected(client, access)

    assert response.status_code == 200
    assert response.json() == {"user_id": user.pk, "username": "alice"}


@override_settings(**_JWT_STACK)
def test_protected_endpoint_rejects_a_missing_token(client, user):
    assert _protected(client).status_code == 401


@override_settings(**_JWT_STACK)
@pytest.mark.parametrize("bad", ["not-a-jwt", "", "eyJhbGciOiJIUzI1NiJ9.e30.wrong"])
def test_protected_endpoint_rejects_unusable_tokens(client, user, bad):
    assert _protected(client, bad).status_code == 401


@override_settings(**_JWT_STACK)
def test_protected_endpoint_rejects_a_refresh_token(client, user):
    """Type confusion in the other direction: a refresh token must not be spendable
    as an access token, even though both are signed with the same key."""
    _, refresh_token = _tokens(client)

    assert _protected(client, refresh_token).status_code == 401


@override_settings(**_JWT_STACK)
def test_protected_endpoint_rejects_an_expired_access_token(client, user):
    token = AccessToken.for_user(user)
    token.set_exp(from_time=token.current_time - timedelta(hours=2),
                  lifetime=timedelta(hours=1))

    assert _protected(client, str(token)).status_code == 401


@override_settings(**_JWT_STACK)
def test_protected_endpoint_rejects_a_forged_access_token(client, user):
    access, _ = _tokens(client)
    claims = pyjwt.decode(access, options={"verify_signature": False})
    forged = pyjwt.encode(claims, "attacker-key-padded-to-32-bytes-min", algorithm="HS256")

    assert _protected(client, forged).status_code == 401


@override_settings(**_JWT_STACK)
def test_deactivating_a_user_rejects_their_access_token(client, user):
    """Enforced per request by JWTAuthentication, so suspension takes effect
    immediately rather than at the next refresh."""
    access, _ = _tokens(client)
    assert _protected(client, access).status_code == 200

    user.is_active = False
    user.save(update_fields=["is_active"])

    assert _protected(client, access).status_code == 401


@override_settings(**_JWT_STACK)
def test_password_change_rejects_existing_access_tokens(client, user):
    """The other half of C3: CHECK_REVOKE_TOKEN must be live for access tokens too."""
    access, _ = _tokens(client)
    assert _protected(client, access).status_code == 200

    user.set_password("a-brand-new-password")
    user.save(update_fields=["password"])

    assert _protected(client, access).status_code == 401


@override_settings(**_JWT_STACK)
def test_logout_does_not_immediately_invalidate_the_access_token(client, user):
    """Documents the accepted limit of stateless JWTs (AUTH_API_CONTRACT.md §3.1):
    logout kills the refresh token, while the access token remains valid until it
    expires. If this ever starts failing, the contract must be updated too."""
    access, refresh_token = _tokens(client)
    _logout(client, refresh_token)

    assert _refresh(client, refresh_token).status_code == 401   # session is over...
    assert _protected(client, access).status_code == 200        # ...but this lingers


def test_jwt_authentication_is_installed_exactly_when_the_flag_is_on():
    """Guards the base.py wiring itself: the tests above mount the auth class
    explicitly, so they would still pass if settings forgot to install it."""
    from django.conf import settings

    installed = any("simplejwt" in path for path in
                    settings.REST_FRAMEWORK["DEFAULT_AUTHENTICATION_CLASSES"])
    assert installed is bool(settings.VEDA_JWT_AUTH)


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------


@override_settings(VEDA_JWT_AUTH=True)
def test_logout_revokes_the_refresh_token(client, user):
    _, refresh_token = _tokens(client)

    response = _logout(client, refresh_token)

    assert response.status_code == 200
    assert response.json() == {"status_code": 200, "message": "Logout successful."}
    assert _refresh(client, refresh_token).status_code == 401


@override_settings(VEDA_JWT_AUTH=True)
def test_logout_is_idempotent(client, user):
    _, refresh_token = _tokens(client)

    first = _logout(client, refresh_token)
    second = _logout(client, refresh_token)

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


@override_settings(VEDA_JWT_AUTH=True)
@pytest.mark.parametrize("token", ["not-a-jwt", "", " "])
def test_logout_never_reports_failure_for_an_unusable_token(client, user, token):
    """Success regardless — otherwise logout tells a caller whether the token it
    holds is live. A blank token is still a 400 (malformed body), not a 500."""
    response = client.post(LOGOUT_URL, {"refresh_token": token},
                           content_type="application/json")

    assert response.status_code in (200, 400)
    assert response.status_code == (400 if not token.strip() else 200)


@override_settings(VEDA_JWT_AUTH=True)
def test_logout_of_an_already_rotated_token_succeeds(client, user):
    """A client that refreshed and then logged out with its stale token must not
    see an error — and must not trigger replay revocation of its live session."""
    _, first = _tokens(client)
    rotated = _refresh(client, first).json()["data"]["refresh_token"]

    response = _logout(client, first)

    assert response.status_code == 200
    # The live token is untouched: logout is not a replay report.
    assert _refresh(client, rotated).status_code == 200


@override_settings(VEDA_JWT_AUTH=True)
def test_logout_rejects_a_malformed_payload(client, user):
    response = client.post(LOGOUT_URL, {}, content_type="application/json")

    assert response.status_code == 400
    assert response.json()["message"] == "Invalid request data."


@override_settings(VEDA_JWT_AUTH=True)
def test_logout_ends_only_the_session_it_was_given(client, user):
    """One device logging out must not sign the user's other devices out."""
    _, phone = _tokens(client)
    _, laptop = _tokens(client)

    _logout(client, phone)

    assert _refresh(client, laptop).status_code == 200


@override_settings(VEDA_JWT_AUTH=True)
def test_logout_rejects_an_access_token_without_revoking_anything(client, user):
    """An access token is not a session handle; passing one must not blacklist a
    jti that belongs to the still-live refresh token."""
    access, refresh_token = _tokens(client)

    assert _logout(client, access).status_code == 200
    assert _refresh(client, refresh_token).status_code == 200


def test_logout_works_even_when_the_flag_is_off(user):
    """Revocation never grants anything, so turning the rollout flag off must not
    strand tokens issued while it was on."""
    with override_settings(VEDA_JWT_AUTH=True):
        refresh_token = AuthService().login("alice", PASSWORD)["refresh_token"]

    with override_settings(VEDA_JWT_AUTH=False):
        AuthService().logout(refresh_token)

    with override_settings(VEDA_JWT_AUTH=True):
        with pytest.raises(InvalidRefreshToken):
            AuthService().refresh(refresh_token)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_login_throttle_scope_is_configured():
    """The view's scope must exist in DEFAULT_THROTTLE_RATES — DRF raises at
    request time if it does not, so a typo here is a 500 on every login."""
    from django.conf import settings

    from apps.authentication.views import LoginView

    rates = settings.REST_FRAMEWORK["DEFAULT_THROTTLE_RATES"]
    assert LoginView.throttle_scope in rates
    assert rates[LoginView.throttle_scope]


def test_token_blacklist_app_is_installed():
    """Rotation, revocation and replay detection all depend on this app's tables
    (and on ``BlacklistMixin`` gating its behaviour on it being installed), and it
    is deliberately NOT behind the rollout flag."""
    from django.conf import settings

    assert "rest_framework_simplejwt.token_blacklist" in settings.INSTALLED_APPS


def test_refresh_throttle_scope_is_configured():
    from django.conf import settings

    from apps.authentication.views import TokenRefreshView

    rates = settings.REST_FRAMEWORK["DEFAULT_THROTTLE_RATES"]
    assert rates.get(TokenRefreshView.throttle_scope)


def test_routes_are_wired():
    """``auth/login`` is the path the frontend already calls — unchanged after the
    move to its own app; the other two sit beside it under the same prefix."""
    from django.urls import resolve

    from apps.authentication.views import LoginView, LogoutView, TokenRefreshView

    assert resolve(LOGIN_URL).func.view_class is LoginView
    assert resolve(REFRESH_URL).func.view_class is TokenRefreshView
    assert resolve(LOGOUT_URL).func.view_class is LogoutView


def test_chat_app_no_longer_serves_auth():
    """The dummy login was MOVED, not copied — a second implementation drifting
    behind the real one is exactly the duplication this refactor removes."""

    assert not hasattr(chat_views, "LoginView")
    assert not hasattr(chat_views, "_authenticate_login")
    assert not hasattr(chat_serializers, "LoginRequestSerializer")
