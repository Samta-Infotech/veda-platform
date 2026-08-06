"""apps.authentication.services — the authentication business logic.

Views are thin (``views.py``): validate a body, call in here, render the result.
Every decision about *whether* a caller is who they claim to be, and what tokens
they may hold, lives in this module.

Nothing here implements cryptography or password hashing:

  * Credentials are checked by ``django.contrib.auth.authenticate`` — Django's
    hashers provide the constant-time comparison, and ``ModelBackend`` already
    runs a dummy hash for an unknown username so the "no such user" branch costs
    the same as a wrong password (its documented timing-attack defence).
  * Tokens are minted, signed, verified and blacklisted by
    ``rest_framework_simplejwt`` (``SIMPLE_JWT`` in ``config/settings/base.py``).

What this module adds on top is the part the library does not give us safely:

  1. **Two-tier login failure counting.** The existing throttles are per-IP only
     (DRF ``AnonRateThrottle``, nginx ``limit_req``), so stuffing spread across
     many addresses is unbounded against one account. Counted in the shared
     ``redis-cache`` via Django's cache API — no new table. See the lockout
     section below for why there are two counters and why only one of them blocks.
  2. **A uniform failure surface.** Unknown user, wrong password and inactive
     account are one indistinguishable 401 (``MSG_INVALID_CREDENTIALS``), so the
     endpoint cannot be used to enumerate accounts.
  3. **Password-change revocation on the refresh path.** simplejwt enforces its
     ``CHECK_REVOKE_TOKEN`` claim only for access tokens (inside
     ``JWTAuthentication.get_user``); rotation would otherwise keep honouring a
     refresh token minted under a password that has since been changed.

Error contract: every expected failure is an ``AuthError`` subclass carrying a
stable ``code`` and a safe, user-facing ``message``. Unexpected failures are left
to propagate — DRF turns them into a 500 and the traceback goes to the log, never
onto the wire.
"""
from __future__ import annotations

import hashlib
import logging
import sys

from django.conf import settings
from django.contrib.auth import authenticate, get_user_model
from django.core.cache import cache
from django.db import transaction
from django.utils import timezone
from rest_framework.throttling import BaseThrottle
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.settings import api_settings as jwt_settings
from rest_framework_simplejwt.token_blacklist.models import BlacklistedToken, OutstandingToken
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.utils import get_md5_hash_password

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client-facing error contract. Codes are stable and safe to branch on in the
# frontend; messages are the only copy that ever reaches a caller. Neither ever
# reveals which half of a credential pair was wrong, whether the account exists,
# or anything about the token internals.
# ---------------------------------------------------------------------------
CODE_INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
MSG_INVALID_CREDENTIALS = "Invalid username or password."
CODE_ACCOUNT_LOCKED = "ACCOUNT_LOCKED"
MSG_ACCOUNT_LOCKED = "Too many failed login attempts. Please try again later."
CODE_INVALID_TOKEN = "INVALID_TOKEN"  # noqa: S105 — an error code, not a credential
MSG_INVALID_TOKEN = "Invalid or expired token."

# OAuth2 bearer-token scheme name returned to the client, matching the
# ``AUTH_HEADER_TYPES`` simplejwt accepts on the way back in.
TOKEN_TYPE = "Bearer"  # noqa: S105 — a scheme name, not a credential

# Access-token value returned while VEDA_JWT_AUTH is off — byte-identical to what
# the pre-JWT LoginView (apps/chat) returned, so the existing frontend contract is
# unchanged until the flag is switched on. See ``jwt_enabled``.
LEGACY_ACCESS_TOKEN = "dummy_access_token"  # noqa: S105 — placeholder, not a secret

# Lockout defaults, overridable per deployment via settings (which read the
# environment, per §9a). See the lockout section of AuthService for the semantics.
_DEFAULT_MAX_FAILURES = 10           # per (account, source IP) — hard refusal
_DEFAULT_ACCOUNT_MAX_FAILURES = 50   # account-wide — soft, never blocks a correct password
_DEFAULT_LOCKOUT_SECONDS = 300

_FAILURE_KEY_PREFIX = "veda:auth:login-failures:"

# Client identity used when the service is called outside an HTTP request (a
# management command, a test): every such call shares one bucket, which is correct
# — there is no source address to distinguish them by.
_NO_REQUEST_IDENT = "-"


def jwt_enabled() -> bool:
    """Whether real JWTs are issued (``VEDA_JWT_AUTH``, default off).

    The single source of truth for the rollout flag, so the service, the DRF
    authentication-class wiring and the tests cannot disagree about it. While it
    is off, login answers with the legacy placeholder token and every other
    endpoint keeps its current behaviour untouched.
    """
    return bool(getattr(settings, "VEDA_JWT_AUTH", False))


class AuthError(Exception):
    """An expected authentication failure with a safe client rendering.

    Subclasses set ``code``/``message``; the HTTP status is chosen by the view
    (``views._ERROR_STATUS``) so this layer stays free of HTTP concerns.
    """

    code = "AUTH_ERROR"
    message = "Authentication failed."


class InvalidCredentials(AuthError):
    """Wrong password, unknown username, or an inactive account — deliberately
    one single error, so the response cannot distinguish the three."""

    code = CODE_INVALID_CREDENTIALS
    message = MSG_INVALID_CREDENTIALS


class AccountLocked(AuthError):
    """The per-account failure counter is above its threshold.

    Not an enumeration leak: the counter is keyed by the *submitted* username, so
    hammering a username that does not exist locks that string out exactly the
    same way a real account would.
    """

    code = CODE_ACCOUNT_LOCKED
    message = MSG_ACCOUNT_LOCKED


class InvalidRefreshToken(AuthError):
    """The presented refresh token is unusable — malformed, wrongly signed,
    expired, of the wrong type, already spent, replayed, or belonging to an
    account that is gone or disabled.

    One error for all of them on purpose: telling a caller *which* of those it was
    tells an attacker holding a captured token whether it is worth more effort.
    """

    code = CODE_INVALID_TOKEN
    message = MSG_INVALID_TOKEN


class _RotatableRefreshToken(RefreshToken):
    """A ``RefreshToken`` whose blacklist lookup is deferred to the rotation step.

    simplejwt checks the blacklist while *constructing* the token
    (``BlacklistMixin.verify`` -> ``check_blacklist``). That read is a TOCTOU
    window: two concurrent refreshes of the same token both SELECT "not
    blacklisted", both proceed, and both mint a fresh family — so the one thing
    rotation exists to catch (a captured token being spent alongside the real
    client's) slips through.

    Deferring the check makes the blacklist INSERT the arbiter instead
    (``AuthService._spend``): the unique constraint on ``BlacklistedToken.token``
    means exactly one caller can ever spend a given ``jti``, and the loser is
    reported by the database rather than by a racy query. It also removes one
    SELECT from the happy path.

    Nothing security-relevant is skipped: ``Token.verify`` still enforces the
    signature, ``exp``, the presence of ``jti`` and the ``token_type`` claim before
    this token is trusted at all. That ordering matters — it is what stops a forged
    or replayed-but-unsigned token from being able to trigger a revocation of a
    real user's sessions.
    """

    def check_blacklist(self) -> None:  # noqa: D102 — see class docstring
        return


class AuthService:
    """Authentication operations for one HTTP request.

    ``request`` is optional so the service is usable (and testable) outside a
    request — it is only needed to pass through to the auth backends and to tag
    log lines with the ambient request id.
    """

    def __init__(self, request=None):
        self._request = request

    # -- login --------------------------------------------------------------

    def login(self, username: str, password: str) -> dict:
        """Authenticate credentials and issue tokens.

        Args:
            username: Submitted username, already validated as a non-blank string.
            password: Submitted password, likewise. Never logged.

        Returns:
            The response ``data`` payload: identity fields plus tokens.

        Raises:
            AccountLocked: too many recent failures from this source for this
                username, or an account-wide flood *and* a wrong password.
            InvalidCredentials: unknown user, wrong password, or inactive account.
        """
        source_key = self._failure_key(username, self._client_ident())
        account_key = self._failure_key(username)

        # The per-source counter is the only one that blocks, and it is checked
        # *before* the hash comparison — refusing early is the point of a lockout
        # (a password check costs ~300ms of CPU, so an attacker must not be able to
        # spend it at will). Being per-source, a third party cannot trip it on a
        # legitimate user's behalf.
        if self._is_locked(source_key, self._max_failures()):
            logger.warning("auth login blocked: too many failures from this source "
                           "username=%s %s", username, self._log_context())
            raise AccountLocked()

        user = authenticate(self._request, username=username, password=password)
        # ModelBackend already rejects inactive users (``user_can_authenticate``),
        # so this second guard is defence-in-depth for a future custom backend —
        # and it collapses into the same generic error either way.
        if user is None or not user.is_active:
            source_failures = self._record_failure(source_key)
            account_failures = self._record_failure(account_key)
            logger.warning("auth login failed username=%s source_failures=%s "
                           "account_failures=%s %s", username, source_failures,
                           account_failures, self._log_context())
            # Account-wide flood: report it as a lockout so the client backs off and
            # the event is distinguishable in logs — but ONLY for a wrong password.
            # A correct password is never refused on this counter, which is what
            # stops it being used to deny service to the real account holder.
            if account_failures >= self._account_max_failures():
                logger.warning("auth login: account-wide failure threshold exceeded "
                               "username=%s (possible distributed attack) %s",
                               username, self._log_context())
                raise AccountLocked()
            raise InvalidCredentials()

        self._clear_failures(source_key)
        self._clear_failures(account_key)
        payload = {**self._identity(user), **self._issue_tokens(user)}
        logger.info("auth login succeeded user_id=%s username=%s jwt=%s %s",
                    user.pk, user.username, jwt_enabled(), self._log_context())
        return payload

    # -- refresh ------------------------------------------------------------

    def refresh(self, raw_refresh_token: str) -> dict:
        """Rotate a refresh token: spend the old one, issue a fresh pair.

        The old token is invalidated whether or not the caller ever sees the new
        pair, so a refresh token is strictly single-use. A second attempt to spend
        the same token is treated as a replay — the token was captured, or a client
        is retrying blindly — and revokes every refresh token the account holds,
        because at that point we cannot tell the legitimate holder from the thief.

        Args:
            raw_refresh_token: The encoded refresh token, validated only as a
                non-blank string. Fully untrusted.

        Returns:
            The response ``data`` payload: identity fields plus a new token pair.

        Raises:
            InvalidRefreshToken: for every failure mode, indistinguishably.
        """
        # With the rollout flag off no real token is ever issued, so nothing can
        # legitimately be rotated. Checked first, before the old token is spent:
        # otherwise flipping the flag off with live tokens in the wild would burn
        # them and hand back a placeholder in exchange.
        if not jwt_enabled():
            logger.info("auth refresh rejected: VEDA_JWT_AUTH is off %s", self._log_context())
            raise InvalidRefreshToken()

        token = self._parse_refresh_token(raw_refresh_token)
        user_id = token.get(jwt_settings.USER_ID_CLAIM)

        # Loaded before spending the token: one query, whose result is reused to
        # mint the new pair (no second lookup), and no pointless write when the
        # account is gone or disabled.
        user = self._load_active_user(user_id)
        if user is None:
            logger.warning("auth refresh rejected: no active user for user_id=%s %s",
                           user_id, self._log_context())
            raise InvalidRefreshToken()

        # Not spent on this reject path, for the same reason as the inactive-user
        # branch above: the token is already worthless, so there is nothing to burn.
        if not self._password_unchanged(token, user):
            logger.warning("auth refresh rejected: token predates a password change "
                           "user_id=%s %s", user_id, self._log_context())
            raise InvalidRefreshToken()

        if not self._spend(token):
            # Lost the race for this jti, or the token was already spent earlier:
            # either way this exact token is being presented for the second time.
            revoked = self._revoke_all_for_user(user_id)
            logger.warning(
                "auth refresh REPLAY detected: jti already spent — revoked all "
                "refresh tokens for user_id=%s (candidates=%s) %s",
                user_id, revoked, self._log_context())
            raise InvalidRefreshToken()

        payload = {**self._identity(user), **self._issue_tokens(user)}
        logger.info("auth refresh succeeded user_id=%s %s", user.pk, self._log_context())
        return payload

    # -- logout -------------------------------------------------------------

    def logout(self, raw_refresh_token: str) -> None:
        """Revoke one refresh token, ending that session.

        Idempotent by contract: logging out twice, or with a token that was already
        rotated away, expired, or never valid at all, is a success. There is no
        state in which a caller should be told "your logout failed" — that only
        leaks whether the token they hold is live, and it gives a client no action
        it could usefully take.

        Deliberately NOT gated on ``jwt_enabled()``: revocation never *grants*
        anything, so a deployment that turned the flag off must still be able to
        kill the tokens it issued while it was on.

        Note the limit inherent to stateless JWTs: this ends the caller's ability
        to obtain new access tokens, but an access token already issued stays valid
        until it expires. That is why ``ACCESS_TOKEN_LIFETIME`` is short, and the
        reason a per-request access-token denylist (a DB read on every single API
        call) is not worth its cost here.

        Args:
            raw_refresh_token: The token to revoke. Fully untrusted.
        """
        try:
            token = _RotatableRefreshToken(raw_refresh_token)
        except TokenError as exc:
            # Unusable token: nothing exists to revoke. The reason is logged, never
            # returned — a 200 here reveals nothing an attacker did not already know.
            logger.info("auth logout: nothing to revoke (%s) %s", exc, self._log_context())
            return

        # Same single "make this token dead" primitive rotation uses; whether we or
        # a previous call did the insert is exactly the difference logout ignores.
        self._spend(token)
        logger.info("auth logout succeeded user_id=%s %s",
                    token.get(jwt_settings.USER_ID_CLAIM), self._log_context())

    def _parse_refresh_token(self, raw_refresh_token: str):
        """Decode and fully verify a refresh token (signature, ``exp``, ``jti``,
        ``token_type``) — everything except the blacklist, which the rotation step
        owns (see ``_RotatableRefreshToken``).

        The library's reason for rejecting is logged but never returned: "signature
        invalid" vs "expired" vs "wrong type" is exactly the feedback an attacker
        probing a captured token wants.
        """
        try:
            return _RotatableRefreshToken(raw_refresh_token)
        except TokenError as exc:
            logger.warning("auth refresh rejected: unusable token (%s) %s",
                           exc, self._log_context())
            raise InvalidRefreshToken() from exc

    @staticmethod
    def _spend(token) -> bool:
        """Consume a refresh token exactly once. True if this caller spent it.

        The arbiter is the unique constraint on ``BlacklistedToken.token``, not the
        surrounding transaction: concurrent callers both attempt the INSERT and the
        database picks one winner, so ``created is False`` is a definitive "someone
        already spent this jti". ``get_or_create`` absorbs the loser's
        ``IntegrityError`` in its own savepoint, which is why the enclosing
        transaction survives it.
        """
        with transaction.atomic():
            _, created = token.blacklist()
        return created

    @staticmethod
    def _revoke_all_for_user(user_id) -> int:
        """Blacklist every outstanding refresh token of one account.

        The response to a detected replay: we cannot distinguish the legitimate
        holder from whoever captured the token, so every session is ended and both
        parties must re-authenticate. simplejwt has no token-family concept to
        revoke a narrower set.

        Already-expired tokens are skipped: they are refused by the ``exp`` check
        anyway, so blacklisting them buys no security and would make this scale with
        an account's entire history rather than its live sessions.

        ``ignore_conflicts`` makes this safe to run concurrently (and idempotent
        over already-blacklisted tokens). Returns the number of candidate rows
        considered — not all of them are necessarily new insertions.
        """
        if not user_id:
            return 0
        outstanding = list(OutstandingToken.objects.filter(
            user_id=user_id, expires_at__gt=timezone.now()))
        BlacklistedToken.objects.bulk_create(
            [BlacklistedToken(token=row) for row in outstanding], ignore_conflicts=True)
        return len(outstanding)

    @staticmethod
    def _password_unchanged(token, user) -> bool:
        """Whether the token was issued under the account's current password.

        simplejwt stamps an md5 of the stored password hash into every token when
        ``CHECK_REVOKE_TOKEN`` is on, but only enforces it for ACCESS tokens inside
        ``JWTAuthentication.get_user``. Rotation has to check it here, or changing a
        password would leave every refresh token minted before the change fully
        usable — and renewable forever, since each rotation mints a fresh pair.

        A token with no claim at all is treated as changed (fail closed): that is
        what a token issued *before* this setting was enabled looks like, and the
        safe response to "cannot prove this is current" is one forced re-login, not
        an indefinite grandfather clause.
        """
        if not jwt_settings.CHECK_REVOKE_TOKEN:
            return True
        claim = token.get(jwt_settings.REVOKE_TOKEN_CLAIM)
        return bool(claim) and claim == get_md5_hash_password(user.password)

    @staticmethod
    def _load_active_user(user_id):
        """The token's subject, or None if it no longer exists or was disabled.

        Deactivating an account therefore ends it at the next refresh: the access
        token already issued stays valid until it expires (that is inherent to
        stateless JWTs, and why the access lifetime is short), but no new one can
        be obtained.
        """
        if not user_id:
            return None
        return get_user_model().objects.filter(
            **{jwt_settings.USER_ID_FIELD: user_id}, is_active=True).first()

    # -- token issuance -----------------------------------------------------

    @staticmethod
    def _identity(user) -> dict:
        """Identity fields of the login response. ``display_name`` keeps the
        pre-JWT view's rule (first name, else username) so the contract the
        frontend already consumes is unchanged."""
        return {
            "user_id": user.pk,
            "username": user.username,
            "display_name": user.first_name or user.username,
        }

    @staticmethod
    def _issue_tokens(user) -> dict:
        """A fresh access/refresh pair — or the legacy placeholder while the
        rollout flag is off.

        ``RefreshToken.for_user`` also records the token's ``jti`` in
        ``OutstandingToken``, which is what later makes rotation and revocation
        possible; the access token is derived from it so both expire relative to
        the same instant.
        """
        if not jwt_enabled():
            return {"access_token": LEGACY_ACCESS_TOKEN, "token_type": TOKEN_TYPE}

        refresh = RefreshToken.for_user(user)
        return {
            "access_token": str(refresh.access_token),
            "refresh_token": str(refresh),
            "token_type": TOKEN_TYPE,
            "expires_in": int(jwt_settings.ACCESS_TOKEN_LIFETIME.total_seconds()),
        }

    # -- login lockout ------------------------------------------------------
    #
    # TWO counters, because one cannot do both jobs safely:
    #
    #   * (account, source IP) — the only one that BLOCKS, and it blocks before any
    #     password hash is computed. Bounds one source guessing at one account, and
    #     an attacker cannot make failures appear against someone else's address, so
    #     it cannot be used to lock a real user out.
    #   * account-wide — catches stuffing distributed over many addresses, but is
    #     SOFT: it only ever changes a *wrong* password's answer from 401 to 429.
    #     A correct password is always honoured. An account-wide counter that
    #     refused correct passwords would hand any anonymous caller a denial of
    #     service against any username they know, which is exactly the defect this
    #     replaced.
    #
    # Every cache operation below is fail-OPEN: if redis-cache is unreachable the
    # counters degrade to "no lockout" rather than failing the login. A cache outage
    # must not lock every account in the deployment out, and the per-IP throttles
    # (DRF + nginx) remain in force regardless. Same fail-soft posture as
    # apps/query/views._audit and apps/query/scope._ready_source_ids.

    def _client_ident(self) -> str:
        """The client's identity for lockout bucketing.

        Delegates to DRF's ``BaseThrottle.get_ident`` — the same function the
        project's existing throttles already use — so there is one definition of
        "which client is this" and it honours ``NUM_PROXIES`` (see base.py). Using a
        raw ``X-Forwarded-For`` here instead would let an attacker mint unlimited
        fresh quota by varying a header they control.
        """
        if self._request is None:
            return _NO_REQUEST_IDENT
        return BaseThrottle().get_ident(self._request) or _NO_REQUEST_IDENT

    @staticmethod
    def _failure_key(username: str, ident: str = "") -> str:
        """Cache key for a failure counter, account-wide or per source.

        Both parts are hashed, not embedded: keys are visible to anyone reading the
        cache (``SCAN``, monitoring dashboards) and a raw username there is a
        needless PII leak. ``casefold`` groups case variants into one counter so an
        attacker cannot reset the count by alternating capitalisation. The username
        is length-prefixed so ("ab", "c") and ("a", "bc") cannot collide.
        """
        name = username.casefold()
        material = f"{len(name)}:{name}:{ident}"
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        return f"{_FAILURE_KEY_PREFIX}{digest[:32]}"

    @classmethod
    def _is_locked(cls, key: str, threshold: int) -> bool:
        try:
            failures = int(cache.get(key) or 0)
        except Exception:  # noqa: BLE001 — cache outage must not fail the login
            cls._log_cache_outage("lockout check unavailable")
            return False
        return failures >= threshold

    @classmethod
    def _record_failure(cls, key: str) -> int:
        """Increment (and start, on first failure) the counter; returns the count.

        Two simultaneous *first* failures can both fall through to the ``set`` and
        each write 1, so a burst may be undercounted by a few. That is the safe
        direction for a lockout — it can delay a lock, never cause a spurious one —
        and avoiding it would cost a Lua script or a lock for no security gain,
        since the per-IP throttles bound burst rate anyway.

        The two cache calls are guarded SEPARATELY on purpose: an exception raised
        inside an ``except`` clause is not caught by a later clause on the same
        ``try``, so folding these together silently turns a cache blip between the
        ``incr`` and the ``set`` into a 500 on the login path.
        """
        try:
            return int(cache.incr(key))
        except ValueError:
            pass  # no counter yet (or it just expired) — start a fresh window below
        except Exception:  # noqa: BLE001 — cache outage must not fail the login
            cls._log_cache_outage("failure counter unavailable")
            return 0

        try:
            cache.set(key, 1, timeout=cls._lockout_seconds())
            return 1
        except Exception:  # noqa: BLE001 — cache outage must not fail the login
            cls._log_cache_outage("failure counter could not be started")
            return 0

    @classmethod
    def _clear_failures(cls, key: str) -> None:
        """Reset a counter after a success, so a legitimate user who mistyped their
        password a few times is not locked out later."""
        try:
            cache.delete(key)
        except Exception:  # noqa: BLE001 — cache outage must not fail the login
            cls._log_cache_outage("failure counter reset failed")

    @staticmethod
    def _log_cache_outage(what: str) -> None:
        """One line per cache failure, deliberately WITHOUT ``exc_info``.

        A redis-cache outage makes this fire on every login attempt; a full Redis
        traceback each time (roughly thirty lines, several times per request) buries
        every other signal in the log exactly when an operator needs it. The
        exception's own message names the host and the error, which is what
        diagnosis actually needs.
        """
        exc = sys.exc_info()[1]
        logger.warning("auth lockout: %s (%s: %s); failing open",
                       what, type(exc).__name__, exc)

    @staticmethod
    def _max_failures() -> int:
        """Per (account, source) threshold — the blocking one."""
        return int(getattr(settings, "VEDA_AUTH_LOGIN_MAX_FAILURES", _DEFAULT_MAX_FAILURES))

    @staticmethod
    def _account_max_failures() -> int:
        """Account-wide threshold — soft; never refuses a correct password."""
        return int(getattr(settings, "VEDA_AUTH_LOGIN_ACCOUNT_MAX_FAILURES",
                           _DEFAULT_ACCOUNT_MAX_FAILURES))

    @staticmethod
    def _lockout_seconds() -> int:
        return int(getattr(settings, "VEDA_AUTH_LOGIN_LOCKOUT_SECONDS", _DEFAULT_LOCKOUT_SECONDS))

    # -- logging ------------------------------------------------------------

    def _log_context(self) -> str:
        """``request_id``/``ip`` suffix shared by every auth audit line.

        Uses the same ``_client_ident`` the lockout keys off, so an audit line and
        the counter it describes can never disagree about which client acted — and
        so the logged address is the proxy-validated one rather than a header the
        caller chose.
        """
        return (f"request_id={getattr(self._request, 'request_id', '')} "
                f"ip={self._client_ident()}")
