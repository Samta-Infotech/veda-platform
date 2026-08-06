"""apps.authentication.views — thin DRF views for the auth endpoints.

Each view does exactly three things: validate the body with a serializer, hand
the validated values to ``AuthService``, and render the outcome. No credential
check, no token handling and no revocation logic lives here — that is all in
``services.py``, which is why these views have no branches beyond "valid payload?"
and "did the service raise?".

The response envelope comes from ``apps.core.api`` — the platform's single
definition of it — so the frontend sees one shape across every endpoint. Auth
failures add a stable ``code`` field, because a client must be able to tell "wrong
password" from "locked out" from "expired token" without parsing human copy.
"""
from __future__ import annotations

import logging

from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from apps.core import api
from apps.core.messages import MESSAGES

from .serializers import LoginRequestSerializer, RefreshTokenRequestSerializer
from .services import (
    AccountLocked,
    AuthError,
    AuthService,
    InvalidCredentials,
    InvalidRefreshToken,
)

logger = logging.getLogger(__name__)

# Exception class -> HTTP status. Keeping the mapping here is what lets
# ``services.py`` stay free of HTTP concerns: it names *what* went wrong, this
# names how that is expressed over HTTP.
_ERROR_STATUS = {
    InvalidCredentials: status.HTTP_401_UNAUTHORIZED,
    AccountLocked: status.HTTP_429_TOO_MANY_REQUESTS,
    InvalidRefreshToken: status.HTTP_401_UNAUTHORIZED,
}
_FALLBACK_ERROR_STATUS = status.HTTP_401_UNAUTHORIZED


def _error_response(exc: AuthError):
    """Render an expected auth failure. Only the exception's curated ``message``
    and ``code`` reach the client — never ``str(exc)``, a traceback, or anything
    the token library said about *why* a token was unusable."""
    return api.error(exc.message, _ERROR_STATUS.get(type(exc), _FALLBACK_ERROR_STATUS),
                 code=exc.code)


class LoginView(APIView):
    """POST /api/v1/auth/login {username, password} -> access + refresh tokens.

    Unauthenticated by definition (``AllowAny``), and throttled on its own
    ``login`` scope — stricter than the global ``anon`` rate, because this is the
    one endpoint where an anonymous caller can guess a secret. That per-IP bound
    composes with the per-account lockout in ``AuthService``: neither one alone
    covers both a single attacker hammering many accounts and many IPs hammering
    one account.
    """

    permission_classes = [AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "login"

    def post(self, request):
        serializer = LoginRequestSerializer(data=request.data)
        if not serializer.is_valid():
            logger.warning("auth login rejected: invalid payload — %s", serializer.errors)
            return api.invalid_payload(serializer.errors)

        credentials = serializer.validated_data
        try:
            data = AuthService(request).login(
                credentials["username"], credentials["password"])
        except AuthError as exc:
            return _error_response(exc)

        return api.success(MESSAGES["auth"]["login_success"], data)


class TokenRefreshView(APIView):
    """POST /api/v1/auth/refresh {refresh_token} -> a new access + refresh pair.

    ``AllowAny`` because the refresh token *is* the credential — the caller's
    access token is expected to be expired by the time they get here, so requiring
    an authenticated request would make the endpoint unusable for its one purpose.

    Throttled on its own scope: this is the other endpoint an anonymous caller can
    submit a guessed or captured secret to.
    """

    permission_classes = [AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "token_refresh"

    def post(self, request):
        serializer = RefreshTokenRequestSerializer(data=request.data)
        if not serializer.is_valid():
            logger.warning("auth refresh rejected: invalid payload — %s", serializer.errors)
            return api.invalid_payload(serializer.errors)

        try:
            data = AuthService(request).refresh(serializer.validated_data["refresh_token"])
        except AuthError as exc:
            return _error_response(exc)

        return api.success(MESSAGES["auth"]["token_refreshed"], data)


class LogoutView(APIView):
    """POST /api/v1/auth/logout {refresh_token} -> 200, always.

    ``AllowAny`` for the same reason as refresh: the refresh token is the
    credential, and a client whose access token has already expired must still be
    able to end its session. Presenting a token you hold in order to destroy it is
    not a privilege the platform needs to guard.

    No error path and no ``data``: the only failure this endpoint could report is
    "that token was already dead", which is indistinguishable from success in
    every way that matters to a client — and reporting it would turn logout into
    an oracle for whether a captured token is still live.
    """

    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RefreshTokenRequestSerializer(data=request.data)
        if not serializer.is_valid():
            logger.warning("auth logout rejected: invalid payload — %s", serializer.errors)
            return api.invalid_payload(serializer.errors)

        AuthService(request).logout(serializer.validated_data["refresh_token"])

        # No ``data``: there is nothing to report beyond the outcome.
        return api.success(MESSAGES["auth"]["logout_success"])
