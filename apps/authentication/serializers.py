"""apps.authentication.serializers — request-payload validation for the auth endpoints.

Same contract as ``apps.chat.serializers`` (see its module docstring): these are
INPUT-only serializers, constructed as ``Serializer(data=request.data)`` and read
via ``.validated_data``; none is ever used to render a response. Response shaping
is done explicitly in ``views.py``.

Validation here is deliberately SHALLOW — presence, type, and blank-ness only. It
is the first gate (a malformed body becomes a 400 before any work starts), never
the authentication decision itself: that lives in ``services.AuthService``. A
serializer that accepts a string must not be mistaken for one that proves the
string is a valid credential or a live token.
"""
from rest_framework import serializers


class LoginRequestSerializer(serializers.Serializer):
    """Credentials for ``POST /api/v1/auth/login``.

    Both fields are mandatory and non-blank so an empty submission is rejected
    here rather than reaching ``django.contrib.auth.authenticate`` — which treats
    a blank password as a normal failed attempt and would waste a password-hash
    comparison on it.

    No ``max_length``/format constraint is imposed: the credential is checked
    against the configured auth backend, and a stricter rule here would only
    reveal which inputs are *shaped* like real usernames.
    """

    username = serializers.CharField(required=True, allow_blank=False)
    password = serializers.CharField(required=True, allow_blank=False)


class RefreshTokenRequestSerializer(serializers.Serializer):
    """Body of ``POST /api/v1/auth/refresh`` **and** ``POST /api/v1/auth/logout``.

    One class serves both endpoints because both take exactly one field with
    identical rules — the refresh token to rotate, or the one to revoke. Splitting
    it into two identical serializers would be duplicated validation with two
    places to keep in sync; the endpoints differ in what they *do* with the
    token (``AuthService.refresh`` vs ``AuthService.logout``), not in how the
    payload is shaped.

    Nothing about the token's validity is asserted here — signature, expiry, type
    and revocation state are all checked by ``AuthService`` against the signing
    key and the blacklist. This only guarantees a non-empty string was sent, so a
    missing field is a 400 rather than a misleading 401.
    """

    refresh_token = serializers.CharField(required=True, allow_blank=False)
