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
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError as DjangoValidationError
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
    #: Sent by the admin frontend only, to claim this login is for it. Absent
    #: entirely from the normal-user frontend's requests — not merely False, so
    #: ``AuthService.login`` can tell "not claiming admin" apart from "claiming
    #: admin explicitly", and only verify the claim when one was actually made.
    is_admin = serializers.BooleanField(required=False)


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


class PasswordChangeRequestSerializer(serializers.Serializer):
    """Body of ``POST /api/v1/auth/password/change``.

    Whether ``current_password`` is actually correct is a credential decision, not
    a shape one — that check belongs to ``AuthService.change_password``, mirroring
    every other place this codebase draws that line (``LoginRequestSerializer``
    does not check the password is *right*, only that it was *sent*).

    ``new_password`` policy IS validated here, via the same ``validate_password()``
    call ``UserCreateSerializer`` already uses — one call site's worth of policy
    logic, reused, not re-implemented. Requires ``context={"request": request}`` so
    ``UserAttributeSimilarityValidator`` can compare against the acting user's own
    username/email, exactly as it does on create.
    """

    current_password = serializers.CharField(
        required=True, allow_blank=False, write_only=True, trim_whitespace=False)
    new_password = serializers.CharField(
        required=True, allow_blank=False, write_only=True, trim_whitespace=False)

    def validate(self, attrs):
        request = self.context.get("request")
        user = getattr(request, "user", None)
        try:
            validate_password(attrs["new_password"], user=user)
        except DjangoValidationError as exc:
            raise serializers.ValidationError({"new_password": list(exc.messages)}) from exc
        return attrs
