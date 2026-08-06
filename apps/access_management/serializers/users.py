"""User-administration request validation.

Same contract as the other serializer modules in this project (see
``apps/chat/serializers.py``): INPUT-only, plain ``serializers.Serializer``,
constructed as ``Serializer(data=request.data)`` and read via ``.validated_data``.
Never used to render a response — response shaping belongs to the view.

Deliberately NOT a ``ModelSerializer``, for two reasons:

  * Consistency — every serializer in this codebase is an input-only plain
    ``Serializer``, and that is documented as intentional rather than accidental.
  * A ``ModelSerializer`` would attach DRF's ``UniqueValidator`` to ``username``,
    which spends a SELECT on every request and still cannot prevent a duplicate:
    two concurrent callers both pass the check, then one INSERT wins. Uniqueness is
    the database's job here (see ``services.UserService``), so paying for a query
    that provides no guarantee is the wrong trade.

Field rules are taken FROM the model rather than restated, so a future migration
that widens ``username`` cannot leave a stale limit behind here.
"""
from __future__ import annotations

from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.contrib.auth.validators import UnicodeUsernameValidator
from django.core.exceptions import ValidationError as DjangoValidationError
from rest_framework import serializers

from .base import PaginatedListSerializer

_USER = get_user_model()
_USERNAME = _USER._meta.get_field("username")
_EMAIL = _USER._meta.get_field("email")
_FIRST_NAME = _USER._meta.get_field("first_name")
_LAST_NAME = _USER._meta.get_field("last_name")

# Fields a caller must never be able to set through ANY endpoint in this app (create
# and update both enforce it). Presence of any of them is rejected outright rather
# than ignored: a client that believes it just created or promoted a superuser must be
# told it did not, instead of silently trusting a downgraded account. This is the
# mass-assignment and privilege-escalation guard, and it is a denylist ON TOP OF each
# serializer's allowlist — the allowlist alone already makes these unreachable, so
# this exists to turn a silent no-op into a loud error.
PRIVILEGED_FIELDS = frozenset({
    "is_staff", "is_superuser", "is_active", "groups", "user_permissions",
    "last_login", "date_joined", "password_hash", "id", "pk",
})

MSG_PRIVILEGED_FIELD = "This field cannot be set through this endpoint."


class UserCreateSerializer(serializers.Serializer):
    """Body of ``POST /api/v1/users/create``.

    Only these five fields are accepted. ``is_staff``/``is_superuser`` and friends
    are not merely absent from the allowlist — submitting them is an error (see
    ``PRIVILEGED_FIELDS``). New users are always created active and unprivileged;
    granting anything beyond that is role assignment, which is a later phase.
    """

    username = serializers.CharField(
        max_length=_USERNAME.max_length,
        # The same validator the model field uses — reused, not re-specified, so the
        # API and the database agree on what a username may contain.
        validators=[UnicodeUsernameValidator()],
    )
    email = serializers.EmailField(max_length=_EMAIL.max_length)
    # write_only is belt-and-braces (this serializer never renders output) and
    # trim_whitespace=False because leading/trailing spaces are legitimate password
    # characters — silently stripping them would lock the user out of the password
    # they think they set.
    password = serializers.CharField(write_only=True, trim_whitespace=False)
    first_name = serializers.CharField(
        max_length=_FIRST_NAME.max_length, required=False, allow_blank=True, default="")
    last_name = serializers.CharField(
        max_length=_LAST_NAME.max_length, required=False, allow_blank=True, default="")

    def validate(self, attrs):
        self._reject_non_string_fields()
        self._reject_privileged_fields()
        self._validate_password_policy(attrs)
        return attrs

    def _reject_non_string_fields(self) -> None:
        """Reject numbers/booleans where a string is expected.

        DRF's ``CharField`` accepts an int and quietly stringifies it, so a client
        sending ``{"username": 12345}`` would create a user literally named "12345"
        and a payload bug would ship as data. Every field here is textual, so a
        non-string is a malformed request, not something to coerce. (``dict``/
        ``list`` are already rejected by ``CharField`` itself.)
        """
        payload = self.initial_data
        if not hasattr(payload, "keys"):
            return
        offending = [
            name for name in ("username", "email", "password", "first_name", "last_name")
            if name in payload and not isinstance(payload[name], str)
        ]
        if offending:
            raise serializers.ValidationError(
                {name: ["This field must be a string."] for name in offending})

    def _reject_privileged_fields(self) -> None:
        """Fail loudly if the payload tried to set anything privileged."""
        payload = self.initial_data
        if not hasattr(payload, "keys"):  # a list/string body — DRF reports it already
            return
        offending = PRIVILEGED_FIELDS.intersection(payload.keys())
        if offending:
            raise serializers.ValidationError(
                {field: [MSG_PRIVILEGED_FIELD] for field in sorted(offending)})

    @staticmethod
    def _validate_password_policy(attrs) -> None:
        """Run the project's configured ``AUTH_PASSWORD_VALIDATORS``.

        Those four validators have been configured in ``config/settings/base.py``
        since the project began but were never invoked — until now nothing set a
        password from user input. Reused as-is; no policy is re-implemented here.

        An *unsaved* ``User`` carrying the submitted fields is passed in so
        ``UserAttributeSimilarityValidator`` can actually do its job (reject a
        password that echoes the username or email). Without it that validator
        silently passes everything.
        """
        candidate = _USER(
            username=attrs.get("username", ""),
            email=attrs.get("email", ""),
            first_name=attrs.get("first_name", ""),
            last_name=attrs.get("last_name", ""),
        )
        try:
            validate_password(attrs["password"], user=candidate)
        except DjangoValidationError as exc:
            # Django speaks in its own ValidationError; DRF needs its own type to
            # render field errors. The messages are the validators' own copy — they
            # describe the submitted password, never any stored secret.
            raise serializers.ValidationError({"password": list(exc.messages)}) from exc


class UserDetailSerializer(serializers.Serializer):
    """Body of ``POST /api/v1/users/detail`` — which user to fetch.

    ``min_value=1`` so a nonsensical id is a 400 (a client bug) rather than a 404 (a
    truthful-looking "no such user"), which keeps the two failures diagnosable apart.
    """

    user_id = serializers.IntegerField(min_value=1)


class UserUpdateSerializer(serializers.Serializer):
    """Body of ``POST /api/v1/users/update`` — profile fields only.

    ``user_id`` identifies the target (in the body, not a path segment — the
    platform's endpoints are all ``POST <resource>/<action>``).

    Partial by definition: every other field is optional and only what is sent is
    changed. A body carrying nothing but ``user_id`` is rejected rather than treated
    as a successful no-op, because a client that sent no changes almost certainly
    meant to send some.

    Three deliberate exclusions, each belonging to a different concern:

      * ``username`` — the login identifier. Renaming an identity is a distinct
        operation with its own audit and cache-invalidation questions, not a profile
        edit. Excluded until it is asked for explicitly.
      * ``password`` — password lifecycle lives in ``apps.authentication`` (and a
        change there revokes tokens; see AUTH_API_CONTRACT.md §3.1). Duplicating it
        here would create a second way to set a credential.
      * ``is_active`` / ``is_staff`` / ``is_superuser`` — deactivation is its own
        endpoint and privilege granting is role assignment. Submitting any of them
        is an error, exactly as on create.
    """

    user_id = serializers.IntegerField(min_value=1)
    email = serializers.EmailField(max_length=_EMAIL.max_length, required=False)
    first_name = serializers.CharField(
        max_length=_FIRST_NAME.max_length, required=False, allow_blank=True)
    last_name = serializers.CharField(
        max_length=_LAST_NAME.max_length, required=False, allow_blank=True)

    #: Fields this endpoint may actually write — ``user_id`` selects the row, it is
    #: not a change. Used by the view to split the target from the changes.
    UPDATABLE_FIELDS = ("email", "first_name", "last_name")

    def validate(self, attrs):
        self._reject_unknown_and_privileged()
        if not any(field in attrs for field in self.UPDATABLE_FIELDS):
            raise serializers.ValidationError(
                {"detail": ["Provide at least one field to update."]})
        return attrs

    def _reject_unknown_and_privileged(self) -> None:
        """Reject privileged fields, and anything not updatable through here.

        Stricter than create: an unknown key on a PATCH is very likely a client
        trying to change something it cannot (``username``, ``password``,
        ``is_staff``), and silently ignoring it would let the caller believe the
        change took effect.
        """
        payload = self.initial_data
        if not hasattr(payload, "keys"):
            return
        allowed = set(self.fields)
        unknown = [key for key in payload if key not in allowed]
        if unknown:
            raise serializers.ValidationError({
                key: [MSG_PRIVILEGED_FIELD if key in PRIVILEGED_FIELDS
                      else "This field cannot be updated through this endpoint."]
                for key in sorted(unknown)
            })


class UserListSerializer(PaginatedListSerializer):
    """Body of ``POST /api/v1/users/list``.

    Paging, ``search`` and ``is_active`` come from ``PaginatedListSerializer``; this
    subclass supplies only what is user-specific — which columns may be sorted on,
    and the default sort. ``search`` is matched against username and email by
    ``UserService``, since what "search" means is a service decision, not a
    validation one.
    """

    # Not the full field list: only columns that are indexed or cheap to sort.
    ORDERING_FIELDS = ("id", "username", "email", "date_joined", "last_login")

    ordering = serializers.CharField(required=False, default="username")
