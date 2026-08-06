"""Role-administration request validation.

Same contract as every serializer module in this project: INPUT-only, plain
``serializers.Serializer``, read via ``.validated_data``, never used to render a
response. Field rules are taken FROM the model so a future migration that widens
``name`` cannot leave a stale limit behind here.

Uniqueness is deliberately NOT validated here. A ``UniqueValidator`` would spend a
SELECT on every request and still not prevent a duplicate — two concurrent callers
both pass the check, then one INSERT wins. The database constraint is the only thing
that can decide, so the service lets the INSERT fail and translates the error
(``services/roles.py``).
"""
from __future__ import annotations

from rest_framework import serializers

from ..models import Role
from .base import PaginatedListSerializer

_NAME = Role._meta.get_field("name")

#: Server-owned columns. Submitting one means the client misunderstood the contract,
#: and silently ignoring it would let the client believe the value took effect.
#:
#: ``role_id`` is deliberately NOT here: it is server-owned as a *value*, but it is
#: also the required target of detail/update. Rejecting a key is the allowlist's job
#: (every serializer passes its own ``self.fields``); this set only exists to give
#: these particular keys a clearer message than "cannot be set".
READ_ONLY_FIELDS = frozenset({"id", "created_at", "updated_at"})

MSG_READ_ONLY_FIELD = "This field is read-only."


def _validate_name(value: str) -> str:
    """A role name must carry an actual label.

    ``allow_blank=False`` already rejects ``""``, but not ``"   "`` — and DRF trims
    whitespace after validation, which would store a blank name that no constraint
    forbids. Rejecting it here keeps the table free of unnameable roles.
    """
    if not value.strip():
        raise serializers.ValidationError("This field may not be blank.")
    return value.strip()


class RoleCreateSerializer(serializers.Serializer):
    """Body of ``POST /api/v1/roles/create``.

    A role is created active; there is no ``is_active`` here because "create a role
    that is already retired" is not a thing an administrator means to do. Retiring is
    an update.
    """

    name = serializers.CharField(max_length=_NAME.max_length, validators=[_validate_name])
    description = serializers.CharField(required=False, allow_blank=True, default="")

    def validate(self, attrs):
        _reject_unknown_fields(self.initial_data, allowed=set(self.fields))
        attrs["name"] = attrs["name"].strip()
        return attrs


class RoleDetailSerializer(serializers.Serializer):
    """Body of ``POST /api/v1/roles/detail`` — which role to fetch.

    ``min_value=1`` so a nonsensical id is a 400 (a client bug) rather than a 404 (a
    truthful-looking "no such role"), which keeps the two failures diagnosable apart.
    """

    role_id = serializers.IntegerField(min_value=1)


class RoleUpdateSerializer(serializers.Serializer):
    """Body of ``POST /api/v1/roles/update`` — partial.

    ``is_active`` is updatable here, and that is how a role is retired: there is no
    delete endpoint. Hard deletion's semantics depend entirely on role *assignment*
    (what happens to the users holding it?), which does not exist yet, and an audit
    trail that says "granted role #7" must still be able to resolve #7.
    """

    role_id = serializers.IntegerField(min_value=1)
    name = serializers.CharField(
        max_length=_NAME.max_length, required=False, validators=[_validate_name])
    description = serializers.CharField(required=False, allow_blank=True)
    is_active = serializers.BooleanField(required=False)

    #: Fields this endpoint may actually write — ``role_id`` selects the row, it is
    #: not a change. Used by the view to split the target from the changes.
    UPDATABLE_FIELDS = ("name", "description", "is_active")

    def validate(self, attrs):
        _reject_unknown_fields(self.initial_data, allowed=set(self.fields))
        if not any(field in attrs for field in self.UPDATABLE_FIELDS):
            raise serializers.ValidationError(
                {"detail": ["Provide at least one field to update."]})
        if "name" in attrs:
            attrs["name"] = attrs["name"].strip()
        return attrs


class RoleListSerializer(PaginatedListSerializer):
    """Body of ``POST /api/v1/roles/list``.

    Paging, ``search`` and ``is_active`` come from ``PaginatedListSerializer``; this
    subclass supplies only what is role-specific. ``search`` is matched against name
    and description by ``RoleService`` — what "search" means is a service decision.
    """

    ORDERING_FIELDS = ("id", "name", "created_at", "updated_at")

    ordering = serializers.CharField(required=False, default="name")


def _reject_unknown_fields(payload, *, allowed: set[str]) -> None:
    """Fail loudly on any key this endpoint does not accept.

    Rejecting rather than ignoring: a client that thinks it set ``id``, or retired a
    role via ``create``, or sent a field that does not exist, must be told — not left
    believing the write happened. The allowlist is the rule; ``READ_ONLY_FIELDS``
    only refines the message for server-owned columns.
    """
    if not hasattr(payload, "keys"):  # a list/string body — DRF reports it already
        return
    offending = {
        key: [MSG_READ_ONLY_FIELD if key in READ_ONLY_FIELDS
              else "This field cannot be set through this endpoint."]
        for key in payload
        if key not in allowed
    }
    if offending:
        raise serializers.ValidationError(dict(sorted(offending.items())))
