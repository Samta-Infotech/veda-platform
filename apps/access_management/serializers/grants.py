"""Grant request validation — role assignment and permission grants.

Bodies name their targets by id, and the resource by path (ADR-0001) — the same
addressing every other part of the RBAC graph uses.
"""
from __future__ import annotations

from rest_framework import serializers

from .. import resource_path as rp
from ..codes import PermissionCode
from ..models import Effect, Permission
from .base import PaginatedListSerializer


def _imply_data_read(attrs):
    """Fill in ``permission_id`` for a resource-scoped grant that omitted it.

    Granting access to a resource is always ``data.read`` ON that path — the caller
    picks a resource in the catalog tree, never a permission. ``data.read`` is
    deliberately hidden from ``permissions/dropdown`` for that reason (see
    ``services/permissions.py::_HIDDEN_FROM_DROPDOWN``), which left a resource screen
    with no way to obtain the id it was nonetheless required to send. Implied here so
    it does not have to be hardcoded in every client.

    Only implied when a ``resource_path`` is present. A blank-path ``data.read`` grant
    covers no resource at all (see ``services/resolver.py``), so defaulting to it for a
    pathless body would mint a grant that looks real and does nothing — that case still
    fails, with the same field error it always had.
    """
    if attrs.get("permission_id") is not None:
        return attrs
    if not attrs.get("resource_path"):
        raise serializers.ValidationError(
            {"permission_id": ["This field is required when no resource_path is given."]})
    permission = (Permission.objects
                  .filter(code=PermissionCode.DATA_READ, is_active=True)
                  .only("id").first())
    if permission is None:
        # Deactivated or unseeded — say which permission is missing rather than
        # failing later on a None pk.
        raise serializers.ValidationError(
            {"permission_id": [f"No active '{PermissionCode.DATA_READ}' permission to imply."]})
    attrs["permission_id"] = permission.pk
    return attrs


class _ResourcePathField(serializers.CharField):
    """A canonical resource path, or "" for a grant that is not resource-scoped.

    Canonicalises on input so a caller sending ``DB:CRM`` grants the same row a caller
    sending ``db:crm`` would — otherwise one resource would be grantable under two
    strings and only one of them would ever match at resolution time.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("required", False)
        kwargs.setdefault("allow_blank", True)
        kwargs.setdefault("default", "")
        kwargs.setdefault("max_length", rp.MAX_LENGTH)
        super().__init__(**kwargs)

    def to_internal_value(self, data):
        value = super().to_internal_value(data)
        if not value:
            return ""
        try:
            return rp.validate(value)
        except rp.InvalidResourcePath as exc:
            raise serializers.ValidationError(str(exc)) from exc


class UserRoleAssignSerializer(serializers.Serializer):
    """Body of ``POST /api/v1/users/roles/assign`` and ``.../revoke``."""

    user_id = serializers.IntegerField(min_value=1)
    role_id = serializers.IntegerField(min_value=1)


class UserRoleListSerializer(PaginatedListSerializer):
    """Body of ``POST /api/v1/users/roles/list``.

    Both filters are optional: omit them for every assignment, pass ``user_id`` for
    "what does this user hold", pass ``role_id`` for "who holds this role".
    """

    ORDERING_FIELDS = ("id", "user", "role", "created_at")

    ordering = serializers.CharField(required=False, default="id")
    user_id = serializers.IntegerField(required=False, min_value=1,
                                       allow_null=True, default=None)
    role_id = serializers.IntegerField(required=False, min_value=1,
                                       allow_null=True, default=None)


class RolePermissionGrantSerializer(serializers.Serializer):
    """Body of ``POST /api/v1/roles/permissions/grant``.

    ``effect`` defaults to ``allow``: the overwhelmingly common case, and an explicit
    DENY should be a deliberate act rather than something a caller can produce by
    typo'ing the field name.
    """

    role_id = serializers.IntegerField(min_value=1)
    #: Optional for a resource-scoped grant — see ``_imply_data_read``.
    permission_id = serializers.IntegerField(min_value=1, required=False)
    resource_path = _ResourcePathField()
    effect = serializers.ChoiceField(choices=Effect.choices, required=False,
                                     default=Effect.ALLOW)

    def validate(self, attrs):
        return _imply_data_read(attrs)


class RolePermissionRevokeSerializer(serializers.Serializer):
    """Body of ``POST /api/v1/roles/permissions/revoke``.

    No ``effect``: revoking removes whatever decision exists for the triple. Accepting
    one would invite "revoke the allow but leave the deny", which is not a thing —
    there is only ever one decision per triple.
    """

    role_id = serializers.IntegerField(min_value=1)
    #: Optional for a resource-scoped revoke — same implication as grant, so a client
    #: can revoke exactly what it granted without having to name data.read.
    permission_id = serializers.IntegerField(min_value=1, required=False)
    resource_path = _ResourcePathField()

    def validate(self, attrs):
        return _imply_data_read(attrs)


class RolePermissionListSerializer(PaginatedListSerializer):
    """Body of ``POST /api/v1/roles/permissions/list``."""

    ORDERING_FIELDS = ("id", "role", "permission", "resource_path", "created_at")

    ordering = serializers.CharField(required=False, default="id")
    role_id = serializers.IntegerField(required=False, min_value=1,
                                       allow_null=True, default=None)
    permission_id = serializers.IntegerField(required=False, min_value=1,
                                             allow_null=True, default=None)
    #: None = no filter. "" would mean "only global grants", which is a real question,
    #: so it must stay distinguishable from "not filtering".
    resource_path = _ResourcePathField(allow_null=True, default=None)
