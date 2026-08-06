"""Grant request validation — role assignment and permission grants.

Bodies name their targets by id, and the resource by path (ADR-0001) — the same
addressing every other part of the RBAC graph uses.
"""
from __future__ import annotations

from rest_framework import serializers

from .. import resource_path as rp
from ..models import Effect
from .base import PaginatedListSerializer


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
    permission_id = serializers.IntegerField(min_value=1)
    resource_path = _ResourcePathField()
    effect = serializers.ChoiceField(choices=Effect.choices, required=False,
                                     default=Effect.ALLOW)


class RolePermissionRevokeSerializer(serializers.Serializer):
    """Body of ``POST /api/v1/roles/permissions/revoke``.

    No ``effect``: revoking removes whatever decision exists for the triple. Accepting
    one would invite "revoke the allow but leave the deny", which is not a thing —
    there is only ever one decision per triple.
    """

    role_id = serializers.IntegerField(min_value=1)
    permission_id = serializers.IntegerField(min_value=1)
    resource_path = _ResourcePathField()


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
