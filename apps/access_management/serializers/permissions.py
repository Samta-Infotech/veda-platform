"""Permission-catalogue request validation.

Only two bodies exist, because the catalogue is read-only (see
``services/permissions.py``). There is no create or update serializer, and adding one
would be the first step toward a permission nothing enforces.
"""
from __future__ import annotations

from rest_framework import serializers

from .base import PaginatedListSerializer


class PermissionDetailSerializer(serializers.Serializer):
    """Body of ``POST /api/v1/permissions/detail`` — which permission to fetch.

    ``min_value=1`` so a nonsensical id is a 400 (a client bug) rather than a 404,
    keeping the two failures diagnosable apart.
    """

    permission_id = serializers.IntegerField(min_value=1)


class PermissionListSerializer(PaginatedListSerializer):
    """Body of ``POST /api/v1/permissions/list``.

    Paging, ``search`` and ``is_active`` come from ``PaginatedListSerializer``; this
    subclass supplies only what is permission-specific.
    """

    ORDERING_FIELDS = ("id", "code", "name")

    ordering = serializers.CharField(required=False, default="code")
