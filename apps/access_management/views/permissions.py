"""Thin DRF views for the permission catalogue — read-only.

Two endpoints, no writes. The catalogue is seeded by migration because only code can
enforce a permission; see ``models/permissions.py`` for the full reasoning.
"""
from __future__ import annotations

from apps.core import api

from ..serializers import PermissionDetailSerializer, PermissionListSerializer
from ..services import AccessManagementError, PermissionService
from .base import AdminView, pagination_payload


def public_fields(permission) -> dict:
    """What a caller may see of a permission — the ONE projection for both endpoints.

    An explicit projection, not a model dump, so a column added by a future migration
    cannot leak into responses by accident.
    """
    return {
        "permission_id": permission.pk,
        "code": permission.code,
        "name": permission.name,
        "description": permission.description,
        "is_active": permission.is_active,
        "created_at": api.iso_z(permission.created_at),
        "updated_at": api.iso_z(permission.updated_at),
    }


class PermissionListView(AdminView):
    """POST /api/v1/permissions/list {page?, page_size?, search?, is_active?, ordering?}.

    The catalogue a role screen renders to ask "what can I grant?".
    """

    serializer_class = PermissionListSerializer
    action = "permission list"
    required_permission = "permission.read"

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        permissions, total = PermissionService(request).list_permissions(**data)
        page, page_size = data["page"], data["page_size"]

        return api.success("Permissions retrieved successfully.", {
            "permissions": [public_fields(p) for p in permissions],
            "pagination": pagination_payload(page, page_size, total),
        })


class PermissionDetailView(AdminView):
    """POST /api/v1/permissions/detail {permission_id} -> one permission."""

    serializer_class = PermissionDetailSerializer
    action = "permission detail"
    required_permission = "permission.read"

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        try:
            permission = PermissionService(request).get_permission(data["permission_id"])
        except AccessManagementError as exc:
            return self.failure(request, exc)

        return api.success("Permission retrieved successfully.", public_fields(permission))
