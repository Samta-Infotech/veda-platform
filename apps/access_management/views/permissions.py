"""Thin DRF views for the permission catalogue — read-only.

Two endpoints, no writes. The catalogue is seeded by migration because only code can
enforce a permission; see ``models/permissions.py`` for the full reasoning.
"""
from __future__ import annotations

from rest_framework.permissions import IsAdminUser

from apps.core import api
from apps.core.messages import MESSAGES

from ..serializers import (
    PermissionDetailSerializer,
    PermissionDropdownSerializer,
    PermissionListSerializer,
)
from ..services import AccessManagementError, PermissionService
from .base import AdminView, pagination_payload


def public_fields(permission) -> dict:

    """What a caller may see of a permission — the ONE projection for both endpoints."""
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
    """GET /api/v1/permissions/list?page=&page_size=&search=&is_active=&ordering="""

    serializer_class = PermissionListSerializer
    action = "permission list"
    # No RBAC permission of its own — staff-only via IsAdminUser. The
    # `permission.read` permission was removed (2026-09-03). RequiresPermission is
    # dropped from permission_classes rather than left with a blank
    # required_permission, because gate.py:90 fails closed on a blank one and would
    # 403 every caller under VEDA_RBAC_MODE=enforce.
    permission_classes = [IsAdminUser]

    def get(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        permissions, total = PermissionService(request).list_permissions(**data)
        page, page_size = data["page"], data["page_size"]

        return api.success(MESSAGES["permission"]["list"], {
            "permissions": [public_fields(p) for p in permissions],
            "pagination": pagination_payload(page, page_size, total),
        })


class PermissionDetailView(AdminView):
    """GET /api/v1/permissions/detail?permission_id= -> one permission."""

    serializer_class = PermissionDetailSerializer
    action = "permission detail"
    # No RBAC permission of its own — staff-only via IsAdminUser. The
    # `permission.read` permission was removed (2026-09-03). RequiresPermission is
    # dropped from permission_classes rather than left with a blank
    # required_permission, because gate.py:90 fails closed on a blank one and would
    # 403 every caller under VEDA_RBAC_MODE=enforce.
    permission_classes = [IsAdminUser]

    def get(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        try:
            permission = PermissionService(request).get_permission(data["permission_id"])
        except AccessManagementError as exc:
            return self.failure(request, exc)

        return api.success(MESSAGES["permission"]["retrieved"], public_fields(permission))


class PermissionDropdownView(AdminView):
    """GET /api/v1/permissions/dropdown -> unpaginated list of active permissions."""

    serializer_class = PermissionDropdownSerializer
    action = "permission dropdown"
    # No RBAC permission of its own — staff-only via IsAdminUser. The
    # `permission.read` permission was removed (2026-09-03). RequiresPermission is
    # dropped from permission_classes rather than left with a blank
    # required_permission, because gate.py:90 fails closed on a blank one and would
    # 403 every caller under VEDA_RBAC_MODE=enforce.
    permission_classes = [IsAdminUser]

    def get(self, request):
        """Return all active permissions, unpaginated."""
        permissions = PermissionService(request).list_active_permissions()

        data = [{"label": p.name, "value": p.pk} for p in permissions]
        return api.success(MESSAGES["permission"]["list"], data)





