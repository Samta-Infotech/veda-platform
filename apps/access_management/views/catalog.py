"""Thin DRF views for the catalog projection — read-only.

Rows come from discovery (``services/catalog.py``), never from the API. See the
serializer module for why there is no write path.
"""
from __future__ import annotations

from rest_framework.permissions import IsAdminUser

from apps.core import api
from apps.core.messages import MESSAGES

from ..serializers import (
    CatalogResourceDetailSerializer,
    CatalogResourceListSerializer,
    CatalogTreeSerializer,
)
from ..services import AccessManagementError, CatalogService
from .base import AdminView, pagination_payload


def public_fields(resource) -> dict:
    """What a caller may see of a catalog resource — the ONE projection for both
    endpoints.

    ``substrate_id`` is exposed because retrieval and the admin UI need to join back
    to the substrate row. It is an internal identifier, but not a secret, and omitting
    it would force clients to re-derive it — which is how two derivations drift apart.
    """
    return {
        "path": resource.path,
        "kind": resource.kind,
        "parent_path": resource.parent_path,
        "source_id": resource.source_id,
        "substrate_id": str(resource.substrate_id) if resource.substrate_id else None,
        "is_active": resource.is_active,
        "created_at": api.iso_z(resource.created_at),
        "updated_at": api.iso_z(resource.updated_at),
    }


class CatalogListView(AdminView):
    """GET /api/v1/catalog/list?page=&page_size=&search=&is_active=&ordering=
    &parent_path=&kind=&source_id=

    The admin tree's data source: pass a node's ``path`` as ``parent_path`` to get
    exactly its children. Read-only, so GET only.
    """

    serializer_class = CatalogResourceListSerializer
    action = "catalog list"
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

        resources, total = CatalogService(request).list_resources(**data)

        return api.success(MESSAGES["catalog"]["list"], {
            "resources": [public_fields(r) for r in resources],
            "pagination": pagination_payload(data["page"], data["page_size"], total),
        })


class CatalogDetailView(AdminView):
    """GET /api/v1/catalog/detail?path= -> one resource."""

    serializer_class = CatalogResourceDetailSerializer
    action = "catalog detail"
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
            resource = CatalogService(request).get_resource(data["path"])
        except AccessManagementError as exc:
            return self.failure(request, exc)

        return api.success(MESSAGES["catalog"]["retrieved"],
                           public_fields(resource))


class CatalogTreeView(AdminView):
    """GET /api/v1/catalog/tree?role_id=&category=&parent_path=&search=

    Hierarchical catalog tree projection with optional role permissions resolution.
    """

    serializer_class = CatalogTreeSerializer
    action = "catalog tree"
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

        tree_data = CatalogService(request).get_tree(**data)
        return api.success(MESSAGES["catalog"]["list"], tree_data)

