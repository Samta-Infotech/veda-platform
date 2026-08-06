"""Thin DRF views for role administration.

Each view validates a body with a serializer, hands the validated values to
``RoleService``, and renders the outcome via ``apps.core.api``. No uniqueness
handling and no lifecycle rules here.

Access control, the service-error-to-status mapping and the validate-or-400 branch
all come from ``views/base.py`` — the same plumbing the user endpoints use, so a
future RBAC permission check replaces ``IsAdminUser`` in one place for both.
"""
from __future__ import annotations

from rest_framework import status

from apps.core import api
from apps.core.messages import MESSAGES

from ..codes import PermissionCode
from ..serializers import (
    RoleCreateSerializer,
    RoleDetailSerializer,
    RoleListSerializer,
    RoleUpdateSerializer,
)
from ..services import AccessManagementError, RoleService
from .base import AdminView, pagination_payload


def public_fields(role) -> dict:
    """What a caller may see of a role — the ONE projection for every role endpoint.

    An explicit projection, not a model dump: a column added by a future migration
    cannot leak into responses by accident. Shared by create/detail/list/update, so
    opening a row returns exactly the shape that was listed.
    """
    return {
        "role_id": role.pk,
        "name": role.name,
        "description": role.description,
        "is_active": role.is_active,
        "created_at": api.iso_z(role.created_at),
        "updated_at": api.iso_z(role.updated_at),
    }


class RoleCreateView(AdminView):
    """POST /api/v1/roles/create {name, description?}.

    Returns 201. A duplicate name is 409 rather than 400, so a client can tell "your
    request was malformed" from "your request was fine but the name is taken" — the
    second is worth retrying with a different value, the first is not.
    """

    serializer_class = RoleCreateSerializer
    action = "role creation"
    required_permission = PermissionCode.ROLE_MANAGE

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        try:
            role = RoleService(request).create_role(**data)
        except AccessManagementError as exc:
            return self.failure(request, exc)

        return api.success(MESSAGES["role"]["created"], public_fields(role),
                           status_code=status.HTTP_201_CREATED)


class RoleDetailView(AdminView):
    """POST /api/v1/roles/detail {role_id} -> one role.

    Same projection as list. No permission or member counts: neither exists yet, and
    a placeholder would be a contract we would have to break.
    """

    serializer_class = RoleDetailSerializer
    action = "role detail"
    required_permission = PermissionCode.ROLE_MANAGE

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        try:
            role = RoleService(request).get_role(data["role_id"])
        except AccessManagementError as exc:
            return self.failure(request, exc)

        return api.success(MESSAGES["role"]["retrieved"], public_fields(role))


class RoleListView(AdminView):
    """POST /api/v1/roles/list {page?, page_size?, search?, is_active?, ordering?}.

    Paginated like every list endpoint here. Roles are few, but a list endpoint whose
    response size depends on the table is a habit worth not forming.
    """

    serializer_class = RoleListSerializer
    action = "role list"
    required_permission = PermissionCode.ROLE_MANAGE

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        roles, total = RoleService(request).list_roles(**data)
        page, page_size = data["page"], data["page_size"]

        return api.success(MESSAGES["role"]["list"], {
            "roles": [public_fields(role) for role in roles],
            "pagination": pagination_payload(page, page_size, total),
        })


class RoleUpdateView(AdminView):
    """POST /api/v1/roles/update {role_id, name?, description?, is_active?}.

    Partial: only the fields present are written. ``is_active: false`` is how a role
    is retired — there is no delete endpoint, because hard-deletion's semantics depend
    on role assignment, which does not exist yet.

    404 when the id does not exist, 409 when the new name belongs to another role.
    """

    serializer_class = RoleUpdateSerializer
    action = "role update"
    required_permission = PermissionCode.ROLE_MANAGE

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        role_id = data.pop("role_id")
        try:
            role = RoleService(request).update_role(role_id, **data)
        except AccessManagementError as exc:
            return self.failure(request, exc)

        return api.success(MESSAGES["role"]["updated"], public_fields(role))
