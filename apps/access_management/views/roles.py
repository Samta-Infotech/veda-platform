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
    RoleDropdownSerializer,
    RoleListSerializer,
    RoleUpdateSerializer,
)
from ..services import AccessManagementError, RoleService, role_stats
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
        "deleted_at": api.iso_z(role.deleted_at),
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

        # Two queries for the whole page — see grants.role_stats.
        stats = role_stats([role.pk for role in roles])

        return api.success(MESSAGES["role"]["list"], {
            "roles": [
                {
                    **public_fields(role),
                    "role_name": role.name,
                    "users_count": stats[role.pk]["users_count"],
                    "connected_sources": stats[role.pk]["connected_sources"],
                    "last_updated": api.human_date(role.updated_at),
                }
                for role in roles
            ],
            "pagination": pagination_payload(page, page_size, total),
        })


class RoleDropdownView(AdminView):
    """POST /api/v1/roles/dropdown {} -> every active role, unpaginated.

    For a picker/select control, not the admin table — ``roles/list`` stays the
    paginated, searchable, sortable view of the same data. Two endpoints because
    they answer two different questions ("show me a page of roles to manage" vs
    "give me every option a dropdown can render"), not two strengths of one.
    """

    serializer_class = RoleDropdownSerializer
    action = "role dropdown"
    required_permission = PermissionCode.ROLE_MANAGE

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        roles = RoleService(request).list_active_roles()

        return api.success(MESSAGES["role"]["dropdown"], {
            "roles": [{"role_id": role.pk, "name": role.name} for role in roles],
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


class RoleDeleteView(AdminView):
    """POST /api/v1/roles/delete {role_id} -> 200.

    A SOFT delete — a named convenience over ``RoleService.update_role(role_id,
    is_active=False)``, not a second code path. No row is removed: hard-deletion's
    semantics depend on role assignment (what happens to the users holding it?),
    and an audit trail that says "granted role #7" must still be able to resolve
    #7. Retiring leaves the row queryable, just no longer grantable.
    """

    serializer_class = RoleDetailSerializer
    action = "role deletion"
    required_permission = PermissionCode.ROLE_MANAGE

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        try:
            role = RoleService(request).update_role(data["role_id"], is_active=False)
        except AccessManagementError as exc:
            return self.failure(request, exc)

        return api.success(MESSAGES["role"]["updated"], public_fields(role))
