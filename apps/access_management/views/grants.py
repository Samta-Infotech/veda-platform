"""Thin DRF views for role assignment and permission grants.

Assign/grant answer **201 when the edge is new and 200 when it already existed** —
both success, because these operations describe a desired state rather than an event
(see ``services/grants.py``). Revoke is always 200.
"""
from __future__ import annotations

from rest_framework import status

from apps.core import api
from apps.core.messages import MESSAGES

from ..codes import PermissionCode
from ..serializers import (
    RolePermissionGrantSerializer,
    RolePermissionListSerializer,
    RolePermissionRevokeSerializer,
    UserRoleAssignSerializer,
    UserRoleListSerializer,
)
from ..services import (
    AccessManagementError,
    RolePermissionService,
    UserRoleService,
)
from .base import AdminView, pagination_payload


def assignment_fields(assignment) -> dict:
    """The projection for a user-role edge."""
    return {
        "user_id": assignment.user_id,
        "role_id": assignment.role_id,
        "granted_by": assignment.granted_by_id,
        "created_at": api.iso_z(assignment.created_at),
    }


def grant_fields(grant, known_paths=frozenset()) -> dict:
    """The projection for a role-permission edge.

    ``resource_exists`` tells an admin UI whether the path still resolves to a live
    catalog resource. A grant on an unknown path is legal — pre-provisioning is
    deliberate — but showing it as ordinary would hide a typo until someone wondered
    why access never worked. Computed from a set the caller fetched in ONE query, so
    a page of grants costs no N+1.
    """
    return {
        "role_id": grant.role_id,
        "permission_id": grant.permission_id,
        "resource_path": grant.resource_path,
        "effect": grant.effect,
        "resource_exists": (True if not grant.resource_path
                            else grant.resource_path in known_paths),
        "granted_by": grant.granted_by_id,
        "created_at": api.iso_z(grant.created_at),
        "updated_at": api.iso_z(grant.updated_at),
    }


class UserRoleAssignView(AdminView):
    """POST /api/v1/users/roles/assign {user_id, role_id}.

    201 when newly assigned, 200 when the user already held the role.
    """

    serializer_class = UserRoleAssignSerializer
    action = "role assignment"
    required_permission = PermissionCode.ROLE_MANAGE

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        try:
            assignment, created = UserRoleService(request).assign(**data)
        except AccessManagementError as exc:
            return self.failure(request, exc)

        return api.success(
            MESSAGES["user_role"]["assigned"] if created else MESSAGES["user_role"]["already_assigned"],
            assignment_fields(assignment),
            status_code=status.HTTP_201_CREATED if created else status.HTTP_200_OK)


class UserRoleRevokeView(AdminView):
    """POST /api/v1/users/roles/revoke {user_id, role_id} -> 200, always.

    Idempotent: revoking a role the user does not hold is success, because the desired
    end state already holds. ``removed`` says which happened.
    """

    serializer_class = UserRoleAssignSerializer
    action = "role revocation"
    required_permission = PermissionCode.ROLE_MANAGE

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        removed = UserRoleService(request).revoke(**data)
        return api.success(MESSAGES["user_role"]["revoked"] if removed
                           else MESSAGES["user_role"]["not_assigned"], {"removed": removed})


class UserRoleListView(AdminView):
    """GET /api/v1/users/roles/list?user_id=&role_id=&page=&page_size="""

    serializer_class = UserRoleListSerializer
    action = "role assignment list"
    required_permission = PermissionCode.ROLE_MANAGE

    def get(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        assignments, total = UserRoleService(request).list_assignments(**data)
        return api.success(MESSAGES["user_role"]["list"], {
            "assignments": [assignment_fields(a) for a in assignments],
            "pagination": pagination_payload(data["page"], data["page_size"], total),
        })


class RolePermissionGrantView(AdminView):
    """POST /api/v1/roles/permissions/grant
    {role_id, permission_id, resource_path?, effect?}.

    201 when the decision is new, 200 when an existing decision was updated —
    re-granting with the opposite effect flips it rather than adding a contradiction.
    """

    serializer_class = RolePermissionGrantSerializer
    action = "permission grant"
    required_permission = PermissionCode.ROLE_MANAGE

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        try:
            grant, created = RolePermissionService(request).grant(**data)
        except AccessManagementError as exc:
            return self.failure(request, exc)

        known = RolePermissionService.known_resource_paths([grant.resource_path])
        return api.success(
            MESSAGES["role_permission"]["granted"] if created else MESSAGES["role_permission"]["updated"],
            grant_fields(grant, known),
            status_code=status.HTTP_201_CREATED if created else status.HTTP_200_OK)


class RolePermissionRevokeView(AdminView):
    """POST /api/v1/roles/permissions/revoke
    {role_id, permission_id, resource_path?} -> 200, always.

    Removing a DENY does not create an ALLOW: with nothing matching, default-deny
    applies (ADR §3.5).
    """

    serializer_class = RolePermissionRevokeSerializer
    action = "permission revocation"
    required_permission = PermissionCode.ROLE_MANAGE

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        try:
            removed = RolePermissionService(request).revoke(**data)
        except AccessManagementError as exc:
            return self.failure(request, exc)

        return api.success(MESSAGES["role_permission"]["revoked"] if removed
                           else MESSAGES["role_permission"]["not_granted"], {"removed": removed})


class RolePermissionListView(AdminView):
    """GET /api/v1/roles/permissions/list?role_id=&permission_id=&resource_path=
    &page=&page_size="""

    serializer_class = RolePermissionListSerializer
    action = "permission grant list"
    required_permission = PermissionCode.ROLE_MANAGE

    def get(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        service = RolePermissionService(request)
        try:
            grants, total = service.list_grants(**data)
        except AccessManagementError as exc:
            return self.failure(request, exc)

        # One query for the whole page — see grant_fields.
        known = service.known_resource_paths([g.resource_path for g in grants])
        return api.success(MESSAGES["role_permission"]["list"], {
            "grants": [grant_fields(g, known) for g in grants],
            "pagination": pagination_payload(data["page"], data["page_size"], total),
        })
