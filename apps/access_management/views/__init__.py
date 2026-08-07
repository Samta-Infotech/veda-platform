"""View layer for access management, one module per domain.

``base.py`` holds what every domain shares (the staff-only rule, the typed-error to
HTTP-status mapping, the validate-or-400 branch); ``users.py`` holds the user
endpoints. Re-exported here so callers — chiefly ``urls/`` — import from the package.
"""
from .base import AdminView, error_status, log_context, pagination_payload
from .catalog import CatalogDetailView, CatalogListView
from .catalog import public_fields as catalog_public_fields
from .grants import (
    RolePermissionGrantView,
    RolePermissionListView,
    RolePermissionRevokeView,
    UserRoleAssignView,
    UserRoleListView,
    UserRoleRevokeView,
    assignment_fields,
    grant_fields,
)
from .permissions import PermissionDetailView, PermissionListView
from .resolver import EffectivePermissionsView
from .permissions import public_fields as permission_public_fields
from .roles import (
    RoleCreateView,
    RoleDeleteView,
    RoleDetailView,
    RoleDropdownView,
    RoleListView,
    RoleUpdateView,
)
from .roles import public_fields as role_public_fields
from .users import (
    UserCreateView,
    UserDeleteView,
    UserDetailView,
    UserListView,
    UserUpdateView,
    public_fields,
)

__all__ = [
    "AdminView",
    "CatalogDetailView",
    "CatalogListView",
    "EffectivePermissionsView",
    "PermissionDetailView",
    "RolePermissionGrantView",
    "RolePermissionListView",
    "RolePermissionRevokeView",
    "UserRoleAssignView",
    "UserRoleListView",
    "UserRoleRevokeView",
    "assignment_fields",
    "grant_fields",
    "PermissionListView",
    "RoleCreateView",
    "RoleDeleteView",
    "RoleDetailView",
    "RoleDropdownView",
    "RoleListView",
    "RoleUpdateView",
    "UserCreateView",
    "UserDeleteView",
    "UserDetailView",
    "UserListView",
    "UserUpdateView",
    "error_status",
    "catalog_public_fields",
    "log_context",
    "pagination_payload",
    "permission_public_fields",
    "public_fields",
    "role_public_fields",
]
