"""Request-validation layer for access management, one module per domain.

Same rationale and same re-export rule as ``services/``: callers import from the
package, so a name can move between modules without breaking them.
"""
from .base import PaginatedListSerializer
from .catalog import (
    CatalogResourceDetailSerializer,
    CatalogResourceListSerializer,
    CatalogTreeSerializer,
)
from .resolver import EffectivePermissionsSerializer
from .grants import (
    RolePermissionGrantSerializer,
    RolePermissionListSerializer,
    RolePermissionRevokeSerializer,
    UserRoleAssignSerializer,
    UserRoleListSerializer,
)
from .permissions import (
    PermissionDetailSerializer,
    PermissionDropdownSerializer,
    PermissionListSerializer,
)
from .roles import (
    MSG_READ_ONLY_FIELD,
    READ_ONLY_FIELDS,
    RoleCreateSerializer,
    RoleDetailSerializer,
    RoleDropdownSerializer,
    RoleListSerializer,
    RoleUpdateSerializer,
)
from .users import (
    MSG_PRIVILEGED_FIELD,
    PRIVILEGED_FIELDS,
    UserCreateSerializer,
    UserDetailSerializer,
    UserListSerializer,
    UserUpdateSerializer,
)

__all__ = [
    "MSG_PRIVILEGED_FIELD",
    "MSG_READ_ONLY_FIELD",
    "PRIVILEGED_FIELDS",
    "CatalogResourceDetailSerializer",
    "CatalogResourceListSerializer",
    "CatalogTreeSerializer",
    "EffectivePermissionsSerializer",
    "PaginatedListSerializer",
    "RolePermissionGrantSerializer",
    "RolePermissionListSerializer",
    "RolePermissionRevokeSerializer",
    "UserRoleAssignSerializer",
    "UserRoleListSerializer",
    "PermissionDetailSerializer",
    "PermissionDropdownSerializer",
    "PermissionListSerializer",
    "READ_ONLY_FIELDS",
    "RoleCreateSerializer",
    "RoleDetailSerializer",
    "RoleDropdownSerializer",
    "RoleListSerializer",
    "RoleUpdateSerializer",
    "UserCreateSerializer",
    "UserDetailSerializer",
    "UserListSerializer",
    "UserUpdateSerializer",
]
