"""Service layer for access management, one module per domain.

Split into a package because this app owns the whole RBAC surface: user
administration today, roles/permissions/resolution next. ``users.py`` stays about
users, and a future ``roles.py`` cannot accidentally grow into it.

Everything public is re-exported here, so callers import from the package
(``from apps.access_management.services import UserService``) and never need to know
which module a name lives in. Moving a class between modules therefore does not
break its importers.
"""
from .admin_guard import (
    ADMIN_ROLE_NAME,
    CODE_LAST_ADMIN_PROTECTED,
    CODE_LAST_ADMIN_ROLE_PROTECTED,
    LastAdminProtected,
    LastAdminRoleProtected,
    active_admins,
    get_admin_role,
    is_last_active_admin,
)
from .base import AccessManagementError, ConflictError, NotFoundError, paginate
from .bootstrap import AdminBootstrapService, AlreadyBootstrapped
from .catalog import (
    CATALOG_LIST_FIELDS,
    CODE_RESOURCE_NOT_FOUND,
    CatalogDiscoveryService,
    CatalogService,
    DiscoveryReport,
    ResourceNotFound,
)
from .data_scope import (
    UNRESOLVED,
    SourceDataScope,
    TableScope,
    compute_data_scope,
    resolve_effective_permissions,
    serialize_data_scope,
)
from .grants import (
    CODE_INVALID_RESOURCE,
    CODE_PERMISSION_INACTIVE,
    CODE_ROLE_INACTIVE,
    InvalidResourcePath,
    PermissionInactive,
    RoleInactive,
    RolePermissionService,
    UserRoleService,
    role_stats,
)
from .permissions import (
    CODE_PERMISSION_NOT_FOUND,
    PERMISSION_LIST_FIELDS,
    PermissionNotFound,
    PermissionService,
)
from .resolver import (
    NO_PERMISSIONS,
    EffectivePermissions,
    Grant,
    PermissionResolver,
)
from .roles import (
    CODE_INVALID_GRANT,
    CODE_ROLE_NAME_TAKEN,
    CODE_ROLE_NOT_FOUND,
    ROLE_LIST_FIELDS,
    InvalidGrant,
    RoleNameTaken,
    RoleNotFound,
    RoleService,
)
from .users import (
    CODE_EMAIL_TAKEN,
    CODE_INVALID_ROLE,
    CODE_USER_CONFLICT,
    CODE_USER_NOT_FOUND,
    CODE_USERNAME_TAKEN,
    DuplicateUser,
    EmailTaken,
    InvalidRole,
    USER_LIST_FIELDS,
    UsernameTaken,
    UserNotFound,
    UserService,
)

__all__ = [
    "ADMIN_ROLE_NAME",
    "AccessManagementError",
    "AdminBootstrapService",
    "AlreadyBootstrapped",
    "CODE_LAST_ADMIN_PROTECTED",
    "CODE_LAST_ADMIN_ROLE_PROTECTED",
    "LastAdminProtected",
    "LastAdminRoleProtected",
    "active_admins",
    "get_admin_role",
    "is_last_active_admin",
    "CATALOG_LIST_FIELDS",
    "CODE_INVALID_RESOURCE",
    "CODE_PERMISSION_INACTIVE",
    "CODE_PERMISSION_NOT_FOUND",
    "CODE_ROLE_INACTIVE",
    "CODE_RESOURCE_NOT_FOUND",
    "CatalogDiscoveryService",
    "CatalogService",
    "DiscoveryReport",
    "SourceDataScope",
    "TableScope",
    "UNRESOLVED",
    "compute_data_scope",
    "resolve_effective_permissions",
    "serialize_data_scope",
    "ConflictError",
    "EffectivePermissions",
    "Grant",
    "NO_PERMISSIONS",
    "PermissionResolver",
    "InvalidResourcePath",
    "PermissionInactive",
    "RoleInactive",
    "RolePermissionService",
    "UserRoleService",
    "ResourceNotFound",
    "NotFoundError",
    "PERMISSION_LIST_FIELDS",
    "PermissionNotFound",
    "PermissionService",
    "CODE_ROLE_NAME_TAKEN",
    "CODE_ROLE_NOT_FOUND",
    "CODE_EMAIL_TAKEN",
    "CODE_INVALID_GRANT",
    "CODE_INVALID_ROLE",
    "CODE_USERNAME_TAKEN",
    "CODE_USER_CONFLICT",
    "CODE_USER_NOT_FOUND",
    "DuplicateUser",
    "EmailTaken",
    "InvalidGrant",
    "InvalidRole",
    "ROLE_LIST_FIELDS",
    "RoleNameTaken",
    "RoleNotFound",
    "RoleService",
    "USER_LIST_FIELDS",
    "UserNotFound",
    "UserService",
    "UsernameTaken",
    "paginate",
    "role_stats",
]
