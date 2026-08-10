"""Grant routes — role assignment and permission grants.

``<resource>/<action>``, all POST, matching the platform convention. Nested under the
noun that owns the edge: assignments hang off ``users/``, grants off ``roles/``.
"""
from django.urls import path

from ..views import (
    RolePermissionGrantView,
    RolePermissionListView,
    RolePermissionRevokeView,
    UserRoleAssignView,
    UserRoleListView,
    UserRoleRevokeView,
)

urlpatterns = [
    path("users/roles/assign", UserRoleAssignView.as_view(), name="user-role-assign"),
    path("users/roles/revoke", UserRoleRevokeView.as_view(), name="user-role-revoke"),
    path("users/roles/list", UserRoleListView.as_view(), name="user-role-list"),
    path("roles/permissions/grant", RolePermissionGrantView.as_view(),
         name="role-permission-grant"),
    path("roles/permissions/revoke", RolePermissionRevokeView.as_view(),
         name="role-permission-revoke"),
    path("roles/permissions/list", RolePermissionListView.as_view(),
         name="role-permission-list"),
]
