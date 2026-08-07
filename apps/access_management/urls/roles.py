"""Role-administration routes.

``<resource>/<action>``, all POST — the platform convention (see ``urls/users.py``).

``roles/delete`` is a SOFT delete — a named convenience over ``roles/update
{is_active: false}``, not a second code path. No row is ever removed:
hard-deletion's semantics depend on role assignment (what happens to the users
holding it?), and an audit trail that says "granted role #7" must still be able to
resolve #7.
"""
from django.urls import path

from ..views import (
    RoleCreateView,
    RoleDeleteView,
    RoleDetailView,
    RoleDropdownView,
    RoleListView,
    RoleUpdateView,
)

urlpatterns = [
    path("roles/create", RoleCreateView.as_view(), name="role-create"),
    path("roles/detail", RoleDetailView.as_view(), name="role-detail"),
    path("roles/list", RoleListView.as_view(), name="role-list"),
    path("roles/dropdown", RoleDropdownView.as_view(), name="role-dropdown"),
    path("roles/update", RoleUpdateView.as_view(), name="role-update"),
    path("roles/delete", RoleDeleteView.as_view(), name="role-delete"),
]
