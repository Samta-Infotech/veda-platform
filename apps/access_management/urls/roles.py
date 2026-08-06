"""Role-administration routes.

``<resource>/<action>``, all POST — the platform convention (see ``urls/users.py``).

No ``roles/delete``: a role is retired with ``roles/update {is_active: false}``.
Hard-deletion semantics depend on role assignment, which does not exist yet, and an
audit trail that says "granted role #7" must still be able to resolve #7.
"""
from django.urls import path

from ..views import RoleCreateView, RoleDetailView, RoleListView, RoleUpdateView

urlpatterns = [
    path("roles/create", RoleCreateView.as_view(), name="role-create"),
    path("roles/detail", RoleDetailView.as_view(), name="role-detail"),
    path("roles/list", RoleListView.as_view(), name="role-list"),
    path("roles/update", RoleUpdateView.as_view(), name="role-update"),
]
