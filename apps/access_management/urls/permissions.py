"""Permission-catalogue routes — read-only.

No ``create``/``update``/``delete``: the catalogue is code-defined and seeded by
migration, because only code can enforce a permission. Adding a write route here
would let an administrator create authority that nothing checks.
"""
from django.urls import path

from ..views import PermissionDetailView, PermissionListView

urlpatterns = [
    path("permissions/detail", PermissionDetailView.as_view(), name="permission-detail"),
    path("permissions/list", PermissionListView.as_view(), name="permission-list"),
]
