"""Django-admin registration for access management.

Roles are editable here because the API cannot bootstrap the first ones any more than
it can bootstrap the first administrator (see ACCESS_MANAGEMENT_API_CONTRACT.md §7) —
an operator needs some way in. Timestamps stay read-only: they are set by the model.
"""
from django.contrib import admin

from .models import CatalogResource, Permission, Role, RolePermission, UserRole


@admin.register(Role)
class RoleAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "is_active", "created_at", "updated_at")
    list_filter = ("is_active",)
    search_fields = ("name", "description")
    readonly_fields = ("created_at", "updated_at")
    ordering = ("name",)


@admin.register(Permission)
class PermissionAdmin(admin.ModelAdmin):
    """Read-mostly: the catalogue is code-defined and seeded by migration 0004.

    Adding and deleting are disabled here for the same reason there is no create API —
    a permission the code does not check enforces nothing. ``is_active`` stays
    editable so an operator can switch a capability off without a deploy.
    """

    list_display = ("id", "code", "name", "is_active", "updated_at")
    list_filter = ("is_active",)
    search_fields = ("code", "name", "description")
    readonly_fields = ("code", "name", "description", "created_at", "updated_at")
    ordering = ("code",)

    def has_add_permission(self, request):
        return False  # seeded by migration; see models/permissions.py

    def has_delete_permission(self, request, obj=None):
        return False  # grants and audit history reference these rows


@admin.register(CatalogResource)
class CatalogResourceAdmin(admin.ModelAdmin):
    """Fully read-only: rows are a projection rebuilt by ``manage.py sync_catalog``.

    Hand-editing would be edited away by the next discovery run, and a hand-created
    row would name a resource nothing upstream corresponds to.
    """

    list_display = ("path", "kind", "source", "is_active", "updated_at")
    list_filter = ("kind", "is_active", "source")
    search_fields = ("path",)
    readonly_fields = [f.name for f in CatalogResource._meta.fields]
    ordering = ("path",)

    def has_add_permission(self, request):
        return False  # produced by discovery, not authored

    def has_change_permission(self, request, obj=None):
        return False  # any edit is overwritten by the next sync

    def has_delete_permission(self, request, obj=None):
        return False  # deleting would silently drop the grants referencing this path


@admin.register(UserRole)
class UserRoleAdmin(admin.ModelAdmin):
    """Read-only in admin: assignments are audited edges, and the API records
    ``granted_by``. Hand-editing here would produce a grant with no attributed actor."""

    list_display = ("id", "user", "role", "granted_by", "created_at")
    list_filter = ("role",)
    search_fields = ("user__username", "role__name")
    readonly_fields = [f.name for f in UserRole._meta.fields]

    def has_add_permission(self, request):
        return False  # use the API, which attributes granted_by

    def has_change_permission(self, request, obj=None):
        return False


@admin.register(RolePermission)
class RolePermissionAdmin(admin.ModelAdmin):
    """Read-only, same reasoning as UserRoleAdmin."""

    list_display = ("id", "role", "permission", "resource_path", "effect", "created_at")
    list_filter = ("effect", "permission")
    search_fields = ("role__name", "permission__code", "resource_path")
    readonly_fields = [f.name for f in RolePermission._meta.fields]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
