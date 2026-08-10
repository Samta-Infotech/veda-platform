"""Domain models for access management, one module per aggregate.

A package rather than a single ``models.py`` for the same reason the other layers in
this app are packages: this is the RBAC bounded context. Each aggregate gets its
own module instead of accreting into one file — the two grant EDGES share
``grants.py`` because they are one concept (see that module's docstring).

Django requires models to be importable from the app's ``models`` namespace, which is
what the re-export below provides.
"""
from .catalog import CatalogResource
from .grants import Effect, RolePermission, UserRole
from .permissions import Permission
from .profile import UserProfile
from .roles import Role

__all__ = ["CatalogResource", "Effect", "Permission", "Role", "RolePermission",
          "UserProfile", "UserRole"]
