"""apps.access_management — identity administration and authorization (RBAC).

The bounded context for *managing* who exists and what they may do: user
administration today, and the home for role management, permission management,
permission resolution and authorization utilities as those phases land.

Deliberately separate from ``apps.authentication``, which owns only identity
*verification* — proving a caller is who they claim (login, refresh, logout, JWT,
password lifecycle). One app answers "who is this?", this one answers "who is
allowed to exist, and to do what?".

Current scope is user creation alone. Nothing here anticipates the RBAC model:
no role field, no permission table, no resolution hook. Those arrive with their
own phases and their own design.
"""
from django.apps import AppConfig


class AccessManagementConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.access_management"
    verbose_name = "Access Management"
