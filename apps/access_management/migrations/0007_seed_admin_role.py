"""Seed the "Admin" role and grant it every permission that exists today.

WHY A MIGRATION, NOT RUNTIME CODE
    Mirrors 0004_seed_permissions: the bootstrap command (management command
    ``bootstrap_admin``) only ASSIGNS this role to the first user — it must not also
    be the thing that decides what the role is allowed to do. Keeping "what Admin can
    do" in a migration means it is versioned, re-runnable, and visible in the same
    place every other permission grant is seeded, rather than being invented lazily
    the first time someone runs the bootstrap command.

WHY EVERY PERMISSION, GLOBALLY (``resource_path=""``)
    This role exists to be authoritative — the account of last resort when nothing
    else can act. A partial grant would defeat that purpose the first time a resource
    is added that Admin was never explicitly granted.

IDEMPOTENT and re-runnable, exactly like 0004: keyed on name / (role, permission,
resource_path) via ``get_or_create``/``update_or_create``. Running this twice grants
nothing extra and revokes nothing an operator has since changed — ``is_active`` on the
role, and ``effect`` on any grant an operator flipped to DENY, are left alone once
they already exist.

Reverse removes only the grants this migration created for the Admin role, and the
role itself only if it is still exactly what this migration made (no other grants,
no assignments) — deleting a role that has since accrued real assignments would
silently break every admin bootstrapped through it.
"""
from django.db import migrations

#: Same literal the model's TextChoices resolves to (see models/grants.py::Effect) —
#: not imported, because a historical migration must not depend on app code that can
#: change shape after this migration is written.
_EFFECT_ALLOW = "allow"

#: The natural key every later piece of code (admin_guard.py, bootstrap_admin) looks
#: this role up by. Case-insensitively unique already (migration 0002's constraint).
ADMIN_ROLE_NAME = "Admin"


def seed_admin_role(apps, schema_editor):
    Role = apps.get_model("access_management", "Role")
    Permission = apps.get_model("access_management", "Permission")
    RolePermission = apps.get_model("access_management", "RolePermission")

    admin_role, _ = Role.objects.get_or_create(
        name=ADMIN_ROLE_NAME,
        defaults={"description": "Full authority over the platform. Reserved for "
                                  "the account(s) bootstrapped as System Admin."})

    for permission in Permission.objects.all():
        RolePermission.objects.update_or_create(
            role=admin_role, permission=permission, resource_path="",
            defaults={"effect": _EFFECT_ALLOW})


def unseed_admin_role(apps, schema_editor):
    Role = apps.get_model("access_management", "Role")
    RolePermission = apps.get_model("access_management", "RolePermission")
    UserRole = apps.get_model("access_management", "UserRole")

    admin_role = Role.objects.filter(name=ADMIN_ROLE_NAME).first()
    if admin_role is None:
        return
    RolePermission.objects.filter(role=admin_role, resource_path="").delete()
    # Only delete the role itself if nothing has come to depend on it since.
    if not UserRole.objects.filter(role=admin_role).exists() and \
            not RolePermission.objects.filter(role=admin_role).exists():
        admin_role.delete()


class Migration(migrations.Migration):

    dependencies = [
        ("access_management", "0006_grants"),
    ]

    operations = [
        migrations.RunPython(seed_admin_role, unseed_admin_role),
    ]
