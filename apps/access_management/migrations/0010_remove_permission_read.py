"""Remove the ``permission.read`` permission.

It gated ``permissions/{list,detail,dropdown}`` and ``catalog/{list,detail,tree}`` — six
read-only endpoints that browse the permission catalogue and the resource tree. Those are
exactly the screens an admin opens *in order to* grant permissions, so a separate
permission to read the list of permissions bought nothing: anyone who could act on the
data already had to be staff. The six views now declare
``permission_classes = [IsAdminUser]`` instead (see views/permissions.py, views/catalog.py)
— dropping RequiresPermission outright rather than leaving a blank
``required_permission``, which gate.py:90 fails closed on and would 403 every caller under
``VEDA_RBAC_MODE=enforce``.

0004_seed_permissions is deliberately left as it was — an applied migration is history.
This one converges both a fresh database (0004 seeds it, this removes it) and an existing
one, so the end state does not depend on when the environment was built.

RolePermission.permission is ``on_delete=PROTECT`` — deliberately, so a permission cannot
be dropped out from under live grants by accident. So the grants go first, explicitly:
12 of them at the time of writing, all of them grants OF this permission. That is the
intent — the permission no longer exists, so a grant of it is meaningless. No other
permission is touched, so every other grant on those roles survives.
"""
from django.db import migrations

CODE = "permission.read"


def remove_permission(apps, schema_editor):
    Permission = apps.get_model("access_management", "Permission")
    RolePermission = apps.get_model("access_management", "RolePermission")
    # PROTECT on the FK means the grants must be cleared before the permission itself.
    RolePermission.objects.filter(permission__code=CODE).delete()
    Permission.objects.filter(code=CODE).delete()


def restore_permission(apps, schema_editor):
    """Recreate the row so the migration is reversible.

    The grants that were cascade-deleted are NOT restored — this migration does not
    record them, and inventing grants during a rollback would hand out authority nobody
    asked for. Re-grant explicitly if a rollback ever needs them.
    """
    Permission = apps.get_model("access_management", "Permission")
    Permission.objects.get_or_create(
        code=CODE,
        defaults={
            "name": "View permissions",
            "description": "View the catalogue of permissions the platform can grant.",
        },
    )


class Migration(migrations.Migration):

    dependencies = [
        ("access_management", "0009_role_deleted_at"),
    ]

    operations = [
        migrations.RunPython(remove_permission, restore_permission),
    ]
