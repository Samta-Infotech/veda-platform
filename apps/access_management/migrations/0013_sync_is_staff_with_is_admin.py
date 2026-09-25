"""Backfill ``is_staff`` for accounts whose ``is_admin`` (= ``is_superuser``) is on.

These are two columns but one decision for this product, and they had drifted. The
API writes ``is_superuser`` (its public name is ``is_admin``, the "which frontend app"
flag checked at login) and deliberately refuses to write ``is_staff`` — no serializer
accepts it. But ``AdminView`` gates every access-management endpoint on
``IsAdminUser``, which reads exactly ``is_staff``.

The result: an account created through the product with ``is_admin: true`` was sent to
the admin frontend at login and then got 403 from every request that frontend made.
Measured on this database — 6 accounts had ``is_superuser=true, is_staff=false``, one of
them holding a role that grants user.manage / role.manage / source.manage in full. RBAC
allowed it; ``is_staff`` denied it; only the bootstrap superuser could use the product.

``UserService.create_user`` / ``update_user`` now keep the two in step going forward;
this carries the accounts that already exist.

Deliberately one-directional: forward promotes ``is_superuser`` accounts to staff, and
the reverse is a NO-OP. Reversing it would have to guess which of the resulting staff
accounts were staff before this ran, and guessing wrong revokes a real administrator's
access — an un-migration is not worth that. Nothing here is destructive to undo.
"""
from django.db import migrations


def sync_forward(apps, schema_editor):
    User = apps.get_model("auth", "User")
    User.objects.filter(is_superuser=True, is_staff=False).update(is_staff=True)


def sync_backward(apps, schema_editor):
    """No-op — see the module docstring."""


class Migration(migrations.Migration):

    dependencies = [
        ("access_management", "0012_rename_permission_display_names"),
        ("auth", "0012_alter_user_first_name_max_length"),
    ]

    operations = [
        migrations.RunPython(sync_forward, sync_backward),
    ]
