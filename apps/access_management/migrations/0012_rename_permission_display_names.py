"""Shorten three permission DISPLAY NAMES to the resource they govern.

    source.manage   "Manage data sources"  ->  "Data Sources"
    user.manage     "Manage users"         ->  "Users"
    role.manage     "Manage roles"         ->  "Roles"

Only ``name`` changes. The ``code`` is the enforcement key — every
``required_permission`` declaration, seeded grant and stored AuthorizationDecision
references it — so renaming a code would silently stop matching existing grants.
``codes.py`` is therefore untouched too.

Why a NEW migration rather than only editing 0004_seed_permissions: 0004 has already
run on every deployed database, and Django never re-runs an applied migration, so an
edit there reaches fresh installs ONLY. Both are needed, and the drift is already
visible in this database — ``data.read`` reads "Read Data Sources" while 0004 still
says "Read data", i.e. a display name was changed without a migration to carry it.

Keyed on ``code`` and matched on the OLD name, so it is idempotent and will not
overwrite a name an operator has since customised. Reverse restores the old names.
"""
from django.db import migrations

#: (code, old_name, new_name)
RENAMES = [
    ("source.manage", "Manage data sources", "Data Sources"),
    ("user.manage", "Manage users", "Users"),
    ("role.manage", "Manage roles", "Roles"),
]


def _apply(apps, schema_editor, forward=True):
    Permission = apps.get_model("access_management", "Permission")
    for code, old_name, new_name in RENAMES:
        frm, to = (old_name, new_name) if forward else (new_name, old_name)
        Permission.objects.filter(code=code, name=frm).update(name=to)


def rename_forward(apps, schema_editor):
    _apply(apps, schema_editor, forward=True)


def rename_backward(apps, schema_editor):
    _apply(apps, schema_editor, forward=False)


class Migration(migrations.Migration):

    dependencies = [
        ("access_management", "0011_authorizationdecision"),
    ]

    operations = [
        migrations.RunPython(rename_forward, rename_backward),
    ]
