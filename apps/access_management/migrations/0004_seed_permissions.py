"""Seed the permission catalogue.

Permissions are CODE-DEFINED (see ``models/permissions.py``): only code can enforce
one, so the catalogue lives here rather than being creatable through the API.

Every entry below maps to an action the platform actually performs today — the
comment names the endpoint or path that will check it. Nothing speculative is seeded:
a permission with no code path to gate it is a promise the system cannot keep, and it
would show an administrator authority that does not exist.

IDEMPOTENT and re-runnable: keyed on ``code`` via ``update_or_create``, so the name
and description in this file stay the source of truth. ``is_active`` is deliberately
NOT in the defaults — an operator who switches a capability off must not have that
undone by a later deploy.

Reverse deletes only the codes this migration introduced, never the whole table.
"""
from django.db import migrations

#: (code, name, description) — each grounded in an existing code path.
PERMISSIONS = [
    # apps/query/views.py::QueryView, apps/chat/views.py::ConversationQueryView
    ("query.execute", "Execute queries",
     "Run analytical queries and conversational turns against the platform."),
    # The data those queries reach. This is the permission the grant phase will
    # scope per source/table/column, against the existing catalog.
    ("data.read", "Read data",
     "Read data from the sources a role has been granted."),
    # apps/sources — source registration and connection configuration.
    ("source.manage", "Manage data sources",
     "Register, configure and retire data sources."),
    # apps/query/views.py::IngestTriggerView (currently IsAdminUser)
    ("ingestion.run", "Run ingestion",
     "Trigger ingestion of a data source."),
    # apps/query/views.py::EvalTriggerView (currently IsAdminUser)
    ("evaluation.run", "Run evaluations",
     "Trigger evaluation runs and read their results."),
    # apps/access_management users/* (currently IsAdminUser)
    ("user.manage", "Manage users",
     "Create, view and update user accounts."),
    # apps/access_management roles/* (currently IsAdminUser)
    ("role.manage", "Manage roles",
     "Create, view, update and retire roles."),
    # apps/access_management permissions/* — read-only by design.
    ("permission.read", "View permissions",
     "View the catalogue of permissions the platform can grant."),
]


def seed_permissions(apps, schema_editor):
    Permission = apps.get_model("access_management", "Permission")
    for code, name, description in PERMISSIONS:
        Permission.objects.update_or_create(
            code=code, defaults={"name": name, "description": description})


def unseed_permissions(apps, schema_editor):
    Permission = apps.get_model("access_management", "Permission")
    Permission.objects.filter(code__in=[code for code, _, _ in PERMISSIONS]).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("access_management", "0003_permission"),
    ]

    operations = [
        migrations.RunPython(seed_permissions, unseed_permissions),
    ]
