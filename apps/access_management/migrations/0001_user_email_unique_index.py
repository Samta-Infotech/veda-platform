"""Case-insensitive unique index on non-blank ``auth_user.email``.

WHY THIS EXISTS
    ``POST /api/v1/users`` must reject a duplicate email, and Django's stock User
    model declares ``email`` as ``unique=False``. Without a database constraint the
    only option is "SELECT, then INSERT if absent", which is a check-then-insert
    race: two concurrent requests both find nothing and both succeed. A uniqueness
    rule that only holds when requests are serial is not a uniqueness rule.

WHY RAW SQL
    The rule is *case-insensitive* and *partial* — ``LOWER(email)`` unique, but only
    ``WHERE email <> ''``. The partial predicate is what preserves the existing
    blank-email accounts (the seeded ``admin`` is one) and keeps email effectively
    optional at the schema level. Expressing this as a Django ``UniqueConstraint``
    would mean altering the state of a model this app does not own (``auth.User``),
    so raw SQL with an explicit reverse is the honest route. Precedent for RunSQL
    DDL in this project: ``apps/substrate/migrations/0002_pgvector.py``.

PORTABILITY
    Verified on both engines this project runs: PostgreSQL, and SQLite 3.46 (partial
    indexes since 3.8, expression indexes since 3.9). Confirmed by test that
    ``'A@b.com'`` and ``'a@B.COM'`` collide while two blank emails coexist.

    ``auth_user`` is hard-coded rather than read from the model's ``db_table``
    because this migration is only valid for the stock, non-swapped User model —
    which is the one dependency below asserts.

SAFETY
    Additive and reversible; no column, table or row is touched. The pre-check
    refuses to run if the data already violates the rule, naming the offending
    addresses, so the failure is actionable instead of a bare
    "UNIQUE constraint failed" from the index build.
"""
from django.conf import settings
from django.db import migrations
from django.db.models import Count
from django.db.models.functions import Lower

INDEX_NAME = "access_mgmt_user_email_ci_uniq"

_CREATE_INDEX = (
    f'CREATE UNIQUE INDEX IF NOT EXISTS "{INDEX_NAME}" '
    'ON "auth_user" (LOWER("email")) '
    "WHERE \"email\" <> '';"
)
_DROP_INDEX = f'DROP INDEX IF EXISTS "{INDEX_NAME}";'


def assert_no_duplicate_emails(apps, schema_editor):
    """Refuse to build the index over data that already breaks the rule.

    Runs before the DDL so a deployment with pre-existing duplicates gets a message
    naming them rather than an opaque constraint error. Remediation is a human
    decision (which account keeps the address), so this deliberately does not
    "fix" anything automatically.
    """
    User = apps.get_model("auth", "User")
    duplicates = (
        User.objects.exclude(email="")
        .annotate(normalized=Lower("email"))
        .values("normalized")
        .annotate(total=Count("id"))
        .filter(total__gt=1)
        .order_by("normalized")
    )
    offenders = [f"{row['normalized']} ({row['total']} accounts)" for row in duplicates]
    if offenders:
        raise RuntimeError(
            "Cannot add a unique index on auth_user.email: these addresses are "
            "already used by more than one account (compared case-insensitively). "
            "Resolve them first, then re-run the migration.\n  "
            + "\n  ".join(offenders)
        )


def noop_reverse(apps, schema_editor):
    """Nothing to undo — the check only reads."""


class Migration(migrations.Migration):

    dependencies = [
        # Ties this migration to the configured user model so it cannot silently
        # apply to a swapped one.
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        # MUST be __latest__, not the __first__ that swappable_dependency resolves
        # to. On SQLite, Django implements ALTER by REBUILDING the table (create
        # new, copy, drop, rename), which silently discards indexes it does not
        # know about. Several auth migrations alter the user model
        # (0004/0005/0008/0009/0012_alter_user_*), so running before them left the
        # migration recorded as applied with no index actually present — uniqueness
        # was silently unenforced. Verified by a test that asserts the constraint,
        # not just the migration record.
        ("auth", "__latest__"),
    ]

    operations = [
        migrations.RunPython(assert_no_duplicate_emails, noop_reverse),
        migrations.RunSQL(sql=_CREATE_INDEX, reverse_sql=_DROP_INDEX),
    ]
