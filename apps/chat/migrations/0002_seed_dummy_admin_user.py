"""Seeds the hardcoded dev/testing login user (username=admin, password=admin123).

Idempotent: get_or_create by username, so re-running (or a fresh DB) always
ends up with exactly this one dummy account. Remove this migration (and the
user) once the real authentication service replaces LoginView.
"""
from django.conf import settings
from django.contrib.auth.hashers import make_password
from django.db import migrations

_DUMMY_USERNAME = "admin"
_DUMMY_PASSWORD = "admin123"


def seed_admin_user(apps, schema_editor):
    User = apps.get_model(settings.AUTH_USER_MODEL)
    User.objects.get_or_create(
        username=_DUMMY_USERNAME,
        defaults={
            "password": make_password(_DUMMY_PASSWORD),
            "first_name": "Administrator",
            "is_active": True,
        },
    )


def unseed_admin_user(apps, schema_editor):
    User = apps.get_model(settings.AUTH_USER_MODEL)
    User.objects.filter(username=_DUMMY_USERNAME).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0001_initial"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.RunPython(seed_admin_user, unseed_admin_user),
    ]
