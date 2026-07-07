"""Local development settings (migration_plan.md §0.1).

Sqlite fallback keeps `manage.py` usable without Postgres/PgBouncer running;
switch to the `default` Postgres alias from base.py once infra is up.
SLM_BACKEND defaults to ollama per §8b (dev/ingestion tier).
"""
import os

from .base import *  # noqa: F401,F403

DEBUG = True
ALLOWED_HOSTS = ["*"]

# Django 4.0+ checks the request Origin against this list for unsafe methods (POST),
# incl. the admin login. Requests arrive through nginx on :8080, so that origin must
# be trusted explicitly (ALLOWED_HOSTS="*" does NOT cover CSRF origin checking).
CSRF_TRUSTED_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

if os.environ.get("VEDA_DB_HOST") is None and os.environ.get("PGBOUNCER_HOST") is None:
    DATABASES["default"] = {  # noqa: F405
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",  # noqa: F405
    }

VEDA.setdefault("SLM_BACKEND", "ollama")  # noqa: F405
