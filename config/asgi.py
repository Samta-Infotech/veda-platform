"""ASGI entrypoint for the Django api tier (migration_plan.md §4).

Note: this is the `api` container's ASGI app (auth, tenancy, admin, DRF).
It is distinct from the standalone `inference/` ASGI service in §8, which
warm-loads the heavy models and is not routed through Django at all.
"""
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")

application = get_asgi_application()
