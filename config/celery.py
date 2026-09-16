"""Celery app for the ingestion tier (migration_plan.md §4.1, §7).

Broker is the dedicated `redis-broker` instance (never the cache instance
— see the split-Redis callout in §3/§1.2). Queues match the L0 stage plan:
`ingestion` for L0 stage tasks, `high` for cache warming, `default` for
everything else.
"""
import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")

app = Celery("veda")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.conf.task_default_queue = "default"
app.conf.task_queues = {
    "ingestion": {},
    "high": {},
    "default": {},
}
app.autodiscover_tasks()


# Startup env-drift check (2026-09-15): the ingest-worker ran for days with a stale
# METAL_EMBED_URL because `docker compose restart` never re-reads .env. Log the
# effective values and warn loudly on drift — see storage_adapters/env_drift.py.
from celery.signals import worker_ready as _worker_ready  # noqa: E402


@_worker_ready.connect
def _log_env_drift(sender=None, **kwargs):
    try:
        from storage_adapters.env_drift import check_env_drift
        check_env_drift(f"celery:{getattr(sender, 'hostname', 'worker')}")
    except Exception:
        pass
