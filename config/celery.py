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
