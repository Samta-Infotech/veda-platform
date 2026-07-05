"""Django project package (migration_plan.md §4).

Imports the Celery app lazily so `config.celery.app` is always reachable
as `app` for `@shared_task` autodiscovery, without hard-crashing modules
(e.g. management commands, tests) that run before Celery is installed.
"""
try:
    from .celery import app as celery_app

    __all__ = ("celery_app",)
except ImportError:
    __all__ = ()
