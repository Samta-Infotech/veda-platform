"""apps.ingestion — Celery L0 pipeline, job tracking, admin actions (migration_plan.md §5, §7)."""
from django.apps import AppConfig


class IngestionConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.ingestion"
