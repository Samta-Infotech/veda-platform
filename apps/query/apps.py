"""apps.query — DRF QueryView, audit (QueryLog), inference client (migration_plan.md §5)."""
from django.apps import AppConfig


class QueryConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.query"
