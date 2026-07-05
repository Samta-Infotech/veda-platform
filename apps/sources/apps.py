"""apps.sources — registry of queryable DBs + connection config (migration_plan.md §5)."""
from django.apps import AppConfig


class SourcesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.sources"
