"""apps.substrate — everything ingestion produces, as models (migration_plan.md §5, §6)."""
from django.apps import AppConfig


class SubstrateConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.substrate"
