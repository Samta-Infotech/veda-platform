"""apps.core — tenancy base models, mixins, health, settings bridge (migration_plan.md §5)."""
from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.core"
