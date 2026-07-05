"""apps.evaluation — eval runs, case results, HTML report artifact (migration_plan.md §5)."""
from django.apps import AppConfig


class EvaluationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.evaluation"
