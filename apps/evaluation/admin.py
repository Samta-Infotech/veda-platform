"""apps.evaluation admin (migration_plan.md §5)."""
from django.contrib import admin

from .models import EvalCaseResult, EvalRun


class EvalCaseResultInline(admin.TabularInline):
    model = EvalCaseResult
    extra = 0


@admin.register(EvalRun)
class EvalRunAdmin(admin.ModelAdmin):
    list_display = ("id", "label", "source", "recall_at_k", "hit_rate", "sql_success_rate", "created_at")
    inlines = [EvalCaseResultInline]
