"""apps.query admin — append-only QueryLog view (migration_plan.md §6.6)."""
from django.contrib import admin

from .models import QueryLog


@admin.register(QueryLog)
class QueryLogAdmin(admin.ModelAdmin):
    list_display = ("id", "tenant", "route", "status", "latency_ms", "created_at")
    list_filter = ("status", "route")
    search_fields = ("query_text", "request_id")
    readonly_fields = [f.name for f in QueryLog._meta.fields]

    def has_add_permission(self, request):
        return False  # append-only, written by the query path

    def has_change_permission(self, request, obj=None):
        return False
