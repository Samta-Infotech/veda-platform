"""apps.sources admin — includes a "test connection" action (migration_plan.md §2.2)."""
from django.contrib import admin

from .models import Source, SourceConnectionProfile


@admin.register(Source)
class SourceAdmin(admin.ModelAdmin):
    list_display = ("name", "dialect", "status", "ready", "last_ingested_at")
    list_filter = ("dialect", "status", "ready")
    actions = ["ingest", "test_connection"]

    @admin.action(description="Ingest source (enqueue ingestion job)")
    def ingest(self, request, queryset):
        """Enqueue task_ingest_source for each selected source on the `ingestion`
        queue (processed by the ingest-worker). Works for first-time ingestion and
        re-ingestion alike."""
        from apps.ingestion.tasks import task_ingest_source
        n = 0
        for src in queryset:
            task_ingest_source.delay(source_id=src.pk, tenant="default", force=True)
            n += 1
        self.message_user(request, f"Enqueued {n} ingestion job(s) on the ingestion queue.")

    @admin.action(description="Test connection")
    def test_connection(self, request, queryset):
        # Phase 6: delegate to a sources service that opens a read-only probe.
        self.message_user(request, "test_connection is a Phase 6 stub.")


@admin.register(SourceConnectionProfile)
class SourceConnectionProfileAdmin(admin.ModelAdmin):
    list_display = ("source", "pool_max_size", "statement_timeout_ms", "read_only_role")
