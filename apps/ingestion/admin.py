"""apps.ingestion admin — job/stage visibility + enqueue actions (migration_plan.md §5, §4.3)."""
from django.contrib import admin

from .models import IngestionJob, IngestionStage


class IngestionStageInline(admin.TabularInline):
    model = IngestionStage
    extra = 0
    readonly_fields = ("order", "name", "status", "row_count", "error_traceback")


@admin.register(IngestionJob)
class IngestionJobAdmin(admin.ModelAdmin):
    list_display = ("id", "source", "tenant", "status", "started_at", "finished_at")
    list_filter = ("status",)
    inlines = [IngestionStageInline]
    actions = ["reingest", "rebuild_embeddings", "regenerate_glossary", "warm_caches"]

    @admin.action(description="Re-ingest source")
    def reingest(self, request, queryset):
        from .tasks import task_ingest_source
        n = 0
        for job in queryset:
            task_ingest_source.delay(source_id=job.source_id, tenant=job.tenant, force=True)
            n += 1
        self.message_user(request, f"Enqueued {n} re-ingestion job(s) on the ingestion queue.")

    @admin.action(description="Rebuild embeddings only")
    def rebuild_embeddings(self, request, queryset):
        self._stub(request)

    @admin.action(description="Regenerate glossary")
    def regenerate_glossary(self, request, queryset):
        self._stub(request)

    @admin.action(description="Warm caches")
    def warm_caches(self, request, queryset):
        self._stub(request)

    def _stub(self, request):
        # Phase 4.3: enqueue the corresponding Celery task.
        self.message_user(request, "Enqueue action is a Phase 4 stub.")
