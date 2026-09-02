"""apps.sources admin — includes a "test connection" action (migration_plan.md §2.2)."""
from django.contrib import admin

from .models import Source, SourceConnectionProfile
from apps.ingestion.tasks import task_ingest_source


@admin.register(Source)
class SourceAdmin(admin.ModelAdmin):
    list_display = ("name", "dialect", "status", "ready", "is_canonical", "last_ingested_at")
    list_filter = ("dialect", "status", "ready", "is_canonical")
    actions = ["ingest", "test_connection"]
    # `description_generated` is provenance the source_profiler manages (True when the
    # description was auto-generated, so a re-ingest may refresh it) — humans read it,
    # never set it. `last_ingested_at`/timestamps are engine-managed too.
    readonly_fields = ("description_generated", "last_ingested_at", "created_at", "updated_at")
    # Group the routing-catalog profile fields (Phase 1) separately from connection config,
    # so the admin editing a source can set domain/description/canonical without hunting
    # through connection fields. Fields left off the connection fieldsets still render under
    # Django's default "everything else" behaviour is disabled once fieldsets are declared,
    # so every editable field a human should touch is listed explicitly below.
    fieldsets = (
        (None, {"fields": ("name", "dialect", "connector_type", "status", "ready")}),
        ("Routing catalog (multi-source routing)", {
            "fields": ("domain_tags", "description", "description_generated", "is_canonical"),
            "description": "Business-facing profile the query router reads. `description` may be "
                           "entered manually (wins) or auto-generated post-ingestion when blank; "
                           "`is_canonical`/`domain_tags` are manual, never auto-inferred.",
        }),
        ("Connection", {
            "classes": ("collapse",),
            "fields": ("host", "port", "dbname", "db_user", "password_env", "password_inline",
                       "connection_secret_ref", "schema_filter", "exclude_tables"),
        }),
        ("Document / datalake", {
            "classes": ("collapse",),
            "fields": ("source_path", "doc_formats", "doc_recursive", "doc_max_file_mb"),
        }),
        ("Timestamps", {
            "classes": ("collapse",),
            "fields": ("last_ingested_at", "created_at", "updated_at"),
        }),
    )

    @admin.action(description="Ingest source (enqueue ingestion job)")
    def ingest(self, request, queryset):
        """Enqueue task_ingest_source for each selected source on the `ingestion`
        queue (processed by the ingest-worker). Works for first-time ingestion and
        re-ingestion alike."""
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
