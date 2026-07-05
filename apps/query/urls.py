"""apps.query URLs — mounted under /api/v1/ by config.urls (migration_plan.md §5).

``POST /api/v1/query`` is the platform's single HTTP entry to the flow. The
QueryView is a Phase 6 skeleton; the route is registered now so the project
wiring (config.urls includes this module) is complete.
"""
from django.urls import path

from .views import EvalTriggerView, IngestTriggerView, QueryView

urlpatterns = [
    path("query", QueryView.as_view(), name="query"),
    path("admin/ingest", IngestTriggerView.as_view(), name="admin-ingest"),  # staff-only (§6.4)
    path("admin/eval", EvalTriggerView.as_view(), name="admin-eval"),        # staff-only (§6.4)
]
