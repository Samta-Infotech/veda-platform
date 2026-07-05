"""Root urlconf (migration_plan.md §4, §5 apps.core).

Health/readiness endpoints live here as thin views per §5 apps.core
("Health & readiness") until that app grows a dedicated views module;
`/readyz` is a liveness-only stub for now — the real checks (Postgres via
PgBouncer, both Redis instances, inference reachability, SLM backend) land
with the api tier in Phase 6.
"""
from django.contrib import admin
from django.http import JsonResponse
from django.urls import include, path

from apps.core import views as core_views


def healthz(request):
    return JsonResponse({"status": "ok"})


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include("apps.query.urls")),
    path("healthz", healthz),
    path("readyz", core_views.readyz),   # real readiness (Postgres via PgBouncer + Redis)
    path("metrics", core_views.metrics),  # Prometheus text (§6.3)
]
