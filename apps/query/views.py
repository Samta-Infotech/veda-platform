"""apps.query.views — QueryView (DRF) (migration_plan.md §5, §6.1, §6.2).

Thin view: validate → resolve tenant (server-side; dev falls back to "default")
→ call InferenceClient → persist QueryLog → return MultiResult with ``status``
preserved verbatim. A refusal or an unreachable inference tier is a structured
JSON payload with an appropriate code, never a leaked 500 (§9a, §18).

Auth/JWT + tenant-from-principal is the Phase 6.2 hardening; dev uses AllowAny and
a request-supplied/default tenant so the end-to-end path is exercisable now.
"""
from __future__ import annotations

import os
import time

try:
    from rest_framework.views import APIView
    from rest_framework.response import Response
    from rest_framework.permissions import AllowAny, IsAdminUser
    _HAVE_DRF = True
except ImportError:  # keep importable without DRF
    APIView = object
    Response = None
    AllowAny = None
    IsAdminUser = None
    _HAVE_DRF = False

from .inference_client import InferenceClient, InferenceUnavailable
from .models import QueryLog


class QueryView(APIView):
    """POST /api/v1/query  {query, source_id?, tenant?}."""

    permission_classes = [AllowAny] if _HAVE_DRF else []

    def post(self, request):
        data = request.data if hasattr(request, "data") else {}
        query = (data.get("query") or "").strip()
        if not query:
            return Response({"status": "invalid", "error": "query is required"}, status=400)

        # Server-side tenant resolution (§6.2). Prod: derive from request.user; dev default.
        tenant = self._resolve_tenant(request, data)
        # Default to the ready source so inference always receives a context (the
        # storage_adapters seam fails closed without one, §4.1). Env-overridable.
        source_id = data.get("source_id") or int(os.environ.get("VEDA_DEFAULT_SOURCE_ID", "1"))

        rid = getattr(request, "request_id", "")
        started = time.time()
        client = InferenceClient()
        try:
            payload = client.run_hybrid_query(query, source_id=source_id, tenant=tenant, request_id=rid)
        except InferenceUnavailable as exc:
            latency = int((time.time() - started) * 1000)
            self._audit(query, tenant, source_id, "exec_error", latency, refusal=str(exc), rid=rid)
            return Response({"status": "exec_error", "error": str(exc)}, status=503)

        latency = int((time.time() - started) * 1000)
        status_str = payload.get("status", "unknown")
        result = payload.get("result", {})
        items = result.get("items", []) if isinstance(result, dict) else []
        route = items[0].get("route") if items and isinstance(items[0], dict) else ""
        item0 = items[0] if items and isinstance(items[0], dict) else {}
        res0 = item0.get("result") or {}
        sql = res0.get("sql") if isinstance(res0, dict) else ""
        # Cache hit: the verified-query path tags the answer table "(cached)" (§6.6).
        cache_hit = isinstance(res0, dict) and res0.get("table") == "(cached)"
        self._audit(query, tenant, source_id, status_str, latency, route=route, sql=sql or "",
                    rid=rid, cache_hit=cache_hit)
        return Response({"status": status_str, "result": result, "latency_ms": latency,
                         "request_id": rid, "cache_hit": cache_hit})

    @staticmethod
    def _resolve_tenant(request, data) -> str:
        user = getattr(request, "user", None)
        if user is not None and getattr(user, "is_authenticated", False):
            return getattr(user, "username", "default") or "default"
        return (data.get("tenant") or "default")

    @staticmethod
    def _audit(query, tenant, source_id, status_str, latency, route="", sql="", refusal="",
               rid="", cache_hit=False):
        try:
            QueryLog.objects.create(
                source_id=source_id, tenant=tenant, query_text=query,
                route=route or "", status=status_str, executed_sql=sql or "",
                refusal_reason=refusal or "", latency_ms=latency, request_id=rid or "",
                cache_hit=cache_hit,
            )
        except Exception:  # audit must never break the response
            pass


class IngestTriggerView(APIView):
    """POST /api/v1/admin/ingest {source_id, tenant?, force?} — enqueue ingestion (§6.4).

    Staff-only. Returns immediately with the Celery task id; the job is tracked as an
    IngestionJob visible in admin.
    """

    permission_classes = [IsAdminUser] if _HAVE_DRF else []

    def post(self, request):
        data = request.data if hasattr(request, "data") else {}
        source_id = data.get("source_id")
        if not source_id:
            return Response({"error": "source_id required"}, status=400)
        from apps.ingestion.tasks import task_ingest_source
        res = task_ingest_source.delay(
            source_id=int(source_id), tenant=data.get("tenant", "default"),
            force=bool(data.get("force", False)),
        )
        return Response({"enqueued": True, "task_id": getattr(res, "id", None),
                         "source_id": int(source_id)}, status=202)


class EvalTriggerView(APIView):
    """POST /api/v1/admin/eval {source_id?, tenant?, label?} — enqueue an eval run (§6.4).
    Staff-only. Returns the Celery task id; results land in EvalRun (admin + API)."""

    permission_classes = [IsAdminUser] if _HAVE_DRF else []

    def post(self, request):
        data = request.data if hasattr(request, "data") else {}
        from apps.evaluation.tasks import task_run_eval
        res = task_run_eval.delay(
            source_id=int(data.get("source_id", 1)),
            tenant=data.get("tenant", "default"),
            label=data.get("label", ""),
        )
        return Response({"enqueued": True, "task_id": getattr(res, "id", None)}, status=202)
