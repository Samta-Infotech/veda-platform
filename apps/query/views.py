"""apps.query.views — QueryView (DRF) (migration_plan.md §5, §6.1, §6.2).

Thin view: validate → resolve tenant (server-side; dev falls back to "default")
→ call InferenceClient → persist QueryLog → return MultiResult with ``status``
preserved verbatim. A refusal or an unreachable inference tier is a structured
JSON payload with an appropriate code, never a leaked 500 (§9a, §18).

Auth/JWT + tenant-from-principal is the Phase 6.2 hardening; dev uses AllowAny and
a request-supplied/default tenant so the end-to-end path is exercisable now.
"""
from __future__ import annotations

import logging
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
from .scope import permitted_source_ids, resolve_query_scope

logger = logging.getLogger(__name__)

DEFAULT_TENANT = "default"

# The verified-query path tags the answer table "(cached)" (§6.6) — this sentinel
# is the cache-hit signal on the wire, not a real table name.
_CACHED_TABLE_SENTINEL = "(cached)"

_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

_STATUS_UNKNOWN = "unknown"
_STATUS_EXEC_ERROR = "exec_error"
_STATUS_FORBIDDEN = "forbidden"

# Gate 1 (User Story 3, Task 17): generic on purpose — never names a resource,
# table, column, or any internal RBAC detail (the brief's own explicit
# requirement). Same copy regardless of WHY nothing was permitted, so the
# response itself carries no signal an attacker could use to enumerate scope.
_FORBIDDEN_MESSAGE = "You do not have permission to access this resource."


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
        user = getattr(request, "user", None)

        # Gate 1 (User Story 3, Task 17): resolve RBAC permissions ONCE per request
        # — reused below by both the source-level check and the table/column
        # payload, never re-resolved layer by layer. Lazy import: this module must
        # stay importable without apps.access_management in INSTALLED_APPS for a
        # caller that never touches RBAC at all (e.g. this app's own minimal-app
        # test harness).
        from apps.access_management.services import (
            compute_data_scope, resolve_effective_permissions, serialize_data_scope,
        )
        effective = resolve_effective_permissions(user)

        # Authenticated + RBAC active but permitted NOTHING -> fail closed with a
        # generic 403 BEFORE any scope resolution or inference call, never a
        # leaked resource/table/column name. `permitted is None` means "no
        # narrowing at all" (RBAC off, or staff) — not this branch.
        permitted = permitted_source_ids(user, effective)
        if permitted is not None and not permitted:
            logger.warning("query denied: user_id=%s has no permitted sources",
                           getattr(user, "pk", None))
            return Response({"status": _STATUS_FORBIDDEN, "error": _FORBIDDEN_MESSAGE},
                            status=403)

        # Resolve the query SCOPE server-side (P5 / cross-source): a source SET, always
        # validated against the ready-source registry — an optional request subset is
        # intersected with ownership, never trusted verbatim (§6.2). `source_id` is the
        # primary (first) member, kept for the single-source execution/audit path.
        source_ids = resolve_query_scope(data, tenant, user=user, effective=effective)
        source_id = source_ids[0]
        # Gate 1 (User Story 3, Task 15): the table/column allow-payload for the
        # now-resolved scope, forwarded across the HTTP boundary alongside it —
        # None (RBAC off / staff) means the inference tier applies no narrowing,
        # exactly as before this change.
        data_scope = serialize_data_scope(compute_data_scope(user, source_ids, effective=effective))

        request_id = getattr(request, "request_id", "")
        started = time.time()
        client = InferenceClient()
        try:
            payload = client.run_hybrid_query(query, source_id=source_id, tenant=tenant,
                                              source_ids=source_ids, request_id=request_id,
                                              data_scope=data_scope)
        except InferenceUnavailable as exc:
            latency = int((time.time() - started) * 1000)
            logger.warning("inference unavailable request_id=%s tenant=%s source_id=%s: %s",
                           request_id, tenant, source_id, exc)
            self._audit(query, tenant, source_id, _STATUS_EXEC_ERROR, latency,
                        refusal=str(exc), rid=request_id)
            return Response({"status": _STATUS_EXEC_ERROR, "error": str(exc)}, status=503)

        latency = int((time.time() - started) * 1000)
        status_str = payload.get("status", _STATUS_UNKNOWN)
        result = payload.get("result", {})
        route, first_result = self._first_item_fields(result)
        sql = first_result.get("sql") or ""
        cache_hit = first_result.get("table") == _CACHED_TABLE_SENTINEL
        usage = first_result.get("usage")
        self._audit(query, tenant, source_id, status_str, latency, route=route, sql=sql,
                    rid=request_id, cache_hit=cache_hit, usage=usage)
        return Response({"status": status_str, "result": result, "latency_ms": latency,
                         "request_id": request_id, "cache_hit": cache_hit,
                         "usage": usage or dict(_ZERO_USAGE)})

    @staticmethod
    def _first_item_fields(result) -> tuple[str, dict]:
        """(route, result-dict) of the MultiResult's first item, defensively — the
        inference tier's payload shape is validated there, but this view must not
        500 on an unexpected/empty envelope."""
        items = result.get("items", []) if isinstance(result, dict) else []
        first_item = items[0] if items and isinstance(items[0], dict) else {}
        route = first_item.get("route") or ""
        first_result = first_item.get("result") or {}
        if not isinstance(first_result, dict):
            return route, {}
        return route, first_result

    @staticmethod
    def _resolve_scope(data, tenant) -> list:
        """Deprecated alias — see ``apps.query.scope.resolve_query_scope``.

        Kept so existing callers/tests referencing ``QueryView._resolve_scope``
        keep working unchanged; new code should import the function directly.
        """
        return resolve_query_scope(data, tenant)

    @staticmethod
    def _resolve_tenant(request, data) -> str:
        """Tenant from the authenticated principal when present (§6.2); dev falls
        back to a request-supplied or default tenant."""
        user = getattr(request, "user", None)
        if user is not None and getattr(user, "is_authenticated", False):
            return getattr(user, "username", DEFAULT_TENANT) or DEFAULT_TENANT
        return data.get("tenant") or DEFAULT_TENANT

    @staticmethod
    def _audit(query, tenant, source_id, status_str, latency, route="", sql="", refusal="",
               rid="", cache_hit=False, usage=None) -> None:
        """Append one QueryLog row (L9 audit, §6.6).

        Best-effort by design: an audit-write failure is logged with its traceback
        but never propagated, because losing an audit row must not turn a
        successfully answered query into a 500 for the caller. (Previously this
        swallowed the exception silently, so a broken audit table was invisible.)
        """
        usage = usage or {}
        try:
            QueryLog.objects.create(
                source_id=source_id, tenant=tenant, query_text=query,
                route=route or "", status=status_str, executed_sql=sql or "",
                refusal_reason=refusal or "", latency_ms=latency, request_id=rid or "",
                cache_hit=cache_hit,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )
        except Exception:  # noqa: BLE001 — audit must never break the response
            logger.exception("query audit write failed request_id=%s tenant=%s status=%s",
                             rid, tenant, status_str)


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
        task = task_ingest_source.delay(
            source_id=int(source_id), tenant=data.get("tenant", DEFAULT_TENANT),
            force=bool(data.get("force", False)),
        )
        logger.info("ingestion enqueued source_id=%s task_id=%s", source_id, getattr(task, "id", None))
        return Response({"enqueued": True, "task_id": getattr(task, "id", None),
                         "source_id": int(source_id)}, status=202)


class EvalTriggerView(APIView):
    """POST /api/v1/admin/eval {source_id?, tenant?, label?} — enqueue an eval run (§6.4).
    Staff-only. Returns the Celery task id; results land in EvalRun (admin + API)."""

    permission_classes = [IsAdminUser] if _HAVE_DRF else []

    def post(self, request):
        data = request.data if hasattr(request, "data") else {}
        from apps.evaluation.tasks import task_run_eval
        task = task_run_eval.delay(
            source_id=int(data.get("source_id", 1)),
            tenant=data.get("tenant", DEFAULT_TENANT),
            label=data.get("label", ""),
        )
        logger.info("eval run enqueued task_id=%s", getattr(task, "id", None))
        return Response({"enqueued": True, "task_id": getattr(task, "id", None)}, status=202)
