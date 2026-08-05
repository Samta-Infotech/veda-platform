"""apps.core.views — health + Prometheus metrics (migration_plan.md §5, §6.3).

/metrics emits Prometheus text-format counters derived from QueryLog (per-status
counts, avg latency) + a PgBouncer-pooled DB-reachability gauge — dependency-free
(no prometheus_client), so the thin api image stays lean. /readyz checks Postgres
(via PgBouncer) + both Redis instances + inference reachability.

Both endpoints are deliberately fail-soft: a probe that cannot run degrades to a
reported failure (readyz) or an omitted metric family (metrics), never a 500 —
an observability endpoint that crashes takes the monitoring down with it.
"""
from __future__ import annotations

import logging
import os
import urllib.request
from contextlib import closing

from django.db.models import Avg, Count
from django.http import HttpResponse, JsonResponse

logger = logging.getLogger(__name__)

# Prometheus text exposition format version served in the Content-Type (fixed by
# the format spec, not a deployment setting — hence a constant, not an env var).
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4"

# QueryLog.status values that count as "the query produced an answer"; every other
# terminal status counts toward the refusal rate.
_ANSWERED_STATUSES = ("ok", "answered")

_UNKNOWN_ROUTE_LABEL = "unknown"

# Readiness probe timeout for the HTTP-reachability checks (inference, SLM
# backend). Short on purpose: /readyz is polled by the orchestrator and must
# answer quickly even when a dependency is hanging.
_PROBE_TIMEOUT_S = 3

# PgBouncer admin console connection (SHOW POOLS). All overridable per deployment.
_ENV_PGBOUNCER_HOST = "PGBOUNCER_HOST"
_ENV_PGBOUNCER_PORT = "PGBOUNCER_PORT"
_ENV_PGBOUNCER_USER = "PGBOUNCER_ADMIN_USER"
_ENV_PGBOUNCER_PASSWORD = "POSTGRES_PASSWORD"
_DEFAULT_PGBOUNCER_HOST = "pgbouncer"
_DEFAULT_PGBOUNCER_PORT = "6432"
_DEFAULT_PGBOUNCER_USER = "pgbouncer_admin"
_DEFAULT_PGBOUNCER_PASSWORD = "change-me"
_PGBOUNCER_ADMIN_DBNAME = "pgbouncer"

_ENV_INFERENCE_URL = "INFERENCE_URL"
_DEFAULT_INFERENCE_URL = "http://inference:8001"
_ENV_SLM_BACKEND = "SLM_BACKEND"
_DEFAULT_SLM_BACKEND = "ollama"
_ENV_VLLM_URL = "VLLM_URL"
_DEFAULT_VLLM_URL = "http://vllm:8000"
_ENV_OLLAMA_URL = "OLLAMA_URL"
_DEFAULT_OLLAMA_URL = "http://ollama:11434"

_REDIS_CHECKS = (
    ("redis_cache", "REDIS_CACHE_URL", "redis://redis-cache:6379/0"),
    ("redis_broker", "REDIS_BROKER_URL", "redis://redis-broker:6379/0"),
)


# ─────────────────────────────────────────────────────────────────────────────
# /metrics
# ─────────────────────────────────────────────────────────────────────────────
def metrics(request) -> HttpResponse:
    """Prometheus text metrics (§6.3): per-status + per-route latency, refusal-rate,
    cache hit/miss, PgBouncer connections in use. Dependency-free."""
    from apps.query.models import QueryLog

    metric_lines: list[str] = []
    total, answered = _append_status_counters(metric_lines, QueryLog)
    _append_refusal_rate(metric_lines, total, answered)
    _append_route_latency(metric_lines, QueryLog)
    _append_cache_counters(metric_lines, QueryLog, total)
    _append_overall_latency(metric_lines, QueryLog)
    _append_pgbouncer_gauge(metric_lines)

    return HttpResponse("\n".join(metric_lines) + "\n", content_type=PROMETHEUS_CONTENT_TYPE)


def _append_status_counters(lines: list[str], query_log_model) -> tuple[int, int]:
    """Emit one counter per terminal status; returns (total, answered) for reuse
    by the refusal-rate and cache-miss families (computed once, not re-queried)."""
    lines += ["# HELP veda_queries_total Total queries by terminal status.",
              "# TYPE veda_queries_total counter"]
    total = 0
    answered = 0
    for row in query_log_model.objects.values("status").annotate(n=Count("id")):
        lines.append(f'veda_queries_total{{status="{row["status"]}"}} {row["n"]}')
        total += row["n"]
        if row["status"] in _ANSWERED_STATUSES:
            answered += row["n"]
    lines += ["# HELP veda_queries_all Total queries.", "# TYPE veda_queries_all counter",
              f"veda_queries_all {total}"]
    return total, answered


def _append_refusal_rate(lines: list[str], total: int, answered: int) -> None:
    refusal_rate = (total - answered) / total if total else 0.0
    lines += ["# HELP veda_refusal_rate Fraction of queries that did not answer.",
              "# TYPE veda_refusal_rate gauge", f"veda_refusal_rate {refusal_rate:.4f}"]


def _append_route_latency(lines: list[str], query_log_model) -> None:
    lines += ["# HELP veda_route_latency_ms_avg Average latency by route.",
              "# TYPE veda_route_latency_ms_avg gauge"]
    for row in query_log_model.objects.values("route").annotate(a=Avg("latency_ms"), n=Count("id")):
        route = row["route"] or _UNKNOWN_ROUTE_LABEL
        lines.append(f'veda_route_latency_ms_avg{{route="{route}"}} {(row["a"] or 0):.1f}')
        lines.append(f'veda_route_queries_total{{route="{route}"}} {row["n"]}')


def _append_cache_counters(lines: list[str], query_log_model, total: int) -> None:
    hits = query_log_model.objects.filter(cache_hit=True).count()
    lines += ["# HELP veda_cache_hits_total Verified-cache hits.",
              "# TYPE veda_cache_hits_total counter", f"veda_cache_hits_total {hits}",
              "# HELP veda_cache_misses_total Verified-cache misses.",
              "# TYPE veda_cache_misses_total counter",
              f"veda_cache_misses_total {max(total - hits, 0)}"]


def _append_overall_latency(lines: list[str], query_log_model) -> None:
    average_latency_ms = query_log_model.objects.aggregate(a=Avg("latency_ms"))["a"] or 0
    lines += ["# HELP veda_query_latency_ms_avg Average query latency (ms).",
              "# TYPE veda_query_latency_ms_avg gauge",
              f"veda_query_latency_ms_avg {average_latency_ms:.1f}"]


def _append_pgbouncer_gauge(lines: list[str]) -> None:
    """The §3 connection ceiling. An unreachable PgBouncer emits the family
    header with no samples rather than failing the whole scrape."""
    lines += ["# HELP veda_pgbouncer_sv_active Server connections active per pool.",
              "# TYPE veda_pgbouncer_sv_active gauge"]
    for database, active in _pgbouncer_pools():
        lines.append(f'veda_pgbouncer_sv_active{{database="{database}"}} {active}')


def _pgbouncer_pools() -> list[tuple[str, int]]:
    """(database, sv_active) from PgBouncer SHOW POOLS — bounded connection visibility.

    ``psycopg2`` is imported lazily so the thin api image stays importable without
    it. Returns an empty list (and logs) when PgBouncer is unreachable or its admin
    console rejects the credentials — metrics must never fail the scrape.

    The connection is closed via ``closing()`` on EVERY path: the previous version
    only closed it on the success path, so a failing ``SHOW POOLS`` (permission
    denied, admin console disabled) leaked a server connection per scrape — against
    the very pool ceiling this gauge exists to watch.
    """
    try:
        import psycopg2
    except ImportError:
        logger.debug("pgbouncer metrics skipped: psycopg2 not installed in this image")
        return []

    try:
        connection = psycopg2.connect(
            host=os.environ.get(_ENV_PGBOUNCER_HOST, _DEFAULT_PGBOUNCER_HOST),
            port=int(os.environ.get(_ENV_PGBOUNCER_PORT, _DEFAULT_PGBOUNCER_PORT)),
            dbname=_PGBOUNCER_ADMIN_DBNAME,
            user=os.environ.get(_ENV_PGBOUNCER_USER, _DEFAULT_PGBOUNCER_USER),
            password=os.environ.get(_ENV_PGBOUNCER_PASSWORD, _DEFAULT_PGBOUNCER_PASSWORD),
        )
    except Exception:  # noqa: BLE001 — any driver/network failure degrades to "no samples"
        logger.warning("pgbouncer metrics unavailable: connect failed", exc_info=True)
        return []

    try:
        with closing(connection):
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute("SHOW POOLS")
                columns = [d[0] for d in cursor.description]
                database_idx, active_idx = columns.index("database"), columns.index("sv_active")
                return [(row[database_idx], row[active_idx]) for row in cursor.fetchall()]
    except Exception:  # noqa: BLE001 — see above
        logger.warning("pgbouncer metrics unavailable: SHOW POOLS failed", exc_info=True)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# /readyz
# ─────────────────────────────────────────────────────────────────────────────
def readyz(request) -> JsonResponse:
    """Readiness probe: Postgres (via PgBouncer) + both Redis instances +
    inference reachability gate readiness; the SLM backend is reported but
    non-gating (dev parity). 200 when every gating check passes, else 503."""
    checks: dict[str, str] = {}
    gating_failures = 0

    gating_failures += _record_check(checks, "postgres", _check_postgres)
    for name, env_key, default_url in _REDIS_CHECKS:
        gating_failures += _record_check(checks, name, lambda e=env_key, d=default_url: _check_redis(e, d))
    gating_failures += _record_check(checks, "inference", _check_inference)
    _record_slm_backend(checks)

    ok = gating_failures == 0
    return JsonResponse({"status": "ready" if ok else "degraded", "checks": checks},
                        status=200 if ok else 503)


def _record_check(checks: dict[str, str], name: str, probe) -> int:
    """Run one gating probe, record its reported detail, and return 1 when it failed.

    ``probe`` returns ``(detail, passed)`` so a reachable-but-unhealthy dependency
    can report its own wording (e.g. ``"status 503"``) while still gating
    readiness — distinct from an outright exception, which reports ``"fail: ..."``.

    Failure detail is echoed into the payload (unchanged behaviour) because
    /readyz is an operator-facing endpoint; see the security note in the review
    if it is ever exposed beyond the cluster.
    """
    try:
        detail, passed = probe()
    except Exception as exc:  # noqa: BLE001 — a probe failure IS the signal, never a 500
        logger.warning("readyz check failed name=%s", name, exc_info=True)
        checks[name] = f"fail: {exc}"
        return 1
    checks[name] = detail
    return 0 if passed else 1


def _check_postgres() -> tuple[str, bool]:
    from django.db import connection

    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
    return "ok", True


def _check_redis(env_key: str, default_url: str) -> tuple[str, bool]:
    import redis  # lazy: the thin image may not carry the client

    redis.Redis.from_url(os.environ.get(env_key, default_url)).ping()
    return "ok", True


def _check_inference() -> tuple[str, bool]:
    """The api never loads models; it calls the inference service."""
    url = _base_url(_ENV_INFERENCE_URL, _DEFAULT_INFERENCE_URL) + "/readyz"
    with urllib.request.urlopen(url, timeout=_PROBE_TIMEOUT_S) as response:
        if response.status != 200:
            return f"status {response.status}", False
        return "ok", True


def _record_slm_backend(checks: dict[str, str]) -> None:
    """Probe the CONFIGURED SLM backend (§10 seam): Ollama → /api/tags, vLLM →
    /v1/models (OpenAI-compatible). Non-gating (dev), reported only — so a
    missing local model server never marks the api unready.

    ``slm_backend_kind`` is recorded only alongside a successful probe, matching
    the endpoint's existing payload shape.
    """
    try:
        backend = os.environ.get(_ENV_SLM_BACKEND, _DEFAULT_SLM_BACKEND).strip().lower()
        url = _slm_probe_url(backend)
        with urllib.request.urlopen(url, timeout=_PROBE_TIMEOUT_S) as response:
            checks["slm_backend"] = "ok" if response.status == 200 else f"status {response.status}"
        checks["slm_backend_kind"] = backend
    except Exception as exc:  # noqa: BLE001 — non-gating by design
        logger.info("readyz slm_backend probe failed", exc_info=True)
        checks["slm_backend"] = f"fail: {exc}"


def _slm_probe_url(backend: str) -> str:
    if backend == "vllm":
        base = _base_url(_ENV_VLLM_URL, _DEFAULT_VLLM_URL)
        if base.endswith("/v1"):
            base = base[: -len("/v1")].rstrip("/")
        return base + "/v1/models"
    return _base_url(_ENV_OLLAMA_URL, _DEFAULT_OLLAMA_URL) + "/api/tags"


def _base_url(env_key: str, default: str) -> str:
    return os.environ.get(env_key, default).rstrip("/")
