"""apps.core.views — health + Prometheus metrics (migration_plan.md §5, §6.3).

/metrics emits Prometheus text-format counters derived from QueryLog (per-status
counts, avg latency) + a PgBouncer-pooled DB-reachability gauge — dependency-free
(no prometheus_client), so the thin api image stays lean. /readyz checks Postgres
(via PgBouncer) + both Redis instances + inference reachability.
"""
from __future__ import annotations

from django.db.models import Avg, Count
from django.http import HttpResponse, JsonResponse


def metrics(request):
    """Prometheus text metrics (§6.3): per-status + per-route latency, refusal-rate,
    cache hit/miss, PgBouncer connections in use. Dependency-free."""
    from apps.query.models import QueryLog

    L = []

    # queries by terminal status
    L += ["# HELP veda_queries_total Total queries by terminal status.",
          "# TYPE veda_queries_total counter"]
    total = 0
    answered = 0
    for row in QueryLog.objects.values("status").annotate(n=Count("id")):
        L.append(f'veda_queries_total{{status="{row["status"]}"}} {row["n"]}')
        total += row["n"]
        if row["status"] in ("ok", "answered"):
            answered += row["n"]
    L += ["# HELP veda_queries_all Total queries.", "# TYPE veda_queries_all counter",
          f"veda_queries_all {total}"]

    # refusal-rate (non-answered / total)
    refusal_rate = (total - answered) / total if total else 0.0
    L += ["# HELP veda_refusal_rate Fraction of queries that did not answer.",
          "# TYPE veda_refusal_rate gauge", f"veda_refusal_rate {refusal_rate:.4f}"]

    # per-route latency (avg ms) + count
    L += ["# HELP veda_route_latency_ms_avg Average latency by route.",
          "# TYPE veda_route_latency_ms_avg gauge"]
    for row in QueryLog.objects.values("route").annotate(a=Avg("latency_ms"), n=Count("id")):
        route = row["route"] or "unknown"
        L.append(f'veda_route_latency_ms_avg{{route="{route}"}} {(row["a"] or 0):.1f}')
        L.append(f'veda_route_queries_total{{route="{route}"}} {row["n"]}')

    # cache hit/miss
    hits = QueryLog.objects.filter(cache_hit=True).count()
    L += ["# HELP veda_cache_hits_total Verified-cache hits.",
          "# TYPE veda_cache_hits_total counter", f"veda_cache_hits_total {hits}",
          "# HELP veda_cache_misses_total Verified-cache misses.",
          "# TYPE veda_cache_misses_total counter", f"veda_cache_misses_total {max(total - hits, 0)}"]

    # overall avg latency
    L += ["# HELP veda_query_latency_ms_avg Average query latency (ms).",
          "# TYPE veda_query_latency_ms_avg gauge",
          f"veda_query_latency_ms_avg {(QueryLog.objects.aggregate(a=Avg('latency_ms'))['a'] or 0):.1f}"]

    # PgBouncer server connections in use (the §3 connection ceiling)
    L += ["# HELP veda_pgbouncer_sv_active Server connections active per pool.",
          "# TYPE veda_pgbouncer_sv_active gauge"]
    for db, active in _pgbouncer_pools():
        L.append(f'veda_pgbouncer_sv_active{{database="{db}"}} {active}')

    return HttpResponse("\n".join(L) + "\n", content_type="text/plain; version=0.0.4")


def _pgbouncer_pools():
    """(database, sv_active) from PgBouncer SHOW POOLS — bounded connection visibility."""
    import os
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.environ.get("PGBOUNCER_HOST", "pgbouncer"),
            port=int(os.environ.get("PGBOUNCER_PORT", "6432")),
            dbname="pgbouncer", user="pgbouncer_admin",
            password=os.environ.get("POSTGRES_PASSWORD", "change-me"),
        )
        conn.autocommit = True
        out = []
        with conn.cursor() as cur:
            cur.execute("SHOW POOLS")
            cols = [d[0] for d in cur.description]
            di, ai = cols.index("database"), cols.index("sv_active")
            for r in cur.fetchall():
                out.append((r[di], r[ai]))
        conn.close()
        return out
    except Exception:
        return []


def readyz(request):
    checks = {}
    ok = True
    # Postgres via PgBouncer
    try:
        from django.db import connection
        with connection.cursor() as cur:
            cur.execute("SELECT 1")
        checks["postgres"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["postgres"] = f"fail: {exc}"; ok = False
    # Redis cache + broker
    import os
    import urllib.request
    for name, env, default in [("redis_cache", "REDIS_CACHE_URL", "redis://redis-cache:6379/0"),
                               ("redis_broker", "REDIS_BROKER_URL", "redis://redis-broker:6379/0")]:
        try:
            import redis
            redis.Redis.from_url(os.environ.get(env, default)).ping()
            checks[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            checks[name] = f"fail: {exc}"; ok = False
    # Inference service reachability (the api never loads models; it calls inference).
    try:
        url = os.environ.get("INFERENCE_URL", "http://inference:8001").rstrip("/") + "/readyz"
        with urllib.request.urlopen(url, timeout=3) as r:
            checks["inference"] = "ok" if r.status == 200 else f"status {r.status}"
            if r.status != 200:
                ok = False
    except Exception as exc:  # noqa: BLE001
        checks["inference"] = f"fail: {exc}"; ok = False
    # SLM backend reachability (Ollama dev / vLLM prod).
    try:
        slm = os.environ.get("OLLAMA_URL", "http://ollama:11434").rstrip("/") + "/api/tags"
        with urllib.request.urlopen(slm, timeout=3) as r:
            checks["slm_backend"] = "ok" if r.status == 200 else f"status {r.status}"
    except Exception as exc:  # noqa: BLE001
        checks["slm_backend"] = f"fail: {exc}"  # non-gating (dev), reported
    return JsonResponse({"status": "ready" if ok else "degraded", "checks": checks},
                        status=200 if ok else 503)
