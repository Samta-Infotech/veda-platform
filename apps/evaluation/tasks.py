"""apps.evaluation.tasks — evaluation as a tracked Celery run (migration_plan.md §5, §6.4).

Runs a query set through the inference service (HTTP, the same path the api uses) and
records an EvalRun + per-query EvalCaseResult, plus a small HTML report artifact. Staff
triggers it via POST /api/v1/admin/eval; status is visible in admin and via the API.
"""
from __future__ import annotations

import time

try:
    from celery import shared_task
except ImportError:
    def shared_task(*d_args, **d_kwargs):
        def _wrap(fn):
            return fn
        return _wrap


# A small default flow-eval set (deterministic count queries answer fast once cached).
DEFAULT_QUERIES = [
    ("D01", "DIRECT", "how many users are there"),
    ("D02", "DIRECT", "count annotations"),
    ("A01", "AGGREGATE", "how many change requests are there"),
    ("S01", "SYNONYM", "number of people"),
]


@shared_task(queue="default")
def task_run_eval(source_id=1, tenant="default", label="", queries=None):
    """Run the query set through inference, store EvalRun + EvalCaseResult (§6.4)."""
    from apps.evaluation.models import EvalCaseResult, EvalRun
    from apps.query.inference_client import InferenceClient, InferenceUnavailable

    qset = queries or DEFAULT_QUERIES
    run = EvalRun.objects.create(source_id=source_id, tenant=tenant, label=label or "adhoc-eval")
    client = InferenceClient()

    n_ok = 0
    rows_html = []
    for qid, qtype, q in qset:
        t0 = time.time()
        status, sql = "exec_error", ""
        try:
            payload = client.run_hybrid_query(q, source_id=source_id, tenant=tenant)
            status = payload.get("status", "unknown")
            items = (payload.get("result") or {}).get("items", [])
            if items:
                sql = (items[0].get("result") or {}).get("sql") or ""
        except InferenceUnavailable as exc:
            sql = str(exc)[:200]
        latency = int((time.time() - t0) * 1000)
        ok = status == "ok"
        n_ok += 1 if ok else 0
        EvalCaseResult.objects.create(
            run=run, query_id=qid, query_type=qtype, difficulty="",
            status=status, hit=ok, details={"sql": sql, "latency_ms": latency, "query": q},
        )
        rows_html.append(
            f"<tr><td>{qid}</td><td>{qtype}</td><td>{q}</td>"
            f"<td>{status}</td><td>{latency}</td><td><code>{sql}</code></td></tr>")

    total = len(qset)
    run.sql_success_rate = round(n_ok / total, 4) if total else 0.0
    run.report_html = (
        "<h2>VEDA Eval Run</h2>"
        f"<p>label={run.label} · source={source_id} · tenant={tenant} · "
        f"success={n_ok}/{total} ({run.sql_success_rate:.0%})</p>"
        "<table border=1 cellpadding=4><tr><th>id</th><th>type</th><th>query</th>"
        "<th>status</th><th>ms</th><th>sql</th></tr>" + "".join(rows_html) + "</table>")
    run.save(update_fields=["sql_success_rate", "report_html"])
    return {"eval_run_id": run.pk, "success_rate": run.sql_success_rate, "n": total, "ok": n_ok}
