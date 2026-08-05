"""apps.evaluation.tasks — evaluation as a tracked Celery run (migration_plan.md §5, §6.4).

Runs a query set through the inference service (HTTP, the same path the api uses) and
records an EvalRun + per-query EvalCaseResult, plus a small HTML report artifact. Staff
triggers it via POST /api/v1/admin/eval; status is visible in admin and via the API.
"""
from __future__ import annotations

import logging
import time

from django.utils.html import escape

try:
    from celery import shared_task
except ImportError:
    def shared_task(*d_args, **d_kwargs):
        def _wrap(fn):
            return fn
        return _wrap


logger = logging.getLogger(__name__)

# A small default flow-eval set (deterministic count queries answer fast once cached).
DEFAULT_QUERIES = [
    ("D01", "DIRECT", "how many users are there"),
    ("D02", "DIRECT", "count annotations"),
    ("A01", "AGGREGATE", "how many change requests are there"),
    ("S01", "SYNONYM", "number of people"),
]

DEFAULT_RUN_LABEL = "adhoc-eval"

# The inference status that counts as a successful case (§6.4 scoring).
_SUCCESS_STATUS = "ok"
_UNKNOWN_STATUS = "unknown"
_ERROR_STATUS = "exec_error"

# An unavailable-inference message is recorded in the case's `sql` slot for
# diagnosis; bounded so a long upstream error can't bloat the JSON details blob.
_ERROR_DETAIL_MAX_CHARS = 200

_SUCCESS_RATE_DECIMALS = 4


@shared_task(queue="default")
def task_run_eval(source_id=1, tenant="default", label="", queries=None):
    """Run the query set through inference, store EvalRun + EvalCaseResult (§6.4)."""
    from apps.evaluation.models import EvalCaseResult, EvalRun
    from apps.query.inference_client import InferenceClient

    query_set = queries or DEFAULT_QUERIES
    run = EvalRun.objects.create(
        source_id=source_id, tenant=tenant, label=label or DEFAULT_RUN_LABEL)
    client = InferenceClient()
    logger.info("eval run started run_id=%s source_id=%s tenant=%s cases=%s",
                run.pk, source_id, tenant, len(query_set))

    success_count = 0
    report_rows: list[str] = []
    for query_id, query_type, query_text in query_set:
        status, sql, latency_ms = _run_one_case(client, query_text, source_id, tenant)
        is_success = status == _SUCCESS_STATUS
        success_count += 1 if is_success else 0
        EvalCaseResult.objects.create(
            run=run, query_id=query_id, query_type=query_type, difficulty="",
            status=status, hit=is_success,
            details={"sql": sql, "latency_ms": latency_ms, "query": query_text},
        )
        report_rows.append(
            _report_row(query_id, query_type, query_text, status, latency_ms, sql))

    total = len(query_set)
    run.sql_success_rate = (
        round(success_count / total, _SUCCESS_RATE_DECIMALS) if total else 0.0)
    run.report_html = _build_report_html(run, source_id, tenant, success_count, total, report_rows)
    run.save(update_fields=["sql_success_rate", "report_html"])
    logger.info("eval run finished run_id=%s success=%s/%s rate=%s",
                run.pk, success_count, total, run.sql_success_rate)
    return {"eval_run_id": run.pk, "success_rate": run.sql_success_rate,
            "n": total, "ok": success_count}


def _run_one_case(client, query_text: str, source_id, tenant) -> tuple[str, str, int]:
    """Execute one eval query against the inference tier.

    Returns ``(status, sql, latency_ms)``. An unreachable inference tier is NOT a
    task failure — it is recorded as an ``exec_error`` case (with the upstream
    detail in the ``sql`` slot, matching the existing report/``details`` shape) so
    the run still produces a complete, comparable scorecard.
    """
    from apps.query.inference_client import InferenceUnavailable

    started = time.time()
    status, sql = _ERROR_STATUS, ""
    try:
        payload = client.run_hybrid_query(query_text, source_id=source_id, tenant=tenant)
        status = payload.get("status", _UNKNOWN_STATUS)
        items = (payload.get("result") or {}).get("items", [])
        if items:
            sql = (items[0].get("result") or {}).get("sql") or ""
    except InferenceUnavailable as exc:
        logger.warning("eval case failed: inference unavailable query=%r", query_text, exc_info=True)
        sql = str(exc)[:_ERROR_DETAIL_MAX_CHARS]
    return status, sql, int((time.time() - started) * 1000)


def _report_row(query_id: str, query_type: str, query_text: str, status: str,
                latency_ms: int, sql: str) -> str:
    """One ``<tr>`` of the stored HTML report.

    EVERY interpolated value is HTML-escaped. The report is persisted to
    ``EvalRun.report_html`` and rendered back in the Django admin, so an
    unescaped query string or SQL fragment containing markup was a stored-XSS
    vector against staff users — the query text is caller-supplied (the
    ``queries`` task argument / ``POST /api/v1/admin/eval``) and the SQL is
    generated upstream from it.
    """
    return (f"<tr><td>{escape(query_id)}</td><td>{escape(query_type)}</td>"
            f"<td>{escape(query_text)}</td><td>{escape(status)}</td>"
            f"<td>{escape(str(latency_ms))}</td><td><code>{escape(sql)}</code></td></tr>")


def _build_report_html(run, source_id, tenant, success_count: int, total: int,
                       report_rows: list[str]) -> str:
    """Assemble the run's report artifact. ``report_rows`` are pre-escaped by
    ``_report_row``; the header values are escaped here for the same reason."""
    header = (f"label={escape(str(run.label))} · source={escape(str(source_id))} · "
              f"tenant={escape(str(tenant))} · "
              f"success={success_count}/{total} ({run.sql_success_rate:.0%})")
    return ("<h2>VEDA Eval Run</h2>"
            f"<p>{header}</p>"
            "<table border=1 cellpadding=4><tr><th>id</th><th>type</th><th>query</th>"
            "<th>status</th><th>ms</th><th>sql</th></tr>" + "".join(report_rows) + "</table>")
