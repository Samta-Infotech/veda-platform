"""Golden-query evaluation harness — regression tracking for the query engine.

Runs a golden set (evaluation/golden_*.jsonl: {query, gold_tables, gold_columns?})
through the engine, scores routing/answer correctness + latency + tokens, and logs
ONE aggregate run per evaluation to a SEPARATE MLflow experiment
(VEDA_MLFLOW_EVAL_EXPERIMENT, default "VEDA-Golden-Eval") so accuracy/latency can be
TRENDED across code/model versions and gated in CI.

Decoupled + testable by design: the query runner is INJECTED (`run_query_fn`), so the
scoring/aggregation logic is fully unit-testable model-free (no SLM/engine). The live
runner (`inference_run_query_fn`) posts to the inference tier over HTTP.

The golden files carry expected TABLES (gold_tables) — the correctness signal is
"did the engine answer AND use the expected table(s)", not answer-value equality
(the golden set has no expected rows/answers). gold_tables=[] → correctness is just
"answered" (nothing to check tables against).
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# query runner contract: query:str -> dict with (at least) keys
#   status, table, sql, usage{prompt/completion/total_tokens}, latency_ms
RunQueryFn = Callable[[str], Dict[str, Any]]

_TABLE_RE = re.compile(r'(?:FROM|JOIN)\s+"?([A-Za-z_][A-Za-z0-9_]*)"?', re.IGNORECASE)


def load_golden(path: str) -> List[dict]:
    """Read a golden JSONL file; skip blank/comment lines. Never raises on a bad
    line — a malformed row is skipped (best-effort), the rest still evaluate."""
    cases: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("query"):
                cases.append(obj)
    return cases


def tables_used(engine_result: dict) -> set:
    """Tables the executed answer actually touched — from the SQL's FROM/JOIN
    clauses, plus the primary `table` the engine reported. Lower-cased for a
    case-insensitive match against gold_tables."""
    used = set()
    sql = engine_result.get("sql") or ""
    for m in _TABLE_RE.findall(sql):
        used.add(m.lower())
    t = engine_result.get("table")
    if t:
        used.add(str(t).lower())
    return used


@dataclass
class CaseResult:
    id: str
    query: str
    status: str
    answered: bool
    gold_tables: List[str]
    used_tables: List[str]
    table_hit: bool          # gold_tables ⊆ used_tables (True when no gold_tables)
    passed: bool             # answered AND table_hit
    latency_ms: Optional[float]
    total_tokens: Optional[float]


def score_case(case: dict, engine_result: dict) -> CaseResult:
    status = str(engine_result.get("status") or "")
    answered = status == "answered"
    gold = [str(t).lower() for t in (case.get("gold_tables") or [])]
    used = tables_used(engine_result)
    table_hit = (not gold) or set(gold).issubset(used)
    usage = engine_result.get("usage") or {}
    _tok = usage.get("total_tokens")
    return CaseResult(
        id=str(case.get("id") or case.get("query", "")[:40]),
        query=case.get("query", ""), status=status, answered=answered,
        gold_tables=gold, used_tables=sorted(used), table_hit=table_hit,
        passed=answered and table_hit,
        latency_ms=_num(engine_result.get("latency_ms")),
        total_tokens=_num(_tok),
    )


def _num(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pct(numer: int, denom: int) -> float:
    return round(numer / denom, 4) if denom else 0.0


def _percentile(values: List[float], q: float) -> Optional[float]:
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    # nearest-rank; small-sample safe
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return round(xs[k], 2)


@dataclass
class EvalReport:
    n_cases: int
    metrics: Dict[str, float] = field(default_factory=dict)
    cases: List[CaseResult] = field(default_factory=list)


def aggregate(results: List[CaseResult]) -> EvalReport:
    n = len(results)
    answered = [r for r in results if r.answered]
    with_gold = [r for r in results if r.gold_tables]
    lat = [r.latency_ms for r in results if r.latency_ms is not None]
    tok = [r.total_tokens for r in results if r.total_tokens is not None]
    metrics = {
        "n_cases": float(n),
        "pass_rate": _pct(sum(1 for r in results if r.passed), n),
        "answered_rate": _pct(len(answered), n),
        "refused_rate": _pct(n - len(answered), n),
        # table-routing accuracy over ONLY the cases that declared gold_tables
        "table_hit_rate": _pct(sum(1 for r in with_gold if r.table_hit), len(with_gold)),
        "n_cases_with_gold_tables": float(len(with_gold)),
    }
    if lat:
        metrics["latency_ms_mean"] = round(statistics.fmean(lat), 2)
        metrics["latency_ms_p50"] = _percentile(lat, 0.50)
        metrics["latency_ms_p95"] = _percentile(lat, 0.95)
    if tok:
        metrics["tokens_mean"] = round(statistics.fmean(tok), 2)
        metrics["tokens_total"] = round(sum(tok), 2)
    return EvalReport(n_cases=n, metrics=metrics, cases=results)


def evaluate_golden(cases: List[dict], run_query_fn: RunQueryFn) -> EvalReport:
    """Run every case through `run_query_fn`, score, aggregate. A per-case failure
    (runner raised) is recorded as a non-answered result — one bad query never
    aborts the whole evaluation."""
    results: List[CaseResult] = []
    for case in cases:
        try:
            er = run_query_fn(case["query"]) or {}
        except Exception as e:
            er = {"status": f"harness_error:{type(e).__name__}"}
        results.append(score_case(case, er))
    return aggregate(results)


# ── MLflow logging (one aggregate run per evaluation) ───────────────────────────
def log_report(report: EvalReport, *, tracking_uri: str, experiment: str,
               golden_file: str = "", environment: str = "local",
               git_sha: Optional[str] = None) -> Optional[str]:
    """Log the aggregate metrics + a per-case artifact to the eval experiment.
    Returns the run_id (None if mlflow is unavailable — never raises)."""
    try:
        import mlflow
    except Exception:
        return None
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"golden-eval:{golden_file.split('/')[-1] or 'set'}") as run:
        mlflow.log_metrics({k: v for k, v in report.metrics.items() if v is not None})
        params = {"golden_file": golden_file, "n_cases": report.n_cases,
                  "environment": environment}
        if git_sha:
            params["git_sha"] = git_sha
        mlflow.log_params(params)
        tags = {"veda.eval": "golden", "veda.environment": environment}
        if git_sha:
            tags["veda.git_sha"] = git_sha
        mlflow.set_tags(tags)
        mlflow.log_text(json.dumps(
            [{"id": c.id, "query": c.query, "status": c.status, "passed": c.passed,
              "table_hit": c.table_hit, "gold_tables": c.gold_tables,
              "used_tables": c.used_tables, "latency_ms": c.latency_ms,
              "total_tokens": c.total_tokens} for c in report.cases],
            indent=2), "eval/cases.json")
        return run.info.run_id


# ── live runner (SLM-dependent; posts to the inference tier) ────────────────────
def inference_run_query_fn(inference_url: str, source_id: Optional[int] = None,
                           tenant: str = "default", timeout: float = 180.0) -> RunQueryFn:
    """Build a runner that POSTs each query to the inference tier and normalizes the
    MultiResult wire shape ({result:{items:[{result:{...}}]}}) to the flat
    {status, table, sql, usage, latency_ms} the scorer expects."""
    import urllib.request

    def _run(query: str) -> Dict[str, Any]:
        body = json.dumps({"query": query, "source_id": source_id,
                           "tenant": tenant}).encode("utf-8")
        req = urllib.request.Request(
            inference_url.rstrip("/") + "/v1/run_hybrid_query",
            data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        result = (payload or {}).get("result") or {}
        items = result.get("items") or []
        res0 = (items[0].get("result") if items and isinstance(items[0], dict) else {}) or {}
        return {
            "status": res0.get("status") or payload.get("status"),
            "table": res0.get("table"),
            "sql": res0.get("sql"),
            "usage": res0.get("usage") or {},
            "latency_ms": res0.get("latency_ms"),
        }

    return _run
