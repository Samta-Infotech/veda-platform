"""Trace record → MLflow run spec (params / metrics / tags / artifacts).

Input is one JSON line written by ExplainTrace.finish()
(veda_core/veda/explain.py):

    {"route", "intent", "table", "anchor", "anchor_conf", "join_conf",
     "action", "status", "confidence", "refusal", "total_ms",
     "full": {"query", "total_ms", "verbose", "sections": {...}} | null}

Mapping strategy (mlflow_impl.md):
  * explicit spec-named metrics/params where a clean source exists
    (total_latency_ms, routing_confidence, repair_count, …);
  * a generic per-section sweep so NO datapoint the pipeline recorded is lost:
    numeric scalar -> metric "<section>.<key>", bool -> 0/1 metric,
    short string -> param "<section>.<key>", list/dict -> section artifact;
  * per-layer artifacts under the spec's file names (query_understanding.json,
    retrieval_candidates.json, graph_expansion.json, routing.json,
    validation.json, generated_sql.sql, final_response.json) plus the raw
    trace (trace/full_trace.json) and a coverage report (coverage.json) that
    lists which spec datapoints this deployment does not emit yet.

This module is pure (no MLflow import) so it is unit-testable anywhere.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_KEY_OK = re.compile(r"[^A-Za-z0-9_\-./ ]+")

# Section display order (mirrors veda/explain.py _SECTIONS).
SECTIONS = [
    "query_understanding", "retrieval", "graph_expansion", "schema_linking",
    "anchor_selection", "join_planning", "sql_planning", "validation", "output",
]

# Spec artifact file name per section (mlflow_impl.md).
_SECTION_ARTIFACT = {
    "query_understanding": "layers/query_understanding.json",
    "retrieval": "layers/retrieval_candidates.json",
    "graph_expansion": "layers/graph_expansion.json",
    "schema_linking": "layers/schema_linking.json",
    "anchor_selection": "layers/routing.json",          # anchor selection == routing layer
    "join_planning": "layers/join_planning.json",
    "sql_planning": "layers/sql_planning.json",
    "validation": "layers/validation.json",
    "output": "layers/final_response.json",
}

# mlflow_impl.md datapoints the platform does NOT emit yet (the engine has no
# token/cost accounting and the trace has no tenant/session identity). Logged
# to coverage.json per run so the gap is visible instead of silently absent.
SPEC_GAPS = [
    "tenant_id", "conversation_id", "session_id", "user_id", "pipeline_version",
    "git_commit", "total_prompt_tokens", "total_completion_tokens",
    "total_tokens", "estimated_cost", "cpu_usage", "memory_usage",
    "rerank_model", "rerank_latency", "sql_model", "prompt_tokens",
    "completion_tokens", "memory_hits", "memory_injected", "summary_model",
    "summary_tokens", "chart_type", "chart_confidence",
]


@dataclass
class RunSpec:
    run_name: str
    params: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    # artifact path -> text content
    artifacts: Dict[str, str] = field(default_factory=dict)


def _san_key(key: str) -> str:
    return _KEY_OK.sub("_", str(key))[:250]


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _short(v: Any, limit: int) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def line_fingerprint(raw_line: str) -> str:
    return hashlib.sha1(raw_line.strip().encode("utf-8", "replace")).hexdigest()


def _section_durations(sections: Dict[str, dict], total_ms: Optional[float]) -> Dict[str, float]:
    """Sections stamp `_ms` (elapsed at first touch). Duration of a stage is
    approximated as the gap to the next stage's first touch (last stage runs
    until total_ms). Ordering is by _ms, not display order."""
    touched: List[Tuple[str, float]] = sorted(
        ((name, float(body["_ms"])) for name, body in sections.items()
         if isinstance(body, dict) and isinstance(body.get("_ms"), (int, float))),
        key=lambda t: t[1])
    out: Dict[str, float] = {}
    for i, (name, start) in enumerate(touched):
        if i + 1 < len(touched):
            out[name] = max(touched[i + 1][1] - start, 0.0)
        elif total_ms is not None:
            out[name] = max(float(total_ms) - start, 0.0)
    return out


def map_record(record: Dict[str, Any], *, raw_line: str = "",
               environment: str = "local", param_value_max: int = 500) -> RunSpec:
    full = record.get("full") or {}
    sections: Dict[str, dict] = full.get("sections") or {}
    query = full.get("query") or record.get("query") or ""
    total_ms = _num(record.get("total_ms"))

    qhash = hashlib.sha256(query.encode("utf-8", "replace")).hexdigest()[:12] if query else "unknown"
    run_name = (query[:60] + "…") if len(query) > 60 else (query or f"query-{qhash}")

    spec = RunSpec(run_name=run_name)
    p, m, t, a = spec.params, spec.metrics, spec.tags, spec.artifacts

    # ── run identity / tags ──────────────────────────────────────────────────
    t["veda.query_hash"] = qhash
    t["veda.environment"] = environment
    t["veda.trace_source"] = "explain_trace.jsonl"
    if raw_line:
        t["veda.line_fingerprint"] = line_fingerprint(raw_line)
    for tag_key in ("route", "status", "intent", "table", "action"):
        v = record.get(tag_key)
        if v not in (None, ""):
            t[f"veda.{tag_key}"] = str(v)[:250]

    # ── spec-named params ────────────────────────────────────────────────────
    if query:
        p["query"] = _short(query, param_value_max)
    for key in ("route", "intent", "table", "anchor", "action", "status", "refusal"):
        v = record.get(key)
        if v not in (None, ""):
            p[key] = _short(v, param_value_max)

    # ── spec-named metrics ───────────────────────────────────────────────────
    if total_ms is not None:
        m["total_latency_ms"] = total_ms
    for src, name in (("anchor_conf", "routing_confidence"),
                      ("join_conf", "join_confidence"),
                      ("confidence", "answer_confidence")):
        v = _num(record.get(src))
        if v is not None:
            m[name] = v
    status = record.get("status") or ""
    m["pipeline_success"] = 1.0 if status == "answered" else 0.0
    m["pipeline_refused"] = 1.0 if status.startswith(("refuse", "ungrounded", "no_table", "invalid")) else 0.0

    # per-stage latency (start offsets + derived durations)
    for name, dur in _section_durations(sections, total_ms).items():
        m[f"{_san_key(name)}.duration_ms"] = round(dur, 2)
    for name, body in sections.items():
        v = _num(body.get("_ms")) if isinstance(body, dict) else None
        if v is not None:
            m[f"{_san_key(name)}.start_offset_ms"] = v

    # ── retrieval / graph / validation counts (spec "contribution metrics") ──
    retr = sections.get("retrieval") or {}
    if isinstance(retr.get("candidate_tables"), list):
        m["retrieval_candidate_tables"] = float(len(retr["candidate_tables"]))
    if _num(retr.get("n_columns")) is not None:
        m["retrieval_candidate_columns"] = _num(retr["n_columns"])

    graph = sections.get("graph_expansion") or {}
    for src, name in (("seeds", "graph_seed_terms"), ("synonyms", "graph_synonyms_followed"),
                      ("added", "graph_columns_added")):
        if isinstance(graph.get(src), list):
            m[name] = float(len(graph[src]))
    if graph:
        m["graph_expansion_used"] = 1.0 if graph.get("added") else 0.0

    val = sections.get("validation") or {}
    checks = val.get("checks") or []
    if checks:
        passed = sum(1 for c in checks if c.get("status") == "pass")
        m["validation_checks_total"] = float(len(checks))
        m["validation_checks_passed"] = float(passed)
        m["validation_checks_failed"] = float(len(checks) - passed)
    repairs = val.get("repairs") or []
    m["repair_count"] = float(len(repairs))

    out = sections.get("output") or {}
    sql = out.get("sql") or ""
    if sql:
        m["sql_length"] = float(len(sql))
        m["sql_param_count"] = float(len(out.get("params") or []))
        m["sql_join_count"] = float(len(re.findall(r"\bJOIN\b", sql, re.IGNORECASE)))
        m["limit_present"] = 1.0 if re.search(r"\bLIMIT\b", sql, re.IGNORECASE) else 0.0

    # ── generic per-section sweep (nothing recorded is dropped) ──────────────
    for name, body in sections.items():
        if not isinstance(body, dict):
            continue
        sec = _san_key(name)
        for key, v in body.items():
            if key in ("_ms", "why"):
                continue
            n = _num(v)
            if n is not None:
                m.setdefault(f"{sec}.{_san_key(key)}", n)
            elif isinstance(v, str) and v:
                p.setdefault(f"{sec}.{_san_key(key)}", _short(v, param_value_max))
        # spec-named per-layer artifact carries the untruncated section
        a[_SECTION_ARTIFACT.get(name, f"layers/{sec}.json")] = json.dumps(
            body, indent=2, default=str)

    # ── artifacts ────────────────────────────────────────────────────────────
    a["trace/full_trace.json"] = json.dumps(record, indent=2, default=str)
    if sql:
        a["sql/generated_sql.sql"] = sql
    why_lines = [f"[{s}] {w}" for s in SECTIONS
                 for w in ((sections.get(s) or {}).get("why") or [])]
    if why_lines:
        a["trace/why.txt"] = "\n".join(why_lines)
    a["coverage.json"] = json.dumps(
        {"captured_sections": sorted(sections.keys()),
         "spec_datapoints_not_yet_emitted_by_engine": SPEC_GAPS},
        indent=2)

    return spec
