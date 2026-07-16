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
    "anchor_selection", "join_planning", "sql_planning", "validation",
    "llm_usage", "output",
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

# mlflow_impl.md datapoints the platform does not necessarily emit (the trace
# has no tenant/session identity; token totals exist only when the engine's
# llm_usage section is present — coverage.json drops from this list whatever
# THIS run actually carried, so the report stays truthful per record).
SPEC_GAPS = [
    "tenant_id", "conversation_id", "session_id", "user_id", "pipeline_version",
    "git_commit", "total_prompt_tokens", "total_completion_tokens",
    "total_tokens", "estimated_cost", "cpu_usage", "memory_usage",
    "rerank_model", "rerank_latency", "sql_model", "prompt_tokens",
    "completion_tokens", "memory_hits", "memory_injected", "summary_model",
    "summary_tokens", "chart_type", "chart_confidence",
]

# Spec Layer-2 "Signal Scores" (mlflow_impl.md). The engine (veda/pipeline.py
# tr.cand("retrieval", "top_columns", ...)) emits each candidate as
#   {col, score, signals: {semantic, sparse, subgraph, fk_path, value}}
# — `score` is final_score (post-rerank when the reranker ran) and the
# per-signal scores sit NESTED under `signals` with short names. Older traces
# and pre-upgrade records carry flat *_score keys instead. _flat_signals()
# normalizes both shapes to the flat spec names below, and map_record()
# promotes EVERY signal key it finds on a candidate, so new signals appear in
# MLflow with zero exporter changes; coverage.json and
# layers/signal_scores.json report per run which are present vs missing.
SIGNAL_SCORE_KEYS = [
    "semantic_score", "bm25_score", "graph_score", "subgraph_score", "fk_score",
    "value_score", "rrf_score", "cross_encoder_score", "final_score", "score",
]

# Engine's nested `signals` short names → flat spec names (`sparse` IS the
# BM25/lexical signal; `fk_path`/`value` are RetrievalResult's
# fk_path_score/value_index_score).
_NESTED_SIGNAL_NAME = {
    "semantic": "semantic_score", "sparse": "bm25_score",
    "subgraph": "subgraph_score", "fk_path": "fk_score", "value": "value_score",
}

# One spec signal, two engine spellings: RetrievalResult calls the graph
# signal `subgraph_score` while mlflow_impl.md calls it `graph_score`, and the
# trace's candidate `score` IS final_score (veda/pipeline.py copies it).
# Either spelling present means the signal is covered.
_ALIAS_GROUPS = [{"graph_score", "subgraph_score"}, {"final_score", "score"}]


def missing_signals(present: List[str]) -> List[str]:
    """Spec signal keys not carried by this run's candidates (alias-aware)."""
    have = set(present)
    for group in _ALIAS_GROUPS:
        if have & group:
            have |= group
    return [k for k in SIGNAL_SCORE_KEYS if k != "score" and k not in have]


def _flat_signals(cand: Dict[str, Any]) -> Dict[str, Any]:
    """Candidate with any nested `signals` dict promoted to flat spec-named
    keys. Flat keys already on the candidate win (older traces carry both)."""
    sig = cand.get("signals")
    if not isinstance(sig, dict):
        return cand
    flat = dict(cand)
    for k, v in sig.items():
        flat.setdefault(_NESTED_SIGNAL_NAME.get(k, k), v)
    return flat


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


def _col_short_name(col_id: Any) -> str:
    """'table.column' -> 'column' (trace column ids are table-qualified)."""
    s = str(col_id)
    return s.split(".", 1)[1] if "." in s else s


def _first_col(seq: Any) -> Optional[str]:
    """Top entry of a candidate list — items are either 'table.col' strings or
    {col: ...} dicts (both shapes appear in traces)."""
    if isinstance(seq, list) and seq:
        head = seq[0]
        return str(head.get("col")) if isinstance(head, dict) else str(head)
    return None


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
        v = graph.get(src)
        if isinstance(v, list):
            m[name] = float(len(v))
        elif isinstance(v, dict):
            # synonyms is {seed: [synonyms]} (graph/query_graph.suggest_expansions)
            m[name] = float(sum(len(x) if isinstance(x, list) else 1 for x in v.values()))
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

    # ── LLM token usage (spec total_*_tokens) — engine's llm_usage section ──
    # (stamped by ExplainTrace.finish() from slm/_call_slm.py accounting; the
    # per-purpose breakdown lands in layers/llm_usage.json via the sweep)
    lu = sections.get("llm_usage") or {}
    for src, name in (("total_prompt_tokens", "total_prompt_tokens"),
                      ("total_completion_tokens", "total_completion_tokens"),
                      ("total_tokens", "total_tokens"),
                      ("calls", "llm_calls")):
        v = _num(lu.get(src))
        if v is not None:
            m[name] = v

    out = sections.get("output") or {}
    sql = out.get("sql") or ""
    if sql:
        m["sql_length"] = float(len(sql))
        m["sql_param_count"] = float(len(out.get("params") or []))
        m["sql_join_count"] = float(len(re.findall(r"\bJOIN\b", sql, re.IGNORECASE)))
        m["limit_present"] = 1.0 if re.search(r"\bLIMIT\b", sql, re.IGNORECASE) else 0.0

    # ── Layer-2 signal scores (spec "Signal Scores") ─────────────────────────
    # retrieval.top_columns is the engine's per-candidate list; nested
    # `signals` short names are flattened to spec keys, and any extra *_score
    # key a future engine adds is promoted too (nothing recorded is dropped).
    top_cols = [_flat_signals(c) for c in (retr.get("top_columns") or [])
                if isinstance(c, dict)]
    signals_present: List[str] = []
    if top_cols:
        m["retrieval.top_columns_count"] = float(len(top_cols))
        p["retrieval.top1_column"] = _short(top_cols[0].get("col", ""), param_value_max)
        p["retrieval.top_columns"] = _short(
            ", ".join(str(c.get("col", "")) for c in top_cols), param_value_max)
        promote = SIGNAL_SCORE_KEYS + sorted(
            {k for c in top_cols for k in c if k.endswith("_score")}
            - set(SIGNAL_SCORE_KEYS))
        for key in promote:
            vals = [v for v in (_num(c.get(key)) for c in top_cols) if v is not None]
            if not vals:
                continue
            signals_present.append(key)
            m[f"retrieval.top1_{key}"] = vals[0]
            m[f"retrieval.{key}_mean"] = round(sum(vals) / len(vals), 6)
            if len(vals) > 1:
                m[f"retrieval.{key}_top1_vs_top2_gap"] = round(vals[0] - vals[1], 6)

    # anchor routing signals (spec Layer 3) — alternatives carry a per-signal
    # breakdown ({table, score, signals}) straight from veda/routing.py.
    anch = sections.get("anchor_selection") or {}
    alts = [x for x in (anch.get("alternatives") or []) if isinstance(x, dict)]
    if alts:
        m["routing.alternatives_count"] = float(len(alts))
        sig0 = alts[0].get("signals")
        if isinstance(sig0, dict):
            for k, v in sig0.items():
                n = _num(v)
                if n is not None:
                    m[f"routing.top1_signal_{_san_key(k)}"] = n

    # ── reranker contribution (spec Layer 2b) ────────────────────────────────
    # No rerank section is emitted yet; recognise one (or before/after lists on
    # retrieval) as soon as it exists. Scalars inside a rerank section are also
    # picked up by the generic sweep below.
    rr = sections.get("rerank") or sections.get("reranking") or {}
    before = _first_col(rr.get("top_before") or retr.get("top_before_rerank"))
    after = _first_col(rr.get("top_after") or retr.get("top_after_rerank"))
    if before is not None and after is not None:
        m["reranker_changed_top1"] = 0.0 if before == after else 1.0
        p["rerank.top1_before"] = _short(before, param_value_max)
        p["rerank.top1_after"] = _short(after, param_value_max)

    # ── selected columns through the funnel (spec "Contribution Metrics") ────
    # retrieval candidates → graph-added → selected table → columns the final
    # SQL actually uses (matched by column name against the generated SQL).
    sl = sections.get("schema_linking") or {}
    graph_added = [str(x) for x in (graph.get("added") or [])]
    cand_cols = [str(c.get("col", "")) for c in top_cols if c.get("col")]
    known_cols = list(dict.fromkeys(cand_cols + graph_added))
    used_in_sql = [c for c in known_cols
                   if re.search(rf"\b{re.escape(_col_short_name(c))}\b", sql)] if sql else []
    if known_cols:
        m["columns.candidate_count"] = float(len(known_cols))
        if sql:
            m["columns.used_in_sql_count"] = float(len(used_in_sql))
            m["columns.selection_ratio"] = round(len(used_in_sql) / len(known_cols), 4)
            if used_in_sql:
                p["columns.used_in_sql"] = _short(", ".join(used_in_sql), param_value_max)

    if top_cols or alts:
        a["layers/signal_scores.json"] = json.dumps({
            "signals_present": signals_present,
            "signals_not_emitted_by_engine": missing_signals(signals_present),
            "note": "`score` on a candidate is the engine's final_score "
                    "(post-rerank when the reranker ran); nested `signals` "
                    "short names (semantic/sparse/subgraph/fk_path/value) are "
                    "normalized to flat spec keys and promoted to metrics.",
            "retrieval_candidates": top_cols,
            "anchor_alternatives": alts,
        }, indent=2, default=str)
    if known_cols or alts or sl:
        a["layers/selected_columns.json"] = json.dumps({
            "selected_table": sl.get("selected_table"),
            "router_primary": sl.get("router_primary"),
            "candidate_tables": retr.get("candidate_tables") or sl.get("candidate_tables"),
            "retrieval_top_columns": top_cols,
            "graph_added_columns": graph_added,
            "anchor_alternatives": alts,
            "columns_used_in_final_sql": used_in_sql,
            "final_sql": sql or None,
        }, indent=2, default=str)

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
    gaps = [k for k in SPEC_GAPS if _num(lu.get(k)) is None]
    a["coverage.json"] = json.dumps(
        {"captured_sections": sorted(sections.keys()),
         "signal_scores_present": signals_present,
         "signal_scores_missing": missing_signals(signals_present),
         "spec_datapoints_not_yet_emitted_by_engine": gaps},
        indent=2)

    return spec
