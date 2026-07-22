# =============================================================================
# veda/explain.py
# Explainability Trace — a structured, JSON-serializable record of WHY the
# pipeline produced (or refused) a given SQL. Captured from data the stages
# already compute (no extra DB/LLM calls), so the overhead is dict appends.
#
# Three cost levels:
#   EXPLAIN_TRACE_ENABLED=False        → new_trace() returns _NullTrace (no-ops).
#   EXPLAIN_TRACE_ENABLED=True         → decisions + confidences + why (compact).
#   + EXPLAIN_TRACE_VERBOSE=True       → also full candidate lists / rejected paths.
#
# Consumed by veda/pipeline.run_query: attached to the result dict and persisted
# to logs/explain_trace.jsonl. .render() gives a human-readable view for debug.
# =============================================================================

import os
import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_TRACE_LOG = "logs/explain_trace.jsonl"

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:  # importable outside the engine cwd too (unit tests)
    import logging
    logger = logging.getLogger("veda.explain")

# The sections, in display order. This is the ONE per-query trace: every stage —
# Tier-1 (pipeline.py), Tier-2 (veda_hybrid/slm_langgraph), retrieval, summary,
# visualization — records into the SAME ExplainTrace, reached via the ambient
# `current_trace()` contextvar (below) so no stage has to thread a `tr` param.
#   llm_usage  — per-purpose SLM TOKEN totals, stamped by finish() from
#                slm/_call_slm.py::get_usage().
#   slm        — per-CALL SLM ledger (purpose/model/duration/ok), appended live by
#                call_slm() via slm_call(); complements the llm_usage totals.
#   totals     — the final one-glance summary, built by finish().
_SECTIONS = [
    "query_understanding", "retrieval", "rrf", "graph_expansion", "reranking",
    "schema_linking", "entity_selection", "projection", "join_planning",
    "tier1", "tier2", "sql_planning", "sql_generation", "validation",
    "execution", "result_analysis", "summary", "visualization",
    "explainability", "slm", "llm_usage", "output", "totals",
]


class _NullTrace:
    """Zero-overhead stand-in used when tracing is disabled. Every method is a
    no-op so call sites stay identical and cost nothing in production-off mode."""
    enabled = False
    verbose = False

    trace_id = ""

    def set(self, *a, **k): pass
    def note(self, *a, **k): pass
    def cand(self, *a, **k): pass
    def check(self, *a, **k): pass
    def repair(self, *a, **k): pass
    def slm_call(self, *a, **k): pass
    def to_dict(self): return None
    def compact(self): return None
    def render(self): return ""
    def finish(self, *a, **k): return None
    def finalize(self, *a, **k): return None


@dataclass
class ExplainTrace:
    query: str
    verbose: bool = False
    enabled: bool = True
    trace_id: str = ""            # ONE correlation id per user query (see new_trace)
    _t0: float = field(default_factory=time.time)
    sections: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0

    # ── recording API (all cheap dict ops) ──────────────────────────────────
    def _sec(self, section: str) -> Dict[str, Any]:
        """Section accessor. Stamps `_ms` (elapsed since trace start) the FIRST
        time a section is touched — per-stage timing for free, since stages touch
        their section when they run. Consumers: latency budgeting/calibration."""
        d = self.sections.get(section)
        if d is None:
            d = self.sections[section] = {"_ms": round((time.time() - self._t0) * 1000, 1)}
        return d

    def set(self, section: str, **data) -> None:
        """Set/merge scalar decision fields for a section."""
        self._sec(section).update(data)

    def note(self, section: str, msg: str) -> None:
        """Append a human 'why' line."""
        self._sec(section).setdefault("why", []).append(msg)

    def cand(self, section: str, key: str, item: Any) -> None:
        """Append to a candidate/rejected list — VERBOSE-only (the heavy data)."""
        if not self.verbose:
            return
        self._sec(section).setdefault(key, []).append(item)

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        """Record a validation check pass/fail."""
        self._sec("validation").setdefault("checks", []).append(
            {"name": name, "status": "pass" if passed else "fail",
             **({"detail": detail} if detail else {})})

    def repair(self, what: str, frm: Any, to: Any) -> None:
        self._sec("validation").setdefault("repairs", []).append(
            {"what": what, "from": frm, "to": to})

    def slm_call(self, purpose: str, model: Optional[str], duration_ms: float,
                 ok: bool, error: Optional[str] = None) -> None:
        """Append one SLM invocation to the per-query ledger. Called from the
        single choke-point call_slm() (slm/_call_slm.py) so EVERY SLM call — for
        any purpose, from any stage — is visible in one place with its duration
        and success. Light metadata only: prompt/response bodies are never
        recorded here (that stays verbose-gated at the call site). Answers, for
        one trace_id: how many SLM calls happened, and why each one."""
        sec = self._sec("slm")
        sec.setdefault("calls", []).append({
            "purpose": purpose,
            "model": model,
            "duration_ms": round(float(duration_ms), 1),
            "ok": bool(ok),
            **({"error": str(error)[:200]} if error else {}),
        })
        sec["count"] = len(sec["calls"])

    # ── serialization ────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "total_ms": self.total_ms,
            "verbose": self.verbose,
            "sections": {s: self.sections.get(s, {}) for s in _SECTIONS
                         if s in self.sections},
        }

    def _build_totals(self) -> Dict[str, Any]:
        """The one-glance per-query summary — the FIRST record to inspect for
        perf/debugging. Built purely from values other stages already recorded
        (no recompute). Per-stage wall-times are approximated from each section's
        `_ms` start offset (same gap-to-next-section method the MLflow mapper
        uses); sections that never ran are simply absent."""
        s = self.sections
        # ordered (section, start_ms) for the sections that actually ran
        touched = [(sec, s[sec]["_ms"]) for sec in _SECTIONS
                   if sec in s and isinstance(s[sec], dict) and "_ms" in s[sec]]
        touched.sort(key=lambda kv: kv[1])
        durations: Dict[str, float] = {}
        for i, (sec, t) in enumerate(touched):
            nxt = touched[i + 1][1] if i + 1 < len(touched) else self.total_ms
            durations[sec] = round(max(0.0, nxt - t), 1)
        slm_calls = (s.get("slm", {}) or {}).get("calls") or []
        exec_sec = s.get("execution", {}) or {}
        viz = s.get("visualization", {}) or {}
        out = s.get("output", {}) or {}
        return {
            "trace_id": self.trace_id,
            "status": out.get("status"),
            "total_duration_ms": self.total_ms,
            "stage_durations_ms": durations,
            "slm_call_count": len(slm_calls),
            "slm_total_duration_ms": round(
                sum(c.get("duration_ms", 0) for c in slm_calls), 1),
            "slm_total_tokens": self.total_tokens,
            "row_count": exec_sec.get("row_count"),
            "column_count": exec_sec.get("column_count"),
            "chart_count": viz.get("selected_count"),
        }

    def compact(self) -> Dict[str, Any]:
        """Production-safe summary for the route log — decisions, confidences,
        outcome, refusal reason. No candidate dumps regardless of verbose."""
        s = self.sections
        out = s.get("output", {})
        return {
            "trace_id": self.trace_id,
            "intent": s.get("query_understanding", {}).get("intent"),
            "table": s.get("schema_linking", {}).get("selected_table"),
            "anchor": s.get("anchor_selection", {}).get("anchor"),
            "anchor_conf": s.get("anchor_selection", {}).get("confidence"),
            "join_conf": s.get("join_planning", {}).get("confidence"),
            "action": s.get("sql_planning", {}).get("action"),
            "status": out.get("status"),
            "confidence": out.get("confidence"),
            "refusal": out.get("refusal"),
            "total_ms": self.total_ms,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }

    # ── persistence + human view ──────────────────────────────────────────────
    def _stamp(self, status: str, force_status: bool = False) -> None:
        """Finalize timing/outcome/usage/totals in place (no persistence)."""
        self.total_ms = round((time.time() - self._t0) * 1000, 2)
        outd = self.sections.setdefault("output", {})
        if force_status or "status" not in outd:
            outd["status"] = status
        try:  # best-effort: token accounting can never break the query path
            from slm._call_slm import get_usage
            u = get_usage()
            if u and u.get("calls"):
                self.sections.setdefault("llm_usage", {}).update(
                    calls=u["calls"],
                    total_prompt_tokens=u["prompt_tokens"],
                    total_completion_tokens=u["completion_tokens"],
                    total_tokens=u["prompt_tokens"] + u["completion_tokens"],
                    per_purpose=u["per_purpose"])
                self.total_prompt_tokens = u["prompt_tokens"]
                self.total_completion_tokens = u["completion_tokens"]
                self.total_tokens = u["prompt_tokens"] + u["completion_tokens"]
        except Exception:
            pass
        # The one-glance summary — built last, from what every stage recorded.
        try:
            self.sections["totals"] = self._build_totals()
        except Exception:
            pass

    def _persist(self, route: str) -> None:
        try:  # one concise INFO line: the first thing a developer greps by trace_id
            t = self.sections.get("totals", {})
            logger.info(
                "query trace_id=%s status=%s total_ms=%s slm_calls=%s rows=%s cols=%s route=%s",
                self.trace_id, t.get("status"), t.get("total_duration_ms"),
                t.get("slm_call_count"), t.get("row_count"), t.get("column_count"),
                route or "?")
        except Exception:
            pass
        try:
            from config import EXPLAIN_TRACE_PERSIST
        except Exception:
            EXPLAIN_TRACE_PERSIST = True
        if EXPLAIN_TRACE_PERSIST:
            try:
                os.makedirs(os.path.dirname(_TRACE_LOG), exist_ok=True)
                rec = {"route": route, **self.compact()}
                rec["full"] = self.to_dict() if self.verbose else None
                with open(_TRACE_LOG, "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
            except Exception:
                pass

    def finish(self, status: str, route: str = "") -> Dict[str, Any]:
        """Stamp timing/outcome and (unless an outer scope owns this trace) persist.

        When this trace is the AMBIENT one — i.e. an outer scope (run_hybrid_query)
        bound it and will call finalize() at the true end of the query — a finish()
        from an inner stage (Tier-1's run_query) is only a CHECKPOINT: it refreshes
        the fields so the returned snapshot is current, but does NOT persist/log, so
        one query yields exactly one trace record even when Tier-1 → Tier-2 both run.
        Standalone (unbound) use persists exactly as before."""
        self._stamp(status, force_status=False)
        if _CURRENT_TRACE.get() is self:
            return self.to_dict()          # owner (run_hybrid_query) will finalize
        self._persist(route)
        return self.to_dict()

    def finalize(self, status: str, route: str = "") -> Dict[str, Any]:
        """The owner's end-of-query call: force the final status, stamp, persist
        once. Used by run_hybrid_query, which owns the ambient trace for the whole
        request (Tier-1 + Tier-2 + summary + viz all recorded into it)."""
        self._stamp(status, force_status=True)
        self._persist(route)
        return self.to_dict()

    def render(self) -> str:
        """Human-readable trace for --debug / logs."""
        return render_trace(self.to_dict(), verbose=self.verbose)


# ---------------------------------------------------------------------------
# Human-friendly renderer (shared by ExplainTrace.render + callers' render_trace,
# so the two views can never drift). Turns the raw section dict into a numbered,
# plain-language walkthrough: each step says WHAT it does, the DECISION it made,
# and WHY — ending in a clear answered/refused verdict.
# ---------------------------------------------------------------------------

# section key → (step title, one-line "what this step does")
_STEP_META = {
    "query_understanding": ("Understand", "what the question is really asking"),
    "retrieval":           ("Find columns", "search the schema for relevant columns"),
    "rrf":                 ("Fuse signals", "reciprocal-rank-fuse the retrieval signals"),
    "graph_expansion":     ("Graph expand", "follow the knowledge graph (synonyms / FK) for missed columns"),
    "reranking":           ("Rerank", "cross-encoder rescoring of the top candidates"),
    "schema_linking":      ("Link to schema", "map the request onto real tables"),
    "entity_selection":    ("Select entities", "primary + secondary tables the answer needs"),
    "projection":          ("Projection", "which columns should appear in the output"),
    "anchor_selection":    ("Pick the subject", "decide which table the question is about"),
    "join_planning":       ("Plan joins", "connect tables along real foreign keys"),
    "tier1":               ("Tier-1 snapshot", "what Tier-1 knew when it handed off to Tier-2"),
    "tier2":               ("Tier-2 / SLM IR", "LLM-generated intermediate representation"),
    "sql_planning":        ("Plan the SQL", "choose the query shape (count / filter / join …)"),
    "sql_generation":      ("Build SQL", "the SELECT/joins/filters actually generated"),
    "validation":          ("Safety checks", "prove the SQL is correct & safe before running"),
    "execution":           ("Execute", "run the SQL and read the result shape"),
    "result_analysis":     ("Analyse result", "shape / dimensions / measures / patterns"),
    "summary":             ("Summarise", "which summariser produced the prose answer"),
    "visualization":       ("Visualise", "which chart was chosen and why"),
    "explainability":      ("Explain", "the end-user explainability payload"),
    "slm":                 ("SLM calls", "every SLM invocation this query made"),
    "output":              ("Result", "the final SQL and outcome"),
}

# field key → friendly label (unknown keys fall back to the raw key)
_FIELD_LABEL = {
    "intent": "asking for", "temporal": "time window", "existence": "existence check",
    "aggregation": "aggregation", "candidate_tables": "candidate tables",
    "n_columns": "columns scanned", "top_columns": "top matches",
    "selected_table": "linked table", "router_primary": "router's pick",
    "anchor": "subject table", "confidence": "confidence", "margin": "lead over 2nd",
    "overrode_router": "overrode router", "source": "decided by", "alternatives": "also considered",
    "action": "query shape", "table": "table", "anchor_conf": "subject confidence",
    "join_path": "join path", "sql": "SQL", "status": "outcome", "refusal": "why refused",
    "seeds": "graph seed terms", "synonyms": "synonyms followed", "added": "columns added via graph",
}

# validation check key → plain-language meaning
_CHECK_MEAN = {
    "value_grounding": "every filter value actually exists in the data",
    "qualifier_completeness": "nothing the user asked for was silently dropped",
    "ir_equivalence": "no unrequested filters / joins / grouping were added",
    "ast_readonly_parameterized_fanout": "read-only, parameterised, no fan-out double-count",
}

_OUTCOME = {
    "answered": "✓ ANSWERED", "refuse": "✗ REFUSED", "clarify": "⚠ NEEDS CLARIFICATION",
    "ungrounded": "✗ REFUSED (a value wasn't in the data)",
    "no_table": "✗ REFUSED (couldn't identify the table)",
    "exec_error": "✗ EXECUTION ERROR", "invalid": "✗ REFUSED (SQL failed a safety check)",
}


def _fmt_val(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str)
    return str(v)


def render_trace(d: Optional[Dict[str, Any]], verbose: bool = False) -> str:
    """Human-readable walkthrough from a to_dict() trace. Numbered steps, plain-language
    decisions, clear verdict. The same view ExplainTrace.render() produces."""
    if not d:
        return ""
    secs = d.get("sections") or {}
    out = (secs.get("output") or {})
    status = out.get("status", "?")
    query = d.get("query", "")
    ms = d.get("total_ms", "?")

    verdict = _OUTCOME.get(status, status.upper() if status else "?")
    W = 74
    L = ["╭" + "─" * W + "╮",
         f"│  VEDA · how it answered",
         f"│  Query    : \"{query}\""[:W + 1],
         f"│  Outcome  : {verdict}   ·   {ms} ms",
         "╰" + "─" * W + "╯", ""]

    step = 0
    for sec in _SECTIONS:
        body = secs.get(sec)
        if not body:
            continue
        title, what = _STEP_META.get(sec, (sec.replace("_", " ").title(), ""))

        # validation gets a dedicated, friendly pass/fail block
        if sec == "validation":
            checks = body.get("checks") or []
            allok = all(c.get("status") == "pass" for c in checks) if checks else True
            L.append(f"  {'✓' if allok else '✗'} {title.upper()} — {what}")
            for c in checks:
                ok = c.get("status") == "pass"
                mean = _CHECK_MEAN.get(c.get("name", ""), c.get("name", ""))
                tail = "" if ok else f"   → {c.get('detail','')}"
                L.append(f"        {'✓' if ok else '✗'} {mean}{tail}")
            L.append("")
            continue

        if sec == "output":
            L.append(f"  ▸ {title.upper()} — {what}")
            if out.get("refusal"):
                L.append(f"        why refused : {out['refusal']}")
            if out.get("sql"):
                L.append(f"        SQL : {out['sql']}")
            if out.get("confidence") is not None:
                L.append(f"        confidence : {_fmt_val(out['confidence'])}")
            L.append("")
            continue

        step += 1
        L.append(f"  {step} ▸ {title.upper()} — {what}")
        for k, v in body.items():
            if k == "why":
                for w in v:
                    L.append(f"        · {w}")
                continue
            if isinstance(v, list) and not verbose:
                # show count + first item only in non-verbose
                head = _fmt_val(v[0])[:80] if v else "—"
                L.append(f"        {_FIELD_LABEL.get(k, k)} : {len(v)} ({head}…)" if v
                         else f"        {_FIELD_LABEL.get(k, k)} : none")
                continue
            L.append(f"        {_FIELD_LABEL.get(k, k)} : {_fmt_val(v)[:300]}")
        L.append("")

    return "\n".join(L).rstrip()


def new_trace(query: str, trace_id: Optional[str] = None):
    """Factory: a real trace if enabled, else a zero-cost _NullTrace.

    Ambient-reuse: if a trace is ALREADY bound to the current context (the top-level
    run_hybrid_query mints/binds one), return THAT trace so every downstream stage —
    Tier-1's run_query included — writes into the ONE per-query trace instead of
    starting a cold second one. Reuse must NOT reset the SLM token accumulator (that
    was reset once when the ambient trace was created); resetting mid-query would
    lose the tokens already spent (e.g. by the L0 simplifier). Only a fresh trace
    (standalone run_query / tests) resets usage — byte-identical to the old behaviour."""
    existing = _CURRENT_TRACE.get()
    if existing is not None and getattr(existing, "enabled", False):
        return existing
    try:  # best-effort: start a fresh per-query SLM token accumulator
        from slm._call_slm import reset_usage
        reset_usage()
    except Exception:
        pass
    try:
        from config import EXPLAIN_TRACE_ENABLED, EXPLAIN_TRACE_VERBOSE
    except Exception:
        return _NullTrace()
    if not EXPLAIN_TRACE_ENABLED:
        return _NullTrace()
    return ExplainTrace(query=query, verbose=bool(EXPLAIN_TRACE_VERBOSE),
                        trace_id=trace_id or mint_trace_id())


# ---------------------------------------------------------------------------
# Ambient current-trace propagation (mirrors context.py / _call_slm.py's
# ContextVar pattern). Set ONCE at the top of a request (run_hybrid_query) and
# read by any stage — Tier-2, retrieval, the SLM choke-point — without threading
# a `tr` parameter through every signature. Worker threads start with an EMPTY
# contextvars context, so parallel sub-queries must re-bind via use_trace()
# (the same caveat _call_slm.py documents for token accounting).
# ---------------------------------------------------------------------------
_NULL_TRACE = _NullTrace()
_CURRENT_TRACE: ContextVar[Optional["ExplainTrace"]] = ContextVar(
    "veda_current_trace", default=None)


def mint_trace_id() -> str:
    """One short correlation id per user query. Reused across API → pipeline →
    Tier-1 → Tier-2 → SLM → execution → response so a developer can grep one id
    and reconstruct the whole lifecycle."""
    return uuid.uuid4().hex[:12]


def current_trace():
    """The trace bound to this context, or a zero-cost _NullTrace when none is
    bound (standalone use / tracing disabled) — so callers never need a guard."""
    return _CURRENT_TRACE.get() or _NULL_TRACE


def bind_trace(tr):
    """Bind `tr` as the ambient trace; returns a token for unbind_trace()."""
    return _CURRENT_TRACE.set(tr)


def unbind_trace(token) -> None:
    try:
        _CURRENT_TRACE.reset(token)
    except Exception:
        pass


@contextmanager
def use_trace(tr):
    """Scope `tr` as the ambient trace for the duration of the block. Used by
    run_hybrid_query (whole request) and by worker threads carrying the parent
    trace into a fresh contextvars context."""
    token = bind_trace(tr)
    try:
        yield tr
    finally:
        unbind_trace(token)


def record_result_stages(*, engine=None, cols=None, row_count=None, truncated=False,
                         ictx=None, answer=None, summary_model=None, summary_ok=None,
                         visualization=None, explain_payload=None) -> None:
    """Record the shared post-execution stages — execution / result_analysis /
    summary / visualization / explainability — into the current trace, reading ONLY
    values the pipeline already computed (the InsightContext, the summary text, the
    chart candidates, the explain payload). No recompute, no new SQL/LLM. Called by
    BOTH Tier-1 (pipeline.py) and Tier-2 (veda_hybrid._tier2_finish) so the two
    tiers tell the SAME structured story for one trace_id."""
    tr = current_trace()
    if not getattr(tr, "enabled", False):
        return
    try:
        _cols = list(cols) if cols is not None else None
        if _cols is not None or row_count is not None:
            tr.set("execution",
                   row_count=row_count,
                   column_count=(len(_cols) if _cols is not None else None),
                   truncated=bool(truncated))
            if _cols and getattr(tr, "verbose", False):   # column NAMES, verbose-only
                tr.set("execution", column_names=_cols[:40])
    except Exception:
        pass
    try:  # result_analysis — straight off the InsightContext (already computed)
        if ictx is not None:
            _dims = [getattr(d, "name", d) for d in (getattr(ictx, "dimensions", None) or [])]
            _meas = [getattr(m, "name", m) for m in (getattr(ictx, "measures", None) or [])]
            _pats = [getattr(p, "detail", str(p)) for p in (getattr(ictx, "patterns", None) or [])]
            tr.set("result_analysis",
                   result_shape=getattr(ictx, "result_shape", None),
                   result_type=getattr(ictx, "result_type", None),
                   dimensions=_dims,
                   measures=_meas,
                   entities=getattr(ictx, "entities", None),
                   pattern_count=len(_pats))
            for _p in _pats[:5]:
                tr.cand("result_analysis", "patterns", _p)
    except Exception:
        pass
    try:  # summary — WHICH summariser produced the prose, model, size, success
        if engine is not None or answer is not None:
            tr.set("summary",
                   engine=engine,
                   model=summary_model,
                   success=summary_ok,
                   answer_chars=(len(answer) if isinstance(answer, str) else None))
    except Exception:
        pass
    try:  # visualization — deterministic candidates on the ctx + the selected chart
        _cands = list(getattr(ictx, "chart_candidates", None) or []) if ictx is not None else []
        _selected = ([visualization] if isinstance(visualization, dict)
                     else (list(visualization) if isinstance(visualization, list) else []))
        if _cands or _selected:
            _shape = getattr(ictx, "result_shape", None) if ictx is not None else None
            _sel_type = (_selected[0].get("type") if _selected and isinstance(_selected[0], dict)
                         else None)
            tr.set("visualization",
                   candidate_count=len(_cands),
                   candidate_charts=[c.get("type") for c in _cands if isinstance(c, dict)],
                   selected_count=len(_selected),
                   selection_reason=(f"{_shape}→{_sel_type}" if (_shape and _sel_type) else None))
            for _c in _selected:                      # verbose-only chart detail
                tr.cand("visualization", "selected", _c)
    except Exception:
        pass
    try:  # explainability — compact only, NEVER the full payload
        if isinstance(explain_payload, dict):
            _checks = explain_payload.get("check_items") or []
            tr.set("explainability",
                   datasets=explain_payload.get("datasets"),
                   operation_count=len(explain_payload.get("operations") or []),
                   filter_count=len(explain_payload.get("filters")
                                    or explain_payload.get("filter_phrases") or []),
                   validation_passed=(all(
                       str(c.get("status")).lower() in ("pass", "true", "ok")
                       for c in _checks) if _checks else None))
    except Exception:
        pass
