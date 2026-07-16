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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_TRACE_LOG = "logs/explain_trace.jsonl"

# The sections, in display order. llm_usage is stamped by finish() (per-query
# SLM token accounting from slm/_call_slm.py), not by a pipeline stage.
_SECTIONS = [
    "query_understanding", "retrieval", "graph_expansion", "schema_linking",
    "anchor_selection", "join_planning", "sql_planning", "validation",
    "llm_usage", "output",
]


class _NullTrace:
    """Zero-overhead stand-in used when tracing is disabled. Every method is a
    no-op so call sites stay identical and cost nothing in production-off mode."""
    enabled = False
    verbose = False

    def set(self, *a, **k): pass
    def note(self, *a, **k): pass
    def cand(self, *a, **k): pass
    def check(self, *a, **k): pass
    def repair(self, *a, **k): pass
    def to_dict(self): return None
    def compact(self): return None
    def render(self): return ""
    def finish(self, *a, **k): return None


@dataclass
class ExplainTrace:
    query: str
    verbose: bool = False
    enabled: bool = True
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

    # ── serialization ────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "total_ms": self.total_ms,
            "verbose": self.verbose,
            "sections": {s: self.sections.get(s, {}) for s in _SECTIONS
                         if s in self.sections},
        }

    def compact(self) -> Dict[str, Any]:
        """Production-safe summary for the route log — decisions, confidences,
        outcome, refusal reason. No candidate dumps regardless of verbose."""
        s = self.sections
        out = s.get("output", {})
        return {
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
    def finish(self, status: str, route: str = "") -> Dict[str, Any]:
        """Stamp timing/outcome, append the compact trace to the trace log, return
        the full dict. Persistence failures never break the query path."""
        self.total_ms = round((time.time() - self._t0) * 1000, 2)
        self.sections.setdefault("output", {}).setdefault("status", status)
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
    "graph_expansion":     ("Graph expand", "follow the knowledge graph (synonyms / FK) for missed columns"),
    "schema_linking":      ("Link to schema", "map the request onto real tables"),
    "anchor_selection":    ("Pick the subject", "decide which table the question is about"),
    "join_planning":       ("Plan joins", "connect tables along real foreign keys"),
    "sql_planning":        ("Plan the SQL", "choose the query shape (count / filter / join …)"),
    "validation":          ("Safety checks", "prove the SQL is correct & safe before running"),
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


def new_trace(query: str):
    """Factory: a real trace if enabled, else a zero-cost _NullTrace."""
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
    return ExplainTrace(query=query, verbose=bool(EXPLAIN_TRACE_VERBOSE))
