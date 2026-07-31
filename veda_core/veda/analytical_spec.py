"""veda.analytical_spec — Phase 1 (ANALYTICAL_SQL_V2, flag-gated, default OFF).

The benchmark showed SQL-generation DROPS the aggregate intent — "% verified" / "payment
method distribution" came back as raw-row LISTs instead of scalar/grouped aggregates,
because the generator free-infers SQL from language. Fix (per the frozen plan): a
lightweight STRUCTURED Analytical Query Specification that SQL-gen CONSUMES — it never
re-infers the aggregation from text.

Scope of Phase 1: SINGLE-ANCHOR analytics only (scalar COUNT/SUM/AVG/MIN/MAX, grouped
GROUP BY on an anchor dimension). Multi-table analytical joins are Phase 2. Deterministic,
reproducible, no LLM. Returns None when it can't build a safe spec → caller falls back to
the existing path (zero regression by construction).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_AGG = {"count", "sum", "avg", "max", "min"}


@dataclass
class AnalyticalSpec:
    anchor: str                              # real table (validated upstream)
    aggregation: str                         # count|sum|avg|max|min
    measure_column: Optional[str] = None     # real column; None ⇒ COUNT(*)
    distinct: bool = False
    group_keys: List[str] = field(default_factory=list)   # real columns on the anchor
    output_shape: str = "scalar"             # scalar | grouped
    top_n: Optional[int] = None
    direction: str = "desc"
    evidence: Dict[str, Any] = field(default_factory=dict)


def _anchor_columns(anchor: str, sm) -> Dict[str, dict]:
    cols = sm.get("columns", {})
    return {k.split(".", 1)[1]: (cols[k] or {}) for k in cols if k.split(".", 1)[0] == anchor}


def _resolve_column(concept: str, anchor: str, sm, *, numeric: bool = False) -> Optional[str]:
    """Ground a measure/dimension CONCEPT to a REAL column of the anchor. Deterministic:
    exact/underscore name match, then business_role/aliases, then (for measures) a numeric
    column whose name shares a token. None if unresolvable — never guesses."""
    if not concept:
        return None
    cols = _anchor_columns(anchor, sm)
    toks = [w for w in re.findall(r"[a-z]+", concept.lower()) if len(w) > 2]
    if not toks:
        return None
    # 1. direct name match (asset "carpet area" → carpet_area)
    joined = "_".join(toks)
    for c in cols:
        cl = c.lower()
        if cl == joined or set(toks) <= set(cl.split("_")):
            return c
    # 2. business_role / aliases text match
    for c, meta in cols.items():
        blob = (str(meta.get("business_role") or "") + " " + str(meta.get("aliases") or "")).lower()
        if blob and all(t in blob for t in toks):
            return c
    return None


_AGG_WORD_RE = re.compile(
    r"\b(?:total|sum|number|count|amount\s+of|no\.?\s+of|average|avg|mean|"
    r"max|maximum|highest|greatest|largest|min|minimum|lowest|smallest|least|most|of)\b")


def _strip_agg_words(concept: str) -> str:
    """Remove the OPERATION words the LLM prepends to a measure phrase so the residual names
    the COLUMN: "total paid amount"→"paid amount", "average carpet area"→"carpet area",
    "highest expected price"→"expected price". Purely lexical; the column match is still
    validated against the real schema downstream."""
    return _AGG_WORD_RE.sub(" ", (concept or "").lower()).strip()


def _num_type(meta: dict) -> bool:
    t = str(meta.get("data_type") or meta.get("type") or "").lower()
    return any(x in t for x in ("int", "numeric", "decimal", "float", "double", "money", "real"))


def derive_spec(grounded_intent, query: str, sm) -> Optional[AnalyticalSpec]:
    """GroundedIntent (+ query, sm) → AnalyticalSpec, or None if not a safe single-anchor
    analytical query. `grounded_intent` must expose .intent, .anchor, .secondaries,
    .measure (a GroundedMeasure|None), and ideally the raw concepts for grouping."""
    gi = grounded_intent
    if gi is None or getattr(gi, "anchor", None) is None:
        return None
    intent = getattr(gi, "intent", None)
    if intent not in _AGG:
        return None
    # Phase 1 = SINGLE anchor. If understanding resolved a second entity, this needs a
    # join → defer to Phase 2 (existing path), don't build here.
    if getattr(gi, "secondaries", None):
        return None
    anchor = gi.anchor

    agg = "count" if intent == "count" else intent
    measure_col = None
    distinct = False
    if agg == "count":
        # "how many X" → COUNT(*); "count of DISTINCT thing" handled by DISTINCT below
        distinct = bool(re.search(r"\bdistinct\b", query.lower()))
    else:
        # sum/avg/max/min NEED a numeric column — from the measure concept. The LLM prepends
        # the OPERATION as words ("total paid amount", "average carpet area", "highest expected
        # price"); those name the aggregation, not the column, so strip them before resolving —
        # else "total paid amount" never matches the paid_amount column ('total' isn't in it).
        mc = getattr(getattr(gi, "measure", None), "concept", None) or getattr(gi, "measure", None)
        mc = _strip_agg_words(str(mc) if mc else "")
        measure_col = _resolve_column(mc, anchor, sm, numeric=True)
        if measure_col is None:
            return None                      # can't ground the measure column → defer

    # group keys: the query's "per/by <dimension>" resolved to a REAL anchor column
    group_keys: List[str] = []
    gm = re.search(r"\b(?:per|by)\s+([a-z_][a-z_ ]{2,40})", query.lower())
    if gm:
        gcol = _resolve_column(gm.group(1).strip(), anchor, sm)
        if gcol:
            group_keys = [gcol]
    shape = "grouped" if group_keys else "scalar"

    # ranking (top N) — only meaningful for grouped/measure output
    _tn = re.search(r"\b(?:top|bottom|highest|lowest)\s+(\d+)\b", query.lower())
    top_n = int(_tn.group(1)) if _tn else None
    direction = "asc" if re.search(r"\b(bottom|lowest|least|fewest)\b", query.lower()) else "desc"

    return AnalyticalSpec(anchor=anchor, aggregation=agg, measure_column=measure_col,
                          distinct=distinct, group_keys=group_keys, output_shape=shape,
                          top_n=top_n, direction=direction,
                          evidence={"intent": intent, "source": "analytical_spec_v2"})


def emit_sql(spec: AnalyticalSpec, sm) -> Optional[str]:
    """AQS → SQL by REUSING the existing deterministic builder
    (planning.build_aggregate_sql, single-anchor branch) — NOT a parallel builder.
    AnalyticalSpec is an IR only; SQL construction stays in the one production builder
    (audit rule: no 7th SQL path). Returns None → caller falls back to existing path."""
    if spec is None:
        return None
    try:
        from veda.planning import build_aggregate_sql
    except Exception:
        return None
    sql, _tables = build_aggregate_sql(
        spec.anchor, [], sm,
        top_n=spec.top_n, direction=spec.direction,
        group_col=(spec.group_keys[0] if spec.group_keys else None),
        measure_agg=spec.aggregation, measure_column=spec.measure_column,
        distinct=spec.distinct)
    return sql
