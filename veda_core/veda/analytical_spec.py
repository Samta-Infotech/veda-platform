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
    where_sql: Optional[str] = None          # M2: grounded predicates on the anchor (alias t0)
    evidence: Dict[str, Any] = field(default_factory=dict)


def _anchor_columns(anchor: str, sm) -> Dict[str, dict]:
    cols = sm.get("columns", {})
    return {k.split(".", 1)[1]: (cols[k] or {}) for k in cols if k.split(".", 1)[0] == anchor}


def _resolve_column(concept: str, anchor: str, sm, *, numeric: bool = False,
                    kinds=None) -> Optional[str]:
    """Ground a measure/dimension CONCEPT to a REAL column of the anchor — the ONE
    resolver lives in veda.understanding.grounding (M2, 2026-09-16); `numeric=True` is
    now enforced there (a SUM/AVG can only land on a numeric column), and `kinds`
    restricts dimensions to groupable columns. None if unresolvable — never guesses."""
    from veda.understanding.grounding import resolve_anchor_column
    return resolve_anchor_column(concept, anchor, sm, numeric=numeric, kinds=kinds)


def _sql_lit(v) -> str:
    return str(v).replace("'", "''")


def _where_sql(gi) -> Optional[str]:
    """Grounded filters + time window → one predicate string on alias t0, or None.
    Categorical: case-insensitive equality on the arbiter's normalised value (same form
    the deterministic value-filter branch emits); several '=' values on one column → IN;
    numeric: bare comparator on a validated numeric column; time: BETWEEN on the grounded
    TEMPORAL column. Composed with AND across columns."""
    parts: List[str] = []
    by_col: Dict[str, dict] = {}
    for f in (getattr(gi, "filters", None) or []):
        if f.op in ("IS NOT NULL", "IS NULL"):
            parts.append(f't0."{f.column}" {f.op}')
            continue
        if getattr(f, "numeric", False):
            try:
                num = float(f.value)
            except (TypeError, ValueError):
                continue
            num_s = str(int(num)) if num.is_integer() else repr(num)
            parts.append(f't0."{f.column}" {f.op} {num_s}')
            continue
        slot = by_col.setdefault(f.column, {"pos": [], "neg": []})
        (slot["neg"] if f.op in ("!=", "<>") else slot["pos"]).append(_sql_lit(str(f.value).lower()))
    for col, slot in by_col.items():
        if slot["pos"]:
            vals = sorted(set(slot["pos"]))
            parts.append(f'LOWER(CAST(t0."{col}" AS TEXT)) = \'{vals[0]}\'' if len(vals) == 1
                         else f'LOWER(CAST(t0."{col}" AS TEXT)) IN ({", ".join(chr(39) + v + chr(39) for v in vals)})')
        if slot["neg"]:
            vals = sorted(set(slot["neg"]))
            parts.append(f'LOWER(CAST(t0."{col}" AS TEXT)) NOT IN ({", ".join(chr(39) + v + chr(39) for v in vals)})')
    tw = getattr(gi, "time", None) or {}
    if tw.get("column"):
        parts.append(f't0."{tw["column"]}" BETWEEN \'{_sql_lit(tw.get("start") or "1900-01-01")}\' '
                     f'AND \'{_sql_lit(tw.get("end") or "2999-12-31")}\'')
    return " AND ".join(parts) if parts else None


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
    # Phase 1 = SINGLE anchor. The LLM's extra `entities` (secondaries) do NOT veto the
    # spec any more (2026-09-16): every slot the spec uses — measure, dimensions, filters,
    # time — is grounded ON THE ANCHOR or the intent was refused upstream, so a listed
    # second entity ("accounts_paymenttype" next to "…paymenttransaction") is informational.
    # A question that truly needs a join fails anchor-grounding of its dimension/filter
    # and never reaches here (advisory clarify → the join planner decides).
    anchor = gi.anchor
    if intent == "list":
        # M2 (2026-09-16): a plain list with GROUNDED predicates ("properties with more
        # than 3 floors", "maintenance records with amount above 500") — the spec carries
        # only the anchor + where; the pipeline composes the projection it already
        # computed (no SQL is built here; emit_sql returns None for it).
        _w = _where_sql(gi)
        if not _w:
            return None
        return AnalyticalSpec(anchor=anchor, aggregation="list", output_shape="list",
                              where_sql=_w,
                              evidence={"intent": intent, "source": "analytical_spec_v2",
                                        "filters": [(f.column, f.op, f.value) for f in (gi.filters or [])],
                                        "time": getattr(gi, "time", None)})
    if intent not in _AGG:
        return None

    agg = "count" if intent == "count" else intent
    measure_col = None
    distinct = False
    if agg == "count":
        # "how many X" → COUNT(*); "how many amenity categories" → COUNT(DISTINCT category)
        # (the dimension column grounding validated on the anchor, M2).
        _dc = getattr(gi, "distinct_column", None)
        if _dc:
            measure_col, distinct = _dc, True
        else:
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

    # group keys: the GROUNDED dimensions first (understanding layer, type-checked); else
    # the query's "per/by <dimension>" resolved to a REAL groupable anchor column. A
    # grouping phrase that resolves to NOTHING → None: building a scalar instead would
    # silently answer a different question (the §5 "average payment amount broken down
    # by currency" regression, 2026-09-15).
    group_keys: List[str] = [d.column for d in (getattr(gi, "dimensions", None) or []) if getattr(d, "column", None)]
    gm = re.search(r"\b(?:per|by|broken down by|grouped by)\s+([a-z_][a-z_ ]{2,40})", query.lower())
    if not group_keys and gm:
        gcol = _resolve_column(gm.group(1).strip(), anchor, sm, kinds={"dimension"})
        if gcol is None:
            return None                      # unresolvable GROUP BY → defer, never scalar
        group_keys = [gcol]
    shape = "grouped" if group_keys else "scalar"

    # ranking (top N) — only meaningful for grouped/measure output
    _tn = re.search(r"\b(?:top|bottom|highest|lowest)\s+(\d+)\b", query.lower())
    top_n = int(_tn.group(1)) if _tn else None
    direction = "asc" if re.search(r"\b(bottom|lowest|least|fewest)\b", query.lower()) else "desc"

    return AnalyticalSpec(anchor=anchor, aggregation=agg, measure_column=measure_col,
                          distinct=distinct, group_keys=group_keys, output_shape=shape,
                          top_n=top_n, direction=direction, where_sql=_where_sql(gi),
                          evidence={"intent": intent, "source": "analytical_spec_v2",
                                    "filters": [(f.column, f.op, f.value) for f in (getattr(gi, "filters", None) or [])],
                                    "time": getattr(gi, "time", None)})


def emit_sql(spec: AnalyticalSpec, sm) -> Optional[str]:
    """AQS → SQL by REUSING the existing deterministic builder
    (planning.build_aggregate_sql, single-anchor branch) — NOT a parallel builder.
    AnalyticalSpec is an IR only; SQL construction stays in the one production builder
    (audit rule: no 7th SQL path). Returns None → caller falls back to existing path."""
    if spec is None or spec.aggregation == "list":
        return None                          # list specs are composed by the pipeline
    try:
        from veda.planning import build_aggregate_sql
    except Exception:
        return None
    sql, _tables = build_aggregate_sql(
        spec.anchor, [], sm,
        top_n=spec.top_n, direction=spec.direction,
        group_col=(spec.group_keys[0] if spec.group_keys else None),
        measure_agg=spec.aggregation, measure_column=spec.measure_column,
        distinct=spec.distinct, where_sql=spec.where_sql)
    return sql
