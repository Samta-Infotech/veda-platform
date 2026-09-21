"""veda.ir — the canonical QueryIR (M3 checkpoint 1, 2026-09-16).

ONE structured description of the question a head is about to answer, built by every head
right before it executes SQL, and checked against that SQL by veda.firewall. It is the
convergence point of the three representations that already exist:

    understanding.GroundedIntent   (LLM extraction → deterministic grounding)
    analytical_spec.AnalyticalSpec (single-anchor aggregate spec)
    query.intent.QueryIntent       (fast path / Tier-2 envelope)

plus the deterministic branches' own state in veda/pipeline.py (_arb_filters, _tpred,
_rank, grouping/aggregate modes) via `from_branch_state`.

Completeness is explicit: a head that emitted SQL without structured state (the LLM
`generate_sql` branch, Tier-2's IR→sql_builder path, verified-cache replay) produces a
PARTIAL IR — `ir_partial=True`, the slots it does know filled, the rest None — and the
firewall falls back to the existing text heuristics for the unknown slots. `partial_slots`
names which. Checkpoint 2 (compile-from-IR) makes every head complete; the battery's
`ir_partial` count is how we watch that happen.

Nothing here reads the DB or the model — pure data + adapters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

IR_VERSION = 1

# every slot the firewall can check; a partial IR lists the ones it could NOT fill
SLOTS = ("anchor", "measure", "filters", "group_keys", "time_window", "order", "limit", "distinct")


@dataclass
class IRFilter:
    table: Optional[str]
    column: str
    op: str                          # = != > >= < <= IN NOT IN BETWEEN IS NULL IS NOT NULL
    value: Any = None                # normalised literal / number / [values] / (start, end)
    grounding: str = "unknown"       # value_arbiter | numeric | temporal | fk | branch | llm | unknown
    concept: str = ""


@dataclass
class IRMeasure:
    aggregation: str                 # count | sum | avg | min | max | none
    column: Optional[str] = None     # None ⇒ COUNT(*)
    table: Optional[str] = None
    distinct: bool = False


@dataclass
class QueryIR:
    anchor: Optional[str]
    secondaries: List[str] = field(default_factory=list)
    measure: Optional[IRMeasure] = None
    filters: List[IRFilter] = field(default_factory=list)
    group_keys: List[str] = field(default_factory=list)        # "table.col" or bare col on anchor
    time_window: Optional[Dict[str, Any]] = None               # {"column","start","end"}
    order: Optional[Dict[str, Any]] = None                     # {"column","direction"}
    limit: Optional[int] = None
    distinct: bool = False
    source_scope: List[str] = field(default_factory=list)      # source ids the SQL may touch
    grounding_method: Dict[str, str] = field(default_factory=dict)   # slot → method
    confidence: float = 1.0
    head: str = "unknown"                                      # which head built it
    ir_partial: bool = False
    partial_slots: List[str] = field(default_factory=list)
    version: int = IR_VERSION

    # ── helpers the firewall reads ─────────────────────────────────────────
    def knows(self, slot: str) -> bool:
        return not (self.ir_partial and slot in self.partial_slots)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def partial(head: str, anchor: Optional[str] = None, *, known: Optional[Dict[str, Any]] = None,
            source_scope: Optional[List[str]] = None) -> QueryIR:
    """A PARTIAL IR: the slots in `known` are trusted, every other slot is unknown and
    the firewall keeps its text heuristic for it."""
    known = dict(known or {})
    ir = QueryIR(anchor=anchor, head=head, ir_partial=True,
                 source_scope=list(source_scope or []))
    for k, v in known.items():
        setattr(ir, k, v)
    ir.partial_slots = [s for s in SLOTS if s not in known and not (s == "anchor" and anchor)]
    return ir


# ── adapters ──────────────────────────────────────────────────────────────────
def from_grounded_intent(gi, spec=None, head: str = "understanding", source_scope=None) -> QueryIR:
    """GroundedIntent (+ the AnalyticalSpec derived from it, when any) → complete IR."""
    filters = [IRFilter(table=f.table, column=f.column, op=f.op, value=f.value,
                        grounding=("numeric" if getattr(f, "numeric", False)
                                   else "temporal" if f.op in ("IS NULL", "IS NOT NULL") and False
                                   else "value_arbiter"),
                        concept=getattr(f, "concept", ""))
               for f in (getattr(gi, "filters", None) or [])]
    tw = getattr(gi, "time", None)
    measure = None
    agg = getattr(spec, "aggregation", None) if spec is not None else None
    if agg and agg != "list":
        measure = IRMeasure(aggregation=agg, column=getattr(spec, "measure_column", None),
                            table=gi.anchor, distinct=bool(getattr(spec, "distinct", False)))
    elif getattr(gi, "measure", None) is not None:
        measure = IRMeasure(aggregation=gi.measure.kind, column=gi.measure.column,
                            table=gi.measure.table or gi.anchor,
                            distinct=bool(getattr(gi, "distinct_column", None)))
    group_keys = list(getattr(spec, "group_keys", None) or []) if spec is not None else \
        [d.column for d in (getattr(gi, "dimensions", None) or [])]
    order = None
    if spec is not None and getattr(spec, "top_n", None):
        order = {"column": None, "direction": getattr(spec, "direction", "desc")}
    gm = {"anchor": getattr(gi, "anchor_method", "retrieval")}
    for f in filters:
        gm[f"filter:{f.column}"] = f.grounding
    for g in group_keys:
        gm[f"group:{g}"] = "type_checked"
    return QueryIR(anchor=gi.anchor, secondaries=list(getattr(gi, "secondaries", None) or []),
                   measure=measure, filters=filters, group_keys=group_keys,
                   time_window=tw, order=order,
                   limit=(getattr(spec, "top_n", None) if spec is not None else None),
                   distinct=bool(getattr(spec, "distinct", False)) if spec is not None else False,
                   source_scope=list(source_scope or []), grounding_method=gm,
                   confidence=float(getattr(gi, "confidence", 1.0) or 1.0), head=head)


def from_query_intent(qi, head: str = "fast_path", source_scope=None) -> QueryIR:
    """query.intent.QueryIntent (fast path / envelope) → IR. Filters carry the intent's
    own op; a BETWEEN filter is the time window."""
    filters, tw = [], None
    for f in (getattr(qi, "filters", None) or []):
        op = getattr(f, "op", "=")
        vals = list(getattr(f, "values", None) or [])
        if op == "BETWEEN" and len(vals) == 2:
            tw = {"column": f.col, "start": vals[0], "end": vals[1]}
            continue
        filters.append(IRFilter(table=getattr(qi, "subject_table", None), column=f.col, op=op,
                                value=(vals[0] if len(vals) == 1 else vals), grounding="registry"))
    qt = getattr(qi, "query_type", "") or ""
    agg = {"count": "count", "measure": (getattr(qi, "metric_id", "") or "").split("_")[0] or "sum",
           "ratio": "ratio", "trend": "count", "compare": "count", "dimension_list": "none",
           "list": "none"}.get(qt, "none")
    sel = getattr(qi, "select_expr", "") or ""
    m = re.match(r"\s*(SUM|AVG|MIN|MAX|COUNT)\s*\(\s*(?:DISTINCT\s+)?\"?([A-Za-z0-9_]+)\"?", sel, re.I)
    if m:
        agg = m.group(1).lower()
        mcol = None if m.group(2) in ("*", "id") and agg == "count" else m.group(2)
    else:
        mcol = None
    measure = None if agg == "none" else IRMeasure(aggregation=agg, column=mcol,
                                                   table=getattr(qi, "subject_table", None))
    gk = [getattr(qi, "group_col")] if getattr(qi, "group_col", None) else []
    if getattr(qi, "time_bucket", None) and getattr(qi, "time_col", None):
        gk = gk + [f"{qi.time_bucket}({qi.time_col})"]
    # dimension_list ("list all users" → SELECT DISTINCT first_name … ORDER BY …): the
    # intent's group_col is the LISTED column, not a GROUP BY — leave group_keys empty
    # for it, and let DISTINCT be checked as the shape (2026-09-18: the first battery run
    # refused "list all users" as shape_mismatch because the IR carried group_col as a
    # group key the SQL had no GROUP BY for).
    if qt == "dimension_list":
        gk = []
    return QueryIR(anchor=getattr(qi, "subject_table", None), measure=measure, filters=filters,
                   group_keys=gk, time_window=tw,
                   order=({"column": getattr(qi, "order_col", None), "direction": getattr(qi, "order_dir", "desc")}
                          if getattr(qi, "order_col", None) else None),
                   limit=getattr(qi, "limit", None), distinct=(qt == "dimension_list"),
                   source_scope=list(source_scope or []),
                   grounding_method={"anchor": "registry"}, head=head)


# sqlglot predicate class name → SQL operator, for from_sql_facts below.
_SQLGLOT_OP = {"EQ": "=", "NEQ": "!=", "GT": ">", "GTE": ">=", "LT": "<", "LTE": "<=",
               "In": "IN", "Like": "LIKE", "ILike": "LIKE", "Is": "IS NULL",
               "Between": "BETWEEN", "NotIn": "NOT IN"}


def from_sql_facts(facts: Dict[str, Any], head: str, *, source_scope=None) -> QueryIR:
    """The EXECUTED SQL's own AST facts (business_explain.extract_sql_facts) → IR.

    For heads that never built structured state — Tier-2, the LLM-SQL branch, a cache
    replay — this reverse-engineers the IR from the statement that actually ran. It is
    used ONLY for session memory (the api tier stacks it and applies the next turn's
    delta to it); it is never handed to the firewall, which would be circular — checking
    SQL against an IR derived from that same SQL proves nothing.

    Always `ir_partial=True`. The structural slots are real (they came from the AST), but
    the per-slot GROUNDING provenance is not recoverable from SQL: this cannot distinguish
    a filter value the value-arbiter grounded from one an LLM wrote. Marking it partial is
    what makes the chat tier fall back to the text restatement for these frames rather
    than trusting slots whose origin is unknown — and the partial COUNT is the measure of
    how much of the stack is genuinely structured (see this module's docstring).
    """
    ents = list(facts.get("entities") or [])
    anchor = ents[0] if ents else None
    filters = []
    for f in (facts.get("filters") or []):
        try:
            col, kind, val = f
        except (TypeError, ValueError):
            continue
        filters.append(IRFilter(table=anchor, column=col,
                                op=_SQLGLOT_OP.get(str(kind), "="), value=val,
                                grounding="sql_ast"))
    measure = None
    for agg in (facts.get("aggregations") or []):
        try:
            fn, col = agg
        except (TypeError, ValueError):
            continue
        measure = IRMeasure(aggregation=str(fn).lower(), column=col, table=anchor,
                            distinct=bool(facts.get("distinct")))
        break                      # the first aggregate is the headline measure
    orderings = list(facts.get("orderings") or [])
    order = None
    if orderings:
        _c, _desc = orderings[0]
        order = {"column": _c, "direction": "desc" if _desc else "asc"}
    ir = QueryIR(anchor=anchor, secondaries=ents[1:], measure=measure, filters=filters,
                 group_keys=list(facts.get("groupings") or []), order=order,
                 limit=facts.get("limit"), distinct=bool(facts.get("distinct")),
                 source_scope=list(source_scope or []),
                 grounding_method={"anchor": "sql_ast"}, head=head)
    ir.ir_partial = True
    ir.partial_slots = list(SLOTS)
    return ir


def from_branch_state(head: str, primary: str, *, arb_filters=None, tpred_col=None, tf=None,
                      agg: Optional[str] = None, measure_col: Optional[str] = None,
                      group_col: Optional[str] = None, rank=None, rank_col: Optional[str] = None,
                      distinct: bool = False, extra_filters=None, source_scope=None,
                      complete: bool = True) -> QueryIR:
    """The deterministic branches' own state → IR. `complete=False` marks a branch that
    knows its anchor and filters but not whether it aggregates/groups (the firewall keeps
    the text heuristics for those slots)."""
    filters = [IRFilter(table=primary, column=f["column"], op=f.get("op", "="),
                        value=f.get("value_norm", f.get("value")), grounding="value_arbiter")
               for f in (arb_filters or [])]
    for ef in (extra_filters or []):
        filters.append(IRFilter(table=primary, **ef))
    tw = None
    if tpred_col and tf is not None and (getattr(tf, "start", None) or getattr(tf, "end", None)):
        tw = {"column": tpred_col, "start": getattr(tf, "start", None), "end": getattr(tf, "end", None)}
    measure = IRMeasure(aggregation=agg, column=measure_col, table=primary, distinct=distinct) if agg else None
    order = None
    limit = None
    if rank is not None and getattr(rank, "top_n", None) is not None:
        limit = int(rank.top_n)
        order = {"column": rank_col, "direction": getattr(rank, "direction", "desc")}
    ir = QueryIR(anchor=primary, measure=measure, filters=filters,
                 group_keys=[group_col] if group_col else [], time_window=tw, order=order,
                 limit=limit, distinct=distinct, source_scope=list(source_scope or []),
                 grounding_method={"anchor": "router", **{f"filter:{f.column}": f.grounding for f in filters}},
                 head=head)
    if not complete:
        ir.ir_partial = True
        ir.partial_slots = [s for s in ("measure", "group_keys", "order", "limit", "distinct")
                            if getattr(ir, s) in (None, [], False)]
    return ir
