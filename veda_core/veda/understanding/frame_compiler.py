"""veda.understanding.frame_compiler — Stage 5: GroundedFrame → QueryIR → SQL.

ONE compiler for frame-path answers. It never re-reads the question: every clause comes
from a grounded slot, so the IR it returns is COMPLETE (`ir_partial=False`, head
"frame…") and the firewall checks the SQL against it structurally.

Builders:
  * scalar aggregates (COUNT / COUNT DISTINCT / SUM / AVG / MIN / MAX, no grouping) go
    through the existing production builder `planning.build_aggregate_sql` (single-anchor
    mode, `where_sql`), exactly as analytical_spec.emit_sql does;
  * lists and grouped aggregates are a single-anchor SELECT with LEFT JOINs to FK
    parents only (N:1 — never fans out rows), built here, deterministic.
A slot the builders cannot express makes compile() DECLINE with that slot named — the
caller turns it into a clarify; it never emits a bare projection or a bare LIMIT instead.

Literals are written inline (categorical ones through LOWER(CAST(… AS TEXT)), the value
arbiter's form); `validation.validate_and_parameterize` parameterises every one of them
before execution, as it does for every head.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from veda.ir import QueryIR, IRFilter, IRMeasure
from veda.understanding.frame_grounding import GroundedFrame, _bare


@dataclass
class Compiled:
    sql: str
    ir: QueryIR
    tables: List[str]
    columns: List[str]
    head: str = "frame"
    notes: List[str] = field(default_factory=list)


@dataclass
class Declined:
    slot: str
    reason: str


def _q(ident: str) -> str:
    return '"' + str(ident).replace('"', '""') + '"'


def _lit(v) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def _num(v) -> str:
    f = float(v)
    return str(int(f)) if f.is_integer() else repr(f)


def _alias_map(gf: GroundedFrame) -> Dict[str, str]:
    m = {gf.anchor: "t0"}
    for i, j in enumerate(gf.joins, start=1):
        m.setdefault(j.table, f"t{i}")
    return m


def predicate(f, alias: str) -> Optional[str]:
    """One grounded filter → SQL predicate on `alias` (None = not expressible)."""
    col = f"{alias}.{_q(f.column)}"
    op = f.op
    if op in ("IS NULL", "IS NOT NULL"):
        return f"{col} {op}"
    if f.grounding == "numeric":
        if op == "between":
            lo, hi = f.value
            return f"{col} BETWEEN {_num(lo)} AND {_num(hi)}"
        if op in (">", ">=", "<", "<=", "=", "!="):
            return f"{col} {op} {_num(f.value)}"
        return None
    if f.grounding == "temporal":
        if op == "between" and isinstance(f.value, (list, tuple)) and len(f.value) == 2:
            return f"{col} BETWEEN {_lit(f.value[0])} AND {_lit(f.value[1])}"
        if op in (">", ">=", "<", "<="):
            return f"{col} {op} {_lit(f.value)}"
        return None
    low = f"LOWER(CAST({col} AS TEXT))"
    if op == "IN":
        vals = sorted({str(v).lower() for v in (f.value or [])})
        return f"{low} IN ({', '.join(_lit(v) for v in vals)})" if vals else None
    if op in ("=", "!="):
        return f"{low} {op} {_lit(str(f.value).lower())}"
    return None


def where_parts(gf: GroundedFrame, am: Dict[str, str]) -> Tuple[List[str], Optional[Declined]]:
    parts = []
    for f in gf.filters:
        p = predicate(f, am.get(f.table, "t0"))
        if p is None:
            return [], Declined(f"filter:{f.column}", f"operator {f.op} on {f.column} is not expressible")
        parts.append(p)
    if gf.time:
        a = am.get(gf.time["table"], "t0")
        col = f"{a}.{_q(gf.time['column'])}"
        if gf.time.get("start"):
            parts.append(f"{col} >= {_lit(gf.time['start'])}")
        if gf.time.get("end"):
            parts.append(f"{col} <= {_lit(gf.time['end'])}")
    return parts, None


def to_ir(gf: GroundedFrame, source_scope=None) -> QueryIR:
    filters = [IRFilter(table=f.table, column=f.column,
                        op=("BETWEEN" if f.op == "between" else f.op),
                        value=f.value, grounding=f.grounding, concept=f.concept) for f in gf.filters]
    measure = None
    if gf.measure:
        agg, col = gf.measure
        measure = IRMeasure(aggregation=agg, column=col, table=gf.anchor, distinct=bool(gf.distinct and col))
    order = None
    if gf.order:
        order = {"column": gf.order[1], "table": gf.order[0], "direction": gf.order[2]}
    tw = None
    if gf.time:
        tw = {"column": gf.time["column"], "start": gf.time.get("start"), "end": gf.time.get("end")}
    gm = {"anchor": gf.anchor_method}
    for f in gf.filters:
        gm[f"filter:{f.column}"] = f.grounding
    return QueryIR(anchor=gf.anchor, secondaries=[j.table for j in gf.joins], measure=measure,
                   filters=filters, group_keys=[c for _t, c in gf.group_by], time_window=tw,
                   order=order, limit=gf.limit, distinct=bool(gf.distinct),
                   source_scope=list(source_scope or []), grounding_method=gm,
                   confidence=float(gf.confidence or 0.0), head=f"frame.{gf.anchor_method.lower()}")


def compile_frame(gf: GroundedFrame, sm, source_scope=None):
    """GroundedFrame → Compiled | Declined."""
    am = _alias_map(gf)
    parts, dec = where_parts(gf, am)
    if dec:
        return dec
    ir = to_ir(gf, source_scope)
    joins_sql = " ".join(
        f'LEFT JOIN {_q(_bare(j.table))} {am[j.table]} ON {am[j.table]}.{_q(j.pk_column)} = t0.{_q(j.fk_column)}'
        for j in gf.joins)
    frm = f'FROM {_q(_bare(gf.anchor))} t0' + (f" {joins_sql}" if joins_sql else "")
    where = (" WHERE " + " AND ".join(parts)) if parts else ""
    tables = [gf.anchor] + [j.table for j in gf.joins]

    # ── scalar aggregate: the production builder ──
    if gf.measure and not gf.group_by:
        agg, col = gf.measure
        if gf.joins:
            return Declined("measure", "an aggregate over a joined table is not expressible here")
        try:
            from veda.planning import build_aggregate_sql
            sql, _t = build_aggregate_sql(_bare(gf.anchor), [], sm, measure_agg=agg,
                                          measure_column=col, distinct=bool(gf.distinct and col),
                                          where_sql=(" AND ".join(parts) or None))
        except Exception as e:
            return Declined("measure", f"aggregate builder failed: {type(e).__name__}")
        if not sql:
            return Declined("measure", "aggregate builder returned nothing")
        if gf.order or gf.limit:
            # an ordering / row count on a single scalar is meaningless — say so
            if gf.limit and gf.limit > 1:
                return Declined("limit", "a single total cannot be limited to several rows")
        return Compiled(sql=sql, ir=ir, tables=tables, columns=[c for c in [col] if c])

    # ── grouped aggregate ──
    if gf.group_by:
        agg, col = gf.measure or ("count", None)
        gcols = [f"{am.get(t, 't0')}.{_q(c)}" for t, c in gf.group_by]
        if agg == "count":
            aexpr = f"COUNT(DISTINCT t0.{_q(col)})" if (col and gf.distinct) else "COUNT(*)"
            alias = "count"
        else:
            aexpr = f"{agg.upper()}(t0.{_q(col)})"
            alias = f"{agg}_{col}"
        sel = ", ".join(gcols + [f"{aexpr} AS {_q(alias)}"])
        if gf.order:
            ot, oc, od = gf.order
            if (ot, oc) in gf.group_by:
                oexpr = f"{am.get(ot, 't0')}.{_q(oc)}"
            elif oc == col or oc == alias:
                oexpr = _q(alias)
                ir.order = {"column": alias, "alias": alias, "table": None, "direction": od}
            elif agg == "count":
                # a grouped COUNT ordered by some other column ('grouped by currency in
                # descending order' + an order concept the SLM tied to price): the only
                # order the answer's rows can carry is the group key
                gt0, gc0 = gf.group_by[0]
                oexpr = f"{am.get(gt0, 't0')}.{_q(gc0)}"
                ir.order = {"column": gc0, "table": gt0, "direction": od}
                gf.order = (gt0, gc0, od)
            else:
                return Declined("order", f"ordering a grouped answer by {oc} is not expressible")
            order_sql = f" ORDER BY {oexpr} {od.upper()}"
        else:
            order_sql = f" ORDER BY {_q(alias)} DESC"
        # No LIMIT unless the frame named one explicitly — the executor's
        # EXECUTION_RESULT_LIMIT caps rows for every head; a compiler-side default
        # silently truncated grouped answers below that cap (§10.7).
        limit_sql = f" LIMIT {int(gf.limit)}" if gf.limit else ""
        sql = f"SELECT {sel} {frm}{where} GROUP BY {', '.join(gcols)}{order_sql}{limit_sql}"
        return Compiled(sql=sql, ir=ir, tables=tables,
                        columns=[c for _t, c in gf.group_by] + ([col] if col else []))

    # ── list ──
    proj = gf.projection or [(gf.anchor, "id")]
    seen, sel = set(), []
    for t, c in proj:
        a = am.get(t)
        if a is None:
            continue
        label = c if t == gf.anchor else f"{_bare(t).split('_')[-1]}_{c}"
        if label in seen:
            continue
        seen.add(label)
        sel.append(f"{a}.{_q(c)}" + (f" AS {_q(label)}" if label != c else ""))
    distinct = "DISTINCT " if gf.distinct else ""
    order_sql = ""
    if gf.order:
        ot, oc, od = gf.order
        a = am.get(ot, "t0")
        tie = f", t0.{_q('id')} {od.upper()}" if not (ot == gf.anchor and oc == "id") and "id" in {c for _t, c in proj} and not gf.distinct else ""
        order_sql = f" ORDER BY {a}.{_q(oc)} {od.upper()}{tie}"
    # No LIMIT unless the frame named one explicitly (see the grouped branch above).
    limit_sql = f" LIMIT {int(gf.limit)}" if gf.limit else ""
    sql = f"SELECT {distinct}{', '.join(sel)} {frm}{where}{order_sql}{limit_sql}"
    if gf.limit is None:
        ir.limit = None          # no limit was asked for; the executor caps rows
    return Compiled(sql=sql, ir=ir, tables=tables, columns=[c for _t, c in proj])
