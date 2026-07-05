"""Task 8 — Canonical IR v2 emission for the deterministic engine (directive Phase 3).

The directive: "Preserve existing reasoning logic. Change output contract only."

So the deterministic engine keeps producing SQL exactly as before; this module derives the
join-free Canonical IR v2 FROM that SQL (sqlglot), giving the engine an IR output contract
WITHOUT rewriting every reasoning branch. The extracted IR deliberately DROPS joins (entities
survive only as table-qualified field refs) — proving the SQL's intent round-trips to the
join-free contract the compiler (Task 10) will consume.

Pure-logic: sqlglot only. No DB, no Ollama.
"""
from typing import Any, Dict, List, Optional

from veda.ir_validator import empty_ir_v2, validate_ir_v2

_OP = {
    "EQ": "=", "NEQ": "!=", "Is": "is_null", "GT": ">", "GTE": ">=",
    "LT": "<", "LTE": "<=", "Like": "like", "In": "in",
}
_AGG = {"Count": "count", "Sum": "sum", "Avg": "avg", "Min": "min", "Max": "max"}


def build_ir(anchor: str, *, projections=None, filters=None, aggregations=None,
             group_by=None, order_by=None, limit=None, temporal=None,
             confidence: float = 0.0, provenance: str = "deterministic") -> Dict[str, Any]:
    """Assemble a Canonical IR v2 dict from already-resolved intent fields."""
    ir = empty_ir_v2()
    ir.update({
        "anchor": anchor or "",
        "projections": list(projections or []),
        "filters": list(filters or []),
        "aggregations": list(aggregations or []),
        "group_by": list(group_by or []),
        "order_by": list(order_by or []),
        "limit": limit,
        "temporal": temporal,
        "confidence": float(confidence),
        "provenance": provenance,
    })
    return ir


def ir_from_sql(sql: str, anchor: Optional[str] = None,
                confidence: float = 1.0, provenance: str = "deterministic") -> Dict[str, Any]:
    """Derive join-free Canonical IR v2 from a SELECT. Raises on unparseable SQL."""
    import sqlglot
    from sqlglot import exp

    tree = sqlglot.parse_one(sql, read="postgres")
    if not isinstance(tree, exp.Select):
        raise ValueError("ir_from_sql expects a single SELECT")

    # alias → real table name (so "t1.name" resolves to "role.name")
    alias_map: Dict[str, str] = {}
    base_table = None
    for t in tree.find_all(exp.Table):
        name = t.name
        alias_map[t.alias_or_name] = name
        alias_map[name] = name
        if base_table is None:
            base_table = name
    anchor = anchor or base_table or ""

    def _qualify(col: "exp.Column") -> Optional[str]:
        tbl = alias_map.get(col.table, col.table) if col.table else anchor
        return f"{tbl}.{col.name}" if tbl and col.name else None

    projections: List[str] = []
    aggregations: List[dict] = []
    for proj in tree.expressions:
        node = proj.this if isinstance(proj, exp.Alias) else proj
        alias = proj.alias if isinstance(proj, exp.Alias) else None
        agg_name = _AGG.get(type(node).__name__)
        if agg_name:
            inner = node.this
            if isinstance(inner, exp.Column):
                field = _qualify(inner) or "*"
            elif isinstance(inner, exp.Star) or inner is None:
                field = "*"
            else:
                field = "*"
            aggregations.append({"func": agg_name, "field": field,
                                 **({"alias": alias} if alias else {})})
        elif isinstance(node, exp.Column):
            q = _qualify(node)
            if q:
                projections.append(q)

    # filters — collect every comparison node in the WHERE (robust to AND/OR nesting)
    filters: List[dict] = []
    where = tree.args.get("where")
    if where:
        _cmp = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.In)
        for pred in where.find_all(*_cmp):
            op = _OP.get(type(pred).__name__)
            left = pred.this
            right = pred.args.get("expression") or pred.args.get("query")
            if op and isinstance(left, exp.Column):
                fld = _qualify(left)
                val = right.name if isinstance(right, exp.Literal) else (
                    right.sql() if right is not None else None)
                if fld:
                    filters.append({"field": fld, "op": op, "value": val})

    group_by = []
    grp = tree.args.get("group")
    if grp:
        for g in grp.expressions:
            if isinstance(g, exp.Column):
                q = _qualify(g)
                if q:
                    group_by.append(q)

    order_by = []
    order = tree.args.get("order")
    if order:
        for o in order.expressions:
            col = o.this
            if isinstance(col, exp.Column):
                q = _qualify(col)
                if q:
                    order_by.append({"field": q, "dir": "desc" if o.args.get("desc") else "asc"})

    limit = None
    lim = tree.args.get("limit")
    if lim is not None:
        try:
            limit = int(lim.expression.name)
        except Exception:
            limit = None

    return build_ir(anchor, projections=projections, filters=filters,
                    aggregations=aggregations, group_by=group_by, order_by=order_by,
                    limit=limit, confidence=confidence, provenance=provenance)


def normalize_to_ir_v2(raw: Dict[str, Any], anchor: Optional[str] = None,
                       provenance: str = "llm") -> Dict[str, Any]:
    """Task 9 — coerce a LangGraph/SLM output dict into pure Canonical IR v2.

    The LLM engine must never author joins or SQL. This DROPS any forbidden keys
    (joins/sql/raw_sql/from/where) the model emitted, keeps only the v2 intent fields,
    and qualifies bare field names with the anchor (entities survive only as dotted refs).
    Output always satisfies validate_ir_v2 or the caller can inspect the returned errors.
    """
    raw = raw or {}
    anchor = anchor or raw.get("anchor") or ""

    def _q(field):
        if not isinstance(field, str) or not field:
            return None
        return field if "." in field else (f"{anchor}.{field}" if anchor else None)

    projections = [p for p in (_q(x) for x in (raw.get("projections") or [])) if p]

    filters = []
    for f in (raw.get("filters") or []):
        if isinstance(f, dict) and f.get("field"):
            fld = _q(f["field"])
            if fld:
                filters.append({"field": fld, "op": str(f.get("op", "=")).lower(),
                                "value": f.get("value")})

    aggregations = []
    for a in (raw.get("aggregations") or []):
        if isinstance(a, dict) and a.get("func"):
            fld = a.get("field")
            fld = "*" if fld in ("*", None) else (_q(fld) or "*")
            agg = {"func": str(a["func"]).lower(), "field": fld}
            if a.get("alias"):
                agg["alias"] = a["alias"]
            aggregations.append(agg)

    group_by = [g for g in (_q(x) for x in (raw.get("group_by") or [])) if g]

    return build_ir(
        anchor,
        projections=projections,
        filters=filters,
        aggregations=aggregations,
        group_by=group_by,
        order_by=raw.get("order_by") or [],
        limit=raw.get("limit"),
        temporal=raw.get("temporal"),
        confidence=float(raw.get("confidence", 0.0) or 0.0),
        provenance=provenance,
    )
    # NB: joins / sql / raw_sql / from / where keys in `raw` are intentionally NOT copied.


def safe_ir_from_sql(sql, **kw):
    """Best-effort: returns (ir, errors). Never raises — for additive pipeline use."""
    try:
        ir = ir_from_sql(sql, **kw)
        return ir, validate_ir_v2(ir)
    except Exception as e:
        return None, [f"{type(e).__name__}: {e}"]
