"""Task 2 — Canonical IR v2 validator (ARCHITECTURE_HYBRID.md §1, directive Phase 1).

The IR is the contract between reasoning (deterministic + LLM engines) and the compiler.
This validator enforces the load-bearing invariant of the whole architecture:

    Join authorship and SQL authorship are structurally IMPOSSIBLE above the compiler.

A reasoning engine may only describe INTENT (anchor, projections, filters, aggregations,
grouping, ordering, limit, temporal). It may never describe STRUCTURE (joins) or OUTPUT
(SQL). Those belong solely to the graph compiler (Task 10).

Usage:
    errs = validate_ir_v2(ir)         # [] → valid; list of strings → reasons it's invalid
    if not is_valid_ir_v2(ir): ...
"""
from typing import Any, Dict, List

# Keys whose mere PRESENCE means a reasoning layer tried to author structure or SQL.
# This is the firewall against the #1 hallucination surface — it is not negotiable.
FORBIDDEN_KEYS = frozenset({
    "joins", "join", "join_path",          # joins are inferred by the compiler, never authored
    "sql", "raw_sql", "sql_query", "query",  # SQL is a compiler artifact, never a reasoning artifact
    "from", "where",                        # SQL clause fragments
})

REQUIRED_KEYS = frozenset({
    "anchor", "projections", "filters", "aggregations",
    "group_by", "order_by", "limit", "temporal",
})

_ALLOWED_OPS = frozenset({"=", "!=", "<", "<=", ">", ">=", "in", "not_in",
                          "between", "like", "is_null", "is_not_null"})
_ALLOWED_AGG = frozenset({"count", "sum", "avg", "min", "max"})


def _is_dotted(field: Any) -> bool:
    """Entities may appear ONLY as table-qualified field names (table.column)."""
    return isinstance(field, str) and field.count(".") == 1 and all(field.split("."))


def validate_ir_v2(ir: Any) -> List[str]:
    """Return a list of reasons `ir` is not valid Canonical IR v2 ([] == valid)."""
    errs: List[str] = []

    if not isinstance(ir, dict):
        return [f"IR must be a dict, got {type(ir).__name__}"]

    # ── 1. Structural impossibility — forbidden keys (the core invariant) ──────
    for k in ir:
        if k.lower() in FORBIDDEN_KEYS:
            errs.append(f"forbidden key '{k}': joins/SQL may not be authored above the compiler")

    # ── 2. Required shape ──────────────────────────────────────────────────────
    for k in REQUIRED_KEYS:
        if k not in ir:
            errs.append(f"missing required key '{k}'")

    # ── 3. Field-level contract ────────────────────────────────────────────────
    anchor = ir.get("anchor")
    if "anchor" in ir and (not isinstance(anchor, str) or not anchor.strip()):
        errs.append("anchor must be a non-empty string (the query subject)")

    projections = ir.get("projections")
    if "projections" in ir:
        if not isinstance(projections, list):
            errs.append("projections must be a list")
        else:
            for p in projections:
                if not _is_dotted(p):
                    errs.append(f"projection '{p}' must be table-qualified (table.column)")

    filters = ir.get("filters")
    if "filters" in ir:
        if not isinstance(filters, list):
            errs.append("filters must be a list")
        else:
            for f in filters:
                if not isinstance(f, dict):
                    errs.append(f"filter must be an object, got {type(f).__name__}")
                    continue
                if not _is_dotted(f.get("field")):
                    errs.append(f"filter.field '{f.get('field')}' must be table-qualified")
                op = str(f.get("op", "")).lower()
                if op not in _ALLOWED_OPS:
                    errs.append(f"filter.op '{f.get('op')}' not in allowed operators")

    aggs = ir.get("aggregations")
    if "aggregations" in ir:
        if not isinstance(aggs, list):
            errs.append("aggregations must be a list")
        else:
            for a in aggs:
                if not isinstance(a, dict):
                    errs.append(f"aggregation must be an object, got {type(a).__name__}")
                    continue
                if str(a.get("func", "")).lower() not in _ALLOWED_AGG:
                    errs.append(f"aggregation.func '{a.get('func')}' not in {sorted(_ALLOWED_AGG)}")
                fld = a.get("field")
                # '*' is allowed only for COUNT
                if fld not in ("*", None) and not _is_dotted(fld):
                    errs.append(f"aggregation.field '{fld}' must be table-qualified or '*'")

    for listkey in ("group_by", "order_by"):
        if listkey in ir and not isinstance(ir.get(listkey), list):
            errs.append(f"{listkey} must be a list")

    if "limit" in ir and ir["limit"] is not None and not isinstance(ir["limit"], int):
        errs.append("limit must be an int or null")

    if "temporal" in ir and ir["temporal"] is not None and not isinstance(ir["temporal"], dict):
        errs.append("temporal must be an object or null")

    return errs


def is_valid_ir_v2(ir: Any) -> bool:
    return not validate_ir_v2(ir)


def empty_ir_v2() -> Dict[str, Any]:
    """The canonical empty IR v2 (the exact shape engines must emit)."""
    return {
        "anchor": "",
        "projections": [],
        "filters": [],
        "aggregations": [],
        "group_by": [],
        "order_by": [],
        "limit": None,
        "temporal": None,
        "confidence": 0.0,
        "provenance": "",
    }
