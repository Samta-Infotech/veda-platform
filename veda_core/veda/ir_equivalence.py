# =============================================================================
# veda/ir_equivalence.py
# IR Equivalence Validation — refuse SQL that introduces semantics the user never
# requested (the "how many workflow state" → WHERE is_final=… class of failure).
#
# We do NOT diff a fragile hand-parsed QueryIR against the SQL IR (a too-thin
# QueryIR false-rejects legitimate mappings like "active workflows" → is_active).
# Instead we extract the SQL IR (sqlglot) and check every introduced element is
# LICENSED by the query: its column (name/alias) or its value is named by the user,
# or it is a SYSTEM predicate (soft-delete / polymorphic discriminator / temporal
# binding). Licensing reuses qualifier_completeness's content-token model, so the
# two gates agree on what counts as "the user asked for it".
#
# Only runs on LLM-GENERATED SQL (single-table generate_sql, skeleton-fill
# generate_join_sql). The deterministic builders (existence/aggregate) are trusted.
# =============================================================================

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SqlIR:
    entities:     List[str]              = field(default_factory=list)
    filters:      List[Tuple[str, str, Optional[str]]] = field(default_factory=list)  # (col, op, value)
    aggregations: List[str]              = field(default_factory=list)
    groupings:    List[str]              = field(default_factory=list)
    orderings:    List[str]              = field(default_factory=list)
    ordering_agg: List[bool]             = field(default_factory=list)  # per ordering: key is an aggregate expr
    agg_aliases:  set                    = field(default_factory=set)   # SELECT aliases of aggregate exprs
    distinct:     bool                   = False


def extract_sql_ir(sql: str) -> SqlIR:
    """Parse generated SQL into a comparable IR. Predicates inside EXISTS/subqueries
    are skipped — in VEDA those are deterministic (the planner's semi-join), not LLM
    output, so they can't be hallucinated."""
    import sqlglot
    from sqlglot import exp
    ir = SqlIR()
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return ir
    if tree is None:
        return ir

    ir.entities = sorted({t.name for t in tree.find_all(exp.Table) if t.name})
    ir.distinct = tree.find(exp.Distinct) is not None
    ir.aggregations = sorted({a.key.upper() for a in tree.find_all(exp.AggFunc)})

    # SELECT aliases whose expression is an aggregate (SUM(x) AS total_amount) — an
    # ORDER BY on one of these re-sorts a grouped result by its own measure; it can
    # never widen/narrow the answer set. Collected so Rule 4 can license that case.
    for a in tree.find_all(exp.Alias):
        if a.alias and a.find(exp.AggFunc) is not None:
            ir.agg_aliases.add(a.alias)

    grp = tree.find(exp.Group)
    if grp is not None:
        for e in grp.expressions:
            c = e.find(exp.Column)
            if c is not None:
                ir.groupings.append(c.name)
    order = tree.find(exp.Order)
    if order is not None:
        for e in order.expressions:
            c = e.find(exp.Column)
            if c is not None:
                ir.orderings.append(c.name)
                ir.ordering_agg.append(e.find(exp.AggFunc) is not None)

    def _in_subquery(node) -> bool:
        p = node.parent
        while p is not None:
            if isinstance(p, (exp.Exists, exp.Subquery)):
                return True
            p = p.parent
        return False

    where = tree.find(exp.Where)
    if where is not None:
        ops = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.In, exp.Is)
        for pred in where.find_all(ops):
            if _in_subquery(pred):
                continue
            col = pred.find(exp.Column)
            if col is None:
                continue
            # the non-column operand → the literal/value
            lit = pred.find(exp.Literal)
            val = None
            if lit is not None:
                val = lit.name
            else:
                b = pred.find(exp.Boolean)
                if b is not None:
                    val = str(b.this)
                else:
                    other = [a for a in (pred.this, getattr(pred, "expression", None))
                             if a is not None and a is not col]
                    val = other[0].sql() if other else None
            ir.filters.append((col.name, type(pred).__name__, val))
    return ir


# ---------------------------------------------------------------------------
# Licensing — is each SQL element backed by the query?
# ---------------------------------------------------------------------------
def _content_tokens(query: str):
    from veda.validation import _gate_strip
    from retrieval.query_enrichment import _singularize
    gate = _gate_strip()
    return {_singularize(w) for w in re.findall(r"[a-z]+", query.lower())
            if len(w) > 2 and w not in gate and _singularize(w) not in gate}


def _toks(name: str):
    from retrieval.query_enrichment import _singularize
    return {_singularize(t) for t in re.split(r"[_\s]+", name.lower()) if len(t) > 2}


def _named(content: set, target_toks: set) -> bool:
    """A content token names a target if it matches a target token exactly (singular)
    or by ≥4-char substring (absorbs morphology: active↔is_active, flagged↔flag)."""
    for c in content:
        for t in target_toks:
            if c == t or (len(c) >= 4 and (c in t or t in c)):
                return True
    return False


def _col_tokens(col: str, cols_meta: dict, allowed_tables) -> set:
    """Name tokens + business aliases of a column, across the in-scope tables."""
    toks = _toks(col)
    tabs = allowed_tables or set()
    for key, meta in cols_meta.items():
        t, c = key.split(".", 1) if "." in key else ("", key)
        if c == col and (not tabs or t in tabs):
            for a in (meta.get("aliases") or []):
                toks |= _toks(a)
            for f in ("business_role", "business_domain"):
                if meta.get(f):
                    toks |= _toks(str(meta[f]))
    return toks


_SOFT_DELETE = re.compile(r"(^|_)(deleted|is_deleted)(_|$)", re.I)


def _filter_licensed(col, val, content, cols_meta, allowed_tables, temporal_cols) -> bool:
    cl = col.lower()
    if _SOFT_DELETE.search(cl) or cl in temporal_cols:
        return True                                   # system / temporal binding
    if _named(content, _col_tokens(col, cols_meta, allowed_tables)):
        return True                                   # the column concept is named
    if val is not None:                               # the VALUE is named ("open orders")
        vtoks = {w for w in re.findall(r"[a-z]+", str(val).lower()) if len(w) > 2}
        if vtoks & content:
            return True
    return False


def _has(query: str, pat: str) -> bool:
    return re.search(pat, query.lower()) is not None


def validate_ir_equivalence(query, sql, sm, *, allowed_tables=None,
                            skip_predicate_cols=None, temporal_cols=None,
                            llm_generated=True):
    """Returns (ok, violations). Only enforces on LLM-generated SQL. Conservative:
    flags an element only when it is CLEARLY unlicensed by the query."""
    try:
        from config import IR_EQUIVALENCE_ENABLED
    except Exception:
        IR_EQUIVALENCE_ENABLED = False
    if not IR_EQUIVALENCE_ENABLED or not llm_generated:
        return True, []

    ir = extract_sql_ir(sql)
    content = _content_tokens(query)
    # Entity/table words name WHICH table, not WHICH predicate — strip them so naming
    # the table ("workflow state") can't license a filter on one of its columns
    # ("is_final"). Only DISTINCTIVE concept/value tokens may license a predicate.
    tabtoks = set()
    for t in (sm.get("tables", {}) if sm else {}):
        tabtoks |= _toks(t)
    content = content - tabtoks
    cols_meta = sm.get("columns", {}) if sm else {}
    skip = {c.lower() for c in (skip_predicate_cols or set())}
    tcols = {c.lower() for c in (temporal_cols or set())}
    v: List[str] = []

    # Rule 1 — no extra filters (the demonstrated failure class)
    for col, op, val in ir.filters:
        if col.lower() in skip:
            continue
        if not _filter_licensed(col, val, content, cols_meta, allowed_tables, tcols):
            v.append(f"filter {col}={val!r} not requested by the query")

    # Rule 3 — no extra grouping
    if ir.groupings and not _has(query, r"\b(per|by|each|every|group|breakdown|split)\b"):
        v.append(f"GROUP BY {ir.groupings} not requested")

    # Rule 4 — no extra ordering. Carve-out (flag-guarded): when grouping IS licensed
    # (Rule 3's own words) and every ordering key is a projected aggregate or its
    # SELECT alias, the sort only re-orders the licensed breakdown by its own measure
    # — presentation, not semantics; it admits no rows/filters/joins. Anything else
    # (ordering by a raw column, unlicensed grouping) stays refused.
    if ir.orderings and not _has(query, r"\b(top|highest|lowest|most|least|largest|"
                                        r"smallest|sorted?|rank|order|first|last|recent)\b"):
        try:
            from config import IR_ORDERBY_GROUPED_MEASURE_OK
        except Exception:
            IR_ORDERBY_GROUPED_MEASURE_OK = False
        _grouped_measure_sort = (
            IR_ORDERBY_GROUPED_MEASURE_OK and ir.groupings
            and _has(query, r"\b(per|by|each|every|group|breakdown|split)\b")
            and len(ir.ordering_agg) == len(ir.orderings)
            and all(is_agg or name in ir.agg_aliases
                    for name, is_agg in zip(ir.orderings, ir.ordering_agg)))
        if not _grouped_measure_sort:
            v.append(f"ORDER BY {ir.orderings} not requested")

    # Rule 2 — no unrequested DISTINCT (COUNT(DISTINCT) grain is allowed)
    if ir.distinct and "COUNT" not in ir.aggregations \
            and not _has(query, r"\b(distinct|unique|different)\b"):
        v.append("DISTINCT not requested")

    # Rule 5 — no joins beyond the planned tables (defense vs LLM-introduced joins)
    if allowed_tables:
        extra = [e for e in ir.entities if e not in allowed_tables]
        if extra:
            v.append(f"joins to {extra} outside the planned tables")

    return (len(v) == 0), v
