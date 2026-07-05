# =============================================================================
# query/intent.py
# VEDA — structured Query Intent: the enterprise replacement for regex routing.
#
# A QueryIntent is a TYPED, registry-resolved description of what a query asks.
# It is produced by ANY front-end (the regex fast-lane today; an LLM extractor
# next) and consumed by deterministic builders. The contract:
#
#   front-end  →  QueryIntent  →  validate_intent  →  build_sql  →  SQL
#                 (declarative)    (the firewall)      (deterministic)
#
# validate_intent is the firewall the LLM will lean on: every column/metric it
# names must resolve against the real schema, and every filter value must exist
# (grounding). An intent that doesn't resolve is DECLINED — never turned into SQL.
# build_sql contains NO English and NO regex; it only renders resolved intents.
# =============================================================================

import os
import json
from dataclasses import dataclass, field
from typing import List, Optional

from semantic import registry as reg

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Schema knowledge — real column set per table (for the existence firewall)
# ---------------------------------------------------------------------------
_COLS_CACHE = {"v": None}


def _table_columns():
    if _COLS_CACHE["v"] is None:
        path = os.path.join(_ROOT, "data", "veda_semantic_model.json")
        m = {}
        try:
            sm = json.load(open(path))
            for col_id in sm.get("columns", {}):
                t, c = col_id.split(".", 1)
                m.setdefault(t, set()).add(c)
        except Exception:
            pass
        _COLS_CACHE["v"] = m
    return _COLS_CACHE["v"]


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


# ---------------------------------------------------------------------------
# The IR
# ---------------------------------------------------------------------------
@dataclass
class Filter:
    col: str = ""
    op: str = "="            # = | <> | IN | NOT IN | BETWEEN
    values: list = field(default_factory=list)
    raw: Optional[str] = None  # pre-formed deterministic fragment (e.g. soft-delete)


@dataclass
class QueryIntent:
    # what kind of question
    query_type: str                              # count|measure|ratio|trend|compare|dimension_list|filter_lookup
    subject_table: str = ""
    # metric / projection
    metric_id: Optional[str] = None
    select_expr: Optional[str] = None            # metric expression, table-prefix stripped
    metric_alias: Optional[str] = None
    # shaping
    filters: List[Filter] = field(default_factory=list)
    group_col: Optional[str] = None
    time_bucket: Optional[str] = None            # day|week|month|quarter|year
    time_col: Optional[str] = None               # column the bucket/compare runs on
    # ratio / compare specifics (resolved values, never English)
    ratio_col: Optional[str] = None
    ratio_value: Optional[str] = None
    compare: Optional[dict] = None               # {unit, this:(s,e), last:(s,e)}
    # listing / lookup
    display_cols: List[str] = field(default_factory=list)
    # extra tables/columns a raw subquery filter references (cross-entity value filter)
    # — added to the validator's allow-list so the subquery's table/cols aren't flagged.
    extra_tables: List[str] = field(default_factory=list)
    extra_columns: List[str] = field(default_factory=list)
    # provenance
    route: str = ""
    why: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validator — the firewall (column existence + metric existence + grounding)
# ---------------------------------------------------------------------------
def validate_intent(intent: QueryIntent, ground_fn=None):
    """Resolve every reference in `intent` against the real schema.

    Returns (status, reason):
      'ok'      → safe to build
      'decline' → not expressible on the fast path; caller falls through (no error)
      'refuse'  → expressible-shape but a referenced object/value doesn't exist
                  (this is what catches an LLM that invents a column or value)

    ground_fn(table, col, value) -> bool : optional live-DB existence check for
    filter values. Omitted offline (skipped); supplied in the real env."""
    t = intent.subject_table
    cols = _table_columns().get(t, set())
    if not t or not cols:
        return "decline", f"unknown table {t!r}"

    def _col_ok(c):
        return c in cols

    # every referenced column must exist on the subject table
    refs = []
    if intent.group_col:  refs.append(intent.group_col)
    if intent.time_col:   refs.append(intent.time_col)
    if intent.ratio_col:  refs.append(intent.ratio_col)
    refs += intent.display_cols
    for f in intent.filters:
        if f.col:
            refs.append(f.col)
    for c in refs:
        if not _col_ok(c):
            return "refuse", f"column {t}.{c} does not exist"

    # metric must resolve in the registry
    if intent.metric_id and reg.get_metric(intent.metric_id) is None:
        return "refuse", f"unknown metric {intent.metric_id!r}"

    # filter-value grounding (real env only): a value that isn't in the column is
    # a fabricated mapping — refuse rather than silently return empty/garbage.
    if ground_fn is not None:
        for f in intent.filters:
            if f.raw or f.op == "BETWEEN":
                continue
            for v in f.values:
                if not ground_fn(t, f.col, v):
                    return "refuse", f"value {v!r} not present in {t}.{f.col}"

    return "ok", ""


# ---------------------------------------------------------------------------
# Builders — pure, deterministic, English-free
# ---------------------------------------------------------------------------
def _render_filters(filters: List[Filter]) -> str:
    parts = []
    for f in filters:
        if f.raw:
            parts.append(f.raw); continue
        col = _q(f.col)
        if f.op in ("=", "<>"):
            parts.append(f"{col} {f.op} '{f.values[0]}'")
        elif f.op in ("IN", "NOT IN"):
            parts.append(f"{col} {f.op} ({', '.join(repr_sql(v) for v in f.values)})")
        elif f.op == "BETWEEN":
            parts.append(f"{col} BETWEEN '{f.values[0]}' AND '{f.values[1]}'")
    return " AND ".join(parts)


def repr_sql(v) -> str:
    return f"'{v}'"


def build_sql(intent: QueryIntent):
    """QueryIntent → (sql, tables, columns, route, why). Assumes validated."""
    t = intent.subject_table
    where = _render_filters(intent.filters)
    w = f" WHERE {where}" if where else ""
    ref_cols, qt = [], intent.query_type

    # The metric's grain key (e.g. incident.id in COUNT(DISTINCT incident.id)) is
    # referenced inside select_expr — it must be in the allowed-column set or the AST
    # validator rejects it as an "unknown column". Every branch that emits select_expr
    # needs it, not just the bare count (the trend/group branches dropped it before).
    def _grain_pk():
        if intent.metric_id:
            grain = (reg.get_metric(intent.metric_id) or {}).get("grain", "")
            if "." in grain and not grain.endswith(".*"):
                return grain.split(".", 1)[1]
        return None
    _pk = _grain_pk()

    if qt == "ratio":
        col = _q(intent.ratio_col)
        sql = (f"SELECT ROUND(SUM(CASE WHEN {col} = '{intent.ratio_value}' "
               f"THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 1) "
               f"AS pct_{intent.ratio_col} FROM {_q(t)}")
        ref_cols = [intent.ratio_col]

    elif qt == "compare":
        c = intent.compare
        tcol, unit = _q(intent.time_col), c["unit"]
        (ts, te), (ps, pe) = c["this"], c["last"]
        sql = (f"SELECT "
               f"SUM(CASE WHEN {tcol} >= '{ts}' AND {tcol} < '{te}' "
               f"THEN 1 ELSE 0 END) AS this_{unit}, "
               f"SUM(CASE WHEN {tcol} >= '{ps}' AND {tcol} < '{pe}' "
               f"THEN 1 ELSE 0 END) AS last_{unit} "
               f"FROM {_q(t)}")
        ref_cols = [intent.time_col]

    elif qt == "trend":
        tcol = _q(intent.time_col)
        sql = (f"SELECT DATE_TRUNC('{intent.time_bucket}', {tcol}) AS period, "
               f"{intent.select_expr} AS {intent.metric_alias} FROM {_q(t)}"
               f"{w} GROUP BY period ORDER BY period")
        ref_cols = [f.col for f in intent.filters if f.col] + [intent.time_col]
        if _pk:
            ref_cols.append(_pk)

    elif qt == "count" and intent.group_col:
        g = _q(intent.group_col)
        sql = (f"SELECT {g}, {intent.select_expr} AS {intent.metric_alias} "
               f"FROM {_q(t)}{w} GROUP BY {g} ORDER BY {intent.metric_alias} DESC")
        ref_cols = [f.col for f in intent.filters if f.col] + [intent.group_col]
        if _pk:
            ref_cols.append(_pk)

    elif qt in ("count", "measure"):
        sql = f"SELECT {intent.select_expr} AS {intent.metric_alias} FROM {_q(t)}{w}"
        ref_cols = [f.col for f in intent.filters if f.col]
        if _pk:
            ref_cols = [_pk] + ref_cols

    elif qt == "dimension_list":
        g = _q(intent.group_col)
        sql = (f"SELECT DISTINCT {g} FROM {_q(t)} "
               f"WHERE {g} IS NOT NULL ORDER BY {g}")
        ref_cols = [intent.group_col]

    elif qt == "filter_lookup":
        sel = ", ".join(_q(c) for c in intent.display_cols) if intent.display_cols else "*"
        f0 = intent.filters[0]
        sql = f"SELECT {sel} FROM {_q(t)} WHERE {_q(f0.col)} = {f0.values[0]}"
        ref_cols = list(intent.display_cols) + [f0.col]

    else:
        raise ValueError(f"unbuildable intent: {qt}")

    cols = list(dict.fromkeys(c for c in ref_cols + list(intent.extra_columns) if c))
    tables = {t} | set(intent.extra_tables)
    return sql, tables, cols, intent.route, list(intent.why)
