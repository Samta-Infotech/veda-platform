"""VEDA · Intent Envelope → QueryIntent mapper (frozen contract v1).

Deterministic. Turns the LLM's closed-enum intent envelope into a QueryIntent, which the
caller routes through validate_intent → build_sql (the single SQL path). The LLM decides
MEANING (intent + which entity/columns/values, as opaque handles); this file contains NO
regex, NO English, NO keyword lists. It resolves handles → names, looks up PRE-MATERIALIZED
metrics (never synthesizes — so grain_suspect guarding holds), and returns None whenever a
reference can't be resolved or the shape isn't a single-table analytical one (the caller then
keeps its existing behaviour: IR→sql_builder, or refuse).

See INTENT_ENVELOPE_CONTRACT.md (v1, frozen).
"""
import datetime
from typing import Optional, Dict

from query.intent import QueryIntent, Filter
from semantic import registry as reg

_INTENTS = {"count", "measure", "ratio", "trend", "compare", "group", "dimension_list"}
_BUCKETS = {"day", "week", "month", "quarter", "year"}
_UNITS   = {"week", "month", "year"}


def _resolve(handle, handle_map):
    """handle → (table_name, col_name|None). col_name is None for a table handle."""
    h = handle_map.get(handle)
    if not h:
        return None, None
    return h.get("table"), h.get("col")


def _count_metric(table):
    return reg.get_metric(f"{table}_count")


def _dimension_cols(table):
    """Groupable dimension column names for a table — group_col must be one of these, so the
    LLM can't group by a high-card / free-text / non-dimension column (valid SQL, useless)."""
    try:
        return {d.get("col_name") for d in reg.dimensions_for_table(table)
                if d.get("groupable", True) and d.get("col_name")}
    except Exception:
        return set()


def _measure_metric(table, col, agg):
    """The PRE-MATERIALIZED SUM/AVG metric for (table, col, agg), or None. Never synthesizes,
    so grain_suspect / fanout_safe guarding is preserved (see contract §5/§9.4)."""
    want = agg.upper()
    ref = f"{table}.{col}".lower()
    for _mid, mm in reg.active()["metrics"].items():
        if mm.get("source_table") != table:
            continue
        if (mm.get("aggregation") or mm.get("kind") or "").upper() != want:
            continue
        if ref in (mm.get("expression") or "").lower():
            return mm
    return None


def _build_filters(envelope, table, handle_map):
    """Resolve + coalesce envelope filters → [Filter], or None on an unresolved/cross-table
    reference (caller falls back). eq→ '='/'IN', ne→ '<>'/'NOT IN'; empty values dropped."""
    eq: Dict[str, list] = {}
    ne: Dict[str, list] = {}
    for f in (envelope.get("filters") or []):
        t, c = _resolve(f.get("col"), handle_map)
        if c is None:
            return None
        if t != table:
            return None                       # cross-table filter → belongs to the join planner
        v = f.get("value")
        if v is None or str(v).strip() == "":
            continue                          # drop empty value (never emit col = '')
        (ne if f.get("op") == "ne" else eq).setdefault(c, []).append(v)
    out = []
    for c, vs in eq.items():
        out.append(Filter(col=c, op=("=" if len(vs) == 1 else "IN"), values=vs))
    for c, vs in ne.items():
        out.append(Filter(col=c, op=("<>" if len(vs) == 1 else "NOT IN"), values=vs))
    return out


def map_envelope_to_intent(envelope, handle_map, tf=None) -> Optional[QueryIntent]:
    """Frozen-contract mapper. envelope: the LLM JSON; handle_map: {handle: {"table","col"}}
    resolved from the retrieved candidates; tf: L1 TemporalFilter (optional)."""
    intent = (envelope or {}).get("intent")
    if intent not in _INTENTS:
        return None
    table, _tcol = _resolve(envelope.get("entity"), handle_map)
    if not table or table not in reg.active()["concepts"]:
        return None

    filters = _build_filters(envelope, table, handle_map)
    if filters is None:
        return None

    def _strip(expr):
        return (expr or "").replace(f"{table}.", "")

    def _add_tf(metric):
        tdim = (metric or {}).get("allowed_time_dimension")
        if tf and tdim and (getattr(tf, "start", None) or getattr(tf, "end", None)):
            filters.append(Filter(col=tdim.split(".", 1)[1], op="BETWEEN",
                                  values=[tf.start or "1900-01-01", tf.end or "2999-12-31"]))

    # ── ratio ──────────────────────────────────────────────────────────────────
    if intent == "ratio":
        r = envelope.get("ratio") or {}
        t, c = _resolve(r.get("col"), handle_map)
        v = r.get("value")
        if c is None or t != table or v is None or str(v).strip() == "":
            return None
        return QueryIntent(query_type="ratio", subject_table=table, ratio_col=c,
                           ratio_value=v, route="envelope.ratio",
                           why=[f"ratio of {c}={v} over all {table}"])

    # ── compare (the one place the mapper does date math) ──────────────────────
    if intent == "compare":
        m = _count_metric(table)
        if not m or m.get("grain_suspect"):
            return None
        tdim = m.get("allowed_time_dimension")
        if not tdim:
            return None                        # time_col unresolved → own refuse case
        unit = envelope.get("compare_unit")
        if unit not in _UNITS:
            return None
        today = datetime.date.today()
        if unit == "month":
            t0 = today.replace(day=1); p0 = (t0 - datetime.timedelta(days=1)).replace(day=1)
        elif unit == "week":
            t0 = today - datetime.timedelta(days=today.weekday()); p0 = t0 - datetime.timedelta(days=7)
        else:
            t0 = today.replace(month=1, day=1); p0 = t0.replace(year=t0.year - 1)
        nxt = today + datetime.timedelta(days=1)
        return QueryIntent(query_type="compare", subject_table=table,
                           time_col=tdim.split(".", 1)[1],
                           compare={"unit": unit, "this": (str(t0), str(nxt)),
                                    "last": (str(p0), str(t0))},
                           route="envelope.compare", why=[f"this {unit} vs last {unit}"])

    # ── trend ──────────────────────────────────────────────────────────────────
    if intent == "trend":
        m = _count_metric(table)
        if not m or m.get("grain_suspect"):
            return None
        tdim = m.get("allowed_time_dimension")
        if not tdim:
            return None
        bucket = envelope.get("time_bucket")
        if bucket not in _BUCKETS:
            return None
        _add_tf(m)
        return QueryIntent(query_type="trend", subject_table=table, metric_id=m["metric_id"],
                           select_expr=_strip(m["expression"]), metric_alias=m["metric_id"],
                           time_col=tdim.split(".", 1)[1], time_bucket=bucket, filters=filters,
                           route="envelope.trend", why=[f"trend by {bucket}"])

    # ── dimension_list ─────────────────────────────────────────────────────────
    if intent == "dimension_list":
        t, c = _resolve(envelope.get("group_col"), handle_map)
        if c is None or t != table or c not in _dimension_cols(table):
            return None
        return QueryIntent(query_type="dimension_list", subject_table=table, group_col=c,
                           route="envelope.dimension_list", why=[f"list {c}"])

    # ── count / measure / group (group = count|measure + group_col) ────────────
    meas = envelope.get("measure")
    if intent == "measure" or (intent == "group" and meas):
        if not meas:
            return None
        t, c = _resolve(meas.get("col"), handle_map)
        agg = (meas.get("agg") or "").lower()
        if c is None or t != table or agg not in ("sum", "avg"):
            return None
        m = _measure_metric(table, c, agg)
    else:
        m = _count_metric(table)
    if not m or m.get("grain_suspect"):
        return None
    _add_tf(m)

    group_col = None
    if intent == "group":
        t, gc = _resolve(envelope.get("group_col"), handle_map)
        if gc is None or t != table or gc not in _dimension_cols(table):
            return None
        # grouped measure: the group dimension must be in the metric's declared safe set
        if meas:
            allowed = {a.split(".", 1)[1] for a in (m.get("allowed_dimensions") or []) if "." in a}
            if allowed and gc not in allowed:
                return None
        group_col = gc

    return QueryIntent(query_type=("measure" if intent == "measure" else "count"),
                       subject_table=table, metric_id=m["metric_id"],
                       select_expr=_strip(m["expression"]), metric_alias=m["metric_id"],
                       group_col=group_col, filters=filters,
                       route=f"envelope.{intent}", why=[f"metric {m['metric_id']}"])
