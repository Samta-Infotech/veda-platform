# =============================================================================
# query/fast_path.py
# VEDA — deterministic fast paths (Phase-1 semantic-layer slice).
#
# Resolves count / aggregate / dimension-list questions DIRECTLY against the
# compiled registries (semantic/*.json) — no retrieval, no join planner, no LLM.
# Returns a FastPathResult (deterministic SQL + the table/column allow-lists the
# existing validator needs), or None to fall through to the full pipeline.
#
# SAFETY: a fast-path result is still run through value_grounding +
# validate_and_parameterize + execute_sql by the caller. This path generates SQL
# faster; it does NOT trust it more. Matching is conservative — anything ambiguous
# or multi-entity (a real join) returns None and falls through untouched.
# =============================================================================

import os
import re
import json
import time
from dataclasses import dataclass, field
from typing import Optional, List

from semantic import registry as reg
from query.intent import QueryIntent, Filter, validate_intent, build_sql

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROUTE_LOG = os.path.join(_ROOT, "logs", "route_log.jsonl")

_COUNT_TRIGGERS = ("how many", "number of", "count of", "count the", "total number",
                   "# of", "no. of")
_COUNT_WORDS    = {"count", "counts"}
_SUM_VERBS      = ("sum of", "total ")
_AVG_VERBS      = ("average ", "avg ", "mean ")
_LIST_VERBS     = ("list", "show", "what are", "which", "distinct", "unique",
                   "possible", "available")
# Words that imply a relationship to ANOTHER entity → a real join → fall through.
# "per" and "each" are NOT here — they're grouping prepositions ("sum of X per/each Y"),
# handled by the group_dim/vals_hit/bucket checks below; treating them as a blanket join
# signal blocked every grouped SUM/AVG/MAX/MIN/COUNT query ("<metric> per/for each Y")
# before the grouping dimension was even resolved — the query then fell to the full
# pipeline, mis-anchored, and refused (qualifier_dropped).
_JOIN_HINTS     = {"with", "without", "their", "its", "whose",
                   "having", "have", "has", "across", "joined"}
_MAX_VERBS      = ("max ", "maximum ", "highest ", "largest ")
_MIN_VERBS      = ("min ", "minimum ", "lowest ", "smallest ")

# The query LANGUAGE layer comes from config (closed linguistic classes, per-language,
# NOT schema vocabulary). Any query token left over AFTER removing these and the tokens
# actually consumed by the entity / value / dimension / bucket is an UNMODELLED
# QUALIFIER — e.g. "assigned to abhijit", "active" — meaning the query asks for
# something narrower than the SQL we'd emit. Then we MUST fall through to the full
# pipeline (which filters or refuses), never emit a count that drops the qualifier.
def _language_vocab():
    from config import QUERY_GRAMMAR, QUERY_LANGUAGE
    v = set()
    for ops in QUERY_GRAMMAR.values():
        for w in ops:
            v.update(w.split())
    for cls in QUERY_LANGUAGE.values():
        v.update(cls)
    return v


_LANG_VOCAB = None


def _unmodelled_residual(qtoks, consumed):
    """Singularized query tokens neither consumed nor part of the config LANGUAGE layer.
    Non-empty ⇒ an unhandled qualifier ⇒ caller falls through."""
    global _LANG_VOCAB
    if _LANG_VOCAB is None:
        _LANG_VOCAB = _language_vocab()
    sing = reg._singularize
    used = {sing(w) for w in consumed}
    used |= {sing(w) for w in _LANG_VOCAB}
    return {sing(w) for w in qtoks} - used


_GRAPH_CACHE = {"g": None}


def _graph():
    if _GRAPH_CACHE["g"] is None:
        # Route through config.RELATIONSHIP_GRAPH_FILE (tenant/source/version-scoped)
        # instead of a hardcoded repo-root path, matching veda.runtime.get_graph() /
        # veda.graph_guard — else a non-default-scope source silently falls back to the
        # legacy default-scope file (or an empty graph).
        try:
            from config import RELATIONSHIP_GRAPH_FILE
            p = RELATIONSHIP_GRAPH_FILE if os.path.isabs(RELATIONSHIP_GRAPH_FILE) \
                else os.path.join(_ROOT, RELATIONSHIP_GRAPH_FILE)
        except Exception:
            p = os.path.join(_ROOT, "data", "veda_relationship_graph.json")
        _GRAPH_CACHE["g"] = json.load(open(p)) if os.path.exists(p) else {"edges": []}
    return _GRAPH_CACHE["g"]


@dataclass
class FastPathResult:
    sql:     str
    tables:  set
    columns: List[str]
    primary: str
    route:   str                       # e.g. "metric.count", "metric.count.group", "dimension.list"
    why:     List[str] = field(default_factory=list)


def log_route(route: str, query: str, latency_ms: float, **extra):
    """Best-effort append to logs/route_log.jsonl. Never raises."""
    try:
        try:
            from config import ROUTE_LOG_INCLUDE_QUERY
        except Exception:
            ROUTE_LOG_INCLUDE_QUERY = True
        rec = {"t": round(time.time(), 3), "route": route,
               "latency_ms": round(latency_ms, 1)}
        if ROUTE_LOG_INCLUDE_QUERY:
            rec["query"] = query
        rec.update(extra)
        os.makedirs(os.path.dirname(_ROUTE_LOG), exist_ok=True)
        with open(_ROUTE_LOG, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _rewrite_agg(expr: str, func: str) -> str:
    """Swap the aggregate function in an expression: AVG(col) → MAX(col)."""
    m = re.match(r'[A-Z]+\((.+)\)', expr, re.I)
    return f"{func}({m.group(1)})" if m else expr


def _measure_col(expr: str):
    """The bare measure column inside an aggregate expression: 'MAX(amount)' → 'amount'."""
    m = re.match(r'[A-Z]+\(\s*(?:DISTINCT\s+)?([A-Za-z_][A-Za-z0-9_.]*)\s*\)', expr or "", re.I)
    if not m:
        return None
    col = m.group(1)
    return col.split(".")[-1]


def _fk_to(a_table: str, b_table: str, graph: dict):
    """(fk_col_on_A, pk_col_on_B) if table A has a DIRECT FK to B, else None."""
    for e in graph.get("edges", []):
        if e.get("source_table") == a_table and e.get("target_table") == b_table:
            return e.get("source_column"), e.get("target_column")
    return None


def _metric_entity_group(query, query_l, qtoks, table_a, agg_func, measure_col,
                         malias, why):
    """MULTI-TABLE metric aggregation (one FK hop): an <AGG> of a measure on table A,
    grouped by an ENTITY on table B that the query NAMES and that A reaches via a single
    FK (A.fk → B.pk), plus an optional dimension ON A. Deterministic — the join is a real
    FK edge (the firewall's graph-guard re-verifies it). Returns a FastPathResult, or None
    to fall back to the single-table path.

      "maximum financial value per property by entry type"
      → SELECT b."<name>", a."entry_type", MAX(a."amount")
        FROM accounts_generalledger a JOIN assets_asset b ON a.asset_id = b.id
        GROUP BY b."<name>", a."entry_type"
    """
    if not measure_col:
        return None
    graph = _graph()
    if not graph.get("edges"):
        return None
    # grouping entity named in the query, reachable from A by ONE FK, and not A itself
    edge = disp = b_table = None
    for concept, _score in reg.match_concepts(qtoks):
        bt = (concept.get("resolves_to") or {}).get("table")
        if not bt or bt == table_a:
            continue
        e = _fk_to(table_a, bt, graph)
        if e is None:
            continue
        dcols = [c.split(".", 1)[1] for c in (concept.get("default_display_columns") or [])
                 if "." in c]
        disp = dcols[0] if dcols else None
        if not disp:
            # prefer a human label column on B over the bare pk (id) for a readable group
            _cols = _sm().get("columns", {})
            disp = next((cn for cn in ("name", "project_name", "title", "display_name",
                                        "full_name", "label")
                         if f"{bt}.{cn}" in _cols), (e[1] or "id"))
        edge, b_table = e, bt
        break
    if edge is None:
        return None
    fk_col, pk_col = edge

    # optional dimension ON A ("... by entry type")
    dim = None
    if _has(query_l, (" by ", " per ", " each ", "grouped", "breakdown",
                       "broken down", "distribution")):
        gd = [d for d in reg.match_dimensions_in_table(table_a, qtoks, query_l, k=2)
              if d["col_name"] not in (fk_col, measure_col, pk_col)]
        if gd:
            dim = gd[0]["col_name"]

    a, b = "a", "b"
    entity_alias = b_table.split("_")[-1]
    ref_cols = [measure_col, fk_col, pk_col, disp]
    sel = [f'{b}.{_q(disp)} AS {_q(entity_alias)}']
    grp = [f'{b}.{_q(disp)}']
    if dim:
        sel.append(f'{a}.{_q(dim)}')
        grp.append(f'{a}.{_q(dim)}')
        ref_cols.append(dim)
    sel.append(f'{agg_func}({a}.{_q(measure_col)}) AS {_q(malias)}')
    sql = (f'SELECT {", ".join(sel)} FROM {_q(table_a)} {a} '
           f'JOIN {_q(b_table)} {b} ON {a}.{_q(fk_col)} = {b}.{_q(pk_col)} '
           f'GROUP BY {", ".join(grp)} ORDER BY {_q(malias)} DESC LIMIT 100')
    return FastPathResult(
        sql=sql, tables={table_a, b_table}, columns=list(dict.fromkeys(ref_cols)),
        primary=table_a, route=f"metric.{agg_func.lower()}.entitygroup",
        why=why + [f"{agg_func}({measure_col}) per {b_table}" + (f" by {dim}" if dim else "")])


def _has(query_l: str, needles) -> bool:
    return any(n in query_l for n in needles)


def _count_intent(query_l: str, qtoks: set) -> bool:
    return _has(query_l, _COUNT_TRIGGERS) or bool(_COUNT_WORDS & qtoks)


def _single_entity(qtoks: set):
    """Return the one dominant entity concept, or None when zero / ambiguous-multi.
    Two distinct strongly-matched entity tables ⇒ a join ⇒ fall through (None)."""
    hits = reg.match_concepts(qtoks)
    if not hits:
        return None
    # A runner-up counts as a SECOND entity (a join, fall through) only if it brings a
    # query token the top concept does not already cover. Otherwise it merely shares a
    # token with the top concept (e.g. "incident" is in both `incident` and
    # `incident_signal_score`) and must not be read as a separately-requested entity —
    # that false multi-entity reading made every count query fall through.
    top = hits[0][0]
    top_table = top["resolves_to"]["table"]
    top_matched = set(top["match_tokens"]) & qtoks
    # A token that is a VALUE on the top entity (object_type='Level 1' → "level","1") is a
    # FILTER, not a competing entity — even when some other table is named after it
    # (a `level` / `signal_levels` concept). Discount such tokens so a value word doesn't
    # masquerade as a second entity and force a needless fall-through.
    val_toks: set = set()
    _vh = reg.match_values_in_table(top_table, qtoks)
    if _vh:
        for _v in _vh[1]:
            val_toks |= set(re.findall(r"[a-z0-9]+", str(_v).lower()))
    for c, score in hits[1:]:
        if score[0] < 1:
            continue
        if c["resolves_to"]["table"] == top_table:
            continue
        if (set(c["match_tokens"]) & qtoks) - top_matched - val_toks:
            return None                # genuine multi-entity -> let the planner join
    return top


def _time_clause(metric, tf, params_cols):
    """(sql_fragment, ()) for a temporal BETWEEN on the metric's time dimension."""
    if not (tf and (getattr(tf, "start", None) or getattr(tf, "end", None))):
        return None
    tdim = metric.get("allowed_time_dimension")
    if not tdim:
        return None
    col = tdim.split(".", 1)[1]
    params_cols.append(col)
    start = tf.start or "1900-01-01"
    end   = tf.end or "2999-12-31"
    return f"{_q(col)} BETWEEN '{start}' AND '{end}'"


_SM_CACHE = {"v": None}


def _sm():
    if _SM_CACHE["v"] is None:
        try:
            from config import SEMANTIC_MODEL_FILE
            path = SEMANTIC_MODEL_FILE if os.path.isabs(SEMANTIC_MODEL_FILE) \
                else os.path.join(_ROOT, SEMANTIC_MODEL_FILE)
            _SM_CACHE["v"] = json.load(open(path))
        except Exception:
            _SM_CACHE["v"] = {}
    return _SM_CACHE["v"]


def _finalize(query, intent, ground_fn=None) -> Optional[FastPathResult]:
    """Validate an extracted intent against the schema (the firewall), then build SQL.
    Non-'ok' → None (fall through). This is the SHARED path that an LLM extractor
    will use too — the regex branches below are just one intent source."""
    status, _reason = validate_intent(intent, ground_fn=ground_fn)
    if status != "ok":
        return None
    sql, tables, columns, route, why = build_sql(intent)

    # Fast-path self-completeness (count/measure only — where value filters live):
    # if the built SQL DROPS a content token the user named (e.g. a value the
    # registry's sampled set couldn't match — "incidents that are waived" when
    # 'Waive' wasn't sampled), DON'T return a filter-dropping count. Fall through to
    # the full pipeline (LLM + live value-grounding), which can resolve the value
    # and put it IN the SQL. This is the SAME gate the pipeline applies to the result
    # anyway, so it never hurts a passing query — it only converts a would-be refusal
    # into a real attempt. Scoped to count/measure so intent words on ratio/trend
    # (percentage/monthly/…) can't trip it.
    # dimension_list included: it projects ONE column, so a multi-attribute list
    # ("role names AND role codes") drops a column → fall through to the single-table
    # path, which projects all named columns. (Single-attribute lists pass unchanged.)
    if intent.query_type in ("count", "measure", "dimension_list"):
        try:
            from veda.validation import qualifier_completeness
            ok_q, _missing = qualifier_completeness(query, sql, _sm())
            if not ok_q:
                return None
        except Exception:
            pass

    return FastPathResult(sql=sql, tables=tables, columns=columns,
                          primary=intent.subject_table, route=route, why=why)


def try_fast_path(query: str, tf=None) -> Optional[FastPathResult]:
    """Attempt a deterministic single-table fast path. Returns None to fall through.

    This function only EXTRACTS a QueryIntent from the query (regex fast-lane) and
    hands it to _finalize → validate → build. The intent IR + validator + builders
    live in query/intent.py and are front-end-agnostic: an LLM extractor producing
    the same QueryIntent gets the same validation and the same SQL, for free."""
    if not reg.is_ready():
        return None
    query_l = " " + query.lower().strip() + " "
    qtoks   = reg.query_tokens(query)

    # A relationship word strongly implies a join to another entity. Existence/“with”
    # is handled by the existing deterministic existence path, not here.
    join_hint = bool(_JOIN_HINTS & qtoks)

    # ── 0a. RATIO: "percentage of incidents that are escalated" ──────────────
    if re.search(r"\b(percentage|percent|proportion|share)\b|%", query_l):
        entity = _single_entity(qtoks)
        if entity is not None:
            table = entity["resolves_to"]["table"]
            metric = reg.get_metric(entity["default_metric"])
            if metric and not metric.get("grain_suspect"):
                hit = reg.match_values_in_table(table, qtoks)
                if hit and len(hit[1]) == 1:
                    d, v = hit[0], hit[1][0]
                    return _finalize(query, QueryIntent(
                        query_type="ratio", subject_table=table,
                        ratio_col=d["col_name"], ratio_value=v,
                        route="metric.ratio",
                        why=[f"ratio of {d['col_name']} = {v} over all {table}"]))

    # ── 0b. PERIOD COMPARISON: "incidents this month vs last month" ──────────
    mcmp = re.search(r"\b(?:this|current)\s+(week|month|year)\s+"
                     r"(?:vs|versus|compared\s+(?:to|with)|against)\s+"
                     r"(?:the\s+)?(?:last|previous)\s+\1\b", query_l)
    if mcmp:
        entity = _single_entity(qtoks)
        if entity is not None:
            table = entity["resolves_to"]["table"]
            metric = reg.get_metric(entity["default_metric"])
            tdim = (metric or {}).get("allowed_time_dimension")
            if metric and tdim and not metric.get("grain_suspect"):
                import datetime as _dt
                unit = mcmp.group(1)
                today = _dt.date.today()
                if unit == "month":
                    t0 = today.replace(day=1)
                    p0 = (t0 - _dt.timedelta(days=1)).replace(day=1)
                elif unit == "week":
                    t0 = today - _dt.timedelta(days=today.weekday())
                    p0 = t0 - _dt.timedelta(days=7)
                else:
                    t0 = today.replace(month=1, day=1)
                    p0 = t0.replace(year=t0.year - 1)
                tcol = tdim.split(".", 1)[1]
                nxt = today + _dt.timedelta(days=1)
                return _finalize(query, QueryIntent(
                    query_type="compare", subject_table=table, time_col=tcol,
                    compare={"unit": unit, "this": (str(t0), str(nxt)),
                             "last": (str(p0), str(t0))},
                    route="metric.compare",
                    why=[f"this {unit} [{t0}..{nxt}) vs last {unit} [{p0}..{t0}) on {tcol}"]))

    # ── 1. COUNT metric (optionally grouped / filtered / time-bounded) ────────
    if _count_intent(query_l, qtoks):
        entity = _single_entity(qtoks)
        if entity is not None:
            table = entity["resolves_to"]["table"]
            metric = reg.get_metric(entity["default_metric"])
            if metric and metric.get("grain_suspect"):
                return None             # entries-per-entity table → live-data decision
            if metric:
                filters, why = [], []
                try:
                    from config import COUNT_EXCLUDE_SOFT_DELETED
                except Exception:
                    COUNT_EXCLUDE_SOFT_DELETED = False
                _sdf = metric.get("soft_delete_filter")
                if _sdf and COUNT_EXCLUDE_SOFT_DELETED:
                    filters.append(Filter(raw=_sdf))
                    why.append("live rows only")

                # time-bucket trend takes precedence over dimension grouping
                bucket = None
                mb = re.search(r"\b(?:per|by|every)\s+(day|week|month|quarter|year)\b",
                               query_l)
                if mb:
                    bucket = mb.group(1)
                else:
                    for word, b in (("daily", "day"), ("weekly", "week"),
                                    ("monthly", "month"), ("quarterly", "quarter"),
                                    ("yearly", "year"), ("trend", "month"),
                                    ("over time", "month")):
                        if word in query_l:
                            bucket = b
                            break
                if bucket and not metric.get("allowed_time_dimension"):
                    bucket = None

                group_dim = None
                group_dim2 = None
                if bucket is None and _has(query_l, (" by ", " per ", " each ",
                                                     "grouped", "breakdown",
                                                     "broken down", "distribution")):
                    _gds = reg.match_dimensions_in_table(table, qtoks, query_l, k=2)
                    group_dim = _gds[0] if _gds else None
                    group_dim2 = _gds[1] if len(_gds) > 1 else None

                vals_hit = reg.match_values_in_table(table, qtoks)
                # Registry sample missed → optionally resolve the value against the
                # LIVE DB (sample-independent, grounded). Gated OFF by default; only
                # consulted when the sampled values couldn't match. A unique column
                # match becomes the filter; ambiguous/none → unchanged (falls through
                # via the residual guard below).
                _extra_tables, _extra_columns, _sub_consumed = [], [], set()
                if vals_hit is None:
                    try:
                        from config import VALUE_RESOLVER_LIVE_DB
                    except Exception:
                        VALUE_RESOLVER_LIVE_DB = False
                    if VALUE_RESOLVER_LIVE_DB:
                        try:
                            from query.value_resolver import (resolve_value_filter,
                                                              column_values_lookup)
                            from veda.runtime import _pg
                            _acols = {c.split(".", 1)[1] for c in _sm().get("columns", {})
                                      if c.split(".", 1)[0] == table}
                            _desc = resolve_value_filter(table, qtoks, _graph(),
                                                         column_values_lookup(_pg),
                                                         anchor_cols=_acols)
                        except Exception:
                            _desc = None
                        if _desc and _desc.get("kind") == "direct":
                            # value lives on the anchor table → ordinary filter
                            vals_hit = ({"col_name": _desc["column"], "labels": []},
                                        [_desc["value"]])
                            why.append("value resolved via live data")
                        elif _desc and _desc.get("kind") == "subquery":
                            # value lives in an FK-reachable table → grounded subquery,
                            # no join / no fan-out. Mark the value + relation tokens
                            # consumed so the residual guard doesn't fall through.
                            filters.append(Filter(raw=(
                                f'{_q(_desc["anchor_col"])} IN (SELECT '
                                f'{_q(_desc["target_col"])} FROM {_q(_desc["target"])} '
                                f'WHERE {_q(_desc["filter_col"])} = \'{_desc["value"]}\')')))
                            _extra_tables = [_desc["target"]]
                            _extra_columns = [_desc["anchor_col"], _desc["target_col"],
                                              _desc["filter_col"]]
                            _sub_consumed = set(re.findall(r"[a-z0-9]+",
                                                str(_desc["value"]).lower()))
                            _sub_consumed |= {p for p in _desc["anchor_col"].split("_")
                                              if len(p) > 2}
                            why.append(f"value '{_desc['value']}' → {_desc['target']}."
                                       f"{_desc['filter_col']} (data-driven, via FK)")
                if join_hint and group_dim is None and vals_hit is None \
                        and bucket is None and not _extra_tables:
                    return None

                # Qualifier guard: every content token must be accounted for, else the
                # query asks something narrower than a bare count ("...assigned to abhijit",
                # "active roles"). Build the consumed set from what we actually used, then
                # fall through if anything is left over.
                consumed = set(entity.get("match_tokens", [])) | _sub_consumed
                if vals_hit is not None:
                    _d, _vs = vals_hit
                    consumed |= {t for v in _vs for t in re.findall(r"[a-z0-9]+", str(v).lower())}
                    consumed |= set(_d["col_name"].split("_"))
                    for _lab in _d.get("labels", []):
                        consumed |= set(re.findall(r"[a-z0-9]+", _lab.lower()))
                if group_dim is not None:
                    consumed |= set(group_dim["col_name"].split("_"))
                    for _lab in group_dim.get("labels", []):
                        consumed |= set(re.findall(r"[a-z0-9]+", _lab.lower()))
                if group_dim2 is not None:
                    consumed |= set(group_dim2["col_name"].split("_"))
                    for _lab in group_dim2.get("labels", []):
                        consumed |= set(re.findall(r"[a-z0-9]+", _lab.lower()))
                if bucket is not None:
                    consumed |= {bucket, "daily", "weekly", "monthly", "quarterly",
                                 "yearly", "trend", "over"}
                if _unmodelled_residual(qtoks, consumed):
                    return None             # unhandled qualifier → full pipeline

                if vals_hit is not None:
                    d, vs = vals_hit
                    def _negated(v):
                        return bool(re.search(
                            rf"\b(?:not|excluding|except)\s+(?:\w+\s+)?{re.escape(v.lower())}\b",
                            query_l))
                    negs = [_negated(v) for v in vs]
                    if any(negs) and not all(negs):
                        return None
                    neg = all(negs)
                    op = ("<>" if neg else "=") if len(vs) == 1 else ("NOT IN" if neg else "IN")
                    filters.append(Filter(col=d["col_name"], op=op, values=list(vs)))
                    why.append(f"filter {d['col_name']} "
                               f"{'NOT ' if neg else ''}{'IN ' if len(vs) > 1 else '= '}{vs}")

                has_temporal = False
                if tf and (getattr(tf, "start", None) or getattr(tf, "end", None)) \
                        and metric.get("allowed_time_dimension"):
                    tcol = metric["allowed_time_dimension"].split(".", 1)[1]
                    filters.append(Filter(col=tcol, op="BETWEEN",
                                          values=[tf.start or "1900-01-01",
                                                  tf.end or "2999-12-31"]))
                    has_temporal = True
                    why.append("temporal window")

                select_expr = metric["expression"].replace(f"{table}.", "")
                alias = metric["metric_id"]
                base_why = [f"metric {alias}"]
                # multi-table: "how many <A> per <entity on B via FK> [by <dim on A>]"
                # (only when no value filter is in play, so we never drop a WHERE clause).
                if not filters and not vals_hit:
                    _mtc = _metric_entity_group(query, query_l, qtoks, table, "COUNT",
                                                _measure_col(select_expr) or "id", alias,
                                                base_why + why)
                    if _mtc is not None:
                        return _mtc
                if bucket is not None:
                    tcol = metric["allowed_time_dimension"].split(".", 1)[1]
                    return _finalize(query, QueryIntent(
                        query_type="trend", subject_table=table, metric_id=alias,
                        select_expr=select_expr, metric_alias=alias,
                        time_col=tcol, time_bucket=bucket, filters=filters,
                        extra_tables=_extra_tables, extra_columns=_extra_columns,
                        route="metric.count.trend",
                        why=base_why + why + [f"bucket by {bucket} on {tcol}"]))
                if group_dim is not None:
                    g = group_dim["col_name"]
                    g2 = group_dim2["col_name"] if group_dim2 is not None else None
                    return _finalize(query, QueryIntent(
                        query_type="count", subject_table=table, metric_id=alias,
                        select_expr=select_expr, metric_alias=alias,
                        group_col=g, group_col2=g2, filters=filters,
                        extra_tables=_extra_tables, extra_columns=_extra_columns,
                        route="metric.count.group",
                        why=base_why + why + [f"group by {g}" + (f", {g2}" if g2 else "")]))
                route = ("metric.count" + (".filter" if (vals_hit or _extra_tables) else "")
                         + (".temporal" if has_temporal else ""))
                return _finalize(query, QueryIntent(
                    query_type="count", subject_table=table, metric_id=alias,
                    select_expr=select_expr, metric_alias=alias, filters=filters,
                    extra_tables=_extra_tables, extra_columns=_extra_columns,
                    route=route, why=base_why + why))

    # ── 2. SUM / AVG measure metric (single table, optionally grouped) ────────
    if _has(query_l, _SUM_VERBS) or _has(query_l, _AVG_VERBS):
        for metric, _ in reg.match_metric_labels(query_l):
            table = metric["source_table"]
            if join_hint or metric.get("grain_suspect"):
                continue
            filters, why = [], [f"metric {metric['metric_id']}"]
            if tf and (getattr(tf, "start", None) or getattr(tf, "end", None)) \
                    and metric.get("allowed_time_dimension"):
                tcol = metric["allowed_time_dimension"].split(".", 1)[1]
                filters.append(Filter(col=tcol, op="BETWEEN",
                                      values=[tf.start or "1900-01-01",
                                              tf.end or "2999-12-31"]))
                why.append("temporal window")
            expr = metric["expression"].replace(f"{table}.", "")
            # multi-table: "<total/average X> per <entity on another table> [by <dim>]"
            _mt = _metric_entity_group(query, query_l, qtoks, table, metric.get("kind", "SUM"),
                                       _measure_col(expr), metric["metric_id"], why)
            if _mt is not None:
                return _mt
            group_dims = []
            if _has(query_l, (" by ", " per ", " each ", "grouped", "breakdown",
                               "broken down", "distribution")):
                group_dims = reg.match_dimensions_in_table(table, qtoks, query_l, k=2)
            if group_dims:
                g = group_dims[0]["col_name"]
                g2 = group_dims[1]["col_name"] if len(group_dims) > 1 else None
                return _finalize(query, QueryIntent(
                    query_type="count", subject_table=table,
                    select_expr=expr, metric_alias=metric["metric_id"],
                    group_col=g, group_col2=g2, filters=filters,
                    route="metric.measure.group",
                    why=why + [f"group by {g}" + (f", {g2}" if g2 else "")]))
            return _finalize(query, QueryIntent(
                query_type="measure", subject_table=table, metric_id=metric["metric_id"],
                select_expr=expr, metric_alias=metric["metric_id"], filters=filters,
                route="metric.measure", why=why))

    # ── 2b. MAX / MIN metric (single table, optionally grouped) ───────────────
    if _has(query_l, _MAX_VERBS) or _has(query_l, _MIN_VERBS):
        func = "MAX" if _has(query_l, _MAX_VERBS) else "MIN"
        for metric, _ in reg.match_metric_labels(query_l):
            table = metric["source_table"]
            if join_hint or metric.get("grain_suspect"):
                continue
            base_expr = metric["expression"].replace(f"{table}.", "")
            expr  = _rewrite_agg(base_expr, func)
            alias = f"{func.lower()}_{metric['metric_id']}"
            filters, why = [], [f"{func}({metric['metric_id']})"]
            # multi-table: "<AGG> per <entity on another table> [by <dim>]" (one FK hop)
            _mt = _metric_entity_group(query, query_l, qtoks, table, func,
                                       _measure_col(base_expr), alias, why)
            if _mt is not None:
                return _mt
            group_dims = []
            if _has(query_l, (" by ", " per ", " each ", "grouped", "breakdown",
                               "broken down", "distribution")):
                group_dims = reg.match_dimensions_in_table(table, qtoks, query_l, k=2)
            if group_dims:
                g = group_dims[0]["col_name"]
                g2 = group_dims[1]["col_name"] if len(group_dims) > 1 else None
                return _finalize(query, QueryIntent(
                    query_type="count", subject_table=table,
                    select_expr=expr, metric_alias=alias,
                    group_col=g, group_col2=g2, filters=filters,
                    route=f"metric.{func.lower()}.group",
                    why=why + [f"group by {g}" + (f", {g2}" if g2 else "")]))
            return _finalize(query, QueryIntent(
                query_type="measure", subject_table=table,
                select_expr=expr, metric_alias=alias, filters=filters,
                route=f"metric.{func.lower()}", why=why))

    # ── 3. Dimension list ("what are the incident statuses") ──────────────────
    if _has(query_l, _LIST_VERBS) and not _count_intent(query_l, qtoks) and not join_hint:
        entity = _single_entity(qtoks)
        if entity is not None:
            table = entity["resolves_to"]["table"]
            dim   = reg.match_dimension_in_table(table, qtoks, query_l)
            if dim is not None and dim.get("groupable", True):
                g = dim["col_name"]
                # Require a DISTINCTIVE dimension token to actually be named — a token
                # from the dimension's name/aliases that is NOT just the entity name.
                # Without this, "show incidents assigned to ekaansh" spuriously matches
                # incident_status via the shared "incident" token and lists statuses,
                # silently dropping the real "assigned to <person>" filter.
                _ent = set(entity.get("match_tokens", []))
                _distinct = {t for t in g.split("_") if len(t) > 2}
                for _lab in (dim.get("labels") or []):
                    _distinct |= {t for t in _lab.split() if len(t) > 2}
                _distinct -= _ent
                if _distinct & qtoks:
                    return _finalize(query, QueryIntent(
                        query_type="dimension_list", subject_table=table, group_col=g,
                        route="dimension.list", why=[f"distinct {table}.{g}"]))
                # else: not a real dimension-list request → fall through to full pipeline

    # ── 4. Subject + related-entity-id filter ("roles which belong to org 1") ─
    if not _count_intent(query_l, qtoks):
        hits = reg.match_concepts(qtoks)
        filt = None
        for c, _s in hits[:6]:
            for lab in sorted(c.get("labels", []), key=len, reverse=True):
                m = re.search(rf"\b{re.escape(lab)}\s+(?:id\s+|#\s*)?(\d+)\b", query_l)
                if m:
                    filt = (c, int(m.group(1)))
                    break
            if filt:
                break
        if filt:
            fc, fval = filt
            ftable = fc["resolves_to"]["table"]
            subj, best_pos = None, 10 ** 9
            for c, _s in hits[:6]:
                if c["resolves_to"]["table"] == ftable:
                    continue
                pos = min((query_l.find(lab) for lab in c.get("labels", [])
                           if lab and query_l.find(lab) >= 0), default=10 ** 9)
                if pos < best_pos:
                    subj, best_pos = c, pos
            if subj is not None:
                stable = subj["resolves_to"]["table"]
                edges = [e for e in _graph().get("edges", [])
                         if e.get("source_table") == stable
                         and e.get("target_table") == ftable
                         and e.get("relationship_type") not in
                         ("audit", "history", "polymorphic")]
                if len(edges) == 1:
                    fk = edges[0]["source_column"]
                    disp = [k.split(".", 1)[1]
                            for k in subj.get("default_display_columns", [])]
                    return _finalize(query, QueryIntent(
                        query_type="filter_lookup", subject_table=stable,
                        display_cols=disp,
                        filters=[Filter(col=fk, op="=", values=[fval])],
                        route="entity.filter_lookup",
                        why=[f"subject {stable}",
                             f"filter {fk} = {fval} (entity '{ftable}' + number = filter)"]))

    return None


# ---------------------------------------------------------------------------
# Self-test / diagnostic — runs WITHOUT the DB or the LLM. Verifies registries
# load and the router maps representative queries. In the real env:
#   python3 -m query.fast_path
# ---------------------------------------------------------------------------
def _diagnose(query, tf=None):
    qtoks = reg.query_tokens(query)
    hits  = [(h[0]["concept_id"], h[1]) for h in reg.match_concepts(qtoks)[:3]]
    fp    = try_fast_path(query, tf)
    print(f"\n  Q: {query}")
    print(f"    qtokens : {sorted(qtoks)}")
    print(f"    concepts: {hits}")
    if fp:
        print(f"    ROUTE   : {fp.route}   why={fp.why}")
        print(f"    SQL     : {fp.sql}")
    else:
        print(f"    ROUTE   : (fall-through → full pipeline)")


if __name__ == "__main__":
    _rst = reg.active()
    print(f"registry ready: {reg.is_ready()}  "
          f"(concepts={len(_rst['concepts'])}, "
          f"dimensions={len(_rst['dimensions'])}, "
          f"metrics={len(_rst['metrics'])})")
    for q in [
        "how many incidents",
        "count of incidents that are escalated",
        "number of incidents by status",
        "how many counterparties",
        "what are the incident statuses",
        "incidents with their assigned users",        # expect fall-through (join)
        "counterparties with more than one alias",     # expect fall-through
    ]:
        _diagnose(q)
