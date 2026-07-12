"""VEDA · L4b/L4c — deterministic existence, grain pre-aggregation, join orchestration."""
import os, re, sys, time, json, logging, threading
from veda.generation import _resolve_display_column, generate_join_sql
from veda.routing import _name_toks
from veda.runtime import IMPORTANCE_WEIGHTS, JOIN_CONFIDENCE_FLOOR, get_graph
from veda.validation import qualifier_completeness


def existence_mode(query):
    """Classify a query as an existence operation (semi/anti-join), or None if it's
    a full join / grouped aggregate. These are query-grammar operators (with/without/
    how-many) — universal, not domain vocabulary.
      exists           : "counterparties with annotations"
      not_exists       : "counterparties without annotations"
      exists_count     : "how many counterparties have annotations"
      not_exists_count : "how many counterparties have no annotations"
    """
    from config import QUERY_GRAMMAR, QUERY_LANGUAGE
    ql = f" {query.lower()} "

    def has(group):
        for w in QUERY_GRAMMAR.get(group, []):
            if (" " in w and w in ql) or re.search(rf"\b{re.escape(w)}\b", ql):
                return True
        return False

    # quantity ("more than one") and grouping ("per") are HAVING-counts / grouped
    # aggregates, not existence — leave them to the aggregate path. Ratio/percentage
    # ("what PERCENTAGE of X have Y") wants a CASE-WHEN ratio, not a bare EXISTS check,
    # and a ranking word ("which X HAVE the HIGHEST Y") wants GROUP BY + ORDER BY, not
    # a semi-join — both are misclassified by the bare "have"/"with" existence trigger.
    if (has("quantity") or re.search(r"[<>]=?", ql) or has("grouping") or has("ratio")
            or any(re.search(rf"\b{re.escape(w)}\b", ql) for w in QUERY_LANGUAGE["ranking"])):
        return None
    # Possessive / projection phrasing ("X with THEIR Y", "X and their Y names") asks
    # to SHOW the related entity's columns — that's a projection JOIN, not an existence
    # filter. Existence is "entities that HAVE Y" (bare related entity), e.g.
    # "counterparties with annotations". So a possessive kicks it to the join path.
    if re.search(r"\b(their|its|whose|with the)\b", ql):
        return None
    counting = has("counting")
    if has("negation"):
        return "not_exists_count" if counting else "not_exists"
    if has("existence"):
        return "exists_count" if counting else "exists"
    return None


_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
              "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def aggregate_mode(query):
    """Classify a per-anchor child aggregation, or None.

    Grammar-level (no schema vocabulary):
      "counterparties with their annotation count"   -> {"threshold": None}
      "organizations with user count and role count" -> {"threshold": None}
      "counterparties with more than one annotation" -> {"threshold": 1, "op": ">"}
      "roles with at least 3 permissions"            -> {"threshold": 3, "op": ">="}

    Existence (how-many-have / with / without) is classified FIRST by the caller —
    this only sees what existence_mode declined (e.g. possessive counting)."""
    from config import QUERY_GRAMMAR, QUERY_LANGUAGE
    ql = f" {query.lower()} "

    mtop = re.search(r"\btop\s+(\d+)\b", ql)
    top_n = int(mtop.group(1)) if mtop else None

    # A ranking word ("highest", "most"...) without an explicit number ("which
    # projects have the HIGHEST number of tenants") still means "sort by the metric
    # descending" — just without a hard LIMIT. Distinct from top_n (which also caps
    # the row count); ranked=True only adds ORDER BY.
    ranked = bool(re.search(r"\b(?:" + "|".join(re.escape(w) for w in QUERY_LANGUAGE["ranking"])
                            + r")\b", ql))

    m = re.search(r"\b(more than|over|at least|minimum of)\s+(\d+|one|two|three|four|"
                  r"five|six|seven|eight|nine|ten)\b", ql)
    if m:
        n = _NUM_WORDS.get(m.group(2), None)
        n = int(m.group(2)) if n is None else n
        op = ">=" if m.group(1) in ("at least", "minimum of") else ">"
        return {"threshold": n, "op": op, "top_n": top_n, "ranked": ranked}
    if re.search(r"\bmultiple\b", ql):
        return {"threshold": 1, "op": ">", "top_n": top_n, "ranked": ranked}

    # "distribution"/"breakdown" of X across/per Y is grammatically a grouped COUNT
    # of X by Y, same as "how many X per Y" — just phrased as a noun instead of a
    # question. Grouped into "counting" rather than a new top-level QUERY_GRAMMAR
    # class since it resolves to the exact same aggregate shape.
    counting = (any((" " in w and w in ql) or re.search(rf"\b{re.escape(w)}\b", ql)
                    for w in QUERY_GRAMMAR.get("counting", []))
               or bool(re.search(r"\b(distribution|breakdown)\b", ql)))
    if counting or top_n is not None or ranked:  # "top 5"/"highest" implies ranking by count
        return {"threshold": None, "op": None, "top_n": top_n, "ranked": ranked}
    return None


def superlative_mode(query):
    """Grammar-level superlative-by-dimension detection, or None.

    "which category contributes the highest value ..." / "what user has the most ..."
    declares a RANKED AGGREGATION (group by the asked dimension, order by an
    aggregate, LIMIT 1) — never a plain row lookup. Deliberately SEPARATE from
    aggregate_mode: that dict drives the COUNT-based grain planner
    (build_aggregate_sql), and a superlative usually ranks by a MEASURE, so feeding
    it the same shape would silently plan the wrong aggregate. Consumers today use
    this as a routing/trace signal (send the query through join/grain planning
    rather than single-table lookup); the deterministic SUM/MAX grain builder is the
    follow-up. Grammar words only (config.QUERY_GRAMMAR — the language layer), no
    schema vocabulary."""
    from config import QUERY_GRAMMAR
    ql = f" {query.lower()} "
    if not re.search(r"\b(which|what|who|whose)\b", ql):
        return None
    for direction, key in (("max", "superlative_max"), ("min", "superlative_min")):
        for w in QUERY_GRAMMAR.get(key, []):
            if re.search(rf"\b{re.escape(w)}\b", ql):
                return {"superlative": direction, "term": w}
    return None


def grouped_mode(query):
    """Grammar-level grouped-measure breakdown detection, or None.

    "how much does each category contribute" / "total amount per type" declares a
    GROUPED SUM over a measure (GROUP BY dim) — the non-ranked sibling of the
    superlative shape. Deliberately SEPARATE from aggregate_mode (whose dict drives
    the COUNT-based grain planner) and mutually exclusive with superlative_mode
    (ranking owns the query when both could match). Grammar words only
    (config.QUERY_GRAMMAR — the language layer), no schema vocabulary."""
    from config import QUERY_GRAMMAR
    ql = f" {query.lower()} "

    def has(group):
        for w in QUERY_GRAMMAR.get(group, []):
            if (" " in w and w in ql) or re.search(rf"\b{re.escape(w)}\b", ql):
                return True
        return False

    if superlative_mode(query):
        return None                     # ranked aggregation — the superlative planner owns it
    if not (has("grouping") and has("measure_agg")):
        return None
    return {"grouped": True}


def ratio_mode(query):
    """Grammar-level ratio detection, or None.

    "what is the ratio of completed payment value to pending value" declares a
    DIVIDED PAIR of aggregates — SUM(side A) / SUM(side B) on one anchor — never
    a row lookup. Returns the two side phrases verbatim; the ratio planner
    resolves each side to a grounded filter or a measure column (refuse/clarify
    otherwise). Grammar words only (config.QUERY_GRAMMAR — the language layer)."""
    from config import QUERY_GRAMMAR
    words = QUERY_GRAMMAR.get("ratio") or []
    if not words:
        return None
    ql = query.lower().rstrip(" ?.!")
    m = re.search(rf"\b(?:{'|'.join(re.escape(w) for w in words)})\b"
                  rf"\s+(?:of|between)\s+(.+?)\s+(?:to|and|vs)\s+(.+)$", ql)
    if not m:
        return None
    return {"side_a": m.group(1).strip(), "side_b": m.group(2).strip()}


def build_aggregate_sql(anchor, child_specs, sm, threshold=None, op=">",
                        top_n=None, group_col=None, ranked=False):
    """Deterministic pre-aggregation: one CTE per child relation, each grouped by
    its FK to the anchor, then joined to the anchor grain. No LLM, and structurally
    fan-out-free — two child CTEs can never cross-multiply (the 'organizations with
    user count AND incident count' double-count bug is impossible by construction).

    child_specs: [(display_name, edge_or_chain)] — edge_or_chain is either the single
    anchor-touching graph edge (the child side is whichever side isn't the anchor), or
    — for a child several hops from the anchor (_resolve_agg_chain) — an ordered hop
    list from the anchor down to the actual child table. A polymorphic
    predicate is applied INSIDE the child's CTE. threshold turns LEFT JOIN + COALESCE
    into INNER JOIN + HAVING (the 'more than N' filter)."""
    ctes, joins, selects, tables = [], [], [], {anchor}
    metrics = []
    if group_col:
        # Grouped variant: "annotation count per counterparty type" — sum the
        # per-entity CTE counts across the dimension. Still fan-out-free: each
        # CTE is one row per entity before the dimension grouping.
        anchor_sel = [f't0."{group_col}"']
    else:
        display = _resolve_display_column(anchor, sm)
        anchor_sel = [f't0."{display}"'] if display else ['t0.*']

    for i, (name, spec) in enumerate(child_specs):
        cte, al, metric = f"agg_{i}", f"a{i}", f"{name}_count"
        having = (f" HAVING COUNT(*) {op} {int(threshold)}"
                  if threshold is not None else "")
        if isinstance(spec, list):
            # Multi-hop chain (_resolve_agg_chain): anchor -[hop0]->
            # bridge -[hop1..]-> ... -> the actual child/target table. Every hop was
            # already verified to fan TOWARD the child (each far-side row maps to
            # exactly one near-side row), so COUNT(DISTINCT child.id) — not COUNT(*)
            # — makes this safe regardless of how many 1:N hops sit in between.
            hops = spec
            anchor_col = hops[0]["near_col"]
            bridge_table, bridge_col = hops[0]["far_table"], hops[0]["far_col"]
            tables.add(bridge_table)
            from_sql, prev_alias, preds = f'"{bridge_table}" c0', "c0", []
            if hops[0].get("requires_predicate"):
                preds.append(hops[0]["requires_predicate"])
            for j, h in enumerate(hops[1:], start=1):
                alias = f"c{j}"
                tables.add(h["far_table"])
                from_sql += (f' JOIN "{h["far_table"]}" {alias} '
                            f'ON {alias}."{h["far_col"]}" = {prev_alias}."{h["near_col"]}"')
                if h.get("requires_predicate"):
                    preds.append(h["requires_predicate"])
                prev_alias = alias
            pred = f" WHERE {' AND '.join(preds)}" if preds else ""
            ctes.append(f'{cte} AS (SELECT c0."{bridge_col}" AS agg_key, '
                        f'COUNT(DISTINCT {prev_alias}."id") AS {metric} '
                        f'FROM {from_sql}{pred} GROUP BY c0."{bridge_col}"{having})')
        else:
            edge = spec
            if edge["target_table"] == anchor:
                child, child_col, anchor_col = (edge["source_table"],
                                                edge["source_column"], edge["target_column"])
            else:
                child, child_col, anchor_col = (edge["target_table"],
                                                edge["target_column"], edge["source_column"])
            tables.add(child)
            pred = f" WHERE {edge['requires_predicate']}" if edge.get("requires_predicate") else ""
            ctes.append(f'{cte} AS (SELECT "{child_col}" AS agg_key, COUNT(*) AS {metric} '
                        f'FROM "{child}"{pred} GROUP BY "{child_col}"{having})')
        jtype = "JOIN" if threshold is not None else "LEFT JOIN"
        joins.append(f'{jtype} {cte} {al} ON {al}.agg_key = t0."{anchor_col}"')
        metrics.append(metric)
        if group_col:
            selects.append(f"COALESCE(SUM({al}.{metric}), 0) AS {metric}")
        elif threshold is not None:
            selects.append(f"{al}.{metric}")
        else:
            selects.append(f"COALESCE({al}.{metric}, 0) AS {metric}")

    tail = f' GROUP BY t0."{group_col}"' if group_col else ""
    if top_n is not None and metrics:
        tail += f" ORDER BY {metrics[0]} DESC LIMIT {int(top_n)}"
    elif ranked and metrics:
        # A ranking word ("highest"/"most") with NO explicit number still means sort
        # by the metric descending — just without a hard cap beyond the default LIMIT.
        tail += f" ORDER BY {metrics[0]} DESC LIMIT 100"
    else:
        tail += " LIMIT 100"
    sql = (f"WITH {', '.join(ctes)} "
           f"SELECT {', '.join(anchor_sel + selects)} "
           f'FROM "{anchor}" t0 ' + " ".join(joins) + tail)
    return sql, tables


def _resolve_agg_chain(anchor, tgt, graph, max_hops=6):
    """BFS over the FULL relationship graph from `anchor` to `tgt`, traversing an edge
    ONLY in the 'child fans toward anchor' direction — cardinality '1:N' with
    anchor-side=source, 'N:1' with anchor-side=target, or '1:1' either way (an even
    SAFER special case: at most one far-side row per near-side row instead of many).
    Baking this into the search itself — not filtering a path AFTER the fact — means a
    coincidental cross-reference through an unrelated table (e.g. a ledger row that
    happens to carry both an asset_id and a lease_transaction_id independently) is
    never even a candidate hop: it's a dead end the moment its onward edge fails the
    direction check, so a shortest-path cost tie can never prefer a semantically wrong
    route over a genuine containment chain. Lets the grain planner pre-aggregate a
    child several hops from the anchor ('lease transactions per PROJECT' is
    project→asset→leaseunit→leasetransaction), not just a direct anchor-touching
    child, WITHOUT the restrictive named-tables-only intermediate set the main
    join-tree search uses (an unnamed bridge entity like 'asset' is the normal case
    here, not a query-named table). Returns an ordered hop list — each hop a dict of
    near/far table+column — or None if no such safe path exists within `max_hops`."""
    from collections import deque
    if anchor == tgt:
        return []
    by_table = {}
    for e in graph.get("edges", []):
        by_table.setdefault(e["source_table"], []).append(e)
        by_table.setdefault(e["target_table"], []).append(e)
    visited, parent, q = {anchor}, {}, deque([(anchor, 0)])
    while q:
        cur, depth = q.popleft()
        if depth >= max_hops:
            continue
        for e in by_table.get(cur, []):
            if e["source_table"] == cur:
                far_table, near_col, far_col = e["target_table"], e["source_column"], e["target_column"]
                safe = e.get("cardinality") in ("1:N", "1:1")
            elif e["target_table"] == cur:
                far_table, near_col, far_col = e["source_table"], e["target_column"], e["source_column"]
                safe = e.get("cardinality") in ("N:1", "1:1")
            else:
                continue
            if not safe or far_table in visited:
                continue
            visited.add(far_table)
            parent[far_table] = {"near_table": cur, "near_col": near_col,
                                 "far_table": far_table, "far_col": far_col,
                                 "requires_predicate": e.get("requires_predicate")}
            if far_table == tgt:
                q.clear()
                break
            q.append((far_table, depth + 1))
    if tgt not in parent:
        return None
    hops, node = [], tgt
    while node != anchor:
        h = parent[node]
        hops.append(h)
        node = h["near_table"]
    hops.reverse()
    return hops


def build_existence_sql(anchor, edges, mode):
    """Deterministic EXISTS / NOT EXISTS over the anchor (no LLM, no fan-out — a
    semi-join returns each anchor row once). One subquery per anchor-touching edge,
    AND-combined, so a multi-relation query ("permission not assigned to role or user")
    becomes NOT EXISTS(role_permissions) AND NOT EXISTS(user_permissions) instead of
    silently dropping all but the first relation. The child is the edge side that isn't
    the anchor; a polymorphic predicate (if any) is carried INTO its subquery. Each
    subquery is its own scope, so they all reuse alias 'b' without colliding. Accepts a
    single edge dict (back-compat) or a list of edges."""
    if isinstance(edges, dict):
        edges = [edges]
    op = "NOT EXISTS" if mode.startswith("not_exists") else "EXISTS"
    tables = {anchor}
    clauses = []
    for edge in edges:
        if edge["source_table"] == anchor:
            child, child_col, anchor_col = edge["target_table"], edge["target_column"], edge["source_column"]
        else:
            child, child_col, anchor_col = edge["source_table"], edge["source_column"], edge["target_column"]
        tables.add(child)
        pred = ""
        if edge.get("requires_predicate"):
            # rewrite the predicate's table to the right alias: 'a' if it's on the anchor,
            # else 'b' (the child). e.g. "annotation_record.object_type = 'counterparty'".
            p = edge["requires_predicate"]
            ptable = p.split(".", 1)[0].strip()
            alias = "a" if ptable == anchor else "b"
            pred = " AND " + re.sub(r"^\s*\w+\.", f"{alias}.", p)
        subq = (f'SELECT 1 FROM "{child}" b '
                f'WHERE b."{child_col}" = a."{anchor_col}"{pred}')
        clauses.append(f"{op} ({subq})")
    where = " AND ".join(clauses)
    if mode.endswith("count"):
        sql = f'SELECT COUNT(*) AS n FROM "{anchor}" a WHERE {where} LIMIT 100'
    else:
        sql = f'SELECT a.* FROM "{anchor}" a WHERE {where} LIMIT 100'
    return sql, tables


_JUNCTION_CACHE = None


_CONTENT_COL_RE = re.compile(
    r"(_id|_at|_datetime|_epoch|_by|_by_id|_by_group)$|^(id|owned_by|assigned_to)", re.I)
# Structural glue inside compound (snake_case) table NAMES — English connectives, not
# entity words. Stripped only when tokenizing a table name into its entity tokens, so
# "investigation_and_research_counter_party" → {investigation, research, counter, party}
# and a query's "and" can't spuriously match it. Queries themselves are never filtered,
# so "X and Y" still joins (X and Y match their own tables). Language layer, not schema.


def _junction_tables(graph, sm):
    """Pure bridge tables (e.g. user_roles, role_permissions, document_tag_mapping):
    ≥2 outgoing FKs and ≤1 real content column. These may sit on a join PATH as
    connectors; any other unrequested table on a path is contamination (necessity)."""
    global _JUNCTION_CACHE
    if _JUNCTION_CACHE is not None:
        return _JUNCTION_CACHE
    fk_out = {}
    for e in graph.get("edges", []):
        fk_out.setdefault(e["source_table"], set()).add(e["target_table"])
    cols_by_t = {}
    for k in sm.get("columns", {}):
        t, c = k.split(".", 1)
        cols_by_t.setdefault(t, []).append(c)
    junctions = set()
    for t, cols in cols_by_t.items():
        content = [c for c in cols if not _CONTENT_COL_RE.search(c)]
        if len(fk_out.get(t, ())) >= 2 and len(content) <= 1:
            junctions.add(t)
    _JUNCTION_CACHE = junctions
    return junctions


def _match_measure_metric(query):
    """Non-COUNT (SUM/AVG) metric named by the query, or None — a thin wrapper over
    the semantic metric registry's label matcher ('average carpet area' ->
    avg_assets_asset_carpet_area), so 'AVERAGE X PER Y' reuses the SAME compiled
    metric→table→column→function knowledge the single-table measure path already
    trusts, instead of inventing new column-resolution vocabulary."""
    from semantic import registry as reg
    hits = reg.match_metric_labels(query.lower())
    return hits[0][0] if hits else None


def build_measure_by_entity_sql(anchor, metric, chain, sm, ranked=False):
    """'AVERAGE/TOTAL <column> PER <entity>' where the metric's own table is several
    hops from the grouping anchor ('average carpet area per project': metric =
    AVG(assets_asset.carpet_area), chain = project→asset). Structurally the same
    pre-aggregation shape as build_aggregate_sql's multi-hop branch — one CTE walking
    the safe chain (_resolve_agg_chain already verified every hop fans toward the
    child) — just applying the metric's OWN aggregation function to its OWN column
    instead of COUNT(DISTINCT pk). `chain=[]` means the metric already lives on the
    anchor table (plain single-table GROUP BY); returns (None, set()) if the metric's
    expression can't be parsed into a function+column (never emits a guess)."""
    import re as _re
    m = _re.match(r'^(\w+)\(\s*(?:DISTINCT\s+)?[\w]+\.(\w+)\s*\)$', metric["expression"], _re.I)
    if not m:
        return None, set()
    func, col = m.group(1).upper(), m.group(2)
    display = _resolve_display_column(anchor, sm)
    anchor_sel = f't0."{display}"' if display else "t0.*"
    metric_alias = f"{func.lower()}_{col}"
    order = f" ORDER BY {metric_alias} DESC" if ranked else ""
    if not chain:
        sql = (f'SELECT {anchor_sel}, {func}("{col}") AS {metric_alias} FROM "{anchor}" t0 '
               f'GROUP BY {anchor_sel}{order} LIMIT 100')
        return sql, {anchor}
    bridge_table, bridge_col, anchor_col = chain[0]["far_table"], chain[0]["far_col"], chain[0]["near_col"]
    tables = {anchor, bridge_table}
    from_sql, prev_alias = f'"{bridge_table}" c0', "c0"
    for j, h in enumerate(chain[1:], start=1):
        alias = f"c{j}"
        tables.add(h["far_table"])
        from_sql += f' JOIN "{h["far_table"]}" {alias} ON {alias}."{h["far_col"]}" = {prev_alias}."{h["near_col"]}"'
        prev_alias = alias
    sql = (f'WITH agg_0 AS (SELECT c0."{bridge_col}" AS agg_key, '
           f'{func}({prev_alias}."{col}") AS {metric_alias} '
           f'FROM {from_sql} GROUP BY c0."{bridge_col}") '
           f'SELECT {anchor_sel}, agg_0.{metric_alias} '
           f'FROM "{anchor}" t0 LEFT JOIN agg_0 ON agg_0.agg_key = t0."{anchor_col}"{order} LIMIT 100')
    return sql, tables


def _resolve_entity_phrase(phrase, graph_tables, junctions):
    """The single graph table a noun PHRASE names (whole-entity match, same rule as
    the grouped-aggregate `_group_anchor` fallback in try_multitable): every token of
    `phrase` must be a subset of the table's segmented name tokens, preferring the
    SMALLEST such table (most specific match, so a generic word doesn't get pulled
    toward an unrelated, more-specific-sounding table). None if nothing matches."""
    from retrieval.query_enrichment import _singularize as _sg_ep
    toks = {_sg_ep(w) for w in phrase.split() if len(w) > 2}
    if not toks:
        return None
    cands = [(t, len(_name_toks(t))) for t in graph_tables
            if t not in junctions and toks <= _name_toks(t)]
    if not cands:
        return None
    cands.sort(key=lambda x: (x[1], len(x[0])))
    return cands[0][0]


def _try_measure_by_entity(query, sm, all_cols, graph, junctions):
    """Early, narrow check for a single named metric grouped by a single named
    entity, independent of the count-oriented target-selection try_multitable
    otherwise does — two phrasings:
      'AVERAGE/TOTAL <column> PER <entity>'            (entity trails "per"/"by")
      'WHICH <entity> HAVE/GENERATE the HIGHEST <col>'  (entity is the subject)
    Returns an action dict, or None to decline (falls through to the normal
    try_multitable flow) when the phrasing doesn't match, the metric/anchor can't be
    resolved, or the metric is already on the anchor's own table (the single-table
    path's job) — never a wrong guess."""
    import re as _re
    metric = _match_measure_metric(query)
    if metric is None:
        return None
    ql = query.lower()
    graph_tables = set(graph.get("tables", []))
    from config import QUERY_LANGUAGE as _QL_measure
    has_ranking = bool(re.search(r"\b(?:" + "|".join(re.escape(w) for w in _QL_measure["ranking"])
                                 + r")\b", ql))
    anchor = None
    mg = _re.search(r"\b(?:per|by)\s+([a-z_][a-z_ ]{2,40})", ql)
    if mg:
        anchor = _resolve_entity_phrase(mg.group(1).strip(), graph_tables, junctions)
    if anchor is None and has_ranking:
        # No "per/by X" trailer — try "WHICH X have/generate/produce the highest Y",
        # where X (the grouping entity) is the query's SUBJECT. Gated on a ranking
        # word so a plain single-table measure question is never hijacked.
        mwhich = _re.match(r"^\s*which\s+([a-z_][a-z_ ]{2,40}?)\s+(?:have|has|generate[s]?|"
                           r"produce[s]?|show[s]?|contribute[s]?|receive[s]?|earn[s]?)\b", ql)
        if mwhich:
            anchor = _resolve_entity_phrase(mwhich.group(1).strip(), graph_tables, junctions)
    if anchor is None or anchor not in graph_tables:
        return None
    metric_table = metric["source_table"]
    if anchor == metric_table:
        return None
    chain = _resolve_agg_chain(anchor, metric_table, graph)
    if not chain:
        return None
    sql, tables = build_measure_by_entity_sql(anchor, metric, chain, sm, ranked=has_ranking)
    if sql is None:
        return None
    cols = [k.split(".", 1)[1] for k in all_cols if k.split(".", 1)[0] in tables]
    plan = {"join_path": [], "confidence": 1.0, "max_fanout": None, "why": [],
           "unreachable": [], "ambiguous": []}
    return {"action": "aggregate", "sql": sql, "tables": tables, "columns": cols,
            "plan": plan, "anchor": anchor, "metrics": [metric["metric_id"]],
            "threshold": None, "top_n": None, "group_col": None}


def try_multitable(query, results, sm, all_cols, tf, primary=None):
    """Attempt a deterministic multi-table plan. Returns an action dict:
      {action: sql|clarify|refuse|fallback, ...}"""
    from query.join_planner import select_anchor, plan_joins, build_skeleton
    graph = get_graph()
    graph_tables = set(graph.get("tables", []))
    if not graph_tables:
        return {"action": "fallback"}
    junctions = _junction_tables(graph, sm)

    from config import GRAIN_PLANNER_ENABLED as _GPE_measure
    if _GPE_measure:
        _measure = _try_measure_by_entity(query, sm, all_cols, graph, junctions)
        if _measure is not None:
            return _measure

    cols_meta = sm.get("columns", {})
    score = {}
    for r in results:
        t = r.col_id.split(".")[0]
        imp = (cols_meta.get(r.col_id, {}) or {}).get("importance_class", "MEDIUM")
        score[t] = max(score.get(t, 0.0), r.final_score * IMPORTANCE_WEIGHTS.get(imp, 0.6))
    import re as _re
    from query.join_planner import _adjacency, _shortest_path
    from retrieval.query_enrichment import _singularize

    ranked = [t for t in sorted(score, key=score.get, reverse=True) if t in graph_tables][:8]
    # Self-join exception to the two-table minimum: a single retrieved table whose
    # SELF-edge the query names ("organizations and their parent organization") is a
    # legitimate multi-table plan — same table, two occurrences.
    _self_join_possible = False
    if ranked:
        from config import JOIN_TREE_PLANNER_ENABLED as _tree_flag
        if _tree_flag:
            from query.join_planner import _edges_between as _eb, \
                _edge_tokens as _et, _query_tokens as _qt
            _self_join_possible = any(_qt(query) & _et(e)
                                      for e in _eb(graph, ranked[0], ranked[0]))
    # Join-partner recovery: retrieval sometimes surfaces only ONE table for a clear
    # join query ("signal rules with their category" → only signal_rules), which would
    # collapse a MULTI_TABLE intent to single-table. If the query NAMES another graph
    # table that is FK-connected to what we have, add it so the join planner gets a
    # real chance. The planner still validates reachability and refuses if no path.
    if len(ranked) < 2 and not _self_join_possible and ranked:
        _qtoks_rec = {_singularize(w) for w in _re.findall(r"[a-z]+", query.lower()) if len(w) > 2}
        _adj_rec = _adjacency(graph)
        _anchor0 = primary if (primary and primary in graph_tables) else ranked[0]
        _neighbours = {n for n, _ in _adj_rec.get(_anchor0, [])}
        _recovered = [t for t in graph_tables
                      if t not in junctions and t not in ranked
                      and (_name_toks(t) & _qtoks_rec) and t in _neighbours]
        for t in _recovered[:1]:
            ranked.append(t)
    if len(ranked) < 2 and not _self_join_possible:
        return {"action": "fallback"}
    # A junction/bridge table is never the subject of a query — it only connects
    # entities. Exclude it from anchor candidates ("roles with their permission names"
    # must anchor on role, not role_permissions).
    # Anchor = the routed primary table when it's join-viable (a non-junction graph
    # table). Routing already blends semantic + lexical + retrieval and is 100% on the
    # per-table suite, so the join grain must AGREE with it — re-deriving the anchor
    # from raw retrieval scores let an off-topic table (investigation_…_party) win on
    # "counterparties …" and refuse a valid join. Fall back to select_anchor only when
    # the routed primary can't anchor a join (missing / junction).
    # A table whose own NAME is a query token is a legitimate subject/anchor even when
    # retrieval ranked junctions or merely-token-sharing tables above it: "roles with
    # more than 3 permissions" must keep `role` as a candidate, not only
    # user_user_permissions (which just shares the "permission" token) — otherwise the
    # subject loses and a wrong-grain table anchors the join. Only widens the candidate
    # pool (score_anchors still decides); junctions stay excluded. This path runs only
    # when the routed primary can't anchor (junction/missing), so blast radius is small.
    _qtoks_early = {_singularize(w) for w in _re.findall(r"[a-z]+", query.lower()) if len(w) > 2}
    _named_subjects = [t for t in graph_tables
                       if t not in junctions and (_name_toks(t) & _qtoks_early)]
    anchor_cands = list(dict.fromkeys(
        [t for t in ranked[:4] if t not in junctions] + _named_subjects)) or ranked[:4]
    # Grouped aggregate "X count per/by Y": the grain is per-Y, so the table owning the
    # Y dimension must anchor — not the higher-ranked metric table X ("annotation count
    # per counterparty type" must anchor on counterparty_details, not annotation_record).
    # Only for counting/grain queries; EXACT dimension match (alias/business_role/name)
    # avoids hijacking unrelated "by" phrases. A wrong pick finds no fanning edge in the
    # grain branch → falls through to the skeleton (never a wrong answer).
    _group_anchor = None
    if aggregate_mode(query):
        _mg = _re.search(r"\b(?:per|by)\s+([a-z_][a-z_ ]{2,40})", query.lower())
        if _mg:
            _phrase = _mg.group(1).strip()
            for _k, _cm in cols_meta.items():
                _tn, _cn = _k.split(".", 1)
                if (_tn not in graph_tables or _tn in junctions
                        or _cm.get("analytics_role") != "DIMENSION"):
                    continue
                _cands = [a.lower() for a in (_cm.get("aliases") or [])]
                _cands.append(_cn.replace("_", " "))
                if _cm.get("business_role"):
                    _cands.append(_cm["business_role"].lower())
                if any(a == _phrase and len(a) > 3 for a in _cands):
                    _group_anchor = _tn
                    break
        # No dimension-COLUMN match ("per counterparty type") — try a whole-ENTITY
        # match: "per project" means group by the project TABLE itself, however many
        # join hops it takes to reach it from the metric table (_resolve_agg_chain
        # handles the multi-hop CTE). Prefer the SMALLEST name-token superset among
        # matches so a generic phrase word doesn't get pulled toward an unrelated,
        # more-specific-sounding table.
        if _group_anchor is None:
            _phrase_toks = {_singularize(w) for w in _phrase.split() if len(w) > 2} if _mg else set()
            if _phrase_toks:
                _entity_cands = [(_tn, len(_name_toks(_tn))) for _tn in graph_tables
                                 if _tn not in junctions and _phrase_toks <= _name_toks(_tn)]
                if _entity_cands:
                    _entity_cands.sort(key=lambda x: (x[1], len(x[0])))
                    _group_anchor = _entity_cands[0][0]
    if _group_anchor:
        anchor = _group_anchor
    elif primary and primary in graph_tables and primary not in junctions:
        # Routing already blends semantic+lexical+retrieval and is 100% on the
        # per-table suite — trust it as the grain when it's join-viable.
        anchor = primary
    else:
        # No usable routed primary → score candidates over multiple signals and read
        # the top-two MARGIN as confidence. A near-tie means the SUBJECT/grain is
        # genuinely ambiguous; clarify rather than silently emit SQL at the wrong grain.
        from query.join_planner import score_anchors
        from config import (ANCHOR_CONFIDENCE_MARGIN, ANCHOR_CONFIDENCE_GATE,
                            ANCHOR_CONFLICT_MULT)
        ranked_anchors = score_anchors(query, anchor_cands, score, graph=graph, sm=sm)
        if not ranked_anchors:
            return {"action": "fallback"}
        anchor = ranked_anchors[0].table
        if ANCHOR_CONFIDENCE_GATE and len(ranked_anchors) > 1:
            a, b = ranked_anchors[0], ranked_anchors[1]
            # signal conflict: a strictly-earlier-mentioned candidate than the
            # composite winner means position (subject prior) disagrees with the
            # lexical pick → demand a much clearer win before committing.
            earliest = max(ranked_anchors, key=lambda r: r.signals["position"])
            conflict = earliest.signals["position"] > a.signals["position"]
            need = ANCHOR_CONFIDENCE_MARGIN * (ANCHOR_CONFLICT_MULT if conflict else 1.0)
            # Subject-prior agreement: when the top composite anchor is ALSO the uniquely
            # earliest-mentioned entity, the grain is grammatically unambiguous — "roles
            # with more than 3 permissions" is plainly per-role. Commit instead of asking
            # the obvious. Fires only on a CLEAN position win (strictly ahead of #2), so a
            # genuine near-tie still clarifies (refuse-over-guess preserved).
            _pos = sorted((r.signals["position"] for r in ranked_anchors), reverse=True)
            subject_clear = (a.signals["position"] == _pos[0] and _pos[0] > _pos[1])
            if not subject_clear and (a.score - b.score) < need:
                alt = earliest.table if conflict else b.table
                return {"action": "clarify",
                        "msg": (f"ambiguous subject — should rows be per {a.table} "
                                f"or {alt}? (confidence {a.score} vs {b.score})")}

    # A table is "requested" if a token of its name NOT shared with the anchor appears
    # in the query — e.g. "annotation" → annotation_record (vs anchor counterparty_details).
    # This separates "dashboards" from "dashboard_items" and ignores spurious near-dups.
    # word-only tokens (strips punctuation like "annotation.") + singularized
    qtoks = {_singularize(w) for w in _re.findall(r"[a-z]+", query.lower()) if len(w) > 2}
    anchor_toks = _name_toks(anchor)

    def _requested(t):
        distinctive = [d for d in _name_toks(t) if d not in anchor_toks]
        return bool(distinctive) and any(d in qtoks for d in distinctive)

    # Consider score-ranked tables AND any name-mentioned graph table retrieval missed.
    pool = list(dict.fromkeys(ranked + [t for t in graph_tables if _requested(t)]))
    others = [t for t in pool if t != anchor]

    # Tables explicitly NAMED in the query are legitimate hubs to route through
    # ("signal assignments per incident" → incident is a valid intermediate). Combined
    # with junctions, this is the set of tables allowed inside the join tree.
    named_tables = {t for t in graph_tables if _name_toks(t) & qtoks}
    allowed_intermediates = junctions | named_tables | {anchor}

    adj = _adjacency(graph)
    def _reachable(t):
        return _shortest_path(adj, anchor, t, allowed=allowed_intermediates | {t}) is not None

    # A "requested" table name is NOT a join target if the anchor already provides
    # that concept as a column: "incident status and workflow state" wants
    # incident.workflow_state, not a join to the workflow table. Only join when the
    # concept genuinely lives in another table (join necessity, column-level).
    anchor_col_toks = {_singularize(tok) for k in all_cols if k.split(".", 1)[0] == anchor
                       for tok in k.split(".", 1)[1].split("_") if len(tok) > 2}
    anchor_cols = [k.split(".", 1)[1] for k in all_cols if k.split(".", 1)[0] == anchor]
    from config import JOIN_TREE_PLANNER_ENABLED as _jt_flag
    # "organization NAMES" asks for a label; role.organization_id satisfies the
    # token but cannot show a name — in that case the table is NOT satisfied by
    # the anchor and must be joined. Display-word detection is query-grammar.
    _display_ask = bool(qtoks & {"name", "names", "title", "titles",
                                 "label", "labels", "description", "descriptions"})

    def _satisfied_by_anchor(t):
        distinctive = [d for d in _name_toks(t) if d not in anchor_toks]
        if not (distinctive and all(d in anchor_col_toks for d in distinctive)):
            return False
        if _jt_flag and _display_ask:
            for d in distinctive:
                if any(d in c and not c.endswith(("_id", "_by"))
                       for c in anchor_cols):
                    return True       # a non-id anchor column genuinely provides it
            return False              # only id columns cover it → join for the label
        return True

    from config import USE_NEW_TARGET_SELECTION, TARGET_SELECTION, QUERY_GRAMMAR
    if USE_NEW_TARGET_SELECTION:
        # Capability check: "more than one <scalar>" is inexpressible when the noun is a
        # scalar column with no 1:N child relation (separate concern from selection).
        _ql = f" {query.lower()} "
        is_quantity = any((" " in w and w in _ql) or _re.search(rf"\b{_re.escape(w)}\b", _ql)
                          for w in QUERY_GRAMMAR.get("quantity", []))
        if is_quantity:
            child_tables = {e["source_table"] for e in graph.get("edges", [])
                            if e["target_table"] == anchor and e.get("cardinality") == "N:1"}
            child_toks = set().union(*[_name_toks(t) for t in child_tables]) if child_tables else set()
            scalar_nouns = [w for w in qtoks
                            if w in anchor_col_toks and w not in child_toks and w not in anchor_toks]
            if scalar_nouns and not (qtoks & child_toks):
                return {"action": "refuse",
                        "msg": f"'{scalar_nouns[0]}' is a single-valued attribute on {anchor}, not a "
                               f"repeating relation — 'more than one {scalar_nouns[0]}' isn't expressible"}

        # Evidence-based target selection (Stage 1). Returns requested entities only;
        # the JoinPlanner introduces required junctions/hubs downstream.
        from query.target_selection import select_targets
        hi = max(score.values()) if score else 1.0
        retr = {t: (score.get(t, 0.0) / hi if hi else 0.0) for t in set(others) | set(score)}
        tr = select_targets(anchor, others, qtoks=qtoks, anchor_toks=anchor_toks,
                            anchor_col_toks=anchor_col_toks, retrieval=retr,
                            junctions=junctions, name_toks=_name_toks, cfg=TARGET_SELECTION)
        if tr.ambiguous:
            a, b = tr.ambiguous[0]
            return {"action": "refuse",
                    "msg": f"ambiguous which entity is meant — {a.table} ({a.confidence}) vs "
                           f"{b.table} ({b.confidence}); too close to choose"}
        if tr.uncertain and not tr.requested:
            u = tr.uncertain[0]
            return {"action": "refuse",
                    "msg": f"not confident which entity is requested (best {u.table} "
                           f"@ {u.confidence} < accept {TARGET_SELECTION['ACCEPT']})"}
        from config import JOIN_TREE_PLANNER_ENABLED, JOIN_MAX_TARGETS
        _tcap = JOIN_MAX_TARGETS if JOIN_TREE_PLANNER_ENABLED else 2
        targets = [t.table for t in tr.requested if _reachable(t.table)][:_tcap]
        # Display-ask recovery: "X with their Y name" where the anchor only exposes Y_id
        # (an FK). The reranker maps "category" → signal_rules.category_id, starving
        # signal_categories of confidence so select_targets drops it — yet a NAME needs the
        # join. Force FK-linked, reachable, name-requested tables the anchor can't satisfy.
        # (Ports the legacy arm's _display_ask logic into the live target-selection arm.)
        if _display_ask:
            _forced = [t for t in others
                       if t not in targets and _reachable(t)
                       and not _satisfied_by_anchor(t)
                       and ((_name_toks(t) - anchor_toks) & qtoks)]
            targets = (targets + _forced)[:_tcap]
        if not targets:
            unreachable = [t.table for t in tr.requested if not _reachable(t.table)]
            if unreachable:
                return {"action": "refuse",
                        "msg": f"{anchor} and {', '.join(unreachable)} are not directly related "
                               f"in the schema — no join path exists"}
            return {"action": "fallback"}
    else:
        # Legacy boolean target selection (behaviour-preserving; the OLD benchmark arm).
        # A junction/bridge is NEVER a requested entity — it only enters the tree as a
        # planner bridge. Excluding it here stops a query token like "roles" pulling in
        # `user_roles` (which shares the "role" token) as a spurious join target.
        from config import JOIN_TREE_PLANNER_ENABLED, JOIN_MAX_TARGETS
        _tcap = JOIN_MAX_TARGETS if JOIN_TREE_PLANNER_ENABLED else 2
        requested = [t for t in others
                     if _requested(t) and not _satisfied_by_anchor(t) and t not in junctions]
        # Domination filter (tree arm only): among candidates matched by the SAME query
        # tokens, keep the most name-specific one. The old [:2] cap masked this; at cap 5
        # the bare token "user" would otherwise drag in user_prompt / user_profile /
        # user_user_permissions alongside `user` itself.
        if JOIN_TREE_PLANNER_ENABLED and len(requested) > 1:
            _best = {}
            for t in requested:
                _d = [x for x in _name_toks(t) if x not in anchor_toks]
                _m = frozenset(x for x in _d if x in qtoks)
                _cov = (len(_m) / len(_d)) if _d else 0.0
                if _m and (_m not in _best or _cov > _best[_m][1]):
                    _best[_m] = (t, _cov)
            _kept = {v[0] for v in _best.values()}
            requested = [t for t in requested if t in _kept]
        targets = [t for t in requested if _reachable(t)][:_tcap]
        if not targets and JOIN_TREE_PLANNER_ENABLED:
            # Self-join hook: no OTHER entity requested, but the query names one of the
            # anchor's self-relations ("organizations and their parent organization") —
            # the anchor itself becomes the target; the tree planner resolves the
            # self-edge by token match (organizations.parent_id → organizations).
            from query.join_planner import _edges_between, _edge_tokens, _query_tokens
            _selfqt = _query_tokens(query)
            if any(_selfqt & _edge_tokens(e) for e in _edges_between(graph, anchor, anchor)):
                targets = [anchor]
        if not targets:
            requested_unreachable = [t for t in requested if not _reachable(t)]
            if requested_unreachable:
                return {"action": "refuse",
                        "msg": f"{anchor} and {', '.join(requested_unreachable)} "
                               f"are not directly related in the schema — no join path exists"}
            return {"action": "fallback"}

    return _plan_and_build(query, sm, all_cols, tf, graph=graph, junctions=junctions,
                           anchor=anchor, targets=targets,
                           allowed_intermediates=allowed_intermediates)


def build_from_entities(query, sm, all_cols, tf, anchor, targets):
    """Drive the SHARED join engine with EXTERNALLY chosen entities — e.g. the LLM's
    entity selection in the LangGraph path. The LLM never invents joins; the graph-
    verified plan_join_tree does (one join engine for both paths). Same action-dict
    contract as try_multitable. Caller maps its entity ids → table names first."""
    import re as _re
    from veda.routing import _name_toks
    from retrieval.query_enrichment import _singularize
    graph = get_graph()
    if not graph.get("tables") or anchor not in set(graph.get("tables", [])):
        return {"action": "fallback"}
    junctions = _junction_tables(graph, sm)
    qtoks = {_singularize(w) for w in _re.findall(r"[a-z]+", query.lower()) if len(w) > 2}
    named_tables = {t for t in graph.get("tables", []) if _name_toks(t) & qtoks}
    tgts = [t for t in dict.fromkeys(targets) if t]
    allowed_intermediates = junctions | named_tables | {anchor} | set(tgts)
    return _plan_and_build(query, sm, all_cols, tf, graph=graph, junctions=junctions,
                           anchor=anchor, targets=tgts,
                           allowed_intermediates=allowed_intermediates)


def _plan_and_build(query, sm, all_cols, tf, *, graph, junctions, anchor, targets,
                    allowed_intermediates):
    """Shared join ENGINE. Given a chosen anchor + targets, plan the graph-verified
    join tree (plan_join_tree), apply necessity pruning + ambiguity / reachability /
    confidence guards, then build SQL (existence / grain pre-aggregation / projection
    join via build_skeleton + generate_join_sql). Driven by EITHER the deterministic
    selector (try_multitable) OR external entity selection (build_from_entities) — so
    join reasoning lives in ONE place. Returns the same action dict as try_multitable."""
    # Pass the allowed-intermediate set so the planner only routes through junctions,
    # query-named hubs, and the targets themselves (join necessity, in-search).
    from query.join_planner import plan_joins, build_skeleton
    from config import JOIN_TREE_PLANNER_ENABLED as _tree_on
    if _tree_on:
        from query.join_planner import plan_join_tree
        plan = plan_join_tree(anchor, targets, graph, query,
                              allowed_intermediates=allowed_intermediates | set(targets),
                              junctions=junctions)
    else:
        plan = plan_joins(anchor, targets, graph, query,
                          allowed_intermediates=allowed_intermediates | set(targets))

    # Join-necessity guard: every table in the tree must be the anchor, a requested
    # target, or a true bridge BETWEEN needed tables. A real bridge is never a LEAF
    # (it sits in the middle), so iteratively dropping leaf tables that are neither
    # anchor nor target removes any table contributing no projection/filter/grouping —
    # the belt-and-suspenders catch for spurious joins regardless of upstream cause.
    _needed = set(targets) | {anchor}
    _pruned = list(plan["join_path"])
    _changed = True
    while _changed and _pruned:
        _changed = False
        from collections import Counter as _Counter
        _deg = _Counter()
        for _e in _pruned:
            _deg[_e["source_table"]] += 1
            _deg[_e["target_table"]] += 1
        for _e in list(_pruned):
            _leaf = next((s for s in (_e["source_table"], _e["target_table"])
                          if _deg[s] == 1 and s not in _needed), None)
            if _leaf is not None:
                _pruned.remove(_e)
                _changed = True
                break
    if len(_pruned) != len(plan["join_path"]):
        plan["join_path"] = _pruned
        plan["why"] = [w for w in plan.get("why", [])]   # keep explanations intact

    if plan["ambiguous"]:
        a = plan["ambiguous"][0]
        return {"action": "clarify",
                "msg": f"ambiguous join to {a['target']} — which key: {', '.join(a['options'])}?"}
    # A requested target that can only be reached by tunnelling through an unrelated
    # table is refused, NOT silently narrowed to single-table (correctness over
    # coverage) — UNLESS this is a grain/aggregate query ("X count per Y"), where the
    # target sitting several hops away through an UNNAMED bridge entity is the NORMAL
    # case ("lease transactions per project" needs project→asset→leaseunit, and
    # neither bridge table is named in the query). `plan` here was built with the
    # RESTRICTIVE intermediate set (named tables/junctions only), so defer the refuse
    # and let the grain branch below retry with its own safety-integrated BFS
    # (_resolve_agg_chain) over the full graph — only refuse for real if THAT also
    # fails (right before the skeleton build).
    from config import GRAIN_PLANNER_ENABLED as _GPE_early
    _defer_unreachable = bool(plan["unreachable"]) and _GPE_early and bool(aggregate_mode(query))
    if plan["unreachable"] and not _defer_unreachable:
        return {"action": "refuse",
                "msg": f"{anchor} and {', '.join(plan['unreachable'])} are not directly "
                       f"related — a join would have to pass through unrelated tables"}
    if not plan["join_path"] and not _defer_unreachable:
        return {"action": "fallback"}
    if plan["confidence"] < JOIN_CONFIDENCE_FLOOR and not _defer_unreachable:
        return {"action": "refuse",
                "msg": f"join confidence {plan['confidence']} < {JOIN_CONFIDENCE_FLOOR}"}

    # Existence semantics (with / without / how-many-have): deterministic EXISTS /
    # NOT EXISTS — no LLM, no fan-out, returns each anchor once (fixes the duplicate-
    # rows bug of the old INNER-JOIN "with X"). Temporal + existence is ambiguous → clarify.
    mode = existence_mode(query)
    if mode:
        if tf and (tf.start or tf.end):
            return {"action": "clarify",
                    "msg": ("a time window with an existence query is ambiguous — does the window "
                            "apply to the entity or to the related records? please rephrase.")}
        # The SUBJECT (anchor we return) is the entity named EARLIEST in the query
        # ("counterparties with annotations" → counterparties). Singularize so
        # "counterparties" matches the "counterparty" table token.
        from retrieval.query_enrichment import _singularize
        qwords = [_singularize(w) for w in re.findall(r"[a-z]+", query.lower())]
        def _first_pos(t):
            # superset of the legacy underscore tokens: segmentation-aware entity
            # words (paymenttransaction → payment, transaction) plus the raw parts,
            # so short (≤2 char) name parts keep matching exactly as before.
            ttoks = _name_toks(t, sm) | {_singularize(tok) for tok in t.split("_")}
            for i, w in enumerate(qwords):
                if w in ttoks:
                    return i
            return 10 ** 9
        ex_anchor = min([anchor] + list(targets), key=_first_pos)
        # One (NOT) EXISTS per relation that DIRECTLY touches the anchor — so a query
        # naming several relations ("permission not assigned to role or user") becomes
        # NOT EXISTS(role_permissions) AND NOT EXISTS(user_permissions) instead of
        # silently dropping all but the first (which the qualifier gate then refused).
        # Relations reached only multi-hop have no anchor-touching edge here, so they're
        # left out and the projection-join skeleton handles them (correctness over guess).
        ex_edges, _seen_children = [], set()
        for e in plan["join_path"]:
            if ex_anchor not in (e["source_table"], e["target_table"]):
                continue
            child = e["target_table"] if e["source_table"] == ex_anchor else e["source_table"]
            if child in _seen_children:
                continue
            _seen_children.add(child)
            ex_edges.append(e)
        if ex_edges:
            ex_sql, ex_tables = build_existence_sql(ex_anchor, ex_edges, mode)
            ex_cols = [k.split(".", 1)[1] for k in all_cols if k.split(".", 1)[0] in ex_tables]
            return {"action": "existence", "sql": ex_sql, "tables": ex_tables, "columns": ex_cols,
                    "plan": plan, "mode": mode, "anchor": ex_anchor}

    # Grain planner: per-anchor child aggregation ("X with their Y count",
    # "X with more than N Y") → deterministic pre-aggregation CTEs, no LLM.
    # Each child aggregates in its own CTE keyed by its FK, so multiple children
    # can never cross-multiply (the classic double-count). Conditions are strict —
    # every target must hang off a single fanning anchor-touching edge; anything
    # else falls through to the skeleton+LLM path (where the fan-out guard rules).
    from config import GRAIN_PLANNER_ENABLED
    agg = aggregate_mode(query) if GRAIN_PLANNER_ENABLED else None
    if agg:
        anchor_edges = [e for e in plan["join_path"]
                        if anchor in (e["source_table"], e["target_table"])]

        def _other(e):
            return e["target_table"] if e["source_table"] == anchor else e["source_table"]

        def _fans(e):
            if e["target_table"] == anchor:          # child = source side (child.fk → anchor.pk)
                return e.get("cardinality") in ("N:1", "N:M")
            return e.get("cardinality") in ("1:N", "N:M")

        specs, agg_ok = [], True
        for tgt in targets:
            if tgt == anchor:                        # self-join → not an aggregation target
                agg_ok = False
                break
            direct = next((e for e in anchor_edges if _other(e) == tgt), None)
            if direct is None and len(anchor_edges) == 1 and len(targets) == 1:
                # junction hop: counting role_permissions rows per role IS the
                # permission count — the relationship grain lives on the bridge.
                direct = anchor_edges[0]
            if direct is not None and _fans(direct):
                specs.append((tgt, direct))
                continue
            # No direct anchor-touching fan-out edge — try a multi-hop chain ("lease
            # transactions per PROJECT" is project→asset→leaseunit→leasetransaction,
            # 3 hops). plan["join_path"] was built with the RESTRICTIVE intermediate
            # set (junctions/named-tables only), which excludes unnamed bridge
            # entities like assets_asset here — so search the FULL graph instead, via
            # _resolve_agg_chain's safety-integrated BFS (never affects the
            # skeleton/LLM path's own `plan`/join_path).
            hops = _resolve_agg_chain(anchor, tgt, graph)
            if not hops:
                agg_ok = False
                break
            specs.append((tgt, hops))
        if agg_ok and specs:
            # Optional grouping dimension on the ANCHOR ("annotation count per
            # counterparty type"). Alias/phrase match only — no token guessing;
            # no match → per-entity listing (today's behavior).
            group_col = None
            mgrp = re.search(r"\b(?:per|by)\s+([a-z_][a-z_ ]{2,40})", query.lower())
            if mgrp:
                phrase = mgrp.group(1).strip()
                for k, cmeta in sm.get("columns", {}).items():
                    tname, cname = k.split(".", 1)
                    if tname != anchor or cmeta.get("analytics_role") != "DIMENSION":
                        continue
                    cands = [a.lower() for a in (cmeta.get("aliases") or [])]
                    cands.append(cname.replace("_", " "))
                    if cmeta.get("business_role"):
                        cands.append(cmeta["business_role"].lower())
                    # EXACT phrase only — substring matching let "counterparty"
                    # (an alias of party_name) hijack "per counterparty type".
                    # business_role included: "Counterparty Type" is exactly how
                    # users name individual_or_entity_type.
                    if any(a == phrase and len(a) > 3 for a in cands):
                        group_col = cname
                        break
            agg_sql, agg_tables = build_aggregate_sql(
                anchor, specs, sm, threshold=agg["threshold"], op=agg["op"] or ">",
                top_n=agg.get("top_n"), group_col=group_col, ranked=agg.get("ranked", False))
            agg_cols = [k.split(".", 1)[1] for k in all_cols
                        if k.split(".", 1)[0] in agg_tables]
            return {"action": "aggregate", "sql": agg_sql, "tables": agg_tables,
                    "columns": agg_cols, "plan": plan, "anchor": anchor,
                    "metrics": [f"{t}_count" for t, _ in specs],
                    "threshold": agg["threshold"], "top_n": agg.get("top_n"),
                    "group_col": group_col}

    if _defer_unreachable:
        # The grain branch's wide-search chain resolution also failed (or GRAIN_PLANNER
        # declined for another reason) — the original restrictive plan really can't
        # reach the target, and building a skeleton from it would be wrong/incomplete.
        return {"action": "refuse",
                "msg": f"{anchor} and {', '.join(plan['unreachable'])} are not directly "
                       f"related — a join would have to pass through unrelated tables"}

    skeleton, alias_map = build_skeleton(plan)     # alias_map: alias -> table (occurrence-keyed)
    # Alias-qualified join keys from OUR OWN skeleton (deterministic format), for
    # ON-integrity. Name-only pairs collide when two joins share column names
    # (both ON ..."id"); alias-qualified pairs ({t1.id, t0.from_state_id}) don't.
    qualified_key_pairs = [
        frozenset({f"{m.group(1)}.{m.group(2)}", f"{m.group(3)}.{m.group(4)}"})
        for ln in skeleton.splitlines()
        for m in [re.search(r'ON (\w+)\."([^"]+)" = (\w+)\."([^"]+)"', ln)] if m
    ]
    sql = generate_join_sql(query, skeleton, alias_map, sm, tf)
    allowed_tables = set(alias_map.values())
    allowed_columns = [k.split(".", 1)[1] for k in all_cols
                       if k.split(".", 1)[0] in allowed_tables]
    # parent-side tables (the "1" side of a 1:N / both sides of N:M) → fan-out risk
    parent_tables = set()
    for e in plan["join_path"]:
        c = e.get("cardinality")
        if c == "N:1":
            parent_tables.add(e["target_table"])
        elif c == "1:N":
            parent_tables.add(e["source_table"])
        elif c == "N:M":
            parent_tables.update([e["source_table"], e["target_table"]])
    # occurrence-keyed alias_map: a parent table may sit under several aliases
    # (same-table-twice) — every one of them is fan-out-relevant.
    parent_aliases = {al for al, t in alias_map.items() if t in parent_tables}
    # Bare column names owned ONLY by a parent-side table (absent from every child/
    # anchor table in this join) — lets the fan-out guard also catch an unqualified
    # COUNT(col) when the LLM drops the alias. Restricted to parent-exclusive names
    # so an ambiguous name shared with a child can never trigger a false reject.
    _parent_cols    = {k.split(".", 1)[1] for k in all_cols
                       if k.split(".", 1)[0] in parent_tables}
    _nonparent_cols = {k.split(".", 1)[1] for k in all_cols
                       if k.split(".", 1)[0] in allowed_tables
                       and k.split(".", 1)[0] not in parent_tables}
    parent_only_cols = _parent_cols - _nonparent_cols
    return {"action": "sql", "sql": sql, "tables": allowed_tables,
            "columns": allowed_columns, "plan": plan, "skeleton": skeleton,
            "parent_aliases": parent_aliases, "parent_only_cols": parent_only_cols,
            "qualified_key_pairs": qualified_key_pairs}


# Vocabulary that is NOT a content qualifier: aggregation/list verbs, relationship
# verbs, query glue, temporal words, number words, and the grammar operators. Anything
# in the query that ISN'T one of these is a content token (an entity, attribute, or
# VALUE) that the SQL must account for — otherwise the SQL silently answers a broader
# question. (See qualifier_completeness.)
