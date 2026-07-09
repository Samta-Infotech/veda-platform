# =============================================================================
# query/federated_route.py — cross-source federated NL route (MULTI_SOURCE_SERVING MS-6)
#
# When a query's scope spans ≥ 2 sources and retrieval selects columns from more than
# one of them, the deterministic single-DB SQL head cannot answer (no FK path exists
# across sources — those are `cross_source_fk` graph edges, not schema FKs). This route:
#   1. retrieves the relevant columns across the merged scope,
#   2. builds a catalog-qualified federated schema (src_<id>.<table> for parquet,
#      src_<id>.public.<table> for postgres) + the cross_source_fk JOIN hints,
#   3. asks the SLM for ONE read-only DuckDB SELECT over exactly those qualified names,
#   4. executes it through the verified query.cross_source_composer.compose_federated
#      (firewall + FederatedExecutor + provenance).
# Returns None when it should not / cannot federate, so the caller falls back cleanly.
# =============================================================================
from __future__ import annotations

import re
from typing import Dict, List, Optional

from query.cross_source_composer import (
    should_federate, resolve_surface, compose_federated, compose_federated_plan,
    selected_source_ids,
)
from query.federated_executor import catalog_name
from utils.logger import get_logger

logger = get_logger(__name__)

_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")   # safe bare SQL identifier
# Peripheral/attachment-style tables excluded from the join-path graph — they link to many
# entities and produce coincidental shortest paths instead of the semantic junction.
_PERIPHERAL_TABLE_PATTERNS = (
    "attachment", "document", "_log", "comment", "chats_", "message", "notification",
    "_history", "oauth", "token", "permission", "campaign", "newsletter", "subscription",
    "fcmdevice", "sociallogin", "clientsupport", "inspection", "review",
)


def _catalog_table(sid: str, table: str, kind: str) -> str:
    """Fully-qualified DuckDB name: parquet views live at src_<id>.<table>; a postgres
    ATTACH exposes its tables at src_<id>.public.<table>."""
    cat = catalog_name(sid)
    return f'{cat}.public."{table}"' if kind == "postgres" else f'{cat}."{table}"'


def _selected_by_source(cols) -> Dict[str, Dict[str, list]]:
    """{source_id: {table_name: [col_name,...]}} from retrieval results."""
    out: Dict[str, Dict[str, list]] = {}
    for c in cols or []:
        sid = str(getattr(c, "source_id", "") or "")
        t = getattr(c, "table_name", "") or ""
        col = getattr(c, "col_name", "") or ""
        if not (sid and t and col):
            continue
        out.setdefault(sid, {}).setdefault(t, [])
        if col not in out[sid][t]:
            out[sid][t].append(col)
    return out


def _join_hints(by_source: Dict[str, Dict[str, list]]) -> List[dict]:
    """cross_source_fk edges (HIGH first) between the SELECTED tables — the grounded join
    keys for the federated SELECT. Read from the internal graph store."""
    try:
        from ingestion.db_abstraction import (
            get_internal_connection, release_internal_connection)
    except Exception:
        return []
    wanted = {(sid, t) for sid, tabs in by_source.items() for t in tabs}
    if not wanted:
        return []
    conn = get_internal_connection()
    hints: List[dict] = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ns.source_id, ns.table_name, ns.name,
                       nd.source_id, nd.table_name, nd.name,
                       (e.attrs::jsonb->>'tier')
                FROM graph_edges e
                JOIN graph_nodes ns ON ns.node_id = e.src_node_id
                JOIN graph_nodes nd ON nd.node_id = e.dst_node_id
                WHERE e.edge_type = 'cross_source_fk'
            """)
            for ss, st, sn, ds, dt, dn, tier in cur.fetchall():
                # Include an edge when EITHER endpoint table is in scope — retrieval often
                # surfaces the tabular table (amenities_catalog) but not the specific join-key
                # column (amenity_name) or the homzhub side (assets_amenity); the caller pulls
                # any missing endpoint table + key into the schema.
                if (str(ss), st) in wanted or (str(ds), dt) in wanted:
                    hints.append({"a_src": str(ss), "a_tbl": st, "a_col": sn,
                                  "b_src": str(ds), "b_tbl": dt, "b_col": dn,
                                  "tier": (tier or "MEDIUM")})
    except Exception as e:
        logger.warning("federated_route: join-hint load failed (%s)", e)
    finally:
        release_internal_connection(conn)
    # Prefer execution-grade HIGH edges; if any exist, drop the MEDIUM noise entirely.
    high = [h for h in hints if h["tier"] == "HIGH"]
    chosen = high or hints
    # de-dup on the unordered column pair
    seen, uniq = set(), []
    for h in chosen:
        key = frozenset({(h["a_src"], h["a_tbl"], h["a_col"]), (h["b_src"], h["b_tbl"], h["b_col"])})
        if key not in seen:
            seen.add(key)
            uniq.append(h)
    return _best_target_per_child(uniq)[:6]


def _entity_of(col_name: str) -> str:
    """`asset_id` -> `asset`, `city` -> `city` — the referenced entity token of an FK."""
    c = (col_name or "").lower()
    for suf in ("_id", "_ids", "id"):
        if c.endswith(suf) and len(c) > len(suf):
            return c[: -len(suf)].rstrip("_")
    return c


def _best_target_per_child(hints: List[dict]) -> List[dict]:
    """Toy/asset-centric schemas yield several HIGH targets for one FK (asset_id →
    assets_asset.id BUT ALSO → asset_assignment.id → assets_assetspace.asset_id …), and a 7B
    model picks wrong ones → empty joins. For each child column keep only the best-scoring
    target: the table named for the FK's entity (assets_asset for asset_id) + PK/twin-key."""
    def score(h):
        entity = _entity_of(h["a_col"])
        b_tbl_last = (h["b_tbl"] or "").lower().split("_")[-1]
        s = 0
        if b_tbl_last == entity:               # assets_asset (last='asset') for asset_id
            s += 10
        if (h["b_col"] or "").lower() == (h["a_col"] or "").lower():   # twin FK
            s += 5
        if (h["b_col"] or "").lower() == "id":                        # points at a PK
            s += 3
        if entity and entity in (h["b_tbl"] or "").lower():
            s += 1
        return s
    groups: Dict[tuple, list] = {}
    for h in hints:
        groups.setdefault((h["a_src"], h["a_tbl"], h["a_col"]), []).append(h)
    out: List[dict] = []
    for _child, hs in groups.items():
        hs.sort(key=score, reverse=True)
        best = score(hs[0])
        # keep the top target(s); if nothing scored (no clear winner) keep the first only
        out.extend([h for h in hs if score(h) == best][:1] if best > 0 else hs[:1])
    return out


def _focus_by_source(by_source: Dict[str, Dict[str, list]], hints: List[dict],
                     kinds: Dict[str, str]) -> Dict[str, Dict[str, list]]:
    """Trim the schema handed to the SLM to what a cross-source join can actually use:
    every table from the small tabular sources, plus ONLY the relational (homzhub) tables
    that are cross_source_fk join-hint endpoints. Retrieval over-surfaces dozens of homzhub
    tables (noise a 7B model mis-picks); this keeps the plan grounded in the real join graph."""
    hint_tables = set()
    for h in hints:
        hint_tables.add((h["a_src"], h["a_tbl"]))
        hint_tables.add((h["b_src"], h["b_tbl"]))
    focused: Dict[str, Dict[str, list]] = {}
    for sid, tabs in by_source.items():
        is_tabular = kinds.get(sid) != "postgres"
        for t, cols in tabs.items():
            if is_tabular or (sid, t) in hint_tables:
                focused.setdefault(sid, {})[t] = cols
    return focused or by_source


def _load_relational_fk_edges(source_id: str) -> List[tuple]:
    """Real schema FK edges (fk_to only — NOT the noisy discovered_fk) for a relational
    source, as (t1, c1, t2, c2). These let the planner bridge a cross-source target table
    to the group-key hub through the source's own join graph (e.g. assets_amenity →
    assetamenitygroup → assets_asset)."""
    try:
        from ingestion.db_abstraction import (
            get_internal_connection, release_internal_connection)
    except Exception:
        return []
    conn = get_internal_connection()
    out: List[tuple] = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ns.table_name, ns.name, nd.table_name, nd.name
                FROM graph_edges e
                JOIN graph_nodes ns ON ns.node_id = e.src_node_id
                JOIN graph_nodes nd ON nd.node_id = e.dst_node_id
                WHERE e.edge_type = 'fk_to' AND ns.source_id = %s AND nd.source_id = %s
            """, [str(source_id), str(source_id)])
            for t1, c1, t2, c2 in cur.fetchall():
                if not (t1 and c1 and t2 and c2):
                    continue
                # Drop generic audit joins (created_by/updated_by → users_user): they connect
                # almost every table to users, so a shortest-path BFS would route real joins
                # through them and return garbage (this was making the amenity chain go
                # amenity → users_user → asset instead of through the amenity-group junction).
                cl1, cl2 = c1.lower(), c2.lower()
                if (cl1.endswith(("created_by_id", "updated_by_id"))
                        or cl2.endswith(("created_by_id", "updated_by_id"))
                        or "users_user" in (t1.lower(), t2.lower())):
                    continue
                # Drop PERIPHERAL bridge tables (attachments/documents/logs/comments/…): they
                # touch many entities and let a shortest-path BFS take a coincidental route
                # (amenity → its attachment → an asset-document → asset) that returns 0 rows,
                # instead of the meaningful junction (amenity → assetamenitygroup → asset).
                tl1, tl2 = t1.lower(), t2.lower()
                if any(p in tl1 or p in tl2 for p in _PERIPHERAL_TABLE_PATTERNS):
                    continue
                out.append((t1, c1, t2, c2))
    except Exception:
        return []
    finally:
        release_internal_connection(conn)
    return out


def _fk_path(t_from: str, t_to: str, edges: List[tuple], max_hops: int = 3):
    """Undirected BFS over FK edges → shortest list of (t1,c1,t2,c2) join conditions linking
    t_from to t_to, or None. Skips generic audit joins (created_by/updated_by → users_user)."""
    from collections import deque
    adj: Dict[str, list] = {}
    for (t1, c1, t2, c2) in edges:
        if c1.endswith(("created_by_id", "updated_by_id")) or c2 == "id" and t2 == "users_user":
            continue
        adj.setdefault(t1, []).append((t2, (t1, c1, t2, c2)))
        adj.setdefault(t2, []).append((t1, (t1, c1, t2, c2)))
    q = deque([(t_from, [])])
    seen = {t_from}
    while q:
        t, path = q.popleft()
        if t == t_to:
            return path
        if len(path) >= max_hops:
            continue
        for nt, cond in adj.get(t, []):
            if nt not in seen:
                seen.add(nt)
                q.append((nt, path + [cond]))
    return None


def _augment_intra_source(homzhub_tables: set, source_id: str):
    """Given the homzhub tables that are cross-source join targets, find fk_to paths connecting
    the non-hub ones to the hub (the most-referenced target), so a metric can reach the group
    key through homzhub's own joins. Returns (extra_join_conditions, extra_tables{table:set(cols)})."""
    tables = list(homzhub_tables)
    if len(tables) < 2:
        return [], {}
    edges = _load_relational_fk_edges(source_id)
    if not edges:
        return [], {}
    hub = tables[0]                      # caller passes the hub first
    conds: List[tuple] = []
    extra_tables: Dict[str, set] = {}
    seen_cond = set()
    for t in tables[1:]:
        path = _fk_path(hub, t, edges)
        if not path:
            continue
        for (t1, c1, t2, c2) in path:
            key = (t1, c1, t2, c2)
            if key in seen_cond:
                continue
            seen_cond.add(key)
            conds.append((source_id, t1, c1, source_id, t2, c2))
            for tt, cc in ((t1, c1), (t2, c2)):
                if tt not in homzhub_tables:
                    extra_tables.setdefault(tt, set()).add(cc)
    return conds, extra_tables


def _build_schema_text(by_source: Dict[str, Dict[str, list]], kinds: Dict[str, str]) -> str:
    lines: List[str] = []
    for sid, tabs in by_source.items():
        kind = kinds.get(sid, "parquet")
        for t, cols in tabs.items():
            qname = _catalog_table(sid, t, kind)
            lines.append(f"TABLE {qname} (columns: {', '.join(cols)})")
    return "\n".join(lines)


def _join_text(hints: List[dict], kinds: Dict[str, str]) -> str:
    if not hints:
        return "(no grounded cross-source join key found)"
    out = []
    for h in hints:
        a = _catalog_table(h["a_src"], h["a_tbl"], kinds.get(h["a_src"], "parquet"))
        b = _catalog_table(h["b_src"], h["b_tbl"], kinds.get(h["b_src"], "parquet"))
        out.append(f'{a}."{h["a_col"]}" = {b}."{h["b_col"]}"   [{h["tier"]}]')
    return "\n".join(out)


def _extract_sql(text: str) -> str:
    m = _SQL_FENCE.search(text or "")
    sql = (m.group(1) if m else text or "").strip()
    # keep a single statement, drop a trailing semicolon
    sql = sql.split(";")[0].strip()
    return sql


def _generate_federated_sql(query: str, schema_text: str, join_text: str,
                            prior_sql: str = "", prior_error: str = "") -> str:
    from slm._call_slm import call_slm
    system = (
        "You write ONE read-only DuckDB SQL SELECT statement. Rules: use ONLY the fully-"
        "qualified table names given VERBATIM (including the src_<id>. prefix and quotes); "
        "join sources ONLY on the provided join keys; use INNER JOINs ONLY; do NOT join to a "
        "subquery and do NOT use correlated subqueries or EXISTS — compute every total/count/"
        "average with plain aggregates + GROUP BY over the joined tables (use COUNT(DISTINCT x) "
        "for distinct counts); join ONLY the tables the question needs; "
        "no DDL/DML; no comments; no explanation. Return only the SQL, optionally in a ```sql block.")
    user = (
        f"Question: {query}\n\n"
        f"Available tables:\n{schema_text}\n\n"
        f"Cross-source join keys (JOIN across src_ catalogs ONLY on these):\n{join_text}\n\n"
        f"Write the single DuckDB SELECT that answers the question. Always alias tables, "
        f"GROUP BY the grouping column(s), LIMIT 100.")
    if prior_error:
        # One-shot self-repair: the previous SQL failed to execute — show it + the engine
        # error so the model fixes exactly that (usually: replace a subquery/outer join with
        # a flat inner join + GROUP BY).
        user += (f"\n\nYour previous SQL FAILED with this DuckDB error — rewrite it to avoid "
                 f"the error (flat INNER JOINs + GROUP BY, no subquery joins):\n"
                 f"-- previous SQL --\n{prior_sql}\n-- error --\n{prior_error}")
    raw = call_slm(user, system=system, purpose="federated_sql", temperature=0.0)
    return _extract_sql(raw)


def _column_bearing_sources(source_ids) -> set:
    """Which in-scope sources actually have COLUMN nodes (relational/tabular). Doc-only
    sources (chunks, no columns) are excluded. One cheap indexed query — avoids paying a
    full retrieval pass just to discover a doc-inclusive scope can't federate."""
    try:
        from ingestion.db_abstraction import (
            get_internal_connection, release_internal_connection)
    except Exception:
        return set()
    conn = get_internal_connection()
    try:
        with conn.cursor() as cur:
            ph = ",".join(["%s"] * len(source_ids))
            cur.execute(f"SELECT DISTINCT source_id FROM graph_nodes "
                        f"WHERE node_type='column' AND source_id IN ({ph})",
                        [str(s) for s in source_ids])
            return {str(r[0]) for r in cur.fetchall()}
    except Exception:
        return set()
    finally:
        release_internal_connection(conn)


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str):
    import json
    raw = text or ""
    m = _JSON_FENCE.search(raw)
    if m:
        raw = m.group(1)
    # tolerate leading/trailing prose around the object
    i, j = raw.find("{"), raw.rfind("}")
    if i != -1 and j != -1 and j > i:
        raw = raw[i:j + 1]
    try:
        return json.loads(raw)
    except Exception:
        return None


def _generate_federated_plan(query: str, schema_text: str, join_text: str,
                             prior_error: str = "") -> Optional[dict]:
    """Ask the SLM for an aggregate-pushdown PLAN: a group key + one INDEPENDENT aggregate
    SELECT per metric (each grouped by the key AS <group_by>). Independent per-metric
    aggregation is what avoids join fan-out (double-counted SUMs) — the executor materializes
    each and FULL JOINs on the key. Returns the plan dict, or None to fall back to single-SQL."""
    from slm._call_slm import call_slm
    system = (
        "You output a JSON query plan for a federated read-only DuckDB analysis. Output ONLY "
        "JSON: {\"group_by\": <bare column alias, e.g. \"city\", or null>, \"metrics\": "
        "[{\"alias\": <string>, \"sql\": <one DuckDB SELECT>}]}. HARD RULES for every metric.sql:\n"
        "1. Use ONLY the fully-qualified table names given VERBATIM. INNER JOIN across sources "
        "ONLY on the provided join keys.\n"
        "2. Join ONLY the tables THIS ONE metric needs. NEVER join a table used by a different "
        "metric — extra joins multiply rows and corrupt the aggregate.\n"
        "3. If group_by is set, the SELECT list MUST start with the group key column aliased "
        "to <group_by>, then EXACTLY ONE aggregate AS <alias>; and it MUST end with GROUP BY "
        "that same group key column. One row per group, no fan-out.\n"
        "4. Each independent measure is its OWN metric entry.\n"
        "For a non-aggregation question use group_by=null and one metric = the whole SELECT. "
        "No DDL/DML, no comments.\n"
        "EXAMPLE for \"total maintenance and average vendor rating per city\" with keys "
        "maintenance.asset_id=assets_asset.id and vendors.city=assets_asset.city_name:\n"
        "{\"group_by\":\"city\",\"metrics\":["
        "{\"alias\":\"total_maintenance\",\"sql\":\"SELECT a.city_name AS city, SUM(m.amount) AS "
        "total_maintenance FROM src_4.\\\"maintenance\\\" m INNER JOIN src_2.public.\\\"assets_asset\\\" a "
        "ON m.asset_id = a.id GROUP BY a.city_name\"},"
        "{\"alias\":\"avg_rating\",\"sql\":\"SELECT a.city_name AS city, AVG(v.rating) AS avg_rating "
        "FROM src_4.\\\"vendors\\\" v INNER JOIN src_2.public.\\\"assets_asset\\\" a ON v.city = a.city_name "
        "GROUP BY a.city_name\"}]}")
    user = (
        f"Question: {query}\n\nAvailable tables:\n{schema_text}\n\n"
        f"Cross-source join keys (JOIN across src_ catalogs ONLY on these):\n{join_text}\n\n"
        f"Return the JSON plan. Remember: each metric SELECTs the group key aliased to the "
        f"group_by name, joins only its own tables, GROUP BY the key.")
    if prior_error:
        user += (f"\n\nThe previous plan FAILED with this engine error — fix it (keep each "
                 f"metric a single flat INNER-JOIN aggregate grouped by the key):\n{prior_error}")
    raw = call_slm(user, system=system, purpose="federated_plan", temperature=0.0,
                   json_format=True)
    plan = _extract_json(raw)
    if not isinstance(plan, dict):
        return None
    metrics = plan.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        return None
    # keep only well-formed metric entries with a SELECT
    metrics = [m for m in metrics if isinstance(m, dict) and str(m.get("sql", "")).lower().lstrip().startswith("select")]
    if not metrics:
        return None
    gb = plan.get("group_by")
    if gb and len(metrics) > 1:
        # every metric must PROJECT the group key in its SELECT list (so FULL JOIN USING(<gb>)
        # works) AND group by it. Check the key alias appears in the SELECT clause (before FROM)
        # of each metric; otherwise the plan is broken → fall back to the single-SQL path.
        gbl = str(gb).lower()
        for m in metrics:
            s = str(m.get("sql", "")).lower()
            head = s[:s.find(" from ")] if " from " in s else s
            if gbl not in head or "group by" not in s:
                return None
    plan["metrics"] = metrics
    return plan


# ---------------------------------------------------------------------------
# Deterministic join-path planner — the SLM only PICKS (group column + per metric an
# aggregate/table/column); CODE assembles the join chain from the edge graph (cross-source
# hints ∪ homzhub fk_to paths) via BFS. This makes deep-indirect metrics (e.g. catalog
# amenities per city, a 4-hop many-to-many) correct — the model never writes a join.
# ---------------------------------------------------------------------------
_AGG_OK = {"SUM", "AVG", "MIN", "MAX", "COUNT", "COUNT_DISTINCT"}


def _unified_join_edges(hints: List[dict], kinds: Dict[str, str], rel_sid) -> List[tuple]:
    """All usable join conditions as (qualified_table_a, col_a, qualified_table_b, col_b):
    the cross-source hints PLUS the relational source's own fk_to edges."""
    edges: List[tuple] = []
    for h in hints:
        ta = _catalog_table(h["a_src"], h["a_tbl"], kinds.get(str(h["a_src"]), "parquet"))
        tb = _catalog_table(h["b_src"], h["b_tbl"], kinds.get(str(h["b_src"]), "parquet"))
        edges.append((ta, h["a_col"], tb, h["b_col"]))
    if rel_sid is not None:
        for (t1, c1, t2, c2) in _load_relational_fk_edges(str(rel_sid)):
            edges.append((_catalog_table(rel_sid, t1, "postgres"), c1,
                          _catalog_table(rel_sid, t2, "postgres"), c2))
    return edges


def _bfs_join_conditions(from_tbl: str, to_tbl: str, edges: List[tuple]):
    """Shortest ordered list of join conditions (t1,c1,t2,c2) linking from_tbl→to_tbl, or None."""
    from collections import deque
    if from_tbl == to_tbl:
        return []
    adj: Dict[str, list] = {}
    for (ta, ca, tb, cb) in edges:
        adj.setdefault(ta, []).append((tb, (ta, ca, tb, cb)))
        adj.setdefault(tb, []).append((ta, (ta, ca, tb, cb)))
    q = deque([(from_tbl, [])])
    seen = {from_tbl}
    while q:
        t, path = q.popleft()
        for nt, cond in adj.get(t, []):
            if nt in seen:
                continue
            npath = path + [cond]
            if nt == to_tbl:
                return npath
            seen.add(nt)
            q.append((nt, npath))
    return None


def _assemble_metric_sql(group_table: str, group_col: str, group_alias: str,
                         metric: dict, edges: List[tuple]) -> Optional[str]:
    """Deterministically build ONE per-group aggregate SELECT: aggregate the measure column,
    join measure_table → group_table along the discovered path, GROUP BY the group column.
    Returns None if no join path exists (→ caller falls back / refuses)."""
    measure_tbl = metric.get("table")
    measure_col = metric.get("col")
    agg = str(metric.get("agg", "SUM")).upper().replace(" ", "_")
    if not (measure_tbl and measure_col and agg in _AGG_OK and _IDENT.match(group_col)
            and _IDENT.match(str(group_alias))):
        return None
    path = _bfs_join_conditions(measure_tbl, group_table, edges)
    if path is None:
        return None
    alias: Dict[str, str] = {}

    def al(t):
        if t not in alias:
            alias[t] = f"t{len(alias)}"
        return alias[t]

    al(measure_tbl)                      # t0
    joins: List[str] = []
    for (ta, ca, tb, cb) in path:
        if ta in alias and tb not in alias:
            newt = tb
        elif tb in alias and ta not in alias:
            newt = ta
        else:
            continue                     # both known (cycle edge) — skip, tree path won't hit this
        al(newt)
        joins.append(f'INNER JOIN {newt} {alias[newt]} ON '
                     f'{alias[ta]}."{ca}" = {alias[tb]}."{cb}"')
    if group_table not in alias:
        return None
    m_al, g_al = alias[measure_tbl], alias[group_table]
    expr = (f'COUNT(DISTINCT {m_al}."{measure_col}")' if agg == "COUNT_DISTINCT"
            else f'{agg}({m_al}."{measure_col}")')
    return (f'SELECT {g_al}."{group_col}" AS {group_alias}, {expr} AS "{metric["alias"]}" '
            f'FROM {measure_tbl} {m_al} ' + " ".join(joins) +
            f' GROUP BY {g_al}."{group_col}"')


def _generate_structured_plan(query: str, schema_text: str, join_text: str) -> Optional[dict]:
    """SLM picks only fields (group column + per-metric agg/table/col) — no SQL, no joins."""
    from slm._call_slm import call_slm
    system = (
        "You output a JSON analysis plan. Output ONLY JSON: {\"group_table\": <qualified table "
        "name from the list, e.g. src_2.public.\\\"assets_asset\\\">, \"group_col\": <its column "
        "to group by, e.g. city_name>, \"group_alias\": <short name, e.g. city>, \"metrics\": "
        "[{\"alias\": <name>, \"agg\": <SUM|AVG|MIN|MAX|COUNT|COUNT_DISTINCT>, \"table\": "
        "<qualified table the measure lives in>, \"col\": <measure column>}]}. Pick the "
        "group_table/col the question groups results BY. For each measure pick the ONE table + "
        "column it aggregates and the aggregate function (COUNT_DISTINCT for 'how many distinct/"
        "different X'). Do NOT write SQL and do NOT describe joins — the system computes all "
        "joins. Use table/column names EXACTLY as given.")
    user = (f"Question: {query}\n\nAvailable tables (name → columns):\n{schema_text}\n\n"
            f"(Join keys, for context only — you do NOT write joins:\n{join_text})\n\n"
            f"Return the JSON plan.")
    plan = _extract_json(call_slm(user, system=system, purpose="federated_struct_plan",
                                  temperature=0.0, json_format=True))
    if not isinstance(plan, dict):
        return None
    if not (plan.get("group_table") and plan.get("group_col") and plan.get("group_alias")):
        return None
    metrics = [m for m in (plan.get("metrics") or [])
               if isinstance(m, dict) and m.get("alias") and m.get("table") and m.get("col")]
    if not metrics:
        return None
    plan["metrics"] = metrics
    return plan


def run_federated(query: str, tenant: str, source_ids, verbose: bool = False) -> Optional[dict]:
    """Attempt the federated route. Returns a compose_federated payload dict on success,
    or None to signal 'not federated — use the normal path'."""
    sids = [str(s) for s in (source_ids or [])]
    if len(sids) < 2:
        return None
    # Cheap gate BEFORE the expensive retrieval: federation needs ≥2 column-bearing
    # sources. A scope like [homzhub, docs] can never produce a cross-source SQL join, so
    # skip straight to the normal (RAG/entity-bridge) path without a wasted retrieval pass.
    if len(_column_bearing_sources(sids)) < 2:
        return None
    try:
        from query.retrieval_select import select_retrieval
        sel = select_retrieval(query=query, source_ids=sids, intent="sql", verbose=False)
    except Exception as e:
        logger.warning("federated_route: retrieval failed (%s)", e)
        return None
    cols = getattr(sel, "columns", []) or []
    if not should_federate(cols):
        return None                      # single-source plan → normal path

    by_source = _selected_by_source(cols)
    kinds: Dict[str, str] = {}
    for sid in by_source:
        surf = resolve_surface(sid, tenant)
        kinds[sid] = getattr(surf, "kind", "parquet") if surf is not None else "parquet"

    hints = _join_hints(by_source)
    focused = _focus_by_source(by_source, hints, kinds)   # drop homzhub noise tables
    # Ensure EVERY join-hint endpoint table is in the schema with its key column, even if
    # retrieval didn't select that table/column (it often misses the exact join key).
    for h in hints:
        for s, t, c in ((h["a_src"], h["a_tbl"], h["a_col"]),
                        (h["b_src"], h["b_tbl"], h["b_col"])):
            cur = set(focused.setdefault(str(s), {}).get(t, []))
            focused[str(s)][t] = sorted(cur | {c})

    # Bridge indirect metrics: if ≥2 relational (homzhub) tables are cross-source join
    # targets but aren't directly FK-connected (e.g. assets_amenity vs assets_asset), add the
    # source's own fk_to path between them (through intermediate tables) so a metric can reach
    # the group key. Purely additive — no effect when the targets are already one hub table.
    from collections import Counter as _Counter
    rel_sid = next((s for s in by_source if kinds.get(s) == "postgres"), None)
    if rel_sid is not None:
        tcount = _Counter()
        for h in hints:
            if str(h["a_src"]) == str(rel_sid):
                tcount[h["a_tbl"]] += 1
            if str(h["b_src"]) == str(rel_sid):
                tcount[h["b_tbl"]] += 1
        rel_targets = [t for t, _ in tcount.most_common()]   # hub (most-referenced) first
        if len(rel_targets) >= 2:
            extra_conds, extra_tables = _augment_intra_source(set(rel_targets), str(rel_sid))
            for t, extra_cols in extra_tables.items():
                cur = set(focused.setdefault(str(rel_sid), {}).get(t, []))
                focused[str(rel_sid)][t] = sorted(cur | set(extra_cols))
            hints = hints + [
                {"a_src": a_s, "a_tbl": a_t, "a_col": a_c, "b_src": b_s, "b_tbl": b_t,
                 "b_col": b_c, "tier": "FK"}
                for (a_s, a_t, a_c, b_s, b_t, b_c) in extra_conds]

    schema_text = _build_schema_text(focused, kinds)
    join_text = _join_text(hints, kinds)
    if verbose:
        logger.info("federated_route: %d sources, %d tables (focused from %d), %d join hints",
                    len(focused), sum(len(t) for t in focused.values()),
                    sum(len(t) for t in by_source.values()), len(hints))

    gr = getattr(sel, "graph_result", None)
    chunks = list(getattr(gr, "chunks", None) or []) if gr is not None else []

    # PREFERRED: DETERMINISTIC join-path planner. The SLM only picks the group column + each
    # metric's aggregate/table/column; CODE assembles the joins (BFS over the edge graph), so
    # even deep-indirect metrics (4-hop many-to-many) join correctly.
    struct = _generate_structured_plan(query, schema_text, join_text)
    if struct is not None:
        edges = _unified_join_edges(hints, kinds, rel_sid)
        metric_sqls = []
        for m in struct["metrics"]:
            msql = _assemble_metric_sql(struct["group_table"], struct["group_col"],
                                        struct["group_alias"], m, edges)
            if msql:
                metric_sqls.append({"alias": m["alias"], "sql": msql})
        # Only trust the deterministic plan when EVERY metric got a real join path (no silent
        # dropped/NULL metric). Otherwise fall through to the free-form / refuse path.
        if metric_sqls and len(metric_sqls) == len(struct["metrics"]):
            plan = {"group_by": struct["group_alias"], "metrics": metric_sqls}
            payload = compose_federated_plan(query, plan, cols, chunks, tenant=tenant)
            if isinstance(payload, dict) and payload.get("status") == "ok":
                payload["answer"] = _nl_answer(query, payload.get("result") or {})
                return payload
            if verbose:
                logger.info("federated_route: structured plan failed (%s) — trying free-form",
                            payload.get("reason") if isinstance(payload, dict) else payload)

    # FALLBACK 1: free-form aggregate-pushdown plan (SLM writes each metric's SQL).
    plan = _generate_federated_plan(query, schema_text, join_text)
    if plan is not None:
        multi_metric = len(plan.get("metrics") or []) > 1
        payload = compose_federated_plan(query, plan, cols, chunks, tenant=tenant)
        if isinstance(payload, dict) and payload.get("status") == "exec_error_federated":
            if verbose:
                logger.info("federated_route: plan exec error, retry: %s", payload.get("reason"))
            plan2 = _generate_federated_plan(query, schema_text, join_text,
                                             prior_error=str(payload.get("reason")))
            if plan2 is not None:
                payload = compose_federated_plan(query, plan2, cols, chunks, tenant=tenant)
        if isinstance(payload, dict) and payload.get("status") == "ok":
            payload["answer"] = _nl_answer(query, payload.get("result") or {})
            return payload
        # A MULTI-metric plan that still failed must REFUSE — never fall back to a single flat
        # SELECT, which would fan-out (double-count) or invent an unjoinable path. Fast + honest
        # beats an 85s wrong/erroring flat query. Single-metric plans may still try the flat path.
        if multi_metric:
            return {"status": "refused_federated",
                    "reason": ("could not build a correct per-metric plan for this multi-metric "
                               "cross-source query — a required join path is not available. "
                               + str(payload.get("reason") or "")).strip(),
                    "sources": selected_source_ids(cols)}
        # single-metric plan failed → fall through to the single-SQL attempt below.

    # FALLBACK: single flat SELECT (works for simple joins / when plan JSON couldn't be formed).
    sql = _generate_federated_sql(query, schema_text, join_text)
    if not sql or not sql.lower().lstrip().startswith("select"):
        return {"status": "refused_federated", "reason": "no SELECT generated", "sql": sql,
                "sources": selected_source_ids(cols)}
    payload = compose_federated(query, sql, cols, chunks, tenant=tenant)
    if isinstance(payload, dict) and payload.get("status") == "exec_error_federated":
        sql2 = _generate_federated_sql(query, schema_text, join_text,
                                       prior_sql=sql, prior_error=str(payload.get("reason")))
        if sql2 and sql2.lower().lstrip().startswith("select"):
            payload = compose_federated(query, sql2, cols, chunks, tenant=tenant)
    if isinstance(payload, dict) and payload.get("status") == "ok":
        payload["answer"] = _nl_answer(query, payload.get("result") or {})
    return payload


def _nl_answer(query: str, result: dict) -> str:
    """One short natural-language sentence over the federated rows (best-effort; the rows
    + SQL are the source of truth, so a failure here never blocks the answer)."""
    rows = (result or {}).get("rows") or []
    cols = (result or {}).get("columns") or []
    if not rows:
        return "No matching rows were found across the sources."
    try:
        from slm._call_slm import call_slm
        preview = [dict(zip(cols, [str(v) for v in r.values()])) if isinstance(r, dict) else r
                   for r in rows[:15]]
        msg = (f"Question: {query}\nColumns: {cols}\nRows (up to 15): {preview}\n\n"
               "Answer the question in ONE concise sentence using only these rows. No preamble.")
        return call_slm(msg, purpose="federated_answer", temperature=0.0, num_predict=120).strip()
    except Exception:
        return f"{len(rows)} row(s) returned across sources: columns {', '.join(map(str, cols))}."
