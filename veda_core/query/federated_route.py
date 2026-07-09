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
    should_federate, resolve_surface, compose_federated, selected_source_ids,
)
from query.federated_executor import catalog_name
from utils.logger import get_logger

logger = get_logger(__name__)

_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


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
                if (str(ss), st) in wanted and (str(ds), dt) in wanted:
                    hints.append({"a_src": str(ss), "a_tbl": st, "a_col": sn,
                                  "b_src": str(ds), "b_tbl": dt, "b_col": dn,
                                  "tier": (tier or "MEDIUM")})
    except Exception as e:
        logger.warning("federated_route: join-hint load failed (%s)", e)
    finally:
        release_internal_connection(conn)
    # Prefer execution-grade HIGH edges; if any exist, drop the MEDIUM noise entirely so
    # the SLM isn't tempted into spurious multi-hop EXISTS joins. Cap the list.
    high = [h for h in hints if h["tier"] == "HIGH"]
    chosen = high or hints
    # de-dup on the unordered column pair
    seen, out = set(), []
    for h in chosen:
        key = frozenset({(h["a_src"], h["a_tbl"], h["a_col"]), (h["b_src"], h["b_tbl"], h["b_col"])})
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out[:6]


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


def _generate_federated_sql(query: str, schema_text: str, join_text: str) -> str:
    from slm._call_slm import call_slm
    system = (
        "You write ONE read-only DuckDB SQL SELECT statement. Rules: use ONLY the fully-"
        "qualified table names given VERBATIM (including the src_<id>. prefix and quotes); "
        "join sources ONLY on the provided join keys; use the SIMPLEST join that answers the "
        "question — join ONLY the tables the question needs and do NOT add EXISTS or nested "
        "subqueries unless the question explicitly asks to filter by another table; "
        "no DDL/DML; no comments; no explanation. Return only the SQL, optionally in a ```sql block.")
    user = (
        f"Question: {query}\n\n"
        f"Available tables:\n{schema_text}\n\n"
        f"Cross-source join keys (JOIN across src_ catalogs ONLY on these):\n{join_text}\n\n"
        f"Write the single, simplest DuckDB SELECT that answers the question using the fewest "
        f"tables. Prefer aggregates for totals/counts/averages. Always alias tables. LIMIT 100.")
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
    schema_text = _build_schema_text(by_source, kinds)
    join_text = _join_text(hints, kinds)
    if verbose:
        logger.info("federated_route: %d sources, %d tables, %d join hints",
                    len(by_source), sum(len(t) for t in by_source.values()), len(hints))

    sql = _generate_federated_sql(query, schema_text, join_text)
    if not sql or not sql.lower().lstrip().startswith("select"):
        return {"status": "refused_federated", "reason": "no SELECT generated", "sql": sql,
                "sources": selected_source_ids(cols)}

    gr = getattr(sel, "graph_result", None)
    chunks = list(getattr(gr, "chunks", None) or []) if gr is not None else []
    payload = compose_federated(query, sql, cols, chunks, tenant=tenant)
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
