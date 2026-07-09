# =============================================================================
# ingestion/graph_persist.py
# VEDA — Unified Data Graph: Phase 1 (Persistence Layer)
#
# Responsibility:
#   - Persists the in-memory REGGraph (table/column nodes + has_column/fk_to
#     edges) into two relational tables: graph_nodes and graph_edges
#   - Adds discovered-FK edges from the Data Graph (high/medium certainty)
#   - Provides query-time accessors: get_nodes, get_neighbors
#   - In-memory fallback when INTERNAL_DB_AVAILABLE is False
#
# All operations are idempotent — scoped DELETE by source_id before insert.
# This module is only invoked when UNIFIED_GRAPH_ENABLED + GRAPH_PERSIST_ENABLED.
#
# Node id conventions (stable across runs):
#   column → "col:<col_id>"
#   table  → "tbl:<table_id>"
#   chunk  → "chunk:<chunk_id>"
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
from uuid import uuid4
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from ingestion.db_abstraction import (
    INTERNAL_DB_AVAILABLE,
    get_internal_connection,
    release_internal_connection,
    DICT_CURSOR,
)
from config import (
    GRAPH_NODES_TABLE,
    GRAPH_EDGES_TABLE,
    GRAPH_EDGE_WEIGHTS,
    GRAPH_DISCOVERED_FK_TIER_WEIGHT,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# In-memory fallback stores
# =============================================================================

_IN_MEMORY_NODES: List[dict] = []
_IN_MEMORY_EDGES: List[dict] = []


# =============================================================================
# Output data structures
# =============================================================================

@dataclass
class GraphNode:
    node_id: str
    node_type: str
    source_id: str
    ref_id: str
    name: str = ""
    table_id: Optional[str] = None
    table_name: Optional[str] = None
    semantic_type: Optional[str] = None
    data_type: Optional[str] = None
    is_pk: bool = False
    is_fk: bool = False
    attrs: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    edge_id: str
    src_node_id: str
    dst_node_id: str
    edge_type: str
    weight: float
    source_id: str
    evidence: str = ""
    attrs: dict = field(default_factory=dict)


@dataclass
class GraphPersistResult:
    nodes_written: int
    edges_written: int
    source_id: str
    backend: str
    duration_sec: float
    stats: dict = field(default_factory=dict)


# =============================================================================
# Node id helpers
# =============================================================================

def col_node_id(col_id: str) -> str:
    return f"col:{col_id}"


def tbl_node_id(table_id: str) -> str:
    return f"tbl:{table_id}"


def chunk_node_id(chunk_id: str) -> str:
    return f"chunk:{chunk_id}"


# =============================================================================
# Schema management
# =============================================================================

def _create_graph_tables(cursor) -> None:
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {GRAPH_NODES_TABLE} (
            node_id       TEXT PRIMARY KEY,
            node_type     TEXT NOT NULL,
            source_id     TEXT NOT NULL,
            ref_id        TEXT NOT NULL,
            table_id      TEXT,
            name          TEXT,
            table_name    TEXT,
            semantic_type TEXT,
            data_type     TEXT,
            is_pk         BOOLEAN DEFAULT FALSE,
            is_fk         BOOLEAN DEFAULT FALSE,
            attrs         TEXT
        );
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{GRAPH_NODES_TABLE}_source
        ON {GRAPH_NODES_TABLE} (source_id);
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{GRAPH_NODES_TABLE}_type
        ON {GRAPH_NODES_TABLE} (node_type);
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{GRAPH_NODES_TABLE}_ref
        ON {GRAPH_NODES_TABLE} (ref_id);
    """)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {GRAPH_EDGES_TABLE} (
            edge_id     TEXT PRIMARY KEY,
            src_node_id TEXT NOT NULL,
            dst_node_id TEXT NOT NULL,
            edge_type   TEXT NOT NULL,
            weight      DOUBLE PRECISION DEFAULT 1.0,
            source_id   TEXT NOT NULL,
            evidence    TEXT,
            attrs       TEXT
        );
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{GRAPH_EDGES_TABLE}_src
        ON {GRAPH_EDGES_TABLE} (src_node_id);
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{GRAPH_EDGES_TABLE}_dst
        ON {GRAPH_EDGES_TABLE} (dst_node_id);
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{GRAPH_EDGES_TABLE}_type
        ON {GRAPH_EDGES_TABLE} (edge_type);
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{GRAPH_EDGES_TABLE}_source
        ON {GRAPH_EDGES_TABLE} (source_id);
    """)
    cursor.execute(f"""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_{GRAPH_EDGES_TABLE}_triple
        ON {GRAPH_EDGES_TABLE} (src_node_id, dst_node_id, edge_type);
    """)


# =============================================================================
# Upsert helpers
# =============================================================================

def upsert_nodes(nodes: List[GraphNode], verbose: bool = False) -> int:
    """Upserts graph nodes. Returns number written. DB or in-memory fallback."""
    if not nodes:
        return 0

    if not INTERNAL_DB_AVAILABLE:
        global _IN_MEMORY_NODES
        existing = {n["node_id"]: n for n in _IN_MEMORY_NODES}
        for n in nodes:
            existing[n.node_id] = {
                "node_id":       n.node_id,
                "node_type":     n.node_type,
                "source_id":     n.source_id,
                "ref_id":        n.ref_id,
                "table_id":      n.table_id,
                "name":          n.name,
                "table_name":    n.table_name,
                "semantic_type": n.semantic_type,
                "data_type":     n.data_type,
                "is_pk":         n.is_pk,
                "is_fk":         n.is_fk,
                "attrs":         dict(n.attrs),
            }
        _IN_MEMORY_NODES = list(existing.values())
        return len(nodes)

    written = 0
    conn = get_internal_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                _create_graph_tables(cur)
                for n in nodes:
                    try:
                        cur.execute(f"""
                            INSERT INTO {GRAPH_NODES_TABLE}
                                (node_id, node_type, source_id, ref_id, table_id,
                                 name, table_name, semantic_type, data_type,
                                 is_pk, is_fk, attrs)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (node_id) DO UPDATE SET
                                node_type     = EXCLUDED.node_type,
                                source_id     = EXCLUDED.source_id,
                                ref_id        = EXCLUDED.ref_id,
                                table_id      = EXCLUDED.table_id,
                                name          = EXCLUDED.name,
                                table_name    = EXCLUDED.table_name,
                                semantic_type = EXCLUDED.semantic_type,
                                data_type     = EXCLUDED.data_type,
                                is_pk         = EXCLUDED.is_pk,
                                is_fk         = EXCLUDED.is_fk,
                                attrs         = EXCLUDED.attrs;
                        """, (
                            n.node_id, n.node_type, n.source_id, n.ref_id, n.table_id,
                            n.name, n.table_name, n.semantic_type, n.data_type,
                            n.is_pk, n.is_fk, json.dumps(n.attrs or {}),
                        ))
                        written += 1
                    except Exception:
                        logger.warning("node insert skipped: %s", n.node_id)
    finally:
        release_internal_connection(conn)
    return written


def upsert_edges(edges: List[GraphEdge], verbose: bool = False) -> int:
    """Upserts graph edges. Returns number written. DB or in-memory fallback."""
    if not edges:
        return 0

    if not INTERNAL_DB_AVAILABLE:
        global _IN_MEMORY_EDGES
        keyed = {
            (e["src_node_id"], e["dst_node_id"], e["edge_type"]): e
            for e in _IN_MEMORY_EDGES
        }
        for e in edges:
            keyed[(e.src_node_id, e.dst_node_id, e.edge_type)] = {
                "edge_id":     e.edge_id,
                "src_node_id": e.src_node_id,
                "dst_node_id": e.dst_node_id,
                "edge_type":   e.edge_type,
                "weight":      e.weight,
                "source_id":   e.source_id,
                "evidence":    e.evidence,
                "attrs":       dict(e.attrs),
            }
        _IN_MEMORY_EDGES = list(keyed.values())
        return len(edges)

    written = 0
    conn = get_internal_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                _create_graph_tables(cur)
                for e in edges:
                    try:
                        cur.execute(f"""
                            INSERT INTO {GRAPH_EDGES_TABLE}
                                (edge_id, src_node_id, dst_node_id, edge_type,
                                 weight, source_id, evidence, attrs)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (src_node_id, dst_node_id, edge_type) DO UPDATE SET
                                weight    = EXCLUDED.weight,
                                source_id = EXCLUDED.source_id,
                                evidence  = EXCLUDED.evidence,
                                attrs     = EXCLUDED.attrs;
                        """, (
                            e.edge_id, e.src_node_id, e.dst_node_id, e.edge_type,
                            e.weight, e.source_id, e.evidence, json.dumps(e.attrs or {}),
                        ))
                        written += 1
                    except Exception:
                        logger.warning("edge insert skipped: %s->%s", e.src_node_id, e.dst_node_id)
    finally:
        release_internal_connection(conn)
    return written


# =============================================================================
# Main persist entry point
# =============================================================================

def persist_reg_graph(
    graph,
    scan_result,
    dg_result=None,
    source_id: str = "",
    verbose: bool = False,
) -> GraphPersistResult:
    """
    Persists a REGGraph into graph_nodes / graph_edges.

    Parameters
    ----------
    graph       : REGGraph (table_nodes, column_nodes, has_column_edges, fk_to_edges)
    scan_result : schema scan result (accepted for signature stability)
    dg_result   : DataGraphResult — discovered FK edges (optional)
    source_id   : scopes this graph's nodes/edges
    verbose     : print progress
    """
    t0 = time.time()

    # ------------------------------------------------------------------
    # Build nodes
    # ------------------------------------------------------------------
    nodes: List[GraphNode] = []

    for tn in graph.table_nodes:
        nodes.append(GraphNode(
            node_id    = tbl_node_id(tn.table_id),
            node_type  = "table",
            source_id  = source_id,
            ref_id     = tn.table_id,
            name       = tn.table_name,
            table_id   = tn.table_id,
            table_name = tn.table_name,
            attrs      = {
                "row_count": getattr(tn, "row_count", 0),
                "col_count": getattr(tn, "col_count", 0),
            },
        ))

    for cn in graph.column_nodes:
        nodes.append(GraphNode(
            node_id       = col_node_id(cn.col_id),
            node_type     = "column",
            source_id     = source_id,
            ref_id        = cn.col_id,
            name          = cn.col_name,
            table_id      = cn.table_id,
            table_name    = cn.table_name,
            semantic_type = cn.semantic_type,
            data_type     = cn.data_type,
            is_pk         = bool(cn.is_pk),
            is_fk         = bool(cn.is_fk),
            attrs         = {},
        ))

    # ------------------------------------------------------------------
    # Build edges
    # ------------------------------------------------------------------
    edges: List[GraphEdge] = []

    # has_column : (table_idx, col_idx)
    for (t_idx, c_idx) in graph.has_column_edges:
        try:
            t_node = graph.table_nodes[t_idx]
            c_node = graph.column_nodes[c_idx]
        except (IndexError, TypeError):
            continue
        edges.append(GraphEdge(
            edge_id     = str(uuid4()),
            src_node_id = tbl_node_id(t_node.table_id),
            dst_node_id = col_node_id(c_node.col_id),
            edge_type   = "has_column",
            weight      = GRAPH_EDGE_WEIGHTS["has_column"],
            source_id   = source_id,
            evidence    = "schema",
        ))

    # fk_to : (from_col_idx, to_col_idx)
    for (from_idx, to_idx) in graph.fk_to_edges:
        try:
            from_node = graph.column_nodes[from_idx]
            to_node   = graph.column_nodes[to_idx]
        except (IndexError, TypeError):
            continue
        edges.append(GraphEdge(
            edge_id     = str(uuid4()),
            src_node_id = col_node_id(from_node.col_id),
            dst_node_id = col_node_id(to_node.col_id),
            edge_type   = "fk_to",
            weight      = GRAPH_EDGE_WEIGHTS["fk_to"],
            source_id   = source_id,
            evidence    = "declared_fk",
        ))

    # discovered_fk : from Data Graph high + medium certainty (skip SOFT)
    discovered_count = 0
    if dg_result is not None:
        discovered = list(getattr(dg_result, "high_certainty", []) or []) + \
                     list(getattr(dg_result, "medium_certainty", []) or [])
        for de in discovered:
            certainty = getattr(de, "certainty", "MEDIUM")
            tier = GRAPH_DISCOVERED_FK_TIER_WEIGHT.get(certainty)
            if tier is None:
                continue
            weight = GRAPH_EDGE_WEIGHTS["discovered_fk"] * tier
            edges.append(GraphEdge(
                edge_id     = str(uuid4()),
                src_node_id = col_node_id(de.from_col_id),
                dst_node_id = col_node_id(de.to_col_id),
                edge_type   = "discovered_fk",
                weight      = weight,
                source_id   = source_id,
                evidence    = getattr(de, "evidence", ""),
                attrs       = {
                    "certainty":     certainty,
                    "overlap_score": getattr(de, "overlap_score", 0.0),
                },
            ))
            discovered_count += 1

    # ------------------------------------------------------------------
    # Scoped delete then insert (idempotent)
    # ------------------------------------------------------------------
    backend = "in_memory_fallback"
    if INTERNAL_DB_AVAILABLE:
        try:
            conn = get_internal_connection()
            try:
                with conn:
                    with conn.cursor() as cur:
                        _create_graph_tables(cur)
                        cur.execute(
                            f"DELETE FROM {GRAPH_EDGES_TABLE} WHERE source_id = %s "
                            f"AND edge_type IN ('has_column','fk_to','discovered_fk');",
                            (source_id,),
                        )
                        cur.execute(
                            f"DELETE FROM {GRAPH_NODES_TABLE} WHERE source_id = %s "
                            f"AND node_type IN ('table','column');",
                            (source_id,),
                        )
            finally:
                release_internal_connection(conn)
            backend = "postgres"
        except Exception as e:
            logger.warning("scoped delete failed (%s) — in-memory fallback", e)
            backend = "in_memory_fallback"
    else:
        # scoped delete in-memory
        global _IN_MEMORY_NODES, _IN_MEMORY_EDGES
        _IN_MEMORY_NODES = [
            n for n in _IN_MEMORY_NODES
            if not (n["source_id"] == source_id and n["node_type"] in ("table", "column"))
        ]
        _IN_MEMORY_EDGES = [
            e for e in _IN_MEMORY_EDGES
            if not (e["source_id"] == source_id
                    and e["edge_type"] in ("has_column", "fk_to", "discovered_fk"))
        ]

    nodes_written = upsert_nodes(nodes, verbose=verbose)
    edges_written = upsert_edges(edges, verbose=verbose)
    if not INTERNAL_DB_AVAILABLE:
        backend = "in_memory_fallback"

    duration = round(time.time() - t0, 4)
    if verbose:
        logger.info(
            "%d nodes, %d edges (%d discovered_fk), backend=%s, %ss",
            nodes_written, edges_written, discovered_count, backend, duration,
        )

    return GraphPersistResult(
        nodes_written = nodes_written,
        edges_written = edges_written,
        source_id     = source_id,
        backend       = backend,
        duration_sec  = duration,
        stats         = {
            "table_nodes":   len(graph.table_nodes),
            "column_nodes":  len(graph.column_nodes),
            "has_column":    len(graph.has_column_edges),
            "fk_to":         len(graph.fk_to_edges),
            "discovered_fk": discovered_count,
        },
    )


# =============================================================================
# Query-time accessors
# =============================================================================

def _row_to_node(row) -> GraphNode:
    raw_attrs = row["attrs"]
    try:
        attrs = json.loads(raw_attrs) if raw_attrs else {}
    except Exception:
        attrs = {}
    return GraphNode(
        node_id       = row["node_id"],
        node_type     = row["node_type"],
        source_id     = row["source_id"],
        ref_id        = row["ref_id"],
        table_id      = row["table_id"],
        name          = row["name"] or "",
        table_name    = row["table_name"],
        semantic_type = row["semantic_type"],
        data_type     = row["data_type"],
        is_pk         = bool(row["is_pk"]),
        is_fk         = bool(row["is_fk"]),
        attrs         = attrs,
    )


def _dict_to_node(d: dict) -> GraphNode:
    return GraphNode(
        node_id       = d["node_id"],
        node_type     = d["node_type"],
        source_id     = d["source_id"],
        ref_id        = d["ref_id"],
        table_id      = d.get("table_id"),
        name          = d.get("name") or "",
        table_name    = d.get("table_name"),
        semantic_type = d.get("semantic_type"),
        data_type     = d.get("data_type"),
        is_pk         = bool(d.get("is_pk")),
        is_fk         = bool(d.get("is_fk")),
        attrs         = dict(d.get("attrs") or {}),
    )


def get_nodes(node_ids: List[str]) -> List[GraphNode]:
    """Fetches GraphNode objects by node_id. DB or in-memory fallback."""
    if not node_ids:
        return []

    if not INTERNAL_DB_AVAILABLE:
        wanted = set(node_ids)
        return [_dict_to_node(n) for n in _IN_MEMORY_NODES if n["node_id"] in wanted]

    try:
        conn = get_internal_connection()
    except Exception:
        wanted = set(node_ids)
        return [_dict_to_node(n) for n in _IN_MEMORY_NODES if n["node_id"] in wanted]
    try:
        cur = conn.cursor(cursor_factory=DICT_CURSOR)
        placeholders = ",".join(["%s"] * len(node_ids))
        cur.execute(f"""
            SELECT node_id, node_type, source_id, ref_id, table_id, name,
                   table_name, semantic_type, data_type, is_pk, is_fk, attrs
            FROM {GRAPH_NODES_TABLE}
            WHERE node_id IN ({placeholders});
        """, list(node_ids))
        rows = cur.fetchall()
        try: cur.close()
        except Exception: pass
    finally:
        release_internal_connection(conn)

    return [_row_to_node(r) for r in rows]


def _row_to_edge(row) -> GraphEdge:
    raw_attrs = row["attrs"]
    try:
        attrs = json.loads(raw_attrs) if raw_attrs else {}
    except Exception:
        attrs = {}
    return GraphEdge(
        edge_id     = row["edge_id"],
        src_node_id = row["src_node_id"],
        dst_node_id = row["dst_node_id"],
        edge_type   = row["edge_type"],
        weight      = float(row["weight"]),
        source_id   = row["source_id"],
        evidence    = row["evidence"] or "",
        attrs       = attrs,
    )


def _dict_to_edge(d: dict) -> GraphEdge:
    return GraphEdge(
        edge_id     = d["edge_id"],
        src_node_id = d["src_node_id"],
        dst_node_id = d["dst_node_id"],
        edge_type   = d["edge_type"],
        weight      = float(d["weight"]),
        source_id   = d["source_id"],
        evidence    = d.get("evidence") or "",
        attrs       = dict(d.get("attrs") or {}),
    )


def get_node_degrees(node_ids: List[str]) -> Dict[str, int]:
    """Returns {node_id: degree} where degree = total edges incident to that node."""
    if not node_ids:
        return {}

    if not INTERNAL_DB_AVAILABLE:
        counts: Dict[str, int] = {nid: 0 for nid in node_ids}
        wanted = set(node_ids)
        for e in _IN_MEMORY_EDGES:
            if e["src_node_id"] in wanted:
                counts[e["src_node_id"]] = counts.get(e["src_node_id"], 0) + 1
            if e["dst_node_id"] in wanted:
                counts[e["dst_node_id"]] = counts.get(e["dst_node_id"], 0) + 1
        return counts

    try:
        conn = get_internal_connection()
    except Exception:
        return {nid: 0 for nid in node_ids}
    try:
        cur = conn.cursor(cursor_factory=DICT_CURSOR)
        ph = ",".join(["%s"] * len(node_ids))
        params = list(node_ids) + list(node_ids)
        cur.execute(f"""
            SELECT node_id, COUNT(*) AS degree
            FROM (
                SELECT src_node_id AS node_id FROM {GRAPH_EDGES_TABLE}
                WHERE src_node_id IN ({ph})
                UNION ALL
                SELECT dst_node_id AS node_id FROM {GRAPH_EDGES_TABLE}
                WHERE dst_node_id IN ({ph})
            ) sub
            GROUP BY node_id;
        """, params)
        rows = cur.fetchall()
        try: cur.close()
        except Exception: pass
    finally:
        release_internal_connection(conn)

    result = {nid: 0 for nid in node_ids}
    for r in rows:
        result[r["node_id"]] = int(r["degree"])
    return result


def get_neighbors(
    node_ids: List[str],
    edge_types: Optional[List[str]] = None,
    direction: str = "both",
) -> List[GraphEdge]:
    """
    Returns edges incident to any of node_ids.

    direction : "out" (src in node_ids), "in" (dst in node_ids), or "both".
    edge_types: restrict to these edge types (None = all).
    """
    if not node_ids:
        return []

    if not INTERNAL_DB_AVAILABLE:
        wanted = set(node_ids)
        out = []
        for d in _IN_MEMORY_EDGES:
            if edge_types and d["edge_type"] not in edge_types:
                continue
            src_in = d["src_node_id"] in wanted
            dst_in = d["dst_node_id"] in wanted
            if direction == "out" and src_in:
                out.append(_dict_to_edge(d))
            elif direction == "in" and dst_in:
                out.append(_dict_to_edge(d))
            elif direction == "both" and (src_in or dst_in):
                out.append(_dict_to_edge(d))
        return out

    try:
        conn = get_internal_connection()
    except Exception:
        return []
    try:
        cur = conn.cursor(cursor_factory=DICT_CURSOR)
        placeholders = ",".join(["%s"] * len(node_ids))

        if direction == "out":
            where = f"src_node_id IN ({placeholders})"
            params = list(node_ids)
        elif direction == "in":
            where = f"dst_node_id IN ({placeholders})"
            params = list(node_ids)
        else:
            where = f"(src_node_id IN ({placeholders}) OR dst_node_id IN ({placeholders}))"
            params = list(node_ids) + list(node_ids)

        if edge_types:
            et_ph = ",".join(["%s"] * len(edge_types))
            where += f" AND edge_type IN ({et_ph})"
            params += list(edge_types)

        cur.execute(f"""
            SELECT edge_id, src_node_id, dst_node_id, edge_type, weight,
                   source_id, evidence, attrs
            FROM {GRAPH_EDGES_TABLE}
            WHERE {where};
        """, params)
        rows = cur.fetchall()
        try: cur.close()
        except Exception: pass
    finally:
        release_internal_connection(conn)

    return [_row_to_edge(r) for r in rows]
