# =============================================================================
# retrieval/signal_builder.py
# VEDA Phase 2 - FK Adjacency + Subgraph Signals
#
# Purpose:
#   Extract relationship-based ranking signals from schema:
#   - Foreign key adjacency (columns that reference each other)
#   - Subgraph connectivity (related tables)
# =============================================================================

import sys
import os
import json
import logging
from typing import Dict, List, Tuple, Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import get_logger

logger = get_logger(__name__)


def _load_schema_from_substrate():
    """Q-1: reconstruct a FK-only schema dict from the substrate `fk_adjacency` table
    (written at ingestion), so the signal builder never touches the client DB at warm.

    Only FK columns are materialised — that is all `_build_fk_graph` / adjacency use.
    Returns {"tables": [{"table_name", "columns":[{col_name, is_fk, fk_ref_table,
    fk_ref_col}]}]} or None on any failure (caller falls back to live introspection)."""
    try:
        from ingestion.db_abstraction import get_internal_connection, release_internal_connection
    except Exception:
        return None
    conn = None
    try:
        conn = get_internal_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT from_table_name, from_col_name, to_table_name, to_col_name "
                "FROM fk_adjacency")
            rows = cur.fetchall()
    except Exception:
        return None
    finally:
        if conn is not None:
            try:
                release_internal_connection(conn)
            except Exception:
                pass
    if not rows:
        return None
    tables: Dict[str, dict] = {}
    for from_table, from_col, to_table, to_col in rows:
        if not from_table or not from_col:
            continue
        t = tables.setdefault(from_table, {"table_name": from_table, "columns": []})
        t["columns"].append({
            "col_name": from_col, "is_fk": True,
            "fk_ref_table": to_table, "fk_ref_col": to_col,
        })
        tables.setdefault(to_table, {"table_name": to_table, "columns": []})
    return {"tables": list(tables.values())}


class SignalBuilder:
    """Build FK and subgraph signals for column ranking."""

    def __init__(self):
        """Initialize signal builder."""
        self.schema = None
        self.fk_graph = {}  # column_id -> SET of referenced column_ids (P1-3, 2026-09-10:
        # was a plain str->str dict, so a polymorphic column with multiple targets kept
        # only the last edge — the comment already said "list" but the code never did).
        self._referenced_cols = set()  # precomputed union of fk_graph.values() — see
        # _compute_column_signals(), which used to be O(n) per column (O(n²) overall).
        self.table_adjacency = {}  # table_name -> list of related table_names
        self.column_signals = {}  # column_id -> {fk_score, subgraph_score}

    def build_signals(self, semantic_model: Dict) -> Dict[str, Dict[str, float]]:
        """
        Build all signals from schema and semantic model.

        Args:
            semantic_model: Output from semantic_layer_v2.py

        Returns:
            {column_id: {fk_signal: float, subgraph_signal: float}}
        """
        logger.info("Building relationship signals...")

        # WP7: FK signals come ONLY from the substrate fk_adjacency store (written at
        # ingestion). The live information_schema introspection dual was removed — the
        # query tier now makes zero source-DB connections outside L7 execution.
        self.schema = _load_schema_from_substrate() or {"tables": []}
        tables = self.schema.get("tables", [])

        # Build FK graph
        self._build_fk_graph(tables)

        # Build table adjacency graph
        self._build_table_adjacency()

        # Compute signals for each column
        retrieval_docs = semantic_model.get("retrieval_documents", {})

        for col_id in retrieval_docs.keys():
            table_name = col_id.split(".")[0]
            self.column_signals[col_id] = self._compute_column_signals(
                col_id, table_name
            )

        logger.info(f"✓ Built signals for {len(self.column_signals)} columns")

        return self.column_signals

    def _build_fk_graph(self, tables: List[Dict]):
        """Build foreign key graph, preferring the canonical relationship graph.

        The substrate `fk_adjacency` scan (below) only carries DECLARED FKs. The
        relationship graph built at ingestion (ingestion/relationship_graph.py) also
        includes data-inferred/polymorphic edges (no declared FK constraint, discovered
        by cardinality + name-affinity) — real join paths this signal would otherwise
        never see. `veda.runtime.get_graph()` is a local-file read (already loaded/cached
        for query routing), so this adds no source-DB touch and keeps WP7's zero-warm-
        connection guarantee. Falls back to the declared-only scan if the graph is
        unavailable or empty."""
        try:
            from veda.runtime import get_graph
            graph = get_graph() or {}
            edges = graph.get("edges") or []
        except Exception:
            edges = []
        if edges:
            logger.info("Building FK graph from relationship graph...")
            for e in edges:
                s, sc = e.get("source_table"), e.get("source_column")
                t, tc = e.get("target_table"), e.get("target_column")
                if s and sc and t and tc:
                    self.fk_graph.setdefault(f"{s}.{sc}", set()).add(f"{t}.{tc}")
            logger.info(f"✓ Found {len(self.fk_graph)} foreign keys (relationship graph)")
            self._finalize_fk_graph()
            return

        logger.info("Building FK graph...")

        for table_info in tables:
            table_name = table_info["table_name"]
            for col_info in table_info.get("columns", []):
                col_name = col_info.get("col_name") or col_info.get("name")
                is_fk = col_info.get("is_fk", False)

                if is_fk:
                    col_id = f"{table_name}.{col_name}"
                    fk_ref_table = col_info.get("fk_ref_table")
                    fk_ref_col = col_info.get("fk_ref_col")

                    if fk_ref_table and fk_ref_col:
                        ref_col_id = f"{fk_ref_table}.{fk_ref_col}"
                        self.fk_graph.setdefault(col_id, set()).add(ref_col_id)

        logger.info(f"✓ Found {len(self.fk_graph)} foreign keys")
        self._finalize_fk_graph()

    def _finalize_fk_graph(self):
        """Precompute the union of every referenced column (P1-3, 2026-09-10) — once,
        here, instead of scanning all of fk_graph.values() per column inside
        _compute_column_signals() (O(n) per call, O(n²) across all columns)."""
        self._referenced_cols = set().union(*self.fk_graph.values()) if self.fk_graph else set()

    def _build_table_adjacency(self):
        """Build table adjacency graph from FK relationships."""
        logger.info("Building table adjacency...")

        tables_set = {}
        for col_id, ref_col_ids in self.fk_graph.items():
            table1 = col_id.split(".")[0]
            if table1 not in tables_set:
                tables_set[table1] = set()
            # ref_col_ids is a SET now (P1-3): a polymorphic column pointing at
            # multiple target tables must connect to ALL of them, not just the last
            # one a plain str->str dict happened to keep.
            for ref_col_id in ref_col_ids:
                table2 = ref_col_id.split(".")[0]
                if table2 not in tables_set:
                    tables_set[table2] = set()
                tables_set[table1].add(table2)
                tables_set[table2].add(table1)

        self.table_adjacency = {
            table: list(adjacent) for table, adjacent in tables_set.items()
        }

        logger.info(f"✓ Built adjacency for {len(self.table_adjacency)} tables")

    def _compute_column_signals(self, col_id: str, table_name: str) -> Dict[str, float]:
        """
        Compute signals for a column.

        Returns:
            {fk_signal: float (0-1), subgraph_signal: float (0-1)}
        """
        # FK signal: is this column a FK or referenced by FK?
        fk_signal = 0.0
        if col_id in self.fk_graph:
            fk_signal = 0.5  # This column references another
        # Check if referenced — O(1) against the precomputed set (P1-3, 2026-09-10;
        # was `any(ref == col_id for ref in self.fk_graph.values())`, O(n) per column
        # and therefore O(n²) across every column in the schema).
        if col_id in self._referenced_cols:
            fk_signal = max(fk_signal, 0.7)  # This column is referenced

        # Subgraph signal: connectivity to other tables
        subgraph_signal = 0.0
        if table_name in self.table_adjacency:
            degree = len(self.table_adjacency[table_name])
            subgraph_signal = min(degree / 10.0, 1.0)  # Normalize by expected max degree

        return {
            "fk_signal": fk_signal,
            "subgraph_signal": subgraph_signal,
        }

    def get_signal(self, col_id: str, signal_name: str) -> float:
        """Get a specific signal for a column."""
        if col_id not in self.column_signals:
            return 0.0
        return self.column_signals[col_id].get(signal_name, 0.0)

    def build_value_index(self, semantic_model: Dict) -> Dict[str, List[str]]:
        """value_token (lowercased) -> [col_id, ...]. Built ONCE at engine
        warm-load (F3) so Signal 5 becomes a set lookup, not a per-query
        nested scan over every column's sampled values."""
        index: Dict[str, List[str]] = {}
        for col_id, col_meta in semantic_model.get("columns", {}).items():
            for val in (col_meta.get("sample_values") or []):
                token = str(val).strip().lower()
                if not token:
                    continue
                index.setdefault(token, []).append(col_id)
        return index
