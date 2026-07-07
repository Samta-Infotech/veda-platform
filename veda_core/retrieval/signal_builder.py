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

from schema.real_schema import get_real_schema
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
        self.fk_graph = {}  # column_id -> list of referenced column_ids
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

        # Load schema. Q-1: read the FK graph from the substrate (fk_adjacency, written
        # at ingestion) instead of live information_schema on the CLIENT DB every warm —
        # removes the source-DB dependency from the query tier. Flag-gated; falls back
        # to the live introspection when the substrate FK store is empty/unavailable.
        self.schema = None
        try:
            from config import SUBSTRATE_SIGNALS_ENABLED
            if SUBSTRATE_SIGNALS_ENABLED:
                self.schema = _load_schema_from_substrate()
        except Exception as e:
            logger.warning(f"Substrate signal load failed ({e}); using live schema")
            self.schema = None
        if not self.schema or not self.schema.get("tables"):
            self.schema = get_real_schema()
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
        """Build foreign key graph from schema."""
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
                        self.fk_graph[col_id] = ref_col_id

        logger.info(f"✓ Found {len(self.fk_graph)} foreign keys")

    def _build_table_adjacency(self):
        """Build table adjacency graph from FK relationships."""
        logger.info("Building table adjacency...")

        tables_set = {}
        for col_id, ref_col_id in self.fk_graph.items():
            table1 = col_id.split(".")[0]
            table2 = ref_col_id.split(".")[0]

            if table1 not in tables_set:
                tables_set[table1] = set()
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
        # Check if referenced
        if any(ref == col_id for ref in self.fk_graph.values()):
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
