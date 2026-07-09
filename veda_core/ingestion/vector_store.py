# =============================================================================
# ingestion/vector_store.py
# VEDA — metadata stores (FK adjacency, table metadata, column values) + the
# keyword-match retrieval helper.
#
# WP3: the legacy encoder-embedding write path and the ensemble LT/hybrid stores
# (column_embeddings / _lt / _hybrid) were removed — dense embeddings now live in
# column_embeddings_v2 (ingestion/biencoder.py, BGE-M3). What remains here:
#   - store_fk_adjacency / get_fk_adjacency  (the join engine's FK source of truth)
#   - store_table_metadata / get_display_columns
#   - retrieve_cols_by_name_keywords         (keyword safety-net over the v2 store)
#   - RetrievalResult / FKEdge dataclasses consumed across the query tier
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import (
    VECTOR_STORE_TRUNCATE_ON_INGEST,
    TOP_K,
    BIENCODER_COL_TABLE,
)
from utils.logger import get_logger

logger = get_logger(__name__)


from ingestion.db_abstraction import (
    INTERNAL_DB_AVAILABLE as PSYCOPG2_AVAILABLE,
    get_internal_connection as _get_connection,
    release_internal_connection,
    DICT_CURSOR,
)


# =============================================================================
# Output data structures
# =============================================================================

@dataclass
class RetrievalResult:
    """Result of a Top-K cosine similarity retrieval."""
    col_id:        str
    col_name:      str
    table_id:      str
    table_name:    str
    semantic_type: str
    similarity:    float
    source_id:     str = ""
    embedding:     Optional[np.ndarray] = None


# =============================================================================
# In-memory fallback stores
# Two separate module-level lists for ensemble mode.
# Single-encoder modes use _IN_MEMORY_STORE only.
# =============================================================================

_IN_MEMORY_STORE:        List[dict] = []   # single-encoder + ensemble LT
_IN_MEMORY_STORE_HYBRID: List[dict] = []   # ensemble hybrid only

def retrieve_cols_by_name_keywords(
    keywords: List[str],
) -> List[RetrievalResult]:
    """Direct col_name keyword match — no vector search.

    Returns columns whose name parts ALL appear in `keywords`.
    For example, keywords=["workflow","state","incident","status"] matches
    col_name="workflow_state" (parts {"workflow","state"} ⊆ keywords) and
    col_name="incident_status" (parts {"incident","status"} ⊆ keywords).

    Used as a safety net when vector similarity fails to retrieve columns
    that are obvious keyword matches in the query.
    Returns similarity=0.0 for all results (injected, not ranked).
    """
    if not keywords:
        return []

    kw_set = {k.lower() for k in keywords if len(k) > 2}
    if not kw_set:
        return []

    seen: Dict[Tuple[str, str], RetrievalResult] = {}

    if PSYCOPG2_AVAILABLE and not _IN_MEMORY_STORE:
        try:
            conn = _get_connection()
            cur  = conn.cursor()
            like_clause = " OR ".join(f"LOWER(col_name) LIKE %s" for _ in kw_set)
            patterns    = [f"%{kw}%" for kw in kw_set]

            for store in (BIENCODER_COL_TABLE,):   # WP3: the one live column store
                try:
                    cur.execute(
                        f"SELECT col_id, col_name, table_id, table_name, semantic_type, source_id "
                        f"FROM {store} WHERE ({like_clause})",
                        patterns,
                    )
                    for row in cur.fetchall():
                        col_name  = str(row[1])
                        tbl_name  = str(row[3])
                        key       = (tbl_name, col_name)
                        if key in seen:
                            continue
                        parts = {p for p in col_name.lower().split("_") if len(p) > 2}
                        if parts and parts <= kw_set:
                            seen[key] = RetrievalResult(
                                col_id        = str(row[0]),
                                col_name      = col_name,
                                table_id      = str(row[2]),
                                table_name    = tbl_name,
                                semantic_type = str(row[4]),
                                similarity    = 0.0,
                                source_id     = str(row[5]) if row[5] else "",
                                embedding     = None,
                            )
                except Exception:
                    pass

            cur.close()
            release_internal_connection(conn)
            return list(seen.values())
        except Exception:
            pass

    # In-memory fallback
    for row in _IN_MEMORY_STORE:
        col_name = row.get("col_name", "")
        tbl_name = row.get("table_name", "")
        key      = (tbl_name, col_name)
        if key in seen:
            continue
        parts = {p for p in col_name.lower().split("_") if len(p) > 2}
        if parts and parts <= kw_set:
            seen[key] = RetrievalResult(
                col_id        = row.get("col_id", ""),
                col_name      = col_name,
                table_id      = row.get("table_id", ""),
                table_name    = tbl_name,
                semantic_type = row.get("semantic_type", ""),
                similarity    = 0.0,
                embedding     = None,
            )
    return list(seen.values())


# =============================================================================
# FK Adjacency Store
#
# Responsibility:
#   - Persists FK edges from schema_scanner.ScanResult into a plain SQL table
#     (not pgvector — FK edges don't need vector search)
#   - Called independently from main.py after run_schema_scanner()
#   - Provides get_fk_adjacency() for query-time bridge injection
#
# Schema:
#   fk_adjacency (
#       from_col_id    UUID,
#       from_col_name  TEXT,
#       from_table_id  UUID,
#       from_table_name TEXT,
#       to_col_id      UUID,
#       to_col_name    TEXT,
#       to_table_id    UUID,
#       to_table_name  TEXT
#   )
#
# In-memory fallback:
#   _FK_ADJACENCY stores the same data as a list of dicts.
#   get_fk_adjacency() returns from it when psycopg2 is unavailable.
#
# Design: encoder-independent — plain SQL FK edges, no vectors.
# =============================================================================

from config import (
    FK_ADJACENCY_TABLE_NAME,
    FK_MAX_HOP_DEPTH,
)

# In-memory fallback for FK adjacency
_FK_ADJACENCY: List[dict] = []


@dataclass
class FKEdge:
    """
    A single FK relationship between two columns.
    Used by semantic_layer.py bridge injector.
    """
    from_col_id:     str
    from_col_name:   str
    from_table_id:   str
    from_table_name: str
    to_col_id:       str
    to_col_name:     str
    to_table_id:     str
    to_table_name:   str


@dataclass
class FKStoreResult:
    """Result of storing FK adjacency data."""
    edges_written: int
    backend:       str
    duration_sec:  float


# =============================================================================
# FK adjacency — pgvector backend (plain SQL, no vector extension needed)
# =============================================================================

def _create_fk_table(cursor) -> None:
    """Creates the fk_adjacency table if it does not exist."""
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {FK_ADJACENCY_TABLE_NAME} (
            from_col_id     TEXT NOT NULL,
            from_col_name   TEXT NOT NULL,
            from_table_id   TEXT NOT NULL,
            from_table_name TEXT NOT NULL,
            to_col_id       TEXT,
            to_col_name     TEXT NOT NULL,
            to_table_id     TEXT,
            to_table_name   TEXT NOT NULL
        );
    """)
    # Index on from_table_id and to_table_id for fast JOIN path lookups
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{FK_ADJACENCY_TABLE_NAME}_from_table
        ON {FK_ADJACENCY_TABLE_NAME} (from_table_id);
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{FK_ADJACENCY_TABLE_NAME}_to_table
        ON {FK_ADJACENCY_TABLE_NAME} (to_table_id);
    """)


def _upsert_fk_edges(cursor, fk_edges: list) -> int:
    """
    Truncates and reinserts all FK edges.
    Truncate + insert is safe here — FK schema doesn't change between runs
    and it's simpler than upserting without a stable PK.

    Batched via execute_values (F4) — was one INSERT per edge.
    """
    from psycopg2.extras import execute_values
    cursor.execute(f"TRUNCATE TABLE {FK_ADJACENCY_TABLE_NAME};")
    rows = [
        (
            edge.get("from_col_id",   ""),
            edge.get("from_col_name", ""),
            edge.get("from_table_id", ""),
            edge.get("from_table",    ""),   # key is "from_table" not "from_table_name"
            edge.get("to_col_id",     ""),
            edge.get("to_col_name",   ""),
            edge.get("to_table_id",   ""),
            edge.get("to_table",      ""),   # key is "to_table" not "to_table_name"
        )
        for edge in fk_edges
    ]
    if not rows:
        return 0
    execute_values(
        cursor,
        f"""INSERT INTO {FK_ADJACENCY_TABLE_NAME}
            (from_col_id, from_col_name, from_table_id, from_table_name,
             to_col_id,   to_col_name,   to_table_id,   to_table_name)
            VALUES %s""",
        rows,
    )
    return len(rows)


def _query_fk_edges_pgvector(
    table_ids: List[str],
) -> List[FKEdge]:
    """
    Returns all FK edges where either endpoint table is in table_ids.
    Used to find bridge tables between a set of retrieved tables.
    """
    if not table_ids:
        return []

    placeholders = ",".join(["%s"] * len(table_ids))
    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=DICT_CURSOR) as cur:
            cur.execute(f"""
                SELECT from_col_id, from_col_name, from_table_id, from_table_name,
                       to_col_id,   to_col_name,   to_table_id,   to_table_name
                FROM {FK_ADJACENCY_TABLE_NAME}
                WHERE from_table_id IN ({placeholders})
                   OR to_table_id   IN ({placeholders});
            """, table_ids + table_ids)
            rows = cur.fetchall()
    finally:
        release_internal_connection(conn)

    return [
        FKEdge(
            from_col_id     = row["from_col_id"],
            from_col_name   = row["from_col_name"],
            from_table_id   = row["from_table_id"],
            from_table_name = row["from_table_name"],
            to_col_id       = row["to_col_id"]   or "",
            to_col_name     = row["to_col_name"],
            to_table_id     = row["to_table_id"] or "",
            to_table_name   = row["to_table_name"],
        )
        for row in rows
    ]


# =============================================================================
# FK adjacency — in-memory fallback
# =============================================================================

def _query_fk_edges_memory(table_ids: List[str]) -> List[FKEdge]:
    """Returns FK edges from in-memory store where either endpoint is in table_ids."""
    ids = set(table_ids)
    results = []
    for edge in _FK_ADJACENCY:
        if edge.get("from_table_id", "") in ids or edge.get("to_table_id", "") in ids:
            results.append(FKEdge(
                from_col_id     = edge.get("from_col_id",   ""),
                from_col_name   = edge.get("from_col_name", ""),
                from_table_id   = edge.get("from_table_id", ""),
                from_table_name = edge.get("from_table",    ""),   # key is "from_table"
                to_col_id       = edge.get("to_col_id",     ""),
                to_col_name     = edge.get("to_col_name",   ""),
                to_table_id     = edge.get("to_table_id",   ""),
                to_table_name   = edge.get("to_table",      ""),   # key is "to_table"
            ))
    return results


# =============================================================================
# Public FK adjacency entry points
# =============================================================================

def store_fk_adjacency(
    scan_result,
    verbose: bool = False,
) -> FKStoreResult:
    """
    Persists FK edges from schema_scanner.ScanResult.

    Called independently from main.py after run_schema_scanner().
    Completely independent of encoder mode and vector stores.

    Parameters
    ----------
    scan_result : ScanResult
        Output of run_schema_scanner(). Contains fk_edges list.
    verbose : bool

    Returns
    -------
    FKStoreResult
    """
    global _FK_ADJACENCY

    fk_edges = scan_result.fk_edges

    if verbose:
        logger.debug(
            "Storing FK edges... count=%d backend=%s table=%s",
            len(fk_edges), "pgvector" if PSYCOPG2_AVAILABLE else "in_memory_fallback",
            FK_ADJACENCY_TABLE_NAME,
        )

    t0 = time.time()

    if PSYCOPG2_AVAILABLE:
        try:
            conn = _get_connection()
            try:
                with conn:
                    with conn.cursor() as cur:
                        _create_fk_table(cur)
                        written = _upsert_fk_edges(cur, fk_edges)
            finally:
                release_internal_connection(conn)
            backend = "pgvector"
        except Exception as e:
            logger.warning("pgvector failed (%s) — using in-memory fallback", e)
            _FK_ADJACENCY = list(fk_edges)
            written = len(fk_edges)
            backend = "in_memory_fallback"
    else:
        _FK_ADJACENCY = list(fk_edges)
        written = len(fk_edges)
        backend = "in_memory_fallback"

    duration = round(time.time() - t0, 4)

    logger.info("FK adjacency stored: %d edges, backend=%s, duration=%ss", written, backend, duration)

    return FKStoreResult(
        edges_written = written,
        backend       = backend,
        duration_sec  = duration,
    )


def get_fk_adjacency(
    table_ids: List[str],
    verbose:   bool = False,
) -> List[FKEdge]:
    """
    Returns all FK edges where either endpoint table is in table_ids.
    Called by semantic_layer.py bridge injector at query time.

    Parameters
    ----------
    table_ids : List[str]
        UUID strings of tables currently in the top-K result set.

    Returns
    -------
    List[FKEdge]
        All FK edges connecting or touching the given tables.
    """
    if not table_ids:
        return []

    # Phase 3.4 rewire: when a request/task context is set (inference/worker tiers),
    # route through the storage_adapters seam → Django-owned FK substrate, tenant-scoped
    # (§3.4). Falls back to the engine's own store when no context is set (standalone/dev),
    # so this shim can never break a caller.
    try:
        from veda_core.context import try_current
        if try_current() is not None:
            from storage_adapters.reader import get_fk_adjacency as _adapter_fk
            edges = _adapter_fk(table_ids)
            if verbose:
                logger.debug("%d edges via storage_adapters (Django substrate)", len(edges))
            return edges
    except Exception as _e:
        logger.debug("adapter unavailable (%s) — engine store", _e)

    if PSYCOPG2_AVAILABLE and not _FK_ADJACENCY:
        try:
            edges = _query_fk_edges_pgvector(table_ids)
        except Exception as e:
            logger.warning("FK pgvector query failed (%s) — using in-memory", e)
            edges = _query_fk_edges_memory(table_ids)
    else:
        edges = _query_fk_edges_memory(table_ids)

    if verbose:
        logger.debug("Found %d FK edges for %d tables", len(edges), len(table_ids))

    return edges


# =============================================================================
# Table Metadata Store
#
# Persists the primary display column per table identified during
# semantic type inference.
#
# Schema:
#   table_metadata (
#       table_id         TEXT PRIMARY KEY,
#       table_name       TEXT NOT NULL,
#       display_col_id   TEXT,
#       display_col_name TEXT
#   )
#
# Called independently from main.py after run_semantic_type_inference().
# Provides get_display_columns() for query-time injection in semantic_layer.py.
# Encoder-independent — plain metadata store.
# =============================================================================

_TABLE_METADATA_STORE: List[dict] = []   # in-memory fallback
TABLE_METADATA_TABLE_NAME = "table_metadata"


@dataclass
class TableMetadataStoreResult:
    """Result of storing table metadata."""
    rows_written: int
    backend:      str
    duration_sec: float


def store_table_metadata(
    inference_result,
    verbose: bool = False,
) -> TableMetadataStoreResult:
    """
    Persists display_col_map from InferenceResult into table_metadata store.

    Called independently from main.py after run_semantic_type_inference().
    Zero coupling to encoder mode or vector dimensions.

    Parameters
    ----------
    inference_result : InferenceResult
        Output of run_semantic_type_inference(). Contains display_col_map.
    verbose : bool

    Returns
    -------
    TableMetadataStoreResult
    """
    global _TABLE_METADATA_STORE

    display_col_map = getattr(inference_result, "display_col_map", {})

    if verbose:
        logger.debug(
            "Storing display columns... entries=%d backend=%s",
            len(display_col_map), "pgvector" if PSYCOPG2_AVAILABLE else "in_memory_fallback",
        )

    t0 = time.time()

    rows = [
        {
            "table_id":         table_id,
            "table_name":       table_name,
            "display_col_id":   col_id,
            "display_col_name": col_name,
        }
        for table_id, (col_id, col_name, table_name) in display_col_map.items()
    ]

    if PSYCOPG2_AVAILABLE:
        try:
            conn = _get_connection()
            try:
                with conn:
                    with conn.cursor() as cur:
                        # Create table
                        cur.execute(f"""
                            CREATE TABLE IF NOT EXISTS {TABLE_METADATA_TABLE_NAME} (
                                table_id         TEXT PRIMARY KEY,
                                table_name       TEXT NOT NULL,
                                display_col_id   TEXT,
                                display_col_name TEXT
                            );
                        """)
                        # Truncate + reinsert — deterministic per ingestion run
                        cur.execute(f"TRUNCATE TABLE {TABLE_METADATA_TABLE_NAME};")
                        # F5: batched insert — was one execute() per table row.
                        from psycopg2.extras import execute_values
                        _tm_rows = [
                            (row["table_id"], row["table_name"],
                             row["display_col_id"], row["display_col_name"])
                            for row in rows
                        ]
                        if _tm_rows:
                            execute_values(
                                cur,
                                f"""INSERT INTO {TABLE_METADATA_TABLE_NAME}
                                    (table_id, table_name, display_col_id, display_col_name)
                                    VALUES %s""",
                                _tm_rows,
                            )
            finally:
                release_internal_connection(conn)
            backend = "pgvector"
        except Exception as e:
            logger.warning("pgvector failed (%s) — using in-memory fallback", e)
            _TABLE_METADATA_STORE = rows
            backend = "in_memory_fallback"
    else:
        _TABLE_METADATA_STORE = rows
        backend = "in_memory_fallback"

    duration = round(time.time() - t0, 4)

    logger.info("Table metadata stored: %d rows, backend=%s, duration=%ss", len(rows), backend, duration)

    return TableMetadataStoreResult(
        rows_written = len(rows),
        backend      = backend,
        duration_sec = duration,
    )


def get_display_columns(
    table_ids: List[str],
    verbose:   bool = False,
) -> dict:
    """
    Returns display column info for the given table_ids.

    Called by semantic_layer.py Step 4b to inject display columns.

    Parameters
    ----------
    table_ids : List[str]
        UUID strings of tables in the current top-K result set.

    Returns
    -------
    dict : {table_id → {"col_id": str, "col_name": str, "table_name": str}}
    """
    if not table_ids:
        return {}

    ids = set(table_ids)

    if PSYCOPG2_AVAILABLE and not _TABLE_METADATA_STORE:
        try:
            conn = _get_connection()
            try:
                placeholders = ",".join(["%s"] * len(ids))
                with conn.cursor(cursor_factory=DICT_CURSOR) as cur:
                    cur.execute(f"""
                        SELECT table_id, table_name, display_col_id, display_col_name
                        FROM {TABLE_METADATA_TABLE_NAME}
                        WHERE table_id IN ({placeholders});
                    """, list(ids))
                    rows = cur.fetchall()
            finally:
                release_internal_connection(conn)
            return {
                row["table_id"]: {
                    "col_id":     row["display_col_id"],
                    "col_name":   row["display_col_name"],
                    "table_name": row["table_name"],
                }
                for row in rows
                if row["display_col_id"]
            }
        except Exception as e:
            logger.warning("pgvector table_metadata query failed (%s) — using in-memory", e)

    # In-memory fallback
    return {
        row["table_id"]: {
            "col_id":     row["display_col_id"],
            "col_name":   row["display_col_name"],
            "table_name": row["table_name"],
        }
        for row in _TABLE_METADATA_STORE
        if row["table_id"] in ids and row["display_col_id"]
    }