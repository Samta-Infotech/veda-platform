# =============================================================================
# ingestion/vector_store.py
# VEDA POC — Step 5: PostgreSQL + pgvector Metadata Store
#
# Responsibility:
#   - Accepts EncoderResult or EnsembleEncoderResult from relgt_encoder.py
#   - Creates pgvector table(s) if they do not exist
#   - Persists embeddings with upsert — safe to re-run
#   - Creates ivfflat cosine index on each embedding column
#   - Provides retrieval helpers used by semantic_layer.py (L2)
#
# Single-encoder modes (relgt_only, light_text, hybrid):
#   - Writes to VECTOR_TABLE_NAME (column_embeddings)
#   - retrieve_top_k() queries that single table
#
# Ensemble mode:
#   - Writes light_text embeddings → VECTOR_TABLE_NAME_LIGHT_TEXT
#   - Writes hybrid embeddings    → VECTOR_TABLE_NAME_HYBRID
#   - retrieve_top_k_lt()     queries the light_text table (256-dim)
#   - retrieve_top_k_hybrid() queries the hybrid table (640-dim)
#   - semantic_layer.py calls both and merges via RRF
#
# In-memory fallback:
#   - Activates when psycopg2 / pgvector is unavailable
#   - Two separate in-memory stores for ensemble mode
#   - Identical retrieval interface — semantic_layer.py sees no difference
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ingestion.relgt_encoder import (
    EncoderResult,
    EnsembleEncoderResult,
    ColumnEmbedding,
    run_relgt_encoder,
)
from config import (
    ENCODER_MODE,
    VECTOR_TABLE_NAME,
    VECTOR_TABLE_NAME_LIGHT_TEXT,
    VECTOR_TABLE_NAME_HYBRID,
    VECTOR_STORE_TRUNCATE_ON_INGEST,
    VECTOR_DIM,
    LIGHT_TEXT_EMBEDDING_DIM,
    HYBRID_EMBEDDING_DIM,
    TOP_K,
    ENSEMBLE_CANDIDATES_PER_STORE,
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
class StoreResult:
    """Result of a single store operation."""
    rows_written:  int
    rows_skipped:  int
    table_name:    str
    vector_dim:    int
    index_created: bool
    backend:       str      # "pgvector" | "in_memory_fallback"
    duration_sec:  float
    stats:         dict = field(default_factory=dict)


@dataclass
class EnsembleStoreResult:
    """Result of an ensemble store — wraps two StoreResults."""
    lt_result:     StoreResult
    hybrid_result: StoreResult
    backend:       str
    duration_sec:  float
    stats:         dict = field(default_factory=dict)


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

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    na  = float(np.linalg.norm(a))
    nb  = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _store_in_memory_to(
    store:      List[dict],
    embeddings: List[ColumnEmbedding],
    table_name: str,
    vector_dim: int,
) -> StoreResult:
    """Generic in-memory store into the given list."""
    store.clear()
    t0 = time.time()
    for emb in embeddings:
        store.append({
            "col_id":        emb.col_id,
            "col_name":      emb.col_name,
            "table_id":      emb.table_id,
            "table_name":    emb.table_name,
            "semantic_type": emb.semantic_type,
            "embedding":     emb.embedding.copy(),
        })
    return StoreResult(
        rows_written  = len(embeddings),
        rows_skipped  = 0,
        table_name    = table_name,
        vector_dim    = vector_dim,
        index_created = False,
        backend       = "in_memory_fallback",
        duration_sec  = round(time.time() - t0, 4),
        stats         = {"total_stored": len(store)},
    )


def _retrieve_from_memory(
    store:        List[dict],
    query_vector: np.ndarray,
    top_k:        int,
) -> List[RetrievalResult]:
    """Cosine similarity search over a given in-memory store."""
    if not store:
        return []
    scores = [(  _cosine_similarity(query_vector, row["embedding"]), row)
              for row in store]
    scores.sort(key=lambda x: x[0], reverse=True)
    return [
        RetrievalResult(
            col_id        = row["col_id"],
            col_name      = row["col_name"],
            table_id      = row["table_id"],
            table_name    = row["table_name"],
            semantic_type = row["semantic_type"],
            similarity    = round(sim, 6),
            embedding     = row["embedding"],
        )
        for sim, row in scores[:top_k]
    ]


# =============================================================================
# pgvector helpers — generic, table-name and dim parameterised
# =============================================================================

def _ensure_pgvector_extension(cursor) -> None:
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def _create_table_for(cursor, table_name: str, vector_dim: int) -> None:
    """Creates a column_embeddings-style table with the given dim."""
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            col_id        UUID        PRIMARY KEY,
            table_id      UUID        NOT NULL,
            col_name      TEXT        NOT NULL,
            table_name    TEXT        NOT NULL,
            semantic_type TEXT        NOT NULL,
            source_id     TEXT        NOT NULL DEFAULT '',
            embedding     vector({vector_dim})
        );
    """)
    # Migrate tables created before source_id was added to the schema.
    cursor.execute(f"""
        ALTER TABLE {table_name}
        ADD COLUMN IF NOT EXISTS source_id TEXT NOT NULL DEFAULT '';
    """)


def _create_index_for(cursor, table_name: str) -> bool:
    """Creates ivfflat cosine index. Returns True if created."""
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    count = cursor.fetchone()[0]
    if count < 100:
        return False
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding
        ON {table_name}
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 10);
    """)
    return True


def _upsert_to(
    cursor,
    table_name: str,
    embeddings: List[ColumnEmbedding],
    source_id: str = "",
) -> Tuple[int, int]:
    """Upserts embeddings into the given table. Returns (written, skipped)."""
    written = skipped = 0
    for emb in embeddings:
        vec_str = "[" + ",".join(f"{v:.8f}" for v in emb.embedding.tolist()) + "]"
        try:
            cursor.execute(f"""
                INSERT INTO {table_name}
                    (col_id, table_id, col_name, table_name, semantic_type, source_id, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (col_id) DO UPDATE SET
                    semantic_type = EXCLUDED.semantic_type,
                    source_id     = EXCLUDED.source_id,
                    embedding     = EXCLUDED.embedding;
            """, (
                emb.col_id, emb.table_id, emb.col_name,
                emb.table_name, emb.semantic_type, source_id, vec_str,
            ))
            written += 1
        except Exception as e:
            skipped += 1
            if written == 0:
                raise RuntimeError(f"Failed to upsert col '{emb.col_name}': {e}")
    return written, skipped


def _store_pgvector_to(
    table_name: str,
    vector_dim: int,
    embeddings: List[ColumnEmbedding],
    source_id: str = "",
) -> StoreResult:
    """Stores embeddings into the named pgvector table."""
    t0   = time.time()
    conn = _get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                _ensure_pgvector_extension(cur)
                _create_table_for(cur, table_name, vector_dim)
                if VECTOR_STORE_TRUNCATE_ON_INGEST and source_id:
                    cur.execute(
                        f"DELETE FROM {table_name} WHERE source_id = %s;",
                        (source_id,),
                    )
                written, skipped = _upsert_to(cur, table_name, embeddings, source_id)
                index_created    = _create_index_for(cur, table_name)
    finally:
        release_internal_connection(conn)

    return StoreResult(
        rows_written  = written,
        rows_skipped  = skipped,
        table_name    = table_name,
        vector_dim    = vector_dim,
        index_created = index_created,
        backend       = "pgvector",
        duration_sec  = round(time.time() - t0, 4),
        stats         = {"total_stored": written},
    )


def _retrieve_pgvector_from(
    table_name:   str,
    query_vector: np.ndarray,
    top_k:        int,
) -> List[RetrievalResult]:
    """Cosine similarity search from the named pgvector table."""
    vec_str = "[" + ",".join(f"{v:.8f}" for v in query_vector.tolist()) + "]"
    conn    = _get_connection()
    try:
        with conn.cursor(cursor_factory=DICT_CURSOR) as cur:
            cur.execute(f"""
                SELECT col_id, col_name, table_id, table_name, semantic_type, source_id,
                       1 - (embedding <=> %s::vector) AS similarity,
                       embedding::text AS embedding_str
                FROM {table_name}
                ORDER BY embedding <=> %s::vector, col_id ASC
                LIMIT %s;
            """, (vec_str, vec_str, top_k))
            rows = cur.fetchall()
    finally:
        release_internal_connection(conn)

    results = []
    for row in rows:
        try:
            emb_arr = np.array(
                json.loads(row["embedding_str"].replace("{", "[").replace("}", "]")),
                dtype=np.float32,
            )
        except Exception:
            emb_arr = None
        results.append(RetrievalResult(
            col_id        = str(row["col_id"]),
            col_name      = row["col_name"],
            table_id      = str(row["table_id"]),
            table_name    = row["table_name"],
            semantic_type = row["semantic_type"],
            similarity    = round(float(row["similarity"]), 6),
            source_id     = row.get("source_id", ""),
            embedding     = emb_arr,
        ))
    return results


# =============================================================================
# Public store entry point
# =============================================================================

def run_vector_store(
    encoder_result = None,
    source_id: str = "",
    verbose: bool  = False,
):
    """
    Main entry point for Step 5.

    Accepts EncoderResult (single-encoder) or EnsembleEncoderResult (ensemble).
    Routes to the correct store path automatically.

    source_id: identifies which VEDA_SOURCE these embeddings came from.
               When set and VECTOR_STORE_TRUNCATE_ON_INGEST=True, stale rows
               for this source are deleted before inserting, making re-ingestion
               fully idempotent without accumulating duplicate UUID rows.

    Returns StoreResult or EnsembleStoreResult depending on ENCODER_MODE.
    """
    if encoder_result is None:
        encoder_result = run_relgt_encoder(verbose=verbose)

    logger.debug("Starting vector store: mode=%s, source_id=%r",
                 "ensemble" if isinstance(encoder_result, EnsembleEncoderResult) else "single",
                 source_id)

    # ------------------------------------------------------------------
    # Ensemble path — write to two separate tables
    # ------------------------------------------------------------------
    if isinstance(encoder_result, EnsembleEncoderResult):
        return _run_ensemble_store(encoder_result, source_id=source_id, verbose=verbose)

    # ------------------------------------------------------------------
    # Single-encoder path — write to one table
    # ------------------------------------------------------------------
    embeddings = encoder_result.embeddings
    emb_dim    = encoder_result.embedding_dim

    if verbose:
        logger.debug(
            "Persisting embeddings... backend=%s embeddings=%d dim=%d table=%s source_id=%r",
            "pgvector" if PSYCOPG2_AVAILABLE else "in_memory_fallback",
            len(embeddings), emb_dim, VECTOR_TABLE_NAME, source_id,
        )

    if PSYCOPG2_AVAILABLE:
        try:
            result = _store_pgvector_to(VECTOR_TABLE_NAME, emb_dim, embeddings, source_id)
        except Exception as e:
            logger.warning("pgvector failed (%s) — falling back to in-memory store", e)
            result = _store_in_memory_to(
                _IN_MEMORY_STORE, embeddings, VECTOR_TABLE_NAME, emb_dim
            )
    else:
        result = _store_in_memory_to(
            _IN_MEMORY_STORE, embeddings, VECTOR_TABLE_NAME, emb_dim
        )

    logger.info(
        "Vector store complete: %d rows, dim=%d, backend=%s, index_created=%s, duration=%ss",
        result.rows_written, result.vector_dim, result.backend, result.index_created,
        result.duration_sec,
    )

    return result


def _run_ensemble_store(
    enc: EnsembleEncoderResult,
    source_id: str = "",
    verbose: bool = False,
) -> EnsembleStoreResult:
    """
    Writes light_text and hybrid embeddings to their respective tables.
    Both use the same backend (pgvector or in-memory).
    """
    t0 = time.time()

    if verbose:
        logger.debug(
            "Ensemble mode — writing to two stores. backend=%s lt=%s(dim=%d) hybrid=%s(dim=%d) "
            "embeddings_each=%d source_id=%r",
            "pgvector" if PSYCOPG2_AVAILABLE else "in_memory_fallback",
            VECTOR_TABLE_NAME_LIGHT_TEXT, enc.lt_embedding_dim,
            VECTOR_TABLE_NAME_HYBRID, enc.hybrid_embedding_dim,
            len(enc.lt_embeddings), source_id,
        )

    if PSYCOPG2_AVAILABLE:
        try:
            lt_result = _store_pgvector_to(
                VECTOR_TABLE_NAME_LIGHT_TEXT,
                enc.lt_embedding_dim,
                enc.lt_embeddings,
                source_id,
            )
            hybrid_result = _store_pgvector_to(
                VECTOR_TABLE_NAME_HYBRID,
                enc.hybrid_embedding_dim,
                enc.hybrid_embeddings,
                source_id,
            )
            backend = "pgvector"
        except Exception as e:
            logger.warning("pgvector failed (%s) — falling back to in-memory stores", e)
            lt_result = _store_in_memory_to(
                _IN_MEMORY_STORE, enc.lt_embeddings,
                VECTOR_TABLE_NAME_LIGHT_TEXT, enc.lt_embedding_dim,
            )
            hybrid_result = _store_in_memory_to(
                _IN_MEMORY_STORE_HYBRID, enc.hybrid_embeddings,
                VECTOR_TABLE_NAME_HYBRID, enc.hybrid_embedding_dim,
            )
            backend = "in_memory_fallback"
    else:
        lt_result = _store_in_memory_to(
            _IN_MEMORY_STORE, enc.lt_embeddings,
            VECTOR_TABLE_NAME_LIGHT_TEXT, enc.lt_embedding_dim,
        )
        hybrid_result = _store_in_memory_to(
            _IN_MEMORY_STORE_HYBRID, enc.hybrid_embeddings,
            VECTOR_TABLE_NAME_HYBRID, enc.hybrid_embedding_dim,
        )
        backend = "in_memory_fallback"

    duration = round(time.time() - t0, 4)

    logger.info(
        "Ensemble store complete: lt=%d rows (index=%s), hybrid=%d rows (index=%s), "
        "backend=%s, duration=%ss",
        lt_result.rows_written, lt_result.index_created,
        hybrid_result.rows_written, hybrid_result.index_created,
        backend, duration,
    )

    return EnsembleStoreResult(
        lt_result     = lt_result,
        hybrid_result = hybrid_result,
        backend       = backend,
        duration_sec  = duration,
        stats         = {
            "lt_rows":     lt_result.rows_written,
            "hybrid_rows": hybrid_result.rows_written,
            "backend":     backend,
        },
    )


# =============================================================================
# Public retrieval entry points
# =============================================================================

def retrieve_top_k(
    query_vector: np.ndarray,
    top_k:        int  = TOP_K,
    verbose:      bool = False,
) -> List[RetrievalResult]:
    """
    Single-store retrieval — used by single-encoder modes (relgt_only,
    light_text, hybrid) and also as the light_text sub-retrieval in ensemble.

    Queries VECTOR_TABLE_NAME (or in-memory fallback).
    """
    assert query_vector.ndim == 1, "query_vector must be 1-D"

    t0 = time.time()

    if PSYCOPG2_AVAILABLE and not _IN_MEMORY_STORE:
        try:
            results = _retrieve_pgvector_from(VECTOR_TABLE_NAME, query_vector, top_k)
        except Exception as e:
            logger.warning("pgvector retrieval failed (%s) — using in-memory fallback", e)
            results = _retrieve_from_memory(_IN_MEMORY_STORE, query_vector, top_k)
    else:
        results = _retrieve_from_memory(_IN_MEMORY_STORE, query_vector, top_k)

    if verbose:
        logger.debug("Top-%d retrieval in %sms", top_k, round((time.time() - t0) * 1000, 2))
        for r in results[:3]:
            logger.debug("  %s.%-28s sim=%.4f  %s", r.table_name, r.col_name, r.similarity, r.semantic_type)

    return results


def retrieve_top_k_lt(
    query_vector: np.ndarray,
    top_k:        int  = ENSEMBLE_CANDIDATES_PER_STORE,
    verbose:      bool = False,
) -> List[RetrievalResult]:
    """
    Ensemble light-text sub-retrieval.
    Queries VECTOR_TABLE_NAME_LIGHT_TEXT (256-dim store).
    Called exclusively by semantic_layer.py in ensemble mode.
    """
    assert query_vector.shape == (LIGHT_TEXT_EMBEDDING_DIM,), (
        f"LT query vector must be shape ({LIGHT_TEXT_EMBEDDING_DIM},), "
        f"got {query_vector.shape}"
    )

    if PSYCOPG2_AVAILABLE and not _IN_MEMORY_STORE:
        try:
            return _retrieve_pgvector_from(VECTOR_TABLE_NAME_LIGHT_TEXT, query_vector, top_k)
        except Exception as e:
            logger.warning("pgvector LT retrieval failed (%s) — using in-memory", e)
    return _retrieve_from_memory(_IN_MEMORY_STORE, query_vector, top_k)


def retrieve_top_k_hybrid(
    query_vector: np.ndarray,
    top_k:        int  = ENSEMBLE_CANDIDATES_PER_STORE,
    verbose:      bool = False,
) -> List[RetrievalResult]:
    """
    Ensemble hybrid sub-retrieval.
    Queries VECTOR_TABLE_NAME_HYBRID (640-dim store).
    Called exclusively by semantic_layer.py in ensemble mode.
    """
    assert query_vector.shape == (HYBRID_EMBEDDING_DIM,), (
        f"Hybrid query vector must be shape ({HYBRID_EMBEDDING_DIM},), "
        f"got {query_vector.shape}"
    )

    if PSYCOPG2_AVAILABLE and not _IN_MEMORY_STORE_HYBRID:
        try:
            return _retrieve_pgvector_from(VECTOR_TABLE_NAME_HYBRID, query_vector, top_k)
        except Exception as e:
            logger.warning("pgvector hybrid retrieval failed (%s) — using in-memory", e)
    return _retrieve_from_memory(_IN_MEMORY_STORE_HYBRID, query_vector, top_k)


def retrieve_temporal_cols_for_tables(
    table_ids: List[str],
) -> List[RetrievalResult]:
    """Return all TEMPORAL-typed columns for the given table_ids.

    Used by the semantic layer's temporal injection step: when L1 detects a
    date expression, L2 guarantees at least one TEMPORAL column per primary
    table is present in the top-k list so L3/L4 can build a date filter.

    No vector search — this is a direct SQL filter on semantic_type.
    Returns similarity=0.0 for all results (injected, not ranked).

    Args:
        table_ids: UUIDs of tables whose TEMPORAL columns are needed.

    Returns:
        List of RetrievalResult with semantic_type == 'TEMPORAL'.
    """
    if not table_ids:
        return []

    if PSYCOPG2_AVAILABLE and not _IN_MEMORY_STORE:
        try:
            conn = _get_connection()
            cur  = conn.cursor()
            placeholders = ",".join(["%s"] * len(table_ids))
            # In ensemble mode each store has different random UUIDs for the
            # same physical table, so query both stores and deduplicate by col_id.
            seen_cols: Dict[Tuple[str, str], RetrievalResult] = {}
            for store in (VECTOR_TABLE_NAME_LIGHT_TEXT, VECTOR_TABLE_NAME_HYBRID):
                try:
                    cur.execute(
                        f"SELECT col_id, col_name, table_id, table_name, semantic_type "
                        f"FROM {store} "
                        f"WHERE table_id IN ({placeholders}) "
                        f"AND semantic_type = 'TEMPORAL'",
                        table_ids,
                    )
                    for row in cur.fetchall():
                        tbl_name = str(row[3])
                        col_name = str(row[1])
                        key = (tbl_name, col_name)
                        if key not in seen_cols:
                            seen_cols[key] = RetrievalResult(
                                col_id=str(row[0]), col_name=col_name,
                                table_id=str(row[2]), table_name=tbl_name,
                                semantic_type=str(row[4]), similarity=0.0, embedding=None,
                            )
                except Exception:
                    pass
            cur.close()
            conn.close()
            return list(seen_cols.values())
        except Exception:
            pass  # fall through to in-memory scan

    # In-memory fallback
    id_set = set(table_ids)
    results: List[RetrievalResult] = []
    seen: set = set()
    for row in _IN_MEMORY_STORE:
        col_id = row.get("col_id", "")
        if col_id in seen:
            continue
        if (row.get("table_id", "") in id_set
                and row.get("semantic_type", "") == "TEMPORAL"):
            results.append(RetrievalResult(
                col_id=col_id,
                col_name=row.get("col_name", ""),
                table_id=row.get("table_id", ""),
                table_name=row.get("table_name", ""),
                semantic_type="TEMPORAL",
                similarity=0.0,
                embedding=None,
            ))
            seen.add(col_id)
    return results


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

            for store in (VECTOR_TABLE_NAME_LIGHT_TEXT, VECTOR_TABLE_NAME_HYBRID):
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
# Smoke test — python ingestion/vector_store.py
# =============================================================================

if __name__ == "__main__":
    store_result = run_vector_store(verbose=True)

    print("=" * 60)
    print(f"VEDA POC — Vector Store Output  [{ENCODER_MODE}]")
    print("=" * 60)

    if isinstance(store_result, EnsembleStoreResult):
        print(f"  Backend           : {store_result.backend}")
        print(f"  LT rows written   : {store_result.lt_result.rows_written}")
        print(f"  Hybrid rows written: {store_result.hybrid_result.rows_written}")
        print(f"  Duration          : {store_result.duration_sec}s")
        print()

        # Test both retrievals
        lt_q = np.random.randn(LIGHT_TEXT_EMBEDDING_DIM).astype(np.float32)
        lt_q = lt_q / np.linalg.norm(lt_q)
        h_q  = np.random.randn(HYBRID_EMBEDDING_DIM).astype(np.float32)
        h_q  = h_q / np.linalg.norm(h_q)

        print("LT top-3 (random query):")
        for r in retrieve_top_k_lt(lt_q, top_k=3):
            print(f"  {r.table_name}.{r.col_name:<28} sim={r.similarity:.4f}")
        print("Hybrid top-3 (random query):")
        for r in retrieve_top_k_hybrid(h_q, top_k=3):
            print(f"  {r.table_name}.{r.col_name:<28} sim={r.similarity:.4f}")
    else:
        print(f"  Backend          : {store_result.backend}")
        print(f"  Rows written     : {store_result.rows_written}")
        print(f"  Vector dim       : {store_result.vector_dim}")
        print(f"  Index created    : {store_result.index_created}")
        print(f"  Duration         : {store_result.duration_sec}s")
        print()

        dummy_query = np.random.randn(VECTOR_DIM).astype(np.float32)
        dummy_query = dummy_query / np.linalg.norm(dummy_query)
        results = retrieve_top_k(dummy_query, top_k=5, verbose=True)
        print(f"\nTop-5 results:")
        for i, r in enumerate(results):
            print(f"  {i+1}. {r.table_name}.{r.col_name:<28} "
                  f"sim={r.similarity:.4f}  {r.semantic_type}")


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
# Design: independent of ENCODER_MODE — works with all four strategies.
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
    """
    cursor.execute(f"TRUNCATE TABLE {FK_ADJACENCY_TABLE_NAME};")
    written = 0
    for edge in fk_edges:
        cursor.execute(f"""
            INSERT INTO {FK_ADJACENCY_TABLE_NAME}
                (from_col_id, from_col_name, from_table_id, from_table_name,
                 to_col_id,   to_col_name,   to_table_id,   to_table_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            edge.get("from_col_id",   ""),
            edge.get("from_col_name", ""),
            edge.get("from_table_id", ""),
            edge.get("from_table",    ""),   # key is "from_table" not "from_table_name"
            edge.get("to_col_id",     ""),
            edge.get("to_col_name",   ""),
            edge.get("to_table_id",   ""),
            edge.get("to_table",      ""),   # key is "to_table" not "to_table_name"
        ))
        written += 1
    return written


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
# Independent of ENCODER_MODE — works with all four encoder strategies.
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
                        for row in rows:
                            cur.execute(f"""
                                INSERT INTO {TABLE_METADATA_TABLE_NAME}
                                    (table_id, table_name, display_col_id, display_col_name)
                                VALUES (%s, %s, %s, %s);
                            """, (
                                row["table_id"],
                                row["table_name"],
                                row["display_col_id"],
                                row["display_col_name"],
                            ))
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