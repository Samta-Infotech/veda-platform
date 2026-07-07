# =============================================================================
# ingestion/chunk_embedder.py
# VEDA — Step: Document Chunk Embedding (Phase 2)
#
# Responsibility:
#   - Accepts DocumentChunk objects from a document connector
#   - Embeds each chunk using the shared BGE-M3 model singleton (WP3)
#   - Persists chunk embeddings to the doc_chunks table in VEDA_INTERNAL_DB
#   - Provides retrieve_top_k_chunks() for RAG retrieval at query time
#
# doc_chunks uses 1024-dim BGE-M3 embeddings (WP3), the same model + space as the
# column/table/graph stores — one model load per process.
#
# Schema:
#   doc_chunks (
#       chunk_id    TEXT PRIMARY KEY,
#       source_id   TEXT NOT NULL,
#       doc_id      TEXT NOT NULL,
#       doc_name    TEXT NOT NULL,
#       chunk_index INTEGER NOT NULL,
#       text        TEXT NOT NULL,
#       page_num    INTEGER,
#       embedding   vector(1024)
#   )
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from connectors.base import DocumentChunk
from ingestion.db_abstraction import (
    INTERNAL_DB_AVAILABLE,
    get_internal_connection,
    release_internal_connection,
    DICT_CURSOR,
)
from config import DOC_CHUNKS_TABLE_NAME, BIENCODER_DIM
from utils.logger import get_logger

logger = get_logger(__name__)
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# In-memory fallback store (used when pgvector is unavailable or write fails)
# =============================================================================

_IN_MEMORY_CHUNKS: List[dict] = []


def _store_in_memory(chunks: List[DocumentChunk], embeddings: np.ndarray, source_id: str) -> None:
    global _IN_MEMORY_CHUNKS
    _IN_MEMORY_CHUNKS = [r for r in _IN_MEMORY_CHUNKS if r["source_id"] != source_id]
    for chunk, emb in zip(chunks, embeddings):
        _IN_MEMORY_CHUNKS.append({
            "chunk_id":    chunk.chunk_id,
            "source_id":   chunk.source_id,
            "doc_id":      chunk.doc_id,
            "doc_name":    chunk.doc_name,
            "chunk_index": chunk.chunk_index,
            "text":        chunk.text,
            "page_num":    chunk.page_num,
            "embedding":   emb,
        })


def _retrieve_from_memory(
    query_vector: np.ndarray,
    source_ids:   List[str],
    top_k:        int,
) -> List["ChunkRetrievalResult"]:
    pool = _IN_MEMORY_CHUNKS
    if source_ids:
        pool = [r for r in pool if r["source_id"] in source_ids]
    if not pool:
        return []
    mat  = np.stack([r["embedding"] for r in pool])
    sims = mat @ query_vector
    idxs = np.argsort(sims)[::-1][:top_k]
    return [
        ChunkRetrievalResult(
            chunk_id    = pool[i]["chunk_id"],
            source_id   = pool[i]["source_id"],
            doc_id      = pool[i]["doc_id"],
            doc_name    = pool[i]["doc_name"],
            chunk_index = pool[i]["chunk_index"],
            text        = pool[i]["text"],
            page_num    = pool[i]["page_num"],
            similarity  = round(float(sims[i]), 6),
        )
        for i in idxs
    ]


# =============================================================================
# Output data structures
# =============================================================================

@dataclass
class ChunkEmbedderResult:
    """Result of embedding and storing a batch of document chunks."""
    chunks_embedded: int
    chunks_skipped:  int
    docs_processed:  int
    source_id:       str
    backend:         str
    duration_sec:    float
    stats:           dict = field(default_factory=dict)


@dataclass
class ChunkRetrievalResult:
    """A single chunk returned by RAG retrieval."""
    chunk_id:    str
    source_id:   str
    doc_id:      str
    doc_name:    str
    chunk_index: int
    text:        str
    page_num:    Optional[int]
    similarity:  float


# =============================================================================
# Schema management
# =============================================================================

def _create_doc_chunks_table(cursor) -> None:
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    # Drop the table if the embedding dimension changed (MiniLM 384 → BGE-M3 1024, WP3),
    # mirroring the graph_embedder guard — a clean re-ingest recreates it at the new dim.
    try:
        cursor.execute(f"""
            SELECT atttypmod - 4 AS dim FROM pg_attribute
            WHERE attrelid = '{DOC_CHUNKS_TABLE_NAME}'::regclass AND attname = 'embedding'
        """)
        row = cursor.fetchone()
        if row and row[0] != BIENCODER_DIM:
            cursor.execute(f"DROP TABLE IF EXISTS {DOC_CHUNKS_TABLE_NAME};")
    except Exception:
        pass  # table absent → nothing to drop
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {DOC_CHUNKS_TABLE_NAME} (
            chunk_id    TEXT PRIMARY KEY,
            source_id   TEXT NOT NULL,
            doc_id      TEXT NOT NULL,
            doc_name    TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text        TEXT NOT NULL,
            page_num    INTEGER,
            doc_date    TIMESTAMPTZ,
            embedding   vector({BIENCODER_DIM})
        );
    """)
    # Migrate tables created before doc_date was added to the schema.
    cursor.execute(f"""
        ALTER TABLE {DOC_CHUNKS_TABLE_NAME}
        ADD COLUMN IF NOT EXISTS doc_date TIMESTAMPTZ;
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{DOC_CHUNKS_TABLE_NAME}_source
        ON {DOC_CHUNKS_TABLE_NAME} (source_id);
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{DOC_CHUNKS_TABLE_NAME}_embedding
        ON {DOC_CHUNKS_TABLE_NAME}
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200);
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{DOC_CHUNKS_TABLE_NAME}_doc_date
        ON {DOC_CHUNKS_TABLE_NAME} (doc_date)
        WHERE doc_date IS NOT NULL;
    """)


# =============================================================================
# Embedding helper
# =============================================================================

def _embed_chunks(texts: List[str]) -> np.ndarray:
    """Embeds a list of texts using the shared BGE-M3 singleton (WP3, 1024-dim,
    already L2-normalized)."""
    from ingestion import m3_encoder
    return m3_encoder.encode_dense(texts)


# =============================================================================
# Public entry point — ingestion
# =============================================================================

def run_chunk_embedder(
    chunks:    List[DocumentChunk],
    source_id: str,
    verbose:   bool = False,
) -> ChunkEmbedderResult:
    """
    Embeds DocumentChunk objects and persists them to the doc_chunks table.

    Called by main.py during document ingestion (--ingest-docs mode).
    The doc_chunks table is truncated per source_id on each ingestion run
    so re-running is safe and idempotent.

    Parameters
    ----------
    chunks    : List[DocumentChunk] from a document connector
    source_id : identifies which VEDA_SOURCE these chunks came from
    verbose   : print progress

    Returns
    -------
    ChunkEmbedderResult
    """
    t0 = time.time()

    if not chunks:
        return ChunkEmbedderResult(
            chunks_embedded = 0,
            chunks_skipped  = 0,
            docs_processed  = 0,
            source_id       = source_id,
            backend         = "no_chunks",
            duration_sec    = 0.0,
        )

    if verbose:
        print(f"[ChunkEmbedder] Embedding {len(chunks)} chunks from source '{source_id}'...")

    texts       = [c.text for c in chunks]
    doc_ids     = set(c.doc_id for c in chunks)
    skipped     = 0
    backend     = "in_memory_fallback"

    try:
        embeddings = _embed_chunks(texts)
    except Exception as e:
        if verbose:
            print(f"  ⚠ BGE-M3 embedding failed ({e}) — chunks not stored")
        return ChunkEmbedderResult(
            chunks_embedded = 0,
            chunks_skipped  = len(chunks),
            docs_processed  = len(doc_ids),
            source_id       = source_id,
            backend         = "embedding_failed",
            duration_sec    = round(time.time() - t0, 4),
            stats           = {"error": str(e)},
        )

    if INTERNAL_DB_AVAILABLE:
        try:
            conn = get_internal_connection()
            try:
                with conn:
                    with conn.cursor() as cur:
                        _create_doc_chunks_table(cur)
                        # Clear previous run for this source
                        cur.execute(
                            f"DELETE FROM {DOC_CHUNKS_TABLE_NAME} WHERE source_id = %s;",
                            (source_id,),
                        )
                        for chunk, emb in zip(chunks, embeddings):
                            vec_str = "[" + ",".join(f"{v:.8f}" for v in emb.tolist()) + "]"
                            try:
                                cur.execute(f"""
                                    INSERT INTO {DOC_CHUNKS_TABLE_NAME}
                                        (chunk_id, source_id, doc_id, doc_name,
                                         chunk_index, text, page_num, embedding)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                                    ON CONFLICT (chunk_id) DO UPDATE SET
                                        text      = EXCLUDED.text,
                                        embedding = EXCLUDED.embedding;
                                """, (
                                    chunk.chunk_id, chunk.source_id, chunk.doc_id,
                                    chunk.doc_name, chunk.chunk_index, chunk.text,
                                    chunk.page_num, vec_str,
                                ))
                            except Exception:
                                skipped += 1
            finally:
                release_internal_connection(conn)
            backend = "pgvector"
        except Exception as e:
            print(f"  ⚠ [ChunkEmbedder] pgvector store failed ({e}) — falling back to in-memory")
            _store_in_memory(chunks, embeddings, source_id)
    else:
        _store_in_memory(chunks, embeddings, source_id)

    embedded = len(chunks) - skipped
    duration = round(time.time() - t0, 4)

    if verbose:
        print(f"  Chunks embedded  : {embedded}")
        print(f"  Chunks skipped   : {skipped}")
        print(f"  Docs processed   : {len(doc_ids)}")
        print(f"  Backend          : {backend}")
        print(f"  Duration         : {duration}s")
        print("[ChunkEmbedder] Done.\n")

    return ChunkEmbedderResult(
        chunks_embedded = embedded,
        chunks_skipped  = skipped,
        docs_processed  = len(doc_ids),
        source_id       = source_id,
        backend         = backend,
        duration_sec    = duration,
        stats           = {
            "total_chunks": len(chunks),
            "total_docs":   len(doc_ids),
            "backend":      backend,
        },
    )


# =============================================================================
# Public entry point — query time
# =============================================================================

def retrieve_top_k_chunks(
    query_vector:    np.ndarray,
    source_ids:      List[str] = None,
    top_k:           int = 5,
    temporal_filter: object = None,  # TemporalFilter from temporal_parser.py
    verbose:         bool = False,
) -> List[ChunkRetrievalResult]:
    """
    Cosine similarity search over the doc_chunks table.

    Called by query/rag_layer.py at query time.

    Parameters
    ----------
    query_vector    : 1-D float32 array of shape (384,)
    source_ids      : restrict search to these source IDs (None = all sources)
    top_k           : number of chunks to return
    temporal_filter : TemporalFilter from L1. When set, only chunks whose
                      doc_date falls within [start, end] are retrieved.
                      None = no date filtering (default).

    Returns
    -------
    List[ChunkRetrievalResult] sorted by descending similarity
    """
    if not INTERNAL_DB_AVAILABLE:
        results = _retrieve_from_memory(query_vector, list(source_ids or []), top_k)
        if verbose and results:
            print(f"[ChunkRetrieval] in-memory fallback: {len(results)} chunks")
        return results

    vec_str = "[" + ",".join(f"{v:.8f}" for v in query_vector.tolist()) + "]"

    # Build temporal filter clause (Improvement 1)
    temporal_clause = ""
    temporal_params: list = []
    if temporal_filter is not None:
        start = getattr(temporal_filter, 'start', None)
        end   = getattr(temporal_filter, 'end',   None)
        if start and end:
            temporal_clause = "AND doc_date BETWEEN %s AND %s"
            temporal_params = [start, end]
        elif start:
            temporal_clause = "AND doc_date >= %s"
            temporal_params = [start]
        elif end:
            temporal_clause = "AND doc_date <= %s"
            temporal_params = [end]

    try:
        conn = get_internal_connection()
    except Exception as _e:
        if verbose:
            print(f"[ChunkEmbedder] DB unavailable, skipping chunk retrieval: {_e}")
        return _retrieve_from_memory(query_vector, list(source_ids or []), top_k)
    try:
        cur = conn.cursor(cursor_factory=DICT_CURSOR)
        # HNSW: pin ef_search for THIS transaction so the served ANN ordering matches
        # the tuned recall target (WP2). Resolved per-source via the one shared helper
        # (env → SubstrateVersion → default 40); SET LOCAL is released at COMMIT, which
        # is PgBouncer-transaction-pool-safe.
        from storage_adapters.reader import _resolve_ef_search
        _ef = _resolve_ef_search(source_ids[0] if source_ids else None)
        cur.execute("BEGIN")
        cur.execute(f"SET LOCAL hnsw.ef_search = {int(_ef)}")
        if source_ids:
            placeholders = ",".join(["%s"] * len(source_ids))
            cur.execute(f"""
                SELECT chunk_id, source_id, doc_id, doc_name,
                       chunk_index, text, page_num,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {DOC_CHUNKS_TABLE_NAME}
                WHERE source_id IN ({placeholders})
                {temporal_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, [vec_str] + source_ids + temporal_params + [vec_str, top_k])
        else:
            cur.execute(f"""
                SELECT chunk_id, source_id, doc_id, doc_name,
                       chunk_index, text, page_num,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {DOC_CHUNKS_TABLE_NAME}
                WHERE 1=1
                {temporal_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, [vec_str] + temporal_params + [vec_str, top_k])
        rows = cur.fetchall()
        try: cur.execute("COMMIT")
        except Exception: pass
        try: cur.close()
        except Exception: pass
    except Exception as _e:
        # doc_chunks absent/unreadable → document source not ingested yet.
        # Degrade gracefully (empty result) instead of crashing the query;
        # the connection is poisoned after the error, so roll back before reuse.
        try: conn.rollback()
        except Exception: pass
        if "does not exist" in str(_e):
            logger.warning(
                "Doc chunk store '%s' missing for source(s) %s — run document "
                "ingestion (`python main.py --ingest-docs`). Returning no chunks.",
                DOC_CHUNKS_TABLE_NAME, source_ids or "all")
            if verbose:
                print(f"  [RAG] ⚠  No document index — '{DOC_CHUNKS_TABLE_NAME}' not built. "
                      f"Run `python main.py --ingest-docs` to ingest documents.")
        else:
            logger.warning("Chunk retrieval failed: %s", _e)
        return []
    finally:
        release_internal_connection(conn)

    results = [
        ChunkRetrievalResult(
            chunk_id    = row["chunk_id"],
            source_id   = row["source_id"],
            doc_id      = row["doc_id"],
            doc_name    = row["doc_name"],
            chunk_index = row["chunk_index"],
            text        = row["text"],
            page_num    = row["page_num"],
            similarity  = round(float(row["similarity"]), 6),
        )
        for row in rows
    ]

    if not results and _IN_MEMORY_CHUNKS:
        if verbose:
            print("[ChunkRetrieval] pgvector returned 0 rows — falling back to in-memory store")
        results = _retrieve_from_memory(query_vector, list(source_ids or []), top_k)

    if verbose:
        print(f"[ChunkRetrieval] Top-{top_k} chunks retrieved ({len(results)} found)")
        for r in results[:3]:
            print(f"  {r.doc_name}[{r.chunk_index}]  sim={r.similarity:.4f}")

    return results