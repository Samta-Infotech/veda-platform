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

import math
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
from ingestion import m3_encoder
import importlib
from veda.rbac_filter import filter_doc_chunks

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
    # True when this chunk came from unified-graph traversal (PPR over the entity
    # bridge) rather than cosine ANN search. Its `similarity` is then a PPR mass,
    # NOT a cosine score — the two are different scales and must not be compared
    # against cosine thresholds. See run_hybrid_layer()'s doc-filter step.
    from_graph:  bool = False


# =============================================================================
# Schema management
# =============================================================================

def _create_doc_chunks_table(cursor) -> None:
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    # Drop the table if the embedding dimension changed (MiniLM 384 → BGE-M3 1024, WP3),
    # mirroring the graph_embedder guard — a clean re-ingest recreates it at the new dim.
    # Use to_regclass (returns NULL for a missing table) rather than
    # '<table>'::regclass (which RAISES UndefinedTable on first-ever ingestion and
    # aborts the whole transaction, so the CREATE below then fails with "current
    # transaction is aborted" → silent in-memory fallback, embeddings never stored).
    cursor.execute(f"""
        SELECT atttypmod - 4 AS dim FROM pg_attribute
        WHERE attrelid = to_regclass('{DOC_CHUNKS_TABLE_NAME}') AND attname = 'embedding'
    """)
    row = cursor.fetchone()
    if row and row[0] != BIENCODER_DIM:
        cursor.execute(f"DROP TABLE IF EXISTS {DOC_CHUNKS_TABLE_NAME};")
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


def _ensure_chunk_sparse_table(cursor) -> None:
    """Learned-sparse (lexical) weights for document chunks — the doc-chunk mirror of
    ingestion/sparse_index.py's column_sparse_v1/table_sparse_v1. Dense cosine similarity
    alone under-weights a rare, distinctive term a query names verbatim: on a 165-chunk
    document, the one section actually discussing "POSH" scored 0.425 while several
    unrelated sections scored 0.50+ purely on generic phrasing overlap, so it never
    reached the top-k and the query was answered as "not in the document" even though it
    was. This table lets retrieve_top_k_chunks() re-rank the dense candidate pool with a
    lexical signal, the same two-signal idea sparse_ranker.py already uses for columns."""
    from config import CHUNK_SPARSE_TABLE
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {CHUNK_SPARSE_TABLE} (
            chunk_id  TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            weights   JSONB
        );
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{CHUNK_SPARSE_TABLE}_source
        ON {CHUNK_SPARSE_TABLE} (source_id);
    """)


# =============================================================================
# Embedding helper
# =============================================================================

def _embed_chunks(texts: List[str]) -> np.ndarray:
    """Embeds a list of texts using the shared BGE-M3 singleton (WP3, 1024-dim,
    already L2-normalized)."""
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

    # Learned-sparse (lexical) weights alongside the dense embeddings — see
    # _ensure_chunk_sparse_table's docstring for why dense-only similarity misses rare,
    # distinctive terms. Best-effort: a failure here degrades to dense-only retrieval,
    # it must never fail the ingestion the way a dense-embedding failure does above.
    try:
        from ingestion import m3_encoder
        sparse_weights = m3_encoder.encode_sparse(texts)
    except Exception as e:
        if verbose:
            print(f"  ⚠ sparse encoding failed ({e}) — chunks stored dense-only")
        sparse_weights = None

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

                        if sparse_weights is not None:
                            import json as _json
                            from config import CHUNK_SPARSE_TABLE
                            _ensure_chunk_sparse_table(cur)
                            cur.execute(
                                f"DELETE FROM {CHUNK_SPARSE_TABLE} WHERE source_id = %s;",
                                (source_id,),
                            )
                            for chunk, w in zip(chunks, sparse_weights):
                                try:
                                    cur.execute(f"""
                                        INSERT INTO {CHUNK_SPARSE_TABLE}
                                            (chunk_id, source_id, weights)
                                        VALUES (%s, %s, %s)
                                        ON CONFLICT (chunk_id) DO UPDATE SET
                                            weights = EXCLUDED.weights;
                                    """, (chunk.chunk_id, chunk.source_id, _json.dumps(w)))
                                except Exception:
                                    pass
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

def _try_current_context():
    """The ambient ``RequestContext``, or ``None`` — tries both module names a
    caller may have imported this under (same landmine ``veda_hybrid.py``'s
    ``_scope_source_ids`` already works around: ``veda_core.context`` and a bare
    ``context`` can be two distinct module objects with independent ContextVars,
    depending on which sys.path entry resolved the import first)."""
    for modname in ("veda_core.context", "context"):
        try:
            ctx = importlib.import_module(modname).try_current()
            if ctx is not None:
                return ctx
        except Exception:
            continue
    return None


# RAG_TOP_K is small (query-facing result count); over-fetch by this factor so
# RBAC filtering (rbac_filter.filter_doc_chunks, applied after this query) has
# enough same-similarity-order candidates to still fill top_k once denied
# documents' chunks are dropped, without changing the query itself per caller.
_RBAC_OVERFETCH_MULTIPLIER = 4

# RRF constant. The usual k=60 (sparse_ranker.py / retrieval_select.py) is tuned for
# deep candidate lists over large corpora, where a 1-2 position rank gap is noise; a
# per-document RAG corpus here is two orders of magnitude smaller (dozens-low hundreds
# of chunks), so k=60 flattens rank position almost to nothing — a chunk absent from
# the dense pool (sentinel rank = pool size) barely loses ground even against a chunk
# ranked #1-3 by sparse alone (observed live: a chunk sparse-ranked #3 for "POSH" still
# lost to weaker-but-dense-present chunks under k=60). A smaller k lets rank position
# actually decide ties at this scale.
_SPARSE_RRF_K = 10.0

# Hard cap on how many of a scope's sparse rows a single query will load and score in
# Python. A per-document RAG corpus here is small (dozens–low hundreds of chunks per
# source, same order of magnitude as SPARSE_FIT_MAX_DOCS elsewhere in this codebase);
# this is a safety backstop against an unexpectedly huge source, not a tuned limit.
_SPARSE_SCAN_CAP = 4000


def _sparse_candidates(source_ids, query_sparse: dict, limit: int, verbose: bool = False) -> list:
    """Independently score EVERY chunk's persisted sparse weights against the query's
    (bounded by _SPARSE_SCAN_CAP), returning the top `limit` as ChunkRetrievalResult
    (similarity=0.0 — these are ranked by sparse score alone, dense similarity was never
    computed for them). This is a SEPARATE retrieval pass, not a re-score of the dense
    candidate pool: a chunk the dense cosine search ranks outside its own fetch window
    (e.g. a short, keyword-dense passage that dense similarity under-weights) can still
    be found here and unioned in — RRF-fusing only the dense pool's own members can
    never rescue a chunk dense similarity excluded before fusion even runs."""
    if not query_sparse:
        return []
    try:
        from config import CHUNK_SPARSE_TABLE
        conn = get_internal_connection()
        try:
            with conn.cursor(cursor_factory=DICT_CURSOR) as cur:
                if source_ids:
                    placeholders = ",".join(["%s"] * len(source_ids))
                    cur.execute(f"""
                        SELECT dc.chunk_id, dc.source_id, dc.doc_id, dc.doc_name,
                               dc.chunk_index, dc.text, dc.page_num, cs.weights
                        FROM {CHUNK_SPARSE_TABLE} cs
                        JOIN {DOC_CHUNKS_TABLE_NAME} dc ON dc.chunk_id = cs.chunk_id
                        WHERE dc.source_id IN ({placeholders})
                        LIMIT %s
                    """, list(source_ids) + [_SPARSE_SCAN_CAP])
                else:
                    cur.execute(f"""
                        SELECT dc.chunk_id, dc.source_id, dc.doc_id, dc.doc_name,
                               dc.chunk_index, dc.text, dc.page_num, cs.weights
                        FROM {CHUNK_SPARSE_TABLE} cs
                        JOIN {DOC_CHUNKS_TABLE_NAME} dc ON dc.chunk_id = cs.chunk_id
                        LIMIT %s
                    """, [_SPARSE_SCAN_CAP])
                rows = cur.fetchall()
        finally:
            release_internal_connection(conn)
    except Exception as e:
        if verbose:
            print(f"  [ChunkRetrieval] sparse scan unavailable ({e}) — dense-only")
        return []

    # Corpus IDF over THIS scan: BGE-M3's learned-sparse weights rate a token's semantic
    # importance in isolation, not its rarity in THIS specific corpus — "Samta"/"policy"
    # score heavily on nearly every chunk of an HR handbook, since they genuinely are
    # important words, and that alone let two irrelevant sections outrank the one chunk
    # actually discussing "POSH" (observed live: query's two heaviest tokens were generic
    # company/document words present in most chunks, drowning out the 3 tokens unique to
    # the harassment section). Down-weighting by how many chunks a token appears in reproduces
    # classic BM25's core idea on top of the learned-sparse scores, penalizing exactly the
    # tokens common enough to be common to every section.
    doc_freq: dict = {}
    for row in rows:
        for tok in (row["weights"] or {}):
            doc_freq[tok] = doc_freq.get(tok, 0) + 1
    n_docs = max(len(rows), 1)
    idf = {tok: math.log((n_docs + 1.0) / (df + 1.0)) + 1.0 for tok, df in doc_freq.items()}

    def _dot(a: dict, b: dict) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(w * b[k] * idf.get(k, 1.0) for k, w in a.items() if k in b)

    scored = [(_dot(query_sparse, row["weights"] or {}), row) for row in rows]
    scored = [(s, row) for s, row in scored if s > 0.0]
    scored.sort(key=lambda sr: sr[0], reverse=True)
    return [
        ChunkRetrievalResult(
            chunk_id=row["chunk_id"], source_id=row["source_id"], doc_id=row["doc_id"],
            doc_name=row["doc_name"], chunk_index=row["chunk_index"], text=row["text"],
            page_num=row["page_num"], similarity=0.0,
        )
        for _score, row in scored[:limit]
    ]


def _fuse_dense_and_sparse(dense_results: list, source_ids, query_sparse: dict,
                          limit: int, verbose: bool = False) -> list:
    """Union the dense-fetched candidate pool with an independent sparse-side scan
    (_sparse_candidates), then RRF-fuse the two rank orders. A chunk found by only one
    signal keeps that signal's rank and a large (weak) rank for the other — it can still
    win overall if its one signal ranks it very highly, but never as easily as a chunk
    both signals agree on."""
    sparse_results = _sparse_candidates(source_ids, query_sparse, limit, verbose=verbose)
    if not sparse_results:
        return dense_results

    by_id = {r.chunk_id: r for r in dense_results}
    for r in sparse_results:
        by_id.setdefault(r.chunk_id, r)   # dense's own copy (with real similarity) wins ties

    dense_rank = {r.chunk_id: i for i, r in enumerate(dense_results)}
    sparse_rank = {r.chunk_id: i for i, r in enumerate(sparse_results)}
    worst_dense = len(dense_results)
    worst_sparse = len(sparse_results)

    def _rrf(chunk_id: str) -> float:
        dr = dense_rank.get(chunk_id, worst_dense)
        sr = sparse_rank.get(chunk_id, worst_sparse)
        return 1.0 / (_SPARSE_RRF_K + dr) + 1.0 / (_SPARSE_RRF_K + sr)

    fused = sorted(by_id.values(), key=lambda r: _rrf(r.chunk_id), reverse=True)
    return fused


def retrieve_top_k_chunks(
    query_vector:    np.ndarray,
    source_ids:      List[str] = None,
    top_k:           int = 5,
    temporal_filter: object = None,  # TemporalFilter from temporal_parser.py
    verbose:         bool = False,
    query_sparse:    Optional[dict] = None,
) -> List[ChunkRetrievalResult]:
    """
    Cosine similarity search over the doc_chunks table, optionally re-ranked against a
    learned-sparse (lexical) signal via an independent scan (see _fuse_dense_and_sparse).

    Called by query/rag_layer.py at query time.

    Parameters
    ----------
    query_vector    : 1-D float32 array of shape (384,)
    source_ids      : restrict search to these source IDs (None = all sources)
    top_k           : number of chunks to return
    temporal_filter : TemporalFilter from L1. When set, only chunks whose
                      doc_date falls within [start, end] are retrieved.
    query_sparse    : learned-sparse weights for the query (from
                      ingestion.m3_encoder.encode_query()), or None to skip lexical
                      re-ranking and use dense order alone (unchanged prior behavior).
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
        # Over-fetch: rbac_filter.filter_doc_chunks (applied below, after this
        # query, on the candidate list — same pattern as filter_retrieval_results)
        # may drop some of these on a restricted scope, so ask the shared,
        # RBAC-oblivious query for more than top_k up front rather than starving
        # the caller's final result count.
        fetch_limit = top_k * _RBAC_OVERFETCH_MULTIPLIER
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
            """, [vec_str] + source_ids + temporal_params + [vec_str, fetch_limit])
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
            """, [vec_str] + temporal_params + [vec_str, fetch_limit])
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

    # Union with an INDEPENDENT lexical scan, then RRF-fuse — before RBAC narrows the
    # pool. Re-ranking only the dense-fetched candidates can't rescue a chunk dense
    # cosine similarity ranked outside its own fetch window in the first place; the
    # independent sparse-side scan (_sparse_candidates) can still surface it here.
    if query_sparse:
        results = _fuse_dense_and_sparse(results, source_ids, query_sparse,
                                         fetch_limit, verbose=verbose)

    # RBAC (Gate 1): narrow the over-fetched candidate list to what the caller's
    # data-scope actually permits, THEN cap to the requested top_k — mirrors
    # veda.rbac_filter.filter_retrieval_results' own reasoning (the shared query
    # above stays RBAC-oblivious; only this per-request result list is narrowed).
    results = filter_doc_chunks(results, _try_current_context())[:top_k]

    if not results and _IN_MEMORY_CHUNKS:
        if verbose:
            print("[ChunkRetrieval] pgvector returned 0 rows — falling back to in-memory store")
        results = _retrieve_from_memory(query_vector, list(source_ids or []), top_k)

    if verbose:
        print(f"[ChunkRetrieval] Top-{top_k} chunks retrieved ({len(results)} found)")
        for r in results[:3]:
            print(f"  {r.doc_name}[{r.chunk_index}]  sim={r.similarity:.4f}")

    return results