# =============================================================================
# ingestion/graph_embedder.py
# VEDA — Unified Data Graph: Phase 3 (Node Embeddings)
#
# Responsibility:
#   - Embeds column/table graph nodes into graph_node_embeddings using BGE
#   - Reuses existing doc_chunks embeddings for chunk nodes (no re-embedding)
#   - Provides query-time helpers: embed_text_bge, retrieve_graph_seeds
#   - In-memory fallback for seed retrieval
#
# Only invoked when UNIFIED_GRAPH_ENABLED + GRAPH_EMBED_ENABLED.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

from ingestion.db_abstraction import (
    INTERNAL_DB_AVAILABLE,
    get_internal_connection,
    release_internal_connection,
    DICT_CURSOR,
)
from ingestion import graph_persist
from ingestion.column_text import build_enriched_column_text
from config import (
    GRAPH_NODES_TABLE,
    GRAPH_NODE_EMB_TABLE,
    GRAPH_NODE_EMB_DIM,
    GRAPH_TABLE_SENTENCE_TEMPLATE,
    DOC_CHUNKS_TABLE_NAME,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# In-memory fallback store
# =============================================================================

_IN_MEMORY_NODE_EMB: List[dict] = []


def _get_bge_model():
    from ingestion.biencoder import _get_biencoder
    return _get_biencoder()


# =============================================================================
# Output data structure
# =============================================================================

@dataclass
class GraphEmbedResult:
    nodes_embedded: int
    source_id:      str
    backend:        str
    duration_sec:   float
    stats:          dict = field(default_factory=dict)


# =============================================================================
# Schema management
# =============================================================================

def _create_node_emb_table(cursor) -> None:
    # Drop table if embedding dimension has changed (e.g. MiniLM 384 → BGE 1024)
    try:
        cursor.execute(f"""
            SELECT atttypmod - 4 AS dim
            FROM pg_attribute JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
            WHERE pg_class.relname = '{GRAPH_NODE_EMB_TABLE}' AND pg_attribute.attname = 'embedding'
        """)
        row = cursor.fetchone()
        if row and row[0] != GRAPH_NODE_EMB_DIM:
            cursor.execute(f"DROP TABLE IF EXISTS {GRAPH_NODE_EMB_TABLE};")
    except Exception:
        pass
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {GRAPH_NODE_EMB_TABLE} (
            node_id   TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            embedding vector({GRAPH_NODE_EMB_DIM})
        );
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{GRAPH_NODE_EMB_TABLE}_source
        ON {GRAPH_NODE_EMB_TABLE} (source_id);
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{GRAPH_NODE_EMB_TABLE}_type
        ON {GRAPH_NODE_EMB_TABLE} (node_type);
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{GRAPH_NODE_EMB_TABLE}_embedding
        ON {GRAPH_NODE_EMB_TABLE}
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 10);
    """)


# =============================================================================
# Query-time shared helper
# =============================================================================

def embed_text_bge(text: str) -> np.ndarray:
    model = _get_bge_model()
    if model is None:
        raise RuntimeError("BGE model unavailable")
    from config import BIENCODER_PASSAGE_PREFIX
    vec = model.encode(
        [BIENCODER_PASSAGE_PREFIX + text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return np.asarray(vec, dtype=np.float32)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (mat / norms).astype(np.float32)


def _vec_str(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vec.tolist()) + "]"


# =============================================================================
# Load nodes for a source
# =============================================================================

def _load_nodes_for_source(source_id: str, node_types: List[str]) -> List:
    if not INTERNAL_DB_AVAILABLE:
        return [
            graph_persist._dict_to_node(n)
            for n in graph_persist._IN_MEMORY_NODES
            if n["source_id"] == source_id and n["node_type"] in node_types
        ]
    try:
        conn = get_internal_connection()
    except Exception:
        return []
    try:
        cur = conn.cursor(cursor_factory=DICT_CURSOR)
        placeholders = ",".join(["%s"] * len(node_types))
        cur.execute(f"""
            SELECT node_id, node_type, source_id, ref_id, table_id, name,
                   table_name, semantic_type, data_type, is_pk, is_fk, attrs
            FROM {GRAPH_NODES_TABLE}
            WHERE source_id = %s AND node_type IN ({placeholders});
        """, [source_id] + list(node_types))
        rows = cur.fetchall()
        try: cur.close()
        except Exception: pass
    finally:
        release_internal_connection(conn)
    return [graph_persist._row_to_node(r) for r in rows]


# =============================================================================
# Main entry point — embed graph nodes
# =============================================================================

def embed_graph_nodes(source_id: str, verbose: bool = False) -> GraphEmbedResult:
    """
    Embeds column + table nodes for source_id via BGE, copies chunk-node
    vectors from doc_chunks, and persists into graph_node_embeddings.
    """
    t0 = time.time()
    backend = "in_memory_fallback" if not INTERNAL_DB_AVAILABLE else "postgres"

    col_table_nodes = _load_nodes_for_source(source_id, ["column", "table"])

    # Load sampled column values for enriched text (populated by value_sampler in step 6)
    try:
        from ingestion.value_sampler import get_sampled_columns
        _sampled = get_sampled_columns()
    except Exception:
        _sampled = {}

    sentences: List[str] = []
    node_meta: List[Tuple[str, str]] = []   # (node_id, node_type)
    for n in col_table_nodes:
        if n.node_type == "column":
            _attrs = n.attrs or {}
            sent = build_enriched_column_text(
                col_name      = n.name,
                table_name    = n.table_name or "",
                semantic_type = n.semantic_type or "",
                is_pk         = n.is_pk,
                is_fk         = n.is_fk,
                fk_ref_table  = _attrs.get("fk_ref_table"),
                fk_ref_col    = _attrs.get("fk_ref_col"),
                sampled       = _sampled.get(n.ref_id),
                style         = "minilm",
            )
        else:
            sent = GRAPH_TABLE_SENTENCE_TEMPLATE.format(table_name=n.name)
        sentences.append(sent)
        node_meta.append((n.node_id, n.node_type))

    embeddings: Optional[np.ndarray] = None
    if sentences:
        try:
            from config import BIENCODER_PASSAGE_PREFIX
            model = _get_bge_model()
            if model is None:
                raise RuntimeError("BGE model unavailable")
            embeddings = model.encode(
                [BIENCODER_PASSAGE_PREFIX + s for s in sentences],
                normalize_embeddings = True,
                show_progress_bar    = False,
            ).astype(np.float32)
            embeddings = _l2_normalize(embeddings)
        except Exception as e:
            logger.warning("BGE model unavailable (%s) — col/table nodes skipped", e)
            embeddings = None
            node_meta = []

    # ------------------------------------------------------------------
    # Chunk nodes: copy vectors from doc_chunks (no re-embedding)
    # ------------------------------------------------------------------
    chunk_rows: List[Tuple[str, np.ndarray]] = []
    if INTERNAL_DB_AVAILABLE:
        try:
            conn = get_internal_connection()
            try:
                cur = conn.cursor(cursor_factory=DICT_CURSOR)
                cur.execute(
                    f"SELECT chunk_id, embedding FROM {DOC_CHUNKS_TABLE_NAME} "
                    f"WHERE source_id = %s;",
                    (source_id,),
                )
                rows = cur.fetchall()
                try: cur.close()
                except Exception: pass
            finally:
                release_internal_connection(conn)
            for r in rows:
                raw = r["embedding"]
                if raw is None:
                    continue
                if hasattr(raw, "tolist"):
                    fvec = np.array(raw, dtype=np.float32)
                else:
                    fvec = np.array(
                        [float(x) for x in str(raw).strip("[]").split(",") if x.strip()],
                        dtype=np.float32,
                    )
                chunk_rows.append((graph_persist.chunk_node_id(r["chunk_id"]), fvec))
        except Exception as e:
            logger.warning("chunk embedding copy failed (%s)", e)

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    nodes_embedded = 0

    if not INTERNAL_DB_AVAILABLE:
        global _IN_MEMORY_NODE_EMB
        _IN_MEMORY_NODE_EMB = [
            r for r in _IN_MEMORY_NODE_EMB if r["source_id"] != source_id
        ]
        if embeddings is not None:
            for (node_id, node_type), emb in zip(node_meta, embeddings):
                _IN_MEMORY_NODE_EMB.append({
                    "node_id":   node_id,
                    "node_type": node_type,
                    "source_id": source_id,
                    "embedding": emb,
                })
                nodes_embedded += 1
        for (node_id, fvec) in chunk_rows:
            _IN_MEMORY_NODE_EMB.append({
                "node_id":   node_id,
                "node_type": "chunk",
                "source_id": source_id,
                "embedding": fvec,
            })
            nodes_embedded += 1
    else:
        conn = get_internal_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    _create_node_emb_table(cur)
                    cur.execute(
                        f"DELETE FROM {GRAPH_NODE_EMB_TABLE} WHERE source_id = %s;",
                        (source_id,),
                    )
                    if embeddings is not None:
                        for (node_id, node_type), emb in zip(node_meta, embeddings):
                            try:
                                cur.execute(f"""
                                    INSERT INTO {GRAPH_NODE_EMB_TABLE}
                                        (node_id, node_type, source_id, embedding)
                                    VALUES (%s, %s, %s, %s::vector)
                                    ON CONFLICT (node_id) DO UPDATE SET
                                        node_type = EXCLUDED.node_type,
                                        source_id = EXCLUDED.source_id,
                                        embedding = EXCLUDED.embedding;
                                """, (node_id, node_type, source_id, _vec_str(emb)))
                                nodes_embedded += 1
                            except Exception:
                                pass
                    for (node_id, fvec) in chunk_rows:
                        try:
                            cur.execute(f"""
                                INSERT INTO {GRAPH_NODE_EMB_TABLE}
                                    (node_id, node_type, source_id, embedding)
                                VALUES (%s, %s, %s, %s::vector)
                                ON CONFLICT (node_id) DO UPDATE SET
                                    node_type = EXCLUDED.node_type,
                                    source_id = EXCLUDED.source_id,
                                    embedding = EXCLUDED.embedding;
                            """, (node_id, "chunk", source_id, _vec_str(fvec)))
                            nodes_embedded += 1
                        except Exception:
                            pass
        finally:
            release_internal_connection(conn)

    duration = round(time.time() - t0, 4)
    if verbose:
        logger.info(
            "%d node embeddings (%d col/table + %d chunk), backend=%s, %ss",
            nodes_embedded, len(node_meta), len(chunk_rows), backend, duration,
        )

    return GraphEmbedResult(
        nodes_embedded = nodes_embedded,
        source_id      = source_id,
        backend        = backend,
        duration_sec   = duration,
        stats          = {
            "col_table_nodes": len(node_meta),
            "chunk_nodes":     len(chunk_rows),
        },
    )


# =============================================================================
# Query-time — seed retrieval
# =============================================================================

def _retrieve_seeds_from_memory(
    query_vector: np.ndarray,
    top_k: int,
    source_ids: Optional[List[str]],
    node_types: Optional[List[str]],
) -> List[Tuple[str, str, float]]:
    pool = _IN_MEMORY_NODE_EMB
    if source_ids:
        sset = set(source_ids)
        pool = [r for r in pool if r["source_id"] in sset]
    if node_types:
        ntset = set(node_types)
        pool = [r for r in pool if r["node_type"] in ntset]
    if not pool:
        return []
    mat = np.stack([r["embedding"] for r in pool])
    sims = mat @ query_vector
    idxs = np.argsort(sims)[::-1][:top_k]
    return [
        (pool[i]["node_id"], pool[i]["node_type"], round(float(sims[i]), 6))
        for i in idxs
    ]


def retrieve_graph_seeds(
    query_vector: np.ndarray,
    top_k: int,
    source_ids: Optional[List[str]] = None,
    node_types: Optional[List[str]] = None,
) -> List[Tuple[str, str, float]]:
    """
    Cosine ANN over graph_node_embeddings. Returns (node_id, node_type, similarity).
    """
    if not INTERNAL_DB_AVAILABLE:
        return _retrieve_seeds_from_memory(query_vector, top_k, source_ids, node_types)

    vec_str = _vec_str(np.asarray(query_vector, dtype=np.float32))

    where_parts = ["1=1"]
    params: list = [vec_str]
    if source_ids:
        ph = ",".join(["%s"] * len(source_ids))
        where_parts.append(f"source_id IN ({ph})")
        params += list(source_ids)
    if node_types:
        ph = ",".join(["%s"] * len(node_types))
        where_parts.append(f"node_type IN ({ph})")
        params += list(node_types)
    where_sql = " AND ".join(where_parts)
    params += [vec_str, top_k]

    try:
        conn = get_internal_connection()
    except Exception:
        return _retrieve_seeds_from_memory(query_vector, top_k, source_ids, node_types)
    try:
        cur = conn.cursor(cursor_factory=DICT_CURSOR)
        cur.execute(f"""
            SELECT node_id, node_type,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM {GRAPH_NODE_EMB_TABLE}
            WHERE {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, params)
        rows = cur.fetchall()
        try: cur.close()
        except Exception: pass
    except Exception:
        release_internal_connection(conn)
        return _retrieve_seeds_from_memory(query_vector, top_k, source_ids, node_types)
    finally:
        release_internal_connection(conn)

    results = [
        (r["node_id"], r["node_type"], round(float(r["similarity"]), 6))
        for r in rows
    ]
    if not results and _IN_MEMORY_NODE_EMB:
        return _retrieve_seeds_from_memory(query_vector, top_k, source_ids, node_types)
    return results
