# =============================================================================
# ingestion/chunk_linker.py
# VEDA — Unified Data Graph: Phase 2 (Chunk → Column Linking)
#
# Responsibility:
#   - Creates chunk nodes in graph_nodes for each DocumentChunk
#   - Links chunks to column/table nodes via three signals:
#       1. Value-overlap (mentions)   — column sampled values appear in chunk text
#       2. Name-match    (mentions)   — column/table name tokens appear in text
#       3. Embedding sim (about)      — MiniLM cosine between chunk and column sentence
#   - Per-chunk edge cap: keep top-N edges by weight
#   - Idempotent: scoped delete of chunk nodes + mentions/about edges per source_id
#
# Only invoked when UNIFIED_GRAPH_ENABLED + GRAPH_CHUNK_LINKING_ENABLED.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import re
from uuid import uuid4
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np

from ingestion.db_abstraction import (
    INTERNAL_DB_AVAILABLE,
    get_internal_connection,
    release_internal_connection,
    DICT_CURSOR,
)
from ingestion import graph_persist
from ingestion.graph_persist import (
    GraphNode,
    GraphEdge,
    chunk_node_id,
    GRAPH_NODES_TABLE,
    GRAPH_EDGES_TABLE,
)
from config import (
    GRAPH_LINK_VALUE_OVERLAP_MIN,
    GRAPH_LINK_NAME_MIN_TOKEN_LEN,
    GRAPH_LINK_EMBED_SIM_MIN,
    GRAPH_LINK_MAX_EDGES_PER_CHUNK,
    GRAPH_EDGE_WEIGHTS,
    GRAPH_EMBED_ENABLED,
    GRAPH_COLUMN_SENTENCE_TEMPLATE,
    COLUMN_VALUES_TABLE_NAME,
    GRAPH_LINK_NAME_STOPWORDS,
)


# =============================================================================
# Output data structure
# =============================================================================

@dataclass
class ChunkLinkResult:
    chunk_nodes_written: int
    link_edges_written:  int
    source_id:           str
    backend:             str
    duration_sec:        float
    stats:               dict = field(default_factory=dict)


# =============================================================================
# Tokenisation helper
# =============================================================================

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Generic schema tokens that create near-universal edges — loaded from config
# so the set can be tuned in one place (GRAPH_LINK_NAME_STOPWORDS).
_NAME_STOPWORDS = GRAPH_LINK_NAME_STOPWORDS


def _tokenize(text: str) -> List[str]:
    # Splits on non-alphanumeric (including underscores) so that doc text and
    # schema names tokenize consistently — 'force_majeure' → ['force','majeure'].
    return _TOKEN_RE.findall((text or "").lower())


def _split_name_tokens(name: str) -> List[str]:
    """Splits a column/table name on underscores + camelCase, lowercased."""
    if not name:
        return []
    parts = re.split(r"[_\s]+", name)
    out = []
    for p in parts:
        # split camelCase
        for sub in re.findall(r"[A-Za-z][a-z0-9]+|[A-Z]+(?![a-z])|\d+", p) or [p]:
            out.append(sub.lower())
    return [t for t in out if t]


# =============================================================================
# Load column/table nodes from graph_nodes
# =============================================================================

def _load_graph_columns() -> List[GraphNode]:
    """Loads all column + table nodes across all sources from graph_nodes."""
    if not INTERNAL_DB_AVAILABLE:
        return [
            graph_persist._dict_to_node(n)
            for n in graph_persist._IN_MEMORY_NODES
            if n["node_type"] in ("column", "table")
        ]

    try:
        conn = get_internal_connection()
    except Exception:
        return []
    try:
        cur = conn.cursor(cursor_factory=DICT_CURSOR)
        cur.execute(f"""
            SELECT node_id, node_type, source_id, ref_id, table_id, name,
                   table_name, semantic_type, data_type, is_pk, is_fk, attrs
            FROM {GRAPH_NODES_TABLE}
            WHERE node_type IN ('column', 'table');
        """)
        rows = cur.fetchall()
        try: cur.close()
        except Exception: pass
    finally:
        release_internal_connection(conn)
    return [graph_persist._row_to_node(r) for r in rows]


def _load_column_values() -> Dict[str, List[str]]:
    """
    Loads sampled normalised values per col_id from column_values.
    Returns {col_id: [value_norm, ...]}. Empty dict on failure.
    """
    if not INTERNAL_DB_AVAILABLE:
        return {}
    try:
        conn = get_internal_connection()
    except Exception:
        return {}
    try:
        cur = conn.cursor(cursor_factory=DICT_CURSOR)
        cur.execute(f"SELECT col_id, value_norm FROM {COLUMN_VALUES_TABLE_NAME};")
        rows = cur.fetchall()
        try: cur.close()
        except Exception: pass
    except Exception:
        release_internal_connection(conn)
        return {}
    finally:
        release_internal_connection(conn)

    out: Dict[str, List[str]] = {}
    for r in rows:
        out.setdefault(r["col_id"], []).append((r["value_norm"] or "").lower())
    return out


# =============================================================================
# Scoped delete
# =============================================================================

def _scoped_delete(source_id: str, verbose: bool = False) -> None:
    if not INTERNAL_DB_AVAILABLE:
        graph_persist._IN_MEMORY_NODES = [
            n for n in graph_persist._IN_MEMORY_NODES
            if not (n["source_id"] == source_id and n["node_type"] == "chunk")
        ]
        graph_persist._IN_MEMORY_EDGES = [
            e for e in graph_persist._IN_MEMORY_EDGES
            if not (e["source_id"] == source_id
                    and e["edge_type"] in ("mentions", "name_match", "about"))
        ]
        return

    try:
        conn = get_internal_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    graph_persist._create_graph_tables(cur)
                    cur.execute(
                        f"DELETE FROM {GRAPH_EDGES_TABLE} WHERE source_id = %s "
                        f"AND edge_type IN ('mentions','name_match','about');",
                        (source_id,),
                    )
                    cur.execute(
                        f"DELETE FROM {GRAPH_NODES_TABLE} WHERE source_id = %s "
                        f"AND node_type = 'chunk';",
                        (source_id,),
                    )
        finally:
            release_internal_connection(conn)
    except Exception as e:
        if verbose:
            print(f"  ⚠ [chunk_linker] scoped delete failed ({e})")


# =============================================================================
# Main entry point
# =============================================================================

def link_chunks_to_graph(
    chunks,
    chunk_embeddings,
    source_id: str,
    verbose: bool = False,
) -> ChunkLinkResult:
    """
    Links document chunks to graph column/table nodes.

    Parameters
    ----------
    chunks           : List[DocumentChunk]
    chunk_embeddings : np.ndarray (n_chunks, 384) L2-normalized, or None
    source_id        : the document source id
    verbose          : print progress
    """
    t0 = time.time()
    backend = "in_memory_fallback" if not INTERNAL_DB_AVAILABLE else "postgres"

    if not chunks:
        return ChunkLinkResult(
            chunk_nodes_written = 0,
            link_edges_written  = 0,
            source_id           = source_id,
            backend             = "no_chunks",
            duration_sec        = round(time.time() - t0, 4),
        )

    # Scoped delete (idempotent)
    _scoped_delete(source_id, verbose=verbose)

    # ------------------------------------------------------------------
    # 1. Create chunk nodes
    # ------------------------------------------------------------------
    chunk_nodes: List[GraphNode] = []
    for c in chunks:
        chunk_nodes.append(GraphNode(
            node_id   = chunk_node_id(c.chunk_id),
            node_type = "chunk",
            source_id = source_id,
            ref_id    = c.chunk_id,
            name      = c.doc_name,
            attrs     = {
                "doc_id":      c.doc_id,
                "page_num":    c.page_num,
                "chunk_index": c.chunk_index,
            },
        ))
    chunk_nodes_written = graph_persist.upsert_nodes(chunk_nodes, verbose=verbose)

    # ------------------------------------------------------------------
    # Load target column/table nodes
    # ------------------------------------------------------------------
    col_nodes = _load_graph_columns()
    column_nodes = [n for n in col_nodes if n.node_type == "column"]

    # Pre-tokenise each chunk
    chunk_tokens: List[set] = [set(_tokenize(c.text)) for c in chunks]
    chunk_token_counts: List[int] = [max(1, len(_tokenize(c.text))) for c in chunks]

    # Candidate edges keyed per chunk index → list of (target_node_id, weight, edge_type, attrs, evidence)
    per_chunk_candidates: List[List[tuple]] = [[] for _ in chunks]

    stat_value_overlap = 0
    stat_name_match    = 0
    stat_embedding     = 0

    # ------------------------------------------------------------------
    # Signal 1 — Value-overlap (mentions)
    # ------------------------------------------------------------------
    col_values: Dict[str, List[str]] = {}
    try:
        col_values = _load_column_values()
    except Exception:
        col_values = {}

    if col_values:
        for n in column_nodes:
            vals = col_values.get(n.ref_id)
            if not vals:
                continue
            val_token_set = set()
            for v in vals:
                val_token_set.update(_tokenize(v))
            if not val_token_set:
                continue
            for ci, ctoks in enumerate(chunk_tokens):
                hits = len(val_token_set & ctoks)
                if hits < 2:
                    # require at least 2 distinct value tokens to reduce false positives
                    continue
                # overlap = fraction of this column's value tokens found in the chunk
                # (not fraction of chunk tokens — chunks are too large for that denominator)
                overlap_score = hits / len(val_token_set)
                if overlap_score >= GRAPH_LINK_VALUE_OVERLAP_MIN:
                    weight = GRAPH_EDGE_WEIGHTS["mentions"] * min(1.0, overlap_score)
                    per_chunk_candidates[ci].append((
                        n.node_id, weight, "mentions",
                        {"overlap_score": round(overlap_score, 4), "signal": "value_overlap"},
                        f"value_overlap:{hits}",
                    ))
                    stat_value_overlap += 1

    # ------------------------------------------------------------------
    # Signal 2 — Name-match (mentions)
    # Stopwords filtered; ranked by token specificity (rarer token = stronger signal).
    # ------------------------------------------------------------------
    # Pre-compute token → how many column nodes it matches (for specificity ranking)
    token_column_freq: Dict[str, int] = {}
    for n in col_nodes:
        toks = {t for t in _split_name_tokens(n.name) if len(t) >= GRAPH_LINK_NAME_MIN_TOKEN_LEN
                and t not in _NAME_STOPWORDS}
        if n.table_name:
            toks.update(t for t in _split_name_tokens(n.table_name)
                        if len(t) >= GRAPH_LINK_NAME_MIN_TOKEN_LEN and t not in _NAME_STOPWORDS)
        for t in toks:
            token_column_freq[t] = token_column_freq.get(t, 0) + 1

    for n in col_nodes:
        name_tokens = {t for t in _split_name_tokens(n.name)
                       if len(t) >= GRAPH_LINK_NAME_MIN_TOKEN_LEN and t not in _NAME_STOPWORDS}
        if n.table_name:
            name_tokens.update(t for t in _split_name_tokens(n.table_name)
                                if len(t) >= GRAPH_LINK_NAME_MIN_TOKEN_LEN
                                and t not in _NAME_STOPWORDS)
        if not name_tokens:
            continue
        for ci, ctoks in enumerate(chunk_tokens):
            matched = name_tokens & ctoks
            if not matched:
                continue
            # pick the rarest (most specific) matching token; fewer col matches = higher weight
            tok = min(matched, key=lambda t: token_column_freq.get(t, 1))
            specificity = 1.0 / max(1, token_column_freq.get(tok, 1))
            # Use "name_match" edge type (not "mentions") so name-match edges are
            # deduplicated separately from value-overlap edges.  Previously both
            # used "mentions", so a high-specificity name-match (weight→1.0) would
            # silently overwrite a lower value-overlap edge on the same target node,
            # causing name-match to dominate retrieval ranking (B7 fix).
            weight = GRAPH_EDGE_WEIGHTS.get("name_match", GRAPH_EDGE_WEIGHTS["mentions"]) * min(1.0, specificity * 10)
            per_chunk_candidates[ci].append((
                n.node_id, weight, "name_match",
                {"signal": "name_match", "token": tok,
                 "col_freq": token_column_freq.get(tok, 1)},
                f"name_match:{tok}",
            ))
            stat_name_match += 1

    # ------------------------------------------------------------------
    # Signal 3 — Embedding similarity (about)
    # Only when Phase 3 graph embedding is NOT handling it.
    # ------------------------------------------------------------------
    if not GRAPH_EMBED_ENABLED and chunk_embeddings is not None and column_nodes:
        try:
            # WP3: BGE-M3 dense (1024-dim) for the column sentences — same space as the
            # chunk embeddings passed in. NOTE: GRAPH_LINK_EMBED_SIM_MIN was tuned for
            # MiniLM's cosine distribution; M3 cosines run tighter/higher, so re-validate
            # that threshold against a histogram on one ingested doc source.
            from ingestion import m3_encoder
            col_sentences = [
                GRAPH_COLUMN_SENTENCE_TEMPLATE.format(
                    col_name      = n.name,
                    table_name    = n.table_name or "",
                    semantic_type = n.semantic_type or "",
                )
                for n in column_nodes
            ]
            col_embs = m3_encoder.encode_dense(col_sentences)

            chunk_emb_mat = np.asarray(chunk_embeddings, dtype=np.float32)
            # cosine = chunk_emb @ col_embs.T (both normalized)
            sims = chunk_emb_mat @ col_embs.T   # (n_chunks, n_cols)
            for ci in range(sims.shape[0]):
                row = sims[ci]
                for cj in range(row.shape[0]):
                    cos = float(row[cj])
                    if cos >= GRAPH_LINK_EMBED_SIM_MIN:
                        weight = GRAPH_EDGE_WEIGHTS["about"] * cos
                        per_chunk_candidates[ci].append((
                            column_nodes[cj].node_id, weight, "about",
                            {"signal": "embedding", "cosine": round(cos, 4)},
                            f"embedding:{round(cos, 4)}",
                        ))
                        stat_embedding += 1
        except Exception as e:
            if verbose:
                print(f"  ⚠ [chunk_linker] embedding signal unavailable ({e}) — name-match only")

    # ------------------------------------------------------------------
    # Per-chunk: dedup by (target, edge_type) keeping max weight, cap top-N
    # ------------------------------------------------------------------
    edges: List[GraphEdge] = []
    for ci, candidates in enumerate(per_chunk_candidates):
        if not candidates:
            continue
        best: Dict[tuple, tuple] = {}
        for (tgt, w, et, attrs, ev) in candidates:
            key = (tgt, et)
            if key not in best or w > best[key][1]:
                best[key] = (tgt, w, et, attrs, ev)
        ranked = sorted(best.values(), key=lambda x: x[1], reverse=True)
        ranked = ranked[:GRAPH_LINK_MAX_EDGES_PER_CHUNK]
        src_id = chunk_node_id(chunks[ci].chunk_id)
        for (tgt, w, et, attrs, ev) in ranked:
            edges.append(GraphEdge(
                edge_id     = str(uuid4()),
                src_node_id = src_id,
                dst_node_id = tgt,
                edge_type   = et,
                weight      = round(w, 6),
                source_id   = source_id,
                evidence    = ev,
                attrs       = attrs,
            ))

    link_edges_written = graph_persist.upsert_edges(edges, verbose=verbose)

    duration = round(time.time() - t0, 4)
    if verbose:
        print(f"[ChunkLinker] {chunk_nodes_written} chunk nodes, "
              f"{link_edges_written} link edges "
              f"(value={stat_value_overlap}, name={stat_name_match}, "
              f"embed={stat_embedding}), backend={backend}, {duration}s")

    return ChunkLinkResult(
        chunk_nodes_written = chunk_nodes_written,
        link_edges_written  = link_edges_written,
        source_id           = source_id,
        backend             = backend,
        duration_sec        = duration,
        stats               = {
            "value_overlap": stat_value_overlap,
            "name_match":    stat_name_match,
            "embedding":     stat_embedding,
        },
    )
