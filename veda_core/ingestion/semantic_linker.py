# =============================================================================
# ingestion/semantic_linker.py
# VEDA — Semantic bridge: unstructured chunks → structured semantic layer
# (docs/SEMANTIC_ENTITY_BRIDGE.md, Tier A — the additive semantic lane)
#
# The exact bridge (ingestion/entity_linker.py) links a chunk to a column only
# when a stored value appears LITERALLY in the chunk text. That is precise but
# under-links at scale (synonyms / abbreviations / paraphrases are missed), and
# under-linking is the dominant failure mode for a large unstructured corpus.
#
# This module adds the SEMANTIC lane, reusing the structured semantic layer we
# already build for structured data: every column/table node is embedded into
# `graph_node_embeddings` (M3 1024-d, enriched text incl. sampled values —
# ingestion/graph_embedder.py). We match each chunk's M3 vector against those
# column vectors by cosine similarity and emit `semantic_about` (chunk → column)
# edges above a tuned threshold.
#
# SAFETY BY CONSTRUCTION:
#   - `semantic_about` is a chunk→column edge. A SQL join is column↔column, so a
#     semantic edge can NEVER authorize a federated join (the graph-guard only
#     ever sees column↔column `value_of` / HIGH `cross_source_fk`). The fuzzy lane
#     can surface EVIDENCE; it can never fabricate a JOIN.
#   - Weight is kept strictly below the exact `value_of` bridge, so exact links
#     always dominate PPR ranking; semantic links only widen recall.
#   - Threshold + top-K + optional top1/top2 margin quarantine "near many columns"
#     noise; sensitive/metadata columns never participate (same guard as exact).
#
# Additive: entity_linker calls this after its exact/pattern detectors. Absent a
# structured semantic layer (no column embeddings yet) it is a graceful no-op.
# =============================================================================

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

from config import (
    SEMANTIC_BRIDGE_ENABLED,
    SEMANTIC_BRIDGE_MIN_SIM,
    SEMANTIC_BRIDGE_TOPK,
    SEMANTIC_BRIDGE_MARGIN,
    SEMANTIC_BRIDGE_MAX_COLS,
    SEMANTIC_BRIDGE_EDGE_TYPE,
    SEMANTIC_VALUE_BRIDGE_ENABLED,
    SEMANTIC_VALUE_MIN_SIM,
    SEMANTIC_VALUE_TOPK,
    SEMANTIC_VALUE_MAX_SPANS_PER_CHUNK,
    GRAPH_EDGE_WEIGHTS,
    GRAPH_NODE_EMB_TABLE,
    DOC_CHUNKS_TABLE_NAME,
)
from ingestion.db_abstraction import (
    INTERNAL_DB_AVAILABLE,
    get_internal_connection,
    release_internal_connection,
    DICT_CURSOR,
)
from ingestion.graph_persist import GraphEdge, GraphNode, chunk_node_id
from utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- vecs
def _parse_vec(raw) -> Optional[np.ndarray]:
    """pgvector round-trips as either a python list (registered adapter) or the text
    form '[0.1,0.2,...]'. Mirror graph_embedder's tolerant parse."""
    if raw is None:
        return None
    if hasattr(raw, "tolist") or isinstance(raw, (list, tuple)):
        return np.asarray(raw, dtype=np.float32)
    try:
        return np.asarray(
            [float(x) for x in str(raw).strip("[]").split(",") if x.strip()],
            dtype=np.float32,
        )
    except Exception:
        return None


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (mat / norms).astype(np.float32)


# --------------------------------------------------------------------------- load
def _load_column_embeddings(exclude_source_id: Optional[str]
                            ) -> Tuple[List[str], np.ndarray]:
    """Column node vectors from the structured semantic layer (graph_node_embeddings).
    Returns ([col_node_id...], matrix[n, dim] L2-normalized). The doc source itself
    is excluded (a document has no structured columns to bridge to). Capped at
    SEMANTIC_BRIDGE_MAX_COLS as a memory backstop for very wide tenants."""
    if not INTERNAL_DB_AVAILABLE:
        return [], np.zeros((0, 0), dtype=np.float32)
    try:
        conn = get_internal_connection()
    except Exception:
        return [], np.zeros((0, 0), dtype=np.float32)
    try:
        cur = conn.cursor(cursor_factory=DICT_CURSOR)
        params: list = []
        where = ["node_type = 'column'"]
        if exclude_source_id is not None:
            where.append("source_id <> %s")
            params.append(str(exclude_source_id))
        params.append(int(SEMANTIC_BRIDGE_MAX_COLS))
        cur.execute(
            f"SELECT node_id, embedding FROM {GRAPH_NODE_EMB_TABLE} "
            f"WHERE {' AND '.join(where)} LIMIT %s",
            params,
        )
        rows = cur.fetchall()
        try: cur.close()
        except Exception: pass
    except Exception as e:
        logger.warning("semantic_linker: column embedding load failed (%s)", e)
        return [], np.zeros((0, 0), dtype=np.float32)
    finally:
        release_internal_connection(conn)

    ids: List[str] = []
    vecs: List[np.ndarray] = []
    for r in rows:
        v = _parse_vec(r["embedding"])
        if v is None or v.size == 0:
            continue
        ids.append(r["node_id"])
        vecs.append(v)
    if not vecs:
        return [], np.zeros((0, 0), dtype=np.float32)
    return ids, _l2_normalize(np.vstack(vecs))


def _load_chunk_embeddings(chunk_ids: List[str], source_id: str) -> Dict[str, np.ndarray]:
    """Reuse the vectors chunk_embedder just wrote to doc_chunks (no re-encoding).
    Returns {chunk_id: vec}. Missing ids are filled by an on-the-fly M3 encode in
    build_semantic_edges."""
    if not INTERNAL_DB_AVAILABLE or not chunk_ids:
        return {}
    try:
        conn = get_internal_connection()
    except Exception:
        return {}
    try:
        cur = conn.cursor(cursor_factory=DICT_CURSOR)
        cur.execute(
            f"SELECT chunk_id, embedding FROM {DOC_CHUNKS_TABLE_NAME} WHERE source_id = %s",
            (str(source_id),),
        )
        rows = cur.fetchall()
        try: cur.close()
        except Exception: pass
    except Exception as e:
        logger.warning("semantic_linker: chunk embedding load failed (%s)", e)
        return {}
    finally:
        release_internal_connection(conn)
    out: Dict[str, np.ndarray] = {}
    for r in rows:
        v = _parse_vec(r["embedding"])
        if v is not None and v.size:
            out[r["chunk_id"]] = v
    return out


# ------------------------------------------------------------------- pure matcher
def match_chunks_to_columns(
    chunk_mat: np.ndarray,
    chunk_node_ids: List[str],
    col_mat: np.ndarray,
    col_node_ids: List[str],
    min_sim: float = SEMANTIC_BRIDGE_MIN_SIM,
    top_k: int = SEMANTIC_BRIDGE_TOPK,
    margin: float = SEMANTIC_BRIDGE_MARGIN,
) -> List[Tuple[str, str, float]]:
    """Cosine match every chunk against every column, keep per-chunk top-K columns
    with cosine ≥ ``min_sim`` (and, when ``margin`` > 0, only if the top match clears
    the 2nd-best by ``margin`` — drops ambiguous "near everything" chunks). Pure /
    DB-free so it is unit-testable with synthetic vectors. Both matrices must be
    L2-normalized (the DB loaders normalize). Returns [(chunk_node_id, col_node_id,
    cosine)]."""
    out: List[Tuple[str, str, float]] = []
    if chunk_mat.size == 0 or col_mat.size == 0:
        return out
    if chunk_mat.shape[1] != col_mat.shape[1]:
        return out
    sims = chunk_mat @ col_mat.T                     # (n_chunks, n_cols), cosine
    n_cols = col_mat.shape[0]
    k = max(1, min(int(top_k), n_cols))
    for ci in range(sims.shape[0]):
        row = sims[ci]
        # top-k column indices for this chunk, highest first
        top_idx = np.argsort(row)[::-1][:k]
        if margin > 0 and n_cols >= 2:
            # ambiguity guard: skip if the best barely beats the runner-up
            best_two = np.sort(row)[::-1][:2]
            if float(best_two[0] - best_two[1]) < margin:
                continue
        for cj in top_idx:
            cos = float(row[cj])
            if cos < min_sim:
                continue
            out.append((chunk_node_ids[ci], col_node_ids[int(cj)], cos))
    return out


# ------------------------------------------------------------------- main entry
def build_semantic_edges(chunks, source_id: str, tenant: str = "default",
                         verbose: bool = False) -> List[GraphEdge]:
    """Return `semantic_about` (chunk → column) GraphEdges bridging the given doc
    source's chunks to the structured semantic layer by M3 cosine similarity.

    Called by entity_linker.link_entities AFTER its exact/pattern detectors and
    AFTER the chunk nodes are created (this only emits edges — the chunk nodes and
    scoped-delete idempotency are owned by entity_linker, which lists
    ``semantic_about`` among its cleaned edge types). Graceful no-op when disabled,
    when the structured semantic layer isn't embedded yet, or when M3 is
    unavailable."""
    if not SEMANTIC_BRIDGE_ENABLED or not chunks:
        return []
    t0 = time.time()

    col_ids, col_mat = _load_column_embeddings(exclude_source_id=source_id)
    if not col_ids:
        if verbose:
            logger.info("semantic_linker: no structured column embeddings — skipped")
        return []

    # Chunk vectors: reuse doc_chunks (just embedded), encode any misses once.
    ref_ids = [getattr(c, "chunk_id", "") for c in chunks]
    cached = _load_chunk_embeddings(ref_ids, source_id)
    missing = [c for c in chunks if getattr(c, "chunk_id", "") not in cached]
    if missing:
        try:
            from ingestion import m3_encoder
            enc = m3_encoder.encode_dense([getattr(c, "text", "") or "" for c in missing])
            for c, v in zip(missing, enc):
                cached[getattr(c, "chunk_id", "")] = np.asarray(v, dtype=np.float32)
        except Exception as e:
            logger.warning("semantic_linker: chunk encode fallback failed (%s)", e)

    chunk_node_ids: List[str] = []
    chunk_vecs: List[np.ndarray] = []
    for c in chunks:
        cid = getattr(c, "chunk_id", "")
        v = cached.get(cid)
        if v is None or getattr(v, "size", 0) == 0:
            continue
        chunk_node_ids.append(chunk_node_id(cid))
        chunk_vecs.append(v)
    if not chunk_vecs:
        return []
    chunk_mat = _l2_normalize(np.vstack(chunk_vecs))

    matches = match_chunks_to_columns(chunk_mat, chunk_node_ids, col_mat, col_ids)

    base_w = GRAPH_EDGE_WEIGHTS.get(SEMANTIC_BRIDGE_EDGE_TYPE, 0.9)
    edges: List[GraphEdge] = []
    for (cnode, col_node, cos) in matches:
        edges.append(GraphEdge(
            str(uuid4()), cnode, col_node, SEMANTIC_BRIDGE_EDGE_TYPE,
            round(base_w * cos, 6), source_id,
            evidence=f"semantic:{round(cos, 4)}",
            attrs={"signal": "semantic", "cosine": round(cos, 4)},
        ))
    if verbose:
        logger.info("semantic_linker: %d semantic_about edges over %d chunks × %d cols (%.2fs)",
                    len(edges), len(chunk_node_ids), len(col_ids), time.time() - t0)
    return edges


# =============================================================================
# Tier B — value-level bridge (docs/SEMANTIC_ENTITY_BRIDGE.md §3.1)
# A chunk SPAN → the actual DB VALUE it paraphrases ("ACME Corporation" ↔ stored
# "ACME-CORP"). Structured side is ingestion/value_embedder.py (entity_value_embeddings);
# here we extract candidate spans, embed them, ANN-match against that index, and emit
# an entity node + mentions_entity (chunk→entity) + semantic_value_of (entity→column).
# =============================================================================

# Candidate span detectors (deterministic — no training):
#   - Capitalized runs (1–5 tokens): proper nouns / names / places / orgs.
#   - Quoted spans: "..." / '...' — often exact entity references.
# These stay tight on purpose (proper-noun-ish), so we embed few spans per chunk and
# keep precision high; the cosine floor (SEMANTIC_VALUE_MIN_SIM) does the rest.
_CAP_RUN_RE = re.compile(r"\b([A-Z][A-Za-z0-9&.\-/]*(?:\s+[A-Z][A-Za-z0-9&.\-/]*){0,4})\b")
_QUOTED_RE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{3,60})[\"'“”‘’]")
_SPAN_MIN_LEN = 4


def extract_candidate_spans(text: str, max_spans: int = SEMANTIC_VALUE_MAX_SPANS_PER_CHUNK
                            ) -> List[str]:
    """Deterministic candidate entity spans from a chunk (pure / unit-testable):
    capitalized runs + quoted phrases, deduped (case-insensitive), length-filtered,
    generic single words dropped, capped. Order-stable (first occurrence wins)."""
    if not text:
        return []
    from ingestion.entity_linker import _GENERIC_WORDS
    seen: set = set()
    out: List[str] = []
    for m in list(_QUOTED_RE.finditer(text)) + list(_CAP_RUN_RE.finditer(text)):
        span = (m.group(1) or "").strip()
        if len(span) < _SPAN_MIN_LEN:
            continue
        key = span.casefold()
        if key in seen:
            continue
        # a lone generic capitalized word ("Report", "Total") is not an entity
        if " " not in span and key in _GENERIC_WORDS:
            continue
        seen.add(key)
        out.append(span)
        if len(out) >= max_spans:
            break
    return out


def match_spans_to_values(
    span_mat: np.ndarray,
    span_owner_chunk: List[str],
    val_mat: np.ndarray,
    col_node_ids: List[str],
    min_sim: float = SEMANTIC_VALUE_MIN_SIM,
    top_k: int = SEMANTIC_VALUE_TOPK,
) -> List[Tuple[int, int, float]]:
    """Cosine-match each candidate span against every stored value vector, keep per-span
    top-K value hits ≥ ``min_sim``. Pure / DB-free. ``span_owner_chunk`` is unused here
    (kept for caller symmetry). Returns [(span_idx, value_idx, cosine)]. Both matrices
    must be L2-normalized."""
    out: List[Tuple[int, int, float]] = []
    if span_mat.size == 0 or val_mat.size == 0 or span_mat.shape[1] != val_mat.shape[1]:
        return out
    sims = span_mat @ val_mat.T
    n_vals = val_mat.shape[0]
    k = max(1, min(int(top_k), n_vals))
    for si in range(sims.shape[0]):
        row = sims[si]
        for vj in np.argsort(row)[::-1][:k]:
            cos = float(row[int(vj)])
            if cos < min_sim:
                continue
            out.append((si, int(vj), cos))
    return out


def build_value_bridge(chunks, source_id: str, tenant: str = "default",
                       verbose: bool = False) -> Tuple[List[GraphNode], List[GraphEdge]]:
    """Value-level semantic bridge: returns (entity_nodes, edges) linking chunk spans to
    the structured VALUES they paraphrase — entity node + mentions_entity (chunk→entity)
    + semantic_value_of (entity→column). Graceful no-op when disabled / no value index /
    M3 unavailable. Entity ids reuse entity_linker.entity_node_id so semantic entities
    unify with the exact bridge's nodes."""
    if not SEMANTIC_VALUE_BRIDGE_ENABLED or not chunks:
        return [], []
    t0 = time.time()

    from ingestion.value_embedder import load_value_embeddings
    col_nodes, value_norms, displays, value_classes, val_mat = \
        load_value_embeddings(exclude_source_id=source_id)
    if not col_nodes:
        if verbose:
            logger.info("semantic_linker(value): no value index — skipped")
        return [], []

    # Extract spans per chunk, remember which chunk each span came from.
    span_texts: List[str] = []
    span_chunk_node: List[str] = []
    for c in chunks:
        spans = extract_candidate_spans(getattr(c, "text", "") or "")
        cn = chunk_node_id(getattr(c, "chunk_id", ""))
        for s in spans:
            span_texts.append(s)
            span_chunk_node.append(cn)
    if not span_texts:
        return [], []

    try:
        from ingestion import m3_encoder
        span_mat = _l2_normalize(np.asarray(m3_encoder.encode_dense(span_texts), dtype=np.float32))
    except Exception as e:
        logger.warning("semantic_linker(value): span encode failed (%s)", e)
        return [], []

    matches = match_spans_to_values(span_mat, span_chunk_node, val_mat, col_nodes)

    from ingestion.entity_linker import entity_node_id
    me_w = GRAPH_EDGE_WEIGHTS.get("mentions_entity", 1.2)
    vo_w = GRAPH_EDGE_WEIGHTS.get("semantic_value_of", 1.1)
    _cls_map = {"category": "term", "text": "name", "id": "id", "numeric": "id"}

    entity_nodes: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []
    seen_me: set = set()   # (chunk_node, entity_id)
    seen_vo: set = set()   # (entity_id, col_node)
    for (si, vj, cos) in matches:
        vnorm = value_norms[vj]
        cls = _cls_map.get((value_classes[vj] or "").lower(), "name")
        eid = entity_node_id(cls, vnorm)
        cnode = span_chunk_node[si]
        col_node = col_nodes[vj]
        if eid not in entity_nodes:
            entity_nodes[eid] = GraphNode(
                node_id=eid, node_type="entity", source_id=source_id, ref_id=vnorm,
                name=displays[vj], semantic_type=cls,
                attrs={"class": cls, "display": displays[vj], "signal": "semantic_value"})
        if (cnode, eid) not in seen_me:
            seen_me.add((cnode, eid))
            edges.append(GraphEdge(str(uuid4()), cnode, eid, "mentions_entity", me_w,
                                   source_id, evidence=f"semantic_value:{round(cos,4)}",
                                   attrs={"class": cls, "signal": "semantic",
                                          "span": span_texts[si][:80]}))
        if (eid, col_node) not in seen_vo:
            seen_vo.add((eid, col_node))
            edges.append(GraphEdge(str(uuid4()), eid, col_node, "semantic_value_of",
                                   round(vo_w * cos, 6), source_id,
                                   evidence=f"semantic_value_of:{round(cos,4)}",
                                   attrs={"cosine": round(cos, 4), "signal": "semantic"}))
    if verbose:
        logger.info("semantic_linker(value): %d entities, %d edges over %d spans × %d values (%.2fs)",
                    len(entity_nodes), len(edges), len(span_texts), len(col_nodes), time.time() - t0)
    return list(entity_nodes.values()), edges
