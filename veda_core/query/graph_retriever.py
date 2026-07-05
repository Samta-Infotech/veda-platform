# =============================================================================
# query/graph_retriever.py
# VEDA — Unified Data Graph: Phase 4 (Query-time Seed + Expand Retrieval)
#
# Responsibility:
#   - Embeds the user query, seeds top-K graph nodes via cosine ANN
#   - BFS-expands the subgraph over edges (weighted, hop-decayed scoring)
#   - Materializes columns / chunks / tables from the visited subgraph
#   - Adapts column subgraph nodes back into RetrievalResult for L3/L4 reuse
#
# Only invoked when UNIFIED_GRAPH_ENABLED + GRAPH_RETRIEVAL_ENABLED + GRAPH_EMBED_ENABLED.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from config import (
    GRAPH_SEED_TOP_K,
    GRAPH_EXPAND_HOPS,
    GRAPH_EXPAND_MAX_NODES,
    GRAPH_HOP_DECAY,
    GRAPH_EDGE_WEIGHTS,
    GRAPH_HUB_DEGREE_CAP,
    GRAPH_SIBLING_SCORE_FACTOR,
    GRAPH_SIBLING_MAX_PER_TABLE,
    GRAPH_SINGLE_TABLE_SIM,
    GRAPH_SINGLE_TABLE_TOPN,
    GRAPH_SINGLE_TABLE_GAP,
    GRAPH_SEED_SIM_FLOOR,
    GRAPH_MAX_COLS_TO_L3,
    GRAPH_MAX_CHUNKS,
)

_MAX_EDGE_WEIGHT = max(GRAPH_EDGE_WEIGHTS.values())  # 3.0 — used to normalise per spec


# =============================================================================
# Output data structures
# =============================================================================

@dataclass
class SubgraphNode:
    node_id:       str
    node_type:     str
    score:         float
    hop:           int
    name:          str = ""
    table_id:      Optional[str] = None
    table_name:    Optional[str] = None
    semantic_type: Optional[str] = None
    text:          Optional[str] = None


@dataclass
class GraphRetrievalResult:
    columns:       list
    chunks:        list
    tables:        list
    edges_used:    list
    duration_ms:   float
    stats:         dict = field(default_factory=dict)
    short_circuited: bool = False


# =============================================================================
# Chunk text fetch
# =============================================================================

def _fetch_chunk_texts(ref_ids: List[str]) -> Dict[str, dict]:
    """Returns {chunk_id: {text, doc_name, page_num}} for the given chunk ref_ids."""
    if not ref_ids:
        return {}
    from ingestion.db_abstraction import (
        INTERNAL_DB_AVAILABLE,
        get_internal_connection,
        release_internal_connection,
        DICT_CURSOR,
    )
    from config import DOC_CHUNKS_TABLE_NAME

    if not INTERNAL_DB_AVAILABLE:
        from ingestion.chunk_embedder import _IN_MEMORY_CHUNKS
        wanted = set(ref_ids)
        out = {}
        for r in _IN_MEMORY_CHUNKS:
            if r["chunk_id"] in wanted:
                out[r["chunk_id"]] = {
                    "text":     r["text"],
                    "doc_name": r["doc_name"],
                    "page_num": r["page_num"],
                }
        return out

    try:
        conn = get_internal_connection()
    except Exception:
        return {}
    try:
        cur = conn.cursor(cursor_factory=DICT_CURSOR)
        placeholders = ",".join(["%s"] * len(ref_ids))
        cur.execute(f"""
            SELECT chunk_id, text, doc_name, page_num
            FROM {DOC_CHUNKS_TABLE_NAME}
            WHERE chunk_id IN ({placeholders});
        """, list(ref_ids))
        rows = cur.fetchall()
        try: cur.close()
        except Exception: pass
    finally:
        release_internal_connection(conn)
    return {
        r["chunk_id"]: {
            "text": r["text"], "doc_name": r["doc_name"], "page_num": r["page_num"],
        }
        for r in rows
    }


# =============================================================================
# Main entry point
# =============================================================================

def run_graph_retrieval(
    query: str,
    source_ids: Optional[List[str]] = None,
    seed_top_k: Optional[int] = None,
    hops: Optional[int] = None,
    verbose: bool = False,
) -> GraphRetrievalResult:
    """
    Seed-and-expand retrieval over the unified graph.
    Returns columns (as RetrievalResult), chunks (as ChunkRetrievalResult),
    tables (SubgraphNode), and provenance edges.
    """
    t0 = time.time()
    seed_top_k = seed_top_k or GRAPH_SEED_TOP_K
    hops = hops or GRAPH_EXPAND_HOPS

    from ingestion.graph_embedder import embed_text_bge, retrieve_graph_seeds
    from ingestion.graph_persist import get_neighbors, get_nodes

    qvec = embed_text_bge(query)
    seeds = retrieve_graph_seeds(qvec, top_k=seed_top_k, source_ids=source_ids)

    # ------------------------------------------------------------------
    # BFS expansion
    # ------------------------------------------------------------------
    visited: Dict[str, SubgraphNode] = {}
    edges_used: list = []

    for (node_id, node_type, sim) in seeds:
        if node_id not in visited:
            visited[node_id] = SubgraphNode(
                node_id   = node_id,
                node_type = node_type,
                score     = float(sim),
                hop       = 0,
            )

    # ------------------------------------------------------------------
    # Fix C — single-table short-circuit
    # Triggers on seed dominance (gap ≥ GAP) OR all strong seeds agree on
    # one table. Prior unanimity check required all top-3 seeds to share
    # one table, which failed when tail seeds were low-similarity noise.
    # ------------------------------------------------------------------
    short_circuited = False
    sc_table_id = None
    if seeds and seeds[0][2] >= GRAPH_SINGLE_TABLE_SIM:
        strong = [(nid, sim) for nid, _, sim in seeds
                  if nid.startswith("col:") and sim >= GRAPH_SEED_SIM_FLOOR]
        strong_meta = get_nodes([nid for nid, _ in strong]) if strong else []
        strong_tids = {n.table_id for n in strong_meta if n.table_id}

        gap = seeds[0][2] - (seeds[1][2] if len(seeds) > 1 else 0.0)
        dominant   = gap >= GRAPH_SINGLE_TABLE_GAP
        one_strong = len(strong_tids) == 1

        if dominant or one_strong:
            short_circuited = True
            s1_meta = get_nodes([seeds[0][0]])
            sc_table_id = s1_meta[0].table_id if s1_meta else None
            if verbose:
                print(f"[GraphRetriever] single-table short-circuit "
                      f"(dominant={dominant}, one_strong={one_strong}) "
                      f"table_id={sc_table_id} sim={seeds[0][2]:.4f} gap={gap:.3f}")

    # ------------------------------------------------------------------
    # Fix A — BFS with has_column restored + hub degree cap
    # ------------------------------------------------------------------
    if not short_circuited:
        from ingestion.graph_persist import get_node_degrees
        frontier = list(visited.keys())
        stop = False
        for hop in range(1, hops + 1):
            if not frontier or stop:
                break
            degrees = get_node_degrees(frontier)
            edges = get_neighbors(
                frontier,
                edge_types=["has_column", "fk_to", "discovered_fk", "mentions", "about"],
                direction="both",
            )
            next_frontier: List[str] = []
            for e in edges:
                if e.src_node_id in visited:
                    parent_id, neighbor_id = e.src_node_id, e.dst_node_id
                elif e.dst_node_id in visited:
                    parent_id, neighbor_id = e.dst_node_id, e.src_node_id
                else:
                    continue

                # Hub guard: don't expand THROUGH high-degree nodes
                if degrees.get(parent_id, 0) > GRAPH_HUB_DEGREE_CAP:
                    continue

                edges_used.append({
                    "src":       e.src_node_id,
                    "dst":       e.dst_node_id,
                    "edge_type": e.edge_type,
                    "weight":    e.weight,
                    "evidence":  e.evidence,
                })

                if neighbor_id in visited:
                    continue

                if len(visited) >= GRAPH_EXPAND_MAX_NODES:
                    stop = True
                    break

                parent_score = visited[parent_id].score
                score = parent_score * GRAPH_HOP_DECAY * (e.weight / _MAX_EDGE_WEIGHT)
                visited[neighbor_id] = SubgraphNode(
                    node_id   = neighbor_id,
                    node_type = "",
                    score     = score,
                    hop       = hop,
                )
                next_frontier.append(neighbor_id)
            frontier = next_frontier

    # ------------------------------------------------------------------
    # Fix B — bounded sibling inclusion (correctly scored, budget-aware)
    # Adds sibling columns of seed tables after expanded nodes, scored
    # below every real expanded node.
    # ------------------------------------------------------------------
    expanded_scores = [sub.score for sub in visited.values() if sub.hop > 0]
    seed_node_ids = [nid for nid, sub in visited.items()
                     if sub.hop == 0 and nid.startswith("col:")]

    if seed_node_ids and len(visited) < GRAPH_EXPAND_MAX_NODES:
        from ingestion.graph_persist import GRAPH_NODES_TABLE
        from ingestion.db_abstraction import (
            INTERNAL_DB_AVAILABLE, get_internal_connection,
            release_internal_connection, DICT_CURSOR,
        )
        if INTERNAL_DB_AVAILABLE:
            try:
                conn = get_internal_connection()
                try:
                    cur = conn.cursor(cursor_factory=DICT_CURSOR)
                    placeholders = ",".join(["%s"] * len(seed_node_ids))
                    cur.execute(
                        f"SELECT node_id, table_id FROM {GRAPH_NODES_TABLE}"
                        f" WHERE node_id IN ({placeholders}) AND node_type = 'column'"
                        f" AND table_id IS NOT NULL",
                        seed_node_ids,
                    )
                    seed_tid_map: Dict[str, float] = {}
                    for r in cur.fetchall():
                        nid, tid = r["node_id"], r["table_id"]
                        s = visited.get(nid, SubgraphNode("", "", 0.0, 0)).score
                        if tid not in seed_tid_map or s > seed_tid_map[tid]:
                            seed_tid_map[tid] = s

                    for tid, tbl_seed_score in seed_tid_map.items():
                        if len(visited) >= GRAPH_EXPAND_MAX_NODES:
                            break
                        sibling_score = (
                            min(expanded_scores) * GRAPH_SIBLING_SCORE_FACTOR
                            if expanded_scores
                            else tbl_seed_score * GRAPH_SIBLING_SCORE_FACTOR
                        )
                        cur.execute(
                            f"SELECT node_id FROM {GRAPH_NODES_TABLE}"
                            f" WHERE node_type = 'column' AND table_id = %s",
                            (tid,),
                        )
                        # When short-circuited, include all columns of the
                        # focused table — it's a single small table, not a hub.
                        sibling_cap = (
                            GRAPH_EXPAND_MAX_NODES if short_circuited and tid == sc_table_id
                            else GRAPH_SIBLING_MAX_PER_TABLE
                        )
                        added = 0
                        for row in cur.fetchall():
                            nid = row["node_id"]
                            if nid in visited or added >= sibling_cap:
                                continue
                            if len(visited) >= GRAPH_EXPAND_MAX_NODES:
                                break
                            visited[nid] = SubgraphNode(
                                node_id   = nid,
                                node_type = "column",
                                score     = sibling_score,
                                hop       = 1,
                            )
                            added += 1
                    cur.close()
                finally:
                    release_internal_connection(conn)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Fix E — chunk safety net
    # Direct pull of mentions/about neighbors from seed+hop-1 column nodes.
    # Chunks are leaf nodes (never hubs), so degree cap must not block them.
    # ------------------------------------------------------------------
    chunk_source_ids = [
        nid for nid, sub in visited.items()
        if sub.hop <= 1 and nid.startswith("col:")
    ]
    if chunk_source_ids:
        chunk_edges = get_neighbors(
            chunk_source_ids,
            edge_types=["mentions", "about"],
            direction="both",
        )
        existing_chunks = sum(1 for nid in visited if nid.startswith("chunk:"))
        for e in chunk_edges:
            if existing_chunks >= GRAPH_MAX_CHUNKS:
                break
            if e.src_node_id.startswith("chunk:") and e.dst_node_id in visited:
                chunk_nid, parent_nid = e.src_node_id, e.dst_node_id
            elif e.dst_node_id.startswith("chunk:") and e.src_node_id in visited:
                chunk_nid, parent_nid = e.dst_node_id, e.src_node_id
            else:
                continue
            if chunk_nid not in visited:
                parent_score = visited[parent_nid].score
                visited[chunk_nid] = SubgraphNode(
                    node_id   = chunk_nid,
                    node_type = "chunk",
                    score     = parent_score * GRAPH_HOP_DECAY,
                    hop       = 1,
                )
                existing_chunks += 1

    # ------------------------------------------------------------------
    # Materialize node metadata
    # ------------------------------------------------------------------
    all_ids = list(visited.keys())
    node_lookup = {n.node_id: n for n in get_nodes(all_ids)}

    chunk_ref_ids = []
    for nid, sub in visited.items():
        meta = node_lookup.get(nid)
        if meta is None:
            continue
        sub.node_type     = meta.node_type
        sub.name          = meta.name
        sub.table_id      = meta.table_id
        sub.table_name    = meta.table_name
        sub.semantic_type = meta.semantic_type
        if meta.node_type == "chunk":
            chunk_ref_ids.append(meta.ref_id)

    # Fetch chunk texts
    chunk_texts = _fetch_chunk_texts(chunk_ref_ids)
    ref_by_node = {nid: node_lookup[nid].ref_id for nid in node_lookup}

    columns: List[SubgraphNode] = []
    chunks_out: list = []
    tables: List[SubgraphNode] = []

    # Build ChunkRetrievalResult objects for chunks
    from ingestion.chunk_embedder import ChunkRetrievalResult

    for nid, sub in visited.items():
        if sub.node_type == "column":
            columns.append(sub)
        elif sub.node_type == "table":
            tables.append(sub)
        elif sub.node_type == "chunk":
            meta = node_lookup.get(nid)
            ref_id = meta.ref_id if meta else nid.replace("chunk:", "")
            ct = chunk_texts.get(ref_id, {})
            sub.text = ct.get("text")
            chunks_out.append(ChunkRetrievalResult(
                chunk_id    = ref_id,
                source_id   = meta.source_id if meta else "",
                doc_id      = (meta.attrs.get("doc_id") if meta else "") or "",
                doc_name    = ct.get("doc_name") or (meta.name if meta else ""),
                chunk_index = (meta.attrs.get("chunk_index") if meta else 0) or 0,
                text        = ct.get("text") or "",
                page_num    = ct.get("page_num"),
                similarity  = round(sub.score, 6),
            ))

    columns.sort(key=lambda n: n.score, reverse=True)
    columns = columns[:GRAPH_MAX_COLS_TO_L3]  # Fix D: truncate before L3
    chunks_out.sort(key=lambda c: c.similarity, reverse=True)
    tables.sort(key=lambda n: n.score, reverse=True)

    duration_ms = round((time.time() - t0) * 1000, 2)
    if verbose:
        print(f"[GraphRetriever] seeds={len(seeds)} visited={len(visited)} "
              f"cols={len(columns)} chunks={len(chunks_out)} "
              f"tables={len(tables)} edges={len(edges_used)} ({duration_ms}ms)")

    return GraphRetrievalResult(
        columns         = columns,
        chunks          = chunks_out,
        tables          = tables,
        edges_used      = edges_used,
        duration_ms     = duration_ms,
        short_circuited = short_circuited,
        stats           = {
            "seeds":   len(seeds),
            "visited": len(visited),
            "columns": len(columns),
            "chunks":  len(chunks_out),
            "tables":  len(tables),
        },
    )


# =============================================================================
# Adapter — SubgraphNode columns → RetrievalResult (for L3/L4 reuse)
# =============================================================================

def _subgraph_to_retrieval_results(column_nodes: List[SubgraphNode]) -> list:
    from ingestion.vector_store import RetrievalResult
    out = []
    for n in column_nodes:
        out.append(RetrievalResult(
            col_id        = n.node_id.replace("col:", ""),
            col_name      = n.name,
            table_id      = n.table_id or "",
            table_name    = n.table_name or "",
            semantic_type = n.semantic_type or "",
            similarity    = round(n.score, 6),
            source_id     = "",
        ))
    return out
