"""L4 INDEX — embeddings + search structures (the model-inference layer).

Graph persist/embed, ensemble encoder + vector store, BGE biencoder, and the NEW
precompute indexes (BM25 Q-2, enrichment Q-3, rerank docs Q-4 — wired in P6/P7).
All model-bound cost is isolated here.
"""
from __future__ import annotations

from typing import Dict, List

from ingestion.contracts import SourceContext, StageOutcome


def run(ctx: SourceContext, state: Dict, verbose: bool = False) -> List[StageOutcome]:
    out: List[StageOutcome] = []
    source_id = ctx.source_id
    graph = state.get("graph")

    from config import (UNIFIED_GRAPH_ENABLED, GRAPH_PERSIST_ENABLED,
                        GRAPH_EMBED_ENABLED, BIENCODER_ENABLED)

    # --- graph persist (graph_nodes / graph_edges) — non-fatal ----------------
    if UNIFIED_GRAPH_ENABLED and GRAPH_PERSIST_ENABLED:
        try:
            from ingestion.graph_persist import persist_reg_graph
            gp = persist_reg_graph(graph=graph, scan_result=state["scan_result"],
                                   dg_result=state.get("dg_result"),
                                   source_id=source_id, verbose=verbose)
            state["graph_persist_result"] = gp
            out.append(StageOutcome("graph_persist", True, detail=(
                f"{gp.nodes_written} nodes, {gp.edges_written} edges")))
        except Exception as e:
            out.append(StageOutcome("graph_persist", False, fatal=False, error=str(e)))

    # --- graph embed (graph_node_embeddings) — non-fatal ----------------------
    if UNIFIED_GRAPH_ENABLED and GRAPH_EMBED_ENABLED:
        try:
            from ingestion.graph_embedder import embed_graph_nodes
            ge = embed_graph_nodes(source_id=source_id, verbose=verbose)
            state["graph_embed_result"] = ge
            out.append(StageOutcome("graph_embed", True, detail=f"{ge.nodes_embedded} nodes"))
        except Exception as e:
            out.append(StageOutcome("graph_embed", False, fatal=False, error=str(e)))

    # NOTE: the ensemble encoder (relgt_encoder → tfidf/svd pkls) and the ensemble
    # vector store (column_embeddings_lt / _hybrid) were removed — the MiniLM/RELGT
    # ensemble retrieval signal is never executed at query time (MiniLM isn't loaded;
    # retrieval_select degrades to the BGE 5-signal spine), so those artifacts were
    # write-only. The BGE biencoder below is the live retrieval store.

    # --- BGE biencoder (column_embeddings_v2) — non-fatal ---------------------
    if BIENCODER_ENABLED:
        try:
            if ctx.resume and _biencoder_embeddings_exist():
                out.append(StageOutcome("biencoder", True, detail="skipped (resume)"))
            else:
                from ingestion.biencoder import run_biencoder_ingestion
                bge = run_biencoder_ingestion(state["inference_result"],
                                              source_id=source_id, verbose=verbose)
                detail = f"{bge.cols_embedded} cols" if not bge.error else f"warn: {bge.error}"
                out.append(StageOutcome("biencoder", True, detail=detail))
        except Exception as e:
            out.append(StageOutcome("biencoder", False, fatal=False, error=str(e)))

    # --- learned-sparse index (WP3, replaces the BM25 index) — non-fatal ------
    try:
        from ingestion.sparse_index import build_sparse_index
        sp = build_sparse_index(state["inference_result"], source_id=source_id, verbose=verbose)
        detail = f"{sp.cols_indexed} cols, {sp.tables_indexed} tables" if not sp.error else f"warn: {sp.error}"
        out.append(StageOutcome("sparse_index", True, detail=detail))
    except Exception as e:
        out.append(StageOutcome("sparse_index", False, fatal=False, error=str(e)))

    # --- enrichment index (NEW, Q-3) — non-fatal ------------------------------
    try:
        from ingestion.enrichment_index import build_enrichment_index
        ei = build_enrichment_index(source_id=source_id, verbose=verbose)
        out.append(StageOutcome("enrichment_index", True, detail=f"{ei.get('terms', 0)} terms"))
    except Exception as e:
        out.append(StageOutcome("enrichment_index", False, fatal=False, error=str(e)))

    # --- rerank docs (NEW, Q-4) — precomputed cross-encoder text — non-fatal --
    try:
        from ingestion.rerank_docs import build_rerank_docs
        rd = build_rerank_docs(source_id=source_id, verbose=verbose)
        out.append(StageOutcome("rerank_docs", True, detail=f"{rd.get('cols', 0)} cols"))
    except Exception as e:
        out.append(StageOutcome("rerank_docs", False, fatal=False, error=str(e)))

    return out


def _biencoder_embeddings_exist() -> bool:
    from config import BIENCODER_COL_TABLE
    try:
        from ingestion.db_abstraction import get_internal_connection, release_internal_connection
        conn = get_internal_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT 1 FROM {BIENCODER_COL_TABLE} LIMIT 1")
                return cur.fetchone() is not None
        finally:
            release_internal_connection(conn)
    except Exception:
        return False
