"""L2 ANALYZE — pure transforms over extracted data (no source touch, no LLM).

Semantic-type inference, table metadata (display columns), REG graph build, and
the NEW precomputed pairwise join paths (Q-9, wired in P7).
"""
from __future__ import annotations

from typing import Dict, List

from ingestion.contracts import SourceContext, StageOutcome


def run(ctx: SourceContext, state: Dict, verbose: bool = False) -> List[StageOutcome]:
    out: List[StageOutcome] = []

    # --- semantic type inference — FATAL --------------------------------------
    try:
        from ingestion.semantic_type_inference import run_semantic_type_inference
        inference_result = run_semantic_type_inference(
            scan_result=state["scan_result"], verbose=verbose)
        state["inference_result"] = inference_result
        s = inference_result.stats
        out.append(StageOutcome("semantic_types", True, detail=(
            f"avg_conf={s['avg_confidence']}, flagged={s['flagged_count']}")))
    except Exception as e:
        out.append(StageOutcome("semantic_types", False, fatal=True, error=str(e)))
        return out

    # --- table metadata (display columns) — FATAL -----------------------------
    try:
        from ingestion.vector_store import store_table_metadata
        tm = store_table_metadata(inference_result, verbose=verbose)
        state["tm_result"] = tm
        out.append(StageOutcome("table_metadata", True, detail=(
            f"{tm.rows_written} tables, backend={tm.backend}")))
    except Exception as e:
        out.append(StageOutcome("table_metadata", False, fatal=True, error=str(e)))
        return out

    # --- value sampling (L1-extract, sequenced here — needs inference_result) --
    from ingestion.layers import l1_extract
    out.append(l1_extract.run_value_sampling(ctx, state, verbose=verbose))
    # --- cross-source MinHash sketch pass (needs inference_result; P2.4/P4.2) ----
    out.append(l1_extract.run_sketch_pass(ctx, state, verbose=verbose))

    # --- REG builder — FATAL ---------------------------------------------------
    try:
        from ingestion.reg_builder import run_reg_builder
        graph = run_reg_builder(inference_result=inference_result, verbose=verbose)
        state["graph"] = graph
        s = graph.stats
        out.append(StageOutcome("reg_graph", True, detail=(
            f"{s['num_table_nodes']} table nodes, {s['num_column_nodes']} col nodes")))
    except Exception as e:
        out.append(StageOutcome("reg_graph", False, fatal=True, error=str(e)))
        return out

    # --- join paths (NEW, Q-9) — precompute pairwise shortest FK paths ---------
    # Additive + non-fatal: consumed by join_planner when present (P7).
    try:
        from ingestion.join_paths import build_join_paths
        jp = build_join_paths(state["scan_result"], source_id=ctx.source_id, verbose=verbose)
        state["join_paths"] = jp
        out.append(StageOutcome("join_paths", True, detail=f"{len(jp)} table-pairs"))
    except Exception as e:  # module optional until P7 lands its writer
        out.append(StageOutcome("join_paths", False, fatal=False, error=str(e)))

    return out
