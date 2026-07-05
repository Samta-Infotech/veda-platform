"""query/slm_langgraph.py — LangGraph pipeline, drop-in for run_slm_layer()."""

import time
from typing import List, Optional

from langgraph.graph import StateGraph, END

from query.lg_nodes import (
    VEDAQueryState,
    node_classify_intent,
    node_select_entity,
    node_select_columns,
    node_build_filters,
    node_assemble_ir,
    should_continue,
)
from query.slm_layer import (
    SLMResult,
    _compute_must_include,
    _validate_ir,
    _fallback_result,
)
from config import TOP_K_TO_LLM

# =============================================================================
# Graph — compiled once and cached at module level
# =============================================================================

def _build_graph():
    g = StateGraph(VEDAQueryState)

    g.add_node("classify_intent", node_classify_intent)
    g.add_node("select_entity",   node_select_entity)
    g.add_node("select_columns",  node_select_columns)
    g.add_node("build_filters",   node_build_filters)
    g.add_node("assemble_ir",     node_assemble_ir)

    g.set_entry_point("classify_intent")
    g.add_conditional_edges(
        "classify_intent",
        should_continue,
        {"select_entity": "select_entity", "assemble": "assemble_ir"},
    )
    g.add_edge("select_entity",  "select_columns")
    g.add_edge("select_columns", "build_filters")
    g.add_edge("build_filters",  "assemble_ir")
    g.add_edge("assemble_ir",    END)

    return g.compile()


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# =============================================================================
# Public entry point — same signature as run_slm_layer()
# =============================================================================

def run_langgraph_pipeline(
    query:           str,
    temporal_filter,
    top_k_columns:   list,
    join_path:       list,
    verbose:         bool = False,
    is_hybrid:       bool = False,
) -> SLMResult:
    t0 = time.time()

    try:
        # ── Cap + filter (same logic as run_slm_layer) ────────────────────────
        llm_columns   = top_k_columns[:TOP_K_TO_LLM]
        llm_table_ids = {r.table_id for r in llm_columns}
        llm_join_path = [e for e in join_path if e.from_table_id in llm_table_ids]

        must_include_results = _compute_must_include(query, top_k_columns, join_path)

        # ── Convert objects → dicts for state ────────────────────────────────
        cols_dicts = [
            {
                "col_id":        r.col_id,
                "col_name":      r.col_name,
                "table_id":      r.table_id,
                "table_name":    r.table_name,
                "semantic_type": r.semantic_type,
                "similarity":    r.similarity,
            }
            for r in llm_columns
        ]
        join_dicts = [
            {
                "from_table_id":   e.from_table_id,
                "from_table_name": e.from_table_name,
                "from_col_id":     e.from_col_id,
                "from_col_name":   e.from_col_name,
                "to_table_id":     e.to_table_id,
                "to_table_name":   e.to_table_name,
                "to_col_id":       e.to_col_id,
                "to_col_name":     e.to_col_name,
                "join_type":       e.join_type,
            }
            for e in llm_join_path
        ]
        tf_dict = None
        if temporal_filter and (temporal_filter.start or temporal_filter.end):
            tf_dict = {}
            if temporal_filter.start:
                tf_dict["start"] = temporal_filter.start
            if temporal_filter.end:
                tf_dict["end"] = temporal_filter.end

        must_dicts = [
            {
                "col_id":    r.col_id,
                "col_name":  r.col_name,
                "table_id":  r.table_id,
                "table_name": r.table_name,
            }
            for r in must_include_results
        ]

        # ── Build initial state ───────────────────────────────────────────────
        initial_state: VEDAQueryState = {
            "query":           query,
            "temporal_filter": tf_dict,
            "top_k_columns":   cols_dicts,
            "join_path":       join_dicts,
            "must_include":    must_dicts,
            "errors":          [],
            "node_times":      {},
        }

        if verbose:
            print(f"[LangGraph] Columns passed : {len(llm_columns)} (capped from {len(top_k_columns)})")
            print(f"[LangGraph] Joins passed   : {len(llm_join_path)} (filtered from {len(join_path)})")

        # ── Run graph ─────────────────────────────────────────────────────────
        graph       = _get_graph()
        final_state = graph.invoke(initial_state)

        # ── Extract results ───────────────────────────────────────────────────
        ir_json    = final_state.get("ir_json", {})
        intent     = final_state.get("intent", "SELECT")
        complexity = final_state.get("complexity", "SIMPLE")
        confidence = float(final_state.get("confidence", 0.9))
        node_times = final_state.get("node_times", {})
        errors     = final_state.get("errors", [])

        needs_clarification  = bool(final_state.get("needs_clarification", False))
        clarification_reason = final_state.get("clarification_reason")

        # ── Validate UUIDs against llm_columns ───────────────────────────────
        warnings = _validate_ir({}, llm_columns, ir_json)

        dur = round((time.time() - t0) * 1000, 2)

        if verbose:
            total_node_ms = sum(node_times.values())
            print(f"[LangGraph] Node times: {node_times}")
            print(f"[LangGraph] Total node ms : {total_node_ms:.1f}ms  | Wall: {dur}ms")
            print(f"[LangGraph] intent={intent}  complexity={complexity}  warnings={len(warnings)}")
            if errors:
                print(f"[LangGraph] Errors: {errors}")

        result = SLMResult(
            intent               = intent,
            complexity           = complexity,
            needs_clarification  = needs_clarification,
            clarification_reason = clarification_reason,
            confidence           = confidence,
            ir_json              = ir_json,
            raw_response         = str(node_times),
            duration_ms          = dur,
            error                = ("; ".join(errors) if errors else None),
            validation_warnings  = warnings,
        )
        result.node_times = node_times
        return result

    except Exception as exc:
        dur = round((time.time() - t0) * 1000, 2)
        return _fallback_result(query, str(exc), dur)
