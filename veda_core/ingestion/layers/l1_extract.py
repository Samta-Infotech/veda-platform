"""L1 EXTRACT — the only layer that touches the tenant's source.

After L1 completes, ingestion can finish even if the source goes down. Wraps the
existing schema-scan / FK-adjacency / data-graph / value-sampler stages verbatim.
"""
from __future__ import annotations

from typing import Dict, List

from ingestion.contracts import SourceContext, StageOutcome


def run(ctx: SourceContext, state: Dict, verbose: bool = False) -> List[StageOutcome]:
    out: List[StageOutcome] = []
    source_id = ctx.source_id

    # --- schema scan (real, live introspection) — FATAL on failure -------------
    try:
        from schema.real_schema import get_real_schema
        from ingestion.schema_scanner import run_schema_scanner
        raw_schema = get_real_schema()
        scan_result = run_schema_scanner(raw_schema=raw_schema, verbose=verbose)
        state["scan_result"] = scan_result
        s = scan_result.stats
        out.append(StageOutcome("schema_scan", True, detail=(
            f"{s['total_tables']} tables, {s['total_columns']} cols, "
            f"{s['total_fk_edges']} FK edges")))
    except Exception as e:
        out.append(StageOutcome("schema_scan", False, fatal=True, error=str(e)))
        return out

    # --- FK adjacency store — FATAL -------------------------------------------
    try:
        from ingestion.vector_store import store_fk_adjacency
        fk_result = store_fk_adjacency(scan_result, verbose=verbose)
        state["fk_result"] = fk_result
        out.append(StageOutcome("fk_adjacency", True, detail=(
            f"{fk_result.edges_written} edges, backend={fk_result.backend}")))
    except Exception as e:
        out.append(StageOutcome("fk_adjacency", False, fatal=True, error=str(e)))
        return out

    # --- data graph (undeclared FK discovery) — non-fatal ----------------------
    try:
        from ingestion.data_graph import run_data_graph, to_fk_adjacency_rows
        from ingestion.vector_store import store_fk_adjacency
        dg_result = run_data_graph(scan_result, source_id=source_id, verbose=verbose)
        state["dg_result"] = dg_result
        if dg_result.discovered_edges:
            rows = to_fk_adjacency_rows(dg_result, include_soft=False)
            if rows:
                scan_result.fk_edges = list(scan_result.fk_edges) + rows
                store_fk_adjacency(scan_result, verbose=False)
        st = dg_result.stats
        out.append(StageOutcome("data_graph", True, detail=(
            f"HIGH={st.get('high_certainty', 0)} MED={st.get('medium_certainty', 0)} "
            f"SOFT={st.get('soft_certainty', 0)}")))
    except Exception as e:
        out.append(StageOutcome("data_graph", False, fatal=False, error=str(e)))

    return out


def run_value_sampling(ctx: SourceContext, state: Dict, verbose: bool = False) -> StageOutcome:
    """Value sampler — L1 extract but sequenced after L2 type inference (needs
    inference_result), so composed by pipeline after l2. Non-fatal (matches today)."""
    try:
        from ingestion.value_sampler import run_value_sampler
        vs = run_value_sampler(state["inference_result"], source_id=ctx.source_id, verbose=verbose)
        state["vs_result"] = vs
        return StageOutcome("value_profiling", True, detail=(
            f"{vs.columns_sampled} cols, {vs.total_values} values, backend={vs.backend}"))
    except Exception as e:
        return StageOutcome("value_profiling", False, fatal=False, error=str(e))
