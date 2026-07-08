"""Type-aware ingestion dispatcher (Track 2/3 — P4/P5).

The single entry the API-triggered worker calls with a resolved ``SourceContext``.
Routes by ``source.type``:

    relational → the layered L1–L5 pipeline (layers.pipeline.run_layered_ingestion)
    document   → doc plan (chunk → embed → publish) via source_dispatcher
    nosql/datalake → schema-pipeline plan via source_dispatcher

This replaces the hardcoded "always run the primary relational pipeline" path
(I-2): the passed context's source is honoured, never re-derived (§3.2).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from ingestion.contracts import SourceContext, StageOutcome


def dispatch(
    ctx: SourceContext,
    verbose: bool = False,
    on_stage: Optional[Callable[[StageOutcome], None]] = None,
) -> Dict:
    """Route the given source to the matching pipeline. Returns a result dict with
    at least ``source_id``, ``source_type``, ``success``."""
    if ctx.type == "relational":
        from ingestion.layers.pipeline import run_layered_ingestion
        state = run_layered_ingestion(ctx, verbose=verbose, on_stage=on_stage)
        return {"source_id": ctx.source_id, "source_type": "relational",
                "success": True, "state": state}

    # Non-relational: delegate to the existing type-aware source_dispatcher, which
    # honours the passed source_config (document/nosql/datalake connectors).
    from ingestion.source_dispatcher import dispatch_ingestion
    cfg = {
        "id": ctx.source_id, "type": ctx.type, "engine": ctx.engine,
        "enabled": True, "role": "queryable", **ctx.connection,
        "exclude_tables": ctx.exclude_tables, "schema": ctx.schema_filter,
        "industry_vertical": ctx.industry_vertical,
    }
    result = dispatch_ingestion(cfg, verbose=verbose)
    return {"source_id": ctx.source_id, "source_type": ctx.type,
            "success": bool(getattr(result, "success", False)), "result": result}
