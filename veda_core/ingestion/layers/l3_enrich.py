"""L3 ENRICH — the only LLM layer (Qwen semantic layer v2 + glossary + concepts).

``skip_llm`` skips exactly this layer; everything else still produces a queryable
(if less enriched) substrate. Glossary is force-regenerated every ingest so the
query-time enricher never lags the schema (I-5 fix, already wired).
"""
from __future__ import annotations

import os
from typing import Dict, List

from ingestion.contracts import SourceContext, StageOutcome


def run(ctx: SourceContext, state: Dict, verbose: bool = False) -> List[StageOutcome]:
    out: List[StageOutcome] = []

    from config import SEMANTIC_LAYER_V2_ENABLED, SEMANTIC_MODEL_FILE

    # Resume-skip: skip this expensive LLM stage if its output already exists.
    if ctx.resume and os.path.exists(SEMANTIC_MODEL_FILE):
        state["semantic_model"] = None
        out.append(StageOutcome("semantic_layer", True, detail="skipped (resume: model exists)"))
        return out

    if ctx.skip_llm or not SEMANTIC_LAYER_V2_ENABLED:
        state["semantic_model"] = None
        out.append(StageOutcome("semantic_layer", True,
                                detail="skipped (skip_llm)" if ctx.skip_llm else "disabled"))
        return out

    try:
        from schema.real_schema import get_real_schema
        from ingestion.semantic_layer_v2 import run_full_semantic_layer, save_semantic_model
        raw = get_real_schema()
        schema_dict = {
            t["table_name"]: {"columns": t.get("columns", [])}
            for t in raw.get("tables", [])
        }
        # force_glossary=True → regenerate the glossary every ingest (I-5).
        semantic_model = run_full_semantic_layer(
            schema_dict=schema_dict, profiling=None, glossary=None, force_glossary=True)
        save_semantic_model(semantic_model, SEMANTIC_MODEL_FILE)
        state["semantic_model"] = semantic_model
        out.append(StageOutcome("semantic_layer", True, detail=(
            f"{len(semantic_model.get('tables', {}))} tables, "
            f"{len(semantic_model.get('domain_synonyms', {}))} synonyms, "
            f"{len(semantic_model.get('concept_graph', {}))} concepts")))
    except Exception as e:
        # Non-fatal: biencoder falls back to structural text (matches run_ingestion).
        out.append(StageOutcome("semantic_layer", False, fatal=False, error=str(e)))

    return out
