"""L3 ENRICH — the only LLM layer (Qwen semantic layer v2 + glossary + concepts).

``skip_llm`` skips exactly this layer; everything else still produces a queryable
(if less enriched) substrate. Glossary is force-regenerated every ingest so the
query-time enricher never lags the schema (I-5 fix, already wired).
"""
from __future__ import annotations

import os
from typing import Dict, List

from ingestion.contracts import SourceContext, StageOutcome


def _artifact_paths(ctx: SourceContext):
    """Per-(tenant, source) paths for this layer's 4 outputs (P0-5, 2026-09-11) —
    unconditionally, via config.source_artifact_path(), not gated behind
    VEDA_ARTIFACT_SCOPE (P0-7). Before this fix these were ALL flat files shared by
    every source: source B's ingest could resume-skip because source A's semantic
    model happened to exist on disk (the exact P0-3 gap left open earlier), and
    then, had it not skipped, would have overwritten A's model/glossary/synonyms/
    concept-graph on save — the same class of bug P0-4 fixed for the relationship
    graph."""
    from config import source_artifact_path
    sid, tenant = ctx.source_id, ctx.tenant
    return {
        "semantic_model": source_artifact_path("veda_semantic_model.json", sid, tenant),
        "domain_synonyms": source_artifact_path("veda_domain_synonyms.json", sid, tenant),
        "concept_graph": source_artifact_path("veda_concept_graph.json", sid, tenant),
        "glossary": source_artifact_path("veda_glossary.json", sid, tenant),
    }


def run(ctx: SourceContext, state: Dict, verbose: bool = False) -> List[StageOutcome]:
    out: List[StageOutcome] = []

    from config import SEMANTIC_LAYER_V2_ENABLED
    paths = _artifact_paths(ctx)

    # Resume-skip: skip this expensive LLM stage if THIS SOURCE's own output already
    # exists (P0-3/P0-5, 2026-09-11 — was `os.path.exists(SEMANTIC_MODEL_FILE)`, the
    # flat path every source shared, so source B's ingest could see source A's
    # model and wrongly skip its own).
    if ctx.resume and os.path.exists(paths["semantic_model"]):
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
            schema_dict=schema_dict, profiling=None, glossary=None, force_glossary=True,
            domain_synonyms_file=paths["domain_synonyms"],
            concept_graph_file=paths["concept_graph"],
            glossary_file=paths["glossary"])
        save_semantic_model(semantic_model, paths["semantic_model"])
        state["semantic_model"] = semantic_model
        out.append(StageOutcome("semantic_layer", True, detail=(
            f"{len(semantic_model.get('tables', {}))} tables, "
            f"{len(semantic_model.get('domain_synonyms', {}))} synonyms, "
            f"{len(semantic_model.get('concept_graph', {}))} concepts")))
    except Exception as e:
        # Non-fatal: biencoder falls back to structural text (matches run_ingestion).
        out.append(StageOutcome("semantic_layer", False, fatal=False, error=str(e)))

    return out
