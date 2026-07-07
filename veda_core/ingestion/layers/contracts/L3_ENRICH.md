# L3 — ENRICH (contract)

> **Role:** the *only* LLM layer (Qwen via Ollama). `skip_llm` skips exactly this
> layer; everything else still yields a queryable (if less enriched) substrate.

Module: `ingestion/layers/l3_enrich.py` — wraps `semantic_layer_v2`
(`run_full_semantic_layer`, `save_semantic_model`).

## Consumes

| Input | Source |
|---|---|
| `ctx: SourceContext` | `skip_llm`, `resume`. |
| live schema | `schema.real_schema.get_real_schema()` → `{table: {columns}}`. |
| config | `SEMANTIC_LAYER_V2_ENABLED`, `SEMANTIC_MODEL_FILE`. |

## Produces

| Stage | `state` key | Side effect | Fatal? |
|---|---|---|---|
| `semantic_layer` | `semantic_model` (tables, `domain_synonyms`, `concept_graph`) or `None` | writes `SEMANTIC_MODEL_FILE` (`veda_semantic_model.json`) | non-fatal |

## Guarantees / invariants

- **Glossary is force-regenerated every ingest** (`force_glossary=True`) so the
  query-time enricher never lags the schema (I-5).
- **Skip paths, all producing a valid `semantic_model=None` state:**
  - `ctx.resume` and the model file already exists → `skipped (resume: model exists)`.
  - `ctx.skip_llm` → `skipped (skip_llm)`.
  - `not SEMANTIC_LAYER_V2_ENABLED` → `disabled`.
- On any LLM failure the biencoder (L4) falls back to structural text — the build
  never aborts here.

## Failure semantics

**Non-fatal by design.** L3 is the only optional-quality layer: without it the
substrate is still fully queryable; retrieval just relies on structural column text
instead of LLM-authored business descriptions/synonyms.

## Downstream consumers

`semantic_model` (+ `veda_semantic_model.json`) → query-time routing, value
grounding, qualifier gate, grain; L4 BGE biencoder (embeds retrieval docs);
L5 semantic registry compile; query-time enrichment (synonyms / concept graph).
