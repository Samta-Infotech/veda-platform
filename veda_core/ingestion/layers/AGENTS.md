# veda_core/ingestion/layers/ — the L1–L5 pipeline

The live ingestion orchestration. Each layer is a **thin wrapper** that sequences stage
functions from `../` (a move, not a rewrite). Contracts:
[`contracts/README.md`](contracts/README.md) + `L1_EXTRACT.md` … `L5_PUBLISH.md`.
Full reference: [../../../docs/INGESTION.md](../../../docs/INGESTION.md).

| File | Role |
|------|------|
| `pipeline.py` | `run_layered_ingestion(ctx, on_stage=)` — composes `_LAYERS = [l1_extract.run, l2_analyze.run, l3_enrich.run, l4_index.run, l5_publish.run]`, threads one in-memory `state` dict, emits `[[STAGE]] <layer> <stage> <ok|fail|fatal>` on stdout per `StageOutcome`, aborts (`raise RuntimeError`) on `fatal + not-ok`. **THE live orchestrator.** |
| `l1_extract.py` | **L1 EXTRACT** — the only layer that touches the source. `schema_scan` (FATAL) + parquet-materialize (tabular) + `fk_adjacency` (FATAL) + `data_graph` (non-fatal). Also exposes `run_value_sampling` + `run_sketch_pass` (sequenced by L2). Connector-aware: tabular engines use `TabularFileConnector`, else `get_real_schema()`. |
| `l2_analyze.py` | **L2 ANALYZE** — pure transforms, no source, no LLM. `semantic_types` (FATAL) + `table_metadata` (FATAL) + [delegates value_profiling + sketch_pass to L1] + `reg_graph` (FATAL) + `join_paths` (non-fatal). |
| `l3_enrich.py` | **L3 ENRICH** — the only LLM layer. `semantic_layer` only (`semantic_layer_v2.run_full_semantic_layer`, `force_glossary=True`). Skips on `ctx.resume` + model-exists, `ctx.skip_llm`, or `not SEMANTIC_LAYER_V2_ENABLED`. Non-fatal. |
| `l4_index.py` | **L4 INDEX** — model inference, all non-fatal. `graph_persist` + `graph_embed` (gated) + `biencoder` (resume-aware) + `sparse_index` + `enrichment_index` + `rerank_docs`. |
| `l5_publish.py` | **L5 PUBLISH** — derived registries, all non-fatal. `relationship_graph` + `semantic_registry` (`compile_all`) + `value_referents` + `hnsw_tune` (inline `clamp(40 + (n_tables//20)*20, 40, 200)` → `veda_hnsw.json`) + `value_mirror` + `unified_graph` (always rebuilt, NOT gated) + `cross_source_fk`. |
| `__init__.py` | Package docstring describing L1–L5. |
| `contracts/*.md` | Per-layer consume/produce/fatal/consumers contracts. **Two stale spots:** `L4_INDEX.md` lists a `bm25_index` stage (it's `sparse_index` now, WP3); `L5_PUBLISH.md` overstates atomicity (the version flip is actually in `apps/ingestion/tasks.py::task_warm_caches` → `storage_adapters/writer.warm`, and readiness is still gated by `Source.ready`, not a per-artifact version pointer). |

## Gotchas
- **L5 "atomic activate" is partly aspirational** — L5 writes flat global files and tunes
  HNSW; the actual version flip + rehydrate happens later in `task_warm_caches`.
- **Resume is stage-level skip, not resume-from-N**: L3 skips if `veda_semantic_model.json`
  exists; L4 `biencoder` skips if `column_embeddings_v2` has rows (`_biencoder_embeddings_exist`).
  L1/L2 always re-run to rebuild the in-memory `state`.
- Cross-source stages (`column_sketches`, `cross_source_fk`, `entity_linker`,
  `semantic_linker`) are a later plan layered on top — wired, mostly non-fatal/best-effort.
