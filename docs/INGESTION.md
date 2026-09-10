# INGESTION — the offline build

Deep reference for `veda_core/main.py::run_ingestion` and `veda_core/ingestion/**` (incl.
`layers/`), `veda_core/connectors/**`, `veda_core/schema/**`. Expands
[ARCHITECTURE.md](ARCHITECTURE.md) §7. Written from a direct read of the source on `master`
(2026-09-09).

> **Authority.** When code and prose disagree, `apps/ingestion/tasks.py` +
> `ingestion/layers/pipeline.py` win. The old `veda_core/ingestion/INGESTION.md` describes a
> **deleted monolith** — do not trust it (it is now a pointer doc).

Related: [ARCHITECTURE.md](ARCHITECTURE.md) · the layer contracts
(`veda_core/ingestion/layers/contracts/*.md`) · operator runbook
[INGESTION_GUIDE.md](../INGESTION_GUIDE.md) ·
[DATA_SOURCES_EMBEDDINGS_AND_GRAPHS.md](DATA_SOURCES_EMBEDDINGS_AND_GRAPHS.md) ·
[SEMANTIC_ENTITY_BRIDGE.md](SEMANTIC_ENTITY_BRIDGE.md) · [RETRIEVAL.md](RETRIEVAL.md)
(consumes every store this build writes).

---

## 1. One pipeline, not two

The "monolith vs layered" duality was resolved in the cleanup plan
([`archive/CLEANUP_PLAN.md`](archive/CLEANUP_PLAN.md)) Phase 7:
`main.run_ingestion` **no longer has a stage body** — it is a thin shim over the layered
L1–L5 pipeline. The layer modules (`layers/l1_extract.py` … `l5_publish.py`) are **thin
wrappers** that call the exact same stage functions the old monolith called ("a move, not a
rewrite").

```
apps/ingestion/tasks.py::task_ingest_source(source_id, tenant, force, skip_llm, resume)
  │   creates IngestionJob + ordered IngestionStage rows
  │   _guard_embedding_model_change  (EMBEDDING_MODEL_ID vs last successful job)
  │   injects Source.as_engine_env() → VEDA_SOURCE_*
  ▼
_run_engine_pipeline → subprocess.Popen(["python","-u","-c", prog], cwd=veda_core)
  │   (subprocess isolates the engine's top-level `config` module from Django's `config`
  │    package — they collide in one interpreter)
  │   _build_engine_command routes by source.source_kind():
  │     relational          → import main; main.run_ingestion(verbose=False, skip_llm=…)
  │     nosql/document/datalake → ingestion.source_dispatcher.dispatch_ingestion(cfg, …)
  ▼
veda_core/main.py::run_ingestion  (main.py:272)
  │   ctx = SourceContext.from_env(skip_llm=)   # the one place the engine learns its source
  │   optional SLM prewarm
  ▼
ingestion/dispatcher.py::dispatch  (dispatcher.py:21)
  │   ctx.type == "relational"  OR  engine ∈ {csv, csv_lake, parquet, xlsx, excel}
  │        → ingestion.layers.pipeline.run_layered_ingestion(ctx)
  │   else → source_dispatcher.dispatch_ingestion(cfg)
  ▼
ingestion/layers/pipeline.py::run_layered_ingestion  (pipeline.py:29)
  │   _LAYERS = [l1_extract.run, l2_analyze.run, l3_enrich.run, l4_index.run, l5_publish.run]
  │   threads one in-memory `state` dict
  │   emits  [[STAGE]] <layer> <stage> <ok|fail|fatal>  on stdout per StageOutcome
  │   fatal + not-ok → raise RuntimeError, abort
  ▼
(subprocess returns) → tasks.py: mark stages SUCCESS
  ▼
task_warm_caches(source_id, tenant) → storage_adapters/writer.py::warm()
  │   sync_from_engine()  (engine store rows → Django Sm* substrate)
  │   assembler.publish_sm  +  publish_rehydrate(scope="all")
  ▼
Source.objects.filter(pk=source_id).update(ready=True, status=READY, last_ingested_at=now)
  ▼
flag-gated never-raise post-steps: _sync_catalog_if_enabled, profile_source_if_enabled,
                                   build_source_items_if_enabled
```

### `source_dispatcher.py` is not dead

It is the **non-relational path**, reachable from `tasks.py` (nosql/document/datalake) and
from `dispatcher.dispatch` (non-relational, non-tabular). For a *relational* source with
`use_full_pipeline=True` it loops back into the layered pipeline
(`source_dispatcher.py:352` → `from main import run_ingestion`). Its `_run_schema_pipeline`
(`source_dispatcher.py:84-314`) is the standalone schema-only chain for datalake/nosql.

### `tasks.py` carries two stdout protocols (`_consume_engine_output`, `tasks.py:534`)

| Protocol | Regex | Status |
|---|---|---|
| `_STAGE_EVENT_RE` = `[[STAGE]] <layer> <stage> <status>` | emitted by `pipeline.py:47`, mapped to `IngestionStage` rows via `_LAYER_STAGE_TO_ROW` (`tasks.py:110-128`) | **live** |
| `_MARKER_RE` = `[N/NN] StageName` + `_ENGINE_STEP_TO_STAGE` (`tasks.py:130-140`) | legacy monolith progress markers; `main._step` emits these only for `run_evaluation_pipeline` / `--build-glossary` | **dead for ingestion** — kept, unexercised |

---

## 2. The layered flow, end to end (relational source)

`state` is threaded through all layers (`pipeline.py:35`); a fatal stage aborts.
Key `state` chain: `scan_result → inference_result → graph → semantic_model`.

### L1 EXTRACT — `layers/l1_extract.py::run` — the only layer that touches the source

| Stage | fn (file:line) | Produces | Consumed by | Fatal |
|---|---|---|---|---|
| `schema_scan` | `schema_scanner.run_schema_scanner(get_real_schema())` — `l1_extract.py:40` | `state["scan_result"]` (`.stats`, `.fk_edges`) | L2, L4, L5 | **FATAL** |
| `materialize_parquet` | tabular only — `TabularFileConnector.materialize_parquet` — `l1_extract.py:64-68` | `ARTIFACT_ROOT/<source_id>/tables/*.parquet` | query `cross_source_composer` | non-fatal |
| `fk_adjacency` | `vector_store.store_fk_adjacency(scan_result)` — `l1_extract.py:76` | FK-adjacency store (internal DB) | query join planner | **FATAL** |
| `data_graph` | `data_graph.run_data_graph` + merge HIGH/MED rows via `store_fk_adjacency` — `l1_extract.py:86-94` | `state["dg_result"]`; augments FK-adjacency | L4 graph_persist | non-fatal |

`l1_extract` also exposes `run_value_sampling` and `run_sketch_pass`, **sequenced by L2**.
Connector-aware: tabular engines use `TabularFileConnector`, else `get_real_schema()`.

### L2 ANALYZE — `layers/l2_analyze.py::run` — pure, no source, no LLM

| Stage | fn (file:line) | Produces | Consumed by | Fatal |
|---|---|---|---|---|
| `semantic_types` | `semantic_type_inference.run_semantic_type_inference(scan_result)` — `l2_analyze.py:20` | `state["inference_result"]` (`.typed_columns`) | L3, L4 biencoder/sparse, value sampler | **FATAL** |
| `table_metadata` | `vector_store.store_table_metadata(inference_result, source_id)` — `l2_analyze.py:33` | table-metadata / display-columns store | query display-column resolution | **FATAL** |
| `value_profiling` | `l1_extract.run_value_sampling` → `value_sampler.run_value_sampler` — `l2_analyze.py:42` | `state["vs_result"]`; `column_values` store | query value grounding; L5 value_mirror / value_referents | non-fatal |
| `column_sketches` | `l1_extract.run_sketch_pass` → `column_sketches.persist_sketches` — `l2_analyze.py:44` | `column_sketches` table | L5 cross_source_fk | non-fatal (no-op w/o `datasketch`) |
| `reg_graph` | `reg_builder.run_reg_builder(inference_result)` — `l2_analyze.py:49` | `state["graph"]` (in-mem REG) | L4 graph_persist / embed | **FATAL** |
| `join_paths` | `join_paths.build_join_paths(scan_result, source_id)` — `l2_analyze.py:62` | `veda_join_paths.json` | query `join_planner` | non-fatal |

### L3 ENRICH — `layers/l3_enrich.py::run` — the only LLM layer

| Stage | fn (file:line) | Produces | Consumed by | Fatal |
|---|---|---|---|---|
| `semantic_layer` | `semantic_layer_v2.run_full_semantic_layer(schema_dict, force_glossary=True)` + `save_semantic_model` — `l3_enrich.py:41-43` | `data/veda_semantic_model.json` (tables, `domain_synonyms`, `concept_graph`, `retrieval_documents`), glossary, synonyms, concepts JSON; `state["semantic_model"]` | L4 biencoder; L5 registry/graph; query routing/grounding/qualifier/grain/enrichment | non-fatal |

Skip conditions → `semantic_model=None`, stage still `ok` (`l3_enrich.py:21-30`):
`ctx.resume` + file exists; `ctx.skip_llm`; `not SEMANTIC_LAYER_V2_ENABLED`. The biencoder
then falls back to structural text.

`semantic_layer_v2` internally is a 5-stage hybrid build (profiling → glossary → table
understanding → column understanding → retrieval docs); Qwen via Ollama in stages 2–4.
Sub-modules: `data_profiler` (Stage 1 → `veda_profiling.json`), `glossary_builder`
(Stage 2 → `veda_glossary.json`), `deterministic_metadata` (rule-based semantic_type /
analytics_role / sql_usage / importance_class / aliases).

### L4 INDEX — `layers/l4_index.py::run` — model inference; all non-fatal

| Stage | fn | Produces | Consumed by (query) | Gate |
|---|---|---|---|---|
| `graph_persist` | `graph_persist.persist_reg_graph` | `graph_nodes`, `graph_edges` | `GRAPH_EXPAND`, graph_retriever | `UNIFIED_GRAPH_ENABLED and GRAPH_PERSIST_ENABLED` |
| `graph_embed` | `graph_embedder.embed_graph_nodes` | `graph_node_embeddings` (HNSW) | graph seed retrieval, semantic_linker | `UNIFIED_GRAPH_ENABLED and GRAPH_EMBED_ENABLED` |
| `biencoder` | `biencoder.run_biencoder_ingestion(inference_result, source_id)` | `column_embeddings_v2`, `table_embeddings_v2` (BGE-M3 dense, 1024-dim, HNSW cosine, scoped delete-then-insert) | retrieval Signal 1; table routing | `BIENCODER_ENABLED`; resume-skip if the table already has rows (`l4_index.py:54`) |
| `sparse_index` | `sparse_index.build_sparse_index` | `column_sparse_v1`, `table_sparse_v1` (learned-sparse token→weight) | `retrieval/sparse_ranker.py` Signal 2 | — |
| `enrichment_index` | `enrichment_index.build_enrichment_index` | `veda_enrichment_index.json` | `retrieval/query_enrichment.py` | — |
| `rerank_docs` | `rerank_docs.build_rerank_docs` | `veda_rerank_docs.json` | `query/reranker.py` | — |

> **Contract-doc drift:** `contracts/L4_INDEX.md` lists a `bm25_index` stage. **It does not
> exist** — WP3 replaced it with `sparse_index` (BGE-M3 learned-sparse). The removed
> MiniLM/RELGT ensemble encoder and `_lt` / `_hybrid` stores are intentionally gone.

### L5 PUBLISH — `layers/l5_publish.py::run` — derived registries; all non-fatal

| Stage | fn | Produces | Consumed by | Gate |
|---|---|---|---|---|
| `relationship_graph` | `relationship_graph.build_relationship_graph` | `data/veda_relationship_graph.json` | query join planner / fast path / graph guard | `DERIVED_ARTIFACTS_ENABLED` |
| `semantic_registry` | `semantic.compile_semantic_layer.compile_all(write=True)` | `semantic/{concepts,dimensions,metrics,MANIFEST}.json` | query fast path | `DERIVED_ARTIFACTS_ENABLED` |
| `value_referents` | `value_referents.write_value_referents` | `<ART>/<tenant>/<source>/veda_value_referents.json` | deterministic planners, typed anchors, Tier-2 qualifier gate | `DERIVED_ARTIFACTS_ENABLED` |
| `hnsw_tune` | inline — `clamp(40 + (n_tables // 20) * 20, 40, 200)` — `l5_publish.py:88-93` | `data/veda_hnsw.json` + `state["hnsw_ef_search"]` | `task_warm_caches` → `SubstrateVersion.hnsw_ef_search` → pgvector `SET LOCAL ef_search` | — |
| `value_mirror` | `value_mirror.mirror_values_to_redis` | Redis `value:{tenant}:{source}:{norm}` hashes | query value resolver / arbiter (Postgres fallback) | — |
| `unified_graph` | `unified_graph_builder.build_unified_graph` + `write_unified_graph` | `data/veda_unified_graph.json` | query `GRAPH_EXPAND` | **ALWAYS rebuilt** — deliberately not gated on `UNIFIED_GRAPH_ENABLED` (`l5_publish.py:107-115`) |
| `cross_source_fk` | `cross_source_graph.discover_and_persist(tenant)` | `cross_source_fk` edges in `graph_edges` | query federated join planner | — (no-op until ≥2 sources have sketches) |

> **"Atomic activate" is partly aspirational.** `L5_PUBLISH.md` says the query tier flips
> once L5 publishes. Reality: `l5_publish.py` writes flat global files (`data/veda_*.json`)
> and tunes HNSW; the actual version-flip + rehydrate happens later in `task_warm_caches`
> → `writer.warm()`, and readiness is still gated only by `Source.ready` — there is no
> per-artifact `SubstrateVersion` pointer yet.

---

## 3. Layer status

| Layer | Status | Notes |
|---|---|---|
| L1 EXTRACT | fully wired | schema_scan + fk_adjacency + data_graph live; tabular parquet-materialize wired for file sources |
| L2 ANALYZE | fully wired | all 6 stages run; `join_paths` + `column_sketches` are newer additive stages, wired and consumed |
| L3 ENRICH | fully wired | `semantic_layer_v2` runs; `skip_llm` / `resume` / disabled skip paths all produce a valid degraded state; glossary force-regenerated |
| L4 INDEX | wired, degraded-tolerant | biencoder + sparse_index are the live retrieval stores; `bm25_index` in the contract doesn't exist |
| L5 PUBLISH | wired, not truly atomic | all stages run, all non-fatal; version-flip is in `task_warm_caches`; `artifact_scope` OFF by default → N relational sources overwrite the same `data/veda_*.json` |

Cross-source additions (`column_sketches`, `cross_source_graph`, `entity_linker`,
`semantic_linker`, `value_embedder`, `value_referents`, tabular connector) are a later plan
layered on top of the CLEANUP_PLAN L1–L5 — wired, mostly non-fatal / best-effort.

---

## 4. Connectors — supported source types

`build_connector(source_config)` (`connectors/base.py`) dispatches on `type` / `engine`.

| Source type | Connector | What it does | Ingestion path |
|---|---|---|---|
| **relational** | `RelationalConnector` (`relational.py`, 45 KB) | Live DB-API 2.0 INFORMATION_SCHEMA introspection → `RawSchema`; PostgreSQL/MySQL/SQLite/Oracle/SQL Server. Owns `get_real_schema()`. | full L1–L5 layered pipeline |
| **document** | `FilesystemDocumentConnector` (`document.py`) + `doc_parser.py` | Walk a dir → `DocumentChunk`s (PDF/DOCX/TXT/MD/HTML), layout-aware sections + heading breadcrumbs. Optional deps: pdfplumber, python-docx, bs4, pymupdf4llm. | `source_dispatcher._dispatch_document`: get_chunks → `chunk_embedder` → `doc_chunks`; then `entity_linker` + `semantic_linker` + `cross_source_graph` (best-effort) |
| **nosql** | `NoSQLConnector` (`nosql.py`) | MongoDB / Elasticsearch / DynamoDB. Sample docs → flatten nested fields → infer types → `NoSQLCollection`s. `execute_query()` per-engine at query time. | `source_dispatcher._dispatch_nosql`: `nosql_collections_to_dict` → `_run_schema_pipeline` (no data_graph, no value_sampler) |
| **datalake** | `DatalakeConnector` (`datalake.py`) OR `TabularFileConnector` (`tabular_files.py`) | Delta/Parquet/CSV via in-process DuckDB. `datalake.py` = light schema pipeline + random UUIDs. `tabular_files.py` = FULL relational pipeline + deterministic UUIDv5 (idempotent) + parquet materialization. | tabular engines (csv/csv_lake/parquet/xlsx/excel) → full `run_layered_ingestion`. Non-tabular (delta/iceberg) → generic connector + `_run_schema_pipeline` |

`schema_unifier.py` bridges connector dataclasses (`RawSchema` / `NoSQLCollection`) → the
legacy dict `schema_scanner` expects.

---

## 5. Artifacts produced (exhaustive)

| Artifact | Path / DB table | Producer (stage · file:line) | Consumers |
|---|---|---|---|
| Schema scan result | in-memory `state["scan_result"]` | L1 `schema_scan` · `l1_extract.py:40` | L2, L4, L5 (in-process) |
| FK adjacency store | internal DB (fk_adjacency) | L1 `fk_adjacency` · `l1_extract.py:76`; augmented by data_graph · `:94` | query join planner, `vector_store.get_fk_adjacency` |
| Materialized parquet | `data/<source_id>/tables/*.parquet` | L1 `materialize_parquet` (tabular) · `l1_extract.py:66` | query `cross_source_composer` federated execute |
| `column_values` store | internal DB `column_values` | L2 `value_profiling` → `value_sampler.run_value_sampler` | query value grounding / arbiter; L5 value_mirror + value_referents |
| `column_sketches` | internal DB `column_sketches` | L2 `column_sketches` → `persist_sketches` | L5 `cross_source_graph` |
| Table metadata / display columns | internal DB (table_metadata) | L2 `table_metadata` → `vector_store.store_table_metadata` | query display-column resolution |
| Semantic types | in-memory `state["inference_result"]` | L2 `semantic_types` | L3, L4, value sampler |
| REG graph | in-memory `state["graph"]` | L2 `reg_graph` → `reg_builder.run_reg_builder` | L4 graph_persist / embed (never persisted itself) |
| Join paths | `data/veda_join_paths.json` | L2 `join_paths` → `build_join_paths` | query `join_planner` |
| Semantic model | `data/veda_semantic_model.json` | L3 `semantic_layer` → `save_semantic_model` | query routing / grounding / qualifier / grain; L4 biencoder; L5 registry / graph |
| Domain synonyms | `data/veda_domain_synonyms.json` | L3 (post-processing) | query enrichment, unified_graph |
| Concept graph | `data/veda_concept_graph.json` | L3 | query enrichment, unified_graph |
| Domain glossary | `glossary/domain_glossary.json` + `veda_glossary.json` | L3 (`force_glossary=True`) via `glossary_builder` / `domain_glossary` | query `semantic_layer` enrichment, `enrichment_index` |
| Data profiling | `veda_profiling.json` | L3 stage 1 → `data_profiler.run_profiling` | `semantic_layer_v2` retrieval-doc enrichment |
| Column embeddings (dense) | pgvector `column_embeddings_v2` (1024-dim, HNSW cosine) | L4 `biencoder` · `biencoder.py:190` | retrieval Signal 1 |
| Table embeddings (dense) | pgvector `table_embeddings_v2` | L4 `biencoder` · `biencoder.py:237` | table routing (`route_tables_semantic`) |
| Column sparse index | internal DB `column_sparse_v1` | L4 `sparse_index` → `build_sparse_index` | `retrieval/sparse_ranker.py` Signal 2 |
| Table sparse index | internal DB `table_sparse_v1` | L4 `sparse_index` | WP4 table prior (sparse half) |
| Graph nodes / edges | internal DB `graph_nodes`, `graph_edges` | L4 `graph_persist` → `persist_reg_graph` | query `GRAPH_EXPAND`, graph_retriever, reranker |
| Graph node embeddings | internal DB `graph_node_embeddings` (HNSW) | L4 `graph_embed` → `embed_graph_nodes` | graph seed retrieval, `semantic_linker` |
| Enrichment index | `data/veda_enrichment_index.json` | L4 `enrichment_index` → `build_enrichment_index` | `retrieval/query_enrichment.py` |
| Rerank docs | `data/veda_rerank_docs.json` | L4 `rerank_docs` → `build_rerank_docs` | `query/reranker.py` |
| Relationship graph | `data/veda_relationship_graph.json` | L5 `relationship_graph` → `build_relationship_graph` | query join planner / fast path / graph guard |
| Semantic registry | `semantic/{concepts,dimensions,metrics,MANIFEST}.json` | L5 `semantic_registry` → `compile_semantic_layer.compile_all` | query fast path |
| Value referents | `data/<tenant>/<source>/veda_value_referents.json` | L5 `value_referents` → `write_value_referents` | deterministic planners, typed anchors, Tier-2 qualifier gate |
| HNSW tune | `data/veda_hnsw.json` | L5 `hnsw_tune` (inline) · `l5_publish.py:88-93` | `task_warm_caches` → `SubstrateVersion.hnsw_ef_search` |
| Redis value mirror | Redis `value:{tenant}:{source}:{value_norm}` | L5 `value_mirror` → `mirror_values_to_redis` | query value resolver / arbiter |
| Unified graph | `data/veda_unified_graph.json` | L5 `unified_graph` → `build_unified_graph` | query `GRAPH_EXPAND` |
| Cross-source FK edges | internal DB `graph_edges` (`cross_source_fk` type) | L5 `cross_source_fk` → `discover_and_persist` | query federated join planner + answer composer |
| Doc chunks (embedded) | pgvector `doc_chunks` (1024-dim) | document path → `chunk_embedder.run_chunk_embedder` | query `rag_layer`, graph_retriever |
| Entity graph (docs) | internal DB `graph_nodes`/`graph_edges` (`entity`, `mentions_entity`, `value_of`) | document path → `entity_linker.link_entities` | query cross-source traversal (chunk↔column) |
| Semantic-about edges | internal DB `graph_edges` (`semantic_about`) | document path → `semantic_linker` (Tier A) | query graph expansion (chunk→column, never a join) |
| Entity value embeddings | internal DB `entity_value_embeddings` (HNSW) | `source_dispatcher` step 7d → `value_embedder.embed_source_values` (Tier B) | `semantic_linker` doc-side span match |
| `IngestionJob` / `IngestionStage` | Django DB (`veda`) | `tasks.py:66-72` | admin observability |
| `Sm*` substrate + assembled sm | Django DB + redis-cache | `task_warm_caches` → `writer.warm()` | query tier (canonical runtime sm) |

The **3 gating artifacts**: `veda_semantic_model.json`, `column_embeddings_v2`,
`veda_relationship_graph.json`. The rest are additive — a missing one degrades a specific
query-time signal, not the whole tier.

---

## 6. Encoder / embedding stage

- **Single `BAAI/bge-m3`** — `ingestion/m3_encoder.py`, a process-wide singleton
  (`BGE_MODEL_NAME`, `config.py:1678`). ONE model → 1024-dim dense (columns, tables, graph
  nodes, doc chunks, queries) + learned-sparse lexical weights (replaces BM25).
  `return_colbert_vecs=False` everywhere.
- **No BGE fine-tune stage exists** any more. CLEANUP_PLAN stages 10–11 (synthetic query
  gen, BGE fine-tune → `client_bge`) are removed — `synthetic_query_gen.py` /
  `auto_finetune.py` are gone, `client_bge/` is never loaded. Both tiers use base BGE-M3.
- **`ENCODER_MODE` removed** (WP3, `config.py:6-7`). No code reads it.
- **`EMBEDDING_MODEL_ID = "bge-m3"`** (`config.py:1684`) is the model-change **guard key**:
  `task_ingest_source` stamps it onto `IngestionJob.encoder_mode` (column reused,
  `tasks.py:270`); `_guard_embedding_model_change` (`tasks.py:305`) refuses a run whose id
  differs from the last successful job's unless `force=True` — a changed embedding space
  invalidates every stored vector.
- **Zero-egress:** `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` forced before any model
  import (`main.py:36-37`, `m3_encoder.py:73-74`); weights baked into the image.
- **Metal offload:** if `METAL_EMBED_URL` set, encode calls proxy to
  `scripts/metal_embed_server.py` on the host (device=mps); any transport error →
  in-process CPU fallback (`m3_encoder.py:54-61,106-117`).
- **Passage text strategy:** `EMBED_TEXT_STRATEGY` ∈ {structural, doc, hybrid}
  (`biencoder.py:120-136`); "hybrid" = semantic-model retrieval doc + structural grounding
  tokens, falling back to structural when no retrieval doc exists.

```
                       m3_encoder singleton  (BAAI/bge-m3)
                                │
     ┌──────────────┬───────────┼────────────┬───────────────┬────────────┐
     ▼              ▼           ▼            ▼               ▼            ▼
  biencoder     graph_embedder  chunk_embedder  sparse_index   query encode  (Metal
  cols/tables   graph nodes     doc chunks      learned-sparse  at runtime    offload,
  → *_v2        → node embeds   → doc_chunks    → column_sparse_v1            CPU fallback)
```

---

## 7. Resume logic

- `VEDA_RESUME=1` is set by `_build_subprocess_env` when `_should_resume(job, resume)`
  (`tasks.py:409-417`): explicit `resume=True` **or** a prior FAILED job exists for this
  source (`_should_resume`, `tasks.py:388-394`). Read into `SourceContext.resume` via
  `os.environ.get("VEDA_RESUME") == "1"` (`contracts.py:69`).
- Effect is a **stage-level skip of the two expensive stages**, not true resume-from-N:
  - L3 `semantic_layer`: `if ctx.resume and os.path.exists(SEMANTIC_MODEL_FILE)` → skip
    (`l3_enrich.py:21-24`).
  - L4 `biencoder`: `if ctx.resume and _biencoder_embeddings_exist()` → skip
    (`l4_index.py:54`; `_biencoder_embeddings_exist` = `SELECT 1 FROM column_embeddings_v2
    LIMIT 1`, `l4_index.py:93-105`).
- Fast prep stages (L1/L2) **always re-run** to rebuild the in-memory `state`. The engine
  passes artifacts in-memory between steps, so true resume-from-stage-N would need artifact
  persistence.
- `_table_has_rows` (the name in older docs) no longer exists — replaced by
  `_biencoder_embeddings_exist` + `_should_resume`.

---

## 8. Dead / skeleton code

### Dead (safe-to-delete candidates)

| Item | Evidence |
|---|---|
| `ingestion/chunk_linker.py` | 0 real importers; `link_chunks_to_graph` never called; "replaced and subsumed" by `entity_linker.py` (`main.py:343`, `entity_linker.py:5,300`). Its MiniLM path uses a model no longer loaded anywhere. |
| `ingestion/INGESTION.md` (old text) | Described a retired entrypoint (`veda_ingestion.py`) + pre-P7 architecture — now rewritten as a pointer doc. |
| `_MARKER_RE` / `_ENGINE_STEP_TO_STAGE` / `_apply_step_marker` in `apps/ingestion/tasks.py` | the `[N/NN]` monolith progress protocol; layered path emits only `[[STAGE]]`. Kept as unexercised fallback. |
| `ingestion_mode` param on `task_ingest_source` | accepted for signature compat, does nothing (`tasks.py:196-198`). The monolith body was deleted outright; no `INGESTION_MODE=legacy` escape hatch. |
| `schema/simulate_schema.py` | fallback-only (`config.py`, `reg_builder.py:212`, `schema_scanner.py:22`, `value_sampler.py:732`, `data_graph.py:775`); CLEANUP_PLAN Track 1 #4 marks for removal after cutting those branches. |
| `main.run_ingestion` docstring refs to a "12-step pipeline"; `_step`/`_ok`/`_fail` monolith helpers | `_step` only used by eval + build-glossary now. |
| `column_text.py` header refs to `relgt_encoder.py` | encoder removed; the function is still used for structural text. |

### Skeleton / partially wired

| Item | State |
|---|---|
| `artifact_scope` / per-(tenant,source) artifact isolation | plumbed through `SourceContext`, `config.artifact_scope()`, `artifact_path()`; **OFF by default** (`VEDA_ARTIFACT_SCOPING != "1"`, `tasks.py:400`). N relational sources still overwrite `data/veda_*.json`. Tabular sources **do** scope parquet under `ARTIFACT_ROOT/<source_id>/tables/`. |
| L5 "atomic activate / SubstrateVersion flip" | `SubstrateVersion.hnsw_ef_search` is written, but query-tier gating is still `Source.ready`, not a per-artifact version pointer. |
| `doc_parser.py` "tabular lane" for derived doc tables | `DocTable.is_derived_table()` / `ParsedDoc.derived_tables()` exist; **no caller anywhere**. |
| `value_embedder.py` (Tier B) in the relational path | only runs in `source_dispatcher._run_schema_pipeline` step 7d — **not invoked by the layered L1–L5**. For a normal relational ingest, `entity_value_embeddings` is not populated. |
| Celery per-stage task chain (CLEANUP_PLAN §2.2) | not implemented — one `task_ingest_source` running a subprocess. `STAGE_ORDER` rows are observability only; no per-stage re-enqueue/retry. |

---

## 9. File inventory

Top-level `veda_core/ingestion/*.py` → see
[`veda_core/ingestion/INGESTION.md`](../veda_core/ingestion/INGESTION.md) (compact table)
and [`veda_core/ingestion/AGENTS.md`](../veda_core/ingestion/AGENTS.md). Layer modules →
[`veda_core/ingestion/layers/AGENTS.md`](../veda_core/ingestion/layers/AGENTS.md) and the
per-layer contracts. Connectors →
[`veda_core/connectors/AGENTS.md`](../veda_core/connectors/AGENTS.md).
