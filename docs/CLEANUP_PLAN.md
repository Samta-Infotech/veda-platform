# VEDA Platform — Cleanup, Layered Ingestion & Query-Latency Plan

> **Status:** plan only — no code changed. Based on a direct read of the full repo (`veda_core/`, `apps/`, `inference/`, `storage_adapters/`, `config/`, `docker/`) plus the existing `ARCHITECTURE.md` and `INGESTION_CLEANUP_PLAN.md`.
> **Relationship to `INGESTION_CLEANUP_PLAN.md`:** that document's dead-code inventory (§3, §4, §4b, §4d) is **adopted verbatim** as Track 1 here. This plan supersedes its §5 (API routing) with a fuller layered-architecture design, and adds two tracks it doesn't cover: production-config elimination and query-latency precompute.
> **Non-negotiables carried forward:** both query tiers stay (Tier 1 deterministic head + Tier 2 LLM-IR fallback); the firewall gates are frozen; nothing either tier reads gets removed; the architecture is improved, never degraded.

---

## 0. Ground truth — how both flows work today (verified)

### 0.1 Ingestion flow (as-is)

```
POST /api/v1/admin/ingest {source_id}          (staff, DRF)
  └─ Celery: task_ingest_source(source_id, tenant, force, skip_llm, resume)
       ├─ IngestionJob + IngestionStage rows (observability)
       ├─ ENCODER_MODE guard (refuse silent mode change)
       ├─ env injection: Source.as_engine_env() → VEDA_SOURCE_*
       └─ SUBPROCESS  python -c "import main; main.run_ingestion(...)"   cwd=veda_core
            [1]  Schema Scanner (real schema, live introspection)
            [2]  FK Adjacency Store            → fk_adjacency
            [3]  Data Graph (undeclared FK)    → merged into fk_adjacency
            [4]  Semantic Type Inference       → in-mem (+ persisted table, unread)
            [5]  Table Metadata Store          → table_metadata
            [6]  Value Sampler                 → column_values
            [7]  REG Builder                   → reg_graph.pkl, col_id_to_idx.pkl, kuzu
            [7b] Graph Persist                 → graph_nodes / graph_edges
            [7c] Graph Embedder (BGE)          → graph_node_embeddings
            [7d] GNN                           → DEAD (fn doesn't exist, import always fails)
            [8]  Encoder (ensemble TF-IDF/SVD) → pkls
            [9]  Vector Store                  → column_embeddings_lt / _hybrid
            [9b] Semantic Layer v2 (Qwen LLM)  → veda_semantic_model.json + synonyms + concepts
            [—]  BGE Biencoder                 → column_embeddings_v2 / table_embeddings_v2
            [10] Synthetic Query Gen (LLM)     → training_pairs.jsonl   (feeds only 11)
            [11] BGE Fine-Tune                 → client_bge             (NEVER LOADED)
            [12] Derived artifacts             → relationship_graph.json + semantic/*.json
       └─ on success: task_warm_caches → writer.warm()
            → assembler.persist (Sm* substrate) + sync_from_engine + publish_sm + publish_rehydrate
            → Source.ready = True
```

Structural problems (beyond dead stages):

| # | Problem | Evidence |
|---|---------|----------|
| I-1 | **Monolithic orchestrator.** All stage logic lives inside one 1,182-line `main.run_ingestion` with print-marker progress; the Celery ten-task chain is a `NotImplementedError` skeleton. Stages aren't importable, testable, or independently retryable. | `veda_core/main.py:293-735`, `apps/ingestion/tasks.py:44-94` |
| I-2 | **Source-type blindness.** The task always runs the primary-relational pipeline; `source.type` is ignored; the type-aware `dispatch_ingestion` router exists but is never called by the API path. | `tasks.py:202`, `source_dispatcher.py` |
| I-3 | **Two sources of truth for sources.** `config.VEDA_SOURCES` (hardcoded, client-specific) vs the `Source` DB table. They drift. | `config.py:40-196` |
| I-4 | **Global, single-source artifacts.** Every derived file is one fixed path (`data/veda_semantic_model.json`, `veda_relationship_graph.json`, `veda_unified_graph.json`, glossary, synonyms, pkls). A second source overwrites the first. This is the hard blocker for multi-source. | `config.py:1292-1303` |
| I-5 | **Query-time inputs not guaranteed by ingestion.** Glossary (`query_enrichment.py:88`) and unified graph (`query_graph.py:39`) are read at query time but built only by side CLIs, never by `run_ingestion`. | `main.py:652`, `unified_graph_builder.py` |
| I-6 | **Subprocess coupling.** Progress tracking depends on regex-parsing stdout `[N/NN]` markers; failure granularity and resume are approximations (`_table_has_rows`). | `tasks.py`, `main.py:318` |

### 0.2 Query flow (as-is)

```
POST /api/v1/query {query}                       (Django, thin — no veda_core import)
  → resolve tenant (server-side) + source_id → InferenceClient (HTTP, headers)
  → inference (FastAPI, warm engine per worker) → run_hybrid_query(query)
       classify (keyword router; 'sql' safe default)
       ├─ sql → pipeline.run_query  (Tier 1, deterministic head)
       │    L1 temporal → L4 intent → existence-mode
       │    FAST PATH   (compiled registries — no retrieval, no LLM)          ~ms
       │    VERIFIED CACHE (pgvector cosine ≥ 0.85 when ctx set)              ~ms
       │    L2+ enhance → L2 retrieve (enrich → BGE encode → ANN + BM25
       │        + FK/subgraph signals → RRF → intent boost → cutoff)          ~100–300ms
       │    L2g graph expand + L2b cross-encoder rerank                        ~100–500ms
       │    L3 anchor → branch:
       │        multi-table/aggregate/existence → deterministic planner (LLM fills SELECT/WHERE only)
       │        single-table ladder: answer-entity → FK-value → value-arbiter
       │           → temporal-only → temporal-refuse → LLM generate_sql       LLM = seconds
       │    FIREWALL: value-grounding → qualifier-completeness → IR-equivalence
       │              → AST validate+parameterize+graph-guard
       │    L7 execute (read-only, 30s timeout, fetch ≤20)
       │    L7b NL answer (SLM call)                                          ~1–3s
       │    cache-back → save_verified_query
       ├─ refusal + TIER2_LLM_FALLBACK → _tier2_sql (LLM-IR → sql_builder → same firewall)
       ├─ rag / hybrid / nosql → respective layers
  → QueryLog audit (status, sql placeholder text, latency, cache_hit, request_id)
```

Where query-time work is done that could be done (or bought) at ingestion:

| # | Hot-path cost | Where | Precompute opportunity |
|---|---------------|-------|------------------------|
| Q-1 | **Live source-DB schema introspection at engine warm** — `SignalBuilder.build_signals` calls `get_real_schema()` (information_schema on the *client's* DB) on every process start. | `signal_builder.py:48` | Read FK/adjacency from the substrate (`FkEdge`, `SchemaTable/Column`) written at ingestion. Removes the source-DB dependency from the query tier entirely. |
| Q-2 | **BM25 corpus fit at engine warm** — `bm25_ranker.fit(semantic_model)` rebuilds tokenization + IDF per process. | `retrieval_engine_phase3.py:153` | Precompute the BM25 index (token→postings + IDF) at ingestion; persist; load ready-made. |
| Q-3 | **Enrichment lexicons parsed from JSON files at warm**; glossary/unified-graph may be stale/missing (I-5). | `query_enrichment.py:68-101` | Build a single pre-tokenized, pre-inverted enrichment index at ingestion; serve from substrate/Redis. |
| Q-4 | **Reranker documents assembled per query** — `_table_text`/`_col_text` stitch sm + sampled values per candidate on every query; plus a `SELECT name FROM graph_nodes` DB hit inside `_get_reranker` init. | `reranker.py:63-128` | Precompute the rerank document text per column/table at ingestion (`SmRetrievalDoc` already exists — extend it); cache graph-node names in the assembled sm. |
| Q-5 | **Value-resolver / arbiter DB round trips per query** against `column_values`. Already batched to one query, but still a Postgres round trip per user query. | `value_resolver.py:178-196`, `value_arbiter.py:317` | Mirror hot value_norm→(table,col,raw) into Redis at ingestion (the docstring already claims a Redis SET mirror — actually implement it) with Postgres as fallback. |
| Q-6 | **Fast-path coverage is narrow** — only what the compiled registries (`semantic/*.json`) cover skips retrieval+LLM. | `fast_path.py` | Ingestion-time expansion of compiled metrics/dimensions/concepts (Step 12 already exists — widen what `compile_semantic_layer` emits, incl. per-table count/aggregate templates and display-column answers). |
| Q-7 | **NL answer = 1 SLM call per answered query**, even for trivial shapes (single count, single row, empty set). | `nl_answer.py` | Deterministic answer templates for canonical result shapes at ingestion (per-table phrase from table_metadata); SLM only for genuinely narrative results. |
| Q-8 | **Verified-cache file fallback re-encodes** and grows unbounded when no ctx; pgvector path is per-lookup encode + ANN (fine), but there's no exact-hash short-circuit. | `veda/cache.py` | Add `query_hash` exact-match lookup (already a unique key on the substrate table) *before* the cosine ANN; embed once, store. |
| Q-9 | **Join skeletons planned per query** from the relationship graph. Cheap-ish, but for the top table-pairs it's fully static. | `join_planner.py` | Precompute canonical join paths (pairwise shortest FK paths + key pairs + fan-out direction) at ingestion into the substrate; planner consults the precompiled map first. |
| Q-10 | **Semantic model file fallback** — inference prefers Redis sm but falls back to the on-disk file; global path again. | `veda_hybrid._load_semantic_model` | After Track 3, Redis/substrate becomes the only path; file fallback removed with the global paths. |

---

## Track 1 — Dead code removal (adopt existing plan, plus delta)

Adopt `INGESTION_CLEANUP_PLAN.md` §3/§4/§4b/§4c/§4d wholesale. Summary of what goes:

- **Stages:** 7d GNN (function doesn't exist), 10 Synthetic Query Gen + 11 BGE Fine-Tune (output never loaded — both tiers use base BGE), `column_profile` writes, persisted semantic-type table (after the §6.3 reader check).
- **Files (zero importers):** `query/audit_logger.py`, `query/executor.py`, `query/sql_generator.py`, `query/sql_validator.py`, `veda/consensus.py`, `veda/ir_emit.py`, `ingestion/semantic_postprocessor.py`, plus `synthetic_query_gen.py`, `auto_finetune.py`, and the legacy CLI (`_run_single_query_legacy`, `--legacy-query`, `query/execution_engine.py`).
- **Config:** the 69 orphaned `config.py` keys, deleted in the same commit as their owning subsystem.
- **Artifacts:** `client_bge/`, `client_minilm/`, `training_pairs.jsonl`, `veda_semantic_checkpoint.json` (once resume moves to stage-level state — Track 2), DB table `graph_node_embeddings_gnn`.
- **Keep (both tiers depend):** ensemble encoder + `_lt`/`_hybrid` tables + TF-IDF/SVD/REG pkls, graph persist/embed tables, `table_metadata`/`table_embeddings_v2`, and every Tier-2 query module.

**Delta added by this plan:**

1. `apps/ingestion/tasks.py` skeleton chain — **do not delete**; it becomes the real orchestrator in Track 2 (the existing plan's Phase 4 said "delete or implement"; the decision here is implement).
2. Untracked working-tree artifacts (`parity_baseline.json`, trace logs, pickled encoders, `.omc/` state dirs) — add to `.gitignore`; derived artifacts never live in git.
3. `docker/reingest_chain.sh`, `scripts/fresh_homzhub.sh` — client-named one-off scripts; fold their generic parts into `onboard_source.sh` and delete.
4. `veda_core/schema/simulate_schema.py` — remove after cutting the fallback branches (most of its importers die in Track 1 anyway).

**Acceptance gate:** parity suite + eval run before and after → identical Tier-1 and Tier-2 metrics, measurably faster ingestion (two LLM/training stages gone).

---

## Track 2 — Layered ingestion architecture

### 2.1 Target layer model

Reshape the 15-stage monolith into **five layers with explicit contracts**, each a package of importable, individually testable stage functions. Every stage takes a `SourceContext` and a typed input, returns a typed output, and persists only through `storage_adapters.writer` — no stage opens its own connection or invents a path.

```
veda_core/ingestion/
├── contracts.py          # SourceContext(source_id, tenant, type, engine, conn, artifact_scope)
│                         # + typed stage I/O dataclasses (ScanResult, TypedSchema, …)
├── layers/
│   ├── l1_extract/       # EXTRACT  — touch the source, nothing else
│   │   ├── schema_scan.py        (relational/nosql/datalake via connectors)
│   │   ├── fk_discovery.py       (declared FK + data-graph undeclared FK, merged here)
│   │   └── value_sampling.py
│   ├── l2_analyze/       # ANALYZE — pure functions over extracted data, no LLM, no DB writes
│   │   ├── semantic_types.py
│   │   ├── table_metadata.py     (display columns)
│   │   ├── reg_graph.py          (REG build — in-mem graph)
│   │   └── join_paths.py         (NEW — precompiled pairwise join skeletons, Q-9)
│   ├── l3_enrich/        # ENRICH  — LLM / model-powered derivation (the expensive layer)
│   │   ├── semantic_layer.py     (Qwen: model + synonyms + concept graph)
│   │   ├── glossary.py           (NEW in pipeline — wires domain_glossary, I-5)
│   │   └── unified_graph.py      (NEW in pipeline — wires unified_graph_builder, I-5)
│   ├── l4_index/         # INDEX   — embeddings + search structures
│   │   ├── bge_embed.py          (column/table embeddings_v2)
│   │   ├── ensemble_encode.py    (TF-IDF/SVD + _lt/_hybrid — Tier-2 signal)
│   │   ├── graph_embed.py        (graph persist + graph_node_embeddings)
│   │   ├── bm25_index.py         (NEW — persisted BM25 postings/IDF, Q-2)
│   │   ├── enrichment_index.py   (NEW — pre-tokenized synonym/concept/glossary inverted index, Q-3)
│   │   └── rerank_docs.py        (NEW — precomputed rerank text per column/table, Q-4)
│   └── l5_publish/       # PUBLISH — atomic activation
│       ├── compile_registries.py (relationship graph + semantic registry + fast-path expansion, Q-6)
│       ├── substrate_sync.py     (assembler.persist + sync_from_engine, versioned)
│       └── activate.py           (SubstrateVersion flip + Redis sm + value mirror + rehydrate)
└── dispatcher.py         # type router: relational | document | nosql | datalake → layer plan
```

Layer rules (these are what make it "structured and divided"):

- **L1 is the only layer that touches the tenant's source.** After L1 completes, ingestion can finish even if the source goes down.
- **L2 is pure** — deterministic transforms, unit-testable with fixtures, no network.
- **L3 is the only LLM layer** — `skip_llm` skips exactly L3; everything else still produces a queryable (if less enriched) substrate.
- **L4 is the only model-inference layer** (BGE/MiniLM) — the GPU/MPS-bound cost is isolated and parallelizable per table batch.
- **L5 is atomic** — everything is written under a new `SubstrateVersion`; `activate` flips the version pointer and publishes rehydrate only when the full set is consistent. The query tier never sees a half-built substrate (today this is only guaranteed by `Source.ready`; versioning makes it guaranteed per artifact).

### 2.2 Orchestration: implement the Celery chain

Replace the subprocess + stdout-regex mechanism (I-1, I-6) with the already-modeled chain in `apps/ingestion/tasks.py`:

- Each `STAGE_ORDER` task body becomes a thin call into the corresponding layer function, wrapped in `TenantTask` so context propagates.
- `IngestionStage` rows update from real task lifecycle (no marker parsing). Failure is per-stage; **resume = re-enqueue from the first non-SUCCESS stage** using stage-level state, retiring `VEDA_RESUME` + `_table_has_rows` heuristics and the checkpoint file.
- Heavy L3/L4 stages stay on the `ingestion` queue (ML image); L1/L2/L5 can run on the thin image — the current all-or-nothing "run everything on the inference image" constraint relaxes.
- Keep the subprocess *only* as an escape hatch for one release (`INGESTION_MODE=legacy` env) to allow parity comparison, then delete `run_ingestion`'s orchestration body (stage functions survive as the layer modules).

### 2.3 Migration is a move, not a rewrite

Every layer module above is a **hoist of existing code**: `schema_scanner` → `l1_extract/schema_scan`, `semantic_layer_v2` → `l3_enrich/semantic_layer`, etc. Logic is preserved (that's the same principle `tasks.py` already documents); only orchestration, I/O boundaries, and path resolution change. The genuinely new modules are the six marked NEW, and four of those are precompute additions from Track 4.

---

## Track 3 — API-driven per-tenant/source ingestion + config elimination

### 3.1 Kill the config-file source registry

`config.VEDA_SOURCES` goes away entirely. The `Source` row becomes the single source of truth:

| Today in `config.py` | Target |
|---|---|
| `VEDA_SOURCES` list (hardcoded launchpad DB, dead `dmt` doc source, commented examples) | **Delete.** `Source` table only. `get_source`/`get_enabled_sources`/`get_primary_relational_source` are replaced by the dispatcher receiving a serialized `SourceContext`. |
| `exclude_tables` — 60 OCS/client-specific table names baked into the repo | **New `Source.exclude_tables` (JSONField)** + a small built-in framework-noise default set (django_*, celery_*) applied by the scanner. Client specifics live in the client's row, never in code. |
| Default credentials (`password: "admin"`, `user: postgres`, localhost ports) | **Delete defaults.** Connection fields already exist on `Source` with `password_env`/secret-ref; missing env = hard fail at task start, never a silent localhost fallback. |
| `VEDA_INTERNAL_DB` env-with-defaults | Keep env-driven, **remove the insecure defaults**; fail fast if unset in prod settings. |
| Global artifact paths (`SEMANTIC_MODEL_FILE`, `RELATIONSHIP_GRAPH_FILE`, `UNIFIED_GRAPH_FILE`, glossary/synonyms/concepts, pkls) | **Replace with `artifact_scope` resolution**: every artifact is keyed `(tenant, source_id, substrate_version)`. DB/pgvector artifacts already carry source+tenant columns; file/pkl artifacts move to substrate rows (preferred: relationship graph, unified graph, glossary, synonyms, concept graph, semantic registry → JSON columns on versioned substrate models; sm already has `Sm*`) or, where a file is unavoidable (TF-IDF/SVD pkls, kuzu), to `ARTIFACT_ROOT/<tenant>/<source_id>/<version>/`. This unblocks I-4: N sources coexist. |
| 367 top-level tuning constants | After the 69 dead ones are deleted (Track 1), the live engine flags **stay in `config.py` as the single defaults file** but every value becomes env-overridable through the existing `settings_bridge` pattern (extend the bridge whitelist to all live flags). No client-, path-, or credential-shaped value remains in code. |

### 3.2 The API contract

`POST /api/v1/admin/ingest {source_id, force?, skip_llm?, resume?}` (existing endpoint, kept):

1. View resolves tenant server-side (as today), loads the `Source` row, serializes a `SourceContext` JSON — id, tenant, **type**, engine, connection (secret by reference, resolved inside the worker), exclude_tables, schema filter.
2. `task_ingest_source` passes the context into `dispatcher.dispatch(source_context)`:
   - `relational` → full L1–L5 plan
   - `document` → doc plan (chunk → embed → publish), reusing `run_doc_ingestion` logic hoisted into layers
   - `nosql` / `datalake` → schema-pipeline plan via their connectors
   This fixes I-2/I-3: `_dispatch_relational` honours the passed source instead of re-deriving "primary"; `source_dispatcher`'s copies of the dead stages (synthetic gen + fine-tune in `_run_schema_pipeline`) are removed with Track 1.
3. Job/stage tracking, ENCODER_MODE guard, and `Source.ready`-on-full-success semantics are preserved exactly.
4. Query side: `X-Veda-Source-Id` already flows end-to-end; with per-source artifact scoping, the inference tier loads the sm for `(source, tenant)` from Redis keyed by scope — multiple ready sources are queryable concurrently.

---

## Track 4 — Query-flow latency: shift work into ingestion

Ordered by expected impact. Every item preserves the escalation ladder and firewall unchanged — this only moves *preparation* earlier and widens the deterministic (LLM-free) coverage.

### 4.1 Eliminate LLM/SLM calls where the answer is deterministic (largest wins — seconds each)

| Change | Mechanism | Saves |
|---|---|---|
| **Widen the fast path (Q-6).** During L5 `compile_registries`, precompile per-table count/aggregate SQL templates, display-column "who/what is X" answers, and per-metric canonical queries into the semantic registry. | Fast path already skips retrieval + rerank + LLM entirely; this increases its hit-rate. Measure hit-rate before/after on the eval set. | Whole pipeline (~2–6 s → ~50 ms) per newly covered query |
| **Deterministic NL answers (Q-7).** Ingest-time per-table answer phrase (from `table_metadata` display metadata); at L7b, template the answer for canonical shapes: single scalar, single row, empty set, count. SLM only for multi-row narrative results. | `nl_answer.py` gains a shape check before the SLM call; row-count fallback already exists — promote templates to the primary path for those shapes. | ~1–3 s on a large share of answered queries |
| **Exact-hash verified-cache short-circuit (Q-8).** `query_hash` unique key already exists on `VerifiedQueryCache`; look it up *before* embedding the query for cosine ANN. | One indexed PK lookup vs BGE encode + ANN. Keep the cosine path as the second step. | ~30–100 ms on repeat queries; also removes the file-fallback's unbounded re-encode entirely (file store deleted with Track 3 path removal) |

### 4.2 Move warm-up work from query tier to ingestion (fixes cold start + removes source-DB coupling)

| Change | Mechanism | Saves |
|---|---|---|
| **Substrate-backed signals (Q-1).** `SignalBuilder` reads FK graph + adjacency from `FkEdge`/`SchemaTable` (written at L1) instead of live `information_schema` on the client DB. | New reader in `storage_adapters.reader`; signal maps are also persisted at L2 so warm-up is a load, not a rebuild. **The query tier no longer needs the source DB reachable except at L7 execute.** | Seconds off every process warm; removes a whole failure mode |
| **Persisted BM25 index (Q-2).** L4 `bm25_index` stores tokenized corpus + IDF per `(source, version)`; engine warm loads it. | `bm25_ranker` gains `load(index)`; `fit()` survives only inside ingestion. | Warm-up time; identical scores |
| **Enrichment index (Q-3).** L4 builds one merged, pre-tokenized inverted index (synonyms + concepts + glossary + literal vocab) as a substrate artifact; `QueryEnricher` loads it instead of parsing four JSON files — and the glossary is now *guaranteed fresh* because L3 builds it (I-5 fix). | Also fixes correctness: today expansion silently degrades when glossary/unified-graph files are stale or missing. | Warm-up + a few ms/query + recall correctness |

### 4.3 Trim per-query round trips and rebuild work

| Change | Mechanism | Saves |
|---|---|---|
| **Precomputed rerank docs (Q-4).** L4 `rerank_docs` materializes the exact `_table_text`/`_col_text` strings per candidate into `SmRetrievalDoc` (extend the existing model with a `rerank_text` field); reranker reads them from the assembled sm. Graph-node name set is embedded in the assembled sm, killing the `SELECT name FROM graph_nodes` at reranker init. | Cross-encoder scoring itself stays (it needs the live query) — only document assembly is precomputed. | ~10–50 ms/query + one DB hit at init |
| **Redis value mirror (Q-5).** L5 `activate` mirrors `column_values` into Redis hashes (`value:{scope}:{value_norm}` → JSON list of (table, col, raw)); `value_resolver`/`value_arbiter` check Redis first, Postgres fallback. The substrate docstring already promises this mirror — implement it. | Sub-ms lookup vs a Postgres round trip on nearly every non-fast-path query. | ~5–20 ms/query, less PgBouncer pressure |
| **Precompiled join paths (Q-9).** L2 `join_paths` stores pairwise shortest FK paths (key pairs + fan-out direction) for all table pairs within hop-limit; `join_planner` consults the map before graph traversal. | Deterministic, versioned with the schema; planner logic unchanged for unmapped pairs. | ~5–30 ms on multi-table queries; also more stable plans |
| **HNSW `ef_search` per source (existing knob).** With versioned per-source embeddings, tune `VEDA_HNSW_EF_SEARCH` per source size at L5 and store on the `SubstrateVersion`, instead of one global value. | The `SET LOCAL` mechanism already exists in `reader.ann_search`. | ANN recall/latency balance per source |

### 4.4 What deliberately does *not* move

- **BGE query-encode, cross-encoder scoring, LLM SQL generation, firewall gates, L7 execute** — these need the live query and are the correctness core. Untouched.
- **Tier-2's retrieval spine and the ensemble artifacts** — kept per the locked directive; the "BGE-only Tier-2" trim stays a flagged future option, not part of this plan.

**Expected profile after Track 4** (to be validated with the parity suite + latency histograms from `QueryLog.latency_ms`):
- fast-path / cache-hit queries: **~50–150 ms** end-to-end
- deterministic-branch answered queries (no LLM SQL, templated NL): **~300–800 ms**
- LLM-branch queries: dominated by the SLM as today, minus ~100–200 ms of shaved retrieval/rerank/value overhead — the remaining lever there is the already-recommended vLLM backend (out of scope here, seam exists in `_call_slm`).

---

## Track 5 — Execution order (each phase independently shippable)

| Phase | Scope | Gate before merging |
|---|---|---|
| **P0 — Baseline** | Run `parity_suite.py` + eval; capture latency histograms per terminal status from `QueryLog`; confirm the `INGESTION_CLEANUP_PLAN.md` §6 verifications (semantic-type table reader, doc-ingestion usage, simulate_schema fallbacks). | Baseline artifacts committed |
| **P1 — Dead code (Track 1)** | All Tier-A removals + dead modules + 69 config keys + gitignore/scripts delta. No behavior change intended. | Eval metrics identical for both tiers; ingestion wall-clock reduced |
| **P2 — Guaranteed query inputs** | Wire glossary + unified-graph builds into ingestion (pre-layering, at their §1a insertion points) so query correctness stops depending on side CLIs. | Both files fresh post-ingest; enrichment/expansion verified live |
| **P3 — Config elimination (Track 3.1)** | Delete `VEDA_SOURCES` + exclude-list → `Source.exclude_tables`; remove credential defaults; introduce `artifact_scope` path/substrate resolution (single-source still fine at this point). | Fresh onboarding via API works with zero code edits; repo greps clean for client names, paths, credentials |
| **P4 — Layered ingestion (Track 2)** | Hoist stages into `layers/`, implement the Celery chain, stage-level resume, versioned L5 publish with atomic activate. Legacy subprocess behind `INGESTION_MODE=legacy` for one release. | Chain-run vs legacy-run substrate parity (row counts + sm diff); per-stage retry demonstrated |
| **P5 — Type-aware multi-source (Track 3.2)** | Dispatcher routing by `source.type`; per-source scoped artifacts live; second relational + one document source onboarded side-by-side. | Two ready sources queryable concurrently; no artifact cross-talk |
| **P6 — Precompute wave 1 (Track 4.1–4.2)** | Fast-path expansion, NL templates, exact-hash cache, substrate-backed signals, persisted BM25, enrichment index. | Latency histograms vs P0; fast-path hit-rate up; answers byte-identical on eval where deterministic |
| **P7 — Precompute wave 2 (Track 4.3)** | Rerank docs, Redis value mirror, precompiled join paths, per-source HNSW. Delete legacy subprocess + `run_ingestion` orchestration body + remaining file fallbacks (`_SM` file path, verified-query file store). | Final parity + latency report; single ingestion path remains |

**Standing invariants checked at every phase:** firewall gates unchanged and asserted on the original query; `Source.ready`/version-flip only on full success; tenant context fail-closed; zero-egress preserved (glossary/semantic-layer LLM calls stay on the internal SLM); `QueryLog` audit fields intact.

---

## Appendix — Target ingestion (relational source, post-plan)

```
API → dispatcher(SourceContext{tenant, source, type=relational})
 L1 EXTRACT   schema scan → FK discovery (declared + data-graph) → value sampling
 L2 ANALYZE   semantic types → table metadata → REG graph → join-path precompile
 L3 ENRICH    semantic layer v2 (Qwen) → domain glossary → unified graph          [skip_llm skips exactly this]
 L4 INDEX     BGE embed → ensemble encode → graph embed → BM25 index
              → enrichment index → rerank docs
 L5 PUBLISH   compile registries (+fast-path expansion) → substrate sync (new version)
              → activate: version flip + Redis sm + Redis value mirror + rehydrate fan-out
```

Removed vs today: GNN, synthetic gen, fine-tune, profiler, legacy CLI, dead modules, 69 config keys, `VEDA_SOURCES` + client exclude list + credential defaults, global artifact paths, subprocess orchestration, checkpoint-file resume.
Added: layer contracts, versioned atomic publish, per-(tenant, source) scoping, six precompute artifacts, deterministic NL templates, exact-hash cache, stage-level Celery orchestration.
Unchanged: both query tiers, the escalation ladder, all four firewall gates, read-only parameterized execution, tenancy fail-closed model.