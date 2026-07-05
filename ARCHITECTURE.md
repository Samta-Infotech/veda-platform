# ARCHITECTURE.md — VEDA Technical Architecture

**System**: VEDA — Natural Language → Query engine (SQL-first, multi-modal)
**Shape**: On-premise / zero-egress. All inference local (Ollama + local BGE/MiniLM). No client data leaves the server.
**Last updated**: 2026-07-04 · regenerated from the live codebase + code-review knowledge graph (141 files, 1,556 nodes, 18,358 edges, 230 detected flows, 21 communities).

> This document supersedes the earlier "L1–L4 built, L5–L7 planned" description. The engine has since grown into the **hybrid architecture**: a deterministic SQL head with a full validate-execute-audit firewall, plus RAG / hybrid / NoSQL heads behind a router, plus a phase-3 5-signal retrieval spine and a unified knowledge graph. The canonical target design lives in [ARCHITECTURE_HYBRID.md](ARCHITECTURE_HYBRID.md); this file documents **what is actually wired today** and every execution flow.

---

## 0. Thesis

> **Neither reasoning engine writes SQL structure.** Reasoning produces *intent*; a deterministic compiler produces *structure*; one firewall proves every result before it executes. Everything schema-specific is **derived at ingestion**, never hardcoded.

Three concrete invariants fall out of this and never bend:

- **Read-only, AST-enforced.** No INSERT/UPDATE/DELETE/DDL ever reaches the DB.
- **Parameterized only.** Every literal is bound; no f-string/`%`-formatted values into SQL.
- **Refuse-over-guess.** Ambiguity (two FK paths, an ungrounded value, a dropped qualifier) becomes a clean refusal with a reason — never a silent wrong answer.

---

## 1. Two phases + a multi-head front door

```
                          ┌──────────────────────────────────────────────┐
   INGESTION (per DB) ───►│  Grounding substrate:                        │
                          │   FK graph · embeddings · semantic metadata  │
                          │   · glossary/synonyms · value samples        │
                          │   · unified knowledge graph · doc chunks     │
                          └──────────────────────────────────────────────┘
                                             │  (populated stores)
                                             ▼
   NL query ──►  veda_hybrid.run_hybrid_query   (THE FRONT DOOR)
                       │
                       ├─ decompose?  (compound → N independent sub-queries)
                       │
                       └─ route per sub-query (query/query_router.py):
                            ├─ sql    → veda/pipeline.run_query      [deterministic, CORRECTNESS]
                            ├─ rag    → query/rag_layer.run_rag_layer      [doc retrieval + LLM]
                            ├─ hybrid → query/rag_layer.run_hybrid_layer   [SQL signals + docs, RRF]
                            └─ nosql  → query/nosql_builder.run_nosql_builder [native Mongo/etc.]
                       │
                       ▼
                 MultiResult  (always — 1 item for a plain query, N for a compound)
```

- **`veda_hybrid.py` — `run_hybrid_query(query)`** is the single public entry point. It **always** returns a `MultiResult` (`query/multi_result.py`). Callers branch on `MultiResult`, never on "is this compound", so everything downstream stays single-intent-simple.
- **Compound handling** (`QUERY_DECOMPOSE_ENABLED`): the deterministic SQL head *self-certifies completeness* (its `qualifier_completeness` gate proves a clean answer covers the whole utterance), so a clean SQL answer skips the decomposer entirely — zero added latency on the hot path. A non-deterministic head (RAG/hybrid/NoSQL) cannot self-certify, so those decompose **first** (silent-drop guard). A deterministic *refusal* also triggers decomposition (maybe it was several questions).
- **Routing default is `sql`** — the deterministic head is the safe default whenever the router is off or unavailable.

---

## 2. Repository map — every package, every file

### Root
| File | Purpose |
|------|---------|
| `veda_hybrid.py` | **Front door** — `run_hybrid_query`, router dispatch, decompose/fan-out, `MultiResult` assembly |
| `main.py` | Orchestrator CLI — ingestion pipeline, evaluation pipeline, single-query smoke test, `--report-only` |
| `config.py` | **All** parameters, feature flags, model names, thresholds, DB config (single source of truth) |
| `run_hybrid_suite.py` | Drives the full eval suite through `veda_hybrid` |
| `benchmark_100_queries.py`, `analyze_benchmark.py`, `_sweep_probe.py` | Benchmark harnesses + LoRA sweep probe |
| `mlflow_orchestrator.py` | MLflow experiment tracking wrapper |
| `veda_hybrid.py` / `run_hybrid_suite.py` | (see above) |

### `veda/` — the deterministic engine + firewall (L1→L9)
| File | Role (per ARCHITECTURE_HYBRID.md layer) |
|------|------|
| `pipeline.py` | **`run_query`** — the L1→L7 orchestrator (subject of §5) |
| `runtime.py` | Shared warm resources: DB pool, BGE-M3, retrieval engine, FK graph handles + constants |
| `routing.py` | L2 anchor resolution — `select_primary_table` + `vet_primary` (margin-based confidence) |
| `planning.py` | L4b/L4c — `existence_mode`, `aggregate_mode`, `try_multitable`, deterministic pre-aggregation & join orchestration |
| `generation.py` | L5 — LLM SQL generation (single-table + join-skeleton fill only; never authors joins) |
| `compiler.py` | Graph SQL compiler — join inference from the FK graph |
| `consensus.py` | L4 Consensus engine — field-weighted IR reconciliation (target/partial) |
| `verifier.py` | L5 Verifier — substrate-grounded ambiguity check (target/partial) |
| `ir_emit.py`, `ir_validator.py`, `ir_equivalence.py` | Canonical IR v2 emit / validate / equivalence-gate |
| `validation.py` | L6 firewall — value grounding, qualifier-completeness, AST validate + parameterize, fan-out |
| `graph_guard.py` | Firewall — every join is a real FK edge; no cartesian |
| `execution.py` | L7 — read-only session, `statement_timeout`, bounded fetch |
| `cache.py` | Verified-query cache (file-based, cosine ≥ 0.85) |
| `query_enhancement.py` | Recall-only search enrichment (singularization + term expansion) |
| `explain.py` | Full `EXPLAIN` trace (`new_trace`) — the debugging spine |
| `feedback.py` | Turns a refusal/error into actionable guidance |
| `cli.py` | Single-shot + REPL entry point |

### `query/` — layer plumbing, heads, and legacy pipeline
| File | Role |
|------|------|
| `temporal_parser.py` | L1 — date expression → ISO range (`run_temporal_parser`) |
| `query_router.py` | Modality classifier — sql / rag / hybrid / nosql |
| `intent.py`, `intent_envelope.py`, `envelope_slm.py` | Intent detection (`IntentDetector`) + intent-envelope SLM emission + mapper |
| `fast_path.py` | **T0** deterministic templates (count / exists / aggregate / dimension-list) — no retrieval, no LLM |
| `retrieval_v2.py`, `retrieval_select.py`, `graph_retriever.py` | Retrieval (v2 two-stage), single-source-of-truth column selection, graph retrieval |
| `semantic_layer.py`, `schema_linker.py`, `reranker.py` | Legacy MiniLM ensemble retrieval + spaCy schema linking + cross-encoder rerank (fallback shim) |
| `slm_layer.py` | L3 SLM — IR JSON emit, `_normalize_ir`, `_validate_ir`, `run_slm_layer`, **`run_decomposer`** |
| `slm_langgraph.py`, `lg_nodes.py`, `lg_prompts.py` | LangGraph SLM pipeline (`classify_intent→select_entity→select_columns→build_filters→assemble_ir`) |
| `sql_builder.py` | L4/L6 — IR → parameterized SQL compiler (`run_sql_builder`), the `_Ctx` param binder |
| `sql_generator.py`, `sql_validator.py` | LLM SQL generation helper + supplementary AST/type/join validator (`SQLValidator`) |
| `join_planner.py`, `fk_path_resolver.py` | FK-graph join planning (`plan_join_tree`) + multi-hop junction-membership resolution |
| `value_arbiter.py`, `value_resolver.py`, `value_filter.py` | Value-vs-column arbitration, FK value resolution, value-token → filter injection |
| `answer_entity.py` | WHO-questions → PERSON entity via the concept graph (display name over FK) |
| `target_selection.py` | Scores/buckets candidate *targets* for multi-entity queries |
| `nl_simplifier.py`, `nl_answer.py` | Query pre-simplification + result-rows → prose answer |
| `rag_layer.py` | RAG head (`run_rag_layer`) + hybrid head (`run_hybrid_layer`, `_rrf_fuse_hybrid`) |
| `nosql_builder.py` | NoSQL head — native Mongo/etc. query builder |
| `execution_engine.py`, `executor.py` | Multi-source read-only executor (routes by `source_id`) + result dataclass wrapper |
| `audit_logger.py` | Append-only query log (L9) |
| `multi_result.py` | `MultiResult` / `SubResult` + `STATUS_OK/REFUSED/ERROR` |
| `fast_path.py` (`log_route`) | Route/latency logging |

### `retrieval/` — the 5-signal grounding spine (phase 3)
| File | Role |
|------|------|
| `retrieval_engine_phase3.py` | The 5-signal engine (`get_engine().retrieve`) — the spine both heads read |
| `semantic_search.py` | BGE-M3 dense signal (1024-dim) |
| `bm25_ranker.py` | Keyword signal |
| `signal_builder.py` | FK / subgraph structural signals |
| `rrf_merger.py` | Reciprocal-rank fusion |
| `intent_boosting.py` | Intent-aware re-weighting |
| `adaptive_cutoff.py` | Semantic-cliff variable top-k |
| `query_enrichment.py` | Singularization + high-precision term expansion |
| `retrieval_cache.py` | Retrieval memoization |

### `ingestion/` — build the substrate (L0)
| File | Role |
|------|------|
| `schema_scanner.py` | INFORMATION_SCHEMA → `ScanResult` (UUID per table/column; sensitive-column exclusion) |
| `schema_unifier.py`, `source_dispatcher.py` | Multi-source schema unification + source routing |
| `data_graph.py` | Undeclared-FK discovery via value overlap (threshold 0.70) |
| `vector_store.py` | pgvector store (`RetrievalResult`, `store_fk_adjacency`, `resolve_cols_by_exact_names`) |
| `db_abstraction.py` | Internal + source DB connection pools (`get_internal_connection`) |
| `semantic_type_inference.py` | 3-layer rule engine → semantic types (MONETARY/TEMPORAL/CATEGORICAL/IDENTIFIER/FLAG/TEXT) |
| `value_sampler.py`, `data_profiler.py` | Value samples + column profiling (distinct/top-N) |
| `reg_builder.py`, `relgt_encoder.py` | Relational Entity Graph (PyG `HeteroData`) + RELGT structural encoder (256-dim) |
| `biencoder.py`, `auto_finetune.py` | BGE bi-encoder + per-schema fine-tune |
| `graph_embedder.py`, `graph_store.py`, `graph_persist.py`, `kuzu_store.py` | Graph node embeddings + Kùzu/graph persistence |
| `unified_graph_builder.py`, `relationship_graph.py` | Unified knowledge graph (phase 1) + relationship graph |
| `chunk_embedder.py`, `chunk_linker.py`, `enrich_retrieval_documents.py` | Doc chunk embeddings + chunk→schema linking (RAG substrate) |
| `domain_glossary.py`, `glossary_builder.py` | LLM-generated glossary/synonyms (derived, never hardcoded) |
| `synthetic_query_gen.py`, `column_text.py`, `deterministic_metadata.py` | Synthetic training-pair generation + embed-sentence text + deterministic metadata |
| `semantic_layer_v2.py`, `semantic_postprocessor.py`, `enhance_semantic_model.py` | Semantic model v2 build + post-processing + enrichment |
| `build_intermediate_files.py`, `gen_debug_files.py` | Intermediate artifact + debug-file generation |

### `connectors/` — any source → one vocabulary
`base.py` (`BaseConnector` ABC, `normalise_data_type()`), `relational.py` (Postgres/MySQL/SQLite/Oracle/SQL Server), `datalake.py` (DuckDB over Parquet/Delta/CSV/Iceberg), `nosql.py` (Mongo/Elasticsearch/DynamoDB), `document.py` (PDF/filesystem chunks).

### Other packages
| Package | Files | Role |
|---------|-------|------|
| `graph/` | `api.py`, `query_graph.py` (`suggest_expansions`), `graph_validator.py`, `visualize_graph.py` | Unified knowledge graph: read-only HTTP API, in-memory query engine, integrity checks, visualization |
| `query_engine/` | `intent_detector.py`, `intent_router.py`, `query_cache.py` | Intent detection/routing + query cache |
| `semantic/` | `registry.py`, `compile_semantic_layer.py` + JSON (`concepts`, `dimensions`, `metrics`, `overrides`, `MANIFEST`) | Compiled semantic-layer registry |
| `schema/` | `real_schema.py`, `simulate_schema.py` + `reg_graph.pkl`, `kuzu_graph/` | DB schema introspection + simulated schema + persisted graphs |
| `evaluation/` | `evaluator.py`, `test_queries.py`, `report.py`, `aggressive_eval.py`, `hard_eval.py`, `auto_ground_truth.py`, `anchor_accuracy.py`, `router_anchor_diag.py`, `run_phase3_tests.py` | Eval harnesses + HTML/JSON reporting |
| `training/` | `train_relgt.py` | RELGT encoder training |
| `glossary/` | `domain_glossary*.json`, `static_glossary.json`, `hf_glossary.json`, `slm_glossary.json` | Generated + static glossaries |
| `demo/backend/` | `main.py` | VEDA demo FastAPI backend |
| `utils/` | `logger.py` | Logging |
| `tests/` | `test_nl_simplifier.py` | Unit tests (13 tests tracked in graph) |

---

## 3. Structure as seen by the knowledge graph

The code-review graph clusters the codebase into **21 communities** (Leiden). Largest and their cohesion:

| Community | Size | Cohesion | What it is |
|-----------|------|----------|------------|
| `ingestion-graph` | 371 | 0.15 | Ingestion + graph building (the substrate factory) |
| `query-query` | 368 | 0.16 | The query pipeline (heads, layers, builders) |
| `connectors-schema` | 144 | 0.30 | Connectors + schema introspection |
| `veda-ir` | 134 | 0.12 | The `veda/` deterministic engine + IR + firewall |
| `evaluation-compute` | 85 | 0.11 | Eval harnesses + metrics |
| `retrieval-semantic` | 82 | 0.21 | The 5-signal retrieval spine |
| `graph-node` | 51 | 0.23 | Unified knowledge graph API/query |
| `semantic-match` | 26 | 0.12 | Semantic registry / value matching |
| `query-engine-query` | 23 | 0.23 | Intent detection/routing |

**Coupling hot-spots** (worth watching in review): `ingestion-graph ↔ query-query` (31 CALLS edges — retrieval reads ingestion stores at query time) and `query-query ↔ tests`/`veda-ir` (12 / 10 edges). These are the seams where a change is most likely to ripple.

---

## 4. Ingestion pipeline (L0 — build the substrate)

Runs once per DB (or on schema change) via `python main.py --ingestion-only`. Produces the schema-agnostic grounding substrate every query reads.

1. **Schema scan** (`schema_scanner` ← `schema/real_schema.py`) → tables, columns, PK/FK, UUIDs. Excludes `SENSITIVE_PATTERNS` (password, token, ssn, otp, salt, hash, aadhar, pan_number, cvv).
2. **FK adjacency store** (`vector_store.store_fk_adjacency`) → `fk_adjacency` table (the join engine's source of truth). Mode-independent.
3. **Data graph** (`data_graph`) → undeclared FKs via value overlap (`DATA_GRAPH_OVERLAP_THRESHOLD=0.70`, sample 200 rows), merged into `fk_adjacency`. Non-fatal.
4. **Semantic type inference** (`semantic_type_inference`) → 6 semantic types via a 3-layer rule engine (explicit rules → name/type patterns → sample-data heuristics), with confidence + review flags.
5. **Value sampling / profiling** (`value_sampler`, `data_profiler`) → categorical value samples (grounding + value-arbitration) and per-column distinct/top-N.
6. **Embeddings**: `reg_builder` (PyG `HeteroData`) → `relgt_encoder` (256-dim structural); `biencoder` (BGE-large 1024-dim) with optional per-schema `auto_finetune`; legacy MiniLM (384) + TF-IDF/SVD (256) for ensemble.
7. **Vector store** (`vector_store`) → pgvector tables (see §11).
8. **Derived language** (`domain_glossary`, `glossary_builder`, `synthetic_query_gen`) → **LLM-generated** glossary, synonyms, and synthetic NL↔IR training pairs. Derived, never hardcoded.
9. **Unified knowledge graph** (`unified_graph_builder`, `chunk_embedder`, `chunk_linker`, `graph_embedder`, `graph_persist`/`kuzu_store`) → nodes+edges over schema and doc chunks, embedded and persisted for query-time expansion (`graph.query_graph.suggest_expansions`).

---

## 5. The deterministic query pipeline — `veda/pipeline.py::run_query`

This is the CORRECTNESS head, the default route, and the single most critical flow in the codebase (graph criticality 0.81 for `main`, 0.71 for `run_sql_builder`). It runs an **escalation ladder** and stops at the first firewall-passing answer. `return_result=True` yields a dict `{status, ok, cols, rows, answer, sql, trace}`; every step is recorded into an `explain` trace.

### 5.1 Understand the query
- **L1 Temporal** (`temporal_parser.run_temporal_parser`) → ISO window (or none).
- **L4 Intent** (`query_engine.intent_detector.IntentDetector`) → SIMPLE / MULTI_TABLE / AGGREGATE (falls back to `SIMPLE`).
- **L4a Existence** (`planning.existence_mode`) → semi/anti-join operator ("with"/"without"/"how many have"). Existence queries are **never cached** (embeddings can't tell "with" from "without").

### 5.2 The escalation ladder (first passing answer wins)

**T0 — Fast path** (`fast_path.try_fast_path`, `FAST_PATH_ENABLED`): count / aggregate / dimension-list questions resolve straight from compiled registries — **no retrieval, no planner, no LLM, no `get_engine()`** (fast even cold). Conservative match; falls through on miss.

**T0 — Verified-query cache** (`veda/cache.verified_cache_lookup`, cosine ≥ 0.85): replay prior verified SQL, skipping retrieval + SLM. Skipped for existence/fast-path/temporal.

**T1/T2 — Full path** (on cache/fast-path miss):
1. **L2+ Enhance** (`veda/query_enhancement.enhance_query`, recall-only sidecar — gates always validate the *original* query, asserted in code).
2. **L2 Retrieval** — `get_engine().retrieve(...)` (the 5-signal spine, §7), then two additive, flag-guarded, fully try/excepted boosters:
   - **L2g Graph expand** (`graph.query_graph.suggest_expansions`) — add synonym/alias/FK-neighbour columns.
   - **L2b Primary rerank** (`query.reranker`, cross-encoder BGE-reranker-v2-m3) — re-score so anchor selection reads reranked `final_score`.
3. **L2/L3 Anchor** — `routing.select_primary_table` → `routing.vet_primary` (word-order/grain vet). No anchor → **refuse `no_table`**.
4. **Branch by intent/shape** (`planning.try_multitable` for MULTI_TABLE / AGGREGATE / existence):
   - `clarify` / `refuse` → return with feedback.
   - **L4b Existence** → deterministic EXISTS / NOT EXISTS (no LLM, no fan-out).
   - **L4c Aggregate** → deterministic pre-aggregation CTEs (no LLM, fan-out-free by construction).
   - **L4b Join plan (`sql`)** → planner pins the join skeleton; the LLM fills SELECT/WHERE only, constrained by `join_constraints` (key pairs, predicate cols) + `fanout_guard`. `_llm_sql=True`.
   - **Single-table** → a sub-ladder of deterministic resolvers, each tried in order and each skipping the LLM when it fires:
     - **L4e Answer-entity** (`answer_entity.find_answer_entity`) — WHO/handler → display name over FK.
     - **L4d FK value** (`value_resolver.resolve_value_filter`) — exact value on a related table → `IN (subquery)`.
     - **L4d Multi-hop FK** (`fk_path_resolver.resolve_fk_path`, off by default) — single unambiguous junction path.
     - **L4c Value arbiter** (`value_arbiter.arbitrate`) — categorical value / negation → `WHERE` filter on the anchor.
     - **L4e Temporal-only** — date window on the canonical temporal column.
     - **Temporal-refuse** — temporal question on a table with no date column → **refuse** (never let the LLM invent `created_at`).
     - **L5 Single-table LLM** (`veda/generation.generate_sql`) — last resort; seeded with an in-scope column glossary + phrase→column term map so it can't pick a sibling column.

### 5.3 The unified firewall (identical for every route above)
Order matters — each gate can refuse:
1. **L6a Value grounding** (`validation.value_grounding`) — every filter literal exists in the sampled data (polymorphic-predicate values skipped). Fail → `ungrounded`.
2. **L6b Qualifier completeness** (`validation.qualifier_completeness`) — every user-named qualifier is represented in the SQL. Fail → `qualifier_dropped`. (Hard-asserted to run on the *original* query.)
3. **L6b+ IR equivalence** (`ir_equivalence.validate_ir_equivalence`, LLM SQL only) — no filters/joins/grouping/ordering/DISTINCT the query never asked for. Fail → `ir_mismatch`.
4. **L6c Validate + parameterize** (`validation.validate_and_parameterize`) — AST read-only, table/column-hallucination check, bind every literal, ON-integrity + fan-out firewall (`graph_guard`). Fail → `invalid`.
5. **L7 Execute** (`veda/execution.execute_sql`) — read-only connection, 30s timeout, fetch ≤ 20. Fail → `exec_error`.
6. **L7b NL answer** (`nl_answer.run_nl_answer`, `NL_ANSWER_ENABLED`) — rows → one-line prose (deterministic row-count fallback if Ollama down).
7. **Cache-back** — non-temporal, non-existence, non-fast-path answers with rows are saved to the verified-query cache.

Terminal statuses: `answered · no_table · clarify · refuse · ungrounded · qualifier_dropped · ir_mismatch · invalid · exec_error`.

---

## 6. The other heads (breadth)

- **RAG** (`rag_layer.run_rag_layer`, `RAG_TOP_K=5`) — retrieve doc chunks (`chunk_embedder.retrieve_top_k_chunks`) → LLM synthesis (`_call_ollama`).
- **Hybrid** (`rag_layer.run_hybrid_layer`) — fuse SQL-schema signals + doc chunks via `_rrf_fuse_hybrid`, cap `HYBRID_MAX_RESULT_ROWS=20`.
- **NoSQL** (`nosql_builder.run_nosql_builder`, `NOSQL_MAX_NESTING_DEPTH=3`) — native Mongo/etc. query.
- **Execution** for non-SQL sources routes through `query/execution_engine.py` by `source_id`.

All heads return a `SubResult`; the front door wraps them into the same `MultiResult`.

---

## 7. The retrieval spine (5-signal, phase 3)

`retrieval/retrieval_engine_phase3.py::retrieve` is the one retrieval stack both the SQL and hybrid heads read (`veda/runtime.get_engine()` keeps it warm process-wide):

1. **BGE-M3 dense** (`semantic_search`, 1024-dim) · 2. **BM25 keyword** (`bm25_ranker`) · 3. **FK / subgraph structural** (`signal_builder`) · 4. **Value signal** (sampled column values) → fused by **RRF** (`rrf_merger`), re-weighted by **intent** (`intent_boosting`), cut by **adaptive cutoff** (`adaptive_cutoff`), memoized (`retrieval_cache`).

A legacy two-stage path (`query/retrieval_v2.py::retrieve_v2` → `first_stage_retrieve` → `graph_expand`) and the MiniLM ensemble (`query/semantic_layer.py`, `query/reranker.py`) remain as fallback shims during migration.

---

## 8. The IR contracts

### IR v1 (legacy, `slm_layer` ↔ `sql_builder`) — join-carrying, UUID-based
The contract still honored by `run_sql_builder`. UUID-only references; `_normalize_ir` fixes known hallucinations (`aggregates→aggregations`, `type/function→func`, `field→col_id`, forces `intent`). With `IR_JOIN_FREE_ENABLED=True` the SLM omits `joins[]` and `sql_builder` derives them from `fk_adjacency`.

```json
{ "version":"1.0", "intent":"SELECT|COUNT|AGGREGATE",
  "entities":[{"table_id":"<uuid>","alias":"t1","columns":[{"col_id":"<uuid>"}]}],
  "filter_tree":{"type":"AND|OR|NOT","conditions":[{"type":"EQ|NEQ|GT|GTE|LT|LTE|IN|LIKE|IS_NULL|BETWEEN","col_id":"<uuid>","value":"<scalar|list>"}]},
  "joins":[{"from_table_id":"<uuid>","from_col_id":"<uuid>","to_table_id":"<uuid>","to_col_id":"<uuid>","join_type":"INNER|LEFT|RIGHT"}],
  "aggregations":[{"func":"COUNT|SUM|AVG|MIN|MAX","col_id":"<uuid|*>","alias":"count_result"}],
  "group_by":[{"col_id":"<uuid>"}], "order_by":[{"col_id":"<uuid>","direction":"ASC|DESC"}],
  "limit":null, "schema_version":1 }
```

### IR v2 (canonical target, `veda/ir_emit.py` / `ir_validator.py`) — name-based, **join-free**, dialect-free
The strategic contract (see ARCHITECTURE_HYBRID.md §1). No `joins` key (reasoning *cannot* author a wrong join), no `sql` key, no `raw_sql` escape hatch. Entities are implied by dotted field names; the compiler derives the join path or **refuses** on ambiguity.

```jsonc
{ "anchor":"role", "projections":["role.name","organization.name"],
  "filters":[{"field":"role.status","op":"=","value":"active"}],
  "aggregations":[], "group_by":[], "order_by":[], "limit":null,
  "temporal":null, "confidence":0.0, "provenance":"consensus|deterministic|llm" }
```

Consensus (`veda/consensus.py`) and Verifier (`veda/verifier.py`) reconcile IR₁ (deterministic) and IR₂ (LLM) per-field before the compiler runs — the newest, partially-wired glue toward the full target.

---

## 9. Execution-flow catalog

The graph detected **230 flows**; below are the load-bearing ones by criticality with their call chains. (Full list via `list_flows_tool`; drill in with `get_flow_tool`.)

### Front-door & orchestration
- **`main`** (crit 0.81, 171 nodes) — `main.py` full pipeline (ingestion → eval → report).
- **`run_hybrid_query`** → `classify`/`route_query` → (`_maybe_split` → `run_decomposer` → `_fan_out`/`_run_sub`) → `_dispatch_single` → head → `_to_subresult` → `MultiResult`.
- **`run_decomposer`** (`slm_layer`) → `_call_ollama_decompose` → `_coerce_decompose` → `_extract_json` → `_conf`/`_log_decompose` → `DecomposeResult`.

### Deterministic SQL head
- **`run_query`** (`veda/pipeline.py`) — the escalation ladder of §5 (temporal → intent → fast-path/cache → retrieve → anchor → branch → firewall → execute → answer).
- **`run_sql_builder`** (crit 0.71, 30 nodes) → `_build_sql_from_ir` → `_build_uuid_maps` → `_resolve_ir_entities` → `_infer_primary_table` → `_build_from_join`/`_resolve_join_refs` → `_build_select_cols` (`_add`,`_add_entity_col`,`col_ref`) → `_build_filter` → `_build_group_by`/`_build_order_by`/`_build_limit`, with `_Ctx.p` binding every value and `_remap_to_temporal_col`/`_pick_best_temporal` for temporal remap.
- **`generate_sql`** (`veda/generation.py`, 19 nodes) — single-table + join-skeleton fill.
- **`run_slm_layer`** (20 nodes) / **`run_langgraph_pipeline`** (`slm_langgraph` → `_build_graph`/`_get_graph` → `lg_nodes`: `node_classify_intent → node_select_entity → node_select_columns → node_build_filters → node_assemble_ir`) → `_validate_ir` (`_check_col`,`_check_table`,`_walk_filter`) → `SLMResult`/`_fallback_result`.
- **Planner** — `plan_join_tree` (13), `plan_joins` (6), `resolve_fk_path` (6), `select_anchor` (7), `build_from_entities` (18), `find_value_filter_columns` (11), `build_value_filters` (8), `arbitrate` (9), `find_answer_entity` (5).

### Firewall & execution
- **`validate_sql`** (`sql_validator.SQLValidator`) → `_check_syntax`,`_check_tables`,`_check_columns`,`_check_types`,`_check_aggregations`,`_check_joins`,`_check_read_only` → `_attempt_repair` → `ValidationResult`.
- **`repair_and_validate`** (6), **`execute_sql`** / **`execute_sql_safe`** (5), **`execute_query`**, **`log_query`** (`audit_logger`, 5).

### Retrieval & graph
- **`retrieve_v2`** → `_get_query_encoder` → `_get_pg_conn` → `first_stage_retrieve` → `_cosine_search_v2` → `graph_expand` → `_fetch_columns_by_name`.
- **`rerank_columns`** / **`rerank_tables`** (10 each), **`suggest_expansions`** / **`get_synonyms`** / **`get_related_tables`** / **`get_related_columns`** (graph), **`run_graph_retrieval`**, **`_encode_hybrid_bge`** (11).

### RAG / hybrid / NoSQL heads
- **`run_hybrid_layer`** (14) → `retrieve_top_k_chunks` → `_encode_rag_query` → `_build_hybrid_user_message` → `_call_ollama` → `_rrf_fuse_hybrid` (DB via `db_abstraction.get_internal_connection`).
- **`run_rag_layer`** (12), **`run_nosql_builder`** (12), **`map_envelope_to_intent`** (10), **`enrich_query`**/**`enhance_query`**, **`run_nl_answer`**.

### Connectors & schema
- **`get_real_schema`** (crit 0.77, 25) — INFORMATION_SCHEMA introspection.
- **`get_schema`** / **`get_nosql_schema`** / **`get_chunks`** / **`get_document_count`** / **`sample_column_values`** / **`get_row_count`** / **`connect`** / **`_get_raw_connection`** / **`build_connector`** — per-connector surface.

### Ingestion & eval
- **`run_synthetic_query_gen`** (crit 0.75, 22), **`run_schema_linker`** (11), **`run_hybrid_layer`** (above), **`_diagnose`** (23), **`_read_tables`** (11), **`emit_envelope`** (4), **`batch_route`** / **`classify_batch`** (5).

---

## 10. Key configuration (`config.py` — single source of truth)

**Retrieval / anchor**
| Flag | Default | Purpose |
|------|---------|---------|
| `ENCODER_MODE` | `ensemble` | relgt_only \| light_text \| hybrid \| ensemble |
| `TOP_K` | 15 | Columns retrieved per query |
| `TOP_K_TO_LLM` | 6 | Best-N passed to L3 (token budget) |
| `BIENCODER_MODEL` / `BIENCODER_DIM` | `BAAI/bge-large-en-v1.5` / 1024 | BGE spine |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder rerank |
| `PRIMARY_RERANK_ENABLED` | True | Rerank on the primary path (not just Tier-2) |
| `FK_MAX_HOP_DEPTH` / `FK_MAX_INJECTED_COLS` | 2 / 5 | FK bridge traversal |
| `QUERY_ROUTER_ENABLED` / `..._CONFIDENCE_THRESHOLD` | True / 0.6 | Modality routing |

**Reasoning / SQL**
| Flag | Default | Purpose |
|------|---------|---------|
| `SLM_MODEL_NAME` | `qwen2.5-coder:7b` | L3 model (Ollama) |
| `SLM_TIMEOUT_SECS` / `SLM_MAX_TOKENS` / `SLM_IR_MAX_TOKENS` | 240 / 2048 / 512 | L3 budgets |
| `IR_JOIN_FREE_ENABLED` | True | SLM omits `joins[]`; builder derives from `fk_adjacency` |
| `FAST_PATH_ENABLED` | on | T0 templated answers |
| `QUERY_DECOMPOSE_ENABLED` | flag | Compound-query splitting |
| `SQL_DEFAULT_LIMIT` / `SQL_MAX_SUBQUERY_DEPTH` | 1000 / 3 | SQL bounds |
| `NL_ANSWER_ENABLED` / `NL_ANSWER_MAX_ROWS` | True / 50 | Prose answer |

**Feature-gated deterministic resolvers** (each fully try/excepted, additive): `VALUE_ARBITER_ENABLED`, `FK_VALUE_RESOLUTION_ENABLED`, `MULTIHOP_FK_RESOLUTION_ENABLED`, `ANSWER_ENTITY_DISCOVERY_ENABLED`, `GRAPH_EXPAND_ENABLED`, `QUERY_ENHANCEMENT_ENABLED`, `VALUE_FILTER_ENABLED`, `SCHEMA_LINK_ENABLED`, `TEMPORAL_PARSER_ENABLED`, `FEEDBACK_ENABLED`.

**Ingestion / substrate**: `DATA_GRAPH_ENABLED` (0.70 overlap), `VALUE_SAMPLER_ENABLED`, `SYNTHETIC_QUERY_GEN_ENABLED`, `AUTO_FINETUNE_ENABLED`, `GLOSSARY_GENERATION_ENABLED`, `UNIFIED_GRAPH_ENABLED`, `GRAPH_PERSIST_ENABLED`, `GRAPH_CHUNK_LINKING_ENABLED`, `GRAPH_EMBED_ENABLED`, `GRAPH_RETRIEVAL_ENABLED`, `BIENCODER_ENABLED`, `RERANKER_ENABLED`, `PROFILING_ENABLED`, `TABLE_UNDERSTANDING_ENABLED`, `COLUMN_UNDERSTANDING_ENABLED`.

Full annotated reference: [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md).

---

## 11. Database & stores (PostgreSQL + pgvector)

| Table / store | Purpose | Created by |
|---------------|---------|------------|
| `column_embeddings` | Single-encoder vector store (non-ensemble) | `vector_store.py` |
| `column_embeddings_lt` | Light-text 256-dim (ensemble) | `vector_store.py` |
| `column_embeddings_hybrid` | Hybrid 640-dim (ensemble) | `vector_store.py` |
| `fk_adjacency` | FK edge store — the join engine's source of truth | `vector_store.store_fk_adjacency` |
| `table_metadata` | Display column per table | `vector_store.py` |
| column-value samples | Value grounding + arbitration | `value_sampler.py` |
| Kùzu / persisted graph | Unified knowledge graph | `kuzu_store.py`, `graph_persist.py`, `schema/kuzu_graph/`, `schema/reg_graph.pkl` |
| verified-query cache | File-based cosine ≥ 0.85 | `veda/cache.py` |

**Connections** (`ingestion/db_abstraction.py`, `config.py`): a *source* DB (the data being queried) and an *internal* store DB (embeddings/graph) are separate pools. Encoder-mode dims: RELGT 256, light-text 256, MiniLM 384, hybrid 640, BGE-hybrid 1280 (BGE 1024 + RELGT 256).

---

## 12. Encoder modes

| Mode | Vector dim | Tables | Notes |
|------|-----------|--------|-------|
| `relgt_only` | 256 | `column_embeddings` | Structural only |
| `light_text` | 256 | `column_embeddings` | TF-IDF + SVD |
| `hybrid` | 640 | `column_embeddings` | MiniLM + RELGT |
| `ensemble` | 256 + 640 | `column_embeddings_lt` + `column_embeddings_hybrid` | **Default** — RRF merge |

The 5-signal spine (§7) additionally uses BGE-M3 (1024-dim). Switching `ENCODER_MODE` requires re-running ingestion.

---

## 13. How to run

```bash
ollama serve                                   # local LLM (terminal 1)
python main.py                                 # full ingestion + eval (~40 min first run)
python main.py --ingestion-only                # rebuild substrate only
python veda_hybrid.py "how many incidents are escalated"   # single query via the front door
python main.py --query "show me total users"   # single-query smoke test (~60s)
python main.py --report-only                   # regenerate HTML report from saved JSON (~4s)
python run_hybrid_suite.py                      # full eval suite through veda_hybrid
```
Report output: `evaluation/results/poc_report.html`.

---

## 14. Related docs

| Doc | Scope |
|-----|-------|
| [ARCHITECTURE_HYBRID.md](ARCHITECTURE_HYBRID.md) | **Canonical target design** — IR v2, L0–L9 module map, escalation ladder, genericity proof |
| [FINAL_ARCHITECTURE.md](FINAL_ARCHITECTURE.md) | Consolidated final architecture narrative |
| [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md) | Architecture audit findings |
| [SEMANTIC_LAYER.md](SEMANTIC_LAYER.md) / [RETRIEVAL.md](RETRIEVAL.md) | Semantic-layer + retrieval deep dives |
| [PIPELINE_WALKTHROUGH_HYBRID.md](PIPELINE_WALKTHROUGH_HYBRID.md) | Worked end-to-end query walkthrough |
| [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) | Every config flag, annotated |
| [SPEC.md](SPEC.md) / [CLAUDE.md](CLAUDE.md) | Functional spec / AI-agent rules |

> **Note on drift:** several docstrings and this map describe target-state wiring (IR v2 emit, Consensus, Verifier, dialect emit) that is partially implemented. When code and doc disagree, the code in `veda/pipeline.py` + `veda_hybrid.py` is authoritative for *current* behavior; ARCHITECTURE_HYBRID.md is authoritative for *intended* behavior.
