# ARCHITECTURE.md — VEDA Platform

**System**: VEDA — Natural Language → Query engine (SQL-first, multi-modal), packaged as a Django platform around a preserved research engine.
**Shape**: On-premise, zero-egress. All model inference is local; no source-DB data (schema, values, query text) leaves the server.
**Basis**: Written from a direct read of the source in this repository (`apps/`, `inference/`, `storage_adapters/`, `config/`, `docker/`, `veda_core/`). Everything below is what the code actually does; where a piece is a skeleton or a fallback, it is called out as such. Code comments throughout reference `migration_plan.md` sections (`§N`).

---

## 1. The two tiers

The repository is one Django project (`config/`) with six apps (`apps/`), a separate FastAPI inference service (`inference/`), a storage seam (`storage_adapters/`), and the preserved engine (`veda_core/`). At runtime this is **two process tiers plus stores**:

```
                 nginx  (only published port: 8080→80)
                   │
   ┌───────────────┴───────────────┐           HTTP: X-Veda-Source-Id / X-Veda-Tenant / X-Request-Id
   │  api (Django/DRF)             │ ─────────────────────────────────────────────┐
   │  worker / beat (Celery)       │                                              │
   │  ingest-worker (Celery, ML)   │                                              ▼
   └───────────────────────────────┘                       ┌──────────────────────────────────────┐
   thin tier — imports NO veda_core                         │ inference (FastAPI/ASGI)             │
   (api calls inference over HTTP)                          │  → veda_core.veda_hybrid.run_hybrid… │
                                                            │  warm engine per worker process      │
                                                            └──────────────────────────────────────┘
                                                                          │ raw psycopg2 (via PgBouncer) + Redis
                                                                          ▼
   Postgres+pgvector:  veda (Django substrate)  ·  veda_engine (engine operational store)  ·  source DB (read-only)
   redis-cache (assembled sm, rehydrate pub/sub)   redis-broker (Celery only)   Ollama/vLLM (SLM)
```

Verified in the code:
- **`apps/query/inference_client.py`** talks to the inference service over stdlib `urllib` (no `veda_core` import in the api tier); on timeout/connection failure it raises `InferenceUnavailable`, which **`apps/query/views.py`** turns into a structured `503`, never a 500.
- **`inference/main.py`** builds the FastAPI app, warms the engine once at ASGI lifespan (`inference/loaders.py::hydrate`), and sets the ambient `(source, tenant)` per request from headers.
- **`veda_core/veda/execution.py::execute_sql`** connects to the **source DB** from `VEDA_SOURCE_*` env (`veda/runtime.py::DB_CONFIG`), opens the session `readonly=True, autocommit=True`, sets `statement_timeout = 30000`, and `fetchmany(20)`.

---

## 2. Repository map (what each file actually is)

### `config/` — Django project
| File | What it does (verified) |
|------|-------------------------|
| `settings/base.py` | Two DB aliases (`default`, `source_registry`) both dial `PGBOUNCER_HOST:PORT`; `DISABLE_SERVER_SIDE_CURSORS=True` (transaction pooling); split Redis (`redis-cache` for `CACHES`, `redis-broker` for Celery); DRF token auth + throttles; `VEDA = build_veda_settings()`. |
| `celery.py` | `Celery("veda")`, broker/back-end from settings, task queues `ingestion` / `high` / `default`. |
| `urls.py` | `/admin`, `/api/v1/` (from `apps.query.urls`), `/healthz`, `/readyz`, `/metrics`. |
| `asgi.py` / `wsgi.py` | Django entrypoints (separate from the FastAPI `inference` app). |

### `apps/` — six bounded contexts
| App | Files | What it does |
|-----|-------|--------------|
| `core` | `models.py`, `middleware.py`, `tenant_task.py`, `settings_bridge.py`, `views.py` | `TenantScopedModel` + `TenantManager` (auto-filter by ambient context); `RequestIdMiddleware` (`X-Request-Id`); `TenantTask` (Celery base that binds context in a copied contextvars context); `build_veda_settings()` (config→settings bridge with env override); `/readyz` + `/metrics`. |
| `sources` | `models.py` | `Source` (connection on the row: host/port/dbname/db_user/password_env|inline; `resolve_password`/`connection`/`as_engine_env`) + `SourceConnectionProfile`. `ready` flips only on ingestion success. |
| `substrate` | `models.py` | Every ingestion output as a model + `managed=False` pgvector mirrors. See §4. |
| `ingestion` | `tasks.py`, `models.py` | `task_ingest_source` (runs the engine pipeline in a subprocess, streams stage markers) + `IngestionJob`/`IngestionStage`. See §7. |
| `query` | `views.py`, `inference_client.py`, `models.py`, `urls.py` | `QueryView` (POST `/api/v1/query`), `InferenceClient`, `QueryLog` (audit), `IngestTriggerView`/`EvalTriggerView` (staff). |
| `evaluation` | `tasks.py`, `models.py` | `task_run_eval` runs a query set through inference → `EvalRun`/`EvalCaseResult` + HTML. |

### `inference/` — warm ASGI service
| File | What it does |
|------|--------------|
| `main.py` | FastAPI app; lifespan `hydrate()`; per-request middleware `set_context` from `x-veda-source-id`/`x-veda-tenant`; starts the redis rehydrate subscriber that clears `veda_hybrid._SM` on broadcast. |
| `loaders.py` | `hydrate()` — checks the semantic-model file exists, warms `retrieval_engine_phase3.get_engine()`; returns a readiness dict. Best-effort (never crashes startup). |
| `concurrency.py` | `run_in_threadpool_with_context` — offload `fn` to a thread under `copy_context()`. Raw offload is lint-banned in `inference/`+`veda_core/`. |
| `routes/hybrid.py` | POST `/v1/run_hybrid_query` → `run_hybrid_query(req.query)` verbatim, serialized, top-level `status` surfaced. |
| `routes/retrieve.py` | POST `/v1/retrieve` (`get_engine().retrieve`) + POST `/v1/rehydrate` (drop `_SM`, publish fan-out). |
| `routes/health.py` | `/healthz` + `/readyz`. |

### `storage_adapters/` — substrate I/O seam
| File | What it does |
|------|--------------|
| `reader.py` | Query-time reads, **Django-free** (raw psycopg2 via PgBouncer + Redis): `get_fk_adjacency`, `glossary`, `synonyms`, `value_samples`, `ann_search` (raw pgvector, `SET LOCAL hnsw.ef_search` inside an explicit txn), `verified_cache_lookup`, `save_verified_query` (INSERT … ON CONFLICT DO NOTHING + rehydrate publish). All read `context.current()` (fail-closed). |
| `writer.py` | Ingestion-time persistence via the Django ORM: `store_fk_adjacency`, `store_glossary`, `store_semantic_model`, `sync_from_engine` (pull FK/value-samples/glossary/graph from `veda_engine`), `warm`. `store_column_embeddings` raises `NotImplementedError` (routes to the engine's batched writer — not yet extracted). |
| `assembler.py` | `SemanticModelAssembler` — `assemble(source, tenant)` rebuilds the `sm` dict from `Sm*` rows; `persist` is the inverse; `publish_sm`/`publish_rehydrate` push to redis-cache. |

### `veda_core/` — preserved engine + three new seams
Preserved packages (moved verbatim): `veda/` (deterministic engine + firewall), `query/` (router, heads, IR/SLM, SQL builder, resolvers), `retrieval/` (5-signal spine), `ingestion/` (substrate factory), `connectors/`, `graph/`, `semantic/`, `schema/`, `slm/`, `utils/`, plus `config.py` (engine flags, single source of truth) and `main.py` (L0 orchestrator, run in a subprocess).
New for the platform: `context.py` (ambient `RequestContext`), `slm/_call_slm.py` (SLM Strategy seam), and the redis `sm` load in `veda_hybrid.py`.

---

## 3. The query flow (`veda_core/veda_hybrid.py`)

`run_hybrid_query(query)` is the single front door and **always returns a `MultiResult`** (`query/multi_result.py`: a list of `SubResult`, one item for a plain query, N for a compound). Verified control flow:

1. **`QUERY_DECOMPOSE_ENABLED` off (default)** → `_dispatch_single(query)` → wrap one `SubResult` in a `MultiResult`. No decomposer runs on the hot path.
2. **Decompose on** → `classify(query)`; if the intent is `sql`, run the deterministic head **as a probe first** (stdout captured); if it returns `ok`, replay the trace and return (a clean SQL answer is complete-by-construction, so the decomposer is skipped). Otherwise `_maybe_split` runs `slm_layer.run_decomposer`: `should_split` → `_fan_out` independent sub-queries; `DECOMP_DEPENDENT` (nested) → **refuse** with the ordered parts as guidance; else fall through to the single pipeline.
3. **`classify`** (`veda_hybrid.classify` → `query/query_router.route_query`): returns `sql` when `QUERY_ROUTER_ENABLED` is false or the router raises — **`sql` is the safe default**.
4. **`_dispatch_single`** routes by intent:
   - `sql` → `veda/pipeline.run_query(query, sm, cols, return_result=True)`. If it can't answer (`refuse`/`qualifier_dropped`/`ungrounded`/`no_table`/`clarify`) **and** `TIER2_LLM_FALLBACK` is on → `_tier2_sql` (LLM emits IR → deterministic `sql_builder` → the same firewall gates → execute). The LLM never writes SQL structure even in Tier-2.
   - `rag` → `query/rag_layer.run_rag_layer`.
   - `hybrid` → runs the deterministic head first, feeds its executed rows into `query/rag_layer.run_hybrid_layer` (correct-by-construction numbers, not LLM SQL).
   - `nosql` → `_run_nosql` (resolve source → `connectors.build_connector` → `query/nosql_builder.run_nosql_builder` → execute).
5. **`_fan_out`** runs sub-queries in query order; sequential + live output when `QUERY_DECOMPOSE_MAX_WORKERS == 1`, otherwise a thread pool with a thread-routing stdout and the parent `(source, tenant)` context carried into each worker (`context.set_context` from `try_current()`).

`_load_semantic_model()` prefers the Django-assembled `sm` from redis-cache (`_load_sm_from_redis`, gated on `VEDA_SM_REDIS`), falling back to the on-disk `SEMANTIC_MODEL_FILE`; it caches in-process in `_SM`, which the rehydrate subscriber clears.

**The router** (`query/query_router.py`) is a keyword-signal classifier: it scores `_SQL_KEYWORDS` / `_RAG_KEYWORDS` / `_NOSQL_KEYWORDS` / `_TEMPORAL_KEYWORDS` (temporal counts double toward SQL), discounts RAG when query tokens match sampled DB values (`_check_value_filter`), and picks `sql`/`rag`/`hybrid`/`nosql` by dominant normalized score against the available source types. With no document/nosql sources it always returns `sql`.

---

## 4. The substrate (`apps/substrate/models.py`)

Every model inherits `TenantScopedModel` (UUID PK matching ingestion UUIDs + `source` FK + `tenant` + timestamps). `TenantManager.get_queryset` filters by `context.current()` and falls back to unscoped when no context is set; `all_tenants()` is the explicit escape hatch.

| Group | Models | Backs |
|-------|--------|-------|
| Structural | `SchemaTable`, `SchemaColumn`, `FkEdge`, `TableMetadata` | schema scan; `FkEdge` = the join engine's `fk_adjacency` (undeclared FKs from the data graph carry `is_declared=False`, `overlap_score`). |
| Semantic/language | `SemanticType`, `GlossaryEntry`, `Synonym`, `SyntheticPair`, `SemanticConcept` | semantic-type inference, glossary/synonyms, synthetic pairs, compiled concepts. |
| Value grounding | `ColumnValueSample`, `ColumnProfile` | value sampler (mirrored to a Redis SET per the docstring) + profiler. |
| Embeddings (`managed=False`) | `ColumnEmbedding`/`_LT`/`_Hybrid`/`_BGE`, `ChunkEmbedding`, `RelgtStructural`, `GraphNodeEmbedding` | pgvector tables (real tables + HNSW indexes via RunSQL migration); admin visibility only — ANN is raw SQL in `reader.ann_search`. |
| Graph | `GraphNode`, `GraphEdge`, `GraphArtifact` | unified KG for expansion; artifact registers the KG/relationship-graph file path + version. |
| Verified cache | `VerifiedQueryCache` | hot-path write via ON CONFLICT; unique on `(source, tenant, query_hash)`. |
| Semantic-model | `SubstrateVersion`, `SmTable`, `SmColumn`, `SmRetrievalDoc`, `SmSynonym`, `SmConcept` | the normalized `sm` the assembler rebuilds; `SubstrateVersion` drives rehydrate. |

`QueryLog` (`apps/query/models.py`) mirrors the audit log: query text, tenant, route, one of the frozen `TerminalStatus` values, `executed_sql` (parameterized placeholder text only), `refusal_reason`, `latency_ms`, `cache_hit`, `request_id`.

---

## 5. The deterministic SQL head (`veda_core/veda/pipeline.py::run_query`)

The correctness head and default route. It runs an escalation ladder and stops at the first firewall-passing answer; with `return_result=True` it returns `{status, ok, cols, rows, answer, sql, trace}`. The stage labels below are exactly what the code prints.

**Understand**: L1 temporal (`temporal_parser.run_temporal_parser`) → L4 intent (`query_engine.intent_detector.IntentDetector`, falls back to `SIMPLE`) → L4a existence (`planning.existence_mode`).

**Ladder**:
- **Fast path** (`FAST_PATH_ENABLED`, not existence): `fast_path.try_fast_path` returns SQL straight from compiled registries — no retrieval, no `get_engine()`, no LLM. Falls through on miss.
- **Verified cache** (skipped for existence and when the fast path fired): `veda/cache.verified_cache_lookup(query)` cosine ≥ 0.85. In the platform this routes through `storage_adapters.reader.verified_cache_lookup` when a context is set (pgvector, tenant-scoped); otherwise it uses the legacy file store. Existence queries are never cached/served from cache (near-identical vectors for "with"/"without").
- **Full path**: L2+ enhance (recall-only, `QUERY_ENHANCEMENT_ENABLED`) → **L2 retrieve** `get_engine().retrieve(query=_search, intent, top_k=15)` (§6) → additive boosters L2g graph-expand (`graph.query_graph.suggest_expansions`) and L2b primary cross-encoder rerank (`query.reranker`, rewrites `final_score`) → **L3 anchor** `routing.select_primary_table` + `vet_primary` (no primary → refuse `no_table`) → branch:
  - MULTI_TABLE / AGGREGATE / existence → `planning.try_multitable`: `clarify` / `refuse`; **existence** → deterministic EXISTS/NOT EXISTS (no LLM); **aggregate** → deterministic pre-aggregation CTEs (no LLM); **sql** → planner pins the join skeleton and sets `join_constraints` (key pairs + predicate cols) + `fanout_guard`, the LLM fills SELECT/WHERE only (`_llm_sql=True`).
  - Single-table sub-ladder, each deterministic and each skipping the LLM when it fires: answer-entity (WHO → display name over FK), FK-value (`value_resolver.resolve_value_filter` → `IN (subquery)`), multi-hop FK (`fk_path_resolver`, off by default), value-arbiter (`value_arbiter.arbitrate` categorical/negation), temporal-only (date window on the canonical temporal column), **temporal-refuse** (temporal question on a table with no date column → refuse, never invent `created_at`), else **single-table LLM** (`veda/generation.generate_sql`, seeded with an in-scope column glossary + phrase→column term map).

**Firewall** (same for every branch; the code asserts the gates run on the *original* query, not the enhanced search string):
1. **L6a value grounding** (`validation.value_grounding`) — every filter literal exists in sampled data; deterministic polymorphic-predicate literals are skipped. Fail → `ungrounded`.
2. **L6b qualifier completeness** (`validation.qualifier_completeness`) — every named qualifier is represented. Fail → `qualifier_dropped`.
3. **L6b+ IR equivalence** (`ir_equivalence.validate_ir_equivalence`, LLM SQL only) — no filters/joins/grouping/ordering/DISTINCT the query never asked for. Fail → `ir_mismatch`.
4. **L6c validate + parameterize** (`validation.validate_and_parameterize` + `graph_guard`) — AST check, table/column coverage, bind every literal, join key-pair + fan-out guard. Fail → `invalid`.
5. **L7 execute** (`execution.execute_sql`) — read-only session, 30 s timeout, fetch ≤ 20. Fail → `exec_error`.
6. **L7b NL answer** (`nl_answer.run_nl_answer`, `NL_ANSWER_ENABLED`) — rows → one-line prose (row-count fallback if the SLM is down).
7. **Cache-back**: `save_verified_query(query, sql)` when the answer has rows and is not from-cache, not fast-path, not temporal, not existence.

Terminal statuses: `answered · no_table · clarify · refuse · ungrounded · qualifier_dropped · ir_mismatch · invalid · exec_error`.

---

## 6. Retrieval spine (`veda_core/retrieval/retrieval_engine_phase3.py`)

`get_engine().retrieve(query, intent, top_k=15)` — one warm engine per process (`veda/runtime.get_engine`, which also shares its BGE-M3 with table-routing and the verified cache so the model loads once). Composed signals (verified from the imports/init): `SemanticSearchEngine` (BGE dense), `BM25Ranker` (fit on the semantic model), `SignalBuilder` (FK/subgraph structural), fused by `RRFMerger(k=60)`, re-weighted by `IntentBooster`, cut by `AdaptiveCutoff`. `RetrievalResult` carries `bm25_score`/`rrf_score`/`final_score`; the pipeline's rerank booster overwrites `final_score` so anchor selection reads reranked order.

---

## 7. Ingestion at the platform level (`apps/ingestion/tasks.py`)

`task_ingest_source(source_id, tenant, force, skip_llm, resume)` is the real, wired path:
- Creates an `IngestionJob` + ordered `IngestionStage` rows.
- **ENCODER_MODE guard**: if the last successful job used a different `encoder_mode` and `force` is not set, it marks the job failed and raises (re-ingestion required).
- Injects the source's connection into the subprocess via `Source.as_engine_env()` → the engine's `VEDA_SOURCE_*` env, so ingestion targets that source with no global env/code change.
- **Resume**: if a prior failed job exists or `resume=True`, sets `VEDA_RESUME=1` so the engine skips expensive stages whose output already exists.
- Runs `python -c "import main; main.run_ingestion(...)"` in a **subprocess** with `cwd=veda_core` (isolates the engine's top-level `config` module from the Django `config` package; lets the engine's relative paths resolve), **streams** stdout, and maps the engine's `[N/NN] StageName` markers to live `IngestionStage` RUNNING/SUCCESS updates.
- On success: `task_warm_caches` → `writer.warm()`, then flips `Source.ready=True`, `status=READY`. On exception: marks stages/job failed, `Source.status=FAILED`.

The engine's real L0 order (`veda_core/main.py` `_step` markers): 1 schema scan · 2 FK adjacency · 3 data graph (undeclared FK, overlap 0.70) · 4 semantic-type inference · 5 table-metadata/display columns · 6 value sampler · 7 REG builder · 8 encoder (`ENCODER_MODE`) · 9 vector store · 10 synthetic query gen · 11 BGE fine-tune, plus the LLM semantic-layer-v2 and glossary stages that `VEDA_RESUME` can skip.

There is also a **skeleton** ten-task chain (`STAGE_ORDER` + `task_schema_scan…task_unified_graph`) whose bodies raise `NotImplementedError` via `_todo()` — the target decomposition for when the engine's in-memory orchestration is hoisted into importable stage functions. It is not the path that runs today.

`writer.warm()`: persist the semantic-model file into the `Sm*` substrate (`assembler.persist`), `sync_from_engine()` (FK edges, value samples, glossary→Synonym, KG nodes/edges from `veda_engine`), then `publish_sm` + `publish_rehydrate` so every inference replica clears its `_SM` cache and reloads.

---

## 8. Multi-source onboarding

Onboarding is a data operation, not a code change:
1. Register a `Source` row with its connection (`host/port/dbname/db_user/password_env|password_inline`); secrets by reference.
2. `POST /api/v1/admin/ingest {source_id}` (staff) → `task_ingest_source` reads the row and ingests that source's DB.
3. The query path reads only `ready=True` sources; `ready` flips only on full success.

`scripts/onboard_source.sh` wraps the full reset (wipe `veda_engine` + Django substrate + derived files, re-point `.env`, recreate the `Source` row + containers). `scripts/ingest_baremetal.sh` reads the connection from the `Source` row and runs full ingestion on the host (MPS + host Ollama), then `writer.warm()` + `/v1/rehydrate`.

---

## 9. Tenancy & security (what the code enforces)

| Concern | Mechanism (file) |
|---------|------------------|
| Tenant isolation | Ambient `RequestContext` (`context.py`); `current()` raises when unset; ORM auto-filter (`core/models.TenantManager`); raw reads call `_scope()` (`storage_adapters/*`). |
| Tenant source of truth | Server-resolved (`views._resolve_tenant`), forwarded as headers (`inference_client`), set by inference middleware — never client-supplied for scoping. |
| Read-only source access | `execute_sql` session `readonly=True`; `_FORBIDDEN` DML/DDL word guard (`runtime.py`); AST + coverage + fan-out in `validate_and_parameterize` / `graph_guard`. |
| Parameterized SQL | `validate_and_parameterize` binds every literal; `QueryLog.executed_sql` stores placeholder text. |
| Secrets by reference | `Source.password_env` / `connection_secret_ref`, never plaintext rows. |
| Connection ceiling | Every pool dials PgBouncer (`settings/base.py`); `reader._connection` never sets a session-level READ ONLY (PgBouncer pooling), and `ann_search` uses `SET LOCAL` inside an explicit txn. |
| Request tracing | `RequestIdMiddleware` mints/propagates `X-Request-Id` → forwarded → `QueryLog.request_id`. |
| Zero-egress | `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` set in `veda_hybrid.py` and `main.py` before any model import; SLM on the internal network. |

---

## 10. SLM backend seam (`veda_core/slm/_call_slm.py`)

`call_slm(prompt, *, purpose, timeout=240, **opts) -> str` over a Strategy: `OllamaBackend` (POST `/api/chat`, `keep_alive:"24h"`) and `vLLMBackend` (OpenAI-compatible `/chat/completions`). Backend chosen by `SLM_BACKEND` (default `ollama`), cached per process; both return a plain string with the same error shape. A `_slm_circuit_breaker` context manager is present but is currently a pass-through skeleton. Note: the engine's existing call sites still call their own `_call_ollama` directly; rewiring them onto `call_slm` is incremental.

---

## 11. Deployment (`docker/`, compose)

Nine dev services on `veda_net`; only `nginx` publishes a host port. `api`/`worker`/`beat` build the thin `Dockerfile.api`; `inference`/`ingest-worker` build `Dockerfile.inference` (torch). `postgres` is `pgvector/pgvector:pg16` and hosts both `veda` and `veda_engine` (dev exposes `15432` for bare-metal ingestion). `pgbouncer` fronts all pools (transaction mode). Redis is split: `redis-broker` (Celery only, unbounded) and `redis-cache` (`allkeys-lru`, holds the assembled `sm` and the rehydrate pub/sub). `ollama` is the dev/ingestion SLM; prod adds `vllm`. `inference` runs with `working_dir: /app/veda_core` so engine relative paths resolve; `ingest-worker` runs `celery -A config worker -Q ingestion,high`. `docs/OPERATIONS.md` documents backups, blue/green + engine-only rollback, and scaling (size inference workers by measured RSS; SLM is the throughput bottleneck).

---

## 12. Configuration

- **Engine flags** live in `veda_core/config.py` and reach Django only through `apps/core/settings_bridge.build_veda_settings()` (fallback default → `config.py` value → `VEDA_<FLAG>` env override) — never duplicated in Django settings. Bridged: `ENCODER_MODE`, `TOP_K`, `TOP_K_TO_LLM`, `QUERY_ROUTER_ENABLED`, `SLM_MODEL_NAME`, `SLM_BACKEND`, `IR_JOIN_FREE_ENABLED`, `FAST_PATH_ENABLED`, `QUERY_DECOMPOSE_ENABLED`, `HNSW_M/EF_CONSTRUCTION/EF_SEARCH`.
- **Infra** (DB, Redis, secrets) is env-only.
- **Per-source / runtime env**: `VEDA_SOURCE_*` (source DB the engine reads/executes against, via `runtime.DB_CONFIG`), `INFERENCE_URL`, `PGBOUNCER_*`, `POSTGRES_*`, `REDIS_CACHE_URL`/`REDIS_BROKER_URL`, `OLLAMA_URL`/`VLLM_URL`, `VEDA_SM_REDIS`/`VEDA_SM_SOURCE_ID`/`VEDA_SM_TENANT`, `VEDA_HNSW_EF_SEARCH`, `VEDA_RESUME`, `VEDA_DEFAULT_SOURCE_ID`.

---

## 13. Scenario → code map

### Query-time
| Scenario | Path | Outcome |
|----------|------|---------|
| Plain NL question | `nginx → QueryView.post` → resolve tenant/source → `InferenceClient` (HTTP) → inference middleware `set_context` → `routes/hybrid` → `run_hybrid_query` → head → firewall → `MultiResult` → `QueryLog` | `answered` / a refusal status |
| Router off/unavailable | `veda_hybrid.classify` returns `("sql", None)` | deterministic head |
| Count/aggregate ("how many users") | `run_query` → `fast_path.try_fast_path` (no retrieval/LLM) → firewall → execute | `answered`, fast cold |
| Repeat of a verified query | `run_query` → `cache.verified_cache_lookup` → `reader.verified_cache_lookup` (pgvector, scoped) | `answered`, `QueryLog.cache_hit=True` |
| Join query | `planning.try_multitable` pins join skeleton from `FkEdge`; LLM fills SELECT/WHERE; `graph_guard` fan-out | `answered` / `invalid` / `refuse` |
| "with/without X" | `planning.existence_mode` → deterministic EXISTS/NOT EXISTS; never cached | `answered` |
| Filter value absent | `validation.value_grounding` fails | `ungrounded` |
| User qualifier dropped | `validation.qualifier_completeness` fails | `qualifier_dropped` |
| LLM added unrequested semantics | `ir_equivalence.validate_ir_equivalence` fails | `ir_mismatch` |
| Hallucinated table/column / write attempt | `validate_and_parameterize` / session read-only | `invalid` |
| Temporal question, no date column | single-table temporal-refuse branch | `refuse` |
| No anchor | `routing.select_primary_table` empty | `no_table` |
| Deterministic head refuses + Tier-2 on | `_dispatch_single` → `_tier2_sql` (LLM IR → builder → same gates → execute) | `answered` or kept refusal |
| Compound question | `_maybe_split` → `run_decomposer` → `_fan_out` | N-item `MultiResult` |
| Nested/dependent question | `run_decomposer` → `DECOMP_DEPENDENT` | `refused` with ordered-parts guidance |
| Doc/policy question | router → `rag_layer.run_rag_layer` | `answered` (rag) |
| Mongo/native source | router → `_run_nosql` → `nosql_builder` | `answered` (nosql) |
| Inference slow/unreachable | `InferenceClient._post` raises `InferenceUnavailable` → `503` + `exec_error` audit | no 500/hang |
| Empty query | `QueryView.post` early return | `400` |

### Ingestion & lifecycle
| Scenario | Path | Effect |
|----------|------|--------|
| Onboard a source | register `Source` → `IngestTriggerView` → `task_ingest_source` reads `as_engine_env()` → subprocess | source-specific substrate, no code change |
| Live stage progress | subprocess `[N/NN]` markers → `marker_re` → `IngestionStage` updates | true per-stage progress |
| ENCODER_MODE changed w/o force | guard vs last successful job → raise | re-ingestion required |
| Resume failed job | prior FAILED or `resume=True` → `VEDA_RESUME=1` | engine skips completed expensive stages |
| Fast structural-only | `skip_llm=True` | skips glossary/semantic-layer LLM stages |
| Partial failure | exception → stages/job FAILED, `Source.status=FAILED`, `ready` stays False | query path never reads half-built substrate |
| Warm after ingest | `task_warm_caches` → `writer.warm()` → `assembler.persist` + `sync_from_engine` + `publish_sm/rehydrate` | Django owns substrate; replicas notified |
| Re-ingest reaches replicas | `publish_rehydrate` → redis pub/sub → inference subscriber drops `_SM` | next query reloads assembled `sm` |
| Verified write under N replicas | `reader.save_verified_query` INSERT … ON CONFLICT DO NOTHING + publish | idempotent |

### Platform/ops
| Scenario | Path | Effect |
|----------|------|--------|
| Liveness | `GET /healthz` | `{"status":"ok"}` |
| Readiness | `GET /readyz` → `core/views.readyz` (Postgres/both Redis/inference/SLM) | `ready`(200)/`degraded`(503) |
| Metrics | `GET /metrics` → `core/views.metrics` (from `QueryLog` + PgBouncer `SHOW POOLS`) | Prometheus text, dependency-free |
| Eval run | `EvalTriggerView` → `task_run_eval` → inference → `EvalRun`/`EvalCaseResult` + HTML | tracked artifact |
| Trace across tiers | `RequestIdMiddleware` → forwarded → `QueryLog.request_id` | one id api→inference→logs |

---

## 14. Code changes mapped to scenarios (git history)

| Commit | Change (from the diff) | Scenario it serves |
|--------|------------------------|--------------------|
| `d1f7ac9` Initial commit | Whole platform scaffold: six apps, `inference/`, `storage_adapters/`, `config/`, `docker/`; engine moved into `veda_core/`; `context.py` + `slm/_call_slm.py`; cache.py + `veda_hybrid.py` platform rewires | Establishes the two-tier flow (§1), frozen front door (§3), fail-closed tenancy (§9) — baseline for every §13 scenario |
| `0e6f167` second commit | `main.run_ingestion` resume generalized to `_table_has_rows(table)` + one `VEDA_RESUME` gate across expensive stages; `parity_suite.py` extended; route/explain trace logging | "resume a failed job" / "fast structural-only ingest" (§13); parity gating |
| `3cad291` upgrade for multi source ingestion | `Source` gains connection fields + `resolve_password/connection/as_engine_env`; `task_ingest_source` injects `src.as_engine_env()` into the subprocess; `onboard_source.sh` + `ingest_baremetal.sh` | "onboard a source" / "bare-metal ingestion" as pure data operations (§8) |

Uncommitted working tree: regenerated engine artifacts (`parity_baseline.json`, `veda_glossary.json`, `veda_semantic_checkpoint.json`, trace logs, pickled schema encoders) and the removal of `veda_relationship_graph.json` / `veda_semantic_model.json` now sourced via substrate/redis.

---

## 15. Status (wired vs skeleton, from the code)

**Wired**: the end-to-end query flow (api → inference → `run_hybrid_query` → head → firewall → audit); `task_ingest_source` real ingestion with streamed stage tracking, resume, skip-LLM, ENCODER_MODE guard, per-source connection injection; `storage_adapters.reader` (all reads incl. the ON-CONFLICT verified write) and `writer` (all except the embedding hook); `SemanticModelAssembler`; the SLM Strategy backends; health/readiness/metrics; eval runs; the rehydrate fan-out; the cache.py platform rewire (context-aware verified cache).

**Skeleton / incremental**: the ten-task Celery chain (`NotImplementedError`); `writer.store_column_embeddings`; the `_call_slm` circuit breaker (pass-through) and engine call-site rewire onto `call_slm`; `vLLMBackend` production wiring; auth/JWT + tenant-from-principal (dev `AllowAny` + default tenant); pgvector RunSQL migrations + per-source HNSW tuning.

**Authority**: when code and prose disagree, `veda/pipeline.py` + `veda_hybrid.py` are authoritative for engine behavior; `apps/`, `inference/`, `storage_adapters/` for platform behavior. The `§N` comments in code point to `migration_plan.md` for intended target wiring.
