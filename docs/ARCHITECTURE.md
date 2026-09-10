# ARCHITECTURE — VEDA Platform

**System.** VEDA — natural-language → query engine (SQL-first, multi-modal), packaged as
a Django platform around a preserved research engine.

**Shape.** On-premise, zero-egress. All model inference is local; no source-DB data
(schema, values, query text) leaves the server.

**Basis.** Written from a direct read of the source (`apps/`, `chatbot/`, `inference/`,
`storage_adapters/`, `config/`, `docker/`, `veda_core/`) on 2026-09-09. Where a piece is a
skeleton, dormant behind a flag, or a fallback, it is called out as such. Deeper treatment
of each subsystem lives in its own `docs/*.md`; the layer contracts
(`veda_core/query/contracts/`, `veda_core/ingestion/layers/contracts/`) are the
finest-grained authority and sit next to the code.

> **Authority.** When code and prose disagree: `veda/pipeline.py` + `veda_hybrid.py` are
> authoritative for engine behavior; `apps/`, `inference/`, `storage_adapters/` for
> platform behavior; migrations for what physically exists in the databases.

---

## 1. The two tiers

One Django project (`config/`) with ten apps (`apps/`), a LangGraph conversational package
(`chatbot/`), a separate FastAPI inference service (`inference/`), a storage seam
(`storage_adapters/`), and the preserved engine (`veda_core/`). At runtime this is **two
process tiers plus stores**:

```
                 nginx  (only published port: 8080→80)
                   │
   ┌───────────────┴───────────────┐     HTTP headers: X-Veda-Source-Id / -Source-Ids /
   │  api  (Django / DRF)          │ ───  -Tenant / -Data-Scope / -Source-Profiles / X-Request-Id
   │  worker / beat  (Celery)      │ ─────────────────────────────────────────────┐
   │  ingest-worker  (Celery, ML)  │                                              │
   └───────────────────────────────┘                                             ▼
   thin tier — imports NO veda_core                     ┌──────────────────────────────────────┐
   (calls inference over HTTP; hosts chatbot/,          │ inference  (FastAPI / ASGI)          │
    which also only calls inference over HTTP)          │  → veda_core.veda_hybrid.run_hybrid… │
                                                        │  warm engine per worker, per scope   │
                                                        └──────────────────────────────────────┘
                                                                  │ raw psycopg2 (via PgBouncer) + Redis
                                                                  ▼
   Postgres+pgvector:  veda (Django substrate)  ·  veda_engine (engine store)  ·  source DB(s) (read-only)
   redis-cache (assembled sm, rehydrate pub/sub)  ·  redis-broker (Celery)  ·  redis-stack :6380 (chat checkpointer)
   Ollama / vLLM (SLM)  ·  optional host Metal server (BGE-M3 + reranker on MPS)
```

Verified in the code:

- **`apps/query/inference_client.py`** talks to inference over stdlib `urllib` (no
  `veda_core` import in the api tier). On any transport failure it raises
  `InferenceUnavailable`, which **`apps/query/views.py`** turns into a structured `503`,
  never a `500`. There is **no retry and no circuit breaker** (three docs claimed one;
  they were wrong). Default `INFERENCE_TIMEOUT_S = 300`.
- **`inference/main.py`** builds the FastAPI app, warms the engine + encoders + reranker +
  SLM once at ASGI lifespan (`inference/loaders.py::hydrate`), and sets the ambient
  `(source, tenant, source_ids, allowed_resources)` per request from headers — **only when
  both `x-veda-source-id` and `x-veda-tenant` are present**; otherwise no context is set
  and downstream reads fail closed.
- **`veda_core/veda/execution.py::execute_sql`** connects to the source DB resolved from
  the `Source` row (or `VEDA_SOURCE_*` env when no request context), opens the session
  `readonly=True, autocommit=True`, sets `statement_timeout = 30000`, `SET search_path`
  when the source declares a schema, and fetches up to `EXECUTION_RESULT_LIMIT = 1000`
  rows. A DuckDB path handles parquet/datalake scopes.

---

## 2. Repository map

### `config/` — Django project

| File | What it does |
|------|--------------|
| `settings/base.py` | Ten `INSTALLED_APPS`; two DB aliases (`default`, `source_registry`) both dialing PgBouncer; `DISABLE_SERVER_SIDE_CURSORS=True` (transaction pooling); split Redis (`redis-cache` for `CACHES`, `redis-broker` for Celery); DRF `TokenAuthentication` + `SessionAuthentication` always, `JWTAuthentication` prepended only when `VEDA_JWT_AUTH=1`; global + scoped throttles; `SIMPLE_JWT` (15-min access / 7-day refresh, rotation + blacklist, `CHECK_REVOKE_TOKEN`); feature flags `VEDA_JWT_AUTH` / `VEDA_RBAC_MODE` / `VEDA_ALLOW_ANONYMOUS` / `VEDA_AUTO_SYNC_CATALOG` / `SOURCE_PROFILER_ENABLED`; `VEDA = build_veda_settings()`. |
| `settings/dev.py` | `DEBUG=True`, sqlite fallback when no DB host, `SLM_BACKEND=ollama`. |
| `settings/prod.py` | `DEBUG=False`, refuses boot on the dev `SECRET_KEY` when JWT is on, HSTS/SSL, `SLM_BACKEND=vllm`. |
| `celery.py` | `Celery("veda")`, queues `ingestion` / `high` / `default`. |
| `urls.py` | `/admin/`, `/api/v1/` (includes **five** app urlconfs: `query`, `chat`, `authentication`, `access_management`, `sources`), `/healthz`, `/readyz`, `/metrics`. |

### `apps/` — ten bounded contexts

| App | What it does | Doc |
|-----|--------------|-----|
| `core` | `TenantScopedModel` + `TenantManager` (ambient auto-filter, fail-**open** to unscoped when no context); `RequestIdMiddleware`; `TenantTask` (Celery base that binds context in a copied contextvars context); `build_veda_settings()` (config→settings bridge); `/readyz` + `/metrics`; `token_revocation` (shared leaf); `messages` (all user-facing copy). | — |
| `sources` | `Source` (connection on the row + `dialect` + routing-catalog fields; `resolve_password` / `connection` / `as_engine_env` / `source_kind`); `SourceConnectionProfile`; `SourceItem` (uniform routing item). `ready` flips only on ingestion success. | [`MULTI_SOURCE.md`](MULTI_SOURCE.md) |
| `substrate` | Every ingestion output as a model (structural / semantic / value-grounding / graph / verified-cache / normalized `sm`). Two `managed=False` pgvector mirrors remain (`chunk_embeddings`, `graph_node_embeddings`) — the rest were dropped by migrations 0006–0008; the live ANN store is engine-owned. | — |
| `ingestion` | `task_ingest_source` runs the engine pipeline in a **subprocess**, streams `[[STAGE]]` markers into `IngestionJob` / `IngestionStage` rows, then `task_warm_caches`. A ten-task Celery chain is a `NotImplementedError` skeleton. | [`INGESTION.md`](INGESTION.md) |
| `query` | `QueryView` (`POST /api/v1/query`, `AllowAny`); `scope.py` (server-side source-set resolution, RBAC-narrowed); `InferenceClient`; `QueryLog` (audit); staff `IngestTriggerView` / `EvalTriggerView`. | [`QUERY_ENGINE.md`](QUERY_ENGINE.md) |
| `chat` | `ConversationQueryView` (`AllowAny` + manual 401); `ConversationQueryService` (turn orchestration, SSE bridge); deterministic markdown-table / thinking-message / turn-event / visualization helpers. | [`CHAT.md`](CHAT.md) |
| `authentication` | Login / refresh / logout / password-change. JWT via `simplejwt`, behind `VEDA_JWT_AUTH` (default **off** → login returns a placeholder token). Redis lockout. No models. | [`RBAC.md`](RBAC.md) |
| `access_management` | The RBAC data model (`User`–`UserRole`–`Role`–`RolePermission`–`Permission`) + `CatalogResource` + `PermissionResolver` + Gate 1 / Gate 2, behind `VEDA_RBAC_MODE` (default **off**). Admin CRUD endpoints. | [`RBAC.md`](RBAC.md) |
| `evaluation` | `task_run_eval` runs a query set through inference → `EvalRun` / `EvalCaseResult` + HTML report. | [`EVALUATION.md`](EVALUATION.md) |

### `chatbot/` — LangGraph conversational supervisor (runs inside the api container)

`run.py` (entry `run_chat_turn`), `graph.py` (8-node `StateGraph`), `state.py` (`ChatState`),
`nodes.py` (classify / smalltalk / memory-read / context-resolve / call-engine / memory-write
/ ask-clarification / format-reply + deterministic fast-paths), `llm.py` (backend-agnostic
SLM caller — deliberately **not** an import of `veda_core/slm/`), `checkpointer.py`
(`RedisSaver` on redis-stack `:6380`, needs RediSearch), `memory/` (evidence-only
`QueryFrame` / `DrillStack` / episodic buffer in Redis), `prompts/`. See [`CHAT.md`](CHAT.md).

### `inference/` — warm ASGI service

`main.py` (FastAPI app, lifespan `hydrate()`, `_tenant_context` middleware, rehydrate
subscriber), `loaders.py` (`hydrate()` warms sm-file check + BGE-M3 dense/sparse +
cross-encoder + SLM), `concurrency.py` (`run_in_threadpool_with_context` — raw offload is
lint-banned in `inference/` + `veda_core/`), `routes/hybrid.py` (`POST /v1/run_hybrid_query`
+ `/stream` SSE), `routes/retrieve.py` (`/v1/retrieve` + `/v1/rehydrate`),
`routes/health.py`. `engine.py::get_engine` is **dead** (`NotImplementedError`; superseded
by `veda_core/veda/runtime.get_engine`).

### `storage_adapters/` — substrate I/O seam

| File | What it does |
|------|--------------|
| `reader.py` | Query-time reads, **Django-free** (raw psycopg2 via PgBouncer + Redis): `get_fk_adjacency`, `glossary`, `synonyms`, `value_samples`, `ann_search` (raw pgvector, `SET LOCAL hnsw.ef_search` inside an explicit txn), `verified_cache_lookup` / `verified_cache_exact`, `save_verified_query` (`INSERT … ON CONFLICT DO NOTHING` + rehydrate publish — the one documented inference-tier write). All scope via `context.current()` (fail-closed). |
| `writer.py` | Ingestion-time persistence via the Django ORM: `store_fk_adjacency`, `store_glossary`, `store_semantic_model`, `sync_from_engine` (pull FK / value-samples / glossary / graph from `veda_engine`), `_build_lite_sm_from_graph` (tabular/doc sources), `warm()`. `store_column_embeddings` raises `NotImplementedError` (target table dropped). |
| `assembler.py` | `SemanticModelAssembler` — `assemble(source, tenant)` rebuilds the `sm` dict from `Sm*` rows; `persist` is the inverse; `publish_sm` / `publish_registry` / `publish_rehydrate` push to redis-cache. |

### `veda_core/` — preserved engine + three platform seams

Preserved packages (moved verbatim from `veda-poc`): `veda/` (deterministic L1–L7 head +
firewall), `query/` (router, IR/SLM, SQL builder, resolvers, RAG / hybrid / NoSQL heads,
multi-source coordinator, federated route), `retrieval/` (6-signal spine), `ingestion/`
(offline L1–L5 build), `graph/`, `semantic/`, `glossary/`, `schema/`, `slm/`, `utils/`,
plus `config.py` (engine flags — the single source of truth) and `main.py` (a thin shim
into the layered ingestion pipeline).

New for the platform: `context.py` (ambient `RequestContext`), `slm/_call_slm.py` (SLM
Strategy seam), the Redis `sm` load in `veda_hybrid.py`, and `veda/rbac_filter.py` (Gate 1
engine-side filtering).

---

## 3. The request flow

### 3.1 Two-tier path (`POST /api/v1/query`)

```
client → nginx → QueryView.post (apps/query/views.py)
  ├─ empty query?                                    → 400  {"status":"invalid"}
  ├─ resolve_effective_permissions(user)             (RBAC; None when VEDA_RBAC_MODE=off)
  ├─ permitted_source_ids(user, effective)           empty set → 403 forbidden (no audit row)
  ├─ resolve_query_scope(data, tenant, user, eff)    SourceAccessDenied → 403 · NoReadySource → 503
  │      → source_ids (primary first);  source_id = source_ids[0]
  ├─ serialize_data_scope(compute_data_scope(...))   → X-Veda-Data-Scope  (omitted when None)
  ├─ source_profiles_for(source_ids)                 → X-Veda-Source-Profiles
  ├─ InferenceClient().run_hybrid_query(...)         InferenceUnavailable → 503 exec_error + QueryLog
  └─ 200 {status, result, latency_ms, request_id, cache_hit, usage}  + QueryLog
```

**Denials (403 / 503-unavailable) are not audited** — only successful inference calls and
`InferenceUnavailable` write a `QueryLog` row. `QueryLog.status` can also carry
non-canonical values (`"forbidden"`, `"unavailable"`) since Django `choices` don't validate
on `.create()`.

Tenant is **not yet principal-derived**: `views._resolve_tenant` returns `user.username`
when authenticated, else `data["tenant"] or "default"`.

### 3.2 Inside `run_hybrid_query` — the front door (`veda_core/veda_hybrid.py`)

`run_hybrid_query(query)` mints the one query trace and **always returns a `MultiResult`**
(a list of `SubResult`). Ordered steps, with current-default behavior:

| # | Step | Default behavior |
|---|------|------------------|
| 1 | **L0 NL simplifier** | `NL_SIMPLIFIER_ENABLED = False` → skipped. |
| 2 | **L0 runtime context** | `RUNTIME_CONTEXT_ENABLED = True`. Pure system-value questions ("what's the current date") answer here, before retrieval. |
| 3 | **Multi-source routing coordinator** | `MULTISOURCE_ROUTING_ENABLED = 1` **and** `MULTISOURCE_ROUTING_SHADOW = 1` → the coordinator computes a `RoutingDecision`, records it to the trace, and **returns `None`**. The legacy path answers. Only with `SHADOW=0` does the decision drive the answer (`NO_MATCH` → refuse, `ROUTED/SINGLE` → source agent, `ROUTED/MULTI` → federate / merge). See [`MULTI_SOURCE.md`](MULTI_SOURCE.md). |
| 4 | **Cross-source federated route** (`_maybe_federated`) | No-op unless the ambient context carries **≥ 2 source_ids**. Then `run_federated` → a `federated` `SubResult` with real cross-source SQL. |
| 5 | **Decomposition** | `QUERY_DECOMPOSE_ENABLED = False` ("splits join queries wrongly") → straight to `_dispatch_single(query)`. The whole `run_decomposer` / `_fan_out` / compound-`MultiResult` path is unreachable. |

### 3.3 `classify` — intent routing (`veda_hybrid.classify`)

Runs **before** the keyword router:

1. **Deterministic doc-intent override** — a fixed word list (`document` / `contract` /
   `policy` / `clause` / …) matches **and** the scope has chunk-backed sources → `rag`
   (or `hybrid` if the query also has an aggregation verb).
2. **Evidence-based doc-intent** — `DOC_INTENT_EVIDENCE_ENABLED` defaults **ON**. Reuses
   the coordinator's cosine evidence; a chunk-backed dominant source → `rag` / `hybrid`.
3. `QUERY_ROUTER_ENABLED = True` → `query/query_router.route_query` — a pure keyword
   counter (`_SQL_KEYWORDS`, `_RAG_KEYWORDS` ×discount-on-value-match, `_NOSQL_KEYWORDS`,
   `_TEMPORAL_KEYWORDS` ×2). **With no document/nosql sources configured it returns `sql`
   unconditionally.** There is no embedding fallback (the contract docs' `QUERY_ROUTER_CONFIDENCE_THRESHOLD` is imported and unused).
4. Router raises → `sql` (the safe default).

### 3.4 `_dispatch_single` — per-modality head

| intent | Head |
|--------|------|
| `sql` | `veda/pipeline.run_query` (the deterministic head, §4). On a refuse-class status (`refuse` / `qualifier_dropped` / `ungrounded` / `no_table` / `exec_error`) **and** `TIER2_LLM_FALLBACK = True` **and** the head took ≤ 120 s → `_tier2_sql` (LLM emits a UUID-only IR → deterministic builder → the same firewall → execute). `clarify` / `invalid` / `ir_mismatch` are **not** Tier-2-retried. The LLM never writes SQL structure, even in Tier-2. |
| `rag` | `query/rag_layer.run_rag_layer` — BGE-M3 dense + learned-sparse chunk retrieval → one local-SLM synthesis. |
| `hybrid` | Run the deterministic head first for correct-by-construction rows, feed those rows (as text, not as SQL) + graph chunks into `query/rag_layer.run_hybrid_layer` → one SLM call. The SQL head's own cols/rows/explain/analytics are attached to the result so a hybrid answer tables and charts like a plain SQL one. |
| `nosql` | `_run_nosql` → resolve source → `connectors.build_connector` → `query/nosql_builder.run_nosql_builder` (deterministic, no LLM) → `connector.execute_query`. |

### 3.5 Chat / SSE path (`POST /api/v1/conversations/query`)

`ConversationQueryView` (`AllowAny`, but an unauthenticated caller gets **401**) → the same
`apps/query/scope.py` helpers → `ConversationQueryService.run_turn` →
`chatbot.run.run_chat_turn` → a LangGraph `StateGraph`
(`memory_read → classify → {smalltalk | context_resolve | call_engine} → …`). The graph
calls the engine **only over HTTP**, via `InferenceClient.stream_hybrid_query`. A daemon
thread + `queue.Queue` bridges the graph's synchronous `on_event` callback into the SSE
generator. See [`CHAT.md`](CHAT.md).

---

## 4. The deterministic SQL head (`veda_core/veda/pipeline.py::run_query`)

The correctness head and the default route. It runs an **escalation ladder** and stops at
the first firewall-passing answer.

**Understand:** L1 temporal parse → intent is the literal string `"SIMPLE"`
(`IntentDetector` is **deliberately not wired**; query shape comes from the grammar
classifiers in `veda/planning.py`) → existence / aggregate / superlative / grouped / ratio
grammar classification.

**Ladder** (first firewall-passing answer wins):

1. **Fast path** (`FAST_PATH_ENABLED`, not existence) — SQL straight from compiled
   registries: no retrieval, no engine, no LLM.
2. **Deterministic superlative / grouped / ratio planners** (all default ON) — one-anchor
   analytical SQL without the LLM.
3. **Fast-path evidence guard** — a pick whose tables get zero typed evidence is
   **demoted** to the full pipeline (not refused).
4. **Verified cache** (skipped for existence and after a fast-path hit) — BGE-M3 cosine
   ≥ 0.85, re-checked by the evidence guard **and** a fresh qualifier-completeness pass on
   the cached SQL before it is trusted.
5. **Full path** — L2+ enhance (`QUERY_ENHANCEMENT_ENABLED = False`, so `_search = query`)
   → **L2 retrieve** `get_engine().retrieve(query, "SIMPLE", top_k=15)` (§5) → L2g graph
   expansion booster → RBAC candidate filter → **L2b primary cross-encoder rerank**
   (overwrites `final_score`) → **L3 anchor** `select_primary_table` + `vet_primary`
   (no primary → `no_table`; ambiguous subject → `clarify`) → Entity Resolution V1
   (pin an anchor, or clarify) → branch:
   - **join needed** (existence, or a join phrase / aggregate / grouped / ratio under
     `TYPED_MULTITABLE_ROUTE`, or ≥ 2 resolved entity tables) → `planning.try_multitable`:
     `clarify` / `refuse` / **existence** (deterministic EXISTS/NOT EXISTS, no LLM) /
     **aggregate** (deterministic pre-aggregation CTEs, no LLM, fan-out-free) / **sql**
     (planner pins the FROM/JOIN skeleton + `join_constraints` + `fanout_guard`; the LLM
     fills SELECT/WHERE only).
   - **single table** → a deterministic sub-ladder, each rung skipping the LLM when it
     fires: answer-entity → FK-value (`IN (subquery)`) → multi-hop FK → value-arbiter
     (categorical `=` / negated `!=`) → temporal-only → ranked-temporal-only →
     **temporal-refuse** (a temporal question on a table with no date column → refuse,
     never invent `created_at`) → **single-table LLM** (a deterministic builder is tried
     first for safe projection/date cases).

**Firewall** — every gate runs on the **original query** (there is a literal `assert`;
enhancement only ever reaches retrieval and the rerank text):

| # | Gate | Fail status | Notes |
|---|------|-------------|-------|
| 1 | **Value grounding** (`validation.value_grounding`) | `ungrounded` | every filter literal must exist in sampled data; skipped for datalake scopes. |
| 2 | **Qualifier completeness** (`validation.qualifier_completeness`) | `qualifier_dropped` / `access_denied` / `clarify` | every named qualifier represented. **Salvage:** if the missing token's referent table is outside the SQL, `run_query` retries once with a forced anchor; then a grounded clarify (real FK domain values); then RBAC reclassification. |
| 3–8 | **Silent-wrong alignment guards** (all default ON) — `grouped_shape_ok`, `distinct_shape_ok`, `intent_sql_alignment` (temporal + entity-anchor), `aggregate_presence_ok`, `filter_presence_ok`, `dimension_alignment` | `clarify` | catch "grouped by wrong column", "how many → 100 with no aggregate", "comparison phrase but no filter", "grouped by a column outside the requested family". |
| 9 | **Canonical-intent shadow** | — | observe-only, default OFF. |
| 10 | **IR equivalence** (`ir_equivalence.validate_ir_equivalence`) | `ir_mismatch` | **LLM-generated SQL only.** No filter / join / grouping / ordering / DISTINCT the query never licensed. |
| 11 | **Analytical semantics** (`semantic_validation`) | — | advisory in Tier-1 (trace only); enforceable in Tier-2. |
| 12 | **RBAC final gate** (`rbac_filter.narrow_allowed`) | — | the single allow-list choke point; a later reference to a trimmed name reclassifies to `access_denied`. |
| 13 | **AST validate + parameterize** (`validation.validate_and_parameterize` + `graph_guard`) | `invalid` / `access_denied` | single read-only SELECT; table/column existence; ON-integrity vs planned key pairs; every base-table join key a real FK edge; one connected component; fan-out guard; bind every literal. |
| 14 | **Execute** (`execution.execute_sql`) | `exec_error` | read-only session, 30 s timeout, fetch ≤ `EXECUTION_RESULT_LIMIT = 1000`. DuckDB for parquet scopes. |
| 15 | **NL-back answer** (`nl_answer` / `result_explainer`) | — | one SLM call (insight-engine XOR nl-answer); deterministic row-count fallback if the SLM is down. |
| 16 | **Cache-back** (`save_verified_query`) | — | only when not from cache / fast-path / temporal / existence and there are rows. |

**Terminal statuses:** `answered · no_table · clarify · refuse · ungrounded ·
qualifier_dropped · ir_mismatch · invalid · exec_error · access_denied`, plus
`tier2_rejected` / `tier2_exec_error` from Tier-2, and `runtime_context` / `federated*` /
`no_match` / `conflict` at the front door.

Full treatment: [`QUERY_ENGINE.md`](QUERY_ENGINE.md).

---

## 5. Retrieval spine (`veda_core/retrieval/retrieval_engine_phase3.py`)

`get_engine(sm).retrieve(query, intent, top_k=15)` — one warm engine per `(source, tenant)`
scope (LRU-capped), sharing one BGE-M3 model across scopes.

**Six signals**, fused by weighted RRF (`RRFMerger(k=60)`, `FUSION_WEIGHTS` — currently all
`1.0`, i.e. identity):

| # | Signal | Source |
|---|--------|--------|
| 1 | Dense semantic | BGE-M3 dense over `column_embeddings_v2`, raw query, HNSW `ef_search` per source |
| 2 | Learned-sparse | BGE-M3 lexical weights (**replaced BM25**), carries query enrichment |
| 3 | FK subgraph | static per-column scalar (`min(table_degree/10, 1)`) |
| 4 | FK path / join-key | static per-column scalar (0.5 FK, 0.7 referenced) |
| 5 | Value index | literal-in-query → the column that holds that value |
| 6 | Table-first prior | dense ⊕ sparse table affinity; soft — boosts existing candidates only |

Then `IntentBooster` (additive ±0.1–0.6 deltas on analytics role — these dwarf the RRF
score range) → `AdaptiveCutoff` (cut at the biggest score gap in `[5, 20)`).

**Encoder.** Single `BAAI/bge-m3` (`ingestion/m3_encoder.py`) — dense + learned-sparse, one
process singleton. `ENCODER_MODE` and the relgt / light-text / hybrid / MiniLM ensemble are
**gone**; `EMBEDDING_MODEL_ID = "bge-m3"` is the resume guard.

**Reranker.** `BAAI/bge-reranker-v2-m3` cross-encoder, on the primary path (after retrieve,
before anchor selection) + Tier-2 + RAG chunks. Pair text is **precomputed at ingestion**
(`ingestion/rerank_docs.py`); a missing artifact degrades to bare column names. If the
model isn't cached locally, every query silently degrades to pure RRF order.

**Graph expansion — two mechanisms:**
- **Tier-1 booster** (`graph/query_graph.suggest_expansions`, wired in `veda/pipeline.py`)
  — synonym/alias resolution + 1-hop FK join-key reach. Not PPR, not BFS.
- **Tier-2 / datalake / cross-source** (`query/graph_retriever.run_graph_retrieval`) —
  genuine **Personalized PageRank** (`d = 0.85`, symmetric row-normalized transition
  matrix) over the unified graph, replacing an older hop-decay BFS.

Full treatment: [`RETRIEVAL.md`](RETRIEVAL.md).

---

## 6. The substrate (`apps/substrate/models.py`)

Every model inherits `TenantScopedModel` (UUID PK matching ingestion UUIDs + `source` FK +
`tenant` + timestamps). `TenantManager.get_queryset` filters by `context.current()` and
falls back to unscoped when no context is set; `all_tenants()` is the explicit escape hatch
(ingestion writers use it directly).

| Group | Models | Backs |
|-------|--------|-------|
| Structural | `SchemaTable`, `SchemaColumn`, `FkEdge`, `TableMetadata` | schema scan; `FkEdge` is the join engine's FK source of truth (undeclared FKs from the data graph carry `is_declared=False`, `overlap_score`). |
| Semantic / language | `SemanticType`, `GlossaryEntry`, `Synonym`, `SyntheticPair`, `SemanticConcept` | semantic-type inference, glossary/synonyms, compiled concepts. |
| Value grounding | `ColumnValueSample`, `ColumnProfile` | value sampler (mirrored to Redis) + profiler. |
| Embeddings (`managed=False`) | `ChunkEmbedding`, `GraphNodeEmbedding` | pgvector mirrors for **admin visibility only**. The legacy `ColumnEmbedding` / `_LT` / `_Hybrid` / `_BGE` / `RelgtStructural` models and tables were **dropped** (migrations 0006–0008). The live ANN store `column_embeddings_v2` / `table_embeddings_v2` is engine-owned in `veda_engine` — no Django migration. |
| Graph | `GraphNode`, `GraphEdge`, `GraphArtifact` | unified KG for expansion (Kùzu removed). |
| Verified cache | `VerifiedQueryCache` | hot-path write via `ON CONFLICT`; `query_embedding vector(1024)` added by RunSQL. |
| Normalized `sm` | `SubstrateVersion`, `SmTable`, `SmColumn`, `SmRetrievalDoc`, `SmSynonym`, `SmConcept` | the read-model the assembler rebuilds; `SubstrateVersion` drives rehydrate + carries `hnsw_ef_search`. |

`QueryLog` (`apps/query/models.py`) is the audit mirror (plain model, not tenant-scoped):
query text, tenant, route, terminal status, `executed_sql` (placeholder text only),
`refusal_reason`, `latency_ms`, `cache_hit`, `request_id`, token counts.

---

## 7. Ingestion (`apps/ingestion/tasks.py` → the engine)

`task_ingest_source(source_id, tenant, force, skip_llm, resume)` is the wired path:

- creates an `IngestionJob` + ordered `IngestionStage` rows;
- guards against an embedding-model change vs the last successful job (`EMBEDDING_MODEL_ID`);
- injects the source's connection via `Source.as_engine_env()` → `VEDA_SOURCE_*`;
- runs the engine in a **subprocess** (`cwd=veda_core`, isolates the engine's top-level
  `config` module from the Django `config` package), streaming `[[STAGE]]` events into the
  stage rows;
- on success → `task_warm_caches` → `writer.warm()` → flips `Source.ready=True`.

**One pipeline, not two.** `veda_core/main.py::run_ingestion` is now a thin shim →
`ingestion/dispatcher.dispatch` → `ingestion/layers/pipeline.run_layered_ingestion`, which
composes **L1 EXTRACT → L2 ANALYZE → L3 ENRICH → L4 INDEX → L5 PUBLISH**, threading one
in-memory `state` dict with per-stage fatal semantics. The "monolith" described in the old
`INGESTION.md` no longer exists; the layer modules are thin wrappers over the same stage
functions. Non-relational sources (document / nosol / datalake) route to
`ingestion/source_dispatcher.dispatch_ingestion` instead.

`writer.warm()`: persist the semantic-model file into the `Sm*` substrate, `sync_from_engine()`
(FK / value samples / glossary / KG from `veda_engine`), then `publish_sm` + `publish_rehydrate`
so every inference replica clears its `_SM` cache and reloads.

Full treatment: [`INGESTION.md`](INGESTION.md). Operator runbook: [`INGESTION_GUIDE.md`](../INGESTION_GUIDE.md).

---

## 8. Multi-source & federation

Onboarding is a data operation: register a `Source` row, `POST /api/v1/admin/ingest`,
the query path reads only `ready=True` sources.

Three routing surfaces exist in `veda_core/query/`:

- **The keyword router** (`query_router.py`) — SQL/RAG/hybrid/NoSQL modality within a
  scope. Live.
- **The multi-source coordinator** (`source_coordinator.py`, `routing_policy.py`,
  `agents.py`, …) — which source(s) answer a question, presence tiers, per-source agents,
  independent-result merge with conflict detection. **Runs in shadow only**
  (`MULTISOURCE_ROUTING_SHADOW = 1`): computes and traces a decision, does not act on it.
- **The federated route** (`federated_route.py`, `federated_executor.py`,
  `cross_source_composer.py`) — when the ambient scope spans ≥ 2 sources and retrieval hits
  more than one, generate + run a cross-source query (DuckDB over materialized parquet +
  live sources). Live.

Cross-source links come from ingestion: MinHash `column_sketches` per source (L2) →
`cross_source_graph.discover_and_persist` (tenant-wide, L5) → `cross_source_fk` edges;
plus the document side (`entity_linker` → `entity` nodes → `value_of` columns) and the
semantic bridge (`semantic_linker` → `semantic_about` chunk→column edges, traversed by PPR).

Full treatment: [`MULTI_SOURCE.md`](MULTI_SOURCE.md). Deploy runbook:
[`MULTI_SOURCE_DEPLOYMENT.md`](MULTI_SOURCE_DEPLOYMENT.md).

---

## 9. RBAC, authentication, tenancy

Two apps, deliberately separate:

- **`apps/authentication`** — verifies identity. Login / refresh / logout / password-change,
  JWT via `simplejwt` behind `VEDA_JWT_AUTH` (default **off** — login then returns a
  placeholder token and refresh 401s). Two-tier Redis lockout (per-IP hard, account-wide
  soft), fail-open on a cache outage. Stock `django.contrib.auth.User`; no custom user model,
  no `launchpad` DB, no identity router.
- **`apps/access_management`** — the RBAC model + enforcement. `User`–`UserRole`–`Role`–
  `RolePermission`–`Permission`, where `RolePermission.resource_path` is a **canonical
  string** (`db:crm:employee:salary` — [ADR-0001](adr/0001-rbac-resource-path.md)), not an
  FK, with `effect ∈ {allow, deny}`. `CatalogResource` is an addressable projection of the
  substrate. `PermissionResolver` = deny-wins + strict hierarchy (a table ALLOW grants
  nothing without a source-level ALLOW).

Enforcement is **built and wired end-to-end, flag-gated off** by `VEDA_RBAC_MODE`
(`off` / `shadow` / `enforce`):

- **Gate 2** — `RequiresPermission` DRF permission class on every admin `AdminView`,
  alongside `IsAdminUser` (can only narrow).
- **Gate 1** — in `apps/query/views.py` + `apps/chat/views.py`: `permitted_source_ids`
  narrows the source set; `compute_data_scope` builds a table/column allow payload
  forwarded as `X-Veda-Data-Scope`; the inference middleware parses it (fail-closed) onto
  `RequestContext.allowed_resources`; `veda/rbac_filter.py` applies it at retrieval
  candidates, the `narrow_allowed` choke point, NoSQL collections, and doc chunks — all
  identity no-ops when `allowed_resources is None`.

| Concern | Mechanism |
|---------|-----------|
| Tenant isolation | ambient `RequestContext` (`current()` raises when unset); ORM auto-filter; raw reads call `_scope()`. |
| Tenant source of truth | server-resolved, forwarded as `X-Veda-Tenant`; never client-supplied for scoping. **Not yet principal-derived** (Phase 6.2). |
| Read-only source access | `execute_sql` session `readonly=True`; DML/DDL AST reject; graph guard + fan-out. |
| Parameterized SQL | every literal bound; `QueryLog.executed_sql` stores placeholder text. |
| Secrets by reference | `Source.password_env` / `connection_secret_ref`. |
| Zero-egress | `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` set before any model import; SLM on the internal network. |
| Request tracing | `RequestIdMiddleware` mints/propagates `X-Request-Id` → forwarded → `QueryLog.request_id`. |

Full treatment: [`RBAC.md`](RBAC.md).

---

## 10. SLM backend seam (`veda_core/slm/_call_slm.py`)

`call_slm(prompt, *, purpose, timeout, temperature=0.0, …) -> str` over a Strategy:
`OllamaBackend` (`/api/chat` + `/api/generate`, `keep_alive:"24h"`) and `VLLMBackend`
(OpenAI-compatible `/v1/chat/completions`). Backend from `SLM_BACKEND` (default `ollama`
dev, `vllm` prod), cached per process. ~41 engine call sites use `call_slm`; three modules
(`query/slm_layer.py`, `query/answer_entity.py`, `query/rag_layer.py`) still hold direct
Ollama calls. `_slm_circuit_breaker` is a pass-through skeleton. Token usage is captured
per call and fanned out to `QueryLog`, `ChatMessage.metadata`, and the explain trace.

`SLM_MODEL_NAME` and `SLM_TEMPERATURE` are read from `.env`; the `config.py` defaults
(`qwen2.5-coder:7b`, `0.3`) differ, so a machine without the `.env` override runs a
different model and non-deterministic intent.

---

## 11. Deployment (`docker/`, compose)

**12 dev services** on `veda_net` (13 with the `proxy` profile); only `nginx` publishes a
host port (`8080`). `api` / `worker` / `beat` build the thin `Dockerfile.api` (no torch);
`inference` / `ingest-worker` build `Dockerfile.inference` (torch). `postgres` is
**`pgvector/pgvector:pg17`** and hosts both `veda` and `veda_engine` (dev exposes `15432`).
`pgbouncer` fronts every pool (transaction mode). Redis is triple-split: `redis-broker`
(Celery), `redis-cache` (`allkeys-lru` — assembled `sm` + rehydrate pub/sub), `redis-stack`
(`:6380`, RediSearch — the chat checkpointer). `ollama` is the dev SLM; prod adds `vllm`.

Overrides: `docker-compose.prod.yml` (adds `vllm`, GPU reservations, secrets, replicas),
`docker-compose.demo.yml` (query-only — a shipped dump, no ingestion, `VEDA_ALLOW_ANONYMOUS`),
`docker-compose.mlflow.yml` (adds the MLflow sidecar + exporter — opt-in).

**Prod is not deployable as-is.** `docs/PRODUCTION_READINESS_PLAN.md` B1–B13 (secrets not
read from `/run/secrets`, TLS block commented out, no `release` migrate service, no prod
`ingest-worker`, torch unpinned, vLLM not offline) are all still open. `docs/OPERATIONS.md`
"verified" security bullets are aspirational.

Runbooks: [`DEPLOYMENT.md`](DEPLOYMENT.md), [`OPERATIONS.md`](OPERATIONS.md),
[`MAC3_APP_HOST_SETUP.md`](MAC3_APP_HOST_SETUP.md), [`DEMO_QUERY_ONLY.md`](DEMO_QUERY_ONLY.md).

---

## 12. Observability & health

- **Explain trace** — the engine appends one JSON line per query to
  `veda_core/logs/explain_trace.jsonl` (`EXPLAIN_TRACE_* = True` by default).
- **`mlflow_observability/`** — a **separate process** that tails that file into MLflow.
  Nothing imports it; it never imports the engine. **Not on by default** — no exporter/UI
  in the dev or prod compose; start `docker-compose.mlflow.yml` or
  `python -m mlflow_observability watch`.
- **api tier**: `/healthz` (static), `/readyz` (gates on Postgres + both Redis + inference;
  SLM probed but non-gating; no BGE probe), `/metrics` (Prometheus text from `QueryLog` +
  PgBouncer `SHOW POOLS`, dependency-free).
- **inference tier**: `/healthz` + `/readyz` (`ready` == semantic-model file present). **No
  `/metrics`** — its per-query observability is the explain-trace path.

Full treatment: [`OBSERVABILITY.md`](OBSERVABILITY.md).

---

## 13. Configuration

- **Engine flags** live in `veda_core/config.py` (the single source of truth) and reach
  Django only through `apps/core/settings_bridge.build_veda_settings()` — precedence
  `fallback default → config.py value → VEDA_<FLAG> env`. 14 flags are bridged
  (`EMBEDDING_MODEL_ID`, `TOP_K`, `TOP_K_TO_LLM`, `QUERY_ROUTER_ENABLED`, `SLM_MODEL_NAME`,
  `SLM_BACKEND`, `VLLM_BASE_URL`, `IR_JOIN_FREE_ENABLED`, `FAST_PATH_ENABLED`,
  `QUERY_DECOMPOSE_ENABLED`, `HNSW_M` / `_EF_CONSTRUCTION` / `_EF_SEARCH`).
- **Infra** (DB, Redis, secrets) is env-only.
- **Per-source / runtime env**: `VEDA_SOURCE_*`, `VEDA_INTERNAL_*`, `INFERENCE_URL`,
  `PGBOUNCER_*`, `REDIS_CACHE_URL` / `REDIS_BROKER_URL`, `CHATBOT_CHECKPOINTER_REDIS_URL`,
  `OLLAMA_URL` / `VLLM_URL`, `METAL_EMBED_URL`, `VEDA_SM_REDIS`, `VEDA_HNSW_EF_SEARCH[_<id>]`,
  `VEDA_RESUME`, `VEDA_DEFAULT_SOURCE_ID`, `VEDA_JWT_AUTH`, `VEDA_RBAC_MODE`.

---

## 14. Scenario → code map

### Query-time

| Scenario | Path | Outcome |
|----------|------|---------|
| Plain NL question | `QueryView` → resolve scope → `InferenceClient` → inference `_tenant_context` → `run_hybrid_query` → `_dispatch_single` → head → firewall → `MultiResult` → `QueryLog` | `answered` / a refusal |
| Router off / single relational source | `classify` → `route_query` → `sql` unconditionally | deterministic head |
| Count / aggregate ("how many users") | `run_query` → `try_fast_path` (no retrieval / LLM) → firewall → execute | `answered`, fast |
| Repeat of a verified query | `run_query` → `verified_cache_lookup` (pgvector, scoped) + evidence + qualifier re-check | `answered`, `cache_hit=True` |
| Join query | `try_multitable` pins the skeleton from `FkEdge`; LLM fills SELECT/WHERE; graph guard | `answered` / `invalid` / `refuse` |
| "with / without X" | `existence_mode` → deterministic EXISTS/NOT EXISTS; never cached | `answered` |
| Filter value absent | `value_grounding` fails | `ungrounded` |
| User qualifier dropped | `qualifier_completeness` fails → salvage re-anchor → grounded clarify | `qualifier_dropped` / `clarify` |
| LLM added unrequested semantics | `ir_equivalence` fails | `ir_mismatch` |
| Grouped by the wrong column | `dimension_alignment` fails | `clarify` |
| "how many" answered without an aggregate | `aggregate_presence_ok` fails | `clarify` |
| Hallucinated table/column / write attempt | `validate_and_parameterize` / session read-only | `invalid` |
| Temporal question, no date column | single-table temporal-refuse branch | `refuse` |
| No anchor | `select_primary_table` empty | `no_table` |
| Head refuses + Tier-2 on | `_dispatch_single` → `_tier2_sql` (LLM IR → builder → same gates → execute) | `answered` or `tier2_rejected` |
| Doc / policy question | `classify` doc-intent override → `run_rag_layer` | `answered` (rag) |
| Doc + data question | `classify` → `hybrid` → head rows + chunks → one SLM call | `answered` (hybrid) |
| Mongo / native source | `classify` → `_run_nosql` → `nosql_builder` | `answered` (nosql) |
| Scope spans ≥ 2 sources | `_maybe_federated` → `run_federated` (DuckDB) | `federated` |
| RBAC denies every source (enforce) | `permitted_source_ids` empty | `403 forbidden` (no audit row) |
| Inference slow / unreachable | `InferenceClient` raises `InferenceUnavailable` → `503` + `exec_error` audit | no 500 / hang |
| Empty query | `QueryView.post` early return | `400` |
| Chat follow-up ("and for 2024?") | `chatbot` `memory_read` → `classify` (merged delta) → `context_resolve` (`render_frame_as_query`) → `call_engine` | resolved standalone query |

### Ingestion & lifecycle

| Scenario | Path | Effect |
|----------|------|--------|
| Onboard a source | register `Source` → `IngestTriggerView` → `task_ingest_source` reads `as_engine_env()` → subprocess → L1–L5 | source-specific substrate, no code change |
| Live stage progress | subprocess `[[STAGE]]` events → `_LAYER_STAGE_TO_ROW` → `IngestionStage` updates | per-stage progress |
| Embedding model changed w/o force | guard vs last successful job → raise | re-ingestion required |
| Resume a failed job | prior FAILED or `resume=True` → `VEDA_RESUME=1` | L3 skips if the sm file exists, L4 biencoder skips if `column_embeddings_v2` non-empty; L1/L2 always re-run |
| Fast structural-only | `skip_llm=True` | L3 semantic-layer LLM stage skipped |
| Partial failure | exception → stages/job FAILED, `Source.status=FAILED`, `ready` stays False | query path never reads half-built substrate |
| Warm after ingest | `task_warm_caches` → `writer.warm()` → `assembler.persist` + `sync_from_engine` + `publish_sm/rehydrate` | Django owns the substrate; replicas notified |
| Re-ingest reaches replicas | `publish_rehydrate` → redis pub/sub → inference subscriber drops `_SM` / engines / caches | next query reloads |

### Platform / ops

| Scenario | Path | Effect |
|----------|------|--------|
| Liveness | `GET /healthz` | `{"status":"ok"}` |
| Readiness | `GET /readyz` → `core/views.readyz` | `ready` (200) / `degraded` (503) |
| Metrics | `GET /metrics` → `core/views.metrics` | Prometheus text, dependency-free |
| Eval run | `EvalTriggerView` → `task_run_eval` → inference → `EvalRun` / `EvalCaseResult` + HTML | tracked artifact |
| Per-query trace | engine → `explain_trace.jsonl` → (opt-in) `mlflow_observability watch` → MLflow | one run per query |
| Trace across tiers | `RequestIdMiddleware` → forwarded → `QueryLog.request_id` | one id api → inference → logs |

---

## 15. Status — wired vs skeleton (from the code)

**Wired end-to-end:** the two-tier query flow (api → inference → `run_hybrid_query` → head
→ firewall → audit); the chat SSE turn (`apps/chat` → `chatbot` graph → inference);
`task_ingest_source` real ingestion (streamed stages, resume, skip-LLM, embedding-model
guard, per-source connection injection); the layered L1–L5 ingestion pipeline;
`storage_adapters.reader` / `writer` (all except `store_column_embeddings`);
`SemanticModelAssembler` + Redis `sm` + rehydrate fan-out; the SLM Strategy backends; the
RBAC data model + Gate 1 + Gate 2 + engine-side `rbac_filter` (**flag-gated off**); the JWT
auth stack (**flag-gated off**); health / readiness / metrics; eval runs; the
`mlflow_observability` package (opt-in); the 6-signal retrieval spine + cross-encoder
rerank; the deterministic planners and the ~10-gate firewall; the cross-source federated
route.

**Skeleton / dormant / dead:**
- `query_decompose` (whole compound path — `QUERY_DECOMPOSE_ENABLED = False`).
- The multi-source coordinator's authoritative branches (`MULTISOURCE_ROUTING_SHADOW = 1`).
- `veda/understanding/` (5 files), `veda/analytical_spec.py`, `veda/query_enhancement.py`,
  the Tier-2 repair loop — all flag-OFF.
- `veda/routing_slm.py` — empty file. `veda/canonical_intent_shadow.py` — observe-only.
- `query_engine/intent_detector.py` — deliberately not called by the head.
- `inference/engine.py::get_engine`, `storage_adapters.writer.store_column_embeddings`,
  `_slm_circuit_breaker`, the ten-task Celery ingestion chain — all `NotImplementedError`
  or pass-through.
- `ingestion/chunk_linker.py` — 0 callers, superseded by `entity_linker.py`.
- Tenant-from-principal, per-source HNSW auto-tuning at scale (`artifact_scope` OFF by
  default), prod deployment hardening (B1–B13).

**Recently fixed (2026-09-10):** `storage_adapters/reader.py::ann_search` was reading
`column_embeddings_v2` on the `veda` connection, but that table lives in `veda_engine`, so
Signal-1 dense retrieval was silently falling back to an **unscoped** engine-store query
(leaking candidates across sources in a multi-source scope). `reader.py` now uses a
dedicated `_internal_connection()` for the vector scan. Still worth a live confirmation —
see [`backlog/query-engine-open-items.md`](backlog/query-engine-open-items.md).
