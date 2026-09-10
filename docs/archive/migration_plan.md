# VEDA — Django Migration & Production Architecture

**Full target specification + phased build runbook for an AI agent**

*Deep Django · DRF · Celery · Postgres+pgvector · Redis · Dedicated Inference Service · Docker Compose*

> Flow preserved verbatim from ARCHITECTURE.md. Only ingestion persistence and query-time retrieval are re-homed into Django-native infrastructure.

---

## Contents

- [0. How an AI agent should use this document](#0-how-an-ai-agent-should-use-this-document)
- [1. Executive summary](#1-executive-summary)
- [2. Migration principles (non-negotiable)](#2-migration-principles-non-negotiable)
- [3. Container topology (Docker Compose)](#3-container-topology-docker-compose)
- [4. Target repository layout](#4-target-repository-layout)
- [5. Django apps and responsibilities](#5-django-apps-and-responsibilities)
- [6. The substrate data model — store everything ingestion produces](#6-the-substrate-data-model--store-everything-ingestion-produces)
- [7. Ingestion pipeline as Celery tasks (L0)](#7-ingestion-pipeline-as-celery-tasks-l0)
- [8. The inference service — warm load, memory hydration, query path](#8-the-inference-service--warm-load-memory-hydration-query-path)
- [8a. Semantic-model assembly — normalized rows → the `sm` dict](#8a-semantic-model-assembly--normalized-rows--the-sm-dict)
- [8b. The SLM backend seam — `_call_slm` Strategy (Ollama · vLLM)](#8b-the-slm-backend-seam--_call_slm-strategy-ollama--vllm)
- [9. Configuration, settings, and secrets](#9-configuration-settings-and-secrets)
- [9a. Cross-cutting design patterns & Django standards](#9a-cross-cutting-design-patterns--django-standards)
- **Part B — Phased build runbook**
  - [Phase 0 — Scaffold & preserve](#phase-0--scaffold--preserve)
  - [Phase 1 — Infrastructure & containers](#phase-1--infrastructure--containers)
  - [Phase 2 — Substrate models & migrations](#phase-2--substrate-models--migrations)
  - [Phase 3 — storage_adapters (the seam)](#phase-3--storage_adapters--the-seam)
  - [Phase 4 — Ingestion pipeline (Celery)](#phase-4--ingestion-pipeline-celery)
  - [Phase 5 — Inference service](#phase-5--inference-service)
  - [Phase 6 — DRF API, auth, tenancy, audit](#phase-6--drf-api-auth-tenancy-audit)
  - [Phase 7 — Parity, hardening, production](#phase-7--parity-hardening-production)
- [17. Parity testing strategy](#17-parity-testing-strategy-how-the-agent-proves-flow-unchanged)
- [18. Risk register & mitigations](#18-risk-register--mitigations)
- [19. Appendix — flow-preservation checklist](#19-appendix--flow-preservation-checklist)

---

## 0. How an AI agent should use this document

This document has two halves. **Part A** (§1–§9a) is the **target architecture specification** — what the system must look like when finished: every Django app, model, service boundary, container, and data-flow. **Part B** (Phases 0–7) is the **phased runbook** — an ordered sequence of concrete, verifiable tasks the agent executes to get there.

**Execution contract for the agent:** work strictly phase by phase. Do not begin a phase until the previous phase's **Exit criteria** all pass. Every task lists its inputs, the exact files to create or move, and an acceptance check. Where a task says **PRESERVE**, the referenced logic is copied verbatim — the agent must not refactor, rename, or "improve" it. The query flow described in ARCHITECTURE.md is frozen.

> ### ⚠️ The one rule that governs everything
> **The runtime query flow does not change.** `run_hybrid_query → router → {sql | rag | hybrid | nosql} → firewall → MultiResult` stays byte-for-byte behaviourally identical. We are only changing (a) where the ingestion substrate is stored, and (b) how it is loaded and served at query time. Every escalation-ladder rung, every firewall gate, every refusal status is preserved.

---

## 1. Executive summary

VEDA today is a set of Python packages (`veda/`, `query/`, `retrieval/`, `ingestion/`, `connectors/`, `graph/`, `semantic/`) driven by `main.py` and `veda_hybrid.py`. Ingestion writes a grounding substrate into Postgres+pgvector, Kùzu, and file caches; query time reads that substrate through a process-wide warm engine (`get_engine()`).

The production target wraps this in a **deep Django platform** with four cooperating containers: (1) a **Django/DRF API + admin** tier that owns HTTP, auth, tenancy, all substrate models, and orchestration; (2) a **Celery ingestion** tier that runs the L0 pipeline as tracked, resumable jobs; (3) a dedicated **inference/query service** (ASGI) that warm-loads the heavy models and the 5-signal engine once per process and serves retrieval + the full `run_hybrid_query` flow; and (4) shared **Postgres+pgvector** and **Redis** backing stores. Everything ingestion produces is persisted in Postgres (relational + pgvector) and hot-cached in Redis and in-process memory for query-time speed.

### Why a separate inference service (the key production decision)

| Approach | Verdict | Reason |
|---|---|---|
| Warm singleton inside each Gunicorn worker | **Rejected** | BGE-M3 (1024-dim) + bge-reranker-v2-m3 are hundreds of MB each. Loading them in every sync worker multiplies RAM by worker count and blocks the event loop during inference. |
| Reload models per request | **Rejected** | Cold-load latency (seconds) on the hot path violates the fast-retrieval requirement. |
| Dedicated ASGI inference service, warm-loaded once, scaled independently | **Chosen** | Preserves `get_engine()` semantics exactly (one warm engine per process), keeps Django workers thin, and lets you scale retrieval CPU/GPU separately from API traffic. Standard production ML-serving pattern. |

Django never loads a transformer. It calls the inference service over HTTP (or gRPC) and persists/returns results. This is the only structural change to the runtime path, and it is transparent to the flow: the inference service runs the identical `veda_hybrid.run_hybrid_query` code.

### Two decisions to lock before writing code

Both come out of the production review and are the only changes that are *redesign* rather than tightening:

1. **Query-time SLM runs on a swappable backend, not hardwired to Ollama.** All SLM calls (IR emit, decomposer, RAG synthesis, NL answer) go through a single `_call_slm` Strategy interface (§8b) with two config-flagged backends: Ollama (dev/ingestion) and vLLM (production hot path). vLLM's continuous batching removes the single-instance serialization bottleneck that Ollama imposes across N inference replicas. The interface is the contract; the backend is a flag.
2. **The HNSW-vs-exact-cosine parity contradiction is resolved before the gate can pass.** The retrieval parity gate (§17) must run against the *actual HNSW config that ships* — not exact search — with an `ef_search`/`m` sweep tuned until recall@k = 1.0 on the fixtures (§7.1a). Letting parity pass under exact cosine and then shipping HNSW is the single most likely silent production regression in the whole plan.

---

## 2. Migration principles (non-negotiable)

- **Flow immutability.** The escalation ladder, router, decompose asymmetry, unified firewall, and terminal statuses (`answered · no_table · clarify · refuse · ungrounded · qualifier_dropped · ir_mismatch · invalid · exec_error`) are preserved exactly. Code that implements them is relocated, not rewritten.
- **Substrate is data, engine is code.** Everything ingestion derives (FK graph, embeddings, semantic metadata, glossary, value samples, knowledge graph, doc chunks) becomes first-class Django models and pgvector tables. The reasoning/compiler code stays as an importable library.
- **Two authoritative sources of truth, unchanged.** `config.py` remains the single source of truth for flags/thresholds/model names (surfaced through Django settings). `fk_adjacency` remains the join engine's source of truth (now a Django model).
- **Derived, never hardcoded.** Nothing schema-specific is baked into models. Semantic types, glossary, synonyms, and synthetic pairs are still LLM/rule-derived at ingestion and stored per source.
- **Refuse-over-guess, end to end.** Every new API boundary returns the same refusal semantics. A validation or grounding failure surfaces as a structured refusal, never a 500 or a silent empty result.
- **Tenancy first.** Every substrate row is scoped to a `(source, tenant)` key from day one, so multi-tenant deployment needs no schema change later.
- **Idempotent, resumable ingestion.** Re-running ingestion for a source is safe and picks up where it left off; each L0 stage is a checkpointed Celery task.
- **Zero-egress preserved.** All inference stays local (SLM backend + local BGE/reranker inside the deployment). No client data leaves the deployment. Container network is closed except for the API ingress.
- **Tenancy is ambient, not a parameter.** `(source, tenant)` travels through a request/task-scoped `contextvar` (§4.1), read only inside `storage_adapters`. The engine's public signatures (`run_query`, `get_engine().retrieve`, `verified_cache_lookup`) are frozen and stay tenant-oblivious. Fail-closed: an unset context raises, never defaults to a tenant. The offload primitive that carries the context into the thread pool is wrapped **once** and lint-enforced (§4.1) — coverage is not left to discipline.
- **The `sm` dict is a rebuilt read-model.** The engine consumes one nested `sm` dict, not normalized rows; a `SemanticModelAssembler` (§8a) reconstructs it byte-identically from the substrate. Its deep-equality with the legacy `veda_semantic_model.json` is a parity gate.
- **Retrieval parity is measured against the shipping index, not a stand-in.** The parity gate runs against the live HNSW configuration, tuned to recall@k = 1.0 on fixtures (§7.1a). "Flow frozen" includes the retrieval ordering the flow actually sees in production.
- **Inference is read-mostly, not read-only.** The verified-query cache is written *on the query hot path* (`save_verified_query`); that single write is idempotent (`INSERT … ON CONFLICT`) and off the latency path. New cache entries are broadcast to peer replicas via the same pub/sub fan-out as rehydrate (§8.4) so the fleet converges. Everything else the inference tier touches is a read.

---

## 3. Container topology (Docker Compose)

Nine services in production. All application containers share one internal Docker network (`veda_net`); only `nginx` exposes a port to the host.

| Service | Image / base | Role | Scales |
|---|---|---|---|
| **nginx** | `nginx:alpine` | TLS termination, reverse proxy, static/media, upload size limits, rate limiting. Single ingress. | 1 (or LB) |
| **api** | `python:3.11-slim` + Django/DRF + Gunicorn | HTTP API, admin, auth, tenancy, all substrate ORM writes, ingestion orchestration. **Loads NO models.** | N (stateless) |
| **inference** | `python:3.11-slim` + Uvicorn (ASGI) | Warm-loads BGE-M3, bge-reranker-v2-m3, MiniLM ensemble, 5-signal engine, FK graph handles, verified cache. Serves `/retrieve` and `/run_hybrid_query`. **This IS `get_engine()`.** | N (CPU/GPU) |
| **worker** | same image as api + Celery worker | Runs L0 ingestion as checkpointed tasks (schema scan → embeddings → glossary → graph). Queues: `ingestion`, `high`, `default`. | N |
| **beat** | same image + Celery beat | Scheduled re-ingestion, cache warmers, TTL sweeps, heartbeat pings. | 1 |
| **pgbouncer** | `pgbouncer/pgbouncer` (or `edoburu/pgbouncer`) | Transaction-pooling proxy in front of Postgres. Every api/worker/inference pool connects **through** it so N workers × M replicas × pool size cannot exceed Postgres `max_connections`. | 1 (or per-tier) |
| **postgres** | `pgvector/pgvector:pg16` | Relational substrate + pgvector embedding tables + audit log + job state. Two logical DBs: internal store and (optionally) source registry. | 1 (HA later) |
| **redis-broker** | `redis:7-alpine` | Celery broker + result backend **only**. Unbounded memory; no eviction policy. | 1 (cluster later) |
| **redis-cache** | `redis:7-alpine` | Django cache + hot substrate indices (FK adjacency, glossary, display cols) + retrieval memoization + pub/sub. `maxmemory` + `allkeys-lru`. | 1 (cluster later) |

> ### ⚠️ Split Redis: a cache-eviction storm must not touch the broker
> The dev/single-node profile may run one Redis with separate logical DBs, but **production runs two Redis instances** (`redis-broker` and `redis-cache`), not two DBs on one process. Redis holds the Celery broker, the hot indices, the query cache, and the rehydrate pub/sub — if a cache-eviction storm (allkeys-lru under load) or an OOM hits a *shared* process, ingestion, the query hot path, and rehydrate fan-out all degrade simultaneously. Isolating `maxmemory` to the cache DB is not enough when they share a process. Separate instances; the broker instance is unbounded and never evicts.

> ### ⚠️ Postgres connection ceiling: front every pool with PgBouncer
> Each inference worker holds a source pool + an internal-store pool (§8.1 item 7). `N inference workers × M replicas × pool_size`, plus api/worker pools, blows past Postgres `max_connections` well before any CPU limit — horizontal scaling hits a connection ceiling first. Route every pool through **transaction-pooling PgBouncer** (the same pattern already used on the OCS side). Set PgBouncer `pool_mode = transaction`; keep prepared-statement usage compatible (or set `max_prepared_statements` appropriately for pg16).

### SLM backend placement (Ollama for dev/ingestion, vLLM for the production hot path)

SLM inference is reached only through the `_call_slm` Strategy (§8b), so the backend is a deployment choice, not a code change:

- **Ollama** (`ollama/ollama`, an additional service on `veda_net`) remains the default for **local dev** and for **ingestion-time** generation (glossary, synthetic pairs) called by `worker`, where throughput is not latency-critical.
- **vLLM** (`vllm/vllm-openai` or equivalent) is the **production query-time** backend reached by `inference`. Its continuous batching serves many concurrent SLM calls from N replicas without the single-instance serialization Ollama imposes. Keep the model name in settings (`SLM_MODEL_NAME=qwen2.5-coder:7b`) and the backend in `SLM_BACKEND={ollama|vllm}`. GPU is attached to `inference`, the SLM backend, and `ollama` only.

> ### ⚠️ A single SLM instance serializes every SLM call across the fleet
> With Ollama on the hot path, all `inference` replicas plus `worker` funnel SLM IR emit, decomposer, RAG synthesis, and glossary generation through one process, each with a 240s timeout — a serialization bottleneck. The primary mitigation is moving the **query-time** SLM to vLLM (§8b), whose batching absorbs fleet concurrency. Regardless of backend, keep a **bounded concurrency gate / queue** in front of the SLM service and scale it with the inference tier, preserving GPU affinity. The existing SLM timeout, circuit breaker, and deterministic fallbacks (row-count NL answer, refuse-over-guess) are preserved, so saturation degrades to a fallback/refusal — never a hung request.

> ### Request path at runtime (unchanged flow, new transport)
> ```
> client → nginx → api (DRF QueryView) → HTTP → inference.run_hybrid_query
>        → [pgvector reads via PgBouncer + redis-cache hot indices + SLM backend] → MultiResult
>        → api serializes → client
> ```
> The `api` tier does auth, tenant resolution, rate limit, request logging, and audit persistence. The `inference` tier does the actual VEDA flow with warm models. Retrieval reads pgvector directly (fast, indexed, pooled through PgBouncer) and `redis-cache` for the hottest indices.

---

## 4. Target repository layout

A single repo, two runnable Python roots (`apps/` for Django, `inference/` for the ASGI service), and the **preserved VEDA library** (`veda_core/`) that both import. The library is the old packages, moved wholesale and left behaviourally intact.

```
veda-platform/
├── docker/
│   ├── Dockerfile.api            # Django + Celery (same image)
│   ├── Dockerfile.inference      # ASGI + heavy models
│   ├── nginx.conf
│   ├── pgbouncer.ini             # transaction pooling in front of Postgres
│   └── entrypoint.*.sh
├── docker-compose.yml
├── docker-compose.prod.yml
├── .env.example
│
├── config/                       # Django project (settings, urls, asgi, wsgi, celery)
│   ├── settings/{base,dev,prod}.py
│   ├── celery.py
│   └── urls.py
│
├── apps/                         # DEEP DJANGO — one app per bounded context
│   ├── core/                     # tenancy, base models, mixins, health
│   ├── sources/                  # DB source registry + connection config
│   ├── substrate/                # ALL ingestion outputs as models (§6)
│   ├── ingestion/                # Celery L0 pipeline, job tracking, admin actions
│   ├── query/                    # DRF QueryView, request/audit models, inference client
│   └── evaluation/               # eval runs, results, HTML report models
│
├── inference/                    # ASGI query/retrieval service (the warm engine)
│   ├── main.py                   # FastAPI/Uvicorn app + lifespan warm-load
│   ├── engine.py                 # get_engine() singleton (relocated, PRESERVED)
│   ├── routes/{retrieve,hybrid,health}.py
│   ├── concurrency.py            # run_in_threadpool_with_context helper (§4.1)
│   └── loaders.py                # pgvector→memory hydration on startup
│
├── veda_core/                    # PRESERVED VEDA library (moved, not rewritten)
│   ├── veda/                     # pipeline, routing, planning, generation,
│   │                             #   compiler, consensus, verifier, validation,
│   │                             #   graph_guard, execution, cache, ir_*  (VERBATIM)
│   ├── query/                    # heads, layers, builders, temporal, router (VERBATIM)
│   ├── retrieval/                # 5-signal spine (VERBATIM)
│   ├── connectors/               # base, relational, datalake, nosql, document (VERBATIM)
│   ├── graph/                    # unified KG api/query/validate (VERBATIM)
│   ├── semantic/                 # compiled registry (VERBATIM)
│   ├── ingestion/                # L0 stage functions (called BY Celery tasks)
│   ├── slm/                      # NEW: _call_slm Strategy + Ollama/vLLM backends (§8b)
│   ├── context.py                # NEW: ambient (source, tenant) contextvar (§4.1)
│   └── veda_hybrid.py            # FRONT DOOR (VERBATIM)
│
├── storage_adapters/             # NEW seam: substrate I/O ↔ Django ORM (structured) + raw pgvector (ANN)
│   ├── reader.py                 # query-time reads (tenant-scoped via context.current())
│   ├── writer.py                 # ingestion-time persistence (ORM + raw pgvector, per-stage Unit of Work)
│   └── assembler.py              # SemanticModelAssembler: substrate rows → the `sm` dict (§8a)
│
└── manage.py
```

> ### The storage_adapters seam is the whole trick (Repository + Adapter)
> VEDA code currently calls `vector_store.py`, `db_abstraction.get_internal_connection()`, `cache.py`, etc. Instead of rewriting those call sites, you point them at `storage_adapters` — a **Repository** whose methods keep the **same signatures and return shapes** (e.g. `RetrievalResult`) the callers expect (**Adapter**). Be honest about the backing store: **structured rows → Django ORM**; **ANN / graph / arbitrary read SQL → raw pgvector against the same Postgres Django manages** (the ORM cannot express `embedding <=> %s::vector`, HNSW search, or the connectors' ad-hoc SQL). So "maximise Django" means Django owns the schema, migrations, admin, tenancy, and all *structured* I/O — **not** that every read becomes an ORM query. Do not try to fake a psycopg cursor over the ORM; keep a raw path on the same connection (pooled through PgBouncer). Tenancy is applied **here and only here**, from the ambient context (§4.1), so engine signatures never change.

### 4.1 Tenant & request context — the ambient-context pattern

The engine's public functions are **frozen** and carry no tenant/source argument — `run_query(query, sm, all_cols)`, `verified_cache_lookup(query)`, `get_engine().retrieve(query, …)`. We must not add parameters (that would ripple through every call site and break the "relocated, not rewritten" rule). So `(source, tenant)` travel **out of band** through a `contextvars.ContextVar`, set at the edge and read only by `storage_adapters`:

```python
# veda_core/context.py  (new, tiny, import-light — no Django import)
from contextvars import ContextVar, copy_context
from dataclasses import dataclass

@dataclass(frozen=True)
class RequestContext:
    source_id: int
    tenant: str

_ctx: ContextVar["RequestContext | None"] = ContextVar("veda_ctx", default=None)

def set_context(ctx: RequestContext):
    return _ctx.set(ctx)

def current() -> RequestContext:
    c = _ctx.get()
    if c is None:                       # fail-closed: never silently default a tenant
        raise RuntimeError("no VEDA request context set")
    return c
```

Where each tier sets it:

- **api tier** — a DRF permission/middleware resolves the tenant from the **authenticated principal** (never a client-supplied field) and sends `{source_id, tenant}` in the inference request body.
- **inference tier** — an ASGI middleware calls `set_context(...)` per request. `contextvars` is coroutine- **and** thread-safe, so concurrent requests never bleed. Heavy sync inference dispatched to the thread pool must run under a copied context — but see the mandatory helper below.
- **worker tier** — a small task base class / decorator sets the context from the task's `(source, tenant)` args before any `veda_core` ingestion function runs.
- **storage_adapters** — every ORM query and every raw pgvector/graph query reads `current()` and scopes on `source_id`/`tenant`. This is the single choke point where tenancy is enforced, which is what keeps the engine tenant-oblivious.

> ### ⚠️ Context propagation into the thread pool is wrapped ONCE and lint-enforced — never left to discipline
> `current()` fails closed on an **unset** context, but a thread-pool call that runs under the **wrong (leaked)** context does not raise — it silently reads the wrong tenant. Requiring every future `run_in_threadpool` call site to remember `copy_context().run(...)` is exactly the kind of rule that gets forgotten on the fifth code path added six months later, and the failure is silent-until-it-isn't. So:
> - Provide **one** offload primitive — `inference/concurrency.py::run_in_threadpool_with_context(fn, *a, **kw)` — that snapshots the current context with `copy_context()` and runs `fn` inside it. All heavy sync inference goes through this helper.
> - **Forbid raw `run_in_threadpool` / bare `ThreadPoolExecutor.submit`** in `inference/` and `veda_core/` via a lint rule (flake8 forbidden-import / ruff `flake8-tidy-imports` banned-api, or a `grep` gate in CI). A passing interleaved-context test proves today's paths are correct; the lint rule is what keeps tomorrow's paths correct.

```python
# inference/concurrency.py
import anyio
from contextvars import copy_context

async def run_in_threadpool_with_context(fn, *args, **kwargs):
    ctx = copy_context()
    return await anyio.to_thread.run_sync(lambda: ctx.run(fn, *args, **kwargs))
```

> **Why `contextvars`, not thread-locals:** thread-locals leak or vanish across the ASGI event loop and thread-pool offload. `contextvars` is the async-correct successor and is exactly what Uvicorn + `run_in_threadpool` expect. Fail-closed (`current()` raises when unset) turns a missing-context bug into a loud 500 on that one request, never a cross-tenant read; the single wrapped offload primitive turns a leaked-context bug into an impossibility rather than a code-review hope.

---

## 5. Django apps and responsibilities

### apps.core

- **Tenant / Source scoping mixin** — Abstract `TenantScopedModel` with `source_id` (FK) + `tenant` (indexed). Every substrate model inherits it.
- **TimeStamped / UUID base** — `UUIDPrimaryKeyModel` (matches ingestion's per-table/column UUID scheme) + created/updated.
- **Health & readiness** — `/healthz` (liveness), `/readyz` (checks postgres via PgBouncer, both Redis instances, inference reachability, SLM backend).
- **Settings bridge** — Loads `config.py` values into Django settings so `config.py` stays the single source of truth.

### apps.sources

Registry of every queryable DB and how to connect to it. Replaces ad-hoc `db_abstraction` source config with a managed model.

- **Source** — One row per connectable database: `id, name, dialect (postgres/mysql/sqlite/oracle/sqlserver/duckdb/mongo/es/dynamo), connector_type, connection secret ref (never plaintext), status, last_ingested_at`, plus a `ready` flag flipped only when an ingestion job completes so the query path never reads a half-built substrate.
- **SourceConnectionProfile** — Pool sizing, read-only role, `statement_timeout` override, sensitive-pattern overrides.

### apps.substrate

The heart of the migration: **everything ingestion produces**, as models. Detailed in §6.

### apps.ingestion

- **IngestionJob / IngestionStage** — Tracks a full L0 run per source and each stage's status, checkpoint, row counts, errors, timing. Makes ingestion resumable and observable in admin.
- **Celery task graph** — One task per L0 stage (§7), chained; each writes via `storage_adapters.writer` and advances the stage checkpoint.
- **Admin actions** — "Re-ingest source", "Rebuild embeddings only", "Regenerate glossary", "Warm caches" — buttons that enqueue tasks.

### apps.query

- **QueryView (DRF)** — `POST /api/v1/query` — validates payload, resolves tenant, calls inference client, persists QueryLog, returns serialized MultiResult with the exact `status` field preserved.
- **InferenceClient** — Thin HTTP client to the inference service with timeouts, retries (idempotent GET only), and circuit-breaker. Never imports `veda_core`.
- **QueryLog (audit, L9)** — Append-only: query text, tenant, route taken, sub-results, status, latency, SQL executed (parameterized), refusal reason. Mirrors `audit_logger.py`, now a model.

### apps.evaluation

Wraps `evaluation/` harnesses as tracked runs: `EvalRun`, `EvalCaseResult`, and a stored HTML report artifact (replaces the loose `poc_report.html`). Runs via Celery, viewable in admin.

---

## 6. The substrate data model — store everything ingestion produces

This is the core of your requirement. Every artifact the L0 pipeline derives is persisted as a Django model (structured) and/or a pgvector table (embeddings), then hot-loaded into Redis and process memory at query time. The mapping below is 1:1 with ARCHITECTURE.md §4 (ingestion) and §11 (stores). Nothing is dropped; nothing new is invented.

### 6.1 Structural / schema substrate

| Django model (apps.substrate) | Source in ARCHITECTURE.md | Key fields | Storage |
|---|---|---|---|
| **SchemaTable** | `schema_scanner → ScanResult` | uuid, source, tenant, name, row_count, display_column, is_sensitive | Postgres |
| **SchemaColumn** | `schema_scanner` | uuid, table(FK), name, data_type, is_pk, is_fk, semantic_type, confidence, review_flag, excluded(sensitive) | Postgres |
| **FkEdge** | `vector_store.store_fk_adjacency` + `data_graph` | source, tenant, from_table, from_col, to_table, to_col, join_type, is_declared, overlap_score | Postgres + Redis hot index |
| **TableMetadata** | `table_metadata` store | table(FK), display_column, notes | Postgres + Redis |

> ### fk_adjacency stays the join engine's source of truth
> `FkEdge` is authoritative for join inference (`compiler.py`, `join_planner.py`, `graph_guard.py`). It is loaded into a Redis hash and an in-process adjacency map at inference startup so join derivation never hits the DB on the hot path. `data_graph`'s undeclared FKs (overlap ≥ 0.70) are merged into the same table with `is_declared=False` — exactly as today.

### 6.2 Semantic / language substrate

| Django model | Source | Key fields | Storage |
|---|---|---|---|
| **SemanticType** | `semantic_type_inference` | column(FK), type(MONETARY/TEMPORAL/CATEGORICAL/IDENTIFIER/FLAG/TEXT), confidence, layer_hit, review_flag | Postgres |
| **GlossaryEntry** | `domain_glossary` / `glossary_builder` | source, tenant, term, canonical, definition, provenance(LLM), scope | Postgres + Redis |
| **Synonym** | `glossary_builder` | source, tenant, term, synonym, weight | Postgres + Redis |
| **SyntheticPair** | `synthetic_query_gen` | source, nl_text, ir_json, target_column, used_for_finetune | Postgres |
| **SemanticConcept / Dimension / Metric** | `semantic/registry` (compiled) | name, definition, mapping_json, manifest_version | Postgres (from compiled JSON) |

### 6.3 Value-grounding substrate

| Django model | Source | Key fields | Storage |
|---|---|---|---|
| **ColumnValueSample** | `value_sampler` | column(FK), value, freq, sampled_at (used by value grounding + arbitration) | Postgres + Redis set per column |
| **ColumnProfile** | `data_profiler` | column(FK), distinct_count, top_n_json, null_ratio, min/max | Postgres |

> ### Value grounding must be O(1) at query time
> Gate L6a checks every filter literal against sampled values. Store `ColumnValueSample` both in Postgres (audit/rebuild) and as a Redis SET keyed `vg:{source}:{tenant}:{column_uuid}` so `validation.value_grounding` does a set-membership check, not a table scan.

### 6.4 Vector / embedding substrate (pgvector)

Embeddings stay in pgvector, keyed by column UUID and encoder mode, matching ARCHITECTURE.md §11–§12 dimensions. These are NOT plain Django models for search (ORM can't do ANN); they are pgvector tables with an HNSW/IVFFlat index, accessed via raw SQL in `storage_adapters.reader`. A thin unmanaged Django model mirrors each for admin visibility.

| pgvector table | Dim | Encoder mode / signal | Index |
|---|---|---|---|
| **column_embeddings** | 256 | relgt_only / light_text (single-encoder) | HNSW cosine |
| **column_embeddings_lt** | 256 | ensemble light-text (TF-IDF+SVD) | HNSW cosine |
| **column_embeddings_hybrid** | 640 | ensemble hybrid (MiniLM+RELGT) | HNSW cosine |
| **column_embeddings_bge** | 1024 | BGE-M3 dense (5-signal spine, `semantic_search`) | HNSW cosine |
| **chunk_embeddings** | 1024 | doc chunks (`chunk_embedder`, RAG substrate) | HNSW cosine |
| **relgt_structural** | 256 | RELGT structural encoder (`reg_builder→relgt_encoder`) | HNSW cosine |

Encoder-mode dims preserved exactly: RELGT 256, light-text 256, MiniLM 384, hybrid 640, BGE-hybrid 1280 (BGE 1024 + RELGT 256), BGE-M3 1024. Switching `ENCODER_MODE` still requires re-ingestion — enforced by a guard in the ingestion job.

> ### ⚠️ HNSW is approximate — the parity gate must run against the shipping index
> Every table above uses an **HNSW** index, which is an *approximate* nearest-neighbour structure: on the same data it can return a different top-k ordering than the legacy **exact** cosine scan, especially at low `ef_search`. This is the crux of the parity contradiction (§7.1a, §18): if the retrieval parity gate runs against exact search and you then ship HNSW, retrieval silently drifts after the gate is green. The HNSW build/search parameters (`m`, `ef_construction`, `ef_search`) are therefore **part of the frozen contract** — they are tuned in Phase 7.1a until recall@k = 1.0 against the exact-cosine fixtures, and the parity gate runs against that exact live configuration.

### 6.5 Graph substrate (unified knowledge graph)

| Store | Source | Handling | Storage |
|---|---|---|---|
| **GraphNode / GraphEdge (Django)** | `unified_graph_builder`, `relationship_graph` | Schema+chunk nodes and typed edges for query-time expansion (`suggest_expansions`). | Postgres + Redis adjacency |
| **Kùzu / persisted graph** | `kuzu_store`, `graph_persist`, `reg_graph.pkl` | Keep Kùzu file store on a mounted volume; register its path + version in a `GraphArtifact` model. Loaded into inference memory at startup. | Volume + Postgres metadata |
| **GraphNodeEmbedding** | `graph_embedder` | Node embeddings for graph retrieval. | pgvector |

### 6.6 Cache & operational substrate

| Store | Source | Handling | Storage |
|---|---|---|---|
| **VerifiedQueryCache** | `veda/cache.py` (file-based cosine ≥ 0.85) | Move to Postgres (query_embedding vector + verified SQL + cols + `query_hash` unique) with pgvector lookup; mirror hottest entries in Redis. **Written on the query hot path** by inference via `INSERT … ON CONFLICT (query_hash) DO NOTHING` (idempotent under N replicas), then **broadcast to peer replicas via pub/sub** (§8.4) so the fleet's in-process/Redis mirrors converge without waiting for a rehydrate. Preserves cosine-≥-0.85 replay, skip rules (existence/fast-path/temporal never cached). | Postgres + Redis |
| **RetrievalCache** | `retrieval/retrieval_cache.py` | Redis with TTL, keyed by normalized query + source + encoder mode. | Redis |
| **QueryLog (audit)** | `query/audit_logger.py` (L9) | Append-only Postgres model. | Postgres |
| **IntentEnvelope / SLM artifacts** | `intent_envelope`, `envelope_slm` | Cached per query in Redis; not persisted long-term. | Redis |

> ### ⚠️ Verified-query cache: preserve the skip rules exactly
> The cache is populated only for non-temporal, non-existence, non-fast-path answers with rows (ARCHITECTURE.md §5.3 step 7), and existence queries are NEVER cached (embeddings can't tell "with" from "without"). These rules move verbatim into the writer; do not let the ORM layer accidentally cache a refused or existence result.

> ### ⚠️ The verified cache is a query-time WRITE — inference is not purely read-only
> `save_verified_query` fires inside `run_query` (pipeline.py:738) after a successful, cacheable answer, so in the platform this write hits the shared store. Consequences the plan makes explicit: **(a)** the inference tier needs *write* access to `VerifiedQueryCache` — the one documented exception to "inference reads, ingestion writes"; **(b)** N replicas can race the same query, so the write is `INSERT … ON CONFLICT (query_hash) DO NOTHING` keyed by a normalized-query hash — **never** read-modify-write; **(c)** keep it off the latency-critical path (fire-and-forget via a tiny Celery task or an after-response hook) so cache maintenance never slows the answer the user already has.

> ### ⚠️ Verified-cache READ staleness across replicas is a warm-cache-eventually property, not a bug
> The write race is handled, but the *read* side is distributed: the cache is mirrored in `redis-cache` and in-process on each replica. When replica A writes a new verified query, replicas B…N don't know about it until they rehydrate — so a second identical query routed by the LB to a different replica **misses** and simply re-runs the query (correct, just not cache-fast). Two consequences: **(1)** the "measurably faster second query" acceptance check (§5 Phase 5, exit 4) is inherently flaky depending on LB replica selection and must be phrased as a warm-cache-eventually property (see that criterion); **(2)** to converge faster, new cache entries are pushed through the **same pub/sub fan-out as rehydrate** (§8.4) so peers pick them up promptly. This is a convergence optimization, not a correctness requirement — a miss is always safe.

---

## 7. Ingestion pipeline as Celery tasks (L0)

ARCHITECTURE.md §4 lists nine ingestion steps. Each becomes a Celery task in `apps.ingestion.tasks`, chained into one job, each writing through `storage_adapters.writer` and checkpointing an `IngestionStage`. The step functions themselves live in `veda_core/ingestion/` and are called by the tasks — the logic is preserved; only the driver (was `main.py --ingestion-only`) becomes a task chain.

| Order | Celery task | Calls (veda_core) | Persists to | Queue |
|---|---|---|---|---|
| 1 | `task_schema_scan` | `schema_scanner` (schema/real_schema) | SchemaTable, SchemaColumn (sensitive excluded) | ingestion |
| 2 | `task_fk_adjacency` | `vector_store.store_fk_adjacency` | FkEdge (is_declared=True) | ingestion |
| 3 | `task_data_graph` | `data_graph` (overlap ≥ 0.70, 200 rows) | FkEdge (is_declared=False). Non-fatal. | ingestion |
| 4 | `task_semantic_types` | `semantic_type_inference` (3-layer) | SemanticType | ingestion |
| 5 | `task_value_profiling` | `value_sampler`, `data_profiler` | ColumnValueSample, ColumnProfile | ingestion |
| 6 | `task_embeddings` | `reg_builder→relgt_encoder`; `biencoder(+auto_finetune)`; MiniLM/TF-IDF | all pgvector tables (§6.4) | ingestion |
| 7 | `task_vector_store` | `vector_store` | pgvector index build (HNSW) | ingestion |
| 8 | `task_derived_language` | `domain_glossary`, `glossary_builder`, `synthetic_query_gen` | GlossaryEntry, Synonym, SyntheticPair | ingestion |
| 9 | `task_unified_graph` | `unified_graph_builder`, `chunk_embedder`, `chunk_linker`, `graph_embedder`, `graph_persist`/`kuzu_store` | GraphNode, GraphEdge, GraphNodeEmbedding, chunk_embeddings, GraphArtifact | ingestion |
| 10 | `task_warm_caches` | `storage_adapters.writer.warm()` | Redis hot indices + signal inference to reload | high |

### Job orchestration

- **`task_ingest_source(source_id)`** builds the chain `chain(task_schema_scan.s(...), task_fk_adjacency.s(), ... , task_warm_caches.s())` and creates the IngestionJob.
- **Each task is idempotent:** it upserts by `(source, tenant, uuid/name)` so a re-run overwrites cleanly. On failure the stage is marked failed with the traceback and the chain halts; "resume" restarts from the last incomplete stage.
- **ENCODER_MODE guard:** `task_embeddings` refuses to run if the requested mode differs from the persisted mode without an explicit force flag (re-ingestion required, per §12).
- **SLM-dependent steps** (glossary, synthetic pairs) call the SLM backend over `veda_net` through `_call_slm` (§8b) — Ollama by default at ingestion time; failures are non-fatal where ARCHITECTURE.md marks them optional.

> ### ⚠️ The embedding stage is the one place per-stage `transaction.atomic()` costs more than it buys
> Each L0 stage runs inside one `transaction.atomic()` (Unit of Work, §9a). For most stages that is exactly right. But `task_embeddings` (stage 6) writes **all six pgvector tables** for potentially large schemas — wrapping that in a single atomic transaction means one long-held write lock and a huge WAL burst, and a failure at 95% rolls back the entire embedding set. Because "idempotent upsert by natural key" (below) already makes partial-progress-then-resume safe, full atomicity here buys less than it costs. **Use batched commits within stage 6** — checkpoint per pgvector table (or per N columns) rather than one transaction spanning the largest stage — and record the batch checkpoint on the `IngestionStage` so resume continues mid-stage. All other stages keep the single-transaction Unit of Work.

> ### Ingestion writes DB; query-time reads memory — the split that makes it fast
> Every heavy artifact is written to Postgres/pgvector by the worker (durable, rebuildable). The final `task_warm_caches` pushes the query-critical subset into Redis and signals the inference service to (re)hydrate its in-process structures. So the slow path (embedding, LLM glossary) runs once at ingestion; the hot path (retrieval, join derivation, value grounding) touches only memory + indexed pgvector.

---

## 8. The inference service — warm load, memory hydration, query path

This ASGI service is the production incarnation of `veda/runtime.get_engine()`. It warm-loads once at process start and holds everything the hot path needs in memory. It runs the **identical** `veda_hybrid.run_hybrid_query` and `veda/pipeline.run_query` code from `veda_core/`.

### 8.1 Startup warm-load (lifespan)

On Uvicorn lifespan startup, `inference/loaders.py` hydrates, in order:

1. BGE-M3 (1024-dim) encoder + bge-reranker-v2-m3 cross-encoder + MiniLM ensemble encoders (from local model cache on a mounted volume — no network).
2. The 5-signal engine (`retrieval_engine_phase3.get_engine`) kept warm process-wide — `semantic_search`, `bm25_ranker`, `signal_builder`, `rrf_merger`, `intent_boosting`, `adaptive_cutoff`, `retrieval_cache`.
3. FK adjacency map: load FkEdge rows from Redis hot hash into an in-memory adjacency dict for compiler/join_planner/graph_guard.
4. Glossary + synonyms + display columns: load from Redis into dicts for graph-expand and answer-entity.
5. Verified-query cache: warm the hottest entries from Redis; cold entries resolved via pgvector on miss.
6. Unified knowledge graph: load Kùzu/persisted graph from the mounted volume (path from GraphArtifact) for `suggest_expansions`.
7. DB pools: read-only source pool + internal store pool (via storage_adapters), **both dialed through PgBouncer**, with `statement_timeout` and bounded fetch preserved.
8. **Assembled `sm` read-model** per active `(source, tenant)` via `SemanticModelAssembler` (§8a) — the nested dict the deterministic engine actually consumes — cached in process memory + Redis, keyed by `substrate_version`.

> ### ⚠️ Size worker count from a measured per-worker RSS, not from CPU
> Items 1 hydrates **three transformer models** — BGE-M3, the bge-reranker-v2-m3 cross-encoder (not small), and the MiniLM ensemble — **once per Uvicorn worker**. The real memory budget is `per_worker_RSS × workers_per_replica × replicas`, and the OOM lives here. Do not leave this as "size worker count to RAM" with no number:
> - **Measure** the steady-state RSS of one fully warmed worker (all three models + engine + warm caches loaded) on the target hardware. Record the figure in the deploy notes (e.g. `PER_WORKER_RSS_GB`).
> - **Derive** `workers_per_replica = floor((replica_RAM_GB × safety_fraction) − OS/overhead) / PER_WORKER_RSS_GB)`, with a safety fraction (~0.8) for CUDA context, page cache, and request-time allocations.
> - Prefer **more replicas over more workers per replica** for throughput, since each additional worker re-pays the full model RAM.
> Discover the ceiling on paper here, not during load test (§7.2).

> ### Concurrency model (and how it interacts with the SLM tier)
> Uvicorn with multiple workers, models loaded once per worker (unavoidable) — so size worker count to the measured RSS above and scale horizontally with more `inference` replicas behind the api's client-side load balancing. Heavy sync inference (encoders) runs in a thread pool **via `run_in_threadpool_with_context`** (§4.1) so the event loop stays responsive and the tenant context is carried in. Note the interaction: the thread-pool offload means concurrent requests share the **one model instance per worker** — good for memory, but it **serializes GPU access within a worker**. Combined with the SLM tier, in-worker inference concurrency is gated by *both* the encoder GPU and the SLM backend; vLLM on the SLM side (§8b) relieves the SLM half of that constraint. This preserves the "one warm engine per process" semantic of `get_engine()` exactly — there are just N processes now.

### 8.2 Endpoints

| Endpoint | Body | Runs | Returns |
|---|---|---|---|
| `POST /v1/run_hybrid_query` | {query, source_id, tenant, flags?} | `veda_hybrid.run_hybrid_query` (front door: decompose/route/fan-out/firewall) | MultiResult (status preserved) |
| `POST /v1/retrieve` | {query, source_id, tenant, top_k?} | `get_engine().retrieve` (5-signal spine only) | ranked columns + scores |
| `POST /v1/rehydrate` | {source_id, tenant, scope} | reload FK/glossary/cache/`sm` from Redis/pgvector **on every replica** (§8.4 fan-out) | ok + versions |
| `GET /healthz /readyz` | — | liveness + model-loaded check | 200/503 |

### 8.4 Rehydrate (and new-cache-entry) fan-out must reach every replica (Publisher/Subscriber)

`/v1/rehydrate` hitting one replica behind the client-side load balancer only refreshes *that* process. After a re-ingestion the whole fleet must reload, so rehydrate is a **fan-out, not a local call**: the receiving replica (or the worker's `task_warm_caches` directly) **publishes** a message to a `redis-cache` pub/sub channel (`veda:rehydrate:{source}:{tenant}:{scope}`); every `inference` replica **subscribes** at startup and reloads the named scope (FK adjacency map / glossary+synonyms / KG / verified-cache warm set / assembled `sm`), bumping its in-memory `substrate_version`. A replica that was down during the broadcast catches up on its next lifespan warm-load (it always reads current substrate). The **same channel carries newly-written verified-cache entries** (§6.6) so a query cached on replica A becomes warm on B…N without waiting for a full rehydrate. This keeps "one warm engine per process" while making ingestion's freshness (and hot-path cache writes) reach all N processes.

### 8.3 Query-time read path (the fast path, unchanged flow)

When `run_query` executes its escalation ladder, each store access resolves against memory first, then indexed pgvector — never a cold scan:

| Ladder step (ARCHITECTURE.md §5.2) | Reads from | Backing |
|---|---|---|
| Fast path (T0) | in-memory compiled registries (semantic/) | loaded at startup |
| Verified cache (T0) | Redis hottest → pgvector cosine ≥ 0.85 | Redis + pgvector |
| 5-signal retrieve (L2) | BGE-M3 in-mem encode → pgvector ANN (HNSW) + BM25 + FK signals | memory + pgvector + Redis |
| Graph expand (L2g) | in-memory KG adjacency | loaded from Kùzu volume |
| Primary rerank (L2b) | in-mem cross-encoder | loaded at startup |
| Anchor select/vet (L2/L3) | in-mem schema + FK map | Redis hot hash |
| Value grounding (L6a) | Redis SET membership per column | Redis (rebuilt from pgvector) |
| Join derivation / graph_guard | in-mem FK adjacency | Redis hot hash |
| SLM calls (IR emit / decompose / RAG synth / NL answer) | `_call_slm` → vLLM (prod) / Ollama (dev) | SLM backend (§8b) |
| Execute (L7) | read-only source pool via PgBouncer, 30s timeout, ≤20 rows | source DB |

---

## 8a. Semantic-model assembly — normalized rows → the `sm` dict

The deterministic engine does **not** read `SchemaColumn` / `SemanticType` / `GlossaryEntry` rows. It reads one nested dict, `sm`, whose shape is load-bearing:

```python
sm["columns"]["incident.status"] = {
    "semantic_type": "CATEGORICAL", "aliases": [...], "business_definition": "...", ...
}
sm["domain_synonyms"]["last logged in"] = ["user.last_logged_in"]
# consumed at pipeline.py:17, 346, 553, 563, 613, 631, 660 …
```

Today `veda_hybrid._load_semantic_model()` loads it whole from `data/veda_semantic_model.json`. In the platform that monolithic file no longer exists — the same information lives in normalized tables — so we must **rebuild the identical dict** from the substrate. This is a first-class migration artifact, not an afterthought, because every anchor/grounding/glossary decision reads from it.

- **`SemanticModelAssembler` (Builder pattern)** — `storage_adapters/assembler.py`. Given the ambient `(source, tenant)`, it queries the substrate models with `select_related`/`prefetch_related` (no N+1) and emits the exact `sm` dict **and** the `all_cols` list the engine expects: same keys, same `"table.column"` string formatting, same optional fields present/absent as the legacy JSON.
- **Cached read-model** — the assembled `sm` is expensive to rebuild, so cache it in Redis (`sm:{source}:{tenant}:{substrate_version}`) and in inference process memory. Invalidated by `substrate_version`, bumped when an ingestion job completes and broadcast via §8.4.
- **Parity acceptance (required exit check)** — for the home schema, `assemble(source, tenant)` must be **deep-equal** to the Phase-0 fixture `veda_semantic_model.json` (normalize dict/list ordering before diffing). A mismatch here silently changes engine behaviour, so the query path stays gated behind this check until it passes.

---

## 8b. The SLM backend seam — `_call_slm` Strategy (Ollama · vLLM)

Query-time SLM inference is the one runtime dependency that must **not** be hardwired to Ollama, because a single Ollama instance serializes every SLM call across the whole inference fleet (§3 callout). The fix is a thin Strategy interface that every SLM call site already funnels through, with the backend selected by config.

- **One interface, frozen signature.** All SLM calls (`ir_emit`, `decompose`, `rag_synthesis`, `nl_answer`) route through `veda_core/slm/_call_slm.py::call_slm(prompt, *, purpose, timeout=240, **opts) -> str`. This is the single choke point; call sites in `veda_core` are rewired to it exactly as storage call sites are rewired to `storage_adapters` (Phase 3 pattern) — same signature, swapped backend.
- **Two backends behind the interface.**
  - `OllamaBackend` — the existing HTTP client to `ollama/ollama`. Default for **dev** and for **ingestion-time** generation (glossary, synthetic pairs) on `worker`, where latency isn't critical.
  - `vLLMBackend` — OpenAI-compatible client to a vLLM server. Default for the **production query hot path** on `inference`; continuous batching serves concurrent SLM calls from N replicas without single-instance serialization.
- **Config-flagged, one source of truth.** `SLM_BACKEND={ollama|vllm}` and `SLM_MODEL_NAME=qwen2.5-coder:7b` live in `config.py` and are surfaced through the settings bridge (§9). The tier chooses its backend: `worker` → `ollama`, `inference` (prod) → `vllm`, dev → `ollama` everywhere.
- **Preserved semantics.** The 240s timeout, the circuit breaker, and every deterministic fallback (row-count NL answer, refuse-over-guess) wrap the interface, not a specific backend — so both backends degrade identically to a fallback/refusal, never a hung request. Zero-egress holds: both Ollama and vLLM run inside the deployment on `veda_net`.

```python
# veda_core/slm/_call_slm.py  (interface + registry)
class SLMBackend(Protocol):
    def generate(self, prompt: str, *, purpose: str, timeout: int, **opts) -> str: ...

def call_slm(prompt: str, *, purpose: str, timeout: int = 240, **opts) -> str:
    backend = _get_backend()          # from SLM_BACKEND; cached per process
    with _slm_circuit_breaker(purpose):   # existing breaker, unchanged
        return backend.generate(prompt, purpose=purpose, timeout=timeout, **opts)
```

---

## 9. Configuration, settings, and secrets

`config.py` stays the single source of truth for the engine (ARCHITECTURE.md §10). Django settings import from it rather than duplicating values, so there is exactly one place flags/thresholds/model-names live.

- **Bridge pattern:** `apps/core/settings_bridge.py` imports `veda_core.config` and exposes `VEDA = {...}`; Django prod settings read `os.environ` for infra (DB URLs, Redis, secrets) and `VEDA` for engine flags.
- **Env-overridable engine flags:** `ENCODER_MODE, TOP_K, TOP_K_TO_LLM, QUERY_ROUTER_ENABLED, SLM_MODEL_NAME, SLM_BACKEND, IR_JOIN_FREE_ENABLED, FAST_PATH_ENABLED, QUERY_DECOMPOSE_ENABLED`, plus the HNSW search params (`HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH`) so the tuned parity config (§7.1a) is a setting, not a magic number, and all the feature-gated resolver flags — surfaced as env with `config.py` defaults.
- **Secrets:** source DB credentials and any API keys live in Docker secrets / env, referenced by `SourceConnectionProfile` — never stored in model rows as plaintext.
- **Model cache:** BGE/MiniLM/reranker weights baked into the inference image OR mounted from a volume pre-pulled at build; startup never downloads (zero-egress). The vLLM/Ollama model weights are likewise pre-pulled into their service volumes.

---

## 9a. Cross-cutting design patterns & Django standards

Naming the patterns keeps the agent from re-inventing structure and makes reviews mechanical.

### Patterns already in the engine — name them, preserve them
| Pattern | Where | Rule |
|---|---|---|
| **Facade** | `veda_hybrid.run_hybrid_query` | The single public entry; everything downstream stays single-intent. Keep it the ONLY front door. |
| **Strategy** | connectors (relational/datalake/nosql/document), heads (sql/rag/hybrid/nosql), encoder modes, **SLM backends (§8b)** | Interchangeable behind one contract. New DB/head/encoder/SLM-backend plugs in without touching the flow. |
| **Chain of Responsibility** | escalation ladder (T0→T1→T2→refuse) + firewall gate sequence | Each rung/gate either handles or passes; order is load-bearing and frozen. |
| **Template Method** | `BaseConnector(ABC)` | Skeleton in the base; subclasses fill `get_schema`/`execute`/…. |

### Patterns introduced by the migration
| Pattern | Where | Why |
|---|---|---|
| **Repository** | `storage_adapters.reader/writer` | Hide ORM vs raw-pgvector vs Redis behind intent-named methods (`get_fk_adjacency`, `ann_search`, `value_samples`). VEDA never touches ORM/psycopg directly. |
| **Adapter** | same seam | Preserve legacy signatures/return shapes (`RetrievalResult`) while swapping the backend. |
| **Builder** | `SemanticModelAssembler` | Reconstruct the `sm` read-model from normalized rows (§8a). |
| **Strategy (SLM)** | `veda_core/slm/_call_slm` | Swap Ollama↔vLLM behind one frozen call signature (§8b). |
| **Ambient Context** | `veda_core.context` (contextvars) + `run_in_threadpool_with_context` | Carry `(source, tenant)` out-of-band so frozen signatures stay frozen (§4.1); the single wrapped offload primitive keeps context propagation lint-enforceable, not discipline-dependent. |
| **Unit of Work** | each Celery ingestion stage | One `transaction.atomic()` per stage; partial failure rolls back that stage cleanly and resumes. **Exception: stage 6 (embeddings) uses batched commits** (§7 callout). |
| **Circuit Breaker** | `InferenceClient` (api→inference) + existing SLM breaker | Fail fast, degrade to a structured error, never hang. |
| **Publisher/Subscriber** | `redis-cache` pub/sub rehydrate + new-cache-entry fan-out (§8.4) | Broadcast substrate freshness and hot-path cache writes to all inference replicas. |
| **Object Pool** | DB connection pools (via PgBouncer) + warm model singletons | Preserve `get_engine()`'s "one warm engine per process"; bound total Postgres connections. |

### Django standards to apply
- **Fat models, thin views, explicit service layer.** Orchestration lives in `apps/*/services.py` and `veda_core`; DRF views only validate → call a service → serialize. No business logic in views or serializers.
- **Tenancy via custom manager + QuerySet.** `TenantScopedModel` ships a `TenantManager` whose `get_queryset()` filters by the ambient `(source, tenant)` from `context.current()`, so a forgotten `.filter(tenant=…)` cannot leak data. Provide an explicit `objects.all_tenants()` escape hatch for admin/migrations only.
- **Model-level integrity, not app-level hope.** `UniqueConstraint` on natural keys (`(source, tenant, uuid)`), `CheckConstraint` where an enum isn't a `TextChoices`, `Meta.indexes` for every hot-path lookup in §2.3.
- **`TextChoices` enums** for `dialect`, `semantic_type`, `join_type`, terminal `status`, `provenance` — one definition, DB-checked, admin-friendly.
- **Migrations discipline.** pgvector tables + HNSW indexes via `migrations.RunSQL(forward, reverse)` with the tuned `m`/`ef_construction` from §7.1a; `managed=False` mirror models never emit DDL; every migration is reversible.
- **Settings split + 12-factor.** `settings/{base,dev,prod}` with `django-environ`; infra (DB/Redis/secrets) from env; engine flags only through the `config.py` bridge (one source of truth).
- **Kill N+1 reads.** `select_related`/`prefetch_related` in the assembler and any admin/list path that loops substrate rows.
- **DRF conventions.** `ViewSet` + `Serializer` + throttling; one consistent error envelope so a refusal is a `200` structured payload, never a leaked exception or `500`.
- **Transactions & idempotency.** `transaction.atomic()` per ingestion stage (batched within stage 6) and per audit write; concurrency-safe upserts use `INSERT … ON CONFLICT` (verified cache) or `update_or_create` (substrate), not read-modify-write; `select_for_update` only where a real race exists.
- **Bounded-context app boundaries.** No cross-app model imports except through `core`; inter-app calls go through service functions, matching §5.
- **Ban raw thread-pool offload.** Lint rule forbids bare `run_in_threadpool` / `ThreadPoolExecutor.submit` in `inference/` and `veda_core/`; all offload goes through `run_in_threadpool_with_context` (§4.1).

---

# PART B — Phased build runbook

Eight phases, executed in order. The agent does not advance until every exit criterion of the current phase passes. Tasks are written imperatively; each names the files to create/move and an acceptance check.

---

## Phase 0 — Scaffold & preserve

**Goal:** Stand up the repo skeleton, move the VEDA library in verbatim, and prove it still imports and runs a single query the old way. Nothing behavioural changes.

### 0.1 Create repo skeleton
Create the directory tree from §4. Initialize a Django project `config/` with split settings (`base/dev/prod`), `celery.py`, `asgi.py`, `wsgi.py`. Create empty apps: `core, sources, substrate, ingestion, query, evaluation`.
```bash
django-admin startproject config .
for a in core sources substrate ingestion query evaluation; do \
  python manage.py startapp $a apps/$a; done
```

### 0.2 Move VEDA library in — VERBATIM
Copy the existing packages into `veda_core/` with zero edits: `veda/ query/ retrieval/ connectors/ graph/ semantic/ ingestion/ veda_hybrid.py config.py`. Add `veda_core/__init__.py`. Do not rename symbols, do not reformat.

> ### ⚠️ PRESERVE
> This is a move, not a refactor. If a linter wants to change anything in `veda_core`, disable it for that path. The only later edits permitted are inside call sites that reach storage (Phase 3) and SLM (Phase 3b), and those swap the backend, not the contract.

### 0.3 Config bridge
Create `apps/core/settings_bridge.py` importing `veda_core.config`. In `settings/base.py`, set `VEDA = build_veda_settings()`. Infra values (DB, Redis) come from env.

### 0.4 Smoke test the preserved library
From a shell in the repo, run the old front door directly against a test DB to confirm the moved library is intact:
```bash
python -c "from veda_core.veda_hybrid import run_hybrid_query; \
           print(run_hybrid_query('how many incidents are escalated'))"
```

### 0.5 Capture the golden baseline (before anything changes)
Run the full eval suite through the untouched legacy engine and serialize every MultiResult (status, rows, sql, route, ladder-rung from the explain trace) to a JSON snapshot committed to the repo (§17). **Also capture, per parity query, the exact top-k column ordering from the legacy exact-cosine retrieval** — this is the fixture the HNSW tuning in §7.1a targets.

> **Exit criteria (all must pass)**
> 1. Repo tree matches §4; all six apps registered in `INSTALLED_APPS`.
> 2. `python -c 'import veda_core.veda_hybrid'` succeeds with no import errors.
> 3. The 0.4 smoke query returns a MultiResult against a test source (proves the library still works pre-migration).
> 4. `config.py` values are readable through Django settings via the bridge.
> 5. Golden baseline snapshot (incl. exact-cosine top-k fixtures) committed.

---

## Phase 1 — Infrastructure & containers

**Goal:** Bring up Postgres+pgvector behind PgBouncer, split Redis, the SLM backends, and empty api/inference/worker containers. Everything boots and can talk on `veda_net`.

### 1.1 Postgres + pgvector + PgBouncer
Use `pgvector/pgvector:pg16`. Init script enables the extension and creates the internal store DB. **Put PgBouncer in front** (`pool_mode = transaction`) and point every Django DB alias and every raw pool at PgBouncer's port, not Postgres directly, so `N workers × M replicas × pool_size` cannot exceed Postgres `max_connections`. Two Django DB aliases: `default` (internal store) and `source_registry` if you keep source metadata separate.
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 1.2 Redis — two instances
Run **`redis-broker`** (Celery broker+backend, unbounded, no eviction) and **`redis-cache`** (Django cache DB, hot substrate indices, retrieval memoization, rehydrate pub/sub) as **separate instances**, not two DBs on one process (§3 callout). Set `maxmemory` + `allkeys-lru` on `redis-cache` only; `redis-broker` never evicts. (Dev profile may collapse to one instance with logical DBs; prod must not.)

### 1.3 Dockerfiles
Two application images. `Dockerfile.api` (Django/DRF/Celery, no ML deps) stays small. `Dockerfile.inference` adds torch/transformers/sentence-transformers and BAKES the model weights (BGE-M3, bge-reranker-v2-m3, MiniLM) into the image or a pre-pulled volume so startup is offline.

> ### Keep the two images separate
> Do not put torch in the api image. The whole point of the split is thin API workers + heavy inference workers. A shared base for OS deps is fine; ML deps live only in the inference image.

### 1.4 compose + nginx + SLM backends
Write `docker-compose.yml` with all services on `veda_net`. Only nginx publishes a host port. Add the SLM backends: `ollama/ollama` (dev/ingestion) and, for the prod profile, a `vllm` service; pre-pull `qwen2.5-coder:7b` (SQL/IR generation) **and** `qwen2.5:1.5b-instruct` (NL_SUMMARY_MODEL — the separate small model `query/result_explainer.py` uses to phrase result rows; missing it degrades to deterministic template answers rather than failing, but silently) into named volumes. Attach GPU to inference, vLLM, and ollama.

> **Exit criteria (all must pass)**
> 1. `docker compose up` brings all services healthy; `/healthz` on api and inference return 200.
> 2. From the api container, `psql` reaches Postgres **through PgBouncer** and `SELECT '[1,2,3]'::vector` works.
> 3. Both Redis instances reachable from the right tiers (broker from worker/beat/api; cache from all); the SLM backend responds to a test generate on `veda_net`.
> 4. No application container exposes a host port except nginx.

---

## Phase 2 — Substrate models & migrations

**Goal:** Turn §6 into real Django models and pgvector tables. This is the schema for everything ingestion produces.

### 2.1 core base models
Implement `UUIDPrimaryKeyModel`, `TimeStampedModel`, and `TenantScopedModel` (adds `source` (FK) + `tenant` indexed). Every substrate model inherits TenantScoped.

### 2.2 sources app
Model `Source` (incl. the `ready` flag) and `SourceConnectionProfile` per §5. Secrets by reference only. Add admin with a "test connection" action.

### 2.3 structural + semantic + value models
Implement §6.1–§6.3 models: `SchemaTable, SchemaColumn, FkEdge, TableMetadata, SemanticType, GlossaryEntry, Synonym, SyntheticPair, SemanticConcept/Dimension/Metric, ColumnValueSample, ColumnProfile`. Match field names to the ingestion outputs so the writer maps cleanly.

> ### Index for the hot path now
> Add DB indexes the query path needs: FkEdge on `(source, tenant, from_table)`; ColumnValueSample on `(column)`; GlossaryEntry on `(source, tenant, term)`. These back the Redis rebuild and any cache-miss fallback.

### 2.4 pgvector tables (§6.4)
Create the six pgvector tables via a raw SQL migration (RunSQL) with HNSW cosine indexes and the exact dims from §6.4/§12. Parameterize the HNSW build params (`m`, `ef_construction`) from settings so §7.1a can retune them reversibly. Add unmanaged mirror models (`managed=False`) purely for admin visibility. ANN search is raw SQL in the reader, never ORM.
```sql
CREATE TABLE column_embeddings_bge (
  column_uuid uuid, source_id int, tenant text,
  embedding vector(1024), PRIMARY KEY(column_uuid, source_id, tenant));
CREATE INDEX ON column_embeddings_bge USING hnsw (embedding vector_cosine_ops)
  WITH (m = :hnsw_m, ef_construction = :hnsw_ef_construction);
```

### 2.5 graph + cache + audit models (§6.5–§6.6)
Implement `GraphNode, GraphEdge, GraphNodeEmbedding, GraphArtifact, VerifiedQueryCache (with query_embedding vector), QueryLog`. VerifiedQueryCache lookup is pgvector cosine ≥ 0.85.

> **Exit criteria (all must pass)**
> 1. `python manage.py makemigrations && migrate` applies cleanly; all §6 tables exist.
> 2. The six pgvector tables have HNSW indexes; a hand-inserted vector is retrievable by cosine SQL.
> 3. Django admin lists every substrate model, scoped by source + tenant.
> 4. Field names on structural/semantic/value models line up with ingestion outputs (verified against `veda_core` function returns).

---

## Phase 3 — storage_adapters (the seam)

**Goal:** Redirect VEDA's storage calls to Django ORM + raw pgvector WITHOUT changing their signatures. After this phase the engine reads/writes the new substrate but behaves identically.

### 3.1 Map every storage call site
Enumerate where `veda_core` touches storage: `ingestion/vector_store.py` (store_fk_adjacency, resolve_cols_by_exact_names, RetrievalResult), `ingestion/db_abstraction.py` (get_internal_connection), `veda/cache.py` (verified cache), `query/audit_logger.py`, value-sample reads in `validation.py`, glossary reads in graph-expand. Produce a call-site inventory table.

### 3.2 Implement writer (ingestion side)
In `storage_adapters/writer.py`, implement functions with the SAME names/signatures the ingestion code calls, backed by the ORM: e.g. `store_fk_adjacency(edges)` → bulk_create FkEdge; `store_column_embeddings(mode, rows)` → raw pgvector INSERT (**batched per table**, see Phase 4 stage 6); `store_glossary(entries)` → GlossaryEntry upsert. Plus `warm()` that populates Redis hot indices and publishes the rehydrate fan-out.

### 3.3 Implement reader (query side)
In `storage_adapters/reader.py`, implement the read contracts the hot path uses: `get_fk_adjacency(source, tenant)` (Redis→dict), `ann_search(mode, qvec, top_k)` (raw pgvector cosine over HNSW), `value_samples(column_uuid)` (Redis SET), `verified_cache_lookup(qvec)` (Redis→pgvector), `glossary/synonyms(source, tenant)`.

> ### ⚠️ Signatures are frozen — tenancy rides a contextvar, not a new parameter
> The adapter functions accept/return exactly what the VEDA callers already expect (same shapes/types, e.g. `RetrievalResult`). They do **not** gain a `tenant` argument — they read the ambient `(source, tenant)` from `veda_core.context.current()` (§4.1). And do **not** fake a psycopg cursor over the ORM: where a caller ran raw SQL (ANN cosine, connector SQL), the adapter keeps a **raw pgvector/psycopg path on the same Postgres Django manages** (through PgBouncer); structured rows go through the ORM. "Same Django database, two access styles" is the honest seam.

### 3.4 Rewire call sites (minimal edits)
Change the imports inside the enumerated `veda_core` storage modules to delegate to `storage_adapters`. Prefer a shim: keep `vector_store.store_fk_adjacency` as a one-line pass-through to `storage_adapters.writer.store_fk_adjacency`. This keeps every downstream import path unchanged.

### 3.5 Wire the ambient request/tenant context
Add `veda_core/context.py` (§4.1). Set the context in three places: an **ASGI middleware** on inference, a **DRF layer** on api (which also forwards `{source_id, tenant}` in the outbound inference call), and a **Celery task base/decorator** for every ingestion task. Make `storage_adapters` read `current()` for every scoped query. **Add `inference/concurrency.py::run_in_threadpool_with_context` and the lint rule banning raw offload; route all heavy sync work through the helper.**

### 3.6 Semantic-model assembler
Implement `storage_adapters/assembler.py::SemanticModelAssembler` (§8a): substrate rows → the exact `sm` dict + `all_cols`, `select_related`/`prefetch_related` to avoid N+1, Redis-cached by `substrate_version`. Replace `veda_hybrid._load_semantic_model()`'s file read with a call to the assembler (behind the same function name, so the front door is unchanged).

### 3.7 SLM seam (`_call_slm` Strategy)
Add `veda_core/slm/_call_slm.py` (§8b) with `OllamaBackend` and `vLLMBackend`, selected by `SLM_BACKEND`. Rewire the existing SLM call sites (IR emit, decompose, RAG synth, NL answer) to `call_slm(...)` as pass-through shims — same signature, wrapped by the existing timeout + circuit breaker. Dev/worker default to `ollama`; leave `vllm` selectable by env for Phase 5/7.

> **Exit criteria (all must pass)**
> 1. Every storage call site from 3.1 now routes through `storage_adapters` (inventory table fully checked off).
> 2. A unit test writes a FkEdge via the writer and reads it back via the reader with identical structure to the legacy `fk_adjacency` return.
> 3. pgvector `ann_search` returns the same top-k ordering as the legacy exact cosine on the fixture set **when run against the actual HNSW index** (not exact search) — see §7.1a; if it doesn't yet, this criterion is met by completing the §7.1a tuning, not by falling back to exact search.
> 4. No VEDA function signature changed; `veda_core` still imports standalone.
> 5. **Tenant isolation:** two interleaved `RequestContext`s never cross-read (concurrency test), including work dispatched through `run_in_threadpool_with_context`; `current()` raises when unset (fail-closed); the lint rule fails a build that calls raw `run_in_threadpool`.
> 6. **`sm` parity:** `SemanticModelAssembler.assemble(source, tenant)` is deep-equal to the legacy `veda_semantic_model.json` fixture.
> 7. **SLM seam:** every SLM call routes through `call_slm`; switching `SLM_BACKEND` between `ollama` and `vllm` changes only the backend, not outputs' shape, and both honor the timeout + circuit breaker.

---

## Phase 4 — Ingestion pipeline (Celery)

**Goal:** Turn the L0 pipeline into a resumable Celery task chain that populates the entire substrate via the writer.

### 4.0 Extract the ingestion driver out of `main.py` (PRESERVE the logic, hoist the orchestration)
The step *logic* lives in `veda_core/ingestion/`, but the *orchestration* that sequences it is currently inline in `main.py --ingestion-only` (a 73 KB driver). Before Celery can wrap stages, lift that orchestration into ten importable, side-effect-light **stage functions** in `veda_core/ingestion/` — each takes explicit inputs and **returns its artifact**; persistence happens in the Celery task via the writer, not inside the function. Copy the core logic verbatim; only hoist the glue. This is the one place Phase 4 does real extraction — budget for it.
**Acceptance:** running the ten stage functions in sequence from a plain script reproduces the legacy substrate for a test source (row-count match against a pre-migration run).

### 4.1 Job & stage models + Celery wiring
Implement `IngestionJob` and `IngestionStage`. Configure Celery (`config/celery.py`) with queues `ingestion, high, default` and the **`redis-broker`** instance. Each stage task runs inside `transaction.atomic()` (Unit of Work) so a partial failure rolls back cleanly and "resume" restarts from the last incomplete stage — **except stage 6 (embeddings), which uses batched commits** (see 4.2a). The `IngestionStage` checkpoint records the batch position so resume can continue mid-stage.

### 4.2 One task per L0 stage
Implement the ten tasks from §7, each calling the corresponding `veda_core/ingestion` function and persisting via the writer, updating its IngestionStage checkpoint. Make each idempotent (upsert by natural key).

### 4.2a Batched commits in the embedding stage (stage 6)
`task_embeddings` writes all six pgvector tables and is the largest write in the pipeline. **Do not wrap it in one `transaction.atomic()`.** Commit per pgvector table (or per N columns), advancing the `IngestionStage` batch checkpoint after each commit, so: no single long-held write lock, no one giant WAL burst, and a failure at 95% resumes from the last committed batch rather than rolling back the whole embedding set. Idempotent upsert-by-natural-key (4.2) makes partial-progress-then-resume safe, which is exactly what lets us drop full atomicity here.

### 4.3 Chain + resume + guards
Implement `task_ingest_source` building the chain, the ENCODER_MODE guard, "resume from last incomplete stage" (and mid-stage batch resume for stage 6). Flip the Source `ready` flag only when the whole job succeeds. Add admin actions ("Re-ingest", "Rebuild embeddings", "Regenerate glossary", "Warm caches").

### 4.4 Run full ingestion on a real source
Register a test Source and run `task_ingest_source`. Verify every substrate table fills and `task_warm_caches` populates `redis-cache` + triggers inference rehydrate fan-out.

> **Exit criteria (all must pass)**
> 1. A full ingestion job completes with all ten stages "success" in admin; row counts recorded per stage.
> 2. Substrate tables (structural, semantic, value, all pgvector, graph) are populated for the test source.
> 3. Killing the worker mid-run and resuming continues from the last incomplete stage — **including mid-stage-6 resume from the last committed embedding batch** (no duplicate rows).
> 4. `redis-cache` hot indices (FkEdge hash, value-sample SETs, glossary) are present after `task_warm_caches`.
> 5. Re-running the whole job is idempotent (row counts stable, no dupes).
> 6. Source `ready` flag flips to true only on full success.

---

## Phase 5 — Inference service

**Goal:** Stand up the ASGI service that warm-loads models and runs the identical `run_hybrid_query` flow, reading substrate from memory + pgvector, with the SLM on vLLM.

### 5.1 ASGI app + lifespan warm-load
Implement `inference/main.py` (Uvicorn) with a lifespan handler calling `inference/loaders.py` to hydrate the eight items in §8.1 (pools through PgBouncer). Expose `engine.py` holding the warm `get_engine()` singleton (relocated from `veda/runtime`). **Record the measured per-worker RSS and derive `workers_per_replica` from it (§8.1 callout).**

> ### ⚠️ PRESERVE get_engine semantics
> The warm engine object and its `retrieve()` behaviour come straight from `retrieval_engine_phase3`. You are wrapping it in a lifespan, not rewriting it. One warm engine per process, exactly as today.

### 5.2 Endpoints
Implement the four endpoints in §8.2. `/v1/run_hybrid_query` calls `veda_core.veda_hybrid.run_hybrid_query` verbatim (under an ASGI middleware that sets the request context, §3.5); `/v1/retrieve` calls the warm engine; `/v1/rehydrate` reloads from Redis/pgvector and **fans out via `redis-cache` pub/sub to all replicas** (§8.4); `/readyz` fails until models + FK map + glossary + KG + assembled `sm` + SLM-backend reachability are all green.

### 5.2a Rehydrate + cache-entry fan-out (Publisher/Subscriber)
Every `inference` replica subscribes to `veda:rehydrate:*` at startup. A rehydrate request (or `task_warm_caches`) publishes; each replica reloads the named scope and bumps its in-memory `substrate_version`. **Newly-written verified-cache entries publish on the same channel** so peers warm them (§6.6, §8.4). Verify that a post-ingestion warm reaches the whole fleet, not just the LB-selected replica.

### 5.3 Thread-pool the heavy sync work
Run encoder/reranker inference in a thread pool **via `run_in_threadpool_with_context`** so the event loop stays responsive under concurrency and the tenant context is carried in. Bounded fetch, `statement_timeout`, and read-only session preserved in execution. Note the in-worker GPU-serialization interaction with the SLM tier (§8.1 concurrency callout).

### 5.4 SLM backend = vLLM in prod; model cache offline
Set `SLM_BACKEND=vllm` for the inference tier in the prod profile (dev stays `ollama`). Confirm the image/volume ships BGE-M3, bge-reranker-v2-m3, MiniLM, and that vLLM/Ollama weights are pre-pulled — **both** `SLM_MODEL_NAME` (SQL/IR generation) **and** `NL_SUMMARY_MODEL` (`query/result_explainer.py`'s separate small summarization model, default `qwen2.5:1.5b-instruct`) — a missing `NL_SUMMARY_MODEL` doesn't fail startup, it silently degrades every answer to the deterministic template/row-count fallback. Set HF offline env so startup never reaches the network (zero-egress).

> **Exit criteria (all must pass)**
> 1. `/readyz` returns 200 only after all models + FK map + glossary + KG + `sm` + SLM backend are loaded/reachable.
> 2. POST `/v1/run_hybrid_query` returns a MultiResult identical (same status, rows, SQL) to the Phase 0 baseline for a fixture set — **with retrieval running on the tuned HNSW index (§7.1a), not exact search.**
> 3. Startup performs no network calls (verify with egress blocked).
> 4. **Warm-cache-eventually:** after a first query is cached, a second identical query is measurably faster **on the replica that cached it or after the cache-entry fan-out propagates**; because the LB may route the second query to a cold replica, the check asserts *either* a faster response *or* an observed cache-fan-out event followed by a faster response — not an unconditional speedup on the very next request. (A cross-replica miss re-runs the query correctly.)
> 5. Measured per-worker RSS recorded; `workers_per_replica` derived from it, not from CPU count.

---

## Phase 6 — DRF API, auth, tenancy, audit

**Goal:** Expose the platform over HTTP with the api tier as the only entry, calling inference and persisting audit — flow and refusal semantics preserved end to end.

### 6.1 QueryView + InferenceClient
Implement `POST /api/v1/query` in apps.query: validate, resolve tenant, call `InferenceClient.run_hybrid_query`, persist QueryLog, serialize MultiResult. The serializer preserves the `status` field and all terminal statuses verbatim.

### 6.2 Auth, tenancy, rate limit
Token/JWT auth; resolve tenant from the authenticated principal, never trust a client-supplied tenant for data scoping. DRF throttling + nginx rate limits. Every substrate read is tenant-scoped.

### 6.3 Audit (L9) + observability
QueryLog append-only per §6.6. Structured logging with request id propagated api→inference. Expose Prometheus metrics (latency per route, refusal-rate per status, cache hit-rate **and cross-replica cache-miss rate**, retrieval latency, SLM-backend queue depth, Postgres connections in use behind PgBouncer).

### 6.4 Ingestion & eval endpoints
Admin/DRF endpoints to trigger ingestion jobs and eval runs, guarded by staff permissions. Job status visible via API + admin.

> **Exit criteria (all must pass)**
> 1. End-to-end: `client → nginx → api → inference` returns correct MultiResult with preserved status for answered AND refused fixtures.
> 2. A refusal (e.g. temporal-refuse, no_table, ungrounded) surfaces as a structured refusal payload, not a 500.
> 3. Cross-tenant read attempt is denied; QueryLog records tenant, route, status, latency, parameterized SQL.
> 4. Metrics endpoint shows per-route latency, refusal-rate, cache hit/miss (incl. cross-replica miss), and PgBouncer connection usage; request id traces across both services.

---

## Phase 7 — Parity, hardening, production

**Goal:** Prove behavioural equivalence to the pre-migration engine, then harden for production.

### 7.1 Full parity suite
Run the existing evaluation harnesses (`evaluation/`) through the new `/api/v1/query` path and compare to the Phase 0 baseline captured from the legacy engine. **Status, chosen route, ladder rung, and parameterized SQL text must match exactly.** Result rows are compared as an **ORDER-BY-insensitive multiset**, not a positional list — and for queries that emit `LIMIT` without `ORDER BY` (the deterministic path emits `LIMIT 100`/`LIMIT 20` with no ORDER BY — pipeline.py:445,455,478,498,515,528), assert **row count + SQL text only**, since Postgres does not guarantee *which* rows an unordered LIMIT returns. Anything that diffs in status/route/rung/SQL is a migration bug; a row-multiset diff on an ordered query is a bug; a row diff on an unordered-LIMIT query is expected noise.

> ### ⚠️ Unordered `LIMIT` is a production behavior, not only a test carve-out
> The `LIMIT`-without-`ORDER BY` non-determinism above is inherited from the legacy engine, so it is "flow preserved" — but it isn't merely a parity-testing nuance. **In production the same query can return different rows run-to-run**, and any downstream consumer (or a user comparing two runs, or a screenshot vs. a re-run) that expects row stability will see what looks like a bug and file a ticket. Treat it as a **documented known behavior**: note it in the API docs / release notes and in the flow-preservation checklist (§19 item 15), not just as a test tolerance. Closing it (adding a deterministic tie-breaker `ORDER BY`) would change the flow and is therefore out of scope for the migration — but the behavior must be *flagged*, not silently carried.

### 7.1a HNSW parity tuning — resolve the exact-vs-approximate contradiction (REQUIRED before the parity gate can pass)
The retrieval parity fixtures were captured against **exact cosine** (Phase 0.5), but production serves **approximate HNSW** (§6.4). Running the gate against exact search and shipping HNSW is the single most likely silent regression in the plan, so this task closes the gap explicitly:

1. Build the HNSW indexes with candidate `(m, ef_construction)` values from settings.
2. **Sweep `ef_search`** (e.g. 40 → 100 → 200 → 400 …) against the Phase-0 exact-cosine top-k fixtures, measuring **recall@k** (fraction of queries whose HNSW top-k ordering matches exact) at each setting.
3. Raise `ef_search` (and, if needed, rebuild with larger `m`/`ef_construction`) until **recall@k = 1.0 on the fixtures**. Pin the resulting `HNSW_EF_SEARCH` (and build params) in `config.py`/settings (§9) so the shipping index *is* the gated index.
4. If recall@k = 1.0 proves impractical for a given source, the fallback is a **documented decision**, not a silent one: either (a) accept a stated recall tolerance and record that retrieval is no longer bit-identical (an explicit exception to "flow frozen"), or (b) route only the parity gate through exact search **and** document that production diverges — which the plan discourages. Default target is (3): tune until exact match.

**Acceptance:** the retrieval-parity row of §17 passes against the *live HNSW configuration at the pinned `ef_search`*, with the pinned params committed. No parity result anywhere in Phase 7 is produced under exact search unless option (b) is explicitly recorded.

### 7.2 Load & soak
Load-test the inference tier; **tune worker count to the measured per-worker RSS (§8.1), not CPU**; add replicas for throughput. Confirm PgBouncer keeps total Postgres connections under `max_connections` at peak (§3). Soak ingestion + query concurrently. Verify `redis-cache` eviction only affects the cache instance, never `redis-broker` (§3). Exercise the SLM tier under fleet concurrency and confirm vLLM batching (not a single-instance queue) absorbs it.

### 7.3 Backups, migrations, rollback
Postgres backups (substrate is rebuildable but expensive; back up anyway). Document a rollback: the legacy engine still runs from `veda_core`, so a bad deploy falls back to the Phase 0 path. Blue/green on the api + inference tiers.

### 7.4 prod compose + secrets + TLS
`docker-compose.prod.yml`: resource limits, restart policies, Docker secrets, TLS at nginx, no debug, read-only source roles enforced, **split Redis + PgBouncer + vLLM** wired. Healthchecks gate rollout.

> **Exit criteria (all must pass)**
> 1. Parity suite passes: MultiResult status + rows + route match the legacy baseline across the full eval set, **with retrieval on the tuned HNSW index (§7.1a)**.
> 2. Load test meets target p95 latency; inference scales horizontally without model reload per request; PgBouncer holds the connection ceiling; SLM tier does not serialize under fleet load.
> 3. Rollback rehearsed successfully; backups restore the substrate.
> 4. Production compose passes a security pass (no host ports except nginx, secrets not in env plaintext, zero-egress verified, split Redis + PgBouncer present).

---

## 17. Parity testing strategy (how the agent proves "flow unchanged")

Because the whole contract is "flow preserved", parity testing is not optional polish — it is the acceptance gate for the migration. Capture a golden baseline BEFORE Phase 3 and diff against it after every subsequent phase.

| Layer | What to assert | Fixture |
|---|---|---|
| Retrieval parity | top-k columns + ordering identical, **measured against the live HNSW index tuned to recall@k = 1.0 (§7.1a), never against exact search** | 50 representative queries per source + exact-cosine top-k snapshot |
| Router parity | same modality chosen (sql/rag/hybrid/nosql) | labelled router set |
| Escalation parity | same ladder rung fires (fast-path/cache/existence/aggregate/join/single-table) | explain-trace diff |
| Firewall parity | same gate outcome + same terminal status | grounding/qualifier/ir/invalid fixtures |
| Answer parity | parameterized SQL text identical; rows compared as an ORDER-BY-insensitive multiset; for `LIMIT`-without-`ORDER BY` queries assert row COUNT + SQL only (unordered LIMIT is non-deterministic — also a documented production behavior, §7.1) | full eval suite |
| Cache parity | same skip rules (existence/fast-path/temporal never cached); idempotent `ON CONFLICT` write; cross-replica warm-cache-eventually (a miss re-runs correctly) | cache-behaviour fixtures |
| Tenancy parity | every substrate read scoped to `(source, tenant)`; no cross-tenant leak; `current()` fail-closed; offload runs under copied context | interleaved-context + cross-tenant fixtures |
| SLM-backend parity | `ollama` and `vllm` backends produce shape-identical outputs through `_call_slm`; timeout + breaker honored on both | SLM-backend fixtures |
| `sm` parity | assembled `sm` deep-equal to legacy `veda_semantic_model.json` | home-schema snapshot |

> ### Golden baseline capture
> In Phase 0, run the full eval suite through the untouched legacy engine and serialize every MultiResult (status, rows, sql, route, ladder-rung from the explain trace) to a JSON snapshot committed to the repo — **plus the exact-cosine top-k ordering per retrieval fixture**, which is the target §7.1a tunes HNSW against. Every later phase runs the same suite through the new path and diffs against this snapshot. A diff in status or route is a migration bug, not a tolerance issue.

---

## 18. Risk register & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **HNSW ANN ordering differs from legacy exact cosine (parity gate passes under exact search, then drifts once HNSW is live)** | **Silent retrieval drift → wrong anchors, undetected because the gate was green** | **Run the parity gate against the actual shipping HNSW config; sweep `ef_search` (+ `m`/`ef_construction`) to recall@k = 1.0 on the exact-cosine fixtures and pin it in settings (§7.1a). Never let parity pass under exact search while shipping HNSW; if a recall tolerance is accepted instead, document it as an explicit exception to "flow frozen".** |
| Model RAM blowup on inference tier (three transformers × workers × replicas) | OOM / cost, discovered in load test | Measure per-worker RSS; derive `workers_per_replica` from it, not CPU (§8.1); prefer more replicas over more workers; GPU only on inference+SLM; never load models in api. |
| Verified-cache skip rules lost in ORM move | Cached refusals / existence bugs | Port skip rules verbatim into writer (§6.6 callout); cache-parity fixtures. |
| Verified-cache is a query-time write from N replicas | Race / lost writes | `INSERT … ON CONFLICT (query_hash) DO NOTHING`; fire-and-forget off the latency path; inference granted write access explicitly (§6.6). |
| Verified-cache **read** staleness across replicas | Flaky "second query faster" check; cross-replica misses | Warm-cache-eventually property (safe: a miss re-runs); push new entries through the rehydrate pub/sub fan-out so peers converge (§6.6, §8.4); acceptance check phrased accordingly (Phase 5 exit 4). |
| Tenant context unset or **leaked into thread-pool offload on a future code path** | Cross-tenant read or hard error | `current()` fails closed; **wrap the offload primitive once as `run_in_threadpool_with_context` and ban raw `run_in_threadpool` via lint** (§4.1) so coverage isn't left to discipline; interleaved-context + fail-closed tests; raw adapter queries always include `source_id`/`tenant`. |
| Assembled `sm` drifts from the legacy dict shape | Silent anchor/grounding/glossary changes | Deep-equal parity fixture (§8a); assembler versioned; query path gated until parity passes. |
| Single SLM instance (Ollama) saturates under N inference replicas | SLM/decomposer/RAG timeouts, latency spikes | **Move query-time SLM to vLLM behind `_call_slm` (§8b)**; bounded concurrency gate; scale SLM tier with inference; timeout + breaker + deterministic fallbacks preserved. |
| Embedding stage's single atomic transaction | Long write-lock, huge WAL burst, full rollback at 95% | Batched commits per pgvector table within stage 6 (§4.2a); idempotent upsert makes partial-progress-then-resume safe; other stages keep single-transaction UoW. |
| Rehydrate reaches only one replica | Stale FK/glossary/KG/`sm` on the rest of the fleet | Redis pub/sub fan-out (§8.4); down replicas catch up on next lifespan warm-load. |
| **Redis is a shared SPOF (broker + cache + indices + pub/sub in one process)** | A cache-eviction storm or OOM degrades ingestion, hot path, and rehydrate simultaneously | **Run `redis-broker` and `redis-cache` as separate instances in prod (§3)**; broker unbounded/no-evict; `allkeys-lru` on cache only; cluster later. |
| **Postgres connection-pool exhaustion under two-pool-per-worker × N×M scaling** | Horizontal scaling hits a connection ceiling before a CPU one | **PgBouncer (`pool_mode=transaction`) in front of Postgres; every pool connects through it (§3, Phase 1.1)**; verify under load (§7.2). |
| SLM backend unavailability at query time | NL answer / SLM failures | Deterministic row-count fallback preserved (NL_ANSWER); SLM timeouts preserved; circuit-breaker in `_call_slm` (§8b). |
| Unordered `LIMIT` returns different rows run-to-run in production | Support tickets / apparent instability for downstream consumers | Documented known behavior (§7.1, §19 item 15); inherited from legacy so out of scope to "fix", but flagged in API docs/release notes, not silently carried. |
| Tenant leakage across substrate | Data exposure | TenantScopedModel from day one; tenant resolved server-side; cross-tenant tests in Phase 6. |
| Ingestion partial failure leaves half a substrate | Inconsistent query results | Checkpointed stages + resume; query path reads only from a Source marked `ready` (§5). |
| Drift between `config.py` and Django settings | Two sources of truth | Bridge imports `config.py`; no duplicated values; test asserts equality. |

---

## 19. Appendix — flow-preservation checklist

A final gate before sign-off. Each item is copied from ARCHITECTURE.md and must be demonstrably true in the migrated system.

1. `run_hybrid_query` is the single front door and always returns MultiResult (1 item plain, N compound).
2. Routing default is `sql`; router-off falls to the deterministic head.
3. Decompose asymmetry intact: clean SQL self-certifies and skips decomposer; RAG/hybrid/nosql decompose first; deterministic refusal triggers decompose.
4. Escalation ladder order preserved: temporal → intent → existence → fast-path → cache → retrieve → anchor/vet → branch → firewall → execute → answer.
5. Deterministic resolvers each skip the LLM when they fire; single-table LLM is last resort; LLM never authors joins.
6. Firewall gate order preserved: value grounding → qualifier completeness → IR equivalence (LLM SQL) → validate+parameterize → execute → NL answer → cache-back.
7. Terminal statuses unchanged: `answered · no_table · clarify · refuse · ungrounded · qualifier_dropped · ir_mismatch · invalid · exec_error`.
8. Invariants hold: read-only AST-enforced, parameterized-only, refuse-over-guess.
9. `fk_adjacency` (FkEdge) is the join engine's source of truth; `config.py` is the engine's single source of truth.
10. Zero-egress: all inference local (encoders + SLM backend on `veda_net`); no client data leaves the deployment.
11. Tenancy rides `veda_core.context` (contextvars); engine signatures unchanged; `current()` fails closed when unset; every substrate read scoped to `(source, tenant)`; thread-pool offload runs under a copied context via the single lint-enforced helper.
12. Assembled `sm` dict is deep-equal to the legacy semantic model for the home schema (`SemanticModelAssembler`).
13. Verified-cache write is idempotent (`ON CONFLICT`) and off the latency-critical path — inference's only write — and new entries fan out to peer replicas (§8.4).
14. Rehydrate fans out to every inference replica (Redis pub/sub); no replica serves stale substrate after ingestion.
15. **Retrieval parity is proven against the live HNSW index tuned to recall@k = 1.0 (§7.1a), not exact search; the pinned `ef_search`/build params are the shipping params.**
16. **SLM calls route through `_call_slm` (§8b); the query-time backend is vLLM in prod, Ollama in dev/ingestion; timeout + circuit breaker + deterministic fallbacks preserved on both.**
17. **Unordered-`LIMIT` non-determinism is a documented known production behavior (§7.1), not only a parity carve-out.**

---

*End of plan — build phase by phase, gate on exit criteria, diff against the golden baseline.*
