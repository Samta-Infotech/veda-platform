# inference/ — the warm ASGI query service

The only tier that imports `veda_core`. One warm engine per uvicorn worker, per
`(source, tenant)` scope. The api tier calls it over HTTP.
[../docs/ARCHITECTURE.md §1](../docs/ARCHITECTURE.md).

| File | Role |
|------|------|
| `main.py` | `create_app()` FastAPI. Lifespan `hydrate()` + `_start_rehydrate_subscriber()` (daemon thread, `psubscribe("veda:rehydrate:*")` → clears `_SM` / engines / reader caches / registries). HTTP middleware `_tenant_context` sets `RequestContext` from `x-veda-source-id` / `-source-ids` / `-tenant` — **only when both source-id and tenant are present** — + parses `x-veda-data-scope` (**fail-closed** — malformed → `allowed_resources = ()`) + `x-veda-source-profiles` (fail-open). Mounts `health` / `retrieve` / `hybrid` routers. |
| `loaders.py` | `hydrate()` — checks `SEMANTIC_MODEL_FILE` exists, warms the retrieval engine + BGE-M3 dense/sparse + cross-encoder reranker + SLM `prewarm` + NL-summary SLM. Each best-effort, non-fatal, `[warmup]` to stdout. `_STATE` readiness dict; `readiness()`. |
| `concurrency.py` | `run_in_threadpool_with_context(fn, …)` — `copy_context()` + `anyio.to_thread.run_sync`. **Raw offload is lint-banned** in `inference/` + `veda_core/` (`scripts/lint_no_raw_offload.sh`). |
| `routes/hybrid.py` | `POST /v1/run_hybrid_query` (→ `run_hybrid_query` verbatim in threadpool-with-context) + `POST /v1/run_hybrid_query/stream` (SSE; pipeline on a daemon thread with `copy_context()` + `with_context`). `_serialize()` strips `_INTERNAL_ONLY_KEYS = {context, trace, _debug}` at every depth, `Decimal → float`. |
| `routes/retrieve.py` | `POST /v1/retrieve` (`_load_scoped_sm` → `get_engine(sm).retrieve` → `filter_retrieval_results` RBAC) + `POST /v1/rehydrate` (local clears + re-publish fan-out). |
| `routes/health.py` | `GET /healthz` (always 200) + `GET /readyz` (503 until `_STATE["ready"]` — the semantic-model file exists). |
| `engine.py` | `get_engine()` → **raises `NotImplementedError`** ("Phase 5.1"). **Dead** — the real singleton is `veda_core/veda/runtime.get_engine`. Nothing imports it. |
| `routes/__init__.py` | empty. |

## Gotchas
- **No `/metrics`** on this tier — per-query observability is the explain-trace path.
- A direct/un-headed call to `/v1/run_hybrid_query` sets **no context** → downstream reads
  fail closed (`context.current()` raises).
- The streaming route never 500s mid-answer: exceptions become an SSE `event: error` frame;
  the client maps a mid-stream drop to `InferenceUnavailable`.
- `working_dir` is `/app/veda_core` in compose (engine relative paths).
