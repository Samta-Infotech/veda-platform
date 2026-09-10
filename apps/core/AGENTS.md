# apps/core/ — shared framework pieces

Tenancy primitives, the request-id middleware, the config→settings bridge, health/metrics,
and a few shared leaves. No URLs of its own (health/ready/metrics are wired in
`config/urls.py`).

| File | Role |
|------|------|
| `models.py` | `UUIDPrimaryKeyModel`, `TimeStampedModel`, `TenantQuerySet`, `TenantManager` (ambient `(source,tenant)` auto-filter; **falls back to the UNSCOPED queryset when `context.current()` raises** — a documented fail-open for admin/migrations), `TenantScopedModel` (source FK CASCADE + tenant + `TenantManager`). `all_tenants()` is the explicit bypass — ingestion writers use it directly. Only substrate models inherit this. |
| `middleware.py` | `RequestIdMiddleware` — mints/propagates `X-Request-Id`, sets `request.request_id`, echoes the header. Last in `MIDDLEWARE`. |
| `tenant_task.py` | `TenantTask` — Celery base that binds `RequestContext(source_id, tenant)` from task args inside `copy_context().run(...)`. |
| `settings_bridge.py` | `build_veda_settings()` — bridges 14 engine flags (`EMBEDDING_MODEL_ID`, `TOP_K`, `TOP_K_TO_LLM`, `QUERY_ROUTER_ENABLED`, `SLM_MODEL_NAME`, `SLM_BACKEND`, `VLLM_BASE_URL`, `IR_JOIN_FREE_ENABLED`, `FAST_PATH_ENABLED`, `QUERY_DECOMPOSE_ENABLED`, `HNSW_M` / `_EF_CONSTRUCTION` / `_EF_SEARCH`). Precedence: **fallback default → `veda_core.config` attr → `VEDA_<name>` env**. `_cfg` import guarded (degrades to fallback). |
| `views.py` | `readyz()` — 200/503, gates on Postgres + `redis-cache` + `redis-broker` + inference `/readyz`; SLM probed but non-gating; **no BGE probe**. `metrics()` — Prometheus text from `QueryLog` aggregates + PgBouncer `SHOW POOLS`, dependency-free, fail-soft. |
| `api.py` | Response-envelope helpers `success()` / `error()` / `invalid_payload()` / `iso_z()` / `human_date()`. "Absent, not null." |
| `messages.py` | `MESSAGES` nested dict — every user-facing API string in one place (`auth` / `user` / `role` / `grant` / `resolver` / `chat` / …). |
| `token_revocation.py` | `revoke_all_refresh_tokens(user_id)` — blacklist every live `OutstandingToken`. Shared by `authentication` (replay, password change) and `access_management` (deactivation) so neither imports the other. |
| `admin.py` | empty. |

## Gotchas
- `TenantManager` **fail-opens** to unscoped when no context is set — the opposite of the
  raw `storage_adapters.reader` reads, which fail closed. Know which you're in.
- `metrics()` accepts free-form `QueryLog.status` values (`"forbidden"`, `"unavailable"`)
  that aren't in `TerminalStatus` — Django `choices` don't validate on `.create()`.
