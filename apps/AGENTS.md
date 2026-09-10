# apps/ — the Django platform (ten bounded contexts)

The "thin tier" — it imports **no `veda_core`**; it calls the `inference/` service over
HTTP. Whole-system map: [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

| App | Files | What it does | Deep doc |
|-----|-------|--------------|----------|
| `core` | 12 | `TenantScopedModel` + `TenantManager` (ambient auto-filter, fail-**open** to unscoped when no context); `RequestIdMiddleware`; `TenantTask` (Celery base binding context in a copied contextvars ctx); `build_veda_settings()` (config → settings bridge); `/readyz` + `/metrics`; `token_revocation` (shared leaf); `messages` (all user-facing copy). | — |
| `sources` | 23 | `Source` (connection on the row + `dialect` + routing-catalog fields + `as_engine_env`); `SourceConnectionProfile`; `SourceItem`; the profilers. `ready` flips only on ingestion success. | [MULTI_SOURCE.md](../docs/MULTI_SOURCE.md) |
| `substrate` | 13 | Every ingestion output as a model (structural / semantic / value-grounding / graph / verified-cache / normalized `sm`). 2 `managed=False` pgvector mirrors remain; the rest dropped by migrations 0006–0008. | [ARCHITECTURE.md §6](../docs/ARCHITECTURE.md) |
| `ingestion` | 7 | `task_ingest_source` runs the engine pipeline in a subprocess, streams `[[STAGE]]` markers into `IngestionJob` / `IngestionStage`, then `task_warm_caches`. A 10-task Celery chain is a `NotImplementedError` skeleton. | [INGESTION.md](../docs/INGESTION.md) |
| `query` | 13 | `QueryView` (`POST /api/v1/query`, `AllowAny`); `scope.py` (server-side source-set resolution, RBAC-narrowed); `InferenceClient`; `QueryLog` (audit); staff `IngestTriggerView` / `EvalTriggerView`. | [QUERY_ENGINE.md](../docs/QUERY_ENGINE.md) |
| `chat` | 14 | `ConversationQueryView` (`AllowAny` + manual 401); `ConversationQueryService` (turn orchestration, SSE bridge); deterministic table / thinking-message / turn-event / visualization helpers. | [CHAT.md](../docs/CHAT.md) |
| `authentication` | 7 | Login / refresh / logout / password-change. JWT via `simplejwt`, behind `VEDA_JWT_AUTH` (default **off** → login returns a placeholder token). Redis lockout. No models. | [RBAC.md](../docs/RBAC.md) |
| `access_management` | 61 | The RBAC data model + `PermissionResolver` + Gate 1 / Gate 2, behind `VEDA_RBAC_MODE` (default **off**). Admin CRUD endpoints. | [RBAC.md](../docs/RBAC.md) |
| `evaluation` | 7 | `task_run_eval` runs a query set through inference → `EvalRun` / `EvalCaseResult` + HTML report. | [EVALUATION.md](../docs/EVALUATION.md) |
| `__init__.py` | — | namespace package marker. | — |

## URL roots (`config/urls.py`)

`/api/v1/` includes **five** app urlconfs: `apps.query.urls`, `apps.chat.urls`,
`apps.authentication.urls`, `apps.access_management.urls`, `apps.sources.urls`.
`/admin/`, `/healthz`, `/readyz`, `/metrics` are wired directly in `config/urls.py`.

## Cross-cutting gotchas

- **`/api/v1/query` answers anonymously** (`AllowAny`, tenant `"default"`);
  **`/api/v1/conversations/query` requires a real principal** (401 otherwise).
- **Denials are not audited** — the 403 / 503-unavailable early returns in `QueryView`
  write no `QueryLog` row.
- **RBAC and JWT are built and wired end-to-end** but flag-gated off
  (`VEDA_RBAC_MODE` / `VEDA_JWT_AUTH`). "Skeleton" in older docs understates this.
- Tenant is **not yet principal-derived** — `user.username` when authenticated, else
  `data["tenant"] or "default"`.
- Only substrate models are tenant-scoped; RBAC, chat, and `QueryLog` models are not.
