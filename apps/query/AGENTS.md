# apps/query/ — the REST query entry + audit

`POST /api/v1/query` → resolve scope → call `inference/` over HTTP → `QueryLog`.
Full flow: [../../docs/ARCHITECTURE.md §3](../../docs/ARCHITECTURE.md),
[../../docs/QUERY_ENGINE.md](../../docs/QUERY_ENGINE.md).

| File | Role |
|------|------|
| `views.py` | `QueryView` (`POST /api/v1/query`, `permission_classes = [AllowAny]`) — resolves tenant + RBAC permissions + source scope + data-scope + source-profiles, calls `InferenceClient`, writes `QueryLog`. `IngestTriggerView` / `EvalTriggerView` (`IsAdminUser`). |
| `inference_client.py` | Stdlib-`urllib` HTTP client to inference. `run_hybrid_query` / `stream_hybrid_query` (SSE) / `retrieve`. Sets 8 headers (`X-Veda-Source-Id` / `-Source-Ids` / `-Tenant` / `-Data-Scope` / `-Source-Profiles`, `X-Request-Id`, …). Any transport failure (`HTTPError` **or** `URLError`) → `InferenceUnavailable`. **No `veda_core` import, no retry, no circuit breaker** (three docs claimed one). Timeout `INFERENCE_TIMEOUT_S` default **300 s**. |
| `scope.py` | `resolve_query_scope()` (server-side source-SET resolution, RBAC-narrowed then request-pin intersected), `permitted_source_ids()` (`None` = no narrowing / `set()` = deny), `source_profiles_for()`, `_ready_source_ids()`. `QueryScopeError` / `SourceAccessDenied` (→ 403) / `NoReadySource` (→ 503). |
| `models.py` | `QueryLog` (append-only audit; **plain `models.Model`, not tenant-scoped**, has a `tenant` CharField) + `TerminalStatus` (9 frozen statuses). |
| `urls.py` | `query`, `admin/ingest`, `admin/eval`. |
| `admin.py` | `QueryLogAdmin` — read-only. |
| `migrations/0001..0004` | `QueryLog` init; `0003` `cache_hit`; `0004` token counts. |

**Absent:** no `throttles.py`, no `serializers.py` (throttling is global in
`REST_FRAMEWORK`; the view reads `request.data` directly).

## Gotchas
- **`AllowAny`** — `/api/v1/query` answers anonymously (tenant `"default"`). RBAC narrowing
  still applies when `VEDA_RBAC_MODE != off`.
- **Denials bypass audit** — the 403 (forbidden) and 503 (no-ready-source) early returns
  write **no `QueryLog` row**. Only successful inference calls and `InferenceUnavailable` do.
- **`QueryLog.status` accepts non-canonical values** (`"forbidden"`, `"unavailable"`,
  `"unknown"`) — `choices` don't validate on `.create()`.
- `cache_hit` is derived from the engine tagging the result table `"(cached)"`.
- Related possible bug: `storage_adapters/reader.py::ann_search` may query the wrong
  Postgres DB — see [../../docs/backlog/query-engine-open-items.md](../../docs/backlog/query-engine-open-items.md).
