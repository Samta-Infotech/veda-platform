# apps/sources/ — the source registry

`Source` is the single registry the whole platform routes and connects through. Onboarding
a source is a data operation (register a row → `POST /api/v1/admin/ingest`).
[../../docs/MULTI_SOURCE.md](../../docs/MULTI_SOURCE.md).

| File | Role |
|------|------|
| `models.py` | `Source` (**plain `models.Model`, global, not tenant-scoped**): connection on the row (`host` / `port` / `dbname` / `db_user` / `password_env` \| `password_inline` / `connection_secret_ref`), `dialect` (16 kinds incl. document/datalake), `connector_type`, `ready`, `status`, `exclude_tables` / `schema_filter`; document/datalake fields (`source_path` / `doc_formats` / …); **routing catalog** (`domain_tags` manual, `description` manual-wins-else-auto, `description_generated`, `is_canonical`). Methods: `resolve_password()`, `connection()`, `as_engine_env()` (→ `VEDA_SOURCE_*`), `source_kind()`, `as_source_config()`. `SourceConnectionProfile` (1:1 — pool sizing, `statement_timeout_ms` default 30000). `SourceItem` / `SourceItemType` (uniform per-item routing metadata, global, unique `(source, item_type, item_key)`). |
| `serializers.py` | `DataSourceListSerializer` + `serialize_source()` + `group_by_type()` + a `config`-fallback (`get_config_sources()` from `veda_core.config`). |
| `views.py` | `DataSourceListView` (`GET` / `POST /api/v1/data-sources/list`, extends `AdminView`, `required_permission=SOURCE_MANAGE`) — **ready-only, un-paginated** (deliberate deviation, noted in its docstring). |
| `urls.py` | one route: `data-sources/list`. |
| `admin.py` | `SourceAdmin` — fieldsets, `ingest` / `test_connection` actions (`test_connection` is a Phase-6 **stub**). |
| `item_profiler.py`, `source_profiler.py`, `management/commands/` | item/source profiling for multi-source routing, flag-gated (`SOURCE_PROFILER_ENABLED` / `SOURCE_ITEM_PROFILER_ENABLED`). |

## Gotchas
- **`ready` flips to `True` only on full ingestion success** (`apps/ingestion/tasks.py`); a
  failed job leaves `ready=False`, `status=FAILED`. The query path reads only `ready=True`.
- `Source` is **global** — no tenant column. Tenancy is on the substrate, not the registry.
- Query-tier counterpart: `storage_adapters.reader.source_connection()` resolves the same
  `sources_source` row via raw SQL for L7 execution; `veda/runtime.get_db_config()` uses it
  when a request context is set, else falls back to `VEDA_SOURCE_*` env.
- `run_homzhub_query.sh` (in `veda_core/`) exports `VEDA_SOURCE_*` at the live **DigitalOcean
  prod** source — not the registry. Don't run casually.
