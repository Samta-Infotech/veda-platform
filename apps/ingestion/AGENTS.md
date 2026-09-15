# apps/ingestion/ — the Celery ingestion driver

Drives the engine's offline build in a subprocess and tracks it. Full reference:
[../../docs/INGESTION.md](../../docs/INGESTION.md).

| File | Role |
|------|------|
| `tasks.py` | `task_ingest_source(source_id, tenant, force, skip_llm, resume, ...)` — creates `IngestionJob` + ordered `IngestionStage` rows; `_guard_embedding_model_change` (`EMBEDDING_MODEL_ID` vs last successful job); injects `Source.as_engine_env()`; `_run_engine_pipeline` → `subprocess.Popen(["python","-u","-c", prog], cwd=veda_core)`; `_build_engine_command` routes by `source.source_kind()` (relational → `main.run_ingestion`; else → `source_dispatcher.dispatch_ingestion`); `_consume_engine_output` parses `[[STAGE]]` events (`_LAYER_STAGE_TO_ROW`) into `IngestionStage` transitions (the legacy `[N/NN]` marker protocol — `_MARKER_RE`/`_ENGINE_STEP_TO_STAGE`/`_apply_step_marker` — was dead code for a format the layered path never emitted; removed 2026-09-10, P2-3). On success → `task_warm_caches` → `storage_adapters.writer.warm()` → `Source.ready=True`. Flag-gated post-steps: `_sync_catalog_if_enabled` (RBAC catalog), `profile_source_if_enabled`, `build_source_items_if_enabled`. |
| `models.py` | `IngestionJob` (+ `encoder_mode` column reused for the model-change guard) + `IngestionStage` (per-stage status, `STAGE_ORDER`). |
| `admin.py` | job/stage observability. |
| `apps.py` / `__init__.py` / `migrations/` | config + marker + tables. |

## Gotchas
- **The subprocess is used to isolate the engine's top-level `config` module from the
  Django `config` package** — they collide in one interpreter.
- **Resume is stage-level skip**, not resume-from-N: `VEDA_RESUME=1` is set when
  `resume=True` OR a prior FAILED job exists. L3 skips if the sm file exists; L4 biencoder
  skips if `column_embeddings_v2` has rows. L1/L2 always re-run.
- **The 10-task per-stage Celery chain** (`STAGE_ORDER` + `task_schema_scan` …
  `task_unified_graph`) is a **`NotImplementedError` skeleton** — not the running path.
- `ingestion_mode` param is accepted for signature compat and **does nothing** (the monolith
  was deleted outright, not kept one release).
