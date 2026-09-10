# OBSERVABILITY — explain traces, MLflow, health & metrics

Expands [ARCHITECTURE.md](ARCHITECTURE.md) §12. Written from a read of
`veda_core/veda/explain.py`, `mlflow_observability/`, `apps/core/views.py`,
`inference/routes/health.py` on 2026-09-09.

---

## 1. The design — zero-touch

```
inference process                          separate process (opt-in)
┌──────────────────────────┐               ┌───────────────────────────────┐
│ run_hybrid_query         │  append 1     │ python -m mlflow_observability │
│  └ ExplainTrace          │  JSON line    │        watch                  │
│     veda/explain.py      │  per query    │  · tail (byte-offset + SHA1)  │
│                          │──────────────▶│  · map_record → RunSpec       │
│ EXPLAIN_TRACE_* = True    │ veda_core/    │  · MlflowSink.log             │
│ (config.py, default ON)  │ logs/         │                               │
└──────────────────────────┘ explain_trace │  MLflow tracking server       │
                             .jsonl        └───────────────────────────────┘
```

- **The engine always writes the trace.** `EXPLAIN_TRACE_ENABLED` /
  `_VERBOSE` / `_PERSIST` default to `True` (`veda_core/config.py:2064-2066`). One JSON
  line per query is appended to `veda_core/logs/explain_trace.jsonl`.
- **`mlflow_observability/` is a separate process.** Nothing in `veda_core/`, `apps/`,
  `inference/`, or `chatbot/` imports it, and it never imports the engine. An MLflow
  outage **cannot** affect inference — this holds *structurally*, not by discipline.
- **It is NOT on by default.** The trace file is always written, but nothing consumes it
  unless you start the exporter: `python -m mlflow_observability watch`, or the Docker
  sidecar `docker compose -f docker-compose.yml -f docker-compose.mlflow.yml up -d
  mlflow mlflow-exporter`. Neither the dev nor prod compose includes those services.

---

## 2. The exporter (`mlflow_observability/`)

| File | Role |
|------|------|
| `__init__.py` | Package doc + `__version__`. States the decoupling contract. |
| `__main__.py` | `sys.exit(main())` → `cli.main`. |
| `cli.py` | Subcommands: `export` (one pass), `watch` (sidecar loop), `ui` (launch `mlflow ui`; refuses on Python ≥ 3.14), `status`, `evaluate` (golden set → aggregate run, `--min-pass-rate` CI gate, exit 2), `selftest` (throwaway sqlite; asserts 3 runs + signal metrics + torn-line deferral), `demo` (seed 4 sample runs tagged `veda.demo=true`). |
| `settings.py` | Frozen env-driven `Settings`. Defaults: trace log `veda_core/logs/explain_trace.jsonl`, sqlite store under `mlflow_observability/mlflow_data/`, experiment `VEDA-Query-Observability`, poll 5 s, `param_value_max=500`. |
| `exporter.py` | Checkpointed JSONL tailer. Byte offset + first-line SHA1 (rotation/truncation detector), partial last line deferred, malformed line skipped + counted, an MLflow error stops the pass **before** advancing the checkpoint (retry, never lose). `watch()` reconnects on a tracking-server restart. |
| `mapper.py` | Pure (no `mlflow` import) `map_record(record) -> RunSpec`. Spec-named metrics + a generic per-section sweep (nothing recorded is dropped) + Layer-2 signal-score flattening + column-funnel metrics + refusal taxonomy + `coverage.json` (per-run present-vs-missing datapoints). `SPEC_GAPS` lists what the engine trace still does not emit. |
| `evaluate.py` | Golden-eval harness. `load_golden` (JSONL `{query, gold_tables, gold_columns?}`), `score_case` (`passed = answered AND gold_tables ⊆ tables-used`), `aggregate`, `log_report` → one aggregate run to `VEDA-Golden-Eval`. Injected runner → model-free unit tests. |
| `README.md` | Accurate implementation writeup of the (archived) `mlflow_impl.md` spec + env reference + "Known gaps". |
| `requirements.txt` | `mlflow>=2.14` only. |

Tests: `tests/test_mlflow_mapper.py`, `tests/test_mlflow_evaluate.py` (both model-free).

---

## 3. What is logged per query

**Params**: `query`, `route`, `intent`, `table`, `anchor`, `action`, `status`,
`refusal`, per-section strings, `retrieval.top1_column` / `top_columns`,
`columns.used_in_sql`, `rerank.top1_before` / `after`.

**Metrics**: `total_latency_ms`; `<section>.duration_ms` + `.start_offset_ms` for the 9
layers (durations *derived* from first-touch `_ms` gaps); `routing_confidence`,
`join_confidence`, `answer_confidence`; `pipeline_success` / `pipeline_refused`;
retrieval / graph / validation counts; `repair_count`; `sql_length` / `sql_join_count` /
`limit_present`; token totals (only when the trace carries an `llm_usage` section);
Layer-2 per-signal `retrieval.top1_<signal>_score` / `_mean` / `_top1_vs_top2_gap`
(nested `signals` short names normalized to spec keys); `reranker_changed_top1`;
`columns.candidate_count` / `used_in_sql_count` / `selection_ratio`.

**Tags**: `veda.route` / `status` / `intent` / `table` / `action`, `veda.query_hash`,
`veda.environment`, `veda.line_fingerprint`, `veda.trace_id`, `veda.git_sha`,
`veda.model.sql` / `summary`, `veda.outcome`, `veda.refusal_category`.

**Artifacts**: `layers/*.json`, `sql/generated_sql.sql`, `trace/full_trace.json`,
`trace/why.txt`, `layers/signal_scores.json`, `layers/selected_columns.json`,
`coverage.json`.

**Spans**: a best-effort waterfall replay via the low-level MLflow tracing client
(`_emit_spans`); any failure swallowed.

**Generic sweep**: every scalar in any trace section is promoted (numeric → metric,
string → param) — a new engine datapoint appears with **zero exporter change**.

### Second experiment — `VEDA-Golden-Eval`

`mlflow_observability evaluate` (or `cli.py evaluate`): runs a golden JSONL through the
inference tier, logs ONE aggregate run (pass_rate, table_hit_rate, latency p50/p95,
tokens) + `eval/cases.json`. `--min-pass-rate` gives a CI exit-2 gate.

### Known gaps (`SPEC_GAPS` / README "Known gaps")

The engine trace does **not** emit: `tenant` / `session` / `user` id,
`estimated_cost`, `cpu_usage` / `memory_usage`, `rerank_model` / `rerank_latency`,
memory-layer metrics, visualization metrics. The schema is pre-built for them
(enumerated in `coverage.json`, picked up by the generic sweep the moment the trace
carries them); capturing them needs pipeline edits — see
[`backlog/query-engine-open-items.md`](backlog/query-engine-open-items.md).

`mlflow ui` is broken on Python ≥ 3.14 (guarded in `cli.py`). `mapper.SECTIONS` mirrors
`explain.py._SECTIONS` — keep them in step.

---

## 4. The compose sidecar (`docker-compose.mlflow.yml`)

Additive — zero changes to existing services.

| Service | Image | Notes |
|---------|-------|-------|
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.22.0` | `mlflow server :5000` (sqlite `mlflow.db` on `mlflow_data`, artifacts `/mlflow/artifacts`). Published `${MLFLOW_UI_PORT:-5001}:5000` (5001 — macOS AirPlay squats 5000). Healthcheck `/health`. |
| `mlflow-exporter` | same image | `python -m mlflow_observability watch`, repo mounted **`:ro`**, checkpoint on its own `mlflow_state` volume, `VEDA_ENVIRONMENT` + `VEDA_GIT_SHA` for provenance, `depends_on: mlflow healthy`. |

---

## 5. Health & metrics endpoints

### api tier (`config/urls.py` → `apps/core/views.py`)

| Endpoint | Behavior |
|----------|----------|
| `GET /healthz` | `{"status":"ok"}` always (liveness). Compose healthcheck. |
| `GET /readyz` | `{"status":"ready"|"degraded","checks":{…}}`, **200 / 503**. **Gates on**: Postgres `SELECT 1` (via PgBouncer), `redis-cache` ping, `redis-broker` ping, inference `/readyz` reachability. **Reported but non-gating**: SLM backend probe (Ollama `/api/tags` / vLLM `/v1/models`). All lazy imports, fail-soft. **No BGE / `METAL_EMBED_URL` probe** — a down host Metal server is invisible here. |
| `GET /metrics` | Prometheus text `version=0.0.4`, **dependency-free** (no `prometheus_client`). Families: `veda_queries_total{status}`, `veda_queries_all`, `veda_refusal_rate`, `veda_route_latency_ms_avg{route}` + `veda_route_queries_total{route}`, `veda_cache_hits_total` / `_misses_total`, `veda_query_latency_ms_avg`, `veda_pgbouncer_sv_active{database}` (from `SHOW POOLS`). All derived from `QueryLog` aggregates. Fail-soft. **Not exposed through nginx** (`PRODUCTION_READINESS_PLAN.md` B4 would add an allow-listed location). |

### inference tier (`inference/routes/health.py`, `inference/loaders.py`)

| Endpoint | Behavior |
|----------|----------|
| `GET /healthz` | `{"status":"ok"}` always. |
| `GET /readyz` | 200 / 503 from `loaders.readiness()`. `ready` == the semantic-model file exists (`config.SEMANTIC_MODEL_FILE`) — i.e. at least one ingestion has run. `_STATE` also carries `engine_warm`, `nl_summary_model_warm`. Warm-load happens once per worker in `hydrate()` (lifespan): sm-file check, retrieval engine, BGE-M3 dense + sparse, cross-encoder reranker, SLM `prewarm`, NL-summary SLM — each best-effort, `[warmup]` to stdout. |
| **`/metrics`** | **Does not exist** on the inference tier. Its per-query observability is the explain-trace path above. |

---

## 6. Related

- [`archive/mlflow_impl.md`](archive/mlflow_impl.md) — the original requirements spec
  (aspirational-enterprise; the implementation delivers the honest zero-touch subset).
- `mlflow_observability/README.md` — the living implementation writeup.
- [`EVALUATION.md`](EVALUATION.md) — the eval harnesses that feed `VEDA-Golden-Eval`.
