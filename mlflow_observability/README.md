# VEDA MLflow Observability

Implementation of [`mlflow_impl.md`](../mlflow_impl.md): every user query becomes
**one MLflow run** with per-layer latencies, decisions, confidences and artifacts —
without touching a single line of the existing codebase.

## How it works (zero-touch by design)

The engine **already** records a structured per-query trace: `veda_core/veda/explain.py`
appends one JSON line per query to `logs/explain_trace.jsonl` (relative to the engine
cwd → `veda_core/logs/explain_trace.jsonl`). The three switches that control it are
already on by default in `veda_core/config.py`:

```
EXPLAIN_TRACE_ENABLED = True    # collect decisions/confidences per layer
EXPLAIN_TRACE_VERBOSE = True    # include full sections + candidate lists ("full")
EXPLAIN_TRACE_PERSIST = True    # append to logs/explain_trace.jsonl
```

This package is a **separate process** that tails that file and logs each record to
MLflow. Nothing in `veda_core/`, `apps/`, `inference/` or `chatbot/` imports it, and
it never imports the engine — observability is additive, and an MLflow outage can
never break inference (spec's "Design Principles" hold structurally, not by care).

```
query → pipeline (unchanged) ──append──▶ logs/explain_trace.jsonl
                                              │ tail (checkpointed)
                                              ▼
                              mlflow_observability exporter ──▶ MLflow server / sqlite
                                                                     │
                                                                     ▼
                                                                 MLflow UI
```

## What lands in MLflow per query run

| Kind | Examples |
|---|---|
| **Params** | `query`, `route`, `intent`, `action`, `table`, `anchor`, `status`, `refusal`, per-section strings (`schema_linking.selected_table`, …), `retrieval.top1_column`, `retrieval.top_columns`, `columns.used_in_sql`, `rerank.top1_before/after` |
| **Metrics** | `total_latency_ms`, `<section>.duration_ms` + `<section>.start_offset_ms` for all 9 layers, `routing_confidence`, `join_confidence`, `answer_confidence`, `pipeline_success`, `retrieval_candidate_tables/_columns`, `graph_columns_added`, `validation_checks_passed/failed`, `repair_count`, `sql_length`, `sql_join_count`, `limit_present` |
| **Signal scores** | `retrieval.top1_score` / `retrieval.score_mean` / `retrieval.score_top1_vs_top2_gap` — same trio for every spec Layer-2 signal found on a candidate (`semantic_score`, `bm25_score`, `graph_score`, `fk_score`, `value_score`, `rrf_score`, `cross_encoder_score`, `final_score`); `routing.top1_signal_<name>` per anchor-routing signal; `reranker_changed_top1` when before/after lists exist |
| **Column selection** | `columns.candidate_count`, `columns.used_in_sql_count`, `columns.selection_ratio` — which retrieved/graph-added columns the final SQL actually used |
| **Tags** | `veda.route`, `veda.status`, `veda.intent`, `veda.table`, `veda.query_hash`, `veda.environment`, `veda.line_fingerprint` (dedupe/audit) |
| **Artifacts** | `layers/query_understanding.json`, `layers/retrieval_candidates.json`, `layers/signal_scores.json` (per-candidate score breakdown + present/missing signals), `layers/selected_columns.json` (candidates → graph-added → selected table → columns used in SQL), `layers/graph_expansion.json`, `layers/routing.json`, `layers/validation.json`, `layers/final_response.json`, `sql/generated_sql.sql`, `trace/full_trace.json`, `trace/why.txt`, `coverage.json` |

A generic sweep also promotes **every** scalar the pipeline recorded in any trace
section (numeric → metric `<section>.<key>`, string → param), so new datapoints added
to the trace appear in MLflow automatically — no exporter change needed.

`coverage.json` (per run) lists the spec datapoints the engine does not emit yet
(tokens, cost, tenant/session identity, memory/summary/visualization stats — see
"Known gaps" below).

## Run it locally (Windows/macOS/Linux, no Docker)

> **Python 3.11–3.13 required for the UI.** On Python 3.14 the exporter works but
> `mlflow ui` crash-loops (mlflow imports `importlib.abc.Traversable`, removed in
> 3.14). On Windows use `py -3.11` explicitly — plain `python` may be 3.14.

```bash
cd veda-platform
py -3.11 -m venv .venv-mlflow && .venv-mlflow\Scripts\activate  # or source .venv-mlflow/bin/activate
pip install -r mlflow_observability/requirements.txt

# 0) prove the install works end-to-end (throwaway store, no config needed)
python -m mlflow_observability selftest

# 0b) optional: seed 4 sample runs into the real local store to explore the UI
#     before the engine has traced anything (tagged veda.demo=true)
python -m mlflow_observability demo

# 1) export everything the engine has traced so far, then keep following
python -m mlflow_observability watch          # Ctrl-C to stop (or `export` for one pass)

# 2) in a second terminal — the UI
python -m mlflow_observability ui             # → http://127.0.0.1:5001
```

Local defaults: sqlite store + artifacts + checkpoint under `mlflow_observability/mlflow_data/`
(git-ignored), trace log at `veda_core/logs/explain_trace.jsonl`. `python -m
mlflow_observability status` prints the effective config.

## Run it in production (Docker)

A dedicated override file — existing compose files are untouched:

```bash
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.mlflow.yml"
$COMPOSE up -d mlflow mlflow-exporter
# UI → http://<host>:5001   (MLFLOW_UI_PORT to change; default avoids macOS AirPlay on 5000)
```

* `mlflow` — tracking server + UI, store/artifacts on the `mlflow_data` volume.
  Point `MLFLOW_BACKEND_URI` at Postgres for a HA store; front the UI with nginx
  instead of publishing the port if it must not be host-exposed.
* `mlflow-exporter` — runs `python -m mlflow_observability watch` with the repo
  mounted **read-only**; checkpoint lives on its own `mlflow_state` volume, so
  restarts resume exactly where they left off and never duplicate runs.
* Works with the query-only demo too: add `-f docker-compose.demo.yml` before
  `-f docker-compose.mlflow.yml`.

## Env reference

| Variable | Default | Meaning |
|---|---|---|
| `VEDA_TRACE_LOG` | `<repo>/veda_core/logs/explain_trace.jsonl` | trace file to tail |
| `VEDA_MLFLOW_TRACKING_URI` | local sqlite (or `MLFLOW_TRACKING_URI`) | MLflow backend |
| `VEDA_MLFLOW_EXPERIMENT` | `VEDA-Query-Observability` | experiment name |
| `VEDA_MLFLOW_CHECKPOINT` | `mlflow_data/exporter_checkpoint.json` | tail position |
| `VEDA_MLFLOW_POLL_SECS` | `5` | watch-mode poll interval |
| `VEDA_ENVIRONMENT` | `local` | tagged onto every run (`local` / `production` / …) |
| `VEDA_MLFLOW_DATA_HOME` | `mlflow_observability/mlflow_data` | local store home |

## Useful UI views (spec "Layer Contribution Dashboard")

In the experiment's table view, add columns / use the chart view over:

* **Pipeline latency** — `total_latency_ms`, and `<section>.duration_ms` per layer
  (which layer consumes the most latency).
* **Routing effectiveness** — `routing_confidence` vs `pipeline_success`; filter
  `tags.veda.status = "no_table"` for routing failures.
* **Retrieval contribution** — `retrieval_candidate_columns`, `graph_columns_added`,
  `graph_expansion_used`.
* **Validation** — `validation_checks_failed`, `repair_count`; per-run details in
  `layers/validation.json`.
* Filter any slice with tag queries, e.g. `tags.veda.route = 'sql' and
  metrics.total_latency_ms > 5000`.

## Robustness notes

* **Checkpointed tailing** — byte offset + file-signature; log rotation/truncation
  resets cleanly, a torn (partially-written) last line is deferred to the next pass.
* **Idempotence** — the checkpoint advances per record; each run carries
  `veda.line_fingerprint` for audit/dedupe.
* **Failure isolation** — malformed lines are skipped and counted; MLflow/network
  errors stop the pass *before* advancing the checkpoint, so records are retried,
  never lost. `watch` reconnects if the tracking server restarts.

## Known gaps vs mlflow_impl.md (and the deliberate reason)

The spec asks for token counts, cost, tenant/session identity, and memory/summary/
visualization metrics. The engine's trace does not emit these today, and capturing
them would require touching pipeline code (e.g. a `usage` return in
`slm/_call_slm.py`, `request_id`/tenant stamped into the trace) — explicitly out of
scope for this zero-touch framework.

Per-signal retrieval scores ARE emitted (three additive engine edits, 2026-07-15):
`retrieval_engine_phase3.py::_results_from_tuples` stamps the per-signal fields
onto each `RetrievalResult`, and `veda/pipeline.py` writes every retrieved
candidate (no top-15 cap) into `retrieval.top_columns` with the spec's key names
(`semantic_score`, `bm25_score`, `graph_score`, `fk_score`, `value_score`,
`rrf_score`, `final_score`, plus `cross_encoder_score` and
`top_before_rerank`/`top_after_rerank` when the primary rerank ran). Caveats:
cache-hit retrievals return tuples without signal data, so their signals read
0.0; rerank keys are absent when the rerank was skipped (that absence is itself
the signal); every run's `coverage.json` / `layers/signal_scores.json` still
reports present-vs-missing per run. `python -m mlflow_observability demo` seeds
runs with ALL signals populated so the end-state is explorable without the engine. The schema is designed for them now (per the
spec's "Future Evaluation Support"): they are enumerated in every run's
`coverage.json`, and the generic sweep means that the moment the trace starts
carrying them, they appear in MLflow with **zero exporter changes**.
