# EVALUATION — golden sets, harnesses, CI gates

Expands [ARCHITECTURE.md](ARCHITECTURE.md). Written from a read of `apps/evaluation/`,
`evaluation/`, `scripts/`, `veda_core/doc_bench.py`, `veda_core/datalake_bench.py`,
`mlflow_observability/evaluate.py` on 2026-09-09.

There are **four** distinct evaluation surfaces. They do not share a runner.

---

## 1. `apps/evaluation` — tracked Celery runs

`POST /api/v1/admin/eval` (`EvalTriggerView`, staff-only) →
`task_run_eval.delay(source_id, tenant, label, queries=None)` (queue `default`).

- Runs a query set (default: 4 flow queries `D01 / D02 / A01 / S01`, or a caller-supplied
  `queries` list) through `InferenceClient.run_hybrid_query` — the same HTTP path
  `/api/v1/query` uses.
- Scores `status == "ok"` as success.
- Writes an `EvalRun` (`recall_at_k`, `hit_rate`, `sql_success_rate`, `report_html`) +
  per-query `EvalCaseResult` (`recall`, `hit`, `status`, `details` JSON) + an HTML report
  in `EvalRun.report_html`. **Every interpolated value is HTML-escaped** (deliberate
  stored-XSS fix — query text is caller-supplied).
- Unreachable inference → an `exec_error` case; the run still completes.
- Visible in Django admin (`EvalRunAdmin` + `EvalCaseResultInline`) and via the API.

Files: `apps/evaluation/{models,tasks,admin,apps}.py`, `migrations/0001_initial.py`.

---

## 2. `evaluation/` — file-based harnesses (run manually)

Run in the api/inference container or on the host. Result files are git-ignored.

| File | Role |
|------|------|
| `nl_query_suite.py` | The main answerability battery — a fixed 50-query set (`q01`–`q50`), POSTs `/api/v1/query`, classifies PASS / REFUSED / EMPTY / ERROR / TIMEOUT + advisory structural checks (`top N` → ≤ N rows) + **`GOLDEN_ANCHORS` wrong-table detection** (`GOLDEN-FAIL`). Flags: `--resume`, `--only`, `--tag`, `--recheck` (offline re-apply anchors). Writes `nl_query_suite[_tag]_results.jsonl` + `_report.md`. |
| `ci_checks.sh [results.jsonl]` | The **Phase D gate**, chained: (1) `tests/test_qsr_resolution.py`, (2) `determinism_check.py`, (3) `nl_query_suite.py --recheck` (enforce 0 `GOLDEN-FAIL`), (4) `latency_assert.py`. Exits non-zero on any failure. |
| `determinism_check.py` | Two-seed (`PYTHONHASHSEED=0` vs `1`) subprocess diff of `score_anchors` ranked-anchor decisions over a fixed 10-query battery. Exit 1 on any divergence. Host-runnable. |
| `latency_assert.py <results.jsonl>` | Ratchet SLO gate: enforces fast-lane p50 < 5 s and clarify p95 < 15 s; **reports** (does not enforce) overall p50/p95 vs the 5 s / 30 s / 60 s target. |
| `calibration_report.py <results.jsonl> [out.md]` | Tabulates anchor confidence + top-2 margin + decision `source` vs verdict/golden, per bucket. Data for fitting thresholds instead of hand-picking. |
| `README.md` | The WP0 retrieval-eval workflow doc — **note its `golden_queries.jsonl` and `retrieval_BASELINE.json` are NOT in the repo** (generated per-substrate, git-ignored). |

### Golden sets & query suites

| File | Contents |
|------|----------|
| `golden_homzhub.jsonl` | `{id, query, gold_tables}` for the homzhub source (source 2). |
| `golden_cross_source.jsonl` | Cross-source golden set (Phase 6.1) with a `#`-comment header describing sources 2/3/4/5 + ground-truth joins. |
| `cross_source_entity_labels.jsonl` | Hand-labeled entity gold (29 admitted nodes, `is_bridge`) for entity-linking precision/recall (Phase 6.2). |
| `retrieval_benchmark.json` | 182-query labeled benchmark (`id`, `query`, `category`, `difficulty`, `expected_tables`, `expected_columns`). |
| `suite_{simple,aggregate,filter,grouped,ranking,temporal}.json` | Per-category NL query suites generated from the live homzhub schema (data-validated), with `op` + `expect` (table shape / viz kind). Component-wise judging (table + viz + summary). |
| `results/` (git-ignored) | `.gitkeep` + `cross_source_nogit.json` (an entity-linking precision/recall 1.0, 13/13 bridges + federated-e2e snapshot). |
| `benchmark_archive/2026-07-30_denseON_live/` | The **frozen reference run** — `INDEX.md` + 13 result files. Headline: retrieval fixed (R@1 0.79 / R@5 0.91 / MRR 0.90) but **31 real-client gold: 0/31 answer-level correct** — the wall is downstream (join planner, ratio/%, grain, ~40% semantic entity coverage). |

---

## 3. `scripts/` — WP0 retrieval eval (deterministic, no SLM)

| Script | Role |
|--------|------|
| `build_golden_set.py --source-id --tenant [--min 60]` | Builds `evaluation/golden_queries.jsonl` from `VerifiedQueryCache` (sqlglot parse of `verified_sql` → gold columns/tables) + a `parity_suite.QUERIES` seed. |
| `retrieval_eval.py --source-id --tenant [--label]` | Runs `query/retrieval_select.select_retrieval` over the golden set → recall@5 / @15, MRR, table_recall@3, candidate-set size, per-stage wall-clock → `evaluation/results/retrieval_<sha>.json`, graded vs `retrieval_BASELINE.json`. Deterministic — the WP0 gate. |
| `parity_suite.py` | Phase 7.1: runs the front door twice per query (legacy on-disk `sm` + FK store vs migrated Redis `sm` + `storage_adapters`), diffs status/route/SQL/rows. |
| `tune_fusion_weights.py --source-id --tenant` | WP6: random search + local refine over the 6 `FUSION_WEIGHTS`, scored by recall@10. **Never writes config** — prints the dict for a human to paste. |
| `hnsw_parity_sweep.py` | Phase 7.1a: sweeps HNSW `ef_search` until recall@k = 1.0 vs exact cosine; reports the lowest passing value (→ pinned `VEDA_HNSW_EF_SEARCH`). |

**`golden_queries.jsonl` and `retrieval_BASELINE.json` are not committed** — they are
built against a live substrate and git-ignored (`evaluation/results/` is ignored).
Regenerate per environment.

---

## 4. `mlflow_observability evaluate`

Golden set → one aggregate MLflow run in `VEDA-Golden-Eval`; CI gate via
`--min-pass-rate` (exit 2). See [`OBSERVABILITY.md`](OBSERVABILITY.md) §3.

---

## 5. `veda_core/doc_bench.py` / `datalake_bench.py`

Run in-container (`-w /app/veda_core`):

- `doc_bench.py` — 57 grounded filesystem/document (source 3) queries through the main
  routed pipeline → `doc_bench_results.json` (repo root).
- `datalake_bench.py` — 54 grounded datalake (sources 4 csv + 5 parquet) queries with
  full routing → `datalake_bench_results.json` (repo root).

> **⚠ Fragile.** Both build source profiles **by hand** via `set_source_profiles({...})`,
> passing connector-type strings (`csv_lake`, `parquet`) as `source_type` rather than the
> routing *kind*. `docs/MULTI_SOURCE_DEPLOYMENT.md` §1 documents that this exact pattern
> invalidated three earlier benchmark runs. The committed result JSONs show mostly
> `refused` statuses and very high latencies (52 s, 134 s). Treat them with suspicion
> until the profiles are built the same way the live path builds them.

---

## 6. CI wiring

- `scripts/lint_no_raw_offload.sh` — bans bare `run_in_threadpool(` / `ThreadPoolExecutor(`
  without context-carrying, over `inference/` + `veda_core/`. Non-zero exit fails the build.
- `evaluation/ci_checks.sh` — the Phase D gate (§2).
- `mlflow_observability evaluate --min-pass-rate` — a golden-eval gate.
- `tests/` — pytest; notable suites: `test_execution_state_reuse.py` (14),
  `test_summary_analytics.py`, `test_visualization_matrix.py` (21),
  `test_chat_visualization.py` (24), `test_chatbot_classify.py` (16),
  `test_rbac_filter.py`, `test_inference_retrieve_rbac.py`, `test_admin_bootstrap.py`
  (Postgres-only — the only place `select_for_update` is lock-proven).

---

## 7. Related

- [`RETRIEVAL.md`](RETRIEVAL.md) — what the retrieval-eval harness measures.
- [`QUERY_ENGINE.md`](QUERY_ENGINE.md) — the terminal statuses the answerability suite classifies.
- [`OBSERVABILITY.md`](OBSERVABILITY.md) — the MLflow eval experiment.
- `evaluation/README.md` — the WP0 workflow (verify script names first).
