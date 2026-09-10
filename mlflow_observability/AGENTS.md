# mlflow_observability/ — the explain-trace → MLflow exporter

A **standalone process**. Nothing in the platform imports it; it never imports the engine.
An MLflow outage cannot affect inference (structural, not disciplined).
Full reference: [../docs/OBSERVABILITY.md](../docs/OBSERVABILITY.md).

| File | Role |
|------|------|
| `__init__.py` | Package doc + `__version__`. States the decoupling contract. |
| `__main__.py` | `sys.exit(main())` → `cli.main`. |
| `cli.py` | Subcommands: `export` (one pass), `watch` (sidecar loop), `ui` (refuses on Python ≥ 3.14), `status`, `evaluate` (golden set → aggregate run, `--min-pass-rate` CI gate exit 2), `selftest` (throwaway sqlite), `demo` (4 sample runs). |
| `settings.py` | Frozen env-driven `Settings`. Defaults: trace log `veda_core/logs/explain_trace.jsonl`, sqlite store under `mlflow_data/`, experiment `VEDA-Query-Observability`, poll 5 s. |
| `exporter.py` | Checkpointed JSONL tailer (byte offset + first-line SHA1 rotation detector, partial-line deferral, MLflow error stops **before** advancing the checkpoint). `watch()` reconnects. |
| `mapper.py` | Pure (no `mlflow` import) `map_record(record) -> RunSpec`. Spec-named metrics + a generic per-section sweep + Layer-2 signal-score flattening + `coverage.json`. `SPEC_GAPS` lists what the engine trace still doesn't emit. `SECTIONS` mirrors `veda_core/veda/explain.py._SECTIONS`. |
| `evaluate.py` | Golden-eval harness → one aggregate run to `VEDA-Golden-Eval`. Injected runner → model-free tests. |
| `README.md` | The living implementation writeup (the archived `mlflow_impl.md` was the spec). |
| `requirements.txt` | `mlflow>=2.14` only. |

Compose sidecar: `docker-compose.mlflow.yml` (`mlflow` server + `mlflow-exporter watch`).
**Not in the dev or prod compose** — opt-in only.

Tests: `tests/test_mlflow_mapper.py`, `tests/test_mlflow_evaluate.py` (model-free).
