"""CLI — python -m mlflow_observability <command>

Commands
  export     one pass: export new trace records since the checkpoint, then exit
  watch      follow the trace log forever (production sidecar mode)
  ui         launch the MLflow UI against the configured backing store
  status     show config, checkpoint position and trace-log stats
  selftest   end-to-end smoke test into a throwaway store (no config needed)
  demo       seed sample query runs into the CONFIGURED store (UI walkthrough
             without running the engine); they are tagged veda.demo=true
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from .settings import load


def _cmd_export(args) -> int:
    from .exporter import export_once
    s = load()
    stats = export_once(s, from_start=args.from_start)
    print(f"exported={stats.exported} malformed={stats.skipped_malformed} failed={stats.failed}")
    return 1 if stats.failed else 0


def _cmd_watch(args) -> int:
    from .exporter import watch
    watch(load())
    return 0


def _cmd_ui(args) -> int:
    # mlflow's UI server (as of 3.x) imports importlib.abc.Traversable, removed
    # in Python 3.14 — the workers crash-loop. Exporting works; serving doesn't.
    if sys.version_info >= (3, 14):
        print(f"Python {sys.version.split()[0]} cannot run `mlflow ui` "
              "(importlib.abc.Traversable was removed in 3.14).\n"
              "Recreate the venv with Python 3.11-3.13, e.g.:\n"
              "  deactivate\n"
              "  rmdir /s /q .venv-mlflow   (PowerShell: Remove-Item -Recurse -Force .venv-mlflow)\n"
              "  py -3.11 -m venv .venv-mlflow\n"
              "  .venv-mlflow\\Scripts\\activate\n"
              "  pip install -r mlflow_observability\\requirements.txt")
        return 1
    s = load()
    if s.tracking_uri.startswith(("http://", "https://")):
        print(f"Tracking server is remote - open it directly: {s.tracking_uri}")
        return 0
    cmd = [sys.executable, "-m", "mlflow", "ui",
           "--backend-store-uri", s.tracking_uri,
           "--port", str(args.port), "--host", args.host]
    if s.artifact_location:
        cmd += ["--default-artifact-root", s.artifact_location]
    print("launching:", " ".join(cmd))
    print(f"open http://{args.host}:{args.port}  (experiment: {s.experiment})")
    return subprocess.call(cmd)


def _cmd_status(args) -> int:
    s = load()
    ck = {}
    if s.checkpoint_path.exists():
        ck = json.loads(s.checkpoint_path.read_text(encoding="utf-8"))
    log_exists = s.trace_log.exists()
    print(json.dumps({
        "trace_log": str(s.trace_log),
        "trace_log_exists": log_exists,
        "trace_log_bytes": s.trace_log.stat().st_size if log_exists else 0,
        "tracking_uri": s.tracking_uri,
        "experiment": s.experiment,
        "environment": s.environment,
        "checkpoint": ck or None,
    }, indent=2))
    return 0


def _sample_records():
    """Realistic trace records shaped exactly like ExplainTrace.finish() output."""
    base = {
        "route": "sql", "intent": "COUNT", "table": "incident",
        "anchor": "incident", "anchor_conf": 0.93, "join_conf": None,
        "action": "single_table", "status": "answered", "confidence": 0.9,
        "refusal": None, "total_ms": 1834.2,
        "full": {
            "query": "how many incidents are escalated", "total_ms": 1834.2,
            "verbose": True,
            "sections": {
                "query_understanding": {"_ms": 0.2, "intent": "COUNT",
                                        "temporal": None, "existence": None},
                "retrieval": {"_ms": 41.7,
                              "candidate_tables": ["incident", "workflow_state"],
                              "n_columns": 34},
                "schema_linking": {"_ms": 512.0, "selected_table": "incident"},
                "anchor_selection": {"_ms": 604.9, "anchor": "incident",
                                     "confidence": 0.93, "margin": 0.41},
                "sql_planning": {"_ms": 918.4, "action": "single_table",
                                 "table": "incident"},
                "validation": {"_ms": 1633.0,
                               "checks": [{"name": "ast_readonly_parameterized_fanout",
                                           "status": "pass"}],
                               "repairs": []},
                "output": {"_ms": 1800.1, "status": "answered",
                           "sql": "SELECT COUNT(*) FROM incident WHERE state = %s",
                           "params": ["escalated"], "confidence": 0.9},
            },
        },
    }
    agg = json.loads(json.dumps(base))
    agg["intent"] = "AGGREGATE"
    agg["total_ms"] = agg["full"]["total_ms"] = 3210.5
    agg["full"]["query"] = "total rent collected last month"
    agg["full"]["sections"]["query_understanding"]["temporal"] = {
        "start": "2026-06-01", "end": "2026-06-30"}

    slow = json.loads(json.dumps(base))
    slow["total_ms"] = slow["full"]["total_ms"] = 7422.9
    slow["anchor_conf"] = 0.61
    slow["full"]["query"] = "average resolution time by assignee this quarter"
    slow["full"]["sections"]["anchor_selection"]["confidence"] = 0.61

    refused = json.loads(json.dumps(base))
    refused.update(status="no_table", confidence=None, table=None,
                   refusal="couldn't identify the table", total_ms=422.0)
    refused["full"]["query"] = "show me the flurbs"
    refused["full"]["total_ms"] = 422.0
    refused["full"]["sections"]["output"] = {
        "_ms": 401.0, "status": "no_table",
        "refusal": "couldn't identify the table"}
    return [base, agg, slow, refused]


def _cmd_demo(args) -> int:
    """Seed sample runs into the configured store so the UI can be explored
    before the engine has traced any real queries."""
    from .exporter import MlflowSink
    from .mapper import map_record

    s = load()
    sink = MlflowSink(s)
    for rec in _sample_records():
        spec = map_record(rec, raw_line=json.dumps(rec),
                          environment=s.environment,
                          param_value_max=s.param_value_max)
        spec.tags["veda.demo"] = "true"
        run_id = sink.log(spec)
        print(f"seeded demo run {run_id}: {spec.run_name}")
    print(f"done - open the UI (python -m mlflow_observability ui), "
          f"experiment {s.experiment!r}. Delete demo runs any time by "
          f"filtering tags.veda.demo = 'true'.")
    return 0


def _cmd_selftest(args) -> int:
    """Write two synthetic trace records to a temp file, export them into a
    temp sqlite store, and verify both runs landed with metrics + artifacts."""
    import tempfile

    from .exporter import export_once
    from .settings import Settings

    sample = {
        "route": "sql", "intent": "COUNT", "table": "incident",
        "anchor": "incident", "anchor_conf": 0.93, "join_conf": None,
        "action": "single_table", "status": "answered", "confidence": 0.9,
        "refusal": None, "total_ms": 1834.2,
        "full": {
            "query": "how many incidents are escalated", "total_ms": 1834.2,
            "verbose": True,
            "sections": {
                "query_understanding": {"_ms": 0.2, "intent": "COUNT",
                                        "temporal": None, "existence": None},
                "retrieval": {"_ms": 41.7, "candidate_tables": ["incident", "workflow_state"],
                              "n_columns": 34},
                "schema_linking": {"_ms": 512.0, "selected_table": "incident"},
                "anchor_selection": {"_ms": 604.9, "anchor": "incident",
                                     "confidence": 0.93, "margin": 0.41},
                "sql_planning": {"_ms": 918.4, "action": "single_table",
                                 "table": "incident"},
                "validation": {"_ms": 1633.0,
                               "checks": [{"name": "ast_readonly_parameterized_fanout",
                                           "status": "pass"}],
                               "repairs": []},
                "output": {"_ms": 1800.1, "status": "answered",
                           "sql": "SELECT COUNT(*) FROM incident WHERE state = %s",
                           "params": ["escalated"], "confidence": 0.9},
            },
        },
    }
    refused = json.loads(json.dumps(sample))
    refused.update(status="no_table", confidence=None, refusal="couldn't identify the table",
                   table=None, total_ms=422.0)
    refused["full"]["sections"]["output"] = {"_ms": 401.0, "status": "no_table",
                                             "refusal": "couldn't identify the table"}

    # ignore_cleanup_errors: on Windows the sqlite handle can outlive the test
    with tempfile.TemporaryDirectory(prefix="veda_mlflow_selftest_",
                                     ignore_cleanup_errors=True) as td:
        tdp = Path(td)
        trace = tdp / "explain_trace.jsonl"
        trace.write_text(json.dumps(sample) + "\n" + json.dumps(refused) + "\n",
                         encoding="utf-8")
        s = Settings(
            trace_log=trace,
            tracking_uri="sqlite:///" + (tdp / "mlflow.db").as_posix(),
            experiment="VEDA-Selftest",
            artifact_location=(tdp / "artifacts").as_uri(),
            checkpoint_path=tdp / "ck.json",
            environment="selftest",
        )
        stats = export_once(s)
        assert stats.exported == 2 and stats.failed == 0, stats

        # partial-line + checkpoint behaviour: append one full and one torn line
        with open(trace, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample) + "\n")
            f.write('{"route": "sql", "intent": "COU')  # no newline — in-flight
        stats2 = export_once(s)
        assert stats2.exported == 1 and stats2.failed == 0, stats2

        import mlflow
        mlflow.set_tracking_uri(s.tracking_uri)
        runs = mlflow.search_runs(experiment_names=[s.experiment])
        assert len(runs) == 3, f"expected 3 runs, got {len(runs)}"
        row = runs.iloc[-1]
        assert row["metrics.total_latency_ms"] > 0
        assert row["params.query"].startswith("how many")
        client = mlflow.MlflowClient()
        arts = {a.path for a in client.list_artifacts(row["run_id"])}
        assert "trace" in arts and "layers" in arts and "coverage.json" in arts, arts
    print("selftest OK - 3 runs exported (2 answered, 1 refused), "
          "metrics/params/artifacts verified, torn line correctly deferred")
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=os.environ.get("VEDA_MLFLOW_LOGLEVEL", "INFO"),
                        format="%(asctime)s | %(levelname)-5s | [%(name)s] %(message)s")
    ap = argparse.ArgumentParser(prog="python -m mlflow_observability",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export", help="one export pass, then exit")
    p.add_argument("--from-start", action="store_true",
                   help="ignore the checkpoint and re-read the whole trace log")
    p.set_defaults(fn=_cmd_export)

    p = sub.add_parser("watch", help="follow the trace log forever")
    p.set_defaults(fn=_cmd_watch)

    p = sub.add_parser("ui", help="launch the MLflow UI (local stores)")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", default="127.0.0.1")
    p.set_defaults(fn=_cmd_ui)

    p = sub.add_parser("status", help="show config + checkpoint")
    p.set_defaults(fn=_cmd_status)

    p = sub.add_parser("selftest", help="end-to-end smoke test (throwaway store)")
    p.set_defaults(fn=_cmd_selftest)

    p = sub.add_parser("demo", help="seed sample runs into the configured store")
    p.set_defaults(fn=_cmd_demo)

    args = ap.parse_args(argv)
    return args.fn(args)
