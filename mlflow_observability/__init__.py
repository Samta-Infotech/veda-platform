"""VEDA MLflow Observability — a fully decoupled exporter (mlflow_impl.md).

Nothing in veda_core / apps / inference imports this package, and this package
never imports the engine. The engine already persists one JSON line per query
to logs/explain_trace.jsonl (veda_core/veda/explain.py — EXPLAIN_TRACE_ENABLED /
_VERBOSE / _PERSIST, all default-on in veda_core/config.py). This package tails
that file and turns every record into one MLflow run: params, per-layer metrics
and artifacts named per mlflow_impl.md.

Failure isolation is structural: the exporter is a separate process. If MLflow
is down, misconfigured, or not installed, inference is untouched.
"""

__version__ = "1.0.0"
