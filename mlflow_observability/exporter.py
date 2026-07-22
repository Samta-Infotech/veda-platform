"""Checkpointed JSONL → MLflow exporter.

Reads new lines from the engine's explain-trace log and logs one MLflow run
per query record. Designed to be boring and unkillable:

  * byte-offset checkpoint (JSON file) — restarts never re-export old lines;
  * file-signature check — a rotated/truncated log resets the offset safely;
  * a partial (unterminated) last line is left for the next pass;
  * a malformed line is skipped and counted, never fatal;
  * MLflow/network errors stop the current pass and retry from the same
    checkpoint next cycle — records are not lost, and the engine is never
    touched (this is a separate process by design).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .mapper import map_record
from .settings import Settings

logger = logging.getLogger("veda.mlflow.exporter")

_SIG_BYTES = 4096  # cap on how much of the FIRST LINE is hashed


@dataclass
class ExportStats:
    exported: int = 0
    skipped_malformed: int = 0
    failed: int = 0


# ── checkpoint ────────────────────────────────────────────────────────────────

def _file_signature(path: Path) -> str:
    """Rotation/truncation detector. Hash only the FIRST LINE (capped): appends
    never change it, while a rotated or rewritten log almost certainly does.
    (Hashing a fixed byte window is NOT append-stable while the file is still
    shorter than the window.)"""
    with open(path, "rb") as f:
        first_line = f.readline(_SIG_BYTES)
    return hashlib.sha1(first_line).hexdigest()


def _load_checkpoint(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_checkpoint(path: Path, ck: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ck, f)
    tmp.replace(path)


# ── MLflow sink ───────────────────────────────────────────────────────────────

class MlflowSink:
    """Thin wrapper that owns the tracking connection. Imported lazily so the
    mapper stays testable without mlflow installed."""

    def __init__(self, settings: Settings):
        import mlflow  # deferred import on purpose
        self._mlflow = mlflow
        self.settings = settings
        mlflow.set_tracking_uri(settings.tracking_uri)
        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name(settings.experiment)
        if exp is None:
            kwargs = {}
            if settings.artifact_location:
                kwargs["artifact_location"] = settings.artifact_location
            exp_id = client.create_experiment(settings.experiment, **kwargs)
        else:
            exp_id = exp.experiment_id
        self.experiment_id = exp_id

    def log(self, spec) -> str:
        mlflow = self._mlflow
        with mlflow.start_run(experiment_id=self.experiment_id,
                              run_name=spec.run_name) as run:
            if spec.params:
                mlflow.log_params(spec.params)
            if spec.metrics:
                mlflow.log_metrics(spec.metrics)
            if spec.tags:
                mlflow.set_tags(spec.tags)
            for artifact_path, content in spec.artifacts.items():
                mlflow.log_text(content, artifact_path)
            if getattr(spec, "spans", None):
                try:
                    self._emit_spans(spec, run.info.run_id)
                except Exception:
                    logger.warning("span emit skipped for run %s (metrics/params/tags "
                                   "still logged)", run.info.run_id, exc_info=True)
            return run.info.run_id

    def _emit_spans(self, spec, run_id: str) -> None:
        """Replay the pipeline-stage timings as an MLflow trace (span waterfall) so
        the Traces UI shows retrieval → SQL-gen → validation → summary as a tree.
        Uses the low-level tracing client with explicit ns timestamps (historical
        replay); the trace's relative offsets/durations give the correct waterfall
        shape — the absolute base time is arbitrary. Best-effort by contract: any
        failure here is swallowed by the caller so run logging is never affected."""
        import time as _time
        mlflow = self._mlflow
        client = mlflow.MlflowClient()
        spans = spec.spans
        base = _time.time_ns()
        total_ms = max((s["start_offset_ms"] + s["duration_ms"]) for s in spans)
        # correlate the trace back to its run + carry a few slice tags (custom
        # veda.* only — never reserved mlflow.* keys).
        tags = {k: v for k, v in (spec.tags or {}).items()
                if k in ("veda.query_hash", "veda.route", "veda.status",
                         "veda.outcome", "veda.git_sha")}
        tags["veda.run_id"] = run_id
        root = client.start_trace(name=(spec.run_name or "query")[:250],
                                  span_type="AGENT", experiment_id=self.experiment_id,
                                  tags=tags, start_time_ns=base)
        rid = root.request_id
        try:
            for s in spans:
                st = base + int(s["start_offset_ms"] * 1_000_000)
                child = client.start_span(
                    name=s["name"], request_id=rid, parent_id=root.span_id,
                    span_type=s["span_type"], start_time_ns=st,
                    attributes={"duration_ms": s["duration_ms"],
                                "start_offset_ms": s["start_offset_ms"]})
                client.end_span(rid, child.span_id,
                                end_time_ns=st + int(s["duration_ms"] * 1_000_000))
        finally:
            client.end_trace(rid, end_time_ns=base + int(total_ms * 1_000_000))


# ── export passes ─────────────────────────────────────────────────────────────

def export_once(settings: Settings, sink: Optional[MlflowSink] = None,
                from_start: bool = False) -> ExportStats:
    """One pass: read every complete new line past the checkpoint, log each to
    MLflow, advance the checkpoint after each successful record."""
    stats = ExportStats()
    log_path = settings.trace_log
    if not log_path.exists():
        logger.info("trace log not found yet: %s", log_path)
        return stats

    ck = {} if from_start else _load_checkpoint(settings.checkpoint_path)
    sig = _file_signature(log_path)
    offset = int(ck.get("offset", 0))
    if ck.get("signature") != sig or offset > log_path.stat().st_size:
        offset = 0  # new/rotated/truncated file — start over

    if sink is None:
        sink = MlflowSink(settings)

    with open(log_path, "rb") as f:
        f.seek(offset)
        while True:
            line_start = f.tell()
            raw = f.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                # partial write in progress — leave it for the next pass
                f.seek(line_start)
                break
            text = raw.decode("utf-8", "replace").strip()
            consumed = f.tell()
            if not text:
                offset = consumed
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                stats.skipped_malformed += 1
                logger.warning("skipping malformed trace line at offset %d", line_start)
                offset = consumed
                continue
            try:
                spec = map_record(record, raw_line=text,
                                  environment=settings.environment,
                                  param_value_max=settings.param_value_max)
                run_id = sink.log(spec)
                stats.exported += 1
                logger.info("exported run %s — %s", run_id, spec.run_name)
            except Exception:
                # MLflow/server hiccup: keep the checkpoint BEFORE this line so
                # the record is retried next cycle, and end this pass.
                stats.failed += 1
                logger.exception("failed to export record at offset %d — will retry", line_start)
                break
            offset = consumed
            _save_checkpoint(settings.checkpoint_path,
                             {"signature": sig, "offset": offset,
                              "trace_log": str(log_path)})

    _save_checkpoint(settings.checkpoint_path,
                     {"signature": sig, "offset": offset, "trace_log": str(log_path)})
    return stats


def watch(settings: Settings) -> None:
    """Follow the trace log forever (production sidecar mode)."""
    logger.info("watching %s → %s (experiment %r, poll %.1fs)",
                settings.trace_log, settings.tracking_uri,
                settings.experiment, settings.poll_seconds)
    sink: Optional[MlflowSink] = None
    while True:
        try:
            if sink is None:
                sink = MlflowSink(settings)
            stats = export_once(settings, sink=sink)
            if stats.exported:
                logger.info("pass complete: %d exported, %d malformed, %d failed",
                            stats.exported, stats.skipped_malformed, stats.failed)
        except KeyboardInterrupt:
            raise
        except Exception:
            sink = None  # tracking server may have restarted — reconnect next loop
            logger.exception("export pass crashed — retrying in %.1fs", settings.poll_seconds)
        time.sleep(settings.poll_seconds)
