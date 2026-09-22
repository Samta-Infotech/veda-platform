# =============================================================================
# veda/exec_records.py
# VEDA — per-source execution records (traceability Phase 1, Part 7).
#
# THE GAP THIS CLOSES
#   Before this, a multi-source answer had NO per-leg accounting. `AgentResult`
#   (query/agents.py) carried status/engine/error but was never traced, and
#   nothing anywhere recorded when a source started, how long it took, how many
#   rows it returned, whether it was retried, or whether a fallback ran. The only
#   execution numbers in the trace were whole-query row_count/column_count.
#
# TWO PROJECTIONS, ONE RECORD
#   `SourceExecutionRecord` is the INTERNAL record: it may hold the raw error.
#   `.as_safe_dict()` is the ONLY thing that may reach a user — it drops the raw
#   error and substitutes a generic, category-based sentence, because a psycopg2
#   / DuckDB message routinely embeds a host, a DSN, a schema name or a column
#   the caller has no access to.
#
# FLAG
#   SOURCE_EXECUTION_RECORDS_ENABLED (config.py, default False).
# =============================================================================

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

TRACE_SECTION = "source_execution"

# ── statuses ─────────────────────────────────────────────────────────────────
PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
PARTIAL = "partial"
FAILED = "failed"
REFUSED = "refused"
SKIPPED = "skipped"
STATUSES = (PENDING, RUNNING, COMPLETED, PARTIAL, FAILED, REFUSED, SKIPPED)

#: Terminal statuses that mean "this source contributed nothing to the answer".
NON_CONTRIBUTING = frozenset({FAILED, REFUSED, SKIPPED})

#: Status -> the sentence a user sees. Never mentions an engine, agent, driver or
#: host. `failed` deliberately does not distinguish "down" from "misconfigured" —
#: that difference is operational, and guessing it out loud would be misleading.
_SAFE_STATUS_MESSAGE = {
    PENDING: "Not started",
    RUNNING: "Running",
    COMPLETED: "Completed",
    PARTIAL: "Partly completed",
    FAILED: "This data source could not be reached",
    REFUSED: "This data source could not answer the question",
    SKIPPED: "Not needed for this question",
}

#: Internal engine name -> what the user is told the system was doing. The engine
#: names (deterministic_sql / rag / nosql) are implementation detail and must not
#: appear in a projection.
_ENGINE_ACTIVITY = {
    "deterministic_sql": "Querying structured data",
    "rag": "Searching relevant documents",
    "nosql": "Searching records",
    "federated": "Combining data across sources",
}


@dataclass
class SourceExecutionRecord:
    """One source's execution, from dispatch to terminal state."""

    source_id: str
    source_type: str = ""
    engine: str = ""                       # INTERNAL — never projected verbatim
    status: str = PENDING
    required: bool = True
    started_at_ms: Optional[int] = None
    completed_at_ms: Optional[int] = None
    duration_ms: Optional[float] = None
    db_execution_ms: Optional[float] = None
    rows_returned: Optional[int] = None
    retry_count: int = 0
    fallback_used: bool = False
    fallback_path: str = ""                # INTERNAL — never projected verbatim
    error: Optional[str] = None            # INTERNAL — never projected verbatim
    error_class: str = ""                  # "transient" | "permanent" | ""

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> "SourceExecutionRecord":
        self.status = RUNNING
        self.started_at_ms = int(time.time() * 1000)
        return self

    def finish(self, status: str, *, rows: Optional[int] = None,
               error: Optional[str] = None, error_class: str = "",
               db_execution_ms: Optional[float] = None) -> "SourceExecutionRecord":
        self.status = status if status in STATUSES else FAILED
        self.completed_at_ms = int(time.time() * 1000)
        if self.started_at_ms is not None:
            self.duration_ms = round(float(self.completed_at_ms - self.started_at_ms), 1)
        if rows is not None:
            self.rows_returned = int(rows)
        if error is not None:
            self.error = str(error)
        if error_class:
            self.error_class = error_class
        if db_execution_ms is not None:
            self.db_execution_ms = round(float(db_execution_ms), 1)
        return self

    # -- projections ---------------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        """The full INTERNAL record (raw error included). Trace/logs only."""
        return asdict(self)

    def as_safe_dict(self) -> Dict[str, Any]:
        """The user-facing projection.

        Drops `error`, `fallback_path` and `engine` entirely. `name`/`type` are
        resolved through veda/source_names.py, which itself falls back to a
        generic label for a source this request may not name.
        """
        from veda import source_names as sn
        out = {
            "id": str(self.source_id),
            "name": sn.display_name(self.source_id),
            "type": sn.display_type(self.source_id),
            "status": self.status,
            "message": _SAFE_STATUS_MESSAGE.get(self.status, "Completed"),
            "required": bool(self.required),
        }
        if self.duration_ms is not None:
            out["duration_ms"] = self.duration_ms
        if self.rows_returned is not None:
            out["rows_returned"] = self.rows_returned
        if self.retry_count:
            out["retried"] = True
        if self.fallback_used:
            out["fallback_used"] = True
        return out

    def activity_message(self) -> str:
        """What to show while this source is RUNNING — business-friendly, and
        preferring the source's own name when this request is allowed to use it."""
        from veda import source_names as sn
        if sn.is_known(self.source_id):
            return f"Retrieving data from {sn.display_name(self.source_id)}"
        return _ENGINE_ACTIVITY.get(self.engine, "Retrieving data")


# ── registry: the per-query collection ───────────────────────────────────────
def _enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "SOURCE_EXECUTION_RECORDS_ENABLED", False))
    except Exception:
        return False


class ExecutionRecorder:
    """Owns the per-source records for ONE query and mirrors each state change
    into the trace and the live timeline (the single-source-of-truth rule)."""

    enabled = True

    def __init__(self, trace=None, timeline=None):
        self._trace = trace
        self._timeline = timeline
        self.records: List[SourceExecutionRecord] = []

    def open(self, source_id, *, source_type="", engine="", required=True
             ) -> SourceExecutionRecord:
        """Create + register a record and mark it RUNNING, streaming the start."""
        rec = SourceExecutionRecord(source_id=str(source_id), source_type=source_type,
                                    engine=engine, required=required).start()
        self.records.append(rec)
        self._append(rec)
        tl = self._timeline
        if tl is not None and getattr(tl, "enabled", False):
            from veda import lifecycle as lc
            tl.started(lc.PHASE_DATA_RETRIEVAL, rec.activity_message(),
                       source_id=rec.source_id)
        return rec

    def close(self, rec: SourceExecutionRecord, status: str, **kw) -> SourceExecutionRecord:
        """Finalize a record and stream its terminal state."""
        rec.finish(status, **kw)
        self._sync(rec)
        tl = self._timeline
        if tl is not None and getattr(tl, "enabled", False):
            from veda import lifecycle as lc
            from veda import source_names as sn
            label = sn.display_name(rec.source_id) if sn.is_known(rec.source_id) else "A data source"
            if status == COMPLETED:
                tl.completed(lc.PHASE_DATA_RETRIEVAL, f"{label} completed",
                             source_id=rec.source_id)
            elif status in NON_CONTRIBUTING:
                tl.warning(lc.PHASE_DATA_RETRIEVAL,
                           f"{label}: {_SAFE_STATUS_MESSAGE.get(status, 'did not return data')}",
                           source_id=rec.source_id)
        return rec

    def has_records(self) -> bool:
        """Whether anything has been recorded for this query yet (EXP-B4: lets a
        gap-filler avoid double-counting a source a head already recorded)."""
        return bool(self.records)

    def _append(self, rec: "SourceExecutionRecord") -> None:
        """Add ONE newly-registered record to the trace, in place.

        `open()` used to call the full `_sync()`, which re-serialises every record —
        so registering n sources was O(n^2) even after `close()` was made targeted,
        and the scaling test caught it (4x the records still cost ~16x the time).
        """
        tr = self._trace
        if tr is None or not getattr(tr, "enabled", False):
            return
        try:
            sec = tr.sections.setdefault(TRACE_SECTION, {})
            sec.setdefault("_ms", 0.0)
            rows = sec.get("records")
            if not isinstance(rows, list):
                rows = sec["records"] = []
            rows.append(rec.as_dict())
            sec["count"] = len(self.records)
        except Exception:
            pass

    def _sync(self, rec: "SourceExecutionRecord | None" = None) -> None:
        """Mirror record state into the trace.

        EXP-B7: with `rec` given, only THAT record's entry is rewritten, in place.
        This used to re-serialise every record on every state change, which is
        O(total) per call and therefore O(n^2) per query — measured 10 records 2 ms,
        50 records 43 ms, 200 records 654 ms. Harmless at today's <=5 sources, but a
        real cliff, and the whole-list rewrite bought nothing: a record is only ever
        mutated by the recorder, so the recorder knows exactly which entry changed.

        Called with no argument it still rebuilds the whole section — used on first
        registration, and as the correctness backstop if a caller mutates a record
        directly (query/source_coordinator does this for retry_count)."""
        tr = self._trace
        if tr is None or not getattr(tr, "enabled", False):
            return
        try:
            sec = tr.sections.setdefault(TRACE_SECTION, {})
            sec.setdefault("_ms", 0.0)
            rows = sec.get("records")
            if rec is not None and isinstance(rows, list) and len(rows) == len(self.records):
                try:
                    rows[self.records.index(rec)] = rec.as_dict()
                    return
                except ValueError:
                    pass          # not registered here — fall through to a full rebuild
            sec["records"] = [r.as_dict() for r in self.records]
            sec["count"] = len(self.records)
        except Exception:
            pass

    # -- rollups -------------------------------------------------------------
    def safe_records(self) -> List[Dict[str, Any]]:
        return [r.as_safe_dict() for r in self.records]

    def any_failed(self) -> bool:
        return any(r.status == FAILED for r in self.records)

    def any_required_failed(self) -> bool:
        return any(r.status == FAILED and r.required for r in self.records)

    def ok_count(self) -> int:
        return sum(1 for r in self.records if r.status == COMPLETED)

    def overall_status(self) -> str:
        """'complete' | 'partial' | 'failed' | 'none' for the whole query."""
        if not self.records:
            return "none"
        if all(r.status == COMPLETED for r in self.records):
            return "complete"
        if self.ok_count() == 0:
            return "failed"
        return "partial"


class _NullRecorder:
    """Zero-cost stand-in when the flag is off."""

    enabled = False
    records: List[SourceExecutionRecord] = []

    def open(self, source_id, **k):
        return SourceExecutionRecord(source_id=str(source_id))

    def close(self, rec, status, **kw):
        return rec

    def has_records(self):
        return False

    def safe_records(self):
        return []

    def any_failed(self):
        return False

    def any_required_failed(self):
        return False

    def ok_count(self):
        return 0

    def overall_status(self):
        return "none"


_NULL_RECORDER = _NullRecorder()


def new_recorder(trace=None, timeline=None):
    if not _enabled():
        return _NULL_RECORDER
    return ExecutionRecorder(trace=trace, timeline=timeline)


# ── ambient access ───────────────────────────────────────────────────────────
from contextvars import ContextVar  # noqa: E402

_CURRENT: ContextVar[Optional[ExecutionRecorder]] = ContextVar(
    "veda_current_recorder", default=None)


def current_recorder():
    return _CURRENT.get() or _NULL_RECORDER


def bind_recorder(rec):
    return _CURRENT.set(rec)


def unbind_recorder(token) -> None:
    try:
        _CURRENT.reset(token)
    except Exception:
        pass


class use_recorder:
    def __init__(self, rec):
        self._rec = rec
        self._token = None

    def __enter__(self):
        self._token = bind_recorder(self._rec)
        return self._rec

    def __exit__(self, *exc):
        unbind_recorder(self._token)
        return False
