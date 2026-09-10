# =============================================================================
# veda/lifecycle.py
# VEDA — the USER-SAFE query execution timeline (traceability Phase 1).
#
# ONE record, three consumers (the "single source of truth" rule):
#
#       emit(...)  ──┬──►  ExplainTrace section "lifecycle"   (internal, persisted)
#                    ├──►  on_event(phase, message, extra)    (live SSE "thinking")
#                    └──►  safe_projection.build_timeline()   (final explainability)
#
# Nothing downstream re-derives "what happened" by guessing — every layer reads
# the SAME LifecycleEvent list. That is why this module owns the emit, and why
# neither the SSE path nor build_explain() reconstructs phases independently.
#
# WHAT THIS IS NOT
#   These are EXECUTION-STATE facts (a stage started / finished / warned), not
#   model thoughts. No prompt, no chain-of-thought, no candidate list, no score,
#   no schema name ever passes through here. `message` is a fixed, human-authored
#   string chosen from _COPY below (or a caller-supplied string that must obey the
#   same rule); `details` is a small dict of already-safe scalars. Anything richer
#   belongs in the internal trace, not in a LifecycleEvent.
#
# FLAG
#   LIFECYCLE_EVENTS_ENABLED (config.py, default False). Off → new_timeline()
#   returns _NullTimeline and every call site costs one attribute lookup, so the
#   production answer path stays byte-identical.
# =============================================================================

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

# ── phases (business-facing, stable; NOT internal stage names) ────────────────
# The closed vocabulary a client may switch on. Internal pipeline stages map ONTO
# these at their emit site (veda_hybrid.py, veda/pipeline.py,
# query/source_coordinator.py via veda/exec_records.py); a stage with no mapping
# emits NOTHING rather than leaking its internal name — Timeline.emit() rejects
# any phase outside PHASES below.
PHASE_RECEIVED = "received"
PHASE_UNDERSTANDING = "understanding"
PHASE_ACCESS_CHECK = "access_check"
PHASE_SOURCE_SELECTION = "source_selection"
PHASE_EXECUTION_PLAN = "execution_plan"
PHASE_DATA_RETRIEVAL = "data_retrieval"
PHASE_CROSS_SOURCE = "cross_source_processing"
PHASE_VALIDATION = "validation"
PHASE_RESULT_PREPARATION = "result_preparation"
PHASE_COMPLETED = "completed"

#: Phases the terminal sweep may never resolve NEGATIVELY.
#:
#: The sweep exists to stop a spinner, not to attribute blame. It infers `failed`
#: from the STATUS OF SOME OTHER STAGE, which is a guess — and for these phases a
#: wrong guess is actively harmful:
#:
#: `access_check` — measured live on a data-lake question. Tier-1 ended in
#: `qualifier_dropped`, so the sweep marked the still-open access_check `failed`;
#: Tier-2 then recovered and produced a clarify about an ambiguous column, and the
#: user was left with a red cross on "Checking access permissions" above a reply
#: that had nothing to do with permissions. Telling someone they lack permission
#: they actually have is the worst thing this layer can get wrong.
#:
#: It is SAFE to resolve these positively, because a real failure is emitted
#: EXPLICITLY by the code that makes the decision — the routing permission
#: pre-check, and the terminal-feedback reconciliation at the front door. An
#: unresolved access_check therefore means no access decision blocked the query,
#: which is a fact, not an assumption.
NEVER_SWEEP_TO_FAILED = frozenset({"access_check"})

PHASES = (
    PHASE_RECEIVED, PHASE_UNDERSTANDING, PHASE_ACCESS_CHECK, PHASE_SOURCE_SELECTION,
    PHASE_EXECUTION_PLAN, PHASE_DATA_RETRIEVAL, PHASE_CROSS_SOURCE, PHASE_VALIDATION,
    PHASE_RESULT_PREPARATION, PHASE_COMPLETED,
)

# ── statuses ─────────────────────────────────────────────────────────────────
STATUS_STARTED = "started"
STATUS_COMPLETED = "completed"
STATUS_WARNING = "warning"
STATUS_FAILED = "failed"
STATUSES = (STATUS_STARTED, STATUS_COMPLETED, STATUS_WARNING, STATUS_FAILED)

# ── the section the trace stores these under ─────────────────────────────────
TRACE_SECTION = "lifecycle"

# Per-phase display title. One title per phase — the client groups events by
# phase and shows this once as the heading, with each event's `message` beneath.
PHASE_TITLES: Dict[str, str] = {
    PHASE_RECEIVED: "Received your question",
    PHASE_UNDERSTANDING: "Understanding your question",
    PHASE_ACCESS_CHECK: "Checking data access",
    PHASE_SOURCE_SELECTION: "Checking available data",
    PHASE_EXECUTION_PLAN: "Planning the query",
    PHASE_DATA_RETRIEVAL: "Running the query",
    PHASE_CROSS_SOURCE: "Combining data",
    # Pre-FLIGHT, not post-hoc: every check in the ledger (value_grounding,
    # qualifier_completeness, ir_equivalence, ast_readonly_parameterized_fanout)
    # runs BEFORE execute_sql. Live verification confirmed validation completes
    # before data_retrieval, so calling this "checking the result" would describe
    # a result that does not exist yet.
    PHASE_VALIDATION: "Checking the query",
    PHASE_RESULT_PREPARATION: "Preparing your answer",
    PHASE_COMPLETED: "Done",
}

# Default per-(phase, status) copy, used when a call site passes no message. Every
# string here is deliberately generic: it describes the STAGE, never the data.
_COPY: Dict[tuple, str] = {
    (PHASE_RECEIVED, STATUS_COMPLETED): "Got your question",
    (PHASE_UNDERSTANDING, STATUS_STARTED): "Reading your question",
    (PHASE_UNDERSTANDING, STATUS_COMPLETED): "Understood what you're asking for",
    (PHASE_ACCESS_CHECK, STATUS_STARTED): "Checking your access",
    (PHASE_ACCESS_CHECK, STATUS_COMPLETED): "Verified your access",
    (PHASE_ACCESS_CHECK, STATUS_WARNING):
        "Some data could not be included because of access restrictions",
    (PHASE_ACCESS_CHECK, STATUS_FAILED):
        "This question needs data you do not have permission to access",
    (PHASE_SOURCE_SELECTION, STATUS_STARTED): "Looking for relevant data",
    (PHASE_SOURCE_SELECTION, STATUS_COMPLETED): "Found relevant data",
    (PHASE_SOURCE_SELECTION, STATUS_FAILED): "Could not find data relevant to this question",
    (PHASE_EXECUTION_PLAN, STATUS_COMPLETED): "Prepared the query",
    (PHASE_DATA_RETRIEVAL, STATUS_STARTED): "Retrieving results",
    (PHASE_DATA_RETRIEVAL, STATUS_COMPLETED): "Retrieved results",
    (PHASE_DATA_RETRIEVAL, STATUS_WARNING): "Some data could not be retrieved",
    (PHASE_DATA_RETRIEVAL, STATUS_FAILED): "Could not retrieve the results",
    (PHASE_CROSS_SOURCE, STATUS_STARTED): "Combining data from more than one source",
    (PHASE_CROSS_SOURCE, STATUS_COMPLETED): "Combined the data",
    (PHASE_CROSS_SOURCE, STATUS_WARNING): "The data sources did not fully agree",
    (PHASE_VALIDATION, STATUS_STARTED): "Checking the query is safe and complete",
    (PHASE_VALIDATION, STATUS_COMPLETED): "Safety checks passed",
    (PHASE_VALIDATION, STATUS_WARNING): "The query has some limitations",
    (PHASE_VALIDATION, STATUS_FAILED): "The query did not pass a safety check",
    (PHASE_RESULT_PREPARATION, STATUS_STARTED): "Preparing your answer",
    (PHASE_RESULT_PREPARATION, STATUS_COMPLETED): "Answer ready",
    (PHASE_COMPLETED, STATUS_COMPLETED): "Done",
}


def default_message(phase: str, status: str) -> str:
    """The stage-describing copy for a (phase, status), or a safe generic fallback.
    Never returns anything derived from the query, the data, or the schema."""
    return _COPY.get((phase, status)) or PHASE_TITLES.get(phase) or "Working on it"


@dataclass
class LifecycleEvent:
    """One observable execution fact, already safe to show a user.

    `details` carries only pre-sanitized scalars/short lists that a call site has
    explicitly decided are user-safe (e.g. {"source_count": 2}). It is NEVER a
    dump of an internal object — putting a raw dict in here is the one way this
    module can leak, so call sites pass named kwargs, not blobs.
    """

    phase: str
    status: str
    title: str
    message: str
    timestamp_ms: int
    elapsed_ms: float
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def as_sse_extra(self) -> Dict[str, Any]:
        """The structured fields to merge into the existing `thinking` SSE payload.

        Deliberately does NOT include `message` — the api tier owns the displayed
        string (it runs the message through its own business-friendly mapping),
        and duplicating it here would let the two drift. `phase` is included
        because the existing api-tier `on_event` bridge already forwards it.
        """
        out = {
            "status": self.status,
            "title": self.title,
            "timestamp_ms": self.timestamp_ms,
            "elapsed_ms": self.elapsed_ms,
            "details_available": bool(self.details),
        }
        if self.details:
            out["details"] = dict(self.details)
        return out


class _NullTimeline:
    """Zero-cost stand-in when the flag is off. Same surface, every method a no-op,
    so call sites are identical in both flag states."""

    enabled = False
    events: List[LifecycleEvent] = []

    def emit(self, *a, **k):
        return None

    def started(self, *a, **k):
        return None

    def completed(self, *a, **k):
        return None

    def warning(self, *a, **k):
        return None

    def failed(self, *a, **k):
        return None

    def close_open_phases(self, *a, **k):
        return None

    def as_list(self):
        return []


class Timeline:
    """Collects LifecycleEvents for ONE query and fans each one out immediately.

    Fan-out order is deliberate: record into the trace FIRST, then fire the SSE
    callback. A callback that raises must not lose the recorded fact — the trace
    is the durable copy and the stream is best-effort.
    """

    enabled = True

    def __init__(self, on_event: Optional[Callable] = None, trace=None, t0: Optional[float] = None):
        self._on_event = on_event
        self._trace = trace
        self._t0 = t0 if t0 is not None else time.time()
        self.events: List[LifecycleEvent] = []

    # -- core ----------------------------------------------------------------
    def emit(self, phase: str, status: str = STATUS_COMPLETED,
             message: Optional[str] = None, **details) -> Optional[LifecycleEvent]:
        """Record one execution fact and fan it out. Never raises."""
        try:
            if phase not in PHASES or status not in STATUSES:
                # An unknown phase/status is a programming bug, but an observability
                # bug must never fail a query — drop it rather than emit something
                # unmapped (and therefore potentially unsafe) to the user.
                return None
            now = time.time()
            ev = LifecycleEvent(
                phase=phase,
                status=status,
                title=PHASE_TITLES.get(phase, ""),
                message=message or default_message(phase, status),
                timestamp_ms=int(now * 1000),
                elapsed_ms=round((now - self._t0) * 1000, 1),
                details={k: v for k, v in details.items() if v is not None},
            )
            self.events.append(ev)
            self._record(ev)
            self._stream(ev)
            return ev
        except Exception:
            return None

    def _record(self, ev: LifecycleEvent) -> None:
        """Append to the internal trace — the durable copy."""
        tr = self._trace
        if tr is None or not getattr(tr, "enabled", False):
            return
        try:
            sec = tr.sections.setdefault(TRACE_SECTION, {})
            sec.setdefault("_ms", ev.elapsed_ms)
            sec.setdefault("events", []).append(ev.as_dict())
            sec["count"] = len(sec["events"])
        except Exception:
            pass

    def _stream(self, ev: LifecycleEvent) -> None:
        """Fire the live progress callback. Best-effort by contract."""
        if self._on_event is None:
            return
        try:
            self._on_event(ev.phase, ev.message, ev.as_sse_extra())
        except Exception:
            pass

    # -- ergonomic wrappers --------------------------------------------------
    def started(self, phase, message=None, **d):
        return self.emit(phase, STATUS_STARTED, message, **d)

    def completed(self, phase, message=None, **d):
        return self.emit(phase, STATUS_COMPLETED, message, **d)

    def warning(self, phase, message=None, **d):
        return self.emit(phase, STATUS_WARNING, message, **d)

    def failed(self, phase, message=None, **d):
        return self.emit(phase, STATUS_FAILED, message, **d)

    def as_list(self) -> List[Dict[str, Any]]:
        return [e.as_dict() for e in self.events]

    def close_open_phases(self, *, failed: bool = False) -> None:
        """Resolve any phase left `started` and never resolved. Idempotent.

        Called TWICE by design: once by veda/pipeline.py::_done BEFORE it builds the
        explainability payload, and once by the front door's terminal handler.

        The first call is the one that matters for the PERSISTED record. Observed on
        a real stream: `access_check` resolves late (after validation), the front
        door's sweep only ran once the engine had already returned, and so the SAVED
        payload's `timeline_summary` recorded `access_check: "started"` forever — an
        unresolved step inside a finished, stored answer.

        A phase already resolved NEGATIVELY is left alone: a real permission failure
        must never be flipped to "verified" by a sweep.

        And the converse: a phase in NEVER_SWEEP_TO_FAILED is never resolved
        negatively BY the sweep, because `failed` here is inferred from another
        stage's status and is therefore a guess. See that constant for the measured
        case this cost us.
        """
        try:
            resolved = {e.phase for e in self.events
                        if e.status in (STATUS_COMPLETED, STATUS_WARNING, STATUS_FAILED)}
            opened = {e.phase for e in self.events if e.status == STATUS_STARTED}
            for ph in sorted(opened - resolved,
                             key=lambda x: PHASES.index(x) if x in PHASES else 99):
                if failed and ph not in NEVER_SWEEP_TO_FAILED:
                    self.failed(ph)
                else:
                    self.completed(ph)
        except Exception:
            pass


_NULL_TIMELINE = _NullTimeline()


def lifecycle_enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "LIFECYCLE_EVENTS_ENABLED", False))
    except Exception:
        return False


def new_timeline(on_event=None, trace=None, t0=None):
    """Factory: a real Timeline when the flag is on, else the zero-cost null one."""
    if not lifecycle_enabled():
        return _NULL_TIMELINE
    return Timeline(on_event=on_event, trace=trace, t0=t0)


# ── ambient access (mirrors veda/explain.py's contextvar pattern) ─────────────
# Bound once per request by the front door so any stage can emit without threading
# a `timeline` parameter through every signature. Worker threads start with an
# EMPTY contextvars context — the same caveat explain.py documents — so a thread
# that must emit has to re-bind via use_timeline().
from contextvars import ContextVar  # noqa: E402  (kept next to its users)

_CURRENT: ContextVar[Optional[Timeline]] = ContextVar("veda_current_timeline", default=None)


def current_timeline():
    """The timeline bound to this context, or the zero-cost null one."""
    return _CURRENT.get() or _NULL_TIMELINE


def bind_timeline(tl):
    return _CURRENT.set(tl)


def unbind_timeline(token) -> None:
    try:
        _CURRENT.reset(token)
    except Exception:
        pass


class use_timeline:
    """Scope `tl` as the ambient timeline for the duration of the block."""

    def __init__(self, tl):
        self._tl = tl
        self._token = None

    def __enter__(self):
        self._token = bind_timeline(self._tl)
        return self._tl

    def __exit__(self, *exc):
        unbind_timeline(self._token)
        return False
