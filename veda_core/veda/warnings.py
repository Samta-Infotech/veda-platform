# =============================================================================
# veda/warnings.py
# VEDA — the WARNING tier (traceability Phase 1, Part 11).
#
# VEDA has always had exactly two outcomes: the query passed, or it was refused.
# There was no way to say "you got an answer, but read it with this caveat" —
# so truncation, a source that didn't respond, RBAC narrowing, an unresolved
# cross-source conflict and a fallback retrieval path were all INVISIBLE to the
# user. That silence is what makes a partially-wrong answer look complete.
#
# A Warning is a first-class, user-safe fact with a STABLE code. Codes are the
# contract (a client may switch on them); messages are human copy and may be
# reworded without breaking anyone.
#
# ONE record, three consumers — same rule as veda/lifecycle.py:
#       add(...)  ──┬──►  ExplainTrace section "warnings"
#                   ├──►  the lifecycle timeline (so it streams live)
#                   └──►  build_explain()'s "warnings" block
#
# SAFETY
#   A warning says THAT something was limited, never WHICH restricted thing.
#   "Some data could not be included because of access restrictions" is correct;
#   naming the table would itself be the leak the warning is reporting.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

TRACE_SECTION = "warnings"

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITIES = (SEVERITY_INFO, SEVERITY_WARNING)

# ── stable codes (the client contract) ───────────────────────────────────────
PARTIAL_SOURCE_FAILURE = "partial_source_failure"
RESULT_TRUNCATED = "result_truncated"
RESTRICTED_DATA = "restricted_data"
UNMATCHED_RECORDS = "unmatched_records"
SOURCE_CONFLICT = "source_conflict"
FALLBACK_USED = "fallback_used"
LOW_EVIDENCE = "low_evidence"

#: code -> (severity, default user-safe message). The message never names a
#: table, column, source the user can't see, or an internal component.
CATALOG: Dict[str, tuple] = {
    PARTIAL_SOURCE_FAILURE: (
        SEVERITY_WARNING,
        "Results may be incomplete because one data source did not respond."),
    RESULT_TRUNCATED: (
        SEVERITY_INFO,
        "The result was limited to the first {limit} records."),
    RESTRICTED_DATA: (
        SEVERITY_WARNING,
        "Some available data could not be included because of access restrictions."),
    UNMATCHED_RECORDS: (
        SEVERITY_INFO,
        "Some records could not be matched across the data sources."),
    SOURCE_CONFLICT: (
        SEVERITY_WARNING,
        "The data sources returned conflicting values, so no single value is reported."),
    FALLBACK_USED: (
        SEVERITY_INFO,
        # EXP-B6: was "One retrieval method was unavailable…" — wrong for the trigger
        # that actually fires it. The dominant case is the deterministic head handing
        # off because it could not answer, not because anything was down. A genuinely
        # unreachable source is PARTIAL_SOURCE_FAILURE, which has its own copy.
        "The primary method could not answer this, so an alternate method was used."),
    LOW_EVIDENCE: (
        SEVERITY_WARNING,
        "There was limited matching data for this question, so the answer may be incomplete."),
}


#: Wording for a templated CATALOG entry when its substitution values are absent.
#: Every code whose default message contains a "{}" placeholder MUST appear here —
#: test_every_templated_code_has_a_no_arg_message enforces it, so adding a
#: templated warning without a fallback fails the suite rather than shipping a
#: raw "{limit}" to a user.
NO_ARG_MESSAGE: Dict[str, str] = {
    RESULT_TRUNCATED: "The result was limited to a maximum number of records.",
}


@dataclass
class Warning_:
    """One user-safe caveat. Named with a trailing underscore so it cannot shadow
    the builtin `Warning` for any module that does `from veda.warnings import *`."""

    code: str
    severity: str
    message: str
    user_safe: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "QUERY_WARNINGS_ENABLED", False))
    except Exception:
        return False


def build(code: str, message: Optional[str] = None, severity: Optional[str] = None,
          **details) -> Optional[Warning_]:
    """Construct a warning from the catalog. Unknown code -> None (never invent copy).

    `message` overrides the catalog default; `details` both fills `{}` placeholders
    in the default message and rides along as structured, already-safe scalars.
    """
    entry = CATALOG.get(code)
    if entry is None:
        return None
    cat_sev, cat_msg = entry
    text = message or cat_msg
    if "{" in text:
        # Attempted UNCONDITIONALLY, not just when details are supplied: a
        # templated message called with no values would otherwise be shown to the
        # user with its raw "{limit}" placeholder intact.
        try:
            text = text.format(**details)
        except (KeyError, IndexError, ValueError):
            text = NO_ARG_MESSAGE.get(code) or cat_msg.split("{")[0].strip().rstrip(",;:") + "."
    return Warning_(code=code, severity=severity or cat_sev, message=text,
                    details={k: v for k, v in details.items() if v is not None})


def add(code: str, *, trace=None, timeline=None, message: Optional[str] = None,
        severity: Optional[str] = None, **details) -> Optional[Warning_]:
    """Record one warning and fan it out to the trace + the live timeline.

    Idempotent per (code): the same condition detected twice in one query — e.g.
    two stages both noticing truncation — yields ONE warning, so the user never
    sees a duplicated caveat. Never raises.
    """
    if not _enabled():
        return None
    try:
        w = build(code, message=message, severity=severity, **details)
        if w is None:
            return None

        tr = trace
        if tr is None:
            try:
                from veda.explain import current_trace
                tr = current_trace()
            except Exception:
                tr = None

        if tr is not None and getattr(tr, "enabled", False):
            sec = tr.sections.setdefault(TRACE_SECTION, {})
            items = sec.setdefault("items", [])
            if any(i.get("code") == w.code for i in items):
                return None                      # already recorded — stay idempotent
            items.append(w.as_dict())
            sec["count"] = len(items)

        tl = timeline
        if tl is None:
            try:
                from veda.lifecycle import current_timeline
                tl = current_timeline()
            except Exception:
                tl = None
        if tl is not None and getattr(tl, "enabled", False):
            # Streams as a warning on whichever phase best describes the caveat, so
            # the user sees it AS IT IS DISCOVERED rather than only at the end.
            from veda import lifecycle as lc
            phase = _PHASE_FOR_CODE.get(w.code, lc.PHASE_VALIDATION)
            tl.warning(phase, w.message, code=w.code)
        return w
    except Exception:
        return None


def collect(trace=None) -> List[Dict[str, Any]]:
    """Every warning recorded on this query, in discovery order. Safe to expose."""
    tr = trace
    if tr is None:
        try:
            from veda.explain import current_trace
            tr = current_trace()
        except Exception:
            return []
    try:
        return list((tr.sections.get(TRACE_SECTION) or {}).get("items") or [])
    except Exception:
        return []


# Which lifecycle phase each warning naturally belongs to, for live streaming.
from veda import lifecycle as _lc  # noqa: E402  (kept beside its only user)

_PHASE_FOR_CODE = {
    PARTIAL_SOURCE_FAILURE: _lc.PHASE_DATA_RETRIEVAL,
    RESULT_TRUNCATED: _lc.PHASE_RESULT_PREPARATION,
    RESTRICTED_DATA: _lc.PHASE_ACCESS_CHECK,
    UNMATCHED_RECORDS: _lc.PHASE_CROSS_SOURCE,
    SOURCE_CONFLICT: _lc.PHASE_CROSS_SOURCE,
    FALLBACK_USED: _lc.PHASE_DATA_RETRIEVAL,
    LOW_EVIDENCE: _lc.PHASE_SOURCE_SELECTION,
}
