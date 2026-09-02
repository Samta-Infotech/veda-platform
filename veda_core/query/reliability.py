"""query/reliability.py — bounded transient-retry + failure classification (routing Phase 5.1/5.2).

Source agents already NEVER raise (they return a failed ``AgentResult``). This layer adds two things
on top, both deliberately narrow so as not to duplicate the existing Tier-2 IR repair loop:

- **Failure classification** — is a failed AgentResult TRANSIENT (a temporary infra hiccup worth one
  more try: timeout, connection reset, service unavailable, circuit-open) or PERMANENT (retrying
  can't help: auth denied, invalid/not-found, validation, param assembly)? Unknown → PERMANENT
  (conservative: never retry something that won't fix).
- **Bounded transient retry** — re-run the agent at most N times, ONLY while the failure classifies
  as transient. Flag-gated default-OFF; when off, `execute_reliably` is a pass-through.

Per-source PARTIAL-FAILURE handling for MULTI lives in the coordinator (Phase 5.3), which uses
these classifications to decide required-vs-optional outcomes — this module stays pure/testable.
"""
from __future__ import annotations

import re

CLASS_TRANSIENT = "transient"
CLASS_PERMANENT = "permanent"

# Substrings that mark a transient infra failure (safe to retry). Ordered by specificity; matched
# case-insensitively against the AgentResult.error string.
_TRANSIENT_MARKERS = (
    "timeout", "timed out", "temporarily", "temporarily unavailable", "unavailable",
    "connection reset", "connection refused", "econnrefused", "reset by peer",
    "circuit", "429", "503", "502", "504", "deadlock", "could not connect", "broken pipe",
)
# Substrings that mark a permanent failure (retry can't help) — checked FIRST so an "invalid ...
# timeout config" style message doesn't get mis-read as transient.
# NOTE: no bare "refused" here — it would swallow the transient "connection refused"; a query
# refusal surfaces as AgentResult.status="refused" (not "failed"), so it never reaches this classifier.
_PERMANENT_MARKERS = (
    "auth", "unauthor", "permission", "denied", "forbidden", "not found", "invalid",
    "syntax", "param", "validation", "ungrounded", "does not exist", "no such",
    "semantic model",
)

_WORD = re.compile(r"[a-z0-9]+")


def classify_failure(error) -> str:
    """Classify a failure message as transient or permanent. Empty/None → permanent (nothing to
    retry against)."""
    s = str(error or "").lower()
    if not s:
        return CLASS_PERMANENT
    for m in _PERMANENT_MARKERS:
        if m in s:
            return CLASS_PERMANENT
    for m in _TRANSIENT_MARKERS:
        if m in s:
            return CLASS_TRANSIENT
    return CLASS_PERMANENT


def _retry_config():
    try:
        import config as _cfg
        return (bool(getattr(_cfg, "ROUTING_AGENT_RETRY_ENABLED", False)),
                int(getattr(_cfg, "ROUTING_AGENT_MAX_RETRIES", 1)))
    except Exception:
        return (False, 0)


def execute_reliably(run, *, enabled=None, max_retries=None, classify=None):
    """Run ``run()`` (returns an AgentResult) with bounded transient retry.

    ``run`` must be idempotent (a fresh agent.execute). Retries ONLY while the failure classifies
    transient and attempts remain. When retry is disabled this is a single pass-through call.
    Returns the last AgentResult, annotated with ``reason`` = attempt count when it retried.
    """
    cfg_enabled, cfg_max = _retry_config()
    enabled = cfg_enabled if enabled is None else enabled
    max_retries = cfg_max if max_retries is None else max_retries
    classify = classify or classify_failure

    res = run()
    if not enabled or max_retries <= 0:
        return res

    attempts = 0
    while (getattr(res, "status", "") == "failed"
           and classify(getattr(res, "error", "")) == CLASS_TRANSIENT
           and attempts < max_retries):
        attempts += 1
        res = run()
    if attempts and getattr(res, "reason", "") == "":
        try:
            res.reason = f"retried x{attempts}"
        except Exception:
            pass
    return res


# Federated payload FAILURE statuses (federated_route.py). "ok" is success; a bare None means "not
# federated — use the normal path" (a clean fall-through, never retried).
_FEDERATED_FAILURE_STATUSES = ("exec_error_federated", "refused_federated", "not_federated")


def _federated_retry_config():
    try:
        import config as _cfg
        return (bool(getattr(_cfg, "FEDERATED_TRANSIENT_RETRY_ENABLED", False)),
                int(getattr(_cfg, "FEDERATED_MAX_RETRIES", 1)))
    except Exception:
        return (False, 0)


def federated_transient(payload) -> bool:
    """True when a federated ``run_federated`` payload is a RETRYABLE (transient-infra) failure.

    Reuses the payload's OWN `retryable` label (set by federated_route._labelled_failure, which
    classifies the failure reason with classify_failure); falls back to classifying the reason
    string when the label is absent. A success (`status == "ok"`), a non-dict (None → 'not
    federated'), or a permanent failure → False (never retried)."""
    if not isinstance(payload, dict):
        return False
    if payload.get("status") == "ok" or payload.get("status") not in _FEDERATED_FAILURE_STATUSES:
        return False
    if "retryable" in payload:
        return bool(payload.get("retryable"))
    return classify_failure(payload.get("reason") or payload.get("error")) == CLASS_TRANSIENT


def execute_federated_reliably(run, *, enabled=None, max_retries=None):
    """Bounded transient-retry for the federated route. ``run()`` returns run_federated's dict payload
    (or None). Retries ONLY while the payload is a transient-infra failure and attempts remain; a
    permanent failure, a success, or None returns immediately. Flag-gated default-OFF → a single
    pass-through call. On retry, annotates the payload with ``retry_attempts`` for diagnostics."""
    cfg_enabled, cfg_max = _federated_retry_config()
    enabled = cfg_enabled if enabled is None else enabled
    max_retries = cfg_max if max_retries is None else max_retries

    res = run()
    if not enabled or max_retries <= 0:
        return res

    attempts = 0
    while federated_transient(res) and attempts < max_retries:
        attempts += 1
        res = run()
    if attempts and isinstance(res, dict) and not res.get("retry_attempts"):
        try:
            res["retry_attempts"] = attempts
        except Exception:
            pass
    return res
