"""Tests for query.reliability + per-source partial-failure in MULTI (routing Phase 5.1/5.2/5.3)."""
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORE = os.path.abspath(os.path.join(ROOT, "veda_core"))
sys.path.insert(0, CORE)

from query.reliability import classify_failure, execute_reliably, CLASS_TRANSIENT, CLASS_PERMANENT  # noqa: E402
from query.agents import AgentResult  # noqa: E402
from query.routing_contracts import (  # noqa: E402
    RoutingDecision, CandidateSource, MODE_MULTI, STATUS_ROUTED, RC_SLM_RESOLVED)
import query.agents as A  # noqa: E402
import query.source_coordinator as SC  # noqa: E402


# ── classification ────────────────────────────────────────────────────────────
def test_transient_classification():
    for e in ["connection timed out", "service temporarily unavailable", "503 upstream",
              "connection refused", "circuit open", "deadlock detected"]:
        assert classify_failure(e) == CLASS_TRANSIENT


def test_permanent_classification():
    for e in ["permission denied", "invalid column", "not found", "syntax error",
              "semantic model not provided", "", "some unknown error"]:
        assert classify_failure(e) == CLASS_PERMANENT


# ── bounded retry ─────────────────────────────────────────────────────────────
def test_transient_retried_then_succeeds():
    n = {"c": 0}
    def run():
        n["c"] += 1
        return (AgentResult("5", "relational", "failed", error="connection timed out")
                if n["c"] < 2 else AgentResult("5", "relational", "ok", data={"rows": [[1]]}))
    r = execute_reliably(run, enabled=True, max_retries=2)
    assert r.status == "ok" and n["c"] == 2


def test_permanent_never_retried():
    n = {"c": 0}
    def run():
        n["c"] += 1
        return AgentResult("5", "relational", "failed", error="permission denied")
    execute_reliably(run, enabled=True, max_retries=3)
    assert n["c"] == 1


def test_disabled_is_single_pass():
    n = {"c": 0}
    def run():
        n["c"] += 1
        return AgentResult("5", "relational", "failed", error="timeout")
    execute_reliably(run, enabled=False, max_retries=3)
    assert n["c"] == 1


# ── partial-failure surfacing (5.3) ────────────────────────────────────────────
def test_multi_partial_failure_is_surfaced(monkeypatch):
    seq = iter([{"ok": True, "rows": [[100]]}, RuntimeError("permission denied")])
    def deleg(q, sm, cols, on_event=None):
        v = next(seq)
        if isinstance(v, Exception):
            raise v
        return v
    monkeypatch.setattr(A, "_sql_delegate", deleg)
    d = RoutingDecision(STATUS_ROUTED, mode=MODE_MULTI, source_ids=["5", "7"],
                        reason_code=RC_SLM_RESOLVED,
                        candidate_sources=[CandidateSource("5", source_type="relational"),
                                           CandidateSource("7", source_type="relational")])
    out = SC.execute_decision(d, "q", sm={}, cols=[])
    p = out["partial"]
    assert out["kind"] == "independent"
    assert p["complete"] is False and p["any_required_failed"] is True and p["ok_count"] == 1
    assert p["failures"][0]["source_id"] == "7" and p["failures"][0]["failure_class"] == "permanent"
    assert out["merge"].source_ids == ["5"]      # only the succeeding source is merged


if __name__ == "__main__":
    import traceback

    class _MP:
        def __init__(self): self._u = []
        def setattr(self, o, n, v): self._u.append((o, n, getattr(o, n))); setattr(o, n, v)
        def undo(self):
            for o, n, v in reversed(self._u):
                setattr(o, n, v)

    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        mp = _MP()
        try:
            fn(mp) if fn.__code__.co_argcount else fn(); print("PASS", name)
        except Exception:
            failed += 1; print("FAIL", name); traceback.print_exc()
        finally:
            mp.undo()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
