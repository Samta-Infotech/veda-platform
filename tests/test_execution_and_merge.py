"""Tests for Phase 4 — execution planner, result orchestrator, cross-source guard, MULTI dispatch.

Pure / injected delegates, no DB. Run: `pytest tests/test_execution_and_merge.py`.
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORE = os.path.abspath(os.path.join(ROOT, "veda_core"))
sys.path.insert(0, CORE)

from query.routing_contracts import (  # noqa: E402
    RoutingDecision, CandidateSource, MODE_SINGLE, MODE_MULTI, STATUS_ROUTED, STATUS_CLARIFY,
    RC_RELATIONSHIP_EDGE, RC_SLM_RESOLVED,
)
from query.execution_planner import (  # noqa: E402
    plan_execution, STRATEGY_SINGLE, STRATEGY_FEDERATED, STRATEGY_INDEPENDENT, MODE_PARALLEL)
from query.result_orchestrator import (  # noqa: E402
    merge_results, POLICY_APPEND, POLICY_CANONICAL_PRIORITY, POLICY_CONFLICT_DETECTED)
from query.cross_source_guard import guard_cross_source_answer  # noqa: E402
from query.agents import AgentResult  # noqa: E402
import query.agents as A  # noqa: E402
import query.source_coordinator as SC  # noqa: E402


def _c(sid, typ="relational", canon=False):
    return CandidateSource(source_id=sid, source_type=typ, is_canonical=canon)


def _R(sid, rows, status="ok"):
    return AgentResult(sid, "relational", status, data={"rows": rows, "answer": f"ans{sid}"})


# ── planner ─────────────────────────────────────────────────────────────────────
def test_planner_single():
    d = RoutingDecision(STATUS_ROUTED, mode=MODE_SINGLE, source_ids=["5"], candidate_sources=[_c("5")])
    p = plan_execution(d)
    assert p.strategy == STRATEGY_SINGLE and p.mode == MODE_PARALLEL and len(p.steps) == 1


def test_planner_edge_is_federated():
    d = RoutingDecision(STATUS_ROUTED, mode=MODE_MULTI, source_ids=["5", "7"],
                        reason_code=RC_RELATIONSHIP_EDGE, candidate_sources=[_c("5"), _c("7")])
    assert plan_execution(d).strategy == STRATEGY_FEDERATED


def test_planner_slm_multi_is_independent():
    d = RoutingDecision(STATUS_ROUTED, mode=MODE_MULTI, source_ids=["5", "7"],
                        reason_code=RC_SLM_RESOLVED, candidate_sources=[_c("5"), _c("7")])
    assert plan_execution(d).strategy == STRATEGY_INDEPENDENT


# ── orchestrator ──────────────────────────────────────────────────────────────
def test_merge_append_for_different_answers():
    assert merge_results([_R("5", [[100]]), _R("7", [["text"]])]).policy == POLICY_APPEND


def test_merge_scalar_conflict_no_canonical():
    m = merge_results([_R("5", [[100]]), _R("7", [[120]])])
    assert m.policy == POLICY_CONFLICT_DETECTED and m.needs_clarification


def test_merge_scalar_conflict_canonical_wins():
    m = merge_results([_R("5", [[100]]), _R("7", [[120]])], canonical_ids={"5"})
    assert m.policy == POLICY_CANONICAL_PRIORITY and m.winner_source_id == "5" and not m.needs_clarification


def test_merge_same_value_is_append():
    assert merge_results([_R("5", [[100]]), _R("7", [[100]])]).policy == POLICY_APPEND


def test_merge_drops_failed():
    assert merge_results([_R("5", [[100]]), _R("7", [[120]], status="failed")]).source_ids == ["5"]


# ── cross-source grounding guard ──────────────────────────────────────────────
def test_guard_passes_grounded_numbers():
    m = merge_results([_R("5", [[100]]), _R("7", [[4200]])])
    ok, _ = guard_cross_source_answer("100 revenue, 4200 events", m)
    assert ok


def test_guard_flags_fabricated_number():
    m = merge_results([_R("5", [[100]]), _R("7", [[4200]])])
    ok, reason = guard_cross_source_answer("revenue jumped to 999999", m)
    assert not ok and reason


# ── MULTI dispatch via coordinator ────────────────────────────────────────────
def test_execute_decision_federated(monkeypatch):
    monkeypatch.setattr(SC, "_federated_delegate",
                        lambda q, tenant, sids: {"status": "ok", "sources": sids})
    d = RoutingDecision(STATUS_ROUTED, mode=MODE_MULTI, source_ids=["5", "7"],
                        reason_code=RC_RELATIONSHIP_EDGE,
                        candidate_sources=[_c("5"), _c("7", "datalake")])
    out = SC.execute_decision(d, "q")
    assert out["kind"] == "federated" and out["result"]["status"] == "ok"


def test_execute_decision_independent_merges(monkeypatch):
    vals = iter([[100], [120]])
    monkeypatch.setattr(A, "_sql_delegate",
                        lambda q, sm, cols, on_event=None: {"ok": True, "rows": [next(vals)]})
    d = RoutingDecision(STATUS_ROUTED, mode=MODE_MULTI, source_ids=["5", "7"],
                        reason_code=RC_SLM_RESOLVED,
                        candidate_sources=[_c("5", canon=True), _c("7")])
    out = SC.execute_decision(d, "q", sm={}, cols=[])
    assert out["kind"] == "independent" and out["merge"].policy == POLICY_CANONICAL_PRIORITY


def test_execute_decision_clarify_is_none():
    d = RoutingDecision(STATUS_CLARIFY, mode="NONE")
    assert SC.execute_decision(d, "q")["kind"] == "none"


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
