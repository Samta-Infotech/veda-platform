"""Tests for the authoritative multi-source routing wiring in veda_hybrid (_run_coordinator).

Verifies the coordinator DRIVES the answer when flag on + not shadow, and is a no-op otherwise.
Providers/plan are monkeypatched → no DB/SLM. Run: `python tests/test_authoritative_routing.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

import config  # noqa: E402
import veda_hybrid  # noqa: E402
import query.source_coordinator as SC  # noqa: E402
from context import RequestContext, set_context  # noqa: E402
from query.routing_contracts import (  # noqa: E402
    RoutingDecision, STATUS_ROUTED, STATUS_NO_MATCH, STATUS_CLARIFY, MODE_SINGLE, MODE_MULTI,
    RC_RELATIONSHIP_EDGE, RC_SLM_RESOLVED, CandidateSource)
from query.agents import AgentResult  # noqa: E402
from query.multi_result import STATUS_OK, STATUS_REFUSED, STATUS_ERROR  # noqa: E402


def _merge(**kw):
    return type("M", (), {"policy": kw.get("policy", "APPEND"),
                          "needs_clarification": kw.get("needs_clarification", False),
                          "winner_source_id": kw.get("winner_source_id", ""),
                          "conflict": kw.get("conflict")})()


def _setup(shadow=False):
    config.MULTISOURCE_ROUTING_ENABLED = True
    config.MULTISOURCE_ROUTING_SHADOW = shadow
    set_context(RequestContext(source_id=5, tenant="default", source_ids=(5, 7)))
    veda_hybrid._load_semantic_model = lambda: ({}, [])


def test_flag_off_is_noop():
    config.MULTISOURCE_ROUTING_ENABLED = False
    assert veda_hybrid._run_coordinator("q") is None


def test_shadow_returns_none_but_still_computes():
    _setup(shadow=True)
    SC.plan_route = lambda q, sids, **k: RoutingDecision(status=STATUS_NO_MATCH, reason="x")
    assert veda_hybrid._run_coordinator("q") is None      # observe only, no answer steering


def test_no_match_refuses_without_answer():
    _setup()
    SC.plan_route = lambda q, sids, **k: RoutingDecision(status=STATUS_NO_MATCH,
                                                         reason="No source had evidence.")
    mr = veda_hybrid._run_coordinator("weird")
    assert mr is not None and mr.items[0].status == STATUS_REFUSED
    assert "evidence" in mr.items[0].refuse_reason


def test_clarify_refuses_with_question():
    _setup()
    SC.plan_route = lambda q, sids, **k: RoutingDecision(status=STATUS_CLARIFY,
                                                         reason="Which source: A or B?")
    mr = veda_hybrid._run_coordinator("ambiguous")
    assert mr.items[0].status == STATUS_REFUSED and "Which source" in mr.items[0].refuse_reason


def test_single_dispatches_and_answers():
    _setup()
    SC.plan_route = lambda q, sids, **k: RoutingDecision(status=STATUS_ROUTED, mode=MODE_SINGLE,
                                                         source_ids=["5"])
    SC.execute_decision = lambda dec, q, **k: {"kind": "single", "result": AgentResult(
        "5", "relational", "ok", engine="deterministic_sql",
        data={"cols": ["rev"], "rows": [[100]], "answer": "Revenue is 100"})}
    mr = veda_hybrid._run_coordinator("revenue")
    assert mr.items[0].status == STATUS_OK and mr.items[0].result.get("rows") == [[100]]


def test_single_no_result_constrains_scope_not_federated():
    """Regression: an authoritative SINGLE[s] decision whose agent yields NO result must NOT fall
    through to the cross-source federated path (which would answer from a DIFFERENT source — the
    src_5.amenities_catalog mis-execution). It defers (returns None) but first narrows the ambient
    scope to the routed source so _maybe_federated becomes a no-op."""
    _setup()  # scope starts as (5, 7)
    SC.plan_route = lambda q, sids, **k: RoutingDecision(status=STATUS_ROUTED, mode=MODE_SINGLE,
                                                         source_ids=["5"])
    SC.execute_decision = lambda dec, q, **k: {"kind": "single", "result": None}  # agent no-result
    mr = veda_hybrid._run_coordinator("q")
    assert mr is None                                   # defers to legacy path
    ctx = veda_hybrid._current_ctx()
    assert tuple(ctx.source_ids) == (5,)               # scope narrowed → federated is now a no-op
    assert veda_hybrid._maybe_federated("q") is None   # cross-source path cannot fire on 1 source


def test_is_datalake_source_detects_by_profile_and_candidate():
    """_is_datalake_source recognises a tabular source from its profile OR the decision's candidates."""
    dec = RoutingDecision(status=STATUS_ROUTED, mode=MODE_SINGLE, source_ids=["5"],
                          candidate_sources=[CandidateSource("5", source_type="datalake")])
    assert veda_hybrid._is_datalake_source("5", dec, {"5": {"source_type": "datalake"}}) is True
    assert veda_hybrid._is_datalake_source("5", dec, {}) is True          # via candidate_sources
    dec2 = RoutingDecision(status=STATUS_ROUTED, mode=MODE_SINGLE, source_ids=["2"],
                           candidate_sources=[CandidateSource("2", source_type="relational")])
    assert veda_hybrid._is_datalake_source("2", dec2, {"2": {"source_type": "relational"}}) is False


def test_single_datalake_augments_sm_and_scopes_relational_does_not():
    """A datalake SINGLE route constrains scope + augments the sm (so parquet columns pass the SQL
    validator); a relational SINGLE route does NEITHER (homzhub path stays byte-identical)."""
    _setup()
    calls = {"constrain": [], "augment": []}
    _orig_constrain = veda_hybrid._constrain_scope_to
    _orig_augment = veda_hybrid._augment_sm_for_datalake
    try:
        veda_hybrid._constrain_scope_to = lambda s: calls["constrain"].append(str(s))
        veda_hybrid._augment_sm_for_datalake = lambda sm, cols, s: (calls["augment"].append(str(s)) or ({}, []))
        SC.execute_decision = lambda dec, q, **k: {"kind": "single", "result": AgentResult(
            "x", "y", "ok", data={"rows": [[1]], "answer": "a"})}

        # datalake route → both helpers fire for the routed source
        SC.plan_route = lambda q, sids, **k: RoutingDecision(
            status=STATUS_ROUTED, mode=MODE_SINGLE, source_ids=["5"],
            candidate_sources=[CandidateSource("5", source_type="datalake")])
        veda_hybrid._run_coordinator("catalog q")
        assert calls["constrain"] == ["5"] and calls["augment"] == ["5"]

        # relational route → neither fires (byte-identical homzhub behaviour)
        calls["constrain"].clear(); calls["augment"].clear()
        SC.plan_route = lambda q, sids, **k: RoutingDecision(
            status=STATUS_ROUTED, mode=MODE_SINGLE, source_ids=["2"],
            candidate_sources=[CandidateSource("2", source_type="relational")])
        veda_hybrid._run_coordinator("db q")
        assert calls["constrain"] == [] and calls["augment"] == []
    finally:
        veda_hybrid._constrain_scope_to = _orig_constrain
        veda_hybrid._augment_sm_for_datalake = _orig_augment


def _multi_decision(reason_code):
    return RoutingDecision(status=STATUS_ROUTED, mode=MODE_MULTI, source_ids=["5", "7"],
                           reason_code=reason_code,
                           candidate_sources=[CandidateSource("5", source_type="relational"),
                                              CandidateSource("7", source_type="relational")])


def test_multi_federated_defers_to_legacy():
    _setup()
    SC.plan_route = lambda q, sids, **k: _multi_decision(RC_RELATIONSHIP_EDGE)   # edge → federated
    assert veda_hybrid._run_coordinator("join") is None    # legacy federated path handles genuine joins


def test_multi_independent_append():
    _setup()
    SC.plan_route = lambda q, sids, **k: _multi_decision(RC_SLM_RESOLVED)         # no edge → independent
    SC.execute_decision = lambda d, q, **k: {"kind": "independent",
        "results": [AgentResult("5", "relational", "ok", data={"rows": [[100]], "answer": "a"}),
                    AgentResult("7", "relational", "ok", data={"rows": [["ev"]], "answer": "b"})],
        "merge": _merge(policy="APPEND"),
        "partial": {"failures": [], "any_required_failed": False, "ok_count": 2, "complete": True}}
    mr = veda_hybrid._run_coordinator("combined")
    assert len(mr.items) == 2 and mr.ok


def test_multi_conflict_refuses_not_blended():
    _setup()
    SC.plan_route = lambda q, sids, **k: _multi_decision(RC_SLM_RESOLVED)
    SC.execute_decision = lambda d, q, **k: {"kind": "independent",
        "results": [AgentResult("5", "relational", "ok", data={"rows": [[100]]}),
                    AgentResult("7", "relational", "ok", data={"rows": [[120]]})],
        "merge": _merge(policy="CONFLICT_DETECTED", needs_clarification=True,
                        conflict={"values": [{"source_id": "5", "value": 100},
                                             {"source_id": "7", "value": 120}]}),
        "partial": {"failures": [], "any_required_failed": False, "ok_count": 2, "complete": True}}
    mr = veda_hybrid._run_coordinator("revenue")
    assert mr.items[0].status == STATUS_REFUSED and "100" in mr.items[0].refuse_reason \
        and "120" in mr.items[0].refuse_reason


def test_multi_canonical_priority_winner_only():
    _setup()
    SC.plan_route = lambda q, sids, **k: _multi_decision(RC_SLM_RESOLVED)
    SC.execute_decision = lambda d, q, **k: {"kind": "independent",
        "results": [AgentResult("5", "relational", "ok", data={"rows": [[100]]}),
                    AgentResult("7", "relational", "ok", data={"rows": [[120]]})],
        "merge": _merge(policy="CANONICAL_PRIORITY", winner_source_id="5"),
        "partial": {"failures": [], "any_required_failed": False, "ok_count": 2, "complete": True}}
    mr = veda_hybrid._run_coordinator("revenue")
    assert len(mr.items) == 1 and mr.items[0].result.get("rows") == [[100]]


def test_multi_partial_failure_labelled():
    _setup()
    SC.plan_route = lambda q, sids, **k: _multi_decision(RC_SLM_RESOLVED)
    SC.execute_decision = lambda d, q, **k: {"kind": "independent",
        "results": [AgentResult("5", "relational", "ok", data={"rows": [[100]]}),
                    AgentResult("7", "relational", "failed", error="permission denied")],
        "merge": _merge(policy="APPEND"),
        "partial": {"failures": [{"source_id": "7", "required": True, "error": "permission denied",
                                  "failure_class": "permanent"}],
                    "any_required_failed": True, "ok_count": 1, "complete": False}}
    mr = veda_hybrid._run_coordinator("x")
    assert len(mr.items) == 2 and not mr.ok       # incomplete is visibly incomplete
    assert any(i.status == STATUS_ERROR for i in mr.items)


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except Exception:
            failed += 1; print("FAIL", fn.__name__); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
