"""A follow-up stays on the source its conversation is about (veda_hybrid._run_coordinator).

Measured 2026-09-24: with the user's words reaching the engine unchanged, a drill on
assets_asset (source 2) — "only the Nagpur ones", then "go back" — was scored as a bare
fragment against every ready source, an inaccessible one won, and the turn was answered
"You don't have permission to access this data" in 0.5s.

Providers/plan are monkeypatched → no DB/SLM. Run: `python -m pytest tests/test_conversation_source_pin.py`
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

import pytest  # noqa: E402

import config  # noqa: E402
import veda_hybrid  # noqa: E402
import query.source_coordinator as SC  # noqa: E402
from context import RequestContext, set_context  # noqa: E402
from query.agents import AgentResult  # noqa: E402
from query.multi_result import STATUS_OK  # noqa: E402
from query.routing_contracts import (  # noqa: E402
    RoutingDecision, STATUS_ROUTED, MODE_SINGLE, RC_CONVERSATION_SOURCE)

SQL_CONV = {"user_message": "only the Nagpur ones", "entity_table": "assets_asset",
            "source_id": 2, "route": "deterministic",
            "filters": [{"column": "furnishing", "operator": "equals", "value": "FULL"}]}


@pytest.fixture
def routed(monkeypatch):
    """Coordinator on + authoritative + permission pre-check on, caller permitted (2, 3),
    and an INACCESSIBLE source 9 that wins the global scoring — the measured failure."""
    monkeypatch.setattr(config, "MULTISOURCE_ROUTING_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "MULTISOURCE_ROUTING_SHADOW", False, raising=False)
    monkeypatch.setattr(config, "ROUTING_PERMISSION_PRECHECK_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "ROUTING_PERMISSION_DENY_GAP", 0.12, raising=False)
    monkeypatch.setattr(veda_hybrid, "_load_semantic_model", lambda: ({}, []))
    set_context(RequestContext(source_id=2, tenant="default", source_ids=(2, 3)))

    calls = {"plan": 0, "executed": []}
    monkeypatch.setattr(SC, "all_ready_source_ids", lambda: {"2", "3", "9"}, raising=False)
    monkeypatch.setattr(SC, "best_matching_scored",
                        lambda q, sids, prof: ("9", 0.9) if "9" in sids else ("2", 0.1),
                        raising=False)

    def _plan(q, sids, **k):
        calls["plan"] += 1
        return RoutingDecision(status=STATUS_ROUTED, mode=MODE_SINGLE, source_ids=["3"])
    monkeypatch.setattr(SC, "plan_route", _plan)

    def _exec(dec, q, **k):
        calls["executed"].append(dec)
        return {"kind": "single", "result": AgentResult(
            str(dec.source_ids[0]), "relational", "ok", engine="deterministic_sql",
            data={"cols": ["n"], "rows": [[1]], "answer": "ok"})}
    monkeypatch.setattr(SC, "execute_decision", _exec)
    yield calls
    veda_hybrid._set_conv(None)


def test_a_follow_up_stays_on_its_source_instead_of_being_refused(routed):
    veda_hybrid._set_conv(dict(SQL_CONV))
    mr = veda_hybrid._run_coordinator("only the Nagpur ones")
    assert mr is not None and mr.items[0].status == STATUS_OK      # not the permission refusal
    assert routed["plan"] == 0                                      # fragment never scored
    dec = routed["executed"][0]
    assert dec.source_ids == ["2"] and dec.reason_code == RC_CONVERSATION_SOURCE


def test_go_back_is_pinned_too(routed):
    veda_hybrid._set_conv(dict(SQL_CONV, user_message="go back", filters=[]))
    mr = veda_hybrid._run_coordinator("go back")
    assert mr.items[0].status == STATUS_OK
    assert routed["executed"][0].source_ids == ["2"]


# ── everything else routes exactly as before ─────────────────────────────────────────
def test_a_first_turn_routes_normally(routed):
    veda_hybrid._set_conv(None)
    mr = veda_hybrid._run_coordinator("only the Nagpur ones")
    assert mr.items[0].status != STATUS_OK                          # pre-check still refuses
    assert "permission" in (mr.items[0].refuse_reason or "")


def test_a_new_topic_carries_no_state_and_is_not_pinned(routed):
    veda_hybrid._set_conv({"user_message": "how many vendors are there"})
    veda_hybrid._run_coordinator("how many vendors are there")
    assert routed["executed"] == [] or routed["executed"][0].reason_code != RC_CONVERSATION_SOURCE


def test_a_revoked_source_is_never_pinned(routed):
    """The remembered source is no longer in the caller's scope → normal routing and its
    checks apply; the pin cannot be used to reach a source the caller lost."""
    veda_hybrid._set_conv(dict(SQL_CONV, source_id=9))
    mr = veda_hybrid._run_coordinator("only the Nagpur ones")
    assert "permission" in (mr.items[0].refuse_reason or "")
    assert all(d.source_ids != ["9"] for d in routed["executed"])


def test_a_document_conversation_is_left_to_normal_routing(routed):
    veda_hybrid._set_conv(dict(SQL_CONV, route="rag"))
    assert veda_hybrid._conversation_pinned_source(["2", "3"]) is None


def test_pin_helper_edges():
    try:
        veda_hybrid._set_conv({"entity_table": "assets_asset"})          # no source_id
        assert veda_hybrid._conversation_pinned_source(["2"]) is None
        veda_hybrid._set_conv({"source_id": 2})                          # no table
        assert veda_hybrid._conversation_pinned_source(["2"]) is None
        veda_hybrid._set_conv({"entity_table": "t", "source_id": "2"})   # str vs int scope
        assert veda_hybrid._conversation_pinned_source([2]) == "2"
    finally:
        veda_hybrid._set_conv(None)


# ── the opportunistic federated route honours the same pin ───────────────────────────
@pytest.fixture
def answer_path(monkeypatch):
    """_run_hybrid_query_inner with the coordinator in SHADOW (returns None, as deployed),
    a two-source scope, and every expensive edge stubbed. Records whether federation ran."""
    for flag in ("NL_SIMPLIFIER_ENABLED", "RUNTIME_CONTEXT_ENABLED", "QUERY_DECOMPOSE_ENABLED"):
        monkeypatch.setattr(config, flag, False, raising=False)
    set_context(RequestContext(source_id=2, tenant="default", source_ids=(2, 3)))
    seen = {"federated": 0, "dispatched": 0}
    monkeypatch.setattr(veda_hybrid, "_run_coordinator", lambda *a, **k: None)

    def _fed(*a, **k):
        seen["federated"] += 1
        return None
    monkeypatch.setattr(veda_hybrid, "_maybe_federated", _fed)

    def _dispatch(q, **k):
        seen["dispatched"] += 1
        return "sql", {"status": "answered", "ok": True, "answer": "ok"}
    monkeypatch.setattr(veda_hybrid, "_dispatch_single", _dispatch)
    yield seen
    veda_hybrid._set_conv(None)


def test_a_carried_follow_up_is_not_federated(answer_path):
    """Measured 2026-09-25: a carried "Nagpur" follow-up on assets_asset was answered by
    the federated route 1 run in 3, because the pin lived only in the SHADOW coordinator."""
    veda_hybrid._set_conv(dict(SQL_CONV, user_message="Nagpur", filters=[]))
    veda_hybrid._run_hybrid_query_inner("Nagpur")
    assert answer_path["federated"] == 0 and answer_path["dispatched"] == 1


def test_a_new_topic_may_still_federate(answer_path):
    veda_hybrid._set_conv({"user_message": "vendors and their assets"})
    veda_hybrid._run_hybrid_query_inner("vendors and their assets")
    assert answer_path["federated"] == 1


def test_a_revoked_source_is_not_pinned_here_either(answer_path):
    veda_hybrid._set_conv(dict(SQL_CONV, source_id=9))
    veda_hybrid._run_hybrid_query_inner("only the Nagpur ones")
    assert answer_path["federated"] == 1          # normal behaviour, no pin to a lost source


# ── the document lane (2026-09-25, demo X1 / D3b) ─────────────────────────────────────
DOC_CONV = {"user_message": "what is the late fee? (in msa green tower)",
            "entity_table": "msa green tower", "source_id": 2, "route": "rag"}


def test_a_document_follow_up_is_not_federated(answer_path, monkeypatch):
    monkeypatch.setattr(veda_hybrid, "_scope_has_doc_source", lambda: True)
    veda_hybrid._set_conv(dict(DOC_CONV))
    veda_hybrid._run_hybrid_query_inner("what is the late fee? (in msa green tower)")
    assert answer_path["federated"] == 0 and answer_path["dispatched"] == 1


@pytest.mark.parametrize("q,lane", [("what is the late fee for invoices?", "rag"),
                                    ("what is the total late fee?", "hybrid")])
def test_a_document_follow_up_stays_on_the_document_lane(monkeypatch, q, lane):
    """A money word no longer sends it to a structured table."""
    monkeypatch.setattr(veda_hybrid, "_scope_has_doc_source", lambda: True)
    veda_hybrid._set_conv(dict(DOC_CONV))
    try:
        assert veda_hybrid.classify(q)[0] == lane
    finally:
        veda_hybrid._set_conv(None)


def test_no_document_source_in_scope_no_document_lane(monkeypatch):
    monkeypatch.setattr(veda_hybrid, "_scope_has_doc_source", lambda: False)
    veda_hybrid._set_conv(dict(DOC_CONV))
    try:
        assert veda_hybrid._conversation_document_lane() is False
    finally:
        veda_hybrid._set_conv(None)


def test_a_new_topic_has_no_document_lane():
    veda_hybrid._set_conv({"user_message": "what is the late fee?"})
    try:
        assert veda_hybrid._conversation_document_lane() is False
    finally:
        veda_hybrid._set_conv(None)
