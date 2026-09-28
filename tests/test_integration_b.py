"""Section B of the 2026-09-26 wiring pass: one frame extraction and one routing decision.

  B.1  the compound front door's single-intent frame is held for the SQL head (no second
       extraction), cleared on every other route, never leaks past the request
  B.2  the coordinator: no plan_route when nothing can be authoritative; an authoritative
       SINGLE narrows the scope and falls through to the normal path
  B.3  classify's doc-intent check reuses the coordinator's evidence (one evidence pass)
  B.4  route_query scores the request scope, not the config registry

No SLM, no DB: extractors, evidence providers and plan_route are monkeypatched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "veda_core"
for p in (str(ROOT), str(CORE)):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CORE))

import config as cfg  # noqa: E402
import context as ctxmod  # noqa: E402
import veda_hybrid as vh  # noqa: E402
import query.source_coordinator as SC  # noqa: E402
from query.source_evidence import SourceEvidence  # noqa: E402
from query.routing_contracts import (  # noqa: E402
    RoutingDecision, STATUS_ROUTED, STATUS_NO_MATCH, MODE_SINGLE, MODE_MULTI, CandidateSource)
from veda.understanding.frame import Frame, Intents  # noqa: E402
from veda.understanding import frame_path as FP  # noqa: E402
import veda.explain as X  # noqa: E402

@pytest.fixture(autouse=True)
def _reuse_on(monkeypatch):
    """These tests exercise the B.1 reuse MECHANISM; it ships OFF by default (see
    test_front_door_reuse_is_off_by_default), so switch it on here."""
    monkeypatch.setattr(cfg, "FRONT_DOOR_FRAME_REUSE", True, raising=False)


PROFILES = {"2": {"name": "homzhub", "source_type": "relational"},
            "3": {"name": "docs", "source_type": "document"},
            "4": {"name": "invoices", "source_type": "datalake"}}


@pytest.fixture(autouse=True)
def _scope(monkeypatch):
    ctxmod.set_context(ctxmod.RequestContext(source_id=2, tenant="default", source_ids=(2, 3, 4)))
    ctxmod.set_source_profiles(PROFILES)
    ctxmod.set_conversation_context(None)
    vh._reset_request_state()
    yield
    vh._reset_request_state()


@pytest.fixture
def trace():
    tr = X.new_trace("q")
    with X.use_trace(tr):
        yield tr


# ── B.1 ─────────────────────────────────────────────────────────────────────────────
class _G:
    def __init__(self, kind="sql", outcome="grounded", source_id="2"):
        self.kind, self.outcome, self.source_id = kind, outcome, source_id
        self.part, self.entity, self.entity_name, self.method = "q", "t", "T", "NAME"
        self.depends_on, self.evidence = None, {}


def _front_door(monkeypatch, frames, groundings, plan="single"):
    import veda.understanding.vocabulary as V
    import veda.understanding.frame_extractor as FE
    import veda.understanding.compound as C
    monkeypatch.setattr(cfg, "FRAME_PATH_ENABLED", True)
    monkeypatch.setattr(V, "front_door_vocab", lambda *a, **k: type("V", (), {"cards": {"t": {}}})())
    monkeypatch.setattr(V, "doc_cards_of", lambda v: [])
    monkeypatch.setattr(FE, "extract_intents",
                        lambda q, v, doc_cards=None, stats=None: Intents(intents=list(frames)))
    monkeypatch.setattr(C, "ground_intents", lambda its, v, source_names=None: list(groundings))
    monkeypatch.setattr(C, "plan_compound", lambda its, gs, segs: plan)


def test_single_intent_data_frame_is_held_for_this_query_only(monkeypatch, trace):
    fr = Frame(entity="property", part="how many properties")
    _front_door(monkeypatch, [fr], [_G()])
    assert vh._maybe_compound("how many properties") is None
    assert trace.sections["frame_path"]["front_door"]["inject"] == "held"
    assert vh._take_front_door_frame("a different question") is None     # cleared, not returned
    _front_door(monkeypatch, [fr], [_G()])
    vh._maybe_compound("how many properties")
    held = vh._take_front_door_frame("how many properties")
    assert held is fr and held.provenance.get("_front_door_single") is True
    assert vh._take_front_door_frame("how many properties") is None       # consume-once


@pytest.mark.parametrize("g,why", [(_G(kind="rag", source_id="3"), "skipped:lane_rag"),
                                   (_G(outcome="clarify"), "skipped:clarify"),
                                   (_G(outcome="degrade", kind=None), "skipped:lane_none")])
def test_rag_clarify_or_ungrounded_frame_is_not_held(monkeypatch, trace, g, why):
    _front_door(monkeypatch, [Frame(entity="x", part="q")], [g])
    vh._maybe_compound("q")
    assert trace.sections["frame_path"]["front_door"]["inject"] == why
    assert vh._take_front_door_frame("q") is None


def test_multi_frame_single_is_not_held(monkeypatch, trace):
    """Two frames on one (source, lane) are one join question's clauses: frame[0] alone would
    lose half of it, so the SQL head extracts the whole question itself."""
    _front_door(monkeypatch, [Frame(entity="property", part="properties"),
                              Frame(entity="payment", part="their payments")],
                [_G(), _G()], plan="single")
    assert vh._maybe_compound("properties and their payments") is None
    assert trace.sections["frame_path"]["front_door"]["inject"] == "skipped:multi_frame_single"
    assert vh._take_front_door_frame("properties and their payments") is None


def _dispatch_env(monkeypatch, intent):
    monkeypatch.setattr(vh, "classify", lambda q, verbose=False: (intent, None))
    monkeypatch.setattr(vh, "_load_semantic_model", lambda: ({"tables": {"t": {}}}, []))
    monkeypatch.setattr(cfg, "TIER2_LLM_FALLBACK", False)
    seen = {}
    import veda.pipeline as VP

    def _rq(query, sm, cols, **k):
        seen["injected"] = FP.injected_frame()
        return {"ok": True, "status": "answered", "rows": [], "cols": []}
    monkeypatch.setattr(VP, "run_query", _rq)
    return seen


def test_sql_head_runs_with_the_front_door_frame_injected(monkeypatch):
    seen = _dispatch_env(monkeypatch, "sql")
    fr = Frame(entity="property")
    vh._FRONT_DOOR_FRAME.set(("q", fr))
    vh._dispatch_single_inner("q")
    assert seen["injected"] is fr
    assert FP.injected_frame() is None                  # reset after the head: Tier-2 never sees it
    assert vh._FRONT_DOOR_FRAME.get() is None


def test_rag_route_clears_the_held_frame(monkeypatch):
    _dispatch_env(monkeypatch, "rag")
    import query.rag_layer as RL
    monkeypatch.setattr(RL, "run_rag_layer", lambda *a, **k: type("R", (), {"error": None, "answer": "a",
                                                                          "citations": []})())
    vh._FRONT_DOOR_FRAME.set(("q", Frame(entity="x")))
    route, _ = vh._dispatch_single_inner("q")
    assert route == "rag" and vh._FRONT_DOOR_FRAME.get() is None


def test_sub_query_does_not_inherit_the_whole_message_frame(monkeypatch):
    seen = _dispatch_env(monkeypatch, "sql")
    vh._FRONT_DOOR_FRAME.set(("whole message", Frame(entity="x")))
    vh._dispatch_single_inner("one part of it")
    assert seen["injected"] is None


def test_request_reset_drops_state_left_by_an_earlier_request():
    vh._FRONT_DOOR_FRAME.set(("q", Frame(entity="x")))
    vh._ROUTED_SM.set(("4", {}, []))
    vh._reset_request_state()
    assert vh._FRONT_DOOR_FRAME.get() is None and vh._ROUTED_SM.get() is None


# ── B.2 ─────────────────────────────────────────────────────────────────────────────
def _coord(monkeypatch, modes, decision):
    monkeypatch.setattr(cfg, "MULTISOURCE_ROUTING_ENABLED", True)
    monkeypatch.setattr(cfg, "MULTISOURCE_ROUTING_SHADOW", True)
    monkeypatch.setattr(cfg, "ROUTING_AUTHORITATIVE_MODES", tuple(modes))
    monkeypatch.setattr(cfg, "ROUTING_SHADOW_OBSERVE", False)
    monkeypatch.setattr(cfg, "ROUTING_PERMISSION_PRECHECK_ENABLED", False)
    monkeypatch.setattr(vh, "_load_semantic_model", lambda: ({}, []))
    calls = []

    def _pr(q, sids, **k):
        calls.append(q)
        return decision
    monkeypatch.setattr(SC, "plan_route", _pr)
    ran = []
    monkeypatch.setattr(SC, "execute_decision", lambda *a, **k: ran.append(1))
    return calls, ran


def test_no_authoritative_mode_means_no_plan_route(monkeypatch, trace):
    calls, _ = _coord(monkeypatch, (), RoutingDecision(status=STATUS_NO_MATCH))
    assert vh._run_coordinator("q") is None
    assert calls == []
    r = trace.sections["routing"]
    assert r["plan_route_ran"] is False and r["decision_consumed"] is False


def test_observe_flag_keeps_the_shadow_decision(monkeypatch, trace):
    calls, _ = _coord(monkeypatch, (), RoutingDecision(status=STATUS_ROUTED, mode=MODE_SINGLE,
                                                       source_ids=["2"]))
    monkeypatch.setattr(cfg, "ROUTING_SHADOW_OBSERVE", True)
    assert vh._run_coordinator("q") is None
    assert calls == ["q"] and trace.sections["routing"]["shadow_discarded"] is True
    assert tuple(vh._current_ctx().source_ids) == (2, 3, 4)       # shadow never narrows


def test_authoritative_single_narrows_scope_and_falls_through(monkeypatch, trace):
    calls, ran = _coord(monkeypatch, ("SINGLE",), RoutingDecision(
        status=STATUS_ROUTED, mode=MODE_SINGLE, source_ids=["2"],
        candidate_sources=[CandidateSource("2", source_type="relational")]))
    assert vh._run_coordinator("q") is None                 # → the normal path answers
    assert ran == []                                        # no source-agent dispatch
    assert tuple(vh._current_ctx().source_ids) == (2,)
    r = trace.sections["routing"]
    assert r["decision_consumed"] is True and r["consumed_as"] == "scope"
    assert r["shadow_discarded"] is False
    assert vh._maybe_federated("q") is None                 # one source: federation is a no-op


def test_authoritative_single_on_a_datalake_hands_the_head_its_model(monkeypatch, trace):
    _coord(monkeypatch, ("SINGLE",), RoutingDecision(
        status=STATUS_ROUTED, mode=MODE_SINGLE, source_ids=["4"],
        candidate_sources=[CandidateSource("4", source_type="datalake")]))
    iso = ({"tables": {"invoices": {}}, "__source_isolated__": True}, ["invoices.amount"])
    monkeypatch.setattr(cfg, "SOURCE_ISOLATED_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(vh, "_datalake_isolated_sm", lambda sid: iso)
    vh._run_coordinator("q")
    assert vh._head_semantic_model() == iso
    # a different scope (a later compound part restored the full scope) does not read it
    ctxmod.set_context(ctxmod.RequestContext(source_id=2, tenant="default", source_ids=(2, 3, 4)))
    assert vh._head_semantic_model() == ({}, [])


@pytest.mark.parametrize("decision", [
    RoutingDecision(status=STATUS_NO_MATCH, reason="none"),
    RoutingDecision(status="CLARIFICATION_REQUIRED", reason="which?"),
    RoutingDecision(status=STATUS_ROUTED, mode=MODE_MULTI, source_ids=["2", "4"])])
def test_only_single_authoritative_never_refuses_or_steers_others(monkeypatch, trace, decision):
    _coord(monkeypatch, ("SINGLE",), decision)
    assert vh._run_coordinator("q") is None
    assert tuple(vh._current_ctx().source_ids) == (2, 3, 4)
    assert trace.sections["routing"]["shadow_discarded"] is True


# ── B.3 ─────────────────────────────────────────────────────────────────────────────
def _evidence_providers(monkeypatch):
    n = {"ev": 0}

    def _ev(q, sids):
        n["ev"] += 1
        return [], []

    def _group(cols, chunks):
        return {"2": SourceEvidence("2", top_column_score=0.50),
                "3": SourceEvidence("3", top_chunk_score=0.80),
                "4": SourceEvidence("4", top_column_score=0.40)}
    monkeypatch.setattr(SC, "_default_evidence_provider", _ev)
    monkeypatch.setattr(SC, "group_evidence_by_source", _group)
    monkeypatch.setattr(SC, "_default_item_prior_provider", lambda q, s: {})
    return n


def test_one_evidence_pass_per_query_and_scope(monkeypatch, trace):
    n = _evidence_providers(monkeypatch)
    a = SC.routing_evidence("q", ["2", "3", "4"])
    b = SC.routing_evidence("q", [4, 3, 2])                  # ints / order: the same scope
    assert a is b and n["ev"] == 1 and SC.evidence_passes() == 1
    SC.routing_evidence("another q", ["2", "3", "4"])
    assert n["ev"] == 2
    assert trace.sections["routing"]["evidence_passes"] == 2


def test_narrowed_scope_reuses_the_wider_pass(monkeypatch):
    n = _evidence_providers(monkeypatch)
    SC.routing_evidence("q", ["2", "3", "4"])
    sub = SC.routing_evidence("q", ["2"])
    assert n["ev"] == 1 and list(sub) == ["2"] and sub["2"].presence_tier == "STRONG"


def test_classify_doc_intent_reuses_the_coordinators_evidence(monkeypatch, trace):
    n = _evidence_providers(monkeypatch)
    monkeypatch.setattr(cfg, "DOC_INTENT_EVIDENCE_ENABLED", True)
    SC.routing_evidence("what is the late fee", ["2", "3", "4"])   # the coordinator's pass
    assert vh._doc_intent_by_evidence("what is the late fee") is True
    assert n["ev"] == 1
    assert trace.sections["routing"]["evidence_passes"] == 1


def test_classify_without_coordinator_computes_once_and_caches(monkeypatch):
    n = _evidence_providers(monkeypatch)
    monkeypatch.setattr(cfg, "DOC_INTENT_EVIDENCE_ENABLED", True)
    vh._doc_intent_by_evidence("q")
    vh._doc_intent_by_evidence("q")
    assert n["ev"] == 1


def test_injected_providers_bypass_the_cache(monkeypatch):
    n = _evidence_providers(monkeypatch)
    calls = []
    SC.routing_evidence("q", ["2"], evidence_provider=lambda q, s: calls.append(1) or ([], []))
    SC.routing_evidence("q", ["2"], evidence_provider=lambda q, s: calls.append(1) or ([], []))
    assert len(calls) == 2 and n["ev"] == 0 and SC.evidence_passes() == 0


# ── B.4 ─────────────────────────────────────────────────────────────────────────────
def test_router_scores_the_request_scope(monkeypatch):
    import query.query_router as QR
    monkeypatch.setattr(QR, "QUERY_ROUTER_ENABLED", True)
    monkeypatch.setattr(QR, "_check_value_filter", lambda q: False)
    r = QR.route_query("what does the leave policy say")
    assert r.intent == "rag" and r.source_ids == ["3"]
    ctxmod.set_context(ctxmod.RequestContext(source_id=2, tenant="default", source_ids=(2,)))
    r = QR.route_query("what does the leave policy say")
    assert r.intent == "sql"                          # no document source in THIS scope


def test_router_without_context_uses_the_registry(monkeypatch):
    import query.query_router as QR
    monkeypatch.setattr(QR, "_request_scope_sources", lambda: None)
    import config as C
    monkeypatch.setattr(C, "get_enabled_sources", lambda *a: [{"id": "9", "type": "relational"}])
    r = QR.route_query("what does the leave policy say")
    assert r.intent == "sql" and r.source_ids == ["9"]


def test_router_keywords_match_whole_words_only():
    import query.query_router as QR
    assert QR._count_signal_hits("sort by their respective currency", {"spec"}) == 0
    assert QR._count_signal_hits("the product spec sheet", {"spec"}) == 1


def test_federation_qualification_gate_reuses_the_pass(monkeypatch):
    """The federated qualification gate was a third evidence retrieval in the same turn."""
    n = _evidence_providers(monkeypatch)
    from query.cross_source_composer import _clean_source_scores
    SC.routing_evidence("q", ["2", "3", "4"])
    scores = _clean_source_scores("q", ["2", "3", "4"])
    assert n["ev"] == 1 and scores["2"] == 0.50


# ── classify.lane: the rule that chose the head is recorded per turn ────────────────────────
def _lane_of(monkeypatch, query, conv=None, doc_ref=False, doc_ev=False, router_on=False,
             router_intent="sql", doc_primary=False):
    monkeypatch.setattr(vh, "_cur_conv", lambda: conv or {})
    monkeypatch.setattr(vh, "_scope_has_doc_source", lambda: doc_ref)
    monkeypatch.setattr(vh, "_scope_has_structured_source", lambda: True)
    monkeypatch.setattr(vh, "_doc_intent_by_evidence", lambda q: doc_ev)
    monkeypatch.setattr(vh, "_primary_is_document_source", lambda: doc_primary)
    monkeypatch.setattr(cfg, "QUERY_ROUTER_ENABLED", router_on, raising=False)
    import query.query_router as QR

    class _R:
        intent, source_ids = router_intent, None
    monkeypatch.setattr(QR, "route_query", lambda q, verbose=False: _R())
    tr = X.new_trace(query)
    with X.use_trace(tr):
        intent, _ = vh.classify(query)
    return intent, (tr.sections.get("classify") or {}).get("lane"), tr


def test_classify_lane_continuity(monkeypatch):
    intent, lane, tr = _lane_of(monkeypatch, "only the ones in Pune",
                                conv={"entity_table": "assets_asset", "route": "deterministic"})
    assert (intent, lane) == ("sql", "continuity")
    assert tr.compact()["classify_lane"] == "continuity"


def test_classify_lane_doc_ref_and_evidence(monkeypatch):
    assert _lane_of(monkeypatch, "what does the document say about late fees",
                    doc_ref=True)[1] == "doc_ref"
    assert _lane_of(monkeypatch, "late fee percentage", doc_ev=True)[1] == "doc_evidence"


def test_classify_lane_router_default_and_guard(monkeypatch):
    assert _lane_of(monkeypatch, "how many assets", router_on=True)[1] == "router"
    assert _lane_of(monkeypatch, "how many assets", router_on=False)[1] == "default_sql"
    intent, lane, _ = _lane_of(monkeypatch, "how many assets", router_on=False, doc_primary=True)
    assert (intent, lane) == ("rag", "guard_rag")


def test_guard_rag_drops_the_routers_sql_source_ids(monkeypatch):
    """FS1 (2026-09-27): router said sql over ['2']; the doc-primary guard demoted it to rag
    but kept ['2'], so RAG searched a source with no documents."""
    monkeypatch.setattr(vh, "_cur_conv", lambda: {})
    monkeypatch.setattr(vh, "_scope_has_doc_source", lambda: False)
    monkeypatch.setattr(vh, "_doc_intent_by_evidence", lambda q: False)
    monkeypatch.setattr(vh, "_primary_is_document_source", lambda: True)
    monkeypatch.setattr(cfg, "QUERY_ROUTER_ENABLED", True, raising=False)
    import query.query_router as QR

    class _R:
        intent, source_ids = "sql", ["2"]
    monkeypatch.setattr(QR, "route_query", lambda q, verbose=False: _R())
    assert vh.classify("what is the late fee percentage") == ("rag", None)


def test_router_types_dialect_named_profiles():
    import query.query_router as QR
    import contextvars
    from veda_core.context import RequestContext, set_context, set_source_profiles

    def _probe():
        set_source_profiles({"2": {"source_type": "relational"}, "3": {"source_type": "filesystem"},
                             "4": {"source_type": "csv_lake"}, "5": {"source_type": "parquet"}})
        set_context(RequestContext(source_id=3, tenant="default", source_ids=(2, 3, 4, 5)))
        return {d["id"]: d["type"] for d in (QR._request_scope_sources() or [])}
    got = contextvars.copy_context().run(_probe)     # no context leaks into other tests
    assert got == {"2": "relational", "3": "document", "4": "datalake", "5": "datalake"}


def test_front_door_reuse_is_off_by_default(monkeypatch):
    """Measured 2026-09-27: the front-door frame drops value filters the head's own
    extraction keeps ("vendors in Kochi" -> no WHERE), so reuse ships OFF."""
    import importlib
    monkeypatch.delenv("FRONT_DOOR_FRAME_REUSE", raising=False)
    fresh = importlib.reload(cfg)
    try:
        assert fresh.FRONT_DOOR_FRAME_REUSE is False
        from veda.understanding.compound import GROUNDED

        class _G:
            kind, outcome = "sql", GROUNDED
        assert vh._keep_front_door_frame("q", Frame(entity="x"), _G()) == "skipped:reuse_off"
        assert vh._FRONT_DOOR_FRAME.get() is None
    finally:
        importlib.reload(cfg)
