"""Wiring pass C (2026-09-26): the frame path's router hint, the agent's part budget, the
agent's split / rag plans, and a multi-source agent plan narrowing the request scope.

No SLM, no DB: the planner agent and the part runners are stubs. Run:
    python -m pytest tests/test_integration_c.py -q
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "veda_core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import pytest  # noqa: E402

from test_frame_path import SM as FSM, _vocab  # noqa: E402
from veda.understanding import frame_path as FP  # noqa: E402
from veda.understanding.frame import Frame  # noqa: E402


@pytest.fixture
def flags(monkeypatch):
    import config as _cfg
    import veda.understanding.vocabulary as VOC
    monkeypatch.setattr(_cfg, "FRAME_PROBES_ENABLED", False, raising=False)
    monkeypatch.setattr(_cfg, "AGENT_PLANNER_ENABLED", True, raising=False)
    monkeypatch.setattr(VOC, "scope_vocab", lambda sm, *a, **k: _vocab())
    return _cfg


def _model_frame():
    # the question names no entity; the extractor's pick is the only evidence → MODEL
    fr = Frame(entity="thing", aggregation="count")
    fr.provenance["entity_table"] = "listing"
    return fr


def _run(query, fr, **kw):
    tok = FP.inject_frame(fr)
    try:
        return FP.run_frame_path(query, FSM, **kw)
    finally:
        FP.reset_injected(tok)


# ── C.1 router hint ────────────────────────────────────────────────────────────────────
def test_model_anchor_agreeing_with_router_primary_answers(flags, monkeypatch):
    monkeypatch.setattr(FP, "_question_evidence", lambda a, q, sm: "column:status")
    calls = []

    def hint():
        calls.append(1)
        return "listing", [("listing", 0.91), ("ledger", 0.4)]
    res = _run("how many are there", _model_frame(), router_hint=hint)
    assert res.kind == "sql", (res.kind, res.reason)
    assert res.grounded.anchor_method == "MODEL" and res.anchor == "listing"
    assert calls == [1]                                   # asked once
    assert res.trace["router"]["primary"] == "listing"
    assert res.trace["router"]["scores"][0] == ("listing", 0.91)
    assert res.trace["grounding"]["router_agrees"] is True
    assert res.trace["grounding"]["router_evidence"] == "column:status"


def test_agreement_without_question_evidence_degrades(flags, monkeypatch):
    """'how many gadgets are there' on a one-table source: the router's primary is the only
    table, so agreeing with it is no evidence — the old chain's typed clarify stands."""
    monkeypatch.setattr(FP, "_question_evidence", lambda a, q, sm: None)
    res = _run("how many are there", _model_frame(), router_hint=lambda: ("listing", []))
    assert res.kind == "degrade" and res.reason == "advisory_anchor_unevidenced:MODEL"


def test_question_evidence_names_a_column():
    assert FP._question_evidence("listing", "average expected price", FSM) == "column:expected_price"
    assert FP._question_evidence("listing", "how many gadgets are there", FSM) is None


def test_model_anchor_disagreeing_with_router_primary_degrades(flags, monkeypatch):
    import veda.agent.planner as APL
    monkeypatch.setattr(APL, "run_planner", lambda *a, **k: (_ for _ in ()).throw(AssertionError("agent")))
    res = _run("how many are there", _model_frame(), router_hint=lambda: ("ledger", []))
    assert res.kind == "degrade" and res.reason == "advisory_anchor:MODEL"


def test_name_grounded_anchor_never_asks_the_router(flags):
    def hint():
        raise AssertionError("retrieval paid on a NAME-grounded turn")
    fr = Frame(entity="sale listing", aggregation="count")
    res = _run("How many sale listings are there", fr, router_hint=hint)
    assert res.kind == "sql" and res.grounded.anchor_method in ("NAME", "NAME+COVERAGE")
    assert "router" not in res.trace


def test_router_hint_failure_degrades_like_no_router(flags):
    def hint():
        raise RuntimeError("retrieval down")
    res = _run("how many are there", _model_frame(), router_hint=hint)
    assert res.kind == "degrade"
    assert "RuntimeError" in res.trace["router"]["error"]


def test_router_primary_maps_bare_table_onto_scope_key():
    v = _vocab()
    v.cards = {"src4.vendors": {"_bare_table": "vendors"}, "listing": {"_bare_table": "listing"}}
    tr = {}
    rp = FP._RouterPrimary(None, lambda: ("vendors", None), v, tr)
    assert rp() == "src4.vendors" and rp() == "src4.vendors" and tr["router"]["primary"] == "src4.vendors"


# ── C.2 the agent's wall is what is left of the part ──────────────────────────────────
def test_agent_wall_is_capped_by_the_part_deadline(monkeypatch):
    import config as _cfg
    from slm._call_slm import slm_deadline
    monkeypatch.setattr(_cfg, "AGENT_PART_BUDGET_S", 90.0, raising=False)
    monkeypatch.setattr(_cfg, "AGENT_TAIL_RESERVE_S", 10.0, raising=False)
    monkeypatch.setattr(_cfg, "AGENT_MIN_WALL_S", 8.0, raising=False)
    assert FP._agent_wall() == 90.0                       # no part deadline: the ceiling
    tok = slm_deadline.set(time.time() + 40.0)
    try:
        assert 29.0 <= FP._agent_wall() <= 30.0           # 40 left − 10 reserve
    finally:
        slm_deadline.reset(tok)
    tok = slm_deadline.set(time.time() + 12.0)
    try:
        assert FP._agent_wall() is None                   # 2 s is no time to plan
    finally:
        slm_deadline.reset(tok)


def test_agent_skipped_when_part_budget_is_spent(flags, monkeypatch):
    import veda.agent.planner as APL
    from slm._call_slm import slm_deadline
    monkeypatch.setattr(APL, "run_planner", lambda *a, **k: (_ for _ in ()).throw(AssertionError("agent")))
    res0 = FP.FramePathResult("clarify", "slot:x", message="which?")
    tok = slm_deadline.set(time.time() + 5.0)
    try:
        out = FP._agent_or(res0, "q", FSM, _vocab(), None, "clarify:x", {})
    finally:
        slm_deadline.reset(tok)
    assert out is res0


def test_agent_gets_the_derived_wall(flags, monkeypatch):
    import veda.agent.planner as APL
    from slm._call_slm import slm_deadline
    seen = {}

    def fake(q, sm, vocab, **k):
        seen.update(k)
        return APL.AgentResult(kind="fail", reason="stub")
    monkeypatch.setattr(APL, "run_planner", fake)
    tok = slm_deadline.set(time.time() + 50.0)
    try:
        FP._agent_or(FP.FramePathResult("clarify", "x", message="m"), "q", FSM, _vocab(), None, "x", {})
    finally:
        slm_deadline.reset(tok)
    assert 39.0 <= seen["wall_s"] <= 40.0


# ── C.3 split / rag plans are consumed, not dropped ───────────────────────────────────
def _stub_agent(monkeypatch, result):
    import veda.agent.planner as APL
    monkeypatch.setattr(APL, "run_planner", lambda *a, **k: result)
    return APL


def test_split_plan_becomes_a_split_result(flags, monkeypatch):
    import veda.agent.planner as APL
    _stub_agent(monkeypatch, APL.AgentResult(kind="split", parts=["how many listings", "how many ledger entries"],
                                             reason="model_split"))
    tr = {}
    out = FP._agent_or(FP.FramePathResult("clarify", "x", message="m"), "q", FSM, _vocab(), None, "x", tr)
    assert out.kind == "split" and out.parts == ["how many listings", "how many ledger entries"]
    assert tr["agent"]["consumed"] == "split"


def test_split_inside_a_split_keeps_the_frame_result(flags, monkeypatch):
    import veda.agent.planner as APL
    _stub_agent(monkeypatch, APL.AgentResult(kind="split", parts=["a", "b"], reason="model_split"))
    res0 = FP.FramePathResult("clarify", "x", message="m")
    tr = {}
    tok = FP.enter_split()
    try:
        out = FP._agent_or(res0, "q", FSM, _vocab(), None, "x", tr)
    finally:
        FP.exit_split(tok)
    assert out is res0 and tr["agent"]["refused"] == "split_inside_split"


def test_rag_plan_goes_to_the_document_sources(flags, monkeypatch):
    import veda.agent.planner as APL
    _stub_agent(monkeypatch, APL.AgentResult(kind="rag", reason="document question"))
    monkeypatch.setattr(FP, "_doc_sources", lambda scope: ["3"])
    out = FP._agent_or(FP.FramePathResult("clarify", "x", message="m"), "q", FSM, _vocab(), None, "x",
                       {"scope_ids": ["2", "3"]})
    assert out.kind == "rag" and out.rag_sources == ["3"]


def test_rag_plan_without_a_document_source_keeps_the_frame_result(flags, monkeypatch):
    import veda.agent.planner as APL
    _stub_agent(monkeypatch, APL.AgentResult(kind="rag", reason="document question"))
    monkeypatch.setattr(FP, "_doc_sources", lambda scope: [])
    res0 = FP.FramePathResult("clarify", "x", message="m")
    tr = {"scope_ids": ["2"]}
    assert FP._agent_or(res0, "q", FSM, _vocab(), None, "x", tr) is res0
    assert tr["agent"]["refused"] == "rag_without_document_source"


# ── C.4 multi-source: a plan on another in-scope source runs there ────────────────────
class _Plan:
    def __init__(self, sid, anchor="listing"):
        self.source_id, self.anchor = sid, anchor
        self.order, self.aggregates, self.filters, self.group_by = [], [], [], []
        self.projection, self.limit, self.distinct, self.evidence, self.confidence = [], None, False, {}, 0.9
        self.tables, self.joins, self.time = [anchor], [], None

    def to_dict(self):
        return {"anchor": self.anchor, "source_id": self.source_id}


class _Comp:
    def __init__(self, tables):
        self.sql, self.ir, self.tables, self.columns = "SELECT 1", None, tables, []


def test_plan_on_other_in_scope_source_is_kept_with_its_source(flags, monkeypatch):
    import veda.agent.planner as APL
    v = _vocab()
    v.source_of = {"listing": "4", "ledger": "2", "prop": "2"}
    _stub_agent(monkeypatch, APL.AgentResult(kind="sql", plan=_Plan("4"), compiled=_Comp(["listing"])))
    tr = {"scope_ids": ["2", "4"]}
    out = FP._agent_or(FP.FramePathResult("clarify", "x", message="m"), "q", FSM, v, None, "x", tr)
    assert out.kind == "sql" and out.source_id == "4" and tr["agent"]["source_id"] == "4"


def test_plan_spanning_sources_is_refused(flags, monkeypatch):
    import veda.agent.planner as APL
    v = _vocab()
    v.source_of = {"listing": "4", "ledger": "2", "prop": "2"}
    _stub_agent(monkeypatch, APL.AgentResult(kind="sql", plan=_Plan("4"), compiled=_Comp(["listing", "ledger"])))
    res0 = FP.FramePathResult("clarify", "x", message="m")
    tr = {"scope_ids": ["2", "4"]}
    assert FP._agent_or(res0, "q", FSM, v, None, "x", tr) is res0
    assert tr["agent"]["refused"].startswith("plan_spans_sources")


def test_narrowed_scope_keeps_rbac_and_restores():
    from veda_core.context import RequestContext, set_context, try_current
    base = RequestContext(source_id=2, tenant="t", source_ids=(2, 4), allowed_resources=(("x",),),
                          cache_back=False)
    set_context(base)
    try:
        with FP.narrowed_scope("4"):
            c = try_current()
            assert c.source_id == 4 and c.source_ids == (4,)
            assert c.allowed_resources == base.allowed_resources and c.cache_back is False
        assert try_current() is base
    finally:
        set_context(None)


def test_run_query_restores_a_scope_narrowed_inside_it(monkeypatch):
    from veda import pipeline
    from veda_core.context import RequestContext, set_context, try_current
    base = RequestContext(source_id=2, tenant="t", source_ids=(2, 4))

    def inner(*a, **k):
        set_context(base.narrowed(4))
        return {"status": "answered"}
    monkeypatch.setattr(pipeline, "_run_query", inner)
    set_context(base)
    try:
        assert pipeline.run_query("q", {}, [], return_result=True) == {"status": "answered"}
        assert try_current() is base
    finally:
        set_context(None)


# ── C.3 end to end in veda_hybrid: a split part becomes extra parts ───────────────────
def _hy():
    import veda_hybrid as H
    return H


def test_compound_part_split_yields_extra_parts(monkeypatch):
    H = _hy()
    from query.multi_result import SubResult, STATUS_OK, STATUS_REFUSED

    class G:
        def __init__(self, part, depends_on=None):
            self.part, self.kind, self.source_id, self.depends_on = part, "sql", "2", depends_on

    class Its:
        intents = [Frame(), Frame()]
        relation = "dependent"

    gs = [G("listings and ledger totals"), G("and for those, the city", depends_on=0)]
    seen_parent = {}

    def fake_part(fr, g, parent, b, verbose=False, on_event=None):
        if g.depends_on is None:
            return SubResult(g.part, STATUS_REFUSED, "agent_split",
                             {"status": "split", "msg": "This asks 2 things: a; b",
                              "split_parts": ["how many listings", "total ledger amount"]},
                             "split", part=g.part, outcome="refused", lane="sql", source_id="2")
        seen_parent["p"] = parent
        return SubResult(g.part, STATUS_REFUSED, "deterministic", {"status": "refuse"}, "x",
                         part=g.part, outcome="refused", depends_on=g.depends_on)

    ran = []

    def fake_sub(sq, verbose=False, on_event=None, index=None, total=None):
        from veda.understanding import frame_path as _fp
        ran.append((sq, _fp._SPLIT_DEPTH.get()))
        return SubResult(sq, STATUS_OK, "deterministic", {"status": "answered", "ok": True, "answer": "7",
                                                          "source_id": 2})

    monkeypatch.setattr(H, "_run_part_budgeted", fake_part)
    monkeypatch.setattr(H, "_run_sub", fake_sub)
    monkeypatch.setattr(H, "_summarise_multi_answers", lambda *a, **k: "summary")
    mr = H._run_compound("msg", Its(), gs, time.time())
    parts = [(it.part, it.outcome, it.depends_on) for it in mr.items]
    assert parts == [("listings and ledger totals", "split", None),
                     ("how many listings", "answered", 0),
                     ("total ledger amount", "answered", 0),
                     ("and for those, the city", "refused", 0)], parts
    assert [d for _q, d in ran] == [1, 1]                  # sub-questions cannot split again
    assert mr.items[0].status == STATUS_OK and mr.items[0].result["answer"].startswith("This asks")
    assert seen_parent["p"] is mr.items[0]                 # the dependent part saw the split part


def test_single_message_split_becomes_a_compound_reply(monkeypatch):
    H = _hy()
    from query.multi_result import SubResult, STATUS_OK

    def fake_sub(sq, verbose=False, on_event=None, index=None, total=None):
        return SubResult(sq, STATUS_OK, "deterministic", {"status": "answered", "ok": True, "answer": "1"})
    monkeypatch.setattr(H, "_run_sub", fake_sub)
    monkeypatch.setattr(H, "_summarise_multi_answers", lambda *a, **k: "summary")
    mr = H._split_message("q", ["a?", "b?"])
    assert mr.compound and mr.relation == "independent"
    assert [(it.part, it.outcome, it.depends_on) for it in mr.items] == [("a?", "answered", None),
                                                                         ("b?", "answered", None)]


def test_split_sub_question_over_budget_is_a_timeout_part(monkeypatch):
    H = _hy()
    from query.multi_result import SubResult, STATUS_OK
    import config as _cfg
    monkeypatch.setattr(_cfg, "COMPOUND_PART_BUDGET_S", 3.0, raising=False)

    def fake_sub(sq, verbose=False, on_event=None, index=None, total=None):
        if sq == "slow":
            time.sleep(6)
        return SubResult(sq, STATUS_OK, "deterministic", {"status": "answered", "ok": True})
    monkeypatch.setattr(H, "_run_sub", fake_sub)
    t0 = time.time()
    out = H._run_split(["slow", "fast"], time.time() + 60)
    assert [it.outcome for it in out] == ["timeout", "answered"]
    assert time.time() - t0 < 5.5                          # the slow one returned AT its budget


def test_agent_rag_handoff_runs_the_rag_layer_on_those_sources(monkeypatch):
    H = _hy()
    import query.rag_layer as RL
    seen = {}
    monkeypatch.setattr(RL, "run_rag_layer", lambda q, source_ids=None, **k: seen.update(q=q, s=source_ids) or "RAG")
    assert H._agent_rag_handoff("what does the policy say", {"status": "clarify"}) is None
    assert H._agent_rag_handoff("what does the policy say",
                                {"status": "agent_rag", "rag_source_ids": [3]}) == "RAG"
    assert seen == {"q": "what does the policy say", "s": ["3"]}


# ── a MODEL anchor that now answers must keep the question's shape ────────────────────
def test_per_group_superlative_licenses_max():
    """'highest monthly fee per category' is MAX per group (measured: before this, the
    unlicensed max became a row ORDER, which a GROUP BY cannot carry → COUNT(*) per
    category). Without a breakdown the row-ranking reading stands."""
    from veda.understanding import frame_grounding as G
    fr = Frame(entity="amenity", aggregation="max", measure="monthly fee", group_by=["category"])
    G.license_aggregation(fr, "highest monthly fee per category")
    assert fr.aggregation == "max" and fr.group_by == ["category"]
    fr = Frame(entity="payment", aggregation="min", measure="amount")
    G.license_aggregation(fr, "What were the smallest payments processed recently?")
    assert fr.aggregation == "none" and fr.order is not None
