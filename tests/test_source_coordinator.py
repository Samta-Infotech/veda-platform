"""Tests for query.source_coordinator — the routing brain end-to-end (routing Phase 3.1).

All providers injected → no DB / SLM / model. Run: `pytest tests/test_source_coordinator.py`.
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORE = os.path.abspath(os.path.join(ROOT, "veda_core"))
sys.path.insert(0, CORE)

from query.source_coordinator import plan_route, dispatch  # noqa: E402
import query.agents as A  # noqa: E402


class _Col:
    def __init__(self, cid, cn, tn, sid, sim):
        self.col_id, self.col_name, self.table_name, self.source_id, self.similarity = cid, cn, tn, sid, sim


def _ev(*cols):
    return lambda q, sids: (list(cols), [])


# Equal top scores so both sources stay STRONG after dominance re-tiering (these tests exercise the
# ROUTING LOGIC given two equally-strong candidates, not the tiering itself).
_TWO = (_Col("a.x", "x", "ta", "5", 0.8), _Col("b.y", "y", "tb", "7", 0.8))


def test_single_relevant_source():
    d = plan_route("q", ["5", "7"], evidence_provider=_ev(_Col("t.rev", "rev", "r", "5", 0.82)),
                   edge_provider=lambda s: set(), profile_provider=lambda s: {"5": {"source_type": "relational"}})
    assert d.mode == "SINGLE" and d.source_ids == ["5"]


def test_edge_gives_multi():
    d = plan_route("q", ["5", "7"], evidence_provider=_ev(*_TWO),
                   edge_provider=lambda s: {frozenset({"5", "7"})},
                   profile_provider=lambda s: {"5": {"source_type": "relational"}, "7": {"source_type": "datalake"}})
    assert d.mode == "MULTI" and d.reason_code == "RELATIONSHIP_EDGE"


def test_same_domain_canonical():
    d = plan_route("q", ["5", "7"], evidence_provider=_ev(*_TWO), edge_provider=lambda s: set(),
                   profile_provider=lambda s: {"5": {"source_type": "relational", "is_canonical": True, "domain_tags": ["crm"]},
                                               "7": {"source_type": "relational", "domain_tags": ["crm"]}})
    assert d.mode == "SINGLE" and d.source_ids == ["5"] and d.reason_code == "CANONICAL_SELECTED"


def test_ambiguous_resolved_by_slm():
    d = plan_route("q", ["5", "7"], evidence_provider=_ev(*_TWO), edge_provider=lambda s: set(),
                   profile_provider=lambda s: {"5": {"domain_tags": ["finance"]}, "7": {"domain_tags": ["product"]}},
                   slm_call=lambda s, u: '{"mode":"SINGLE","source_ids":["5"]}')
    assert d.mode == "SINGLE" and d.decision_method == "slm"


def test_ambiguous_slm_hallucination_clarifies():
    d = plan_route("q", ["5", "7"], evidence_provider=_ev(*_TWO), edge_provider=lambda s: set(),
                   profile_provider=lambda s: {"5": {"domain_tags": ["finance"]}, "7": {"domain_tags": ["product"]}},
                   slm_call=lambda s, u: '{"mode":"SINGLE","source_ids":["zzz"]}')
    assert d.status == "CLARIFICATION_REQUIRED"


def test_no_evidence_is_no_match():
    d = plan_route("q", ["5"], evidence_provider=lambda q, s: ([], []),
                   edge_provider=lambda s: set(), profile_provider=lambda s: {})
    assert d.status == "NO_MATCH"


def test_dispatch_single_executes_agent(monkeypatch):
    monkeypatch.setattr(A, "_sql_delegate", lambda q, sm, cols, on_event=None: {
        "ok": True, "cols": ["rev"], "rows": [[100]]})
    d = plan_route("rev", ["5"], evidence_provider=_ev(_Col("t.rev", "rev", "r", "5", 0.82)),
                   edge_provider=lambda s: set(), profile_provider=lambda s: {"5": {"source_type": "relational"}})
    res = dispatch(d, "rev", sm={}, cols=[], profiles={"5": {"source_type": "relational"}})
    assert res.status == "ok" and res.data["rows"] == [[100]]


def test_dispatch_returns_none_for_non_single():
    d = plan_route("q", ["5", "7"], evidence_provider=_ev(*_TWO),
                   edge_provider=lambda s: {frozenset({"5", "7"})},
                   profile_provider=lambda s: {"5": {"source_type": "relational"}, "7": {"source_type": "datalake"}})
    assert d.mode == "MULTI"
    assert dispatch(d, "q") is None      # MULTI handed to Phase 4 federation, not dispatched here


# ── Phase A3: Source Adapter dispatch (flag-gated, default OFF) ──────────────────────────────
import config as _cfg_mod  # noqa: E402  (top-level module — veda_core is on sys.path via CORE above;
                                          # same module source_coordinator.py's `import config as _cfg`
                                          # resolves to, kept as `_cfg_mod` here only to avoid shadowing
                                          # the `config` name test-locally.)


def test_dispatch_default_off_matches_pre_phase_a3_behavior(monkeypatch):
    """Flag OFF (the default) must be byte-identical to calling resolve_agent() directly — this is
    the exact pre-Phase-A3 code path, unchanged."""
    assert _cfg_mod.SOURCE_ADAPTER_DISPATCH_ENABLED is False
    monkeypatch.setattr(A, "_sql_delegate", lambda q, sm, cols, on_event=None: {
        "ok": True, "cols": ["rev"], "rows": [[100]]})
    d = plan_route("rev", ["5"], evidence_provider=_ev(_Col("t.rev", "rev", "r", "5", 0.82)),
                   edge_provider=lambda s: set(), profile_provider=lambda s: {"5": {"source_type": "relational"}})
    res = dispatch(d, "rev", sm={}, cols=[], profiles={"5": {"source_type": "relational"}})
    assert res.status == "ok" and res.data["rows"] == [[100]]


def test_dispatch_with_adapter_flag_on_matches_flag_off_exactly(monkeypatch):
    """Dual-run compatibility check (Phase A4): flipping SOURCE_ADAPTER_DISPATCH_ENABLED on must
    produce an IDENTICAL result to the flag being off, for the same query/decision/inputs."""
    monkeypatch.setattr(A, "_sql_delegate", lambda q, sm, cols, on_event=None: {
        "ok": True, "cols": ["rev"], "rows": [[100]]})
    d = plan_route("rev", ["5"], evidence_provider=_ev(_Col("t.rev", "rev", "r", "5", 0.82)),
                   edge_provider=lambda s: set(), profile_provider=lambda s: {"5": {"source_type": "relational"}})

    monkeypatch.setattr(_cfg_mod, "SOURCE_ADAPTER_DISPATCH_ENABLED", False)
    off = dispatch(d, "rev", sm={}, cols=[], profiles={"5": {"source_type": "relational"}})

    monkeypatch.setattr(_cfg_mod, "SOURCE_ADAPTER_DISPATCH_ENABLED", True)
    on = dispatch(d, "rev", sm={}, cols=[], profiles={"5": {"source_type": "relational"}})

    assert off == on


def test_dispatch_with_adapter_flag_on_still_returns_none_for_non_single(monkeypatch):
    monkeypatch.setattr(_cfg_mod, "SOURCE_ADAPTER_DISPATCH_ENABLED", True)
    d = plan_route("q", ["5", "7"], evidence_provider=_ev(*_TWO),
                   edge_provider=lambda s: {frozenset({"5", "7"})},
                   profile_provider=lambda s: {"5": {"source_type": "relational"}, "7": {"source_type": "datalake"}})
    assert dispatch(d, "q") is None


# ── Phase B2: Execution Request dispatch (flag-gated, SEPARATE flag, default OFF) ────────────────
def test_execution_request_flag_default_off():
    assert _cfg_mod.EXECUTION_REQUEST_DISPATCH_ENABLED is False


def test_dispatch_er_flag_off_uses_legacy_execute_call_spy(monkeypatch):
    """TEST 1 + call-spy: flag OFF -> adapter.execute() (legacy) is invoked, execute_request() is NOT."""
    monkeypatch.setattr(_cfg_mod, "EXECUTION_REQUEST_DISPATCH_ENABLED", False)
    monkeypatch.setattr(_cfg_mod, "SOURCE_ADAPTER_DISPATCH_ENABLED", True)  # use SourceAdapter so both methods exist to spy on
    monkeypatch.setattr(A, "_sql_delegate", lambda q, sm, cols, on_event=None: {
        "ok": True, "cols": ["rev"], "rows": [[100]]})

    calls = {"execute": 0, "execute_request": 0}
    import query.source_adapters as SA
    real_execute = SA.SourceAdapter.execute
    real_execute_request = SA.SourceAdapter.execute_request

    def spy_execute(self, *a, **kw):
        calls["execute"] += 1
        return real_execute(self, *a, **kw)

    def spy_execute_request(self, *a, **kw):
        calls["execute_request"] += 1
        return real_execute_request(self, *a, **kw)

    monkeypatch.setattr(SA.SourceAdapter, "execute", spy_execute)
    monkeypatch.setattr(SA.SourceAdapter, "execute_request", spy_execute_request)

    d = plan_route("rev", ["5"], evidence_provider=_ev(_Col("t.rev", "rev", "r", "5", 0.82)),
                   edge_provider=lambda s: set(), profile_provider=lambda s: {"5": {"source_type": "relational"}})
    res = dispatch(d, "rev", sm={}, cols=[], profiles={"5": {"source_type": "relational"}})
    assert res.status == "ok"
    assert calls == {"execute": 1, "execute_request": 0}


def test_dispatch_er_flag_on_uses_execute_request_call_spy(monkeypatch):
    """TEST 2 + call-spy: flag ON -> ExecutionRequest is constructed and adapter.execute_request()
    is invoked, NOT the legacy execute() directly (execute_request internally delegates to execute,
    so execute() still runs ONCE underneath — the spy distinguishes "called directly by dispatch"
    from "called internally by execute_request" by checking which one dispatch itself invoked)."""
    monkeypatch.setattr(_cfg_mod, "EXECUTION_REQUEST_DISPATCH_ENABLED", True)
    monkeypatch.setattr(A, "_sql_delegate", lambda q, sm, cols, on_event=None: {
        "ok": True, "cols": ["rev"], "rows": [[100]]})

    import query.source_adapters as SA
    calls = {"execute_request": 0}
    real_execute_request = SA.SourceAdapter.execute_request

    def spy_execute_request(self, request, **kw):
        calls["execute_request"] += 1
        assert isinstance(request, __import__("query.execution_request", fromlist=["ExecutionRequest"]).ExecutionRequest)
        return real_execute_request(self, request, **kw)

    monkeypatch.setattr(SA.SourceAdapter, "execute_request", spy_execute_request)

    d = plan_route("rev", ["5"], evidence_provider=_ev(_Col("t.rev", "rev", "r", "5", 0.82)),
                   edge_provider=lambda s: set(), profile_provider=lambda s: {"5": {"source_type": "relational"}})
    res = dispatch(d, "rev", sm={}, cols=[], profiles={"5": {"source_type": "relational"}})
    assert res.status == "ok" and res.data["rows"] == [[100]]
    assert calls["execute_request"] == 1


def test_dispatch_er_flag_implies_adapter_resolution_even_if_other_flag_off(monkeypatch):
    """EXECUTION_REQUEST_DISPATCH_ENABLED alone (SOURCE_ADAPTER_DISPATCH_ENABLED left at its
    default OFF) must still resolve a SourceAdapter, since execute_request() only exists there."""
    monkeypatch.setattr(_cfg_mod, "EXECUTION_REQUEST_DISPATCH_ENABLED", True)
    monkeypatch.setattr(_cfg_mod, "SOURCE_ADAPTER_DISPATCH_ENABLED", False)
    monkeypatch.setattr(A, "_sql_delegate", lambda q, sm, cols, on_event=None: {
        "ok": True, "cols": ["rev"], "rows": [[100]]})
    d = plan_route("rev", ["5"], evidence_provider=_ev(_Col("t.rev", "rev", "r", "5", 0.82)),
                   edge_provider=lambda s: set(), profile_provider=lambda s: {"5": {"source_type": "relational"}})
    res = dispatch(d, "rev", sm={}, cols=[], profiles={"5": {"source_type": "relational"}})
    assert res.status == "ok"   # no AttributeError from a bare agent lacking execute_request


def test_dispatch_er_off_result_equals_er_on_result(monkeypatch):
    """TEST 3: same query/inputs, OFF result == ON result — AgentResult type, data, errors all match."""
    monkeypatch.setattr(A, "_sql_delegate", lambda q, sm, cols, on_event=None: {
        "ok": True, "cols": ["rev"], "rows": [[100]], "sql": "SELECT rev", "table": "t", "explain": {}})
    d = plan_route("rev", ["5"], evidence_provider=_ev(_Col("t.rev", "rev", "r", "5", 0.82)),
                   edge_provider=lambda s: set(), profile_provider=lambda s: {"5": {"source_type": "relational"}})

    monkeypatch.setattr(_cfg_mod, "EXECUTION_REQUEST_DISPATCH_ENABLED", False)
    off = dispatch(d, "rev", sm={}, cols=[], profiles={"5": {"source_type": "relational"}})

    monkeypatch.setattr(_cfg_mod, "EXECUTION_REQUEST_DISPATCH_ENABLED", True)
    on = dispatch(d, "rev", sm={}, cols=[], profiles={"5": {"source_type": "relational"}})

    assert type(off) is type(on)
    assert off == on   # AgentResult is a plain @dataclass -> field-wise equality


def test_dispatch_er_relational_source_receives_sm_and_cols(monkeypatch):
    """TEST 6."""
    seen = {}

    def deleg(q, sm, cols, on_event=None):
        seen["sm"], seen["cols"] = sm, cols
        return {"ok": True, "cols": [], "rows": []}
    monkeypatch.setattr(A, "_sql_delegate", deleg)
    monkeypatch.setattr(_cfg_mod, "EXECUTION_REQUEST_DISPATCH_ENABLED", True)

    d = plan_route("q", ["5"], evidence_provider=_ev(_Col("t.x", "x", "r", "5", 0.82)),
                   edge_provider=lambda s: set(), profile_provider=lambda s: {"5": {"source_type": "relational"}})
    dispatch(d, "q", sm={"tables": ["t"]}, cols=["a", "b"], profiles={"5": {"source_type": "relational"}})
    assert seen["sm"] == {"tables": ["t"]} and seen["cols"] == ["a", "b"]


def test_dispatch_er_document_source_needs_no_sm_or_cols(monkeypatch):
    """TEST 5."""
    class _FakeRag:
        def __init__(self):
            self.answer, self.citations, self.error = "per policy...", ["SLA.pdf"], None
    monkeypatch.setattr(A, "_rag_delegate", lambda q, sids, on_event=None: _FakeRag())
    monkeypatch.setattr(_cfg_mod, "EXECUTION_REQUEST_DISPATCH_ENABLED", True)

    d = plan_route("what does the SLA say", ["9"],
                   evidence_provider=_ev(_Col("doc.chunk", "chunk", "c", "9", 0.9)),
                   edge_provider=lambda s: set(), profile_provider=lambda s: {"9": {"source_type": "document"}})
    res = dispatch(d, "what does the SLA say", profiles={"9": {"source_type": "document"}})
    assert res.status == "ok" and res.data["citations"] == ["SLA.pdf"]


def test_dispatch_er_flag_on_non_single_returns_none():
    """TEST 7: non-SINGLE preserved."""
    import config as _c
    _c.EXECUTION_REQUEST_DISPATCH_ENABLED = True
    try:
        d = plan_route("q", ["5", "7"], evidence_provider=_ev(*_TWO),
                       edge_provider=lambda s: {frozenset({"5", "7"})},
                       profile_provider=lambda s: {"5": {"source_type": "relational"}, "7": {"source_type": "datalake"}})
        assert d.mode == "MULTI"
        assert dispatch(d, "q") is None
    finally:
        _c.EXECUTION_REQUEST_DISPATCH_ENABLED = False


# ── Phase C1: Capability Planning shadow observation (flag-gated, SEPARATE flag, default OFF) ────
def test_capability_shadow_flag_default_off():
    assert _cfg_mod.CAPABILITY_PLANNING_SHADOW_ENABLED is False


def test_plan_route_flag_off_decide_receives_identical_candidates(monkeypatch):
    """TEST A: flag OFF -> decide() receives the EXACT SAME candidate list object build_candidates()
    produced — call-spy proves identity, not just equal-looking output."""
    import query.source_coordinator as SC
    monkeypatch.setattr(_cfg_mod, "CAPABILITY_PLANNING_SHADOW_ENABLED", False)
    seen = {}
    real_build = SC.build_candidates

    def spy_build(*a, **kw):
        result = real_build(*a, **kw)
        seen["built"] = result
        return result

    real_decide = SC.decide

    def spy_decide(candidates, *a, **kw):
        seen["decided_with"] = candidates
        return real_decide(candidates, *a, **kw)

    monkeypatch.setattr(SC, "build_candidates", spy_build)
    monkeypatch.setattr(SC, "decide", spy_decide)

    SC.plan_route("q", ["5"], evidence_provider=_ev(_Col("t.x", "x", "r", "5", 0.82)),
                  edge_provider=lambda s: set(), profile_provider=lambda s: {"5": {"source_type": "relational"}})
    assert seen["decided_with"] is seen["built"]   # identity, not equality


def test_plan_route_flag_on_decide_still_receives_identical_candidates(monkeypatch):
    """TEST B (candidate-list part): flag ON -> shadow observation runs, but decide() STILL
    receives the exact same object — capability planning never touches the list."""
    import query.source_coordinator as SC
    monkeypatch.setattr(_cfg_mod, "CAPABILITY_PLANNING_SHADOW_ENABLED", True)
    seen = {}
    real_build = SC.build_candidates

    def spy_build(*a, **kw):
        result = real_build(*a, **kw)
        seen["built"] = result
        return result

    real_decide = SC.decide

    def spy_decide(candidates, *a, **kw):
        seen["decided_with"] = candidates
        return real_decide(candidates, *a, **kw)

    monkeypatch.setattr(SC, "build_candidates", spy_build)
    monkeypatch.setattr(SC, "decide", spy_decide)

    SC.plan_route("how many projects", ["5"],
                  evidence_provider=_ev(_Col("t.x", "x", "r", "5", 0.82)),
                  edge_provider=lambda s: set(), profile_provider=lambda s: {"5": {"source_type": "relational"}})
    assert seen["decided_with"] is seen["built"]


def test_plan_route_shadow_flag_on_calls_run_capability_planning_shadow(monkeypatch):
    """Call-spy proving the shadow function is genuinely invoked when the flag is on (not just
    output equivalence — the same lesson as Phase A3/B2's call-spy tests)."""
    import query.source_coordinator as SC
    monkeypatch.setattr(_cfg_mod, "CAPABILITY_PLANNING_SHADOW_ENABLED", True)
    calls = {"n": 0}

    import query.capability_observation as CO
    real = CO.run_capability_planning_shadow

    def spy(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(CO, "run_capability_planning_shadow", spy)
    SC.plan_route("how many projects", ["5"],
                  evidence_provider=_ev(_Col("t.x", "x", "r", "5", 0.82)),
                  edge_provider=lambda s: set(), profile_provider=lambda s: {"5": {"source_type": "relational"}})
    assert calls["n"] == 1


def test_plan_route_output_identical_shadow_flag_on_vs_off(monkeypatch):
    """TEST: same query/inputs, shadow flag OFF result == shadow flag ON result — the routing
    decision itself must be byte-for-byte unaffected by capability observation."""
    import query.source_coordinator as SC

    def _run():
        return SC.plan_route("how many projects", ["5", "7"],
                             evidence_provider=_ev(*_TWO),
                             edge_provider=lambda s: {frozenset({"5", "7"})},
                             profile_provider=lambda s: {"5": {"source_type": "relational"},
                                                        "7": {"source_type": "datalake"}})

    monkeypatch.setattr(_cfg_mod, "CAPABILITY_PLANNING_SHADOW_ENABLED", False)
    off = _run()
    monkeypatch.setattr(_cfg_mod, "CAPABILITY_PLANNING_SHADOW_ENABLED", True)
    on = _run()
    assert off == on


def test_plan_route_shadow_on_with_incompatible_candidate_still_included(monkeypatch):
    """CRITICAL: even a candidate the shadow observation would mark capability-incompatible
    (a document source for an aggregation query) MUST still appear in the routing decision —
    C1 never filters."""
    import query.source_coordinator as SC
    monkeypatch.setattr(_cfg_mod, "CAPABILITY_PLANNING_SHADOW_ENABLED", True)
    d = SC.plan_route("how many projects", ["5", "9"],
                      evidence_provider=_ev(_Col("t.x", "x", "ta", "5", 0.8),
                                            _Col("doc.chunk", "chunk", "tb", "9", 0.8)),
                      edge_provider=lambda s: {frozenset({"5", "9"})},
                      profile_provider=lambda s: {"5": {"source_type": "relational"},
                                                 "9": {"source_type": "document"}})
    # document source (no AGGREGATION capability) is still present in the decision's candidates
    all_ids = {c.source_id for c in d.candidate_sources}
    assert "9" in all_ids


def test_dispatch_er_request_object_not_mutated(monkeypatch):
    """TEST 4: the ExecutionRequest built inside dispatch() is frozen — captured via the call-spy
    and asserted unchanged in identity/value after execute_request() returns."""
    monkeypatch.setattr(_cfg_mod, "EXECUTION_REQUEST_DISPATCH_ENABLED", True)
    monkeypatch.setattr(A, "_sql_delegate", lambda q, sm, cols, on_event=None: {
        "ok": True, "cols": [], "rows": []})

    import query.source_adapters as SA
    captured = {}
    real = SA.SourceAdapter.execute_request

    def spy(self, request, **kw):
        captured["before"] = (request.query, request.source_id, request.sm, request.cols)
        result = real(self, request, **kw)
        captured["after"] = (request.query, request.source_id, request.sm, request.cols)
        return result

    monkeypatch.setattr(SA.SourceAdapter, "execute_request", spy)
    d = plan_route("q", ["5"], evidence_provider=_ev(_Col("t.x", "x", "r", "5", 0.82)),
                   edge_provider=lambda s: set(), profile_provider=lambda s: {"5": {"source_type": "relational"}})
    dispatch(d, "q", sm={"a": 1}, cols=["x"], profiles={"5": {"source_type": "relational"}})
    assert captured["before"] == captured["after"]


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
