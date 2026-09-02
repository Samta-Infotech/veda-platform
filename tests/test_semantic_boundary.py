"""Tests for the semantic decision boundary + SINGLE|MULTI|NONE SLM contract (P1/P2/P3).

All providers/SLM injected → no DB/model. Run: `python tests/test_semantic_boundary.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

from query.source_coordinator import plan_route, _decision_boundary, _query_embedding, _ROUTING_QV  # noqa: E402
from query.routing_contracts import CandidateSource, MODE_MULTI, RC_RELATIONSHIP_EDGE, RC_CANONICAL_SELECTED  # noqa: E402
from query.routing_slm import validate_slm_decision, resolve_boundary  # noqa: E402


class _Col:
    def __init__(self, cid, cn, tn, sid, sim):
        self.col_id, self.col_name, self.table_name, self.source_id, self.similarity = cid, cn, tn, sid, sim


def _ev(*cols):
    return lambda q, sids: (list(cols), [])


def _prof(*sids):
    return lambda s: {sid: {"source_type": "relational"} for sid in sids}


def _c(sid, tier="STRONG", score=0.6):
    return CandidateSource(source_id=sid, source_type="relational", presence_tier=tier, top_score=score)


# ── boundary: when the SLM is / isn't invoked ────────────────────────────────────────────────
def test_dominant_single_bypasses_slm():
    # one STRONG, clearly ahead → deterministic, NOT a boundary
    cands = [_c("5", "STRONG", 0.70), _c("7", "WEAK", 0.40)]
    from query.routing_contracts import RoutingDecision, STATUS_ROUTED, MODE_SINGLE, RC_SINGLE_CANDIDATE
    dec = RoutingDecision(STATUS_ROUTED, mode=MODE_SINGLE, source_ids=["5"], reason_code=RC_SINGLE_CANDIDATE,
                          candidate_sources=cands)
    at, _ = _decision_boundary(cands, dec, ambiguous=False)
    assert at is False


def test_structural_multi_bypasses_slm():
    cands = [_c("5", "STRONG", 0.6), _c("7", "STRONG", 0.59)]
    from query.routing_contracts import RoutingDecision, STATUS_ROUTED
    dec = RoutingDecision(STATUS_ROUTED, mode=MODE_MULTI, source_ids=["5", "7"],
                          reason_code=RC_RELATIONSHIP_EDGE, candidate_sources=cands)
    at, _ = _decision_boundary(cands, dec, ambiguous=False)
    assert at is False


def test_canonical_bypasses_slm():
    cands = [_c("5", "STRONG", 0.6), _c("7", "STRONG", 0.59)]
    from query.routing_contracts import RoutingDecision, STATUS_ROUTED, MODE_SINGLE
    dec = RoutingDecision(STATUS_ROUTED, mode=MODE_SINGLE, source_ids=["5"],
                          reason_code=RC_CANONICAL_SELECTED, candidate_sources=cands)
    at, _ = _decision_boundary(cands, dec, ambiguous=False)
    assert at is False


def test_two_strong_is_a_boundary():
    cands = [_c("5", "STRONG", 0.60), _c("7", "STRONG", 0.58)]
    from query.routing_contracts import RoutingDecision, STATUS_ROUTED, MODE_SINGLE, RC_SINGLE_CANDIDATE
    dec = RoutingDecision(STATUS_ROUTED, mode=MODE_SINGLE, source_ids=["5"], reason_code=RC_SINGLE_CANDIDATE,
                          candidate_sources=cands)
    at, subset = _decision_boundary(cands, dec, ambiguous=False)
    assert at is True and len(subset) >= 2


def test_nothing_strong_is_a_boundary():
    cands = [_c("5", "WEAK", 0.40), _c("7", "WEAK", 0.38)]
    from query.routing_contracts import RoutingDecision, STATUS_ROUTED, MODE_SINGLE, RC_SINGLE_CANDIDATE
    dec = RoutingDecision(STATUS_ROUTED, mode=MODE_SINGLE, source_ids=["5"], reason_code=RC_SINGLE_CANDIDATE,
                          candidate_sources=cands)
    at, _ = _decision_boundary(cands, dec, ambiguous=False)
    assert at is True


# ── boundary → SLM outcomes (SINGLE / MULTI / NONE) ──────────────────────────────────────────
def test_boundary_can_become_multi():
    d = plan_route("q", ["5", "7"], evidence_provider=_ev(_Col("a.x", "x", "t", "5", 0.60),
                                                          _Col("b.y", "y", "t", "7", 0.59)),
                   edge_provider=lambda s: set(), profile_provider=_prof("5", "7"),
                   slm_call=lambda s, u: '{"decision":"MULTI","selected_source_ids":["5","7"]}')
    assert d.mode == "MULTI" and set(d.source_ids) == {"5", "7"} and d.decision_method == "slm"


def test_boundary_can_stay_single():
    d = plan_route("q", ["5", "7"], evidence_provider=_ev(_Col("a.x", "x", "t", "5", 0.60),
                                                          _Col("b.y", "y", "t", "7", 0.59)),
                   edge_provider=lambda s: set(), profile_provider=_prof("5", "7"),
                   slm_call=lambda s, u: '{"decision":"SINGLE","selected_source_ids":["5"]}')
    assert d.mode == "SINGLE" and d.source_ids == ["5"]


def test_boundary_can_become_none():
    d = plan_route("book a flight", ["5", "7"],
                   evidence_provider=_ev(_Col("a.x", "x", "t", "5", 0.40), _Col("b.y", "y", "t", "7", 0.38)),
                   edge_provider=lambda s: set(), profile_provider=_prof("5", "7"),
                   slm_call=lambda s, u: '{"decision":"NONE","selected_source_ids":[]}')
    assert d.status == "NO_MATCH"


# ── strict validation ────────────────────────────────────────────────────────────────────────
def test_slm_cannot_emit_non_candidate():
    d = resolve_boundary("q", [_c("5"), _c("7")],
                         slm_call=lambda s, u: '{"decision":"SINGLE","selected_source_ids":["zzz"]}')
    assert d.status == "CLARIFICATION_REQUIRED"


def test_multi_with_one_source_rejected():
    assert not validate_slm_decision({"decision": "MULTI", "selected_source_ids": ["5"]}, {"5", "7"})[0]


def test_single_with_two_sources_rejected():
    assert not validate_slm_decision({"decision": "SINGLE", "selected_source_ids": ["5", "7"]}, {"5", "7"})[0]


def test_none_with_sources_rejected():
    assert not validate_slm_decision({"decision": "NONE", "selected_source_ids": ["5"]}, {"5", "7"})[0]


def test_slm_failure_is_controlled():
    def boom(s, u):
        raise RuntimeError("down")
    d = resolve_boundary("q", [_c("5"), _c("7")], slm_call=boom)
    assert d.status == "CLARIFICATION_REQUIRED"


# ── P3: query embedded once per routing request ──────────────────────────────────────────────
def test_query_embedded_once(monkeypatch):
    calls = {"n": 0}
    import query.rag_layer as rl
    def fake_encode(q, verbose=False):
        calls["n"] += 1
        import numpy as np
        return np.ones(4, dtype="float32")
    monkeypatch.setattr(rl, "_encode_rag_query", fake_encode)
    _ROUTING_QV.set((None, None))
    a = _query_embedding("same query")
    b = _query_embedding("same query")   # cached — no second encode
    assert calls["n"] == 1 and a is b


# ── Required-Source Escalation (flag-gated) ──────────────────────────────────────────────────
import config as _config  # noqa: E402
from query.routing_contracts import RoutingDecision, STATUS_ROUTED, MODE_SINGLE, RC_SINGLE_CANDIDATE  # noqa: E402


def _ci(sid, tier, score, item):
    """CandidateSource with an explicit item-prior (top_item_score) for escalation tests."""
    return CandidateSource(source_id=sid, source_type="relational", presence_tier=tier,
                           top_score=score, top_item_score=item)


def _dominant_decision(cands):
    return RoutingDecision(STATUS_ROUTED, mode=MODE_SINGLE, source_ids=[cands[0].source_id],
                           reason_code=RC_SINGLE_CANDIDATE, candidate_sources=cands)


def _edges(*pairs):
    return {frozenset(p) for p in pairs}


def test_rse_off_dominant_single_byte_identical(mp):
    # flag OFF: even with both signals present, a dominant single must NOT reach the SLM.
    mp.setattr(_config, "REQUIRED_SOURCE_ESCALATION_ENABLED", False)
    cands = [_ci("4", "STRONG", 0.60, 0.60), _ci("2", "WEAK", 0.45, 0.45)]
    at, _ = _decision_boundary(cands, _dominant_decision(cands), ambiguous=False,
                               edge_pairs=_edges({"4", "2"}))
    assert at is False


def test_rse_on_both_signals_escalates(mp):
    # flag ON + edge-connected + item-prior-positive secondary → escalate to the SLM boundary.
    mp.setattr(_config, "REQUIRED_SOURCE_ESCALATION_ENABLED", True)
    cands = [_ci("4", "STRONG", 0.60, 0.60), _ci("2", "WEAK", 0.45, 0.45)]
    at, boundary = _decision_boundary(cands, _dominant_decision(cands), ambiguous=False,
                                      edge_pairs=_edges({"4", "2"}))
    assert at is True
    assert {c.source_id for c in boundary} == {"4", "2"}


def test_rse_on_edge_but_no_item_support_no_escalate(mp):
    # edge-connected but the secondary has NO item-prior (bare shared-column match) → stay deterministic.
    mp.setattr(_config, "REQUIRED_SOURCE_ESCALATION_ENABLED", True)
    cands = [_ci("4", "STRONG", 0.60, 0.60), _ci("2", "WEAK", 0.45, 0.0)]
    at, _ = _decision_boundary(cands, _dominant_decision(cands), ambiguous=False,
                               edge_pairs=_edges({"4", "2"}))
    assert at is False


def test_rse_on_item_support_but_no_edge_no_escalate(mp):
    # item-prior-positive secondary but NO structural edge to the primary → stay deterministic.
    mp.setattr(_config, "REQUIRED_SOURCE_ESCALATION_ENABLED", True)
    cands = [_ci("4", "STRONG", 0.60, 0.60), _ci("2", "WEAK", 0.45, 0.45)]
    at, _ = _decision_boundary(cands, _dominant_decision(cands), ambiguous=False,
                               edge_pairs=_edges())            # no edges
    assert at is False


def test_rse_on_no_secondary_candidate_no_escalate(mp):
    # a genuine lone dominant source (secondary is NONE tier) → no escalation even with flag ON.
    mp.setattr(_config, "REQUIRED_SOURCE_ESCALATION_ENABLED", True)
    cands = [_ci("4", "STRONG", 0.60, 0.60), _ci("2", "NONE", 0.10, 0.10)]
    at, _ = _decision_boundary(cands, _dominant_decision(cands), ambiguous=False,
                               edge_pairs=_edges({"4", "2"}))
    assert at is False


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
