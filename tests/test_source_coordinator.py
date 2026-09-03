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
