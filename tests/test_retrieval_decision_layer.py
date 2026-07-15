"""Tests for the Phase 2 retrieval-decision-layer improvements
(docs/RETRIEVAL_DECISION_LAYER_AUDIT.md):

1. select_primary_table() now records a genuine (non-tautological), always-on
   confidence/margin/alternatives into the shared "anchor_selection" trace
   section, closing the gap where confidence was previously recorded only
   when vet_primary went on to run score_anchors.
2. retrieval_engine_phase3.py's _results_from_tuples() now carries each
   signal's own contribution (semantic/sparse/subgraph/fk_path/value_index/
   rrf/boosted) onto RetrievalResult instead of discarding it.

Pure-python, no DB, no network.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))


class _FakeTrace:
    """Minimal stand-in for veda.explain.ExplainTrace — same merge-into-section
    semantics (.set updates, doesn't replace) and same verbose-gating for .cand."""
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.sections = {}

    def set(self, section, **data):
        self.sections.setdefault(section, {}).update(data)

    def cand(self, section, key, item):
        if not self.verbose:
            return
        self.sections.setdefault(section, {}).setdefault(key, []).append(item)

    def note(self, *a, **k):
        pass


def _rr(col_id, final_score):
    from retrieval.retrieval_engine_phase3 import RetrievalResult
    table_name, col_name = col_id.rsplit(".", 1)
    return RetrievalResult(col_id=col_id, column_name=col_name, table_name=table_name,
                           final_score=final_score)


def test_select_primary_table_records_confidence_without_vet_primary(monkeypatch):
    """The actual gap found in the audit: confidence used to be recorded ONLY when
    vet_primary ran score_anchors. select_primary_table must now record it on its
    own, unconditionally, whenever it's given a trace."""
    import veda.routing as routing

    monkeypatch.setattr(routing, "route_tables_semantic", lambda query, top_n=8: {})

    results = [_rr("orders.total", 0.9), _rr("orders.id", 0.5), _rr("users.email", 0.1)]
    sm = {"columns": {}, "tables": {"orders": {}, "users": {}}}
    trace = _FakeTrace()

    winner = routing.select_primary_table(results, "orders total", sm, trace=trace)

    assert winner == "orders"
    sec = trace.sections["anchor_selection"]
    assert sec["anchor"] == "orders"
    assert sec["source"] == "router"
    assert 0.0 <= sec["confidence"] <= 1.0
    assert sec["margin"] > 0   # orders clearly beats users here
    assert "alternatives" in sec
    assert any(a["table"] == "orders" for a in sec["alternatives"])


def test_select_primary_table_confidence_is_not_tautological(monkeypatch):
    """Guard against the trivial bug of dividing best_score by itself (always 1.0,
    regardless of how close the runner-up was) — confidence must actually reflect
    the margin: a near-tie must score lower than a clear win."""
    import veda.routing as routing

    monkeypatch.setattr(routing, "route_tables_semantic", lambda query, top_n=8: {})
    sm = {"columns": {}, "tables": {"a": {}, "b": {}}}

    # Clear win: a way ahead of b
    clear = [_rr("a.x", 1.0), _rr("b.x", 0.05)]
    t1 = _FakeTrace()
    routing.select_primary_table(clear, "irrelevant text", sm, trace=t1)

    # Near-tie: a barely ahead of b
    tie = [_rr("a.x", 0.51), _rr("b.x", 0.50)]
    t2 = _FakeTrace()
    routing.select_primary_table(tie, "irrelevant text", sm, trace=t2)

    assert t1.sections["anchor_selection"]["confidence"] > t2.sections["anchor_selection"]["confidence"]


def test_select_primary_table_no_trace_is_a_pure_noop():
    """Backward compatibility: trace=None (the default) must not change behavior
    or raise — every existing call site that doesn't pass trace keeps working."""
    import veda.routing as routing

    sm = {"columns": {}, "tables": {"orders": {}}}
    results = [_rr("orders.total", 0.9)]
    winner = routing.select_primary_table(results, "orders total", sm)
    assert winner == "orders"


def test_results_from_tuples_carries_signal_scores():
    """The actual gap found in the audit: RetrievalResult declares semantic_score/
    sparse_score/subgraph_score/fk_path_score/value_index_score/rrf_score/
    boosted_score, but the only constructor left them all at 0.0. They must now
    reflect each signal's real contribution."""
    from retrieval.retrieval_engine_phase3 import RetrievalEnginePhase3

    engine = RetrievalEnginePhase3.__new__(RetrievalEnginePhase3)  # skip heavy __init__
    results = engine._results_from_tuples(
        [("orders.total", 0.42)],
        semantic_map={"orders.total": 0.8},
        sparse_map={"orders.total": 0.6},
        subgraph_map={"orders.total": 0.3},
        fk_map={"orders.total": 0.7},
        value_map={"orders.total": 1.0},
        rrf_map={"orders.total": 0.09},
        boosted_map={"orders.total": 0.42},
    )
    r = results[0]
    assert r.final_score == 0.42
    assert r.semantic_score == 0.8
    assert r.sparse_score == 0.6
    assert r.subgraph_score == 0.3
    assert r.fk_path_score == 0.7
    assert r.value_index_score == 1.0
    assert r.rrf_score == 0.09
    assert r.boosted_score == 0.42


def test_results_from_tuples_without_maps_defaults_to_zero():
    """Backward compatibility: the cache-hit call site (no signal maps available)
    must behave exactly as before — every *_score field at its dataclass default."""
    from retrieval.retrieval_engine_phase3 import RetrievalEnginePhase3

    engine = RetrievalEnginePhase3.__new__(RetrievalEnginePhase3)
    results = engine._results_from_tuples([("orders.total", 0.42)])
    r = results[0]
    assert r.final_score == 0.42
    assert r.semantic_score == 0.0
    assert r.sparse_score == 0.0
    assert r.subgraph_score == 0.0
    assert r.fk_path_score == 0.0
    assert r.value_index_score == 0.0
    assert r.rrf_score == 0.0
    assert r.boosted_score == 0.0


def test_reranker_precomputed_text_helper_still_importable():
    """Sanity check for the pipeline.py primary-rerank fix: _precomputed_rerank_text
    must be importable from query.reranker (pipeline.py now imports it directly)."""
    from query.reranker import _precomputed_rerank_text
    assert callable(_precomputed_rerank_text)
