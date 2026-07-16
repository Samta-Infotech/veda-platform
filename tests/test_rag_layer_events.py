"""Regression tests for veda_core/query/rag_layer.py's on_event support
(2026-07-16). Before this fix, run_rag_layer/run_hybrid_layer were silent
black boxes between the caller's outer "Retrieving relevant documents..."/
"Running SQL and document fusion..." ticks and the terminal "answer" tick —
no visibility into the actual retrieval or the SLM synthesis call, often the
slowest, most opaque part of the whole turn.

Pure-python: embedding/retrieval/SLM calls are all monkeypatched, no DB, no
network, no real Ollama/vector-store dependency."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))


def _fake_chunk(name="doc.pdf", sim=0.8):
    from ingestion.chunk_embedder import ChunkRetrievalResult
    return ChunkRetrievalResult(chunk_id="c1", source_id="s1", doc_id="d1", doc_name=name,
                                chunk_index=0, text="some passage text", page_num=1, similarity=sim)


def _stub_rag_deps(monkeypatch):
    import query.rag_layer as rag_mod
    monkeypatch.setattr(rag_mod, "_encode_rag_query", lambda query, verbose=False: [0.1, 0.2])
    monkeypatch.setattr(rag_mod, "retrieve_top_k_chunks", lambda **k: [_fake_chunk()])
    monkeypatch.setattr(rag_mod, "_call_ollama", lambda system, user: "The answer is X.")
    return rag_mod


def test_run_rag_layer_emits_retrieve_and_synthesize_events(monkeypatch):
    rag_mod = _stub_rag_deps(monkeypatch)
    events = []
    result = rag_mod.run_rag_layer(
        "what does the contract say", source_ids=["s1"],
        on_event=lambda phase, msg, extra: events.append((phase, msg)),
    )
    assert result.answer == "The answer is X."
    phases = [p for p, _ in events]
    assert "rag_retrieve" in phases
    assert "rag_synthesize" in phases
    # retrieve must fire BEFORE synthesize — the retrieval result is what
    # gets synthesized, not the other way round.
    assert phases.index("rag_retrieve") < phases.index("rag_synthesize")


def test_run_rag_layer_on_event_none_never_raises(monkeypatch):
    """on_event is optional — every existing caller that doesn't pass it
    (or explicitly passes None) must see identical behavior to before this
    change, never a crash."""
    rag_mod = _stub_rag_deps(monkeypatch)
    result = rag_mod.run_rag_layer("what does the contract say", source_ids=["s1"])
    assert result.answer == "The answer is X."


def test_run_rag_layer_on_event_callback_exception_never_propagates(monkeypatch):
    """A broken UI callback must never fail the actual query — same
    contract as veda_hybrid.py's own _emit."""
    rag_mod = _stub_rag_deps(monkeypatch)

    def _broken(*a, **k):
        raise RuntimeError("frontend callback exploded")

    result = rag_mod.run_rag_layer("what does the contract say", source_ids=["s1"], on_event=_broken)
    assert result.answer == "The answer is X."


def test_run_hybrid_layer_emits_retrieve_and_synthesize_events(monkeypatch):
    import query.rag_layer as rag_mod
    monkeypatch.setattr(rag_mod, "_encode_rag_query", lambda query, verbose=False: [0.1, 0.2])
    monkeypatch.setattr(rag_mod, "retrieve_top_k_chunks", lambda **k: [_fake_chunk()])
    monkeypatch.setattr(rag_mod, "_call_ollama", lambda system, user: "Fused answer.")

    events = []
    result = rag_mod.run_hybrid_layer(
        "revenue and the contract terms", sql_columns=[], source_ids=["s1"],
        on_event=lambda phase, msg, extra: events.append((phase, msg)),
    )
    assert result.answer == "Fused answer."
    phases = [p for p, _ in events]
    assert "hybrid_retrieve" in phases
    assert "hybrid_synthesize" in phases
    assert phases.index("hybrid_retrieve") < phases.index("hybrid_synthesize")


# ---------------------------------------------------------------------------
# _run_nosql (veda_core/veda_hybrid.py) — same on_event contract, covering
# the LLM-based query-building step specifically (nosql_build).
# ---------------------------------------------------------------------------

class _FakeConnStatus:
    ok = True


class _FakeNosqlConn:
    def connect(self):
        return _FakeConnStatus()

    def get_nosql_schema(self):
        return {"orders": {}}

    def disconnect(self):
        pass

    def execute_query(self, query, row_limit, timeout_sec):
        return type("Res", (), {"row_count": 2, "columns": ["id"], "rows": [(1,), (2,)]})()


def test_run_nosql_emits_build_event(monkeypatch):
    import veda_hybrid
    import config as config_mod
    import connectors.base as connectors_mod
    import query.nosql_builder as nosql_builder_mod

    monkeypatch.setattr(config_mod, "get_source", lambda sid: {"type": "nosql", "engine": "mongodb"})
    monkeypatch.setattr(connectors_mod, "build_connector", lambda src: _FakeNosqlConn())
    fake_nb = type("NB", (), {"error": None, "query_json": {}})()
    monkeypatch.setattr(nosql_builder_mod, "run_nosql_builder", lambda **k: fake_nb)
    # NL-answer summarization is a separate concern (already covered by
    # test_tier2_answer.py/test_result_explainer.py) — disable it here so
    # this test stays focused on the nosql_build event itself.
    monkeypatch.setattr(config_mod, "NL_ANSWER_ENABLED", False)

    events = []
    veda_hybrid._run_nosql("find recent orders", source_ids=["s1"],
                           on_event=lambda phase, msg, extra: events.append((phase, msg)))
    assert "nosql_build" in [p for p, _ in events]


def test_run_nosql_on_event_none_never_raises(monkeypatch):
    import veda_hybrid
    import config as config_mod
    import connectors.base as connectors_mod
    import query.nosql_builder as nosql_builder_mod

    monkeypatch.setattr(config_mod, "get_source", lambda sid: {"type": "nosql", "engine": "mongodb"})
    monkeypatch.setattr(connectors_mod, "build_connector", lambda src: _FakeNosqlConn())
    fake_nb = type("NB", (), {"error": None, "query_json": {}})()
    monkeypatch.setattr(nosql_builder_mod, "run_nosql_builder", lambda **k: fake_nb)
    monkeypatch.setattr(config_mod, "NL_ANSWER_ENABLED", False)

    # No on_event passed at all — every existing caller must be unaffected.
    veda_hybrid._run_nosql("find recent orders", source_ids=["s1"])
