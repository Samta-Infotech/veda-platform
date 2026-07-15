"""Regression + behavior tests for the Tier1 -> Tier2 ExecutionState propagation
refactor (veda/execution_state.py, veda/pipeline.py, veda_hybrid.py,
query/retrieval_v2.py, query/retrieval_select.py).

Covers:
1. ExecutionState is lightweight (no full trace) — the "keep it lightweight"
   requirement is a structural property, tested directly.
2. _merge_seed_candidates (retrieval_v2.py) merges Tier1 seed candidates without
   a second DB lookup (synthetic table_id) and dedups against existing candidates.
3. _tier2_sql(execution_state=None) behaves exactly as before (regression guard).
4. _tier2_sql(execution_state=<populated>) reuses temporal parsing and seeds
   select_retrieval with Tier1's candidate_fields instead of recomputing.
5. Tier1's refusal_reason seeds the existing repair-hint mechanism on attempt 0.
6. ExecutionState never crosses the HTTP boundary (inference/routes/hybrid.py's
   _serialize() strips "context"/"trace" from every response) — this is the fix
   for the critical finding from the production-readiness review.
7. The Tier2 reuse log only claims what's actually functionally consumed — the
   fix for the "misleading logging" finding from the same review.

Pure-python, no DB, no network — envelope_slm.emit_envelope (which would otherwise
make a real Ollama HTTP call) is explicitly monkeypatched to fail fast, exactly
mirroring its own documented "Ollama down -> fall through to IR path" behavior.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))


def _stop_envelope(monkeypatch):
    """Force the envelope fast-path to fail deterministically (no network), so
    every test below exercises the IR path via the mocked run_slm_layer."""
    import query.envelope_slm as envelope_slm

    def _raise(*a, **k):
        raise RuntimeError("no network in tests")

    monkeypatch.setattr(envelope_slm, "emit_envelope", _raise)


def test_execution_state_is_lightweight():
    from veda.execution_state import ExecutionState

    es = ExecutionState()
    assert not hasattr(es, "trace"), "ExecutionState must not carry the full trace"
    assert es.candidate_fields == []
    assert es.candidate_tables == []
    assert es.refusal_reason is None
    assert hasattr(es, "sql_planning")
    assert hasattr(es, "primary_table")
    assert hasattr(es, "temporal_result")
    assert hasattr(es, "query_understanding")


def test_merge_seed_candidates_adds_new_and_dedups():
    from query.retrieval_v2 import _merge_seed_candidates
    from ingestion.vector_store import RetrievalResult

    existing_col = RetrievalResult(col_id="users.email", col_name="email", table_id="t1",
                                   table_name="users", semantic_type="TEXT", similarity=0.9)
    existing_tbl = RetrievalResult(col_id="", col_name="", table_id="t1", table_name="users",
                                   semantic_type="UNKNOWN", similarity=0.5)

    seeds = [
        {"table_name": "users", "col_name": "email", "score": 0.99},   # dup, must not duplicate
        {"table_name": "orders", "col_name": "total", "score": 0.77},  # new
    ]

    cols, tbls = _merge_seed_candidates([existing_col], [existing_tbl], seeds, verbose=False)

    assert len(cols) == 2
    added = [c for c in cols if c.table_name == "orders"]
    assert len(added) == 1 and added[0].col_name == "total"
    assert added[0].similarity == 0.77
    assert added[0].table_id == ""   # synthetic — no DB lookup performed for the seed

    assert {t.table_name for t in tbls} == {"users", "orders"}


def test_merge_seed_candidates_noop_without_seeds():
    from query.retrieval_v2 import _merge_seed_candidates

    cols, tbls = _merge_seed_candidates([1, 2], [3], None)
    assert cols == [1, 2] and tbls == [3]

    cols, tbls = _merge_seed_candidates([1, 2], [3], [])
    assert cols == [1, 2] and tbls == [3]


def test_tier2_sql_execution_state_none_preserves_prior_behavior(monkeypatch):
    """Regression guard: execution_state=None (the default) must recompute
    temporal parsing and pass seed_candidates=None to select_retrieval, exactly
    as _tier2_sql did before this refactor."""
    import query.temporal_parser as temporal_parser
    import query.retrieval_select as retrieval_select
    import query.slm_layer as slm_layer
    import veda_hybrid

    _stop_envelope(monkeypatch)
    calls = {"temporal": 0, "kwargs": None}

    class _TF:
        temporal_filter = "computed-fresh"

    def fake_temporal(q):
        calls["temporal"] += 1
        return _TF()

    class _Sel:
        columns = []
        join_path = []

    def fake_select_retrieval(**kwargs):
        calls["kwargs"] = kwargs
        return _Sel()

    class _L3:
        error = "stop-for-test"
        ir_json = None

    monkeypatch.setattr(temporal_parser, "run_temporal_parser", fake_temporal)
    monkeypatch.setattr(retrieval_select, "select_retrieval", fake_select_retrieval)
    monkeypatch.setattr(slm_layer, "run_slm_layer", lambda **k: _L3())

    result = veda_hybrid._tier2_sql("how many users", sm={}, all_cols=[], verbose=False,
                                    execution_state=None)

    assert result is None
    assert calls["temporal"] == 1
    assert calls["kwargs"]["seed_candidates"] is None


def test_tier2_sql_reuses_execution_state(monkeypatch):
    """With a populated ExecutionState: temporal parser is NOT re-run, and
    select_retrieval receives Tier1's candidate_fields as seed_candidates."""
    import query.temporal_parser as temporal_parser
    import query.retrieval_select as retrieval_select
    import query.slm_layer as slm_layer
    import veda_hybrid
    from veda.execution_state import ExecutionState

    _stop_envelope(monkeypatch)
    calls = {"temporal": 0, "kwargs": None}

    def fake_temporal(q):
        calls["temporal"] += 1
        return type("TF", (), {"temporal_filter": None})()

    class _Sel:
        columns = []
        join_path = []

    def fake_select_retrieval(**kwargs):
        calls["kwargs"] = kwargs
        return _Sel()

    class _L3:
        error = "stop-for-test"
        ir_json = None

    monkeypatch.setattr(temporal_parser, "run_temporal_parser", fake_temporal)
    monkeypatch.setattr(retrieval_select, "select_retrieval", fake_select_retrieval)
    monkeypatch.setattr(slm_layer, "run_slm_layer", lambda **k: _L3())

    es = ExecutionState()
    es.temporal_result = type("TR", (), {"temporal_filter": "from-tier1"})()
    es.candidate_fields = [{"table_name": "users", "col_name": "email", "score": 0.9}]
    es.refusal_reason = None

    result = veda_hybrid._tier2_sql("how many users", sm={}, all_cols=[], verbose=False,
                                    execution_state=es)

    assert result is None
    assert calls["temporal"] == 0, "temporal parser must be reused, not recomputed"
    assert calls["kwargs"]["seed_candidates"] == es.candidate_fields


def test_repair_hint_seeded_from_tier1_refusal(monkeypatch):
    """When execution_state carries a refusal_reason and the repair loop is
    enabled, attempt 0 already carries Tier1's reason via the EXISTING
    _repair_hint_for mechanism — not a new retry framework."""
    import config
    import query.retrieval_select as retrieval_select
    import query.slm_layer as slm_layer
    import veda_hybrid
    from veda.execution_state import ExecutionState

    _stop_envelope(monkeypatch)
    monkeypatch.setattr(config, "VALIDATION_REPAIR_LOOP_ENABLED", True)
    monkeypatch.setattr(config, "VALIDATION_MAX_REPAIR_ATTEMPTS", 2)

    seen_queries = []

    class _Sel:
        columns = []
        join_path = []

    class _L3:
        error = "stop-for-test"
        ir_json = None

    monkeypatch.setattr(retrieval_select, "select_retrieval", lambda **k: _Sel())

    def fake_run_slm_layer(**kwargs):
        seen_queries.append(kwargs["query"])
        return _L3()

    monkeypatch.setattr(slm_layer, "run_slm_layer", fake_run_slm_layer)

    es = ExecutionState()
    es.temporal_result = type("TR", (), {"temporal_filter": None})()
    es.refusal_reason = "no_table"

    veda_hybrid._tier2_sql("how many users", sm={}, all_cols=[], verbose=False,
                           execution_state=es)

    # run_slm_layer's error short-circuits the loop after attempt 0 — only one call.
    assert len(seen_queries) == 1
    assert "[REPAIR]" in seen_queries[0]
    assert "no_table" in seen_queries[0]


def test_serialize_strips_internal_only_keys():
    """The actual bug found in review: ExecutionState (under the "context" key) and
    the full debug trace (under "trace") must never reach an HTTP caller.
    inference/routes/hybrid.py::_serialize() is the ONE place every head result
    (SQL/Tier-2/RAG/hybrid/NoSQL) passes through before becoming a wire response —
    this is the actual enforcement point, not just a docstring claim."""
    import inference.routes.hybrid as hybrid_route
    from veda.execution_state import ExecutionState
    from query.multi_result import MultiResult, SubResult, STATUS_OK

    es = ExecutionState()
    es.refusal_reason = "no_table"
    es.candidate_fields = [{"table_name": "users", "col_name": "email", "score": 0.9}]

    head_result = {
        "status": "answered", "ok": True, "cols": ["id"], "rows": [[1]],
        "trace": {"query_understanding": {"secret": "internal-only"}},
        "context": es,
    }
    multi = MultiResult(items=[SubResult("how many users", STATUS_OK, "deterministic", head_result)])

    payload = hybrid_route._serialize(multi)

    assert "context" not in payload["items"][0]["result"]
    assert "trace" not in payload["items"][0]["result"]
    # legitimate fields must survive the strip
    assert payload["items"][0]["result"]["cols"] == ["id"]
    assert payload["items"][0]["result"]["rows"] == [[1]]
    assert payload["items"][0]["result"]["status"] == "answered"


def test_serialize_strips_context_even_as_a_bare_dataclass_field():
    """Belt-and-suspenders: if "context"/"trace" ever became a dataclass's OWN
    field (not just a plain-dict key), it must still be stripped — asdict() runs
    through the same dict-filtering branch, not bypassed."""
    import inference.routes.hybrid as hybrid_route
    from dataclasses import dataclass

    @dataclass
    class FakeResult:
        cols: list
        context: object = None

    payload = hybrid_route._serialize(FakeResult(cols=["a"], context={"leak": True}))
    assert "context" not in payload
    assert payload["cols"] == ["a"]


def test_serialize_converts_decimal_to_float_not_string():
    """Regression: psycopg2 returns Decimal for NUMERIC/SUM/AVG columns (e.g.
    monetary "amount" fields). Decimal used to fall through to the generic
    str(obj) branch, turning every such value into a STRING on the wire
    (e.g. "423.000") — which silently broke apps/chat/visualization.py's
    _is_numeric()/_to_number() downstream (that code's own comment already
    assumes it receives a real Decimal to convert, not a pre-stringified
    one), so no chart was ever produced for a query whose measure was a
    NUMERIC/DECIMAL column — only INTEGER aggregates (COUNT(*)) worked,
    since those already survive as native JSON ints."""
    import inference.routes.hybrid as hybrid_route
    from decimal import Decimal

    out = hybrid_route._serialize(Decimal("423.000"))
    assert out == 423.0
    assert isinstance(out, float)


def test_serialize_decimal_inside_rows_still_a_real_number():
    """Same fix, exercised through the actual nested shape a query result
    takes (rows = list of tuples with mixed types) — not just the bare
    Decimal case above."""
    import inference.routes.hybrid as hybrid_route
    from decimal import Decimal

    payload = hybrid_route._serialize({
        "cols": ["amount", "label"],
        "rows": [[Decimal("423.000"), "rent"], [Decimal("50000.00"), "Rent"]],
    })
    amounts = [row[0] for row in payload["rows"]]
    assert amounts == [423.0, 50000.0]
    assert all(isinstance(v, float) for v in amounts)


def test_serialize_still_passes_through_primitives_unchanged():
    """Regression guard: the Decimal special-case must not affect the
    existing str/int/float/bool/None passthrough behavior."""
    import inference.routes.hybrid as hybrid_route

    assert hybrid_route._serialize("hello") == "hello"
    assert hybrid_route._serialize(42) == 42
    assert hybrid_route._serialize(3.14) == 3.14
    assert hybrid_route._serialize(True) is True
    assert hybrid_route._serialize(None) is None


def test_reuse_log_does_not_overstate_query_understanding_reuse(monkeypatch, capsys):
    """The other confirmed bug: the reuse log must only claim what's ACTUALLY
    functionally consumed. query_understanding/sql_planning are populated on
    ExecutionState but not read by any Tier2 decision — the log must not claim
    otherwise, even when those fields are non-empty."""
    import query.temporal_parser as temporal_parser
    import query.retrieval_select as retrieval_select
    import query.slm_layer as slm_layer
    import veda_hybrid
    from veda.execution_state import ExecutionState

    _stop_envelope(monkeypatch)
    monkeypatch.setattr(retrieval_select, "select_retrieval",
                        lambda **k: type("Sel", (), {"columns": [], "join_path": []})())
    monkeypatch.setattr(slm_layer, "run_slm_layer",
                        lambda **k: type("L3", (), {"error": "stop-for-test", "ir_json": None})())

    es = ExecutionState()
    es.temporal_result = type("TR", (), {"temporal_filter": None})()
    es.query_understanding = {"intent": "SIMPLE", "aggregation": "count"}  # non-empty, unused
    es.candidate_fields = []   # deliberately empty — no seeds this time

    veda_hybrid._tier2_sql("how many users", sm={}, all_cols=[], verbose=True,
                           execution_state=es)

    out = capsys.readouterr().out
    assert "Query Understanding" not in out
    assert "Temporal" in out


def test_repair_hint_not_seeded_when_loop_disabled(monkeypatch):
    """When VALIDATION_REPAIR_LOOP_ENABLED is off (the default), a refusal_reason
    on execution_state must NOT alter the query — zero behavior change when the
    flag is off, matching the existing repair loop's own contract."""
    import config
    import query.retrieval_select as retrieval_select
    import query.slm_layer as slm_layer
    import veda_hybrid
    from veda.execution_state import ExecutionState

    _stop_envelope(monkeypatch)
    monkeypatch.setattr(config, "VALIDATION_REPAIR_LOOP_ENABLED", False)

    seen_queries = []

    class _Sel:
        columns = []
        join_path = []

    class _L3:
        error = "stop-for-test"
        ir_json = None

    monkeypatch.setattr(retrieval_select, "select_retrieval", lambda **k: _Sel())

    def fake_run_slm_layer(**kwargs):
        seen_queries.append(kwargs["query"])
        return _L3()

    monkeypatch.setattr(slm_layer, "run_slm_layer", fake_run_slm_layer)

    es = ExecutionState()
    es.temporal_result = type("TR", (), {"temporal_filter": None})()
    es.refusal_reason = "no_table"

    veda_hybrid._tier2_sql("how many users", sm={}, all_cols=[], verbose=False,
                           execution_state=es)

    assert seen_queries == ["how many users"]
    assert "[REPAIR]" not in seen_queries[0]
