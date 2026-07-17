"""Regression tests for the Tier2 "Recommended Projection" wiring (2026-07):
Tier2's IR-generation SLM previously had zero projection guidance (rule 3's
"only include columns directly relevant... omit irrelevant ones" was the only
instruction), so it defaulted to over-inclusion — the observed bug: "Show top
customers by revenue" returned id/created_at/updated_at/created_by alongside
customer_name/revenue.

Covers:
1. _tier2_sql() computes recommended_projection via the EXISTING
   veda/routing.py::recommended_projection() (the same function Tier1 already
   uses) and passes it into run_slm_layer — no duplicate projection logic.
2. Audit columns (importance_class=LOW) are excluded; the display column and
   HIGH-importance columns survive.
3. run_slm_layer threads the value into BOTH the langgraph (default) and
   non-langgraph prompt builders, each rendering a "Recommended Projection"
   block distinct from the full column reference — and renders NOTHING extra
   when None (existing callers see an unchanged prompt).

Pure-python: no DB, no network, no live Ollama — envelope_slm.emit_envelope
and the actual SLM call are monkeypatched, exactly mirroring
tests/test_execution_state_reuse.py's established pattern.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))

TABLE = "customers"
SM = {"columns": {
    f"{TABLE}.id":            {"table_name": TABLE, "col_name": "id", "importance_class": "LOW"},
    f"{TABLE}.created_at":    {"table_name": TABLE, "col_name": "created_at", "importance_class": "LOW"},
    f"{TABLE}.updated_at":    {"table_name": TABLE, "col_name": "updated_at", "importance_class": "LOW"},
    f"{TABLE}.created_by_id": {"table_name": TABLE, "col_name": "created_by_id", "importance_class": "LOW"},
    f"{TABLE}.customer_name": {"table_name": TABLE, "col_name": "customer_name", "importance_class": "MEDIUM"},
    f"{TABLE}.revenue":       {"table_name": TABLE, "col_name": "revenue", "importance_class": "HIGH"},
    f"{TABLE}.status":        {"table_name": TABLE, "col_name": "status", "importance_class": "MEDIUM"},
}}
ALL_COLS = [f"{TABLE}.{c}" for c in
            ("id", "created_at", "updated_at", "created_by_id", "customer_name", "revenue", "status")]


def _stop_envelope(monkeypatch):
    import query.envelope_slm as envelope_slm

    def _raise(*a, **k):
        raise RuntimeError("no network in tests")
    monkeypatch.setattr(envelope_slm, "emit_envelope", _raise)


def _retrieval_results():
    from ingestion.vector_store import RetrievalResult
    cols = ("id", "created_at", "updated_at", "created_by_id", "customer_name", "revenue", "status")
    return [RetrievalResult(col_id=f"{TABLE}.{c}", col_name=c, table_id="t1",
                            table_name=TABLE, semantic_type="UNKNOWN", similarity=0.5)
            for c in cols]


def test_tier2_sql_computes_and_passes_recommended_projection(monkeypatch):
    """End-to-end (mocked SLM/network) wiring proof: _tier2_sql calls the
    EXISTING recommended_projection() and forwards a non-empty, correctly
    narrowed result into run_slm_layer — audit columns excluded, display +
    HIGH-importance columns included."""
    import query.retrieval_select as retrieval_select
    import query.slm_layer as slm_layer
    import veda_hybrid

    _stop_envelope(monkeypatch)
    captured = {}

    class _Sel:
        columns = _retrieval_results()
        tables = [TABLE]
        join_path = []

    def fake_select_retrieval(**kwargs):
        return _Sel()

    class _L3:
        error = "stop-for-test"
        ir_json = None

    def fake_run_slm_layer(**kwargs):
        captured.update(kwargs)
        return _L3()

    monkeypatch.setattr(retrieval_select, "select_retrieval", fake_select_retrieval)
    monkeypatch.setattr(slm_layer, "run_slm_layer", fake_run_slm_layer)

    veda_hybrid._tier2_sql("Show top customers by revenue", sm=SM, all_cols=ALL_COLS,
                           verbose=False, execution_state=None)

    rec = captured.get("recommended_projection")
    assert rec, "recommended_projection must be computed and passed, not left None"
    rec_names = {r.col_name for r in rec}

    assert "customer_name" in rec_names, "display column must survive"
    assert "revenue" in rec_names, "HIGH-importance column must survive"
    for audit_col in ("id", "created_at", "updated_at", "created_by_id"):
        assert audit_col not in rec_names, f"audit column {audit_col!r} must be excluded"


def test_tier2_sql_without_execution_state_uses_sel_tables_as_primary(monkeypatch):
    """Cold Tier2 call (execution_state=None, no Tier1 primary_table to reuse)
    still computes a projection — falling back to select_retrieval's own
    top-ranked table (sel.tables[0]), not skipping the feature entirely."""
    import query.retrieval_select as retrieval_select
    import query.slm_layer as slm_layer
    import veda_hybrid

    _stop_envelope(monkeypatch)
    captured = {}

    class _Sel:
        columns = _retrieval_results()
        tables = [TABLE]
        join_path = []

    monkeypatch.setattr(retrieval_select, "select_retrieval", lambda **k: _Sel())

    class _L3:
        error = "stop-for-test"
        ir_json = None

    def fake_run_slm_layer(**kwargs):
        captured.update(kwargs)
        return _L3()

    monkeypatch.setattr(slm_layer, "run_slm_layer", fake_run_slm_layer)

    veda_hybrid._tier2_sql("revenue by region", sm=SM, all_cols=ALL_COLS,
                           verbose=False, execution_state=None)

    assert captured.get("recommended_projection"), \
        "projection must still be computed without an ExecutionState"


def test_tier2_sql_recommended_projection_never_crashes_on_missing_tables(monkeypatch):
    """select_retrieval results with no .tables (or empty) and no ExecutionState
    primary_table must degrade to None, never raise — _tier2_sql's own
    try/except around the projection block is the safety net, verified here
    end-to-end rather than just by inspection."""
    import query.retrieval_select as retrieval_select
    import query.slm_layer as slm_layer
    import veda_hybrid

    _stop_envelope(monkeypatch)
    captured = {}

    class _Sel:
        columns = []
        tables = []
        join_path = []

    monkeypatch.setattr(retrieval_select, "select_retrieval", lambda **k: _Sel())

    class _L3:
        error = "stop-for-test"
        ir_json = None

    def fake_run_slm_layer(**kwargs):
        captured.update(kwargs)
        return _L3()

    monkeypatch.setattr(slm_layer, "run_slm_layer", fake_run_slm_layer)

    result = veda_hybrid._tier2_sql("how many customers", sm=SM, all_cols=ALL_COLS,
                                    verbose=False, execution_state=None)

    assert result is None
    assert captured.get("recommended_projection") is None


# ---------------------------------------------------------------------------
# Prompt-rendering level — slm_layer.py (non-langgraph path)
# ---------------------------------------------------------------------------

def test_build_user_message_renders_recommended_projection_block():
    from query.slm_layer import _build_user_message

    cols = _retrieval_results()
    rec = [c for c in cols if c.col_name in ("customer_name", "revenue")]
    msg = _build_user_message("Show top customers by revenue", None, cols, [],
                              recommended_projection_results=rec)
    assert "Recommended Projection" in msg
    assert f"{TABLE}.customer_name" in msg
    assert f"{TABLE}.revenue" in msg


def test_build_user_message_omits_block_when_none():
    """An existing caller that never passes recommended_projection_results
    must see a byte-for-byte unchanged prompt."""
    from query.slm_layer import _build_user_message

    cols = _retrieval_results()
    msg_without = _build_user_message("Show top customers by revenue", None, cols, [])
    msg_explicit_none = _build_user_message("Show top customers by revenue", None, cols, [],
                                            recommended_projection_results=None)
    assert "Recommended Projection" not in msg_without
    assert msg_without == msg_explicit_none


# ---------------------------------------------------------------------------
# Prompt-rendering level — lg_nodes.py (langgraph path, USE_LANGGRAPH default)
# ---------------------------------------------------------------------------

def test_node_select_columns_renders_recommended_projection_block(monkeypatch):
    from query.lg_nodes import node_select_columns

    def fake_call_node(system_prompt, user_msg):
        fake_call_node.captured_user_msg = user_msg
        return {"selected_col_ids": [f"{TABLE}.revenue"], "group_by_col_id": None,
                "order_by_col_id": None, "order_direction": "ASC"}

    import query.lg_nodes as lg_nodes
    monkeypatch.setattr(lg_nodes, "_call_node", fake_call_node)

    top_k = [{"col_id": f"{TABLE}.{c}", "col_name": c, "table_id": "t1", "table_name": TABLE,
             "semantic_type": "UNKNOWN"}
             for c in ("id", "created_at", "customer_name", "revenue")]
    state = {
        "query": "Show top customers by revenue",
        "intent": "SELECT",
        "top_k_columns": top_k,
        "primary_table_id": "t1",
        "secondary_table_ids": [],
        "must_include": [],
        "recommended_projection": [
            {"col_id": f"{TABLE}.customer_name", "col_name": "customer_name", "table_name": TABLE},
            {"col_id": f"{TABLE}.revenue", "col_name": "revenue", "table_name": TABLE},
        ],
        "errors": [],
        "node_times": {},
    }

    node_select_columns(state)

    msg = fake_call_node.captured_user_msg
    assert "RECOMMENDED PROJECTION" in msg
    assert f"{TABLE}.customer_name" in msg
    assert f"{TABLE}.revenue" in msg


def test_node_select_columns_omits_block_when_absent(monkeypatch):
    """No recommended_projection key at all (existing state shape, pre-this-
    change callers) must render the exact same prompt as before."""
    from query.lg_nodes import node_select_columns

    def fake_call_node(system_prompt, user_msg):
        fake_call_node.captured_user_msg = user_msg
        return {"selected_col_ids": [], "group_by_col_id": None,
                "order_by_col_id": None, "order_direction": "ASC"}

    import query.lg_nodes as lg_nodes
    monkeypatch.setattr(lg_nodes, "_call_node", fake_call_node)

    top_k = [{"col_id": f"{TABLE}.id", "col_name": "id", "table_id": "t1", "table_name": TABLE,
             "semantic_type": "UNKNOWN"}]
    state = {
        "query": "how many customers", "intent": "COUNT", "top_k_columns": top_k,
        "primary_table_id": "t1", "secondary_table_ids": [], "must_include": [],
        "errors": [], "node_times": {},
    }

    node_select_columns(state)
    assert "RECOMMENDED PROJECTION" not in fake_call_node.captured_user_msg
