"""Tier-2 LangGraph per-node thinking events (2026-07-17).

The 5 Tier-2 nodes (classify_intent → select_entity → select_columns →
build_filters → assemble_ir) each emit an SSE progress event via the on_event
callback carried in the graph state, so the previously-silent Tier-2 gap now
reports per-step progress — parity with run_rag_layer/run_hybrid_layer's own
sub-steps. Pure-python: exercises the _emit_step contract + the node-level emit
calls + the business-friendly display mapping. No SLM / no network / no DB.
Run from the repo root: ``pytest tests/test_tier2_thinking.py``"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# _emit_step — the state-carried progress callback contract
# ---------------------------------------------------------------------------

def test_emit_step_fires_with_contract():
    from query.lg_nodes import _emit_step
    seen = []
    state = {"on_event": lambda phase, msg, extra: seen.append((phase, msg, extra))}
    _emit_step(state, "tier2_columns", "Selecting the details to include...")
    assert seen == [("tier2_columns", "Selecting the details to include...", {})]


def test_emit_step_noop_without_callback():
    from query.lg_nodes import _emit_step
    _emit_step({}, "tier2_intent", "x")            # no on_event key
    _emit_step({"on_event": None}, "tier2_intent", "x")   # explicit None
    # nothing to assert beyond "does not raise"


def test_emit_step_never_raises_into_node():
    from query.lg_nodes import _emit_step

    def boom(phase, msg, extra):
        raise RuntimeError("callback blew up")

    # progress reporting must not be able to fail query building
    _emit_step({"on_event": boom}, "tier2_filters", "Applying your conditions...")


# ---------------------------------------------------------------------------
# Every Tier-2 node emits its own phase at start (order = graph order)
# ---------------------------------------------------------------------------

def test_all_tier2_nodes_emit_a_phase(monkeypatch):
    """Drive each node with a recording on_event and a stubbed SLM so no network
    is touched — assert each fires exactly its own tier2_* phase first."""
    import query.lg_nodes as ln

    # Stub the single SLM entry point every node funnels through.
    monkeypatch.setattr(ln, "_call_node", lambda *a, **k: {})

    cols = [{"col_id": "c1", "col_name": "amount", "table_id": "t1",
             "table_name": "txn", "semantic_type": "MONETARY", "similarity": 0.9}]
    base = {
        "query": "top payments",
        "top_k_columns": cols,
        "join_path": [],
        "must_include": [],
        "recommended_projection": [],
        "errors": [],
        "node_times": {},
        "primary_table_id": "t1",
        "selected_col_ids": ["c1"],
    }

    cases = [
        (ln.node_classify_intent, "tier2_intent"),
        (ln.node_select_entity,   "tier2_entity"),
        (ln.node_select_columns,  "tier2_columns"),
        (ln.node_build_filters,   "tier2_filters"),
        (ln.node_assemble_ir,     "tier2_assemble"),
    ]
    for fn, expected_phase in cases:
        seen = []
        state = dict(base, on_event=lambda p, m, e: seen.append(p))
        try:
            fn(state)
        except Exception:
            pass   # a node may still fail deeper on the stubbed SLM output — we only
                   # assert it emitted its progress phase BEFORE doing any work
        assert seen and seen[0] == expected_phase, (fn.__name__, seen)


# ---------------------------------------------------------------------------
# Display mapping — internal phase → business-friendly text, no jargon leak
# ---------------------------------------------------------------------------

def test_tier2_phase_display_messages_present_and_clean():
    from apps.chat.thinking_messages import business_friendly_message, THINKING_PHASE_MESSAGES
    for phase in ("tier2_intent", "tier2_entity", "tier2_columns",
                  "tier2_filters", "tier2_assemble"):
        msg = business_friendly_message(phase, "RAW_FALLBACK")
        assert msg in THINKING_PHASE_MESSAGES.values()
        assert msg != "RAW_FALLBACK"                       # actually mapped, not fallen back
        low = msg.lower()
        for banned in ("sql", "tier2", "tier-2", "langgraph", "slm", "llm", "schema", "table"):
            assert banned not in low, (phase, banned, msg)
