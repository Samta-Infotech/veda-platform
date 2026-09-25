import traceback
"""ARCHITECTURAL CONTRACT: what reaches VEDA Core as the query is text the user typed.

This exists so the regression it guards cannot quietly return. The conversation layer may
keep any derived or debug representation it likes, but the EXECUTION query must not be a
memory-enriched rewrite — the engine parses every word of it as data, and words this
layer added are indistinguishable there from words the user chose.

The one deliberate exception is the pair of pure-navigation deltas ("remove", "drill_up"),
whose own words name what to STOP doing rather than what to ask. They replay the user's
OWN earlier question, recorded verbatim when they asked it — still never invented text.

Pure: no DB, no SLM, no network. Run: `python tests/test_resolved_query_contract.py`
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

from chatbot.nodes import context_resolve_node        # noqa: E402

LEDGER = "accounts_generalledger"
LABEL = "Single Financial Transactions"
BASE_Q = "Show the latest 10 ledger entries."


def _frame(**over):
    f = {
        "entity": LEDGER, "entity_display": LABEL,
        "filters": [{"field": "Entry Type", "operator": "equals", "value": "debit"}],
        "group_by": [], "measures": [], "available_measures": [],
        "order_by": [], "limit": None, "source_id": 2,
        "route": "deterministic", "drill_path": [], "base_query": BASE_Q,
        "understanding": "", "last_sql": None, "last_row_count": None,
    }
    f.update(over)
    return f


def _run(message, delta_type, frame=None, action="followup", **extra):
    state = {"message": message, "history": [], "frame": frame if frame is not None else _frame(),
             "drill_stack": [], "delta_type": delta_type, "action": action}
    state.update(extra)
    return context_resolve_node(state, {})


# ── the contract ──────────────────────────────────────────────────────────────────────
def test_every_data_carrying_delta_sends_the_users_own_words():
    msg = "only the debit ones"
    for delta in ("refine", "replace", "drill_down", "new_topic", "ambiguous"):
        out = _run(msg, delta)
        assert out["resolved_query"] == msg, (
            f"{delta}: resolved_query was rewritten to {out['resolved_query']!r}")


def test_the_display_label_never_reaches_the_engine():
    for delta in ("refine", "replace", "drill_down"):
        out = _run("only the debit ones", delta)
        low = out["resolved_query"].lower()
        assert "financial" not in low
        assert LABEL.lower() not in low
        assert LEDGER not in low


def test_navigation_deltas_replay_the_users_earlier_question_not_invented_text():
    # `remove` needs a field the frame actually holds, or apply_context_delta declines it
    # (see the next test) — that guard is existing, measured behaviour, not something
    # this change introduces.
    out = _run("remove the entry type filter", "remove", delta_field="Entry Type",
               delta_value="entry type")
    assert out["resolved_query"] == BASE_Q, (
        f"remove: expected the user's own earlier question, got "
        f"{out['resolved_query']!r}")

    out = _run("go back", "drill_up")
    assert out["resolved_query"] == BASE_Q, (
        f"drill_up: expected the user's own earlier question, got "
        f"{out['resolved_query']!r}")


def test_a_declined_remove_keeps_the_users_own_words():
    """apply_context_delta refuses a delta naming a field the frame does not hold, and the
    node downgrades it to a refinement so the user's request still reaches the engine
    instead of silently re-running the previous query. Unchanged by this work; pinned so
    the boundary change cannot quietly break it."""
    out = _run("remove the city filter", "remove", delta_field="City", delta_value="city")
    assert out["resolved_query"] == "remove the city filter"
    assert out["delta_type"] == "refine"


def test_navigation_delta_without_a_recorded_base_query_keeps_the_message():
    out = _run("go back", "drill_up", frame=_frame(base_query=None))
    assert out["resolved_query"] == "go back"


# ── the structured half ───────────────────────────────────────────────────────────────
def test_the_context_travels_separately_and_carries_the_anchor():
    out = _run("only the debit ones", "refine")
    ctx = out["conversation_context"]
    assert ctx["entity_table"] == LEDGER
    assert ctx["user_message"] == "only the debit ones"
    assert "entity_display" not in ctx


def test_new_topic_sends_no_remembered_state():
    out = _run("how many assets are there", "new_topic", action="answer")
    assert out["resolved_query"] == "how many assets are there"
    assert out["conversation_context"] == {"user_message": "how many assets are there"}


def test_a_turn_with_no_frame_is_untouched():
    out = _run("how many assets are there", "new_topic", frame={}, action="answer")
    assert out["resolved_query"] == "how many assets are there"


# ── drill-up: root vs depth (VEDA_DRILLDOWN_10LEVEL_FEASIBILITY.md §K1) ───────────────
_TWO = [{"field": "Furnishing", "column": "furnishing", "operator": "equals", "value": "FULL"},
        {"field": "City", "column": "city_name", "operator": "equals", "value": "Nagpur"}]


def test_drill_up_to_the_root_replays_the_original_question_with_no_context():
    """Measured 2026-09-24: the root pop sent the base question WITH context, the engine
    treated it as context-dependent, and it came back ungrouped where the first ask had
    answered. Nothing remains to carry, so it goes out exactly as it did the first time."""
    out = _run("go back", "drill_up", frame=_frame(filters=_TWO[:1]),
               drill_stack=[{"dimension": "Furnishing", "value": "FULL"}])
    assert out["resolved_query"] == BASE_Q
    assert out["conversation_context"] == {"user_message": BASE_Q}


def test_drill_up_with_levels_left_still_carries_them():
    out = _run("go back", "drill_up", frame=_frame(filters=list(_TWO)),
               drill_stack=[{"dimension": "Furnishing", "value": "FULL"},
                            {"dimension": "City", "value": "Nagpur"}])
    assert out["resolved_query"] == BASE_Q
    ctx = out["conversation_context"]
    assert [f["column"] for f in ctx["filters"]] == ["furnishing"]
    assert ctx["entity_table"] == LEDGER


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print("PASS", name)
        except Exception:
            failed += 1; print("FAIL", name); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
