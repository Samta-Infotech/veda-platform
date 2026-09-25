import traceback
"""A turn the classifier could not PLACE must not rewrite the conversation's subject.

Measured 2026-09-24, 3 runs of 3: mid-drill, a REFUSED turn made the next follow-up come
back `ambiguous`. An ambiguous turn carries no context (carry_state excludes it), so the
engine answered the bare fragment, landed on a different table, and `is_topic_switch` read
that as a deliberate change of subject — erasing a live drill path at both call sites (the
drill-stack reset in memory_write_node and the frame reset in merge_frame_post_execution).

Pure: no DB, no SLM, no network. Run: `python tests/test_unplaced_turn_subject_guard.py`
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from chatbot.memory.frame import (  # noqa: E402
    hold_subject_on_unplaced_turn as guard,
    is_topic_switch,
    merge_frame_post_execution,
)

FRAME = {"entity": "assets_asset", "entity_display": "Assets",
         "route": "deterministic", "source_id": 2,
         "filters": [{"field": "city_name", "column": "city_name", "value": "Nagpur"}]}
STRAY = {"entity": "client_support_clientsupport", "entity_display": "Support",
         "route": "deterministic", "source_id": 2,
         "last_row_count": 7, "last_sql": "SELECT 1"}
STACK = [{"dimension": "city_name", "column": "city_name", "value": "Nagpur"}]


# ── the measured failure ──────────────────────────────────────────────────────────────
def test_an_ambiguous_turn_does_not_take_over_a_live_drill():
    out = guard(FRAME, STRAY, "ambiguous", STACK)
    assert out["entity"] == "assets_asset"
    assert out["entity_display"] == "Assets"


def test_holding_the_entity_is_what_makes_the_reset_not_fire():
    """Both call sites read is_topic_switch, so the guard has to close both."""
    assert is_topic_switch(FRAME, STRAY) is True
    assert is_topic_switch(FRAME, guard(FRAME, STRAY, "ambiguous", STACK)) is False


def test_the_frame_reset_inside_merge_is_closed_too():
    merged = merge_frame_post_execution(
        FRAME, guard(FRAME, STRAY, "ambiguous", STACK), "ambiguous", "t", "s")
    assert merged["entity"] == "assets_asset"
    assert merged["filters"] == FRAME["filters"]


def test_what_actually_ran_is_still_recorded():
    """Only the subject claim is refused — the facts about the turn are written."""
    out = guard(FRAME, STRAY, "ambiguous", STACK)
    assert out["last_row_count"] == 7
    assert out["last_sql"] == "SELECT 1"


# ── deliberately narrow ───────────────────────────────────────────────────────────────
def test_with_no_drill_live_an_ambiguous_turn_may_move_the_frame():
    """No stack means nothing to protect; a new subject phrased oddly stays free to move."""
    assert guard(FRAME, STRAY, "ambiguous", [])["entity"] == STRAY["entity"]
    assert guard(FRAME, STRAY, "ambiguous", None)["entity"] == STRAY["entity"]


def test_every_other_delta_is_untouched():
    for dt in ("new_topic", "refine", "replace", "remove",
               "drill_down", "drill_up", "compare"):
        assert guard(FRAME, STRAY, dt, STACK)["entity"] == STRAY["entity"], dt


def test_the_same_entity_is_a_no_op():
    same = dict(STRAY, entity="assets_asset")
    assert guard(FRAME, same, "ambiguous", STACK) == same


def test_missing_either_entity_does_nothing():
    assert guard({}, STRAY, "ambiguous", STACK) is STRAY
    assert guard(FRAME, {}, "ambiguous", STACK) == {}
    assert guard(None, STRAY, "ambiguous", STACK) is STRAY
    assert guard(FRAME, dict(STRAY, entity=None), "ambiguous", STACK)["entity"] is None


def test_the_input_is_not_mutated():
    before = dict(STRAY)
    guard(FRAME, STRAY, "ambiguous", STACK)
    assert STRAY == before


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
