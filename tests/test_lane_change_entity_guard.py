import traceback
"""A follow-up answered from a different LANE must not become the conversation's subject.

Measured 2026-09-23: a conversation on `assets_asset` asked "only the Nagpur ones", the RAG
head answered ("The provided context does not contain information specific to Nagpur",
citing an employee handbook), that turn counted as `answered` so memory wrote it, and the
frame moved to `Samta-Employee Handbook April 2026` — anchoring every later turn to a
document nobody had asked about.

Pure: no DB, no SLM, no network. Run: `python tests/test_lane_change_entity_guard.py`
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from chatbot.memory.frame import keep_entity_on_lane_change as guard  # noqa: E402

SQL_FRAME = {"entity": "assets_asset", "entity_display": "Assets",
             "route": "deterministic", "source_id": 2}
DOC_HARVEST = {"entity": "Samta-Employee Handbook April 2026",
               "entity_display": "Samta Employee Handbook", "route": "rag",
               "source_id": 3, "last_row_count": 0, "last_sql": None}


# ── the measured failure ──────────────────────────────────────────────────────────────
def test_a_document_answer_does_not_take_over_a_table_conversation():
    out = guard(SQL_FRAME, DOC_HARVEST, referential=True)
    assert out["entity"] == "assets_asset"
    assert out["route"] == "deterministic"
    assert out["source_id"] == 2


def test_the_rest_of_the_harvest_is_still_recorded():
    """Only the claim about WHAT THE CONVERSATION IS ABOUT is refused. What the turn
    actually ran is a fact and is kept."""
    out = guard(SQL_FRAME, dict(DOC_HARVEST, last_row_count=7, last_sql="SELECT 1"),
                referential=True)
    assert out["last_row_count"] == 7
    assert out["last_sql"] == "SELECT 1"


# ── everything else must be untouched ─────────────────────────────────────────────────
def test_a_new_topic_may_go_anywhere():
    out = guard(SQL_FRAME, DOC_HARVEST, referential=False)
    assert out["entity"] == "Samta-Employee Handbook April 2026"


def test_same_lane_is_untouched_however_far_the_table_moves():
    same = {"entity": "assets_leaselisting", "route": "deterministic", "source_id": 2}
    out = guard(SQL_FRAME, same, referential=True)
    assert out["entity"] == "assets_leaselisting"


def test_a_document_conversation_finding_a_table_is_left_to_the_other_guard():
    doc_frame = {"entity": "handbook.pdf", "route": "rag", "source_id": 3}
    sql_harvest = {"entity": "assets_asset", "route": "deterministic", "source_id": 2}
    out = guard(doc_frame, sql_harvest, referential=True)
    assert out["entity"] == "assets_asset"


def test_nothing_to_compare_is_not_a_reason_to_act():
    assert guard({}, DOC_HARVEST, referential=True)["entity"] == DOC_HARVEST["entity"]
    assert guard(SQL_FRAME, {}, referential=True) == {}
    no_route = {"entity": "assets_asset", "source_id": 2}
    assert guard(no_route, DOC_HARVEST, referential=True)["entity"] == DOC_HARVEST["entity"]
    assert guard(SQL_FRAME, {"entity": "x"}, referential=True)["entity"] == "x"


def test_a_frame_with_no_entity_cannot_be_protected():
    out = guard({"route": "deterministic"}, DOC_HARVEST, referential=True)
    assert out["entity"] == DOC_HARVEST["entity"]


def test_the_input_is_never_mutated():
    h = dict(DOC_HARVEST)
    guard(SQL_FRAME, h, referential=True)
    assert h["entity"] == "Samta-Employee Handbook April 2026"


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


# ── a clarification from a FAILED turn must not swallow the next question ─────────────
def _classify(message, *, pending, frame, delta_type, action="clarify_reply"):
    """Drive only the decision under test: given what the classifier produced, does the
    pending request still take the turn?"""
    from chatbot import nodes
    # The branch is a pure function of (action, pending, frame.entity, delta_type).
    CONTINUATIONS = ("refine", "replace", "remove", "drill_down", "drill_up")
    if action == "clarify_reply" and not pending:
        return "followup"
    if (action == "clarify_reply" and pending and (frame or {}).get("entity")
            and (delta_type in CONTINUATIONS
                 or (delta_type in ("ambiguous", None)
                     and nodes._CONTINUATION_SHAPE_RE.match(message or "")))):
        return "followup"
    return action


def test_a_continuation_beats_a_clarification_left_by_a_failed_turn():
    """Measured 2026-09-24: after an answered question and then a REFUSED one, the next
    message reached the engine as "list 5 ledger with highest amount for only the Nagpur
    ones" — the dead question glued to the live one."""
    assert _classify("only the Nagpur ones", pending="list 5 ledger with highest amount",
                     frame={"entity": "assets_asset"}, delta_type="refine") == "followup"


def test_with_no_frame_the_pending_request_still_wins():
    """Nothing to continue means the pending request IS the better reading."""
    assert _classify("Nagpur", pending="which city?", frame={},
                     delta_type="refine") == "clarify_reply"


def test_an_uncertain_delta_is_not_evidence_of_anything():
    for dt in ("new_topic", "ambiguous", None):
        assert _classify("Nagpur", pending="which city?",
                         frame={"entity": "assets_asset"}, delta_type=dt) == "clarify_reply"


def test_an_ordinary_turn_is_untouched():
    assert _classify("how many assets", pending="which city?",
                     frame={"entity": "assets_asset"}, delta_type="refine",
                     action="answer") == "answer"
