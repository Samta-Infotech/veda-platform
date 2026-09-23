"""The turn's own understanding, surfaced to the user.

`build_explain` has always told the user how the ANSWER was produced. Nothing told
them what their QUESTION was taken to mean — and for a follow-up, that is the half
that can silently go wrong: "what about Mumbai" resolved against the wrong remembered
topic produces a confident, well-explained answer to a question nobody asked.

`_context_used` reports only what the turn ALREADY did — the frame it merged and the
delta it applied. It never re-derives, infers, or asks a model, so it cannot claim an
understanding the turn did not act on.

Pure-python, no Django settings needed for the chatbot half; the accumulator half
imports apps.chat.turn_events, which is also settings-free.

Run: ``pytest tests/test_context_understanding.py``
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.nodes import _context_used  # noqa: E402

_FRAME = {"entity": "assets_asset", "entity_display": "Assets", "source_id": 2,
          "filters": [{"field": "City", "operator": "equals", "value": "Mumbai"},
                      {"field": "Status", "operator": "equals", "value": "ACTIVE"}]}

_RESOLVED = "what about Mumbai (for Assets (assets_asset), City equals Mumbai)"


# ---------------------------------------------------------------------------
# what it reports
# ---------------------------------------------------------------------------

def test_it_reports_the_remembered_topic_and_filters():
    used = _context_used(_FRAME, "replace", "City", "Mumbai", _RESOLVED, "what about Mumbai")
    assert used["carried"]["entity"] == "Assets"
    assert used["carried"]["filters"] == ["City equals Mumbai", "Status equals ACTIVE"]
    assert used["carried"]["source_id"] == 2


def test_it_reports_the_operation_and_its_target():
    used = _context_used(_FRAME, "replace", "City", "Mumbai", _RESOLVED, "what about Mumbai")
    assert used["changed"] == {"operation": "replace", "field": "City", "value": "Mumbai"}


def test_it_reports_the_text_actually_sent_to_the_engine():
    used = _context_used(_FRAME, "replace", "City", "Mumbai", _RESOLVED, "what about Mumbai")
    assert used["resolved_query"] == _RESOLVED


def test_a_removal_reports_no_value():
    used = _context_used(_FRAME, "remove", "City", "", "Assets (assets_asset)",
                         "drop the city filter")
    assert used["changed"] == {"operation": "remove", "field": "City"}
    assert "value" not in used["changed"]


def test_an_operation_with_no_target_still_names_itself():
    used = _context_used(_FRAME, "refine", "", "", _RESOLVED, "only the active ones")
    assert used["changed"] == {"operation": "refine"}


def test_a_shape_delta_names_the_slot_it_changed():
    frame = {**_FRAME, "limit": 10}
    used = _context_used(frame, "replace", "limit", "10",
                         "make it top 10 (for Assets (assets_asset))", "make it top 10")
    assert used["changed"] == {"operation": "replace", "field": "limit", "value": "10"}


# ---------------------------------------------------------------------------
# when it must say nothing — an empty panel is a claim of its own
# ---------------------------------------------------------------------------

def test_a_first_question_reports_nothing():
    assert _context_used({}, "new_topic", "", "", "how many assets", "how many assets") is None


def test_a_frame_with_no_entity_reports_nothing():
    assert _context_used({"filters": []}, "refine", "", "", "x", "y") is None


def test_a_turn_whose_resolved_query_is_just_the_message_reports_nothing():
    """new_topic renders the message unchanged — there is no context to show, and
    "we understood: <your own words>" is noise."""
    assert _context_used(_FRAME, "new_topic", "", "", "how many invoices",
                         "how many invoices") is None
    assert _context_used(_FRAME, "ambiguous", "", "", "  what about it  ",
                         "what about it") is None


def test_a_filter_with_no_value_is_not_rendered_as_none():
    frame = {**_FRAME, "filters": [{"field": "City", "operator": "equals", "value": None},
                                   {"field": "Status", "operator": "equals",
                                    "value": "ACTIVE"}]}
    used = _context_used(frame, "refine", "", "", _RESOLVED, "only the active ones")
    assert used["carried"]["filters"] == ["Status equals ACTIVE"]


# ---------------------------------------------------------------------------
# the accumulator and the persisted metadata
# ---------------------------------------------------------------------------

def test_the_accumulator_folds_and_persists_the_context_event():
    from apps.chat.turn_events import TurnEventAccumulator

    acc = TurnEventAccumulator()
    acc.consume("context", {"carried": {"entity": "Assets"}, "resolved_query": _RESOLVED})
    acc.consume("usage", {"total_tokens": 10})
    md = acc.metadata()
    assert md["context"]["carried"]["entity"] == "Assets"
    assert md["context"]["resolved_query"] == _RESOLVED


def test_a_turn_with_no_context_persists_no_context_key():
    """Absent, not null — the envelope convention this codebase already follows."""
    from apps.chat.turn_events import TurnEventAccumulator

    acc = TurnEventAccumulator()
    acc.consume("usage", {"total_tokens": 10})
    assert "context" not in acc.metadata()


def test_an_unknown_event_still_does_not_break_the_turn():
    from apps.chat.turn_events import TurnEventAccumulator

    acc = TurnEventAccumulator()
    acc.consume("something_new", {"x": 1})
    assert "context" not in acc.metadata()
