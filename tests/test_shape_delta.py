"""Shape deltas — changing WHAT the question measures/groups/ranks by, as opposed to
WHICH ROWS it looks at.

Two halves, tested separately because they fail differently:

  detect_shape_delta   reads the operation off the message, deterministically. The risk
                       here is a FALSE POSITIVE — a real new question ("top 5 cities by
                       assets") read as a re-ranking of the previous one.
  apply_context_delta  performs the mutation. The risk here is a WRONG WRITE — a value
                       the user never typed, or a slot the frame never held.

Assertions are written as ABSENCE wherever absence is the property that matters: "the
limit is now 10" passes on broken code that also kept the old ordering.

Pure-python, no Django settings needed (chatbot.memory.frame imports standalone)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from chatbot.memory.frame import (apply_context_delta, detect_shape_delta,
                                  render_frame_as_query)


def _frame(**overrides):
    base = {
        "entity": "accounts_generalledger", "entity_display": "General Ledger",
        "filters": [{"field": "City", "operator": "equals", "value": "Pune",
                     "source": "executed_sql"}],
        "group_by": ["year"], "measures": ["amount"],
        "order_by": [{"field": "amount", "desc": True}], "limit": 100,
        "drill_path": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# detect_shape_delta — what it must recognise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message,expected", [
    ("make it top 10", ("replace", "limit", "10")),
    ("top 25", ("replace", "limit", "25")),
    ("just the top 5", ("replace", "limit", "5")),
    ("show me the first 50 rows", ("replace", "limit", "50")),
    ("show all of them", ("remove", "limit", "")),
    ("remove the limit", ("remove", "limit", "")),
    ("show it by month instead", ("replace", "group_by", "month")),
    ("by city instead", ("replace", "group_by", "city")),
    ("group by region instead", ("replace", "group_by", "region")),
    ("sort by city instead", ("replace", "order_by", "city")),
    ("don't sort by amount", ("remove", "order_by", "amount")),
    ("do not sort by amount", ("remove", "order_by", "amount")),
    ("stop sorting by amount", ("remove", "order_by", "amount")),
])
def test_recognised_shape_changes(message, expected):
    assert detect_shape_delta(_frame(), message) == expected


# ---------------------------------------------------------------------------
# detect_shape_delta — what it must NOT claim
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "top 5 cities by assets",          # a new question that merely starts the same way
    "list 5 assets",
    "top 20 credit transaction",
    "show me the top 3 invoices",
    "how many assets are there",
    "what about Mumbai",               # a filter swap, not a shape change
    "only the active ones",            # a refinement
    "as a pie chart",                  # presentation
    "go back",
    "hi",
])
def test_does_not_misfire_on_a_real_message(message):
    assert detect_shape_delta(_frame(), message) is None


def test_no_misfire_across_the_whole_real_corpus():
    """Every distinct message ever sent through /api/v1/chat/query, against a frame
    holding every shape slot. All 299 are self-contained questions, so a single match
    would be a false positive."""
    import json
    from pathlib import Path

    corpus = (Path(__file__).resolve().parent.parent
              / "evaluation" / "conversation" / "corpus_real.jsonl")
    if not corpus.exists():                      # the eval harness is optional baggage
        pytest.skip("evaluation/conversation/corpus_real.jsonl not present")
    frame = _frame()
    hits = [json.loads(line)["message"] for line in corpus.read_text().splitlines()
            if line.strip() and detect_shape_delta(frame, json.loads(line)["message"])]
    assert hits == []


def test_a_slot_the_frame_never_held_is_not_invented():
    """"make it top 10" after a question with no row limit must NOT manufacture one —
    the user's own words still reach the engine, which is the recoverable outcome."""
    bare = _frame(limit=None, order_by=[], group_by=[])
    assert detect_shape_delta(bare, "make it top 10") is None
    assert detect_shape_delta(bare, "don't sort by amount") is None
    assert detect_shape_delta(bare, "by month instead") is None


def test_no_frame_and_no_message_are_both_safe():
    assert detect_shape_delta(None, "make it top 10") is None
    assert detect_shape_delta({}, "make it top 10") is None
    assert detect_shape_delta(_frame(), "") is None
    assert detect_shape_delta(_frame(), "   ") is None


# ---------------------------------------------------------------------------
# apply_context_delta — the mutation
# ---------------------------------------------------------------------------

def test_replace_limit_changes_only_the_limit():
    out = apply_context_delta(_frame(), "replace", "limit", "10", "make it top 10")
    assert out["limit"] == 10
    assert out["group_by"] == ["year"]
    assert out["order_by"] == [{"field": "amount", "desc": True}]
    assert [f["value"] for f in out["filters"]] == ["Pune"]


def test_replace_group_by_drops_the_old_grouping():
    out = apply_context_delta(_frame(), "replace", "group_by", "month",
                              "show it by month instead")
    assert out["group_by"] == ["month"]
    assert "year" not in out["group_by"]
    assert out["limit"] == 100


def test_replace_order_by_keeps_the_proven_direction():
    """"sort by city instead" says nothing about ascending/descending; inventing one
    would silently reverse a ranking the user never asked to reverse."""
    out = apply_context_delta(_frame(order_by=[{"field": "amount", "desc": False}]),
                              "replace", "order_by", "city", "sort by city instead")
    assert out["order_by"] == [{"field": "city", "desc": False}]


def test_remove_order_by_names_its_target():
    out = apply_context_delta(_frame(), "remove", "order_by", "amount",
                              "don't sort by amount")
    assert out["order_by"] == []
    assert out["measures"] == ["amount"]        # the MEASURE is untouched by a sort drop


def test_remove_order_by_that_names_something_else_changes_nothing():
    frame = _frame()
    out = apply_context_delta(frame, "remove", "order_by", "city",
                              "don't sort by city")
    assert out is frame                          # declined, caller's object returned


def test_remove_limit_clears_it():
    assert apply_context_delta(_frame(), "remove", "limit", "", "show all")["limit"] is None


def test_a_limit_with_no_number_is_refused():
    frame = _frame()
    assert apply_context_delta(frame, "replace", "limit", "ten", "make it top ten") is frame


def test_a_value_the_user_never_typed_never_reaches_the_frame():
    frame = _frame()
    assert apply_context_delta(frame, "replace", "group_by", "quarter",
                               "show it by month instead") is frame


def test_a_slot_holding_two_values_is_left_alone_on_replace():
    """Nothing in "by month instead" says WHICH of two groupings is being swapped, and
    picking one would be this layer guessing."""
    frame = _frame(group_by=["year", "region"])
    assert apply_context_delta(frame, "replace", "group_by", "month",
                               "show it by month instead") is frame


def test_a_filter_of_the_same_name_still_wins():
    """A source with a real column called "Limit" keeps the behaviour it always had —
    the shape slots are additive, not a hijack of the filter path."""
    frame = _frame(filters=[{"field": "Limit", "operator": "equals", "value": "5",
                             "source": "executed_sql"}], limit=100)
    out = apply_context_delta(frame, "replace", "Limit", "10", "make it limit 10")
    assert out["filters"][0]["value"] == "10"
    assert out["limit"] == 100


def test_the_input_frame_is_never_mutated():
    frame = _frame()
    apply_context_delta(frame, "replace", "limit", "10", "make it top 10")
    apply_context_delta(frame, "remove", "order_by", "amount", "don't sort by amount")
    assert frame["limit"] == 100
    assert frame["order_by"] == [{"field": "amount", "desc": True}]


# ---------------------------------------------------------------------------
# the resolved query
# ---------------------------------------------------------------------------

def test_a_shape_removal_keeps_the_users_words():
    """A filter removal renders context-only, because its words name a value the engine
    must NOT be handed back. A shape removal is the opposite: the shape slots are not in
    the rendered context at all, so the message is the only thing carrying the change."""
    frame = apply_context_delta(_frame(), "remove", "order_by", "amount",
                                "don't sort by amount")
    resolved = render_frame_as_query(frame, "don't sort by amount", "remove",
                                     shape_delta=True)
    assert resolved.startswith("don't sort by amount")
    assert "General Ledger" in resolved


def test_a_filter_removal_still_renders_context_only():
    frame = _frame()
    resolved = render_frame_as_query(frame, "remove Pune", "remove")
    assert "remove" not in resolved.lower()


def test_no_operation_prose_leaks_into_the_resolved_query():
    """The 2026-09-17 incident: "measuring <col>, ranked by <col>, top 100" appended to
    the query made the engine read "measuring" as a data term. The shape slots must stay
    out of the text no matter what the delta did to them."""
    frame = apply_context_delta(_frame(), "replace", "limit", "10", "make it top 10")
    resolved = render_frame_as_query(frame, "make it top 10", "replace", shape_delta=True)
    for word in ("measuring", "ranked by", "sorted by", "group_by", "order_by",
                 "limited to", "limit"):
        assert word not in resolved.lower().replace("make it top 10", "")


# ---------------------------------------------------------------------------
# explicit vs bare — who wins a pending clarification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "make it top 5", "just the top 5", "by city instead", "sort by city instead",
    "don't sort by amount", "show all of them", "remove the limit",
])
def test_an_explicit_reshaping_is_recognised_as_one(message):
    assert detect_shape_delta(_frame(), message, explicit_only=True) is not None


@pytest.mark.parametrize("message", ["top 5", "the top 10", "first 20"])
def test_a_bare_value_is_a_shape_change_but_not_an_explicit_one(message):
    """A clarifying question asks for a bare value, so a bare "top 5" must stay
    available to answer one — while still re-shaping the query when nothing is pending."""
    assert detect_shape_delta(_frame(), message) is not None
    assert detect_shape_delta(_frame(), message, explicit_only=True) is None


def test_a_clarification_value_is_never_an_explicit_reshaping():
    for message in ("Pune", "for 2024", "the count", "ACTIVE"):
        assert detect_shape_delta(_frame(), message, explicit_only=True) is None


def test_an_explicit_reshaping_outranks_a_pending_clarification():
    """Measured on the real engine 2026-09-18: with a clarification pending, "by city
    instead" was read as the ANSWER and glued onto the unanswered request, and the engine
    answered something unrelated with full confidence."""
    import chatbot.nodes as nodes

    frame = {**_frame(), "entity": "assets_asset"}
    for message in ("by city instead", "make it top 5"):
        shape = nodes.memory_frame.detect_shape_delta(frame, message, explicit_only=True)
        assert shape is not None
        assert nodes._is_clarification_answer(message, None, None, False,
                                              bool(shape)) is False


def test_a_bare_value_still_answers_a_pending_clarification():
    import chatbot.nodes as nodes

    frame = {**_frame(), "entity": "assets_asset"}
    for message in ("Pune", "for 2024", "top 5"):
        shape = nodes.memory_frame.detect_shape_delta(frame, message, explicit_only=True)
        assert nodes._is_clarification_answer(message, None, None, False,
                                              bool(shape)) is True
