"""Tests for the QueryFrame's measure/ordering/limit memory (chatbot/memory/frame.py).

The frame used to remember only WHICH ROWS the previous turn was about (entity +
proven filters) and never WHAT ABOUT THEM. So a follow-up that changes only the
ranking — "the most expensive instead", "just the top 10" — reached the engine with
no measure or ordering to change, and had to re-derive one from the words alone:

    turn 1  "the 100 cheapest sale listings"      -> ORDER BY expected_price ASC LIMIT 100
    turn 2  "show me the most expensive instead"
            -> "show me the most expensive instead (for Sale Listings, status = For Sale)"
                                                   ^ nothing about price, order or 100

These are pure functions — no Redis, no LLM, no network.
Run: ``pytest tests/test_memory_frame_ranking.py``
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.memory import frame as F  # noqa: E402


def _answered_result(**over):
    """An engine_result shaped like the real one (verified against a live
    /v1/run_hybrid_query response for "the cheapest properties for sale")."""
    result = {
        "status": "answered",
        "table": "assets_salelisting",
        "sql": "SELECT ... ORDER BY expected_price ASC LIMIT 100",
        "rows": [[1], [2]],
        "explain": {
            "data_used": {"datasets": ["Sale Listings"]},
            "filters": {"applied": [{"field": "Status", "operator": "equals",
                                     "value": "For Sale"}]},
            "operations": [{"type": "sort", "summary": "Sort by Expected Price (lowest first)"},
                           {"type": "limit", "summary": "Return top 100"}],
            "understanding": {"summary": "Find the 100 cheapest sale listings."},
        },
        "analytics": {
            "orderings": [["expected_price", False]],   # False == not DESC == ASC
            "limit": 100,
            "query_measures": [],
            "query_dimensions": [],
        },
    }
    result.update(over)
    return result


def _frame(**over):
    harvested = F.harvest_frame(_answered_result())
    harvested.update(over)
    return F.merge_frame_post_execution(None, harvested, "new_topic", "default", "s1")


# ---- harvest --------------------------------------------------------------------

def test_ordering_and_limit_are_harvested_from_the_executed_sql():
    frame = _frame()
    assert frame["order_by"] == [{"field": "expected_price", "desc": False}]
    assert frame["limit"] == 100


def test_measures_are_harvested_when_the_query_aggregates():
    harvested = F.harvest_frame(_answered_result(
        analytics={"orderings": [["total", True]], "limit": 10,
                   "query_measures": ["rent_amount"], "query_dimensions": ["city"]}))
    assert harvested["measures"] == ["rent_amount"]


def test_a_result_without_analytics_still_harvests_the_rest():
    """Federated results carry no analytics — the frame keeps fewer facts, exactly
    as it did before these fields existed, rather than failing the write."""
    harvested = F.harvest_frame(_answered_result(analytics=None))
    assert harvested["entity"] == "assets_salelisting"
    assert harvested["order_by"] == [] and harvested["limit"] is None


def test_malformed_orderings_are_skipped_not_raised():
    harvested = F.harvest_frame(_answered_result(
        analytics={"orderings": ["expected_price", [], [None, True], ["ok", False]],
                   "limit": None}))
    assert harvested["order_by"] == [{"field": "ok", "desc": False}]


# ---- carrying it into the next turn ---------------------------------------------

def test_a_narrowing_follow_up_carries_entity_and_filters():
    resolved = F.render_frame_as_query(_frame(), "only the ones in Nagpur", "refine")
    # Values, not "Field equals Value" — see chatbot/memory/frame.py::_describe_frame.
    assert "assets_salelisting" in resolved and "For Sale" in resolved
    assert resolved.startswith("only the ones in Nagpur")   # user's words, verbatim, first


def test_the_remembered_ranking_never_leaks_into_the_engines_query():
    """THE guard. A first cut appended "measuring <col>, ranked by <col> (lowest
    first), top 100" to the resolved query. Live test 2026-09-17: the engine read the
    word "measuring" as a data term and spent 54s before answering "Could you clarify
    if 'measuring' is a column name or a value to filter on?".

    The frame still REMEMBERS all three (the tests above) and the classifier prompt
    shows them — that is where a decision is made about the ranking. The resolved query
    goes to a pipeline that parses every word of it as data, so it gets facts only."""
    frame = _frame(measures=["rent_amount"])
    for message, delta in (("only the ones in Nagpur", "refine"),
                           ("show me the most expensive instead", "refine"),
                           ("just the top 10", "refine"),
                           ("go back", "drill_up")):
        resolved = F.render_frame_as_query(frame, message, delta)
        for leaked in ("measuring", "ranked by", "lowest first", "highest first", "top 100"):
            assert leaked not in resolved, f"{leaked!r} leaked for {message!r}: {resolved!r}"


def test_a_new_topic_carries_nothing():
    assert F.render_frame_as_query(_frame(), "and for lease listings", "new_topic") == \
        "and for lease listings"


# ---- source pinning -------------------------------------------------------------

def test_the_frame_records_which_source_answered():
    """An entity name is not unique across a multi-source scope; without this a
    follow-up could re-resolve the same table name in a different source."""
    frame = _frame(source_id=2)
    assert frame["source_id"] == 2


def test_an_empty_frame_declares_every_new_field():
    empty = F.empty_frame("default", "s1")
    for field in ("measures", "order_by", "limit", "source_id"):
        assert field in empty, f"{field} missing from empty_frame"
