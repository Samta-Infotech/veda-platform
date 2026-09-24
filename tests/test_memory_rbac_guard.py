"""Tests for the authorisation guard on REUSED conversational memory.

RBAC is already resolved fresh every turn (apps/chat/views.py computes
permitted_source_ids / resolve_query_scope / compute_data_scope before the service is
built), and every engine-bound turn carries the resulting data_scope. But the QueryFrame
is not metadata: it holds filter VALUES read out of the customer's data, the executed
SQL, the row count, and — for a re-render — the result rows. Three paths answer from it
WITHOUT reaching the engine, so data_scope is never applied to them:

    recall_node            returns the previous SQL, table and row count
    represent_node         redraws the previous rows
    context_resolve_node   injects remembered filter values into the next query

A grant revoked between turns therefore left all three serving content from the withdrawn
source for as long as the memory lived — seven days. Authorisation from a previous turn
is not authorisation.

Run: ``pytest tests/test_memory_rbac_guard.py``
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot import nodes as N  # noqa: E402

_FRAME = {"entity": "assets_salelisting", "entity_display": "Sale Listings",
          "filters": [{"field": "Status", "operator": "equals", "value": "For Sale"}],
          "last_sql": "SELECT * FROM assets_salelisting", "last_row_count": 100,
          "source_id": 2, "version": 3}


# ---- the predicate ---------------------------------------------------------------

def test_a_frame_from_a_still_granted_source_is_kept():
    assert N._frame_still_authorised(_FRAME, {"source_ids": [2, 4]}) is True


def test_a_frame_from_a_revoked_source_is_rejected():
    assert N._frame_still_authorised(_FRAME, {"source_ids": [4, 5]}) is False


def test_string_and_int_source_ids_compare_correctly():
    """The scope arrives from JSON on some paths and from the ORM on others."""
    assert N._frame_still_authorised({"source_id": "2"}, {"source_ids": [2]}) is True
    assert N._frame_still_authorised({"source_id": 2}, {"source_ids": ["2"]}) is True
    assert N._frame_still_authorised({"source_id": "2"}, {"source_ids": ["3"]}) is False


def test_unreadable_ids_fail_closed():
    assert N._frame_still_authorised({"source_id": "abc"}, {"source_ids": [2]}) is False


def test_it_only_decides_where_there_is_something_to_decide():
    """A frame with no recorded source (written before source pinning existed) and a turn
    with no resolved scope AT ALL (a non-HTTP caller such as the CLI) each supply no fact
    to compare, so neither is treated as a denial.

    An EMPTY scope is deliberately NOT in this set — see
    test_an_empty_scope_is_a_denial_not_an_absence. An earlier version of this test
    asserted `{"source_ids": []} is True`, documenting the hole rather than flagging it."""
    assert N._frame_still_authorised({"entity": "x"}, {"source_ids": [2]}) is True
    assert N._frame_still_authorised(_FRAME, {"source_ids": None}) is True
    assert N._frame_still_authorised(_FRAME, {}) is True


# ---- the guard, at the single point every path reads through ----------------------

def _read(frame, source_ids, reset_calls):
    from unittest.mock import patch
    with patch.object(N.MemoryStore, "read_frame", return_value=frame), \
         patch.object(N.MemoryStore, "read_stack", return_value=[{"dimension": "d", "value": "v"}]), \
         patch.object(N.MemoryStore, "read_episodic", return_value=[{"role": "user", "content": "q"}]), \
         patch.object(N.MemoryStore, "reset", side_effect=lambda *a, **k: reset_calls.append(a)):
        return N.memory_read_node({"tenant": "t", "session_id": "s", "message": "anything",
                                   "source_ids": source_ids})


def test_an_authorised_turn_reads_its_memory_normally():
    out = _read(_FRAME, [2], [])
    assert out["frame"]["entity"] == "assets_salelisting"
    assert out["drill_stack"] and out["episodic"]


def test_a_revoked_source_discards_everything_derived_from_it():
    """Not just the frame — the drill stack, the episodic buffer and the retained result
    rows all descend from the same withdrawn source."""
    resets = []
    out = _read(_FRAME, [4, 5], resets)
    assert out["frame"] == {}
    assert out["drill_stack"] == [] and out["episodic"] == []
    assert out["last_result"] == {} and out["pending_clarification"] == {}
    assert resets, "the stale memory should also be cleared from Redis, not just this turn"


def test_no_sql_rows_or_filter_values_survive_the_guard():
    """The concrete exposure: what recall_node and represent_node would have served."""
    out = _read(_FRAME, [9], [])
    serialized = str(out)
    for secret in ("SELECT", "assets_salelisting", "For Sale", "100"):
        assert secret not in serialized, f"{secret!r} survived a revoked grant"


def test_the_guard_runs_before_any_path_can_read_the_frame():
    """It lives in memory_read_node — the one node every route reads through — rather
    than in each consumer, so a path added later is guarded by default."""
    import inspect
    source = inspect.getsource(N.memory_read_node)
    assert "_frame_still_authorised" in source
    for consumer in (N.recall_node, N.represent_node, N.context_resolve_node):
        assert "_frame_still_authorised" not in inspect.getsource(consumer)


def test_recall_on_a_discarded_frame_states_the_absence_rather_than_the_secret():
    """End of the chain: with the frame gone, recall answers "nothing has been run"
    instead of the previous source's SQL."""
    out = N.recall_node({"message": "what sql did you run", "recall_kind": "sql",
                         "frame": {}, "source_profiles": {}})
    assert "don't have an earlier question" in out["reply_text"]
    assert "SELECT" not in out["reply_text"].upper()


def test_represent_on_a_discarded_frame_has_no_rows_to_redraw():
    out = N.classify_node({"message": "as a pie chart", "history": [{}], "frame": {},
                           "engine_result": {}, "last_result": {}}, {})
    assert out["action"] != "represent"


def test_an_empty_scope_is_a_denial_not_an_absence():
    """The single most likely revocation shape — the caller's LAST grant withdrawn —
    arrives as source_ids == []. Treating that like "no scope supplied" let a seven-day
    frame keep serving. `None` (a non-HTTP caller that resolved no scope at all) is the
    only genuine absence."""
    assert N._frame_still_authorised({"source_id": 2}, {"source_ids": []}) is False
    assert N._frame_still_authorised({"source_id": 2}, {"source_ids": None}) is True
    assert N._frame_still_authorised({"source_id": 2}, {}) is True
