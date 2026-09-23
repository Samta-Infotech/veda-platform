"""Tests for the "start over" memory reset — chatbot/nodes.py::reset_node.

Found by an aggressive memory audit against real Redis: memory_read_node DID wipe the
session's keys on "start over", but the turn then carried on to the engine, which
answered something, and memory_write_node wrote a brand-new frame from it. Measured:
the frame came back at version 1 with an entity in it, so the wipe had no lasting
effect at all. Those words also name no data, so the engine round-trip was pointless.

Run: ``pytest tests/test_memory_reset.py``
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot import nodes as N  # noqa: E402


def test_the_reset_flag_ends_the_turn_before_the_engine():
    out = N.classify_node({"message": "start over", "history": [{}], "frame": {},
                           "engine_result": {}, "memory_reset": True}, {})
    assert out["action"] == "reset"


def test_an_ordinary_turn_is_untouched_by_the_flag_being_absent():
    N.call_slm = lambda *a, **k: '{"action": "answer", "delta_type": "new_topic"}'
    out = N.classify_node({"message": "how many assets are there", "history": [],
                           "frame": {}, "engine_result": {}}, {})
    assert out["action"] != "reset"


def test_the_reset_turn_clears_the_carried_result_too():
    """A presentation follow-up after "start over" must not redraw the forgotten
    result — clearing the frame while leaving last_result behind would do exactly that."""
    out = N.classify_node({"message": "start over", "history": [{}], "frame": {},
                           "engine_result": {"rows": [[1]]}, "last_result": {"rows": [[1]]},
                           "memory_reset": True}, {})
    assert out["engine_result"] == {} and out["last_result"] == {}


def test_the_reset_is_acknowledged_rather_than_answered_silently():
    out = N.reset_node({"message": "start over"})
    assert "Cleared" in out["reply_text"]
    assert out["frame"] == {} and out["drill_stack"] == [] and out["last_result"] == {}
    assert [t["role"] for t in out["history"]] == ["user", "assistant"]


def test_the_graph_routes_reset_away_from_the_engine():
    from chatbot.graph import _route_after_classify
    assert _route_after_classify({"action": "reset", "history": [{}]}) == "reset"


def test_the_reset_flag_does_not_stick_to_the_next_turn():
    """The checkpointer persists the whole state across turns, so a flag only ever set
    True stays True. Live test 2026-09-17: the turn AFTER "start over" was itself
    answered as a reset, and every turn after that would have been too. memory_read_node
    must clear it explicitly on every non-reset turn."""
    import inspect

    from chatbot import nodes as N
    source = inspect.getsource(N.memory_read_node)
    assert '"memory_reset": False' in source


def test_a_normal_turn_reports_the_flag_as_cleared():
    from unittest.mock import patch

    from chatbot import nodes as N
    with patch.object(N.MemoryStore, "read_frame", return_value={}), \
         patch.object(N.MemoryStore, "read_stack", return_value=[]), \
         patch.object(N.MemoryStore, "read_episodic", return_value=[]):
        out = N.memory_read_node({"tenant": "t", "session_id": "s",
                                  "message": "how many assets are there"})
    assert out["memory_reset"] is False


def test_the_reset_turn_itself_still_sets_the_flag():
    from unittest.mock import patch

    from chatbot import nodes as N
    with patch.object(N.MemoryStore, "reset", return_value=None):
        out = N.memory_read_node({"tenant": "t", "session_id": "s", "message": "start over"})
    assert out["memory_reset"] is True and out["frame"] == {}
