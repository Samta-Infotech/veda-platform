"""A follow-up that left the conversation's table does not replace its state.

Measured 2026-09-26 (audit scenario A): "Go back" on a Noida/furnished frame was answered
by the federated route with no rows; memory filed entity None and the conversation
wandered into the employee handbook.

Run: `PYTHONPATH=.:veda_core python -m pytest tests/test_followup_write_guard.py -q`
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chatbot.nodes as N  # noqa: E402

FRAME = {"entity": "assets_asset", "source_id": 2, "version": 3, "route": "deterministic",
         "filters": [{"field": "City", "column": "city_name", "operator": "equals",
                      "value": "noida"}], "base_query": "Show properties by city"}


def _explain(datasets, table=None):
    return {"data_used": {"datasets": datasets}, "filters": {"applied": []},
            "operations": []}


@pytest.fixture
def writes(monkeypatch):
    seen = []
    monkeypatch.setattr(N.MemoryStore, "write_frame",
                        staticmethod(lambda *a, **k: seen.append("frame") or True))
    for name in ("write_stack", "push_episodic_turn", "write_reference", "write_topics",
                 "write_comparison", "write_user_sessions"):
        if hasattr(N.MemoryStore, name):
            monkeypatch.setattr(N.MemoryStore, name, staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(N.MemoryStore, "read_user_sessions", staticmethod(lambda *a, **k: []),
                        raising=False)
    return seen


def _state(engine_result, action="followup", **kw):
    return {"status": "answered", "action": action, "tenant": "t", "session_id": "s",
            "frame": dict(FRAME), "drill_stack": [], "delta_type": "drill_up",
            "engine_result": engine_result, "message": "Go back", **kw}


def test_a_federated_no_table_answer_does_not_replace_the_frame(writes):
    er = {"status": "answered", "table": "federated", "rows": [], "route": "federated",
          "explain": _explain(["homzhub", "invoices_csv"])}
    N.memory_write_node(_state(er))
    assert writes == []


def test_a_zero_row_follow_up_does_not_replace_the_frame(writes):
    er = {"status": "answered", "table": "assets_asset", "rows": [], "route": "deterministic",
          "explain": _explain(["Assets"])}
    N.memory_write_node(_state(er))
    assert writes == []


def test_a_document_answer_to_a_table_follow_up_does_not_replace_it(writes):
    er = {"status": "answered", "rows": None, "route": "rag", "answer": "x. Sources: (h.pdf)",
          "explain": _explain(["h"])}
    N.memory_write_node(_state(er))
    assert writes == []


def test_a_normal_follow_up_still_writes(writes):
    er = {"status": "answered", "table": "assets_asset", "rows": [{"n": 1}],
          "route": "deterministic", "explain": _explain(["Assets"])}
    N.memory_write_node(_state(er, delta_type="refine"))
    assert writes == ["frame"]


def test_a_new_question_is_never_held_back(writes):
    er = {"status": "answered", "rows": [], "route": "rag", "answer": "x. Sources: (h.pdf)",
          "explain": _explain(["h"])}
    N.memory_write_node(_state(er, action="answer", delta_type="new_topic"))
    assert writes == ["frame"]
