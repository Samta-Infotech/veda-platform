"""Two follow-up paths measured in the 2026-09-26 audit.

1. A follow-up pointing at an answer whose source was just revoked is told so (P1: "Show me
   those results" was answered from the handbook; "the 2nd one" as smalltalk).
2. A values-only follow-up while the conversation is on a document narrows the latest table
   topic (properties → Nagpur → a handbook question → "only the FULL ones" hit the handbook).

Run: `PYTHONPATH=.:veda_core python -m pytest tests/test_revoked_and_values_on_document.py -q`
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chatbot.nodes as N  # noqa: E402
from chatbot.memory import topics as T  # noqa: E402

HISTORY = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]


def _fail_slm(*a, **k):
    raise AssertionError("no model call expected")


# ── 1. revoked ────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("msg", ["Show me those results", "the 2nd one", "what about the other ones?"])
def test_pointing_at_revoked_memory_is_answered_honestly(monkeypatch, msg):
    monkeypatch.setattr(N, "call_slm", lambda *a, **k: '{"action": "followup"}')
    out = N.classify_node({"message": msg, "history": HISTORY, "frame": {},
                           "memory_revoked_source": "2"}, config={})
    assert out["action"] == "recall" and out["recall_kind"] == "access_revoked"
    reply = N.recall_node({"message": msg, "frame": {}, "recall_kind": "access_revoked",
                           "history": []})["reply_text"]
    assert "no longer" in reply and "access" in reply


def test_a_new_question_after_revocation_is_untouched(monkeypatch):
    monkeypatch.setattr(N, "call_slm", lambda *a, **k: '{"action": "answer"}')
    out = N.classify_node({"message": "What is the probation period for new recruits?",
                           "history": HISTORY, "frame": {}, "memory_revoked_source": "2"},
                          config={})
    assert out["action"] == "answer"


def test_nothing_revoked_nothing_changes(monkeypatch):
    monkeypatch.setattr(N, "call_slm", lambda *a, **k: '{"action": "followup"}')
    out = N.classify_node({"message": "Show me those results", "history": HISTORY,
                           "frame": {}, "memory_revoked_source": None}, config={})
    assert out.get("recall_kind") != "access_revoked"


def test_the_flag_is_declared_in_chatstate():
    from chatbot.state import ChatState
    assert "memory_revoked_source" in ChatState.__annotations__


# ── 2. values on a document ──────────────────────────────────────────────────────────
TABLE = {"entity": "assets_asset", "source_id": 2, "route": "deterministic",
         "base_query": "What is the distribution of properties by facing?",
         "filters": [{"field": "City", "column": "city_name", "operator": "equals",
                      "value": "nagpur"}], "group_by": ["facing"]}
DOC = {"entity": "Samta-Employee Handbook April 2026", "entity_is_document": True,
       "route": "rag", "source_id": 2, "base_query": "probation?"}


def _index():
    return T.upsert(T.upsert([], T.snapshot(TABLE, [{"field": "City", "value": "nagpur"}])),
                    T.snapshot(DOC, []))


def test_values_on_a_document_continue_the_table_topic(monkeypatch):
    monkeypatch.setattr(N, "call_slm", _fail_slm)
    out = N.classify_node({"message": "only the FULL ones", "history": HISTORY, "frame": DOC,
                           "topic_index": _index(), "message_names_only_values": True},
                          config={})
    assert out["action"] == "followup" and out["topic_restore"]["kind"] == "continue"
    assert out["topic_restore"]["topic"]["entity"] == "assets_asset"


def test_the_continue_carries_the_table_state(monkeypatch):
    monkeypatch.setattr(N.MemoryStore, "read_frame", lambda *a, **k: {"version": 4})
    entry = next(e for e in _index() if e["entity"] == "assets_asset")
    out = N._return_to_topic({"message": "only the FULL ones", "frame": DOC, "source_ids": [2]},
                             {"kind": "continue", "topic": entry})
    ctx = out["conversation_context"]
    assert out["resolved_query"] == "only the FULL ones" and out["delta_type"] == "refine"
    assert ctx["entity_table"] == "assets_asset" and ctx["operation"] == "refine"
    assert {"column": "city_name", "operator": "equals", "value": "nagpur"} in ctx["filters"]
    assert out["frame"]["version"] == 4


def test_a_real_question_on_a_document_is_untouched(monkeypatch):
    monkeypatch.setattr(N, "call_slm", lambda *a, **k: '{"action": "followup"}')
    out = N.classify_node({"message": "and how many sick leaves?", "history": HISTORY,
                           "frame": DOC, "topic_index": _index(),
                           "message_names_only_values": False}, config={})
    assert (out.get("topic_restore") or {}).get("kind") != "continue"


def test_no_table_topic_nothing_changes(monkeypatch):
    monkeypatch.setattr(N, "call_slm", lambda *a, **k: '{"action": "followup"}')
    out = N.classify_node({"message": "only the FULL ones", "history": HISTORY, "frame": DOC,
                           "topic_index": [T.snapshot(DOC, [])],
                           "message_names_only_values": True}, config={})
    assert (out.get("topic_restore") or {}).get("kind") != "continue"


def test_memory_read_flags_an_active_source_outside_the_scope(monkeypatch):
    """A narrowed scope reads an empty frame for its own nominal source; the source the
    conversation was on is flagged (not read, not deleted)."""
    S = N.MemoryStore
    for name, val in (("read_frame", None), ("read_stack", []), ("read_episodic", []),
                      ("read_comparison", {}), ("read_topics", []), ("read_reference", None),
                      ("read_user_sessions", []), ("active_source", "2")):
        monkeypatch.setattr(S, name, staticmethod(lambda *a, _v=val, **k: _v), raising=False)
    deleted = []
    monkeypatch.setattr(S, "reset", staticmethod(lambda *a, **k: deleted.append(a)))
    out = N.memory_read_node({"message": "Show me those results", "session_id": "s",
                              "source_id": 3, "source_ids": [3, 4, 5]})
    assert out["memory_revoked_source"] == "2" and deleted == []
    out = N.memory_read_node({"message": "x", "session_id": "s", "source_id": 2,
                              "source_ids": [2, 3]})
    assert out["memory_revoked_source"] is None
