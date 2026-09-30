"""Session memory across chats (VEDA_MEMORY_LAYER_PLAN.md step 6, M2).

Every memory key is per SESSION, so a new chat started with nothing: "what did we look at
before?" and "continue the property analysis" had no answer. The per-user summary carries
topic SNAPSHOTS (never rows or answers) forward; a restore re-executes under the current
turn's authorisation.

Same hermetic harness as tests/test_topic_index.py: the REAL MemoryStore on an in-process
fake Redis, the REAL nodes, and any SLM call fails the test.
Run: `PYTHONPATH=.:veda_core python -m pytest tests/test_session_memory.py -q`
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from test_topic_index import (  # noqa: E402,F401  (fixtures are used by name)
    NAGPUR, PAY_Q, PROPS_Q, Session, _result, no_model, redis)
from chatbot import nodes as N  # noqa: E402
from chatbot.memory import topics as T  # noqa: E402
from chatbot.memory.store import MemoryStore  # noqa: E402
from chatbot.state import ChatState  # noqa: E402

_GROUP = ["facing", "corner_property"]


def _session(user="u1", source_ids=(1, 2, 3)):
    s = Session(source_ids=source_ids)
    s.state["user_id"] = user
    return s


def _properties_in_nagpur(sess):
    sess.answered(PROPS_Q, _result("assets_asset", "Assets", group=_GROUP), "new_topic",
                  action="answer")
    sess.answered("only the Nagpur ones",
                  _result("assets_asset", "Assets", NAGPUR, group=_GROUP), "refine")
    return sess


def test_the_new_keys_are_declared_in_chatstate():
    """LangGraph silently drops an undeclared key, in the input and in node outputs."""
    for key in ("user_id", "previous_topics"):
        assert key in ChatState.__annotations__, key


# ── carried forward ───────────────────────────────────────────────────────────────────
def test_a_new_chat_sees_the_earlier_chats_topics(redis, no_model):
    a = _properties_in_nagpur(_session())
    b = _session()
    b.read("hello")
    prev = b.state["previous_topics"]
    assert [t["entity"] for t in prev] == ["assets_asset"]
    assert prev[0]["from_session"] == a.state["session_id"]
    assert ("city_name", "nagpur") in [(f["column"], f["value"]) for f in prev[0]["filters"]]


def test_the_current_chat_is_not_listed_as_an_earlier_one(redis, no_model):
    a = _properties_in_nagpur(_session())
    a.read("hello")
    assert a.state["previous_topics"] == []


def test_another_user_sees_nothing(redis, no_model):
    _properties_in_nagpur(_session(user="u1"))
    b = _session(user="u2")
    b.read("hello")
    assert b.state["previous_topics"] == []


def test_a_revoked_source_is_invisible(redis, no_model):
    """The snapshot holds filter values from the data: out of scope → never offered."""
    _properties_in_nagpur(_session(source_ids=(1, 2, 3)))
    b = _session(source_ids=(1, 3))
    b.read("hello")
    assert b.state["previous_topics"] == []


def test_no_user_no_cross_session_memory(redis, no_model):
    a = Session()                                   # CLI-style caller: no user_id
    _properties_in_nagpur(a)
    assert not any(":user:" in k for k in redis.kv)


# ── returning to an earlier chat's topic ──────────────────────────────────────────────
def test_continue_the_property_analysis_restores_the_earlier_topic(redis, no_model):
    _properties_in_nagpur(_session())
    b = _session()
    b.classify("continue the property analysis")
    assert b.state["topic_restore"]["kind"] == "restore"
    assert b.state["topic_restore"]["topic"]["entity"] == "assets_asset"
    b.resolve()
    ctx = b.state["conversation_context"]
    assert ctx["user_message"] == PROPS_Q                 # the user's own earlier words
    assert {"column": "city_name", "operator": "equals", "value": "nagpur"} in ctx["filters"]


def test_this_chats_own_topic_wins_over_an_earlier_chat(redis, no_model):
    """Same entity in both: the in-session index is consulted first."""
    _properties_in_nagpur(_session())
    b = _session()
    b.answered(PROPS_Q, _result("assets_asset", "Assets", group=_GROUP), "new_topic",
               action="answer")
    b.answered(PAY_Q, _result("accounts_generalledger", "Payment Transactions",
                              group=["transaction_type"]), "new_topic", action="answer")
    b.classify("go back to the properties")
    topic = b.state["topic_restore"]["topic"]
    assert "from_session" not in topic                   # this chat's snapshot, not A's
    assert all(f.get("value") != "nagpur" for f in topic["filters"])


def test_recall_lists_earlier_chats_when_this_one_is_empty(redis, no_model):
    _properties_in_nagpur(_session())
    b = _session()
    b.classify("what have we looked at?")
    assert b.state["action"] == "recall"
    out = N.recall_node(b.state)
    assert "earlier conversations" in out["reply_text"] and "nagpur" in out["reply_text"]


# ── forgetting ────────────────────────────────────────────────────────────────────────
def test_start_over_forgets_the_chat_across_sessions_too(redis, no_model):
    a = _properties_in_nagpur(_session())
    a.read("start over")
    b = _session()
    b.read("hello")
    assert b.state["previous_topics"] == []


# ── pure helpers ──────────────────────────────────────────────────────────────────────
def _summary(sid, *entities):
    return T.session_summary(sid, [{"entity": e, "source_id": 2, "base_query": "q"}
                                   for e in entities], updated_at=1)


def test_sessions_are_bounded_and_most_recent_first():
    sessions = []
    for i in range(15):
        sessions = T.merge_sessions(sessions, _summary(f"s{i}", "a"))
    assert len(sessions) == T._MAX_SESSIONS and sessions[0]["session_id"] == "s14"


def test_a_session_replaces_its_own_entry():
    sessions = T.merge_sessions([_summary("s1", "a")], _summary("s2", "b"))
    sessions = T.merge_sessions(sessions, _summary("s1", "c"))
    assert [s["session_id"] for s in sessions] == ["s1", "s2"]
    assert sessions[0]["topics"][0]["entity"] == "c"


def test_previous_topics_dedupe_across_sessions_keeping_the_most_recent():
    sessions = [_summary("s2", "a", "b"), _summary("s1", "a", "c")]
    prev = T.previous_topics(sessions, current_session_id="s9")
    assert [t["entity"] for t in prev] == ["a", "b", "c"]
    assert prev[0]["from_session"] == "s2"


def test_nothing_worth_keeping_is_no_summary():
    assert T.session_summary("s1", []) is None
    assert T.session_summary("", [{"entity": "a"}]) is None
