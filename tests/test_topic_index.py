"""Topic index + return-to-topic (VEDA_MEMORY_LAYER_PLAN.md step 5, M3).

Measured problem: on one source, "distribution of properties by facing" -> "only the Nagpur
ones" -> "show payment transactions by transaction type" overwrote the source's frame, so
the properties topic, its Nagpur filter and its drill stack were gone and "go back to the
properties" had nothing to return to.

Hermetic: the REAL MemoryStore code runs against an in-process fake Redis (so the key
layout, the reset paths and the optimistic lock are exercised, not mocked away), and the
REAL graph nodes are driven. No SLM, no engine: call_slm is replaced so that any
unexpected model call fails the test.

Run: `PYTHONPATH=.:veda_core python -m pytest tests/test_topic_index.py -q`
"""
import fnmatch
import json
import os
import sys
import uuid

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chatbot import graph as G  # noqa: E402
from chatbot import nodes as N  # noqa: E402
from chatbot.memory import store as S  # noqa: E402
from chatbot.memory import topics as T  # noqa: E402
from chatbot.memory.store import MemoryStore  # noqa: E402
from chatbot.prompts.delta_types import DELTA_TYPES  # noqa: E402
from chatbot.state import ChatState  # noqa: E402


# ── an in-process Redis, just the commands MemoryStore uses ───────────────────────────
class _Pipe:
    def __init__(self, r):
        self.r, self.q = r, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def watch(self, *k):
        pass

    def unwatch(self):
        pass

    def multi(self):
        pass

    def get(self, k):                      # immediate, as in WATCH mode
        return self.r.get(k)

    def __getattr__(self, name):           # everything else is queued
        def _q(*a, **k):
            self.q.append((name, a, k))
            return self
        return _q

    def execute(self):
        out = [getattr(self.r, n)(*a, **k) for n, a, k in self.q]
        self.q = []
        return out


class FakeRedis:
    def __init__(self):
        self.kv, self.lists, self.sets = {}, {}, {}

    def pipeline(self):
        return _Pipe(self)

    def get(self, k):
        return self.kv.get(k)

    def set(self, k, v, ex=None, nx=False, px=None):
        if nx and k in self.kv:
            return False
        self.kv[k] = v
        return True

    def delete(self, *keys):
        for k in keys:
            self.kv.pop(k, None)
            self.lists.pop(k, None)
            self.sets.pop(k, None)

    def expire(self, *a, **k):
        return True

    def lpush(self, k, v):
        self.lists.setdefault(k, []).insert(0, v)

    def ltrim(self, k, a, b):
        self.lists[k] = self.lists.get(k, [])[a:b + 1]

    def lrange(self, k, a, b):
        return list(self.lists.get(k, [])[a:b + 1])

    def sadd(self, k, v):
        self.sets.setdefault(k, set()).add(v)

    def srem(self, k, v):
        self.sets.get(k, set()).discard(v)

    def smembers(self, k):
        return set(self.sets.get(k, set()))

    def keys(self, pattern):
        return [k for k in list(self.kv) + list(self.lists) if fnmatch.fnmatch(k, pattern)]


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(S, "_client", lambda: fake)
    return fake


@pytest.fixture
def no_model(monkeypatch):
    """Any SLM call is a failure: every path under test here is deterministic."""
    def _boom(*a, **k):
        raise AssertionError("an SLM call was made on a deterministic path")
    monkeypatch.setattr(N, "call_slm", _boom)


# ── engine results as the pipeline hands them over ────────────────────────────────────
def _result(table, display, filters=(), group=(), agg="count", sql="SELECT 1"):
    return {"status": "answered", "table": table, "rows": [["x", 1]], "cols": ["a", "b"],
            "sql": sql, "_route": "deterministic",
            "analytics": {"orderings": [], "limit": None, "query_measures": []},
            "explain": {"data_used": {"datasets": [display]},
                        "filters": {"applied": [
                            {"field": f, "column": c, "operator": "equals", "value": v}
                            for f, c, v in filters]},
                        "operations": ([{"type": "group", "column": g,
                                         "summary": f"Group by {g}"} for g in group]
                                       + [{"type": agg, "summary": agg}]),
                        "understanding": {"summary": "s"}}}


PROPS_Q = "What is the distribution of properties by facing?"
PAY_Q = "show payment transactions by transaction type"
NAGPUR = [("Location", "city_name", "nagpur")]


class Session:
    """Drives the real nodes turn by turn, carrying state the way the checkpointer does."""

    def __init__(self, source_ids=(1, 2, 3)):
        self.state = {"tenant": "t", "session_id": str(uuid.uuid4()), "source_id": 2,
                      "source_ids": list(source_ids), "history": []}

    def read(self, message):
        self.state["message"] = message
        self.state.update(N.memory_read_node(self.state))
        return self.state

    def answered(self, message, engine_result, delta_type, action="followup", **extra):
        """A turn the engine answered: memory_read -> (decisions given) -> memory_write."""
        self.read(message)
        self.state.update({"status": "answered", "engine_result": engine_result,
                           "delta_type": delta_type, "action": action, **extra})
        self.state.update(N.memory_write_node(self.state))
        self.state["history"] = self.state["history"] + [
            {"role": "user", "content": message}, {"role": "assistant", "content": "ok"}]
        return self.state

    def classify(self, message):
        self.read(message)
        self.state.update(N.classify_node(self.state, {}))
        return self.state

    def resolve(self):
        self.state.update(N.context_resolve_node(self.state, {}))
        return self.state


def _three_topics(sess):
    """The measured sequence: properties -> Nagpur -> payments."""
    sess.answered(PROPS_Q, _result("assets_asset", "Assets", group=["facing",
                                                                      "corner_property"]),
                  "new_topic", action="answer")
    sess.answered("only the Nagpur ones",
                  _result("assets_asset", "Assets", NAGPUR, group=["facing",
                                                                    "corner_property"]),
                  "refine")
    sess.answered(PAY_Q, _result("accounts_generalledger", "Payment Transactions",
                                 group=["transaction_type"]), "new_topic", action="answer")
    return sess


# ══ the LangGraph trap ═════════════════════════════════════════════════════════════════
def test_both_new_keys_are_declared_in_chatstate():
    """LangGraph silently drops undeclared keys — this has disabled features twice."""
    assert "topic_index" in ChatState.__annotations__
    assert "topic_restore" in ChatState.__annotations__


def test_the_restore_operation_is_a_member_of_the_closed_set():
    assert N._RESTORE_OPERATION in DELTA_TYPES


# ══ index write / update / move-to-front / bound ═══════════════════════════════════════
def _entry(entity, src=2, q="show x", **kw):
    frame = {"entity": entity, "source_id": src, "base_query": q, "route": "deterministic",
             "filters": [], **kw}
    return T.snapshot(frame, kw.pop("stack", []) if "stack" in kw else [])


def test_upsert_moves_the_same_topic_to_the_front_and_keeps_the_others():
    idx = T.upsert([], _entry("a"))
    idx = T.upsert(idx, _entry("b"))
    idx = T.upsert(idx, _entry("a", q="show a again"))
    assert [e["entity"] for e in idx] == ["a", "b"]
    assert idx[0]["base_query"] == "show a again"


def test_the_index_is_bounded_to_five_most_recent():
    idx = []
    for name in "abcdefg":
        idx = T.upsert(idx, _entry(name))
    assert [e["entity"] for e in idx] == ["g", "f", "e", "d", "c"]


def test_the_same_entity_in_two_sources_is_two_topics():
    idx = T.upsert(T.upsert([], _entry("assets_asset", src=2)), _entry("assets_asset", src=3))
    assert len(idx) == 2


def test_rootless_frames_are_never_indexed_and_documents_are_indexed_as_documents():
    """Documents are indexed since 2026-09-25 (demo X1) — as document entries, returned to
    by naming the document; see tests/test_document_memory.py."""
    doc = T.snapshot({"entity": "Handbook", "entity_is_document": True,
                      "base_query": "notice period", "filters": [{"column": "x"}]}, [{"a": 1}])
    assert doc["entity_is_document"] is True and doc["filters"] == [] \
        and doc["drill_stack"] == []
    assert T.snapshot({"entity": "assets_asset", "base_query": ""}, []) is None
    assert T.snapshot({}, []) is None


def test_memory_write_builds_the_index_and_keeps_the_displaced_topic(redis):
    sess = _three_topics(Session())
    idx = MemoryStore.read_topics("t", sess.state["session_id"])
    assert [e["entity"] for e in idx] == ["accounts_generalledger", "assets_asset"]
    props = idx[1]
    # The displaced topic kept its LAST answered state: the Nagpur filter AND its stack.
    assert [(f["column"], f["value"]) for f in props["filters"]] == [("city_name", "nagpur")]
    assert [lvl["value"] for lvl in props["drill_stack"]] == ["nagpur"]
    assert props["base_query"] == PROPS_Q
    assert props["group_by"] == ["facing", "corner_property"]
    # ...while the SOURCE's frame did move on — which is exactly the problem measured.
    assert MemoryStore.read_frame("t", sess.state["session_id"], 2)["entity"] == \
        "accounts_generalledger"
    # And the index travels in graph state for this turn.
    assert [e["entity"] for e in sess.state["topic_index"]] == [e["entity"] for e in idx]


def test_an_aborted_frame_write_does_not_touch_the_index(redis, monkeypatch):
    sess = Session()
    monkeypatch.setattr(MemoryStore, "write_frame", staticmethod(lambda *a, **k: False))
    sess.answered(PROPS_Q, _result("assets_asset", "Assets"), "new_topic", action="answer")
    assert MemoryStore.read_topics("t", sess.state["session_id"]) == []


def test_a_new_topic_whose_first_question_has_a_filter_is_its_own_root(redis):
    """It used to inherit the PREVIOUS topic's base_query, which a restore would replay."""
    sess = Session()
    sess.answered(PROPS_Q, _result("assets_asset", "Assets"), "new_topic", action="answer")
    sess.answered("show debit payments", _result("accounts_generalledger", "Payments",
                                                 [("Type", "transaction_type", "debit")]),
                  "new_topic", action="answer")
    assert sess.state["frame"]["base_query"] == "show debit payments"
    assert MemoryStore.read_topics("t", sess.state["session_id"])[0]["base_query"] == \
        "show debit payments"


# ══ reset clears it ═══════════════════════════════════════════════════════════════════
def test_whole_session_reset_clears_the_index(redis):
    sess = _three_topics(Session())
    sid = sess.state["session_id"]
    MemoryStore.reset("t", sid)
    assert MemoryStore.read_topics("t", sid) == []


def test_start_over_clears_the_index_and_reports_it_empty(redis):
    sess = _three_topics(Session())
    st = sess.read("start over")
    assert st["topic_index"] == [] and st["topic_restore"] is None
    assert MemoryStore.read_topics("t", sess.state["session_id"]) == []
    assert N.reset_node({**st, "message": "start over"})["topic_index"] == []


def test_per_source_reset_drops_only_that_sources_topics(redis):
    sid = str(uuid.uuid4())
    MemoryStore.write_topics("t", sid, [_entry("a", src=2), _entry("b", src=3)])
    MemoryStore.reset("t", sid, source_id=3)
    assert [e["entity"] for e in MemoryStore.read_topics("t", sid)] == ["a"]


# ══ RBAC: revoked-source topics are invisible and unrestorable ═══════════════════════
def test_a_revoked_sources_topic_is_invisible(redis):
    sess = _three_topics(Session(source_ids=(1, 2, 3)))
    sid = sess.state["session_id"]
    MemoryStore.write_topics("t", sid, MemoryStore.read_topics("t", sid)
                             + [_entry("vendors", src=3, q="list vendors")])
    sess.state["source_ids"] = [2]                     # source 3 revoked
    st = sess.read("go back to the vendors")
    assert "vendors" not in [e["entity"] for e in st["topic_index"]]
    assert N._match_remembered_topic("go back to the vendors", st["topic_index"],
                                     st["frame"]) is None


def test_an_empty_scope_is_a_decision_every_scoped_topic_goes(redis):
    sess = _three_topics(Session())
    sess.state["source_ids"] = []
    assert sess.read("go back to the properties")["topic_index"] == []


def test_a_forged_restore_of_a_revoked_topic_is_refused_without_the_engine(redis):
    sess = _three_topics(Session())
    sess.read("go back to the vendors")
    sess.state["source_ids"] = [2]
    sess.state["topic_restore"] = {"kind": "restore",
                                   "topic": _entry("vendors", src=3, q="list vendors")}
    out = N.context_resolve_node(sess.state, {})
    assert out["engine_result"]["route"] == "reference"
    assert out["conversation_context"] is None
    assert G._route_after_resolve({**sess.state, **out}) == "ask_clarification"


def test_the_revocation_branch_of_memory_read_keeps_other_sources_topics(redis):
    sess = Session()
    sess.answered("list vendors", _result("vendors", "Vendors"), "new_topic", action="answer",
                  source_id=3)
    sess.state["source_id"] = 3
    sid = sess.state["session_id"]
    MemoryStore.write_topics("t", sid, MemoryStore.read_topics("t", sid)
                             + [_entry("assets_asset", src=2, q=PROPS_Q)])
    sess.state["source_ids"] = [2]                     # the frame's source (3) is revoked
    st = sess.read("go back to the properties")
    assert st["frame"] == {}
    assert [e["entity"] for e in st["topic_index"]] == ["assets_asset"]
    assert [e["entity"] for e in MemoryStore.read_topics("t", sid)] == ["assets_asset"]


# ══ return-to-topic restores filters + stack ═════════════════════════════════════════
def test_go_back_to_the_properties_restores_the_nagpur_snapshot(redis, no_model):
    sess = _three_topics(Session())
    st = sess.classify("go back to the properties")
    assert st["action"] == "followup"
    assert st["delta_type"] == N._RESTORE_OPERATION
    assert st["topic_restore"]["kind"] == "restore"
    assert G._route_after_classify(st) == "context_resolve"
    st = sess.resolve()
    assert st["resolved_query"] == PROPS_Q                  # the user's own root question
    assert st["frame"]["entity"] == "assets_asset"
    assert [(f["column"], f["value"]) for f in st["frame"]["filters"]] == \
        [("city_name", "nagpur")]
    assert [lvl["value"] for lvl in st["drill_stack"]] == ["nagpur"]
    ctx = st["conversation_context"]
    assert ctx["user_message"] == PROPS_Q
    assert ctx["entity_table"] == "assets_asset" and ctx["source_id"] == 2
    assert ctx["filters"] == [{"column": "city_name", "operator": "equals",
                               "value": "nagpur"}]
    assert ctx["group_by"] == ["facing", "corner_property"]
    assert ctx["operation"] == N._RESTORE_OPERATION
    assert st["context_used"]["changed"]["operation"] == "return_to_topic"
    assert G._route_after_resolve(st) == "call_engine"


def test_an_answered_restore_commits_the_topic_back_as_current(redis, no_model):
    sess = _three_topics(Session())
    sess.classify("go back to the properties")
    sess.resolve()
    sess.state.update({"status": "answered",
                       "engine_result": _result("assets_asset", "Assets", NAGPUR,
                                                group=["facing", "corner_property"])})
    sess.state.update(N.memory_write_node(sess.state))
    sid = sess.state["session_id"]
    frame = MemoryStore.read_frame("t", sid, 2)
    assert frame["entity"] == "assets_asset"                 # committed despite the old
    assert frame["base_query"] == PROPS_Q                    # snapshot version
    assert [lvl["value"] for lvl in MemoryStore.read_stack("t", sid, 2)] == ["nagpur"]
    assert [e["entity"] for e in MemoryStore.read_topics("t", sid)] == \
        ["assets_asset", "accounts_generalledger"]
    # ...and the reverse direction now works from here.
    st = sess.classify("back to the payments")
    assert st["topic_restore"]["kind"] == "restore"
    st = sess.resolve()
    assert st["resolved_query"] == PAY_Q
    assert st["drill_stack"] == [] and st["frame"]["filters"] == []
    # A filterless topic is its root question verbatim: sent WITHOUT context, exactly as
    # it was the first time (the root-replay rule).
    assert st["conversation_context"] == {"user_message": PAY_Q}


def test_after_a_restore_plain_go_back_pops_the_restored_stack(redis, no_model):
    sess = _three_topics(Session())
    sess.classify("go back to the properties")
    sess.resolve()
    sess.state.update({"status": "answered",
                       "engine_result": _result("assets_asset", "Assets", NAGPUR,
                                                group=["facing", "corner_property"])})
    sess.state.update(N.memory_write_node(sess.state))
    st = sess.classify("go back")
    assert st["delta_type"] == "drill_up" and st["topic_restore"] is None
    st = sess.resolve()
    assert st["drill_stack"] == [] and st["resolved_query"] == PROPS_Q


def test_restoring_a_topic_with_a_valueless_filter_pushes_no_level(redis, no_model):
    """Measured live: "transaction_type IS NOT NULL" is harvested as a filter with value
    None; the snapshot dropped it, so the answered restore saw it as NEW and pushed
    ('Transaction Type', None) onto the stack."""
    notnull = [("Transaction Type", "transaction_type", None)]
    sess = Session()
    sess.answered(PAY_Q, _result("accounts_paymenttransaction", "Payments", notnull),
                  "new_topic", action="answer")
    sess.answered(PROPS_Q, _result("assets_asset", "Assets"), "new_topic", action="answer")
    sess.classify("back to the payments")
    st = sess.resolve()
    assert st["conversation_context"] == {"user_message": PAY_Q}   # value-less: not sent
    sess.state.update({"status": "answered",
                       "engine_result": _result("accounts_paymenttransaction", "Payments",
                                                notnull)})
    sess.state.update(N.memory_write_node(sess.state))
    assert sess.state["drill_stack"] == []
    assert MemoryStore.read_stack("t", sess.state["session_id"], 2) == []


def test_a_failed_replay_changes_nothing(redis, no_model):
    sess = _three_topics(Session())
    sid = sess.state["session_id"]
    before = (json.dumps(MemoryStore.read_topics("t", sid)),
              json.dumps(MemoryStore.read_frame("t", sid, 2)))
    sess.classify("go back to the properties")
    sess.resolve()
    sess.state.update({"status": "refuse", "engine_result": {"status": "refuse"}})
    assert G._route_after_engine(sess.state) == "ask_clarification"
    assert N.memory_write_node(sess.state) == {}
    after = (json.dumps(MemoryStore.read_topics("t", sid)),
             json.dumps(MemoryStore.read_frame("t", sid, 2)))
    assert before == after
    # The next turn re-reads Redis: still on payments, still able to return.
    st = sess.read("hello")
    assert st["frame"]["entity"] == "accounts_generalledger" and st["topic_restore"] is None


def test_a_restore_from_another_source_carries_that_sources_live_version(redis, no_model):
    sess = Session()
    sess.answered("list vendors", _result("vendors", "Vendors"), "new_topic",
                  action="answer", source_id=3)
    sess.state["source_id"] = 2
    sess.answered(PAY_Q, _result("accounts_generalledger", "Payment Transactions"),
                  "new_topic", action="answer")
    st = sess.classify("go back to the vendors")
    assert st["topic_restore"]["topic"]["source_id"] in (3, "3")
    st = sess.resolve()
    live3 = MemoryStore.read_frame("t", sess.state["session_id"], 3)
    assert st["frame"]["version"] == live3["version"]
    assert st["conversation_context"] == {"user_message": "list vendors"}


# ══ plain "go back" is unaffected ════════════════════════════════════════════════════
def test_plain_go_back_still_drills_up_the_current_topic(redis, no_model):
    sess = Session()
    sess.answered(PROPS_Q, _result("assets_asset", "Assets"), "new_topic", action="answer")
    sess.answered("only the Nagpur ones", _result("assets_asset", "Assets", NAGPUR),
                  "refine")
    st = sess.classify("go back")
    assert st["action"] == "followup" and st["delta_type"] == "drill_up"
    assert st["topic_restore"] is None
    st = sess.resolve()
    assert st["drill_stack"] == []


def test_go_back_with_nothing_to_go_back_to_still_says_so(redis, no_model):
    sess = _three_topics(Session())                 # payments is current, no drill on it
    st = sess.classify("go back")
    assert st["action"] == "recall" and st["recall_kind"] == "drill_up_empty"


def test_naming_the_current_topic_is_not_a_return(redis):
    sess = _three_topics(Session())
    assert N._match_remembered_topic("go back to the payments", sess.read("x")["topic_index"],
                                     sess.state["frame"]) is None


@pytest.mark.parametrize("msg", [
    "go back", "go back again", "undo that filter",
    "show payment transactions by month",        # names a topic, no return: a NEW question
    "go back to the properties in 2024",         # a request of its own left over
    "what about the properties?",                # no return word
    "go back to the vendors",                    # never discussed
])
def test_messages_that_are_not_a_return_to_a_remembered_topic(redis, msg):
    sess = _three_topics(Session())
    assert N._match_remembered_topic(msg, sess.read(msg)["topic_index"],
                                     sess.state["frame"]) is None


@pytest.mark.parametrize("msg", [
    "go back to the properties", "go back to the property analysis",
    "back to the Nagpur ones", "can we return to the properties?",
    "let's revisit the properties", "the properties again",
])
def test_phrasings_of_returning_to_the_properties(redis, msg):
    sess = _three_topics(Session())
    hit = N._match_remembered_topic(msg, sess.read(msg)["topic_index"], sess.state["frame"])
    assert hit and hit["kind"] == "restore" and hit["topic"]["entity"] == "assets_asset"


# ══ ambiguity fails closed ═══════════════════════════════════════════════════════════
def test_two_matching_topics_ask_which_without_the_engine(redis, no_model):
    sess = Session()
    sess.answered("show sale listings by city",
                  _result("assets_salelisting", "Sale Listings"), "new_topic", action="answer")
    sess.answered("show lease listings by city",
                  _result("assets_leaselisting", "Lease Listings"), "new_topic",
                  action="answer")
    sess.answered(PAY_Q, _result("accounts_generalledger", "Payment Transactions"),
                  "new_topic", action="answer")
    st = sess.classify("go back to the listings")
    assert st["topic_restore"]["kind"] == "ambiguous"
    assert G._route_after_classify(st) == "context_resolve"
    st = sess.resolve()
    assert G._route_after_resolve(st) == "ask_clarification"
    assert st["conversation_context"] is None
    reply = N.ask_clarification_node(st)
    assert "Sale Listings" in reply["reply_text"] and "Lease Listings" in reply["reply_text"]
    assert reply["needs_clarification"] is False and reply["pending_clarification"] == {}


# ══ no match → byte-identical ════════════════════════════════════════════════════════
def test_no_match_classifies_exactly_as_without_an_index(redis, monkeypatch):
    raw = '{"action": "answer", "delta_type": "new_topic", "slot_candidates": []}'
    monkeypatch.setattr(N, "call_slm", lambda *a, **k: raw)
    sess = _three_topics(Session())
    st = sess.read("how many vendors are there")
    with_index = N.classify_node(dict(st), {})
    without = N.classify_node({**st, "topic_index": []}, {})
    assert with_index == without
    assert with_index["topic_restore"] is None


def test_no_topic_restore_leaves_context_resolve_untouched(redis, monkeypatch):
    raw = '{"action": "followup", "delta_type": "refine", "slot_candidates": ["Pune"]}'
    monkeypatch.setattr(N, "call_slm", lambda *a, **k: raw)
    sess = Session()
    sess.answered(PROPS_Q, _result("assets_asset", "Assets"), "new_topic", action="answer")
    st = sess.classify("only the Pune ones")
    a = N.context_resolve_node(dict(st), {})
    b = N.context_resolve_node({**st, "topic_index": []}, {})
    assert a == b and a["resolved_query"] == "only the Pune ones"


def test_topic_restore_never_survives_into_the_next_turn(redis, no_model):
    sess = _three_topics(Session())
    sess.classify("go back to the properties")
    assert sess.state["topic_restore"]
    assert sess.read("what sql did you run")["topic_restore"] is None


# ══ recall lists the topics ══════════════════════════════════════════════════════════
def test_what_have_we_looked_at_lists_the_index_without_the_engine(redis, no_model):
    sess = _three_topics(Session())
    st = sess.classify("what have we looked at so far?")
    assert st["action"] == "recall" and st["recall_kind"] == "topics"
    reply = N.recall_node(st)["reply_text"]
    assert reply.index("Payment Transactions") < reply.index("Assets")
    assert "nagpur" in reply


def test_recall_of_topics_in_a_fresh_session_says_there_is_nothing(redis, no_model):
    sess = Session()
    st = sess.classify("what have we looked at?")
    assert "haven't looked at anything" in N.recall_node(st)["reply_text"]
