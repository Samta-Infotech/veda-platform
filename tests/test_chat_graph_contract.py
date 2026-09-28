"""GRAPH-LEVEL CONTRACT: the structured conversation context reaches the engine.

The node-level tests call context_resolve_node / call_engine_node directly and hand them
whatever state they like, so they could not see the failure this guards: LangGraph drops
any key a node returns that ChatState does not declare. `conversation_context` was not
declared, so context_resolve_node built it, the graph threw it away, and call_engine_node
sent flags=None on every turn — every engine behaviour gated on conversation context
(lane continuity, anchor hint, agent follow-up, compound skip) was unreachable from chat.

So this drives THREE turns through the real compiled graph via chatbot.run.run_chat_turn,
with only the edges of the system stubbed:
  · the inference tier — a fake InferenceClient that records every call and replies with
    canned engine results, passed through the inference tier's REAL `_serialize` so the
    wire shape (debug trace stripped, agent memory lifted) is the real one;
  · Redis — tests/tools/fake_redis.py behind the REAL MemoryStore code;
  · the checkpointer — LangGraph's in-memory MemorySaver;
  · the supervisor SLM — recorded, and expected to be called only on the first turn.

The scope is two sources, [2, 3], and every answer comes from source 3 — the NON-primary
one — so the turn-to-turn memory read (nodes.py::_read_newest_frame) is exercised too.

Run: pytest tests/test_chat_graph_contract.py
"""
import copy
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import pytest                                              # noqa: E402
from langgraph.checkpoint.memory import MemorySaver        # noqa: E402

from chatbot import graph as chat_graph                    # noqa: E402
from chatbot import nodes, run                             # noqa: E402
from inference.routes.hybrid import (_serialize,           # noqa: E402
                                     _validated_conversation_context)
from tools import fake_redis                               # noqa: E402

SCOPE = [2, 3]
ANSWERING_SOURCE = 3

TURN1_DRAFT = {"tables": ["vendors"], "aggregates": [{"fn": "count", "col": "*"}]}
TURN2_DRAFT = {"tables": ["vendors"], "aggregates": [{"fn": "count", "col": "*"}],
               "filters": [{"col": "vendors.city", "op": "=", "value": "Mumbai"}]}
TURN1_LOG = [{"tool": "find_entities", "args": {"q": "vendors"},
              "result": {"hits": [{"table": "vendors", "why": "name"}]}},
             {"tool": "join_path", "args": {"a": "vendors", "b": "cities"},
              "result": {"a": "vendors", "b": "cities",
                         "routes": [{"id": "r1", "path": ["vendors.city_id=cities.id"]}]}}]


def _engine_result(*, sql, cols, rows, filters, group_ops, draft, question, log):
    """One head result as the ENGINE builds it (pipeline._done shape), trace included —
    `_serialize` below is what turns it into what crosses the wire."""
    return {
        "status": "answered", "ok": True, "table": "vendors", "source_id": ANSWERING_SOURCE,
        "sql": sql, "cols": cols, "rows": rows, "answer": f"{len(rows)} rows",
        "ir": {"anchor": "vendors", "measure": {"aggregation": "count", "column": None},
               "filters": [{"column": f["column"], "op": "=", "value": f["value"]}
                           for f in filters],
               "group_keys": [g for g in group_ops], "ir_partial": False},
        "explain": {
            "data_used": {"datasets": ["Vendors"]},
            "filters": {"applied": [{"field": f["column"].title(), "column": f["column"],
                                     "operator": "equals", "value": f["value"]}
                                    for f in filters]},
            "operations": ([{"type": "count", "summary": "Count"}]
                           + [{"type": "group", "summary": f"Group by {g}", "column": g}
                              for g in group_ops]),
            "sources": [{"id": str(ANSWERING_SOURCE), "name": "vendors-db"}],
        },
        "analytics": {
            "column_stats": [{"name": "city", "role": "dimension",
                              "top_values": ["Mumbai", "Pune", "Kochi"]},
                             {"name": "category", "role": "dimension",
                              "top_values": ["Plumbing", "Electrical"]}],
            "available_dimensions": ["city", "category"],
            "available_measures": ["rating"],
            "query_measures": [], "orderings": [], "result_shape": "table",
        },
        "trace": {"trace_id": "t", "sections": {"agent": {
            "kind": "sql", "reason": "ok", "draft": draft, "question": question,
            "tool_calls": [{"id": f"c{i}", "ms": 1.0, **c} for i, c in enumerate(log)],
            "sql": sql}}},
    }


def _wire(engine_result):
    """The terminal SSE `result` event exactly as inference/routes/hybrid.py builds it."""
    payload = _serialize({"items": [{"status": "ok", "route": "deterministic",
                                     "source_id": ANSWERING_SOURCE, "result": engine_result}],
                          "trace_id": "t"})
    return {"status": "ok", "trace_id": "t", "result": payload}


REPLIES = [
    _wire(_engine_result(sql="SELECT COUNT(*) FROM vendors", cols=["city", "n"],
                         rows=[["Mumbai", 4], ["Pune", 3], ["Kochi", 1]], filters=[],
                         group_ops=[], draft=TURN1_DRAFT,
                         question="how many vendors are there", log=TURN1_LOG)),
    _wire(_engine_result(sql="SELECT COUNT(*) FROM vendors WHERE city = 'Mumbai'",
                         cols=["city", "n"], rows=[["Mumbai", 4]],
                         filters=[{"column": "city", "value": "Mumbai"}], group_ops=[],
                         draft=TURN2_DRAFT, question="only in Mumbai", log=TURN1_LOG[:1])),
    _wire(_engine_result(sql="SELECT city, COUNT(*) FROM vendors WHERE city = 'Mumbai' "
                             "GROUP BY city", cols=["city", "n"], rows=[["Mumbai", 4]],
                         filters=[{"column": "city", "value": "Mumbai"}], group_ops=["city"],
                         draft=TURN2_DRAFT, question="group by city", log=[])),
]


class _FakeInference:
    calls: list = []

    def stream_hybrid_query(self, query, **kwargs):
        _FakeInference.calls.append({"query": query, **copy.deepcopy(kwargs)})
        yield "progress", {"phase": "answer", "message": "…"}
        yield "result", copy.deepcopy(REPLIES[len(_FakeInference.calls) - 1])


@pytest.fixture
def session(monkeypatch):
    fake_redis.install(monkeypatch)
    _FakeInference.calls = []
    slm_calls = []

    def _slm(*a, **k):
        slm_calls.append(k.get("purpose"))
        return None                    # classify falls back to "answer"

    monkeypatch.setattr(nodes, "InferenceClient", _FakeInference)
    monkeypatch.setattr(nodes, "call_slm", _slm)
    monkeypatch.setattr(nodes, "classify_delta",
                        lambda *a, **k: pytest.fail("classify_delta must not be needed"))
    monkeypatch.setattr(chat_graph, "get_checkpointer", MemorySaver)
    compiled = chat_graph.build_graph()
    monkeypatch.setattr(run, "get_graph", lambda: compiled)

    def turn(message):
        return run.run_chat_turn(message, "graph-contract-session", tenant="t",
                                 source_id=SCOPE[0], source_ids=SCOPE,
                                 message_mentions_data=True)
    return turn, slm_calls


def _ctx(call):
    flags = call.get("flags")
    return (flags or {}).get("conversation_context")


def test_three_turn_session_carries_the_structured_context(session):
    turn, slm_calls = session

    r1 = turn("how many vendors are there")
    r2 = turn("only in Mumbai")
    r3 = turn("group by city")
    calls = _FakeInference.calls
    assert [c["query"] for c in calls] == ["how many vendors are there", "only in Mumbai",
                                           "group by city"]
    assert [r["status"] for r in (r1, r2, r3)] == ["answered"] * 3

    # Turn 1 has no history: nothing to carry, and nothing stale either.
    assert calls[0]["flags"] is None

    # Turn 2: the frame turn 1 wrote under source 3 was found although the turn's primary
    # is source 2, and it reached the engine STRUCTURED, with turn 1's agent plan.
    c2 = _ctx(calls[1])
    assert c2 is not None, "turn 2 reached the engine with no conversation_context"
    assert c2["entity_table"] == "vendors"
    assert c2["source_id"] == ANSWERING_SOURCE
    assert c2["user_message"] == "only in Mumbai"
    assert not c2.get("filters")                       # turn 1 applied none
    assert c2["agent_plan"] == TURN1_DRAFT
    assert c2["agent_question"] == "how many vendors are there"
    assert [e["tool"] for e in c2["agent_log"]] == ["find_entities", "join_path"]
    assert c2["agent_log"][1]["result"]["routes"][0]["id"] == "r1"

    # Turn 3: turn 2's executed filter is now remembered, by its real COLUMN, and the
    # agent plan is turn 2's edited one.
    c3 = _ctx(calls[2])
    assert c3 is not None, "turn 3 reached the engine with no conversation_context"
    assert c3["entity_table"] == "vendors"
    assert c3["source_id"] == ANSWERING_SOURCE
    assert c3["filters"] == [{"column": "city", "operator": "equals", "value": "Mumbai"}]
    assert c3["agent_plan"] == TURN2_DRAFT

    # Follow-ups stay on the source that answered.
    assert calls[1]["source_id"] == ANSWERING_SOURCE and calls[2]["source_id"] == ANSWERING_SOURCE

    # The follow-ups were resolved by the rule layer: the only supervisor SLM call was
    # turn 1's classification.
    assert slm_calls == ["classify"]

    # The Turn Entry Gate's report now survives the graph instead of arriving as None.
    for r in (r1, r2, r3):
        assert r["entry_path"] in ("L0_DIRECT", "L0_FASTPATH", "SLM_GATE")
        assert isinstance(r["classification_latency_ms"], float)
        assert r["requires_veda"] is True
    assert r2["requires_context"] is True and r3["requires_context"] is True


def test_inference_validator_keeps_what_chat_sends(session):
    """The same payloads, fed to the inference tier's validator, come out unchanged —
    no key chat sends is silently stripped at the boundary."""
    turn, _ = session
    for m in ("how many vendors are there", "only in Mumbai", "group by city"):
        turn(m)
    for call in _FakeInference.calls[1:]:
        sent = _ctx(call)
        kept = _validated_conversation_context({"conversation_context": copy.deepcopy(sent)})
        assert kept == sent


def test_a_clarify_reply_turn_sends_no_stale_context(session):
    """clarify_reply leaves classify through an early return and goes straight to the
    engine. The context the PREVIOUS turn built must not ride along with it."""
    turn, _ = session
    turn("how many vendors are there")
    turn("only in Mumbai")
    assert _ctx(_FakeInference.calls[1]) is not None
    state = {"message": "2024", "history": [{"role": "user", "content": "x"}],
             "pending_clarification": {"question": "which year?",
                                       "original_query": "vendors onboarded", "missing": []},
             "frame": {}, "last_result": {}}
    out = nodes.classify_with_entry_gate(state, {})
    assert out["action"] == "clarify_reply"
    assert out["conversation_context"] is None
