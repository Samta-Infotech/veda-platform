"""The inference boundary's conversation-context validator, for the keys added after the
original whitelist: the planner agent's memory (agent_plan / agent_log / agent_question),
comparison, entry_path, classification_latency_ms, target_frame_index and part_index.

Each is checked for SHAPE, not only presence. A wrong shape is dropped with a log line,
never raised: this runs on the request thread before the pipeline starts, and a bad client
payload must degrade to "no context", which is how a first turn already behaves.

Also pins the other half of the agent memory's trip: `_serialize` strips the debug trace
from every result, so the agent section is lifted out as `agent_memory` first.

Pure: no DB, no SLM, no network. Run: pytest tests/test_inference_context_agent_keys.py
"""
import copy
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from inference.routes.hybrid import _serialize, _validated_conversation_context   # noqa: E402

PLAN = {"tables": ["vendors", "cities"], "joins": ["vendors.city_id=cities.id"],
        "select": ["vendors.name"],
        "filters": [{"col": "cities.name", "op": "=", "value": "Mumbai"},
                    {"col": "vendors.rating", "op": "between", "value": [3, 5]}],
        "group_by": ["cities.name"], "aggregates": [{"fn": "count", "col": "*"}],
        "order": [{"by": "count_all", "dir": "desc"}], "limit": 5, "distinct": True,
        "time": {"col": "vendors.created_at", "from": "2024-01-01", "to": None}}
LOG = [{"tool": "find_entities", "args": {"q": "vendors"}, "result": {"hits": [{"table": "vendors"}]}},
       {"tool": "join_path", "args": {"a": "vendors", "b": "cities"},
        "result": {"routes": [{"id": "r1", "path": ["vendors.city_id=cities.id"]}]}}]


def _v(**ctx):
    return _validated_conversation_context({"conversation_context": {"user_message": "x", **ctx}})


def test_a_well_formed_agent_memory_passes_unchanged():
    out = _v(agent_plan=copy.deepcopy(PLAN), agent_log=copy.deepcopy(LOG),
             agent_question="how many vendors in Mumbai")
    assert out["agent_plan"] == PLAN
    assert out["agent_log"] == LOG
    assert out["agent_question"] == "how many vendors in Mumbai"


def test_a_plan_with_an_unknown_or_malformed_key_is_rejected_whole():
    for bad in ({**PLAN, "sql": "DROP TABLE x"},                       # unknown key
                {**PLAN, "tables": []},                                 # frame_path needs tables
                {**PLAN, "tables": "vendors"},
                {**PLAN, "filters": [{"col": "a.b", "op": "=", "value": {"x": 1}}]},
                {**PLAN, "order": [{"by": "x", "dir": "sideways"}]},
                {**PLAN, "limit": True},
                {k: v for k, v in PLAN.items() if k != "tables"},
                ["vendors"], "vendors"):
        out = _v(agent_plan=bad, agent_log=LOG, agent_question="q") or {}
        assert "agent_plan" not in out, bad
        # the log and question travel only with their plan
        assert "agent_log" not in out and "agent_question" not in out


def test_a_huge_plan_is_rejected():
    big = {**PLAN, "filters": [{"col": f"t.c{i}", "op": "in", "value": ["v" * 400] * 50}
                               for i in range(20)]}
    assert "agent_plan" not in (_v(agent_plan=big) or {})
    long_value = {**PLAN, "filters": [{"col": "t.c", "op": "=", "value": "v" * 501}]}
    assert "agent_plan" not in (_v(agent_plan=long_value) or {})


def test_malformed_log_entries_are_dropped_individually():
    log = LOG + ["not a dict", {"tool": ""}, {"tool": "x", "args": "no"},
                 {"tool": "join_path", "result": {"routes": [{"path": ["a.b=c.d"]}]}},   # no id
                 {"tool": "join_path", "result": {"routes": [{"id": "r2", "path": "a=b"}]}}]
    out = _v(agent_plan=PLAN, agent_log=log)
    assert out["agent_log"] == LOG


def test_the_log_is_capped():
    out = _v(agent_plan=PLAN, agent_log=[LOG[0]] * 40)
    assert len(out["agent_log"]) == 12
    out = _v(agent_plan=PLAN, agent_log=[{"tool": "t", "result": {"x": "y" * 20_000}}])
    assert out["agent_log"] == []


def test_comparison_keeps_only_what_has_execution_meaning():
    cmp_ = {"primary": {"entity": "vendors", "entity_display": "Vendors!", "label": "Pune",
                        "source_id": "3", "filters": [{"field": "City"}]},
            "comparison": {"entity": "vendors", "label": "Mumbai", "source_id": 3},
            "dimension": "city", "created_at": "2026-09-26", "turn_index": 4}
    out = _v(comparison=cmp_)
    assert out["comparison"] == {"primary": {"entity": "vendors", "label": "Pune", "source_id": 3},
                                 "comparison": {"entity": "vendors", "label": "Mumbai",
                                                "source_id": 3},
                                 "dimension": "city"}
    assert "comparison" not in (_v(comparison={"primary": {"entity": "v"}}) or {})
    assert "comparison" not in (_v(comparison="vendors") or {})


def test_scalar_telemetry_keys_are_type_and_range_checked():
    out = _v(entry_path="SLM_GATE", classification_latency_ms=12.5,
             target_frame_index=-2, part_index=1)
    assert out["entry_path"] == "SLM_GATE"
    assert out["classification_latency_ms"] == 12.5
    assert out["target_frame_index"] == -2 and out["part_index"] == 1
    out = _v(entry_path=5, classification_latency_ms=-1, target_frame_index=True,
             part_index=500)
    for k in ("entry_path", "classification_latency_ms", "target_frame_index", "part_index"):
        assert k not in out


def test_bad_shapes_never_raise():
    for junk in (None, 1, "x", [], {"agent_plan": object()}, {"agent_log": object()},
                 {"comparison": object(), "agent_plan": {"tables": [object()]}}):
        _validated_conversation_context({"conversation_context": junk})


# ── the wire: agent memory survives the trace strip ─────────────────────────────────
def _result_with_agent(kind="sql", draft=PLAN):
    return {"status": "answered", "sql": "SELECT 1",
            "trace": {"sections": {"agent": {"kind": kind, "draft": draft, "question": "q",
                                              "tool_calls": [{"id": "c1", "ms": 3.0, **LOG[0]}],
                                              "steps": [{"raw": "internal"}]}},
                      "timeline": ["internal"]}}


def test_serialize_lifts_agent_memory_and_still_strips_the_trace():
    out = _serialize({"items": [{"status": "ok", "result": _result_with_agent()}]})
    res = out["items"][0]["result"]
    assert "trace" not in res
    assert res["agent_memory"] == {"kind": "sql", "draft": PLAN, "question": "q",
                                   "tool_calls": [LOG[0]]}


def test_serialize_lifts_nothing_for_a_non_sql_agent_turn_or_no_agent():
    for res in (_result_with_agent(kind="clarify"), _result_with_agent(draft=None),
                {"status": "answered", "trace": {"sections": {}}}):
        assert "agent_memory" not in _serialize(res)
