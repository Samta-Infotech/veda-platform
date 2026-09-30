"""One decision record per turn (VEDA_MEMORY_LAYER_PLAN.md step 8, M5).

Answers "why did VEDA interpret this follow-up this way?" from ONE structured record keyed
by request_id: what memory held when the turn started, how it was classified and on what
evidence, what was sent to the engine, and whether memory committed. Must never contain
filter values, message text or SQL.

Run: `PYTHONPATH=.:veda_core python -m pytest tests/test_turn_decision_record.py -q`
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from test_topic_index import NAGPUR, PROPS_Q, Session, _result, no_model, redis  # noqa: E402,F401
from chatbot.state import ChatState  # noqa: E402
from chatbot.telemetry import decision_record, memory_summary  # noqa: E402

SECRET = "Ashutosh-Private-Value-42"

_FINAL = {
    "request_id": "rid-1", "session_id": "s1",
    "message": f"only the ones for {SECRET}",
    "action": "followup", "delta_type": "refine",
    "message_names_only_values": True, "message_mentions_data": True,
    "memory_in": memory_summary({"entity": "assets_asset", "source_id": 2, "version": 3,
                                 "filters": [{"column": "city_name", "value": SECRET}]},
                                [{"dimension": "Location"}], None, [{"entity": "a"}], []),
    "conversation_context": {"user_message": f"only the ones for {SECRET}",
                             "entity_table": "assets_asset", "source_id": 2,
                             "operation": "refine",
                             "filters": [{"column": "city_name", "operator": "equals",
                                          "value": SECRET}],
                             "group_by": ["facing"], "resolved_terms": ["second", "one"]},
    "engine_result": {"status": "answered", "_route": "deterministic", "table": "assets_asset",
                      "rows": [[SECRET, 1]],
                      "sql": f"SELECT * FROM assets_asset WHERE city_name = '{SECRET}'"},
    "status": "answered", "rows": [[SECRET, 1]],
    "frame": {"entity": "assets_asset", "version": 4,
              "filters": [{"column": "city_name", "value": SECRET}]},
    "reply_text": f"There are 17 properties in {SECRET}.",
}


def test_the_record_answers_the_four_questions():
    r = decision_record(_FINAL)
    assert r["request_id"] == "rid-1"
    assert r["memory_in"]["entity"] == "assets_asset" and r["memory_in"]["drill_depth"] == 1
    assert r["interpretation"]["action"] == "followup"
    assert r["interpretation"]["evidence"]["names_only_values"] is True
    assert r["interpretation"]["evidence"]["result_pointer_terms"] == 2
    assert r["context_sent"]["filter_columns"] == ["city_name"]
    assert r["context_sent"]["operation"] == "refine"
    assert r["outcome"]["memory_committed"] is True and r["outcome"]["memory_version"] == 4


def test_no_values_no_message_no_sql_no_prose():
    blob = json.dumps(decision_record(_FINAL), default=str)
    assert SECRET not in blob
    assert "SELECT" not in blob and "There are" not in blob


def test_a_refused_turn_did_not_commit():
    r = decision_record({**_FINAL, "status": "refuse",
                         "frame": {"entity": "assets_asset", "version": 3}})
    assert r["outcome"]["memory_committed"] is False


def test_a_turn_without_the_engine():
    r = decision_record({"action": "smalltalk", "status": None, "request_id": "r"})
    assert r["outcome"]["engine_called"] is False and r["context_sent"] is None


def test_memory_in_is_declared_and_recorded_by_the_real_read(redis, no_model):
    """Declared, or LangGraph drops it; recorded by memory_read_node at turn start."""
    assert "memory_in" in ChatState.__annotations__
    s = Session()
    s.answered(PROPS_Q, _result("assets_asset", "Assets", group=["facing"]), "new_topic",
               action="answer")
    s.answered("only the Nagpur ones", _result("assets_asset", "Assets", NAGPUR,
                                               group=["facing"]), "refine")
    s.read("anything")
    mem = s.state["memory_in"]
    assert mem["entity"] == "assets_asset" and mem["filter_columns"] == ["city_name"]
    assert "nagpur" not in json.dumps(mem)
