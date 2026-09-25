"""Drill-down / drill-up reliability — the two defects DRILLDOWN_QUERY_MATRIX found.

B2 (scenario 1): base "distribution by facing" -> Nagpur -> FULL -> "go back" returned
   1000 raw rows. The conversation layer sent the right context (city filter, group_by,
   aggregation) but the engine could not tell a drill-up REPLAY of the base question from
   a new grouping request, because it inferred that from the text. The layer now says so:
   ConversationContext.operation.

B3 (scenario 2): Pune, then EAST (labelled drill_down) left the stack as
   ['Location', 'Location']. A drill_down turn took its own branch, push_drill(), which
   pushes "the last filter in the list" — the SQL walker's order, not the user's.

Run: `PYTHONPATH=.:veda_core python -m pytest tests/test_drill_reliability.py -q`
"""
import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chatbot import nodes as N  # noqa: E402
from chatbot.memory.context import ConversationContext  # noqa: E402
from chatbot.memory.store import MemoryStore  # noqa: E402


# ── B2: the operation travels, typed ──────────────────────────────────────────────────
_FRAME = {"entity": "assets_asset", "source_id": 2, "route": "deterministic",
          "filters": [{"field": "Location", "column": "city_name", "value": "nagpur"}],
          "group_by": ["facing", "corner_property"], "aggregation": "count"}


def test_drill_up_is_stated_in_the_context():
    ctx = ConversationContext.from_frame(_FRAME, "What is the distribution of properties "
                                         "by facing?", operation="drill_up")
    assert ctx.to_payload()["operation"] == "drill_up"


def test_every_known_delta_travels():
    from chatbot.prompts.delta_types import DELTA_TYPES
    for op in DELTA_TYPES:
        p = ConversationContext.from_frame(_FRAME, "m", operation=op).to_payload()
        assert p.get("operation") == op, op


def test_an_unknown_operation_never_travels():
    for op in ("DROP TABLE", "drill up", "", None, "delete_all"):
        p = ConversationContext.from_frame(_FRAME, "m", operation=op).to_payload()
        assert "operation" not in p, op


def test_case_and_whitespace_are_normalised_not_rejected():
    p = ConversationContext.from_frame(_FRAME, "m", operation=" Drill_Up ").to_payload()
    assert p["operation"] == "drill_up"


def test_no_operation_keeps_the_payload_byte_identical():
    """Callers that pass no operation (every non-chat caller) see no new key."""
    assert "operation" not in ConversationContext.from_frame(_FRAME, "m").to_payload()


def test_a_new_topic_carries_no_state_but_may_state_its_operation():
    p = ConversationContext.from_frame(_FRAME, "m", carry_state=False,
                                       operation="new_topic").to_payload()
    assert p == {"user_message": "m"}      # carry_state=False returns before any state


def test_inference_boundary_accepts_only_identifier_shaped_operations():
    from inference.routes.hybrid import _validated_conversation_context as V
    ok = V({"conversation_context": {"user_message": "m", "operation": "drill_up"}})
    assert ok["operation"] == "drill_up"
    for bad in ("drop table x", "drill_up; --", "X" * 40, 7, ["drill_up"]):
        out = V({"conversation_context": {"user_message": "m", "operation": bad}}) or {}
        assert "operation" not in out, bad


# ── B3: the level pushed is the filter the query ADDED ────────────────────────────────
class TestDrillStackLevel:
    def setup_method(self):
        self._orig = {n: getattr(MemoryStore, n)
                      for n in ("write_frame", "write_stack", "push_episodic_turn",
                                "write_comparison")}
        self.stack = None

        def _stack(*a, **k):
            self.stack = a[2] if len(a) > 2 else k.get("stack")
        for n in self._orig:
            setattr(MemoryStore, n, staticmethod(lambda *a, **k: None))
        MemoryStore.write_stack = staticmethod(_stack)

    def teardown_method(self):
        for n, fn in self._orig.items():
            setattr(MemoryStore, n, fn)

    @staticmethod
    def _state(delta, prev_filters, prev_stack, harvested_filters):
        applied = [{"field": f, "column": c, "operator": "equals", "value": v}
                   for f, c, v in harvested_filters]
        return {"status": "answered", "tenant": "t", "session_id": str(uuid.uuid4()),
                "message": "only the EAST ones", "delta_type": delta, "source_id": 2,
                "action": "followup",
                "frame": {"entity": "assets_asset", "route": "deterministic",
                          "source_id": 2,
                          "filters": [{"field": f, "column": c, "operator": "equals",
                                       "value": v} for f, c, v in prev_filters]},
                "drill_stack": prev_stack,
                "engine_result": {
                    "status": "answered", "table": "assets_asset", "rows": [[1]],
                    "cols": ["a"], "sql": "SELECT 1", "_route": "deterministic",
                    "analytics": {"orderings": [], "limit": None, "query_measures": []},
                    "explain": {"data_used": {"datasets": ["Assets"]},
                                "filters": {"applied": applied},
                                "operations": [], "understanding": {"summary": "s"}}}}

    _PUNE = [("Location", "city_name", "pune")]
    _LVL = [{"dimension": "Location", "column": "city_name", "value": "pune"}]
    # The walker's order: the older filter comes LAST — exactly the measured case.
    _BOTH = [("Direction Facing", "facing", "east"), ("Location", "city_name", "pune")]

    def test_the_measured_case_pushes_the_new_dimension(self):
        N.memory_write_node(self._state("drill_down", self._PUNE, self._LVL, self._BOTH))
        assert [l["dimension"] for l in self.stack] == ["Location", "Direction Facing"]
        assert self.stack[-1]["column"] == "facing" and self.stack[-1]["value"] == "east"

    def test_the_label_does_not_matter(self):
        for delta in ("drill_down", "refine", "ambiguous"):
            N.memory_write_node(self._state(delta, self._PUNE, self._LVL, self._BOTH))
            assert [l["dimension"] for l in self.stack] == ["Location", "Direction Facing"], delta

    def test_a_replaced_value_does_not_deepen_the_path(self):
        """drill_down on a field already constrained is a replacement: one level, new value."""
        N.memory_write_node(self._state("drill_down", self._PUNE, self._LVL,
                                        [("Location", "city_name", "mumbai")]))
        assert [(l["dimension"], l["value"]) for l in self.stack] == [("Location", "mumbai")]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
