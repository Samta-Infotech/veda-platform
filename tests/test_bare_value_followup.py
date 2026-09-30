"""A bare data value keeps the conversation it arrives in — and the api tier's grounding
facts actually reach the graph.

Measured 2026-09-25 (DRILLDOWN_QUERY_MATRIX scenario 27): after "What is the distribution
of properties by facing?", a bare "Nagpur" was labelled answer/ambiguous on one run and
followup on the next. The answer run carried no context, and the engine grounded "Nagpur"
on a city/phone-code lookup table instead of the properties being discussed.

Found alongside it: `message_mentions_data` — the api tier's precomputed "does this name
anything in the data" bool — was never declared in ChatState, so LangGraph dropped it from
the graph input and the guard it feeds had been dead since it became a bool.

Run: `PYTHONPATH=. python -m pytest tests/test_bare_value_followup.py -q`
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chatbot.nodes as nodes  # noqa: E402

_FRAME = {"entity": "assets_asset", "entity_display": "Assets", "source_id": 2,
          "route": "deterministic", "filters": []}
_HISTORY = [{"role": "user", "content": "What is the distribution of properties by facing?"},
            {"role": "assistant", "content": "EAST is the most common facing."}]


def _classify(monkeypatch, message, *, values_only, verdict='{"action":"answer",'
              '"delta_type":"ambiguous"}', frame=_FRAME, history=_HISTORY):
    monkeypatch.setattr(nodes, "call_slm", lambda *a, **k: verdict)
    return nodes.classify_node({"message": message, "history": history, "frame": frame,
                                "message_names_only_values": values_only}, config={})


# ── the measured failure ──────────────────────────────────────────────────────────────
def test_a_bare_value_continues_the_conversation(monkeypatch):
    out = _classify(monkeypatch, "Nagpur", values_only=True)
    assert out["action"] == "followup" and out["delta_type"] == "ambiguous"


def test_also_when_the_model_said_new_topic(monkeypatch):
    out = _classify(monkeypatch, "Nagpur", values_only=True,
                    verdict='{"action":"answer","delta_type":"new_topic"}')
    assert out["action"] == "followup"


# ── deliberately narrow ───────────────────────────────────────────────────────────────
def test_a_word_that_names_a_table_is_a_subject_not_a_value(monkeypatch):
    """names_only_values is False for "vendors" (a table word): the model's call stands."""
    out = _classify(monkeypatch, "vendors", values_only=False,
                    verdict='{"action":"answer","delta_type":"new_topic"}')
    assert out["action"] == "answer" and out["delta_type"] == "new_topic"


def test_undecidable_changes_nothing(monkeypatch):
    out = _classify(monkeypatch, "Nagpur", values_only=None)
    assert out["action"] == "answer"


def test_no_frame_no_continuation(monkeypatch):
    """Nothing to continue: the rule does not fire and the model's call stands."""
    out = _classify(monkeypatch, "Nagpur", values_only=True, frame={})
    assert out["action"] == "answer"


def test_a_placed_turn_is_left_alone(monkeypatch):
    out = _classify(monkeypatch, "Nagpur", values_only=True,
                    verdict='{"action":"followup","delta_type":"refine"}')
    assert out["action"] == "followup" and out["delta_type"] == "refine"


# ── the grounding facts reach the graph ───────────────────────────────────────────────
@pytest.mark.parametrize("key", ["message_mentions_data", "message_names_only_values"])
def test_the_grounding_facts_are_declared(key):
    """LangGraph drops an undeclared INPUT key as silently as an undeclared output."""
    from chatbot.state import ChatState
    assert key in ChatState.__annotations__


# ── the vocabulary decision itself ────────────────────────────────────────────────────
class TestNamesOnlyValues:
    @pytest.fixture(autouse=True)
    def _split(self, monkeypatch):
        dv = pytest.importorskip("apps.query.data_vocabulary")
        monkeypatch.setattr(dv, "_split_for", lambda _s: (
            frozenset({"vendor", "vendors", "furnishing", "city", "name", "property",
                       "properties", "transaction", "transactions"}),
            frozenset({"nagpur", "mumbai", "east", "debit", "full"})))
        self.f = dv.names_only_values

    @pytest.mark.parametrize("m", ["Nagpur", "EAST", "debit", "Mumbai?", "FULL",
                                   "only the Nagpur ones", "only the FULL ones"])
    def test_values(self, m):
        assert self.f(m, [2]) is True

    @pytest.mark.parametrize("m", ["vendors", "properties in Nagpur", "furnishing",
                                   "Nagpur transactions", "zzzunknown"])
    def test_not_values(self, m):
        assert self.f(m, [2]) is False

    def test_nothing_to_decide_with_is_none(self, monkeypatch):
        import apps.query.data_vocabulary as dv
        assert self.f("hello there", [2]) is None           # no content words
        monkeypatch.setattr(dv, "_split_for", lambda _s: (frozenset(), frozenset()))
        assert self.f("Nagpur", [2]) is None                # no vocabulary


def test_followup_new_topic_on_a_value_is_not_a_new_topic(monkeypatch):
    """Edge E7: "what about Pune?" labelled followup/new_topic wiped the drill path."""
    out = _classify(monkeypatch, "what about Pune?", values_only=True,
                    verdict='{"action":"followup","delta_type":"new_topic"}')
    assert out["action"] == "followup" and out["delta_type"] == "ambiguous"
