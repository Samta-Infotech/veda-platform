"""The model's typed `render` output (VEDA_MEMORY_LAYER_PLAN.md step 7).

The whole-message regexes stay the fast path for "show that as a table". Phrasings they
miss — recorded as KNOWN GAPS in evaluation/conversation/build_suite.py: "can i see that
as a chart", "draw it", "show me a graph of that", "as a bar graph instead" — now come
from the supervisor as a closed-set `render` value that code validates. Presentation only
re-renders the previous result; it cannot change the analysis or memory.

Run: `PYTHONPATH=. python -m pytest tests/test_typed_render_operation.py -q`
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chatbot.nodes as nodes  # noqa: E402

_FRAME = {"entity": "assets_asset", "entity_display": "Assets", "source_id": 2,
          "route": "deterministic", "filters": []}
_PREV = {"status": "answered", "cols": ["facing", "n"], "rows": [["EAST", 10], ["WEST", 4]],
         "table": "assets_asset"}
_HISTORY = [{"role": "user", "content": "distribution of properties by facing"},
            {"role": "assistant", "content": "EAST has the most."}]


def _classify(monkeypatch, message, render, *, mentions=False, frame=_FRAME, prev=_PREV):
    monkeypatch.setattr(nodes, "call_slm", lambda *a, **k: (
        '{"action":"followup","delta_type":"ambiguous","render":"%s"}' % render))
    return nodes.classify_node({"message": message, "history": _HISTORY, "frame": frame,
                                "last_result": prev,
                                "message_mentions_data": mentions}, config={})


# ── the recorded gaps now re-render ───────────────────────────────────────────────────
@pytest.mark.parametrize("msg,render,expected", [
    ("can i see that as a chart", "chart", "chart"),
    ("draw it", "chart", "chart"),
    ("show me a graph of that", "graph", "chart"),       # alias folded to the closed set
    ("as a bar graph instead", "bar", "bar"),
])
def test_recorded_gaps_are_honoured(monkeypatch, msg, render, expected):
    out = _classify(monkeypatch, msg, render)
    assert out["action"] == "represent" and out["viz_override"] == expected


# ── fail closed ───────────────────────────────────────────────────────────────────────
def test_a_message_that_names_data_is_a_data_question(monkeypatch):
    """"the sales as a bar chart" names data → the engine, as before."""
    out = _classify(monkeypatch, "show the sales as a bar chart", "bar", mentions=True)
    assert out["action"] != "represent"


def test_an_undecidable_vocabulary_check_never_honours_the_model(monkeypatch):
    out = _classify(monkeypatch, "draw it", "chart", mentions=None)
    assert out["action"] != "represent"


@pytest.mark.parametrize("render", ["none", "", "hologram", "DROP TABLE", "null"])
def test_only_the_closed_set(monkeypatch, render):
    assert _classify(monkeypatch, "draw it", render)["action"] != "represent"


def test_nothing_to_redraw(monkeypatch):
    out = _classify(monkeypatch, "draw it", "chart", prev={"rows": []})
    assert out["action"] != "represent"


def test_a_document_answer_is_not_charted(monkeypatch):
    doc = dict(_FRAME, entity="maintenance_policy.docx", entity_is_document=True,
               route="rag")
    out = _classify(monkeypatch, "draw it", "chart", frame=doc)
    assert out["action"] != "represent"


def test_the_regex_fast_path_still_runs_first(monkeypatch):
    """A phrasing the regex knows never reaches the model at all."""
    def _boom(*a, **k):
        raise AssertionError("the model was called for a regex-covered phrasing")
    monkeypatch.setattr(nodes, "call_slm", _boom)
    out = nodes.classify_node({"message": "show that as a table", "history": _HISTORY,
                               "frame": _FRAME, "last_result": _PREV}, config={})
    assert out["action"] == "represent" and out["viz_override"] == "table"


def test_a_delta_is_still_parsed_when_render_is_none(monkeypatch):
    """Regression guard for the slip made while wiring this: the render block must not
    make the delta parse unreachable."""
    monkeypatch.setattr(nodes, "call_slm", lambda *a, **k: (
        '{"action":"answer","delta_type":"new_topic","render":"none"}'))
    out = nodes.classify_node({"message": "how many vendors are there", "history": _HISTORY,
                               "frame": _FRAME, "last_result": _PREV}, config={})
    assert out["delta_type"] == "new_topic"
