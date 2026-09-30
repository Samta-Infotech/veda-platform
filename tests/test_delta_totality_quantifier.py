"""A totality quantifier ("all", "every", "overall", "entire", "whole") named alongside a
measure means the user wants the UNFILTERED, ungrouped whole set — not a change_measure
refinement of the conversation's current narrowing.

Measured live 2026-09-30: mid-drill (city=Nagpur, furnishing=FULL, grouped by facing),
"what is the average carpet area of all properties?" was still classified `change_measure`
by `measure_word_named_column` — "all" sits in `_STOP` with the rest of the closed
function-word set and was silently dropped — so the answer stayed filtered to Nagpur+FULL
and grouped by facing, the opposite of what was asked ("all properties").

Fix is narrow: rule 6 (measure_word_named_column) is veto'd by a totality word with no
anaphora, falling back to OP_AMBIGUOUS (the SLM decides, same as any case the rules aren't
sure of) rather than forcing either a wrong "stay filtered" or a guessed "definitely a new
topic" outcome. The SAME question WITHOUT "all" is untouched — it still means the current
set (see test_the_rule_is_unaffected_without_a_totality_word below).

Run: `PYTHONPATH=.:veda_core python -m pytest tests/test_delta_totality_quantifier.py -q`
"""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import chatbot.memory.delta as D  # noqa: E402

ENTRY = {
    "question": "What is the distribution of properties by facing?",
    "drill_options": {"dimensions": ["facing", "corner_property"],
                      "measures": ["carpet_area", "expected_price"]},
    "result": {"top_values": {}},
}


@pytest.mark.parametrize("msg", [
    "what is the average carpet area of all properties?",
    "what is the average carpet area of every property?",
    "what is the overall average carpet area?",
    "show the entire carpet area average",
])
def test_a_totality_word_with_no_anaphora_vetoes_change_measure(msg):
    out = D.detect(msg, ENTRY, stack=None)
    assert out["op"] == D.OP_AMBIGUOUS
    assert out["rule"] == "totality_quantifier_no_anaphora"
    # Not flagged a continuation either — an SLM "new_topic" verdict for this message must
    # not be vetoed back into ambiguous by continues_thread().
    assert not D.continues_thread(out, msg)


def test_the_rule_is_unaffected_without_a_totality_word():
    out = D.detect("what is the average carpet area?", ENTRY, stack=None)
    assert out["op"] == D.OP_CHANGE_MEASURE
    assert out["rule"] == "measure_word_named_column"
    assert out["concept"] == "carpet_area"


def test_a_totality_word_WITH_anaphora_still_means_the_current_set():
    """"what is the average carpet area of all of THOSE" still has a back-reference —
    "those" is anaphora, so it stays a measure change on the current result, exactly as
    before. Totality alone is not enough; it has to be un-anchored."""
    out = D.detect("what is the average carpet area of all of those?", ENTRY, stack=None)
    assert out["op"] == D.OP_CHANGE_MEASURE
    assert out["rule"] == "measure_word"


def test_totality_words_elsewhere_in_a_filter_phrase_are_not_touched():
    """The veto only fires on rule 6's own path (a measure word named). A totality word
    in an unrelated follow-up shape is untouched — e.g. a pure drill/filter phrase never
    reaches rule 6 at all."""
    out = D.detect("only the FULL ones", ENTRY, stack=None)
    assert out["rule"] != "totality_quantifier_no_anaphora"
