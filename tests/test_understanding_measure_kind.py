"""veda.understanding.grounding — the aggregation kind must come from RawIntent.intent
when the measure PHRASE carries no aggregation word.

BUG THIS PINS: ground_measure derived `kind` ONLY by string-matching _AGG_WORDS inside
the measure phrase. But the aggregation is already classified in RawIntent.intent (closed
INTENTS vocabulary) — the measure field is meant to name WHAT is aggregated, not to repeat
the aggregation. So the measure survived only when the extractor was REDUNDANT and was
dropped exactly when it behaved correctly.

Measured on the live extractor (qwen2.5-coder:7b via the external SLM):
    "What is the total carpet area across all properties?"
        -> intent="sum", grain="property", measure="carpet area"
           -> no agg word in "carpet area" -> kind=None -> MEASURE DROPPED
    "What is the total rent across all lease transactions?"
        -> intent="sum", grain="lease transaction", measure="total rent amount"
           -> "total" found -> kind=sum -> kept
Both are correct extractions; only the second happened to be redundant.

Same operator-dropping class as the Tier-2 assemble_ir COUNT(*) defect: the requested
aggregation is available upstream and thrown away downstream.

Pure-python: no DB, no network, no SLM — ground_measure is deterministic.
Run per-file:  python -m pytest tests/test_understanding_measure_kind.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))

import pytest                                                    # noqa: E402
from veda.understanding.grounding import ground_measure          # noqa: E402

TABLES = ["assets_asset", "accounts_paymenttransaction", "users_user"]
JUNC = set()


def _kind(concept, intent=None, anchor="assets_asset"):
    m = ground_measure(concept, anchor, TABLES, JUNC, sm=None, intent=intent)
    return m.kind if m else None


# ══ the regression: aggregation lives in `intent`, phrase is clean ══════════════

@pytest.mark.parametrize("intent,expected", [
    ("sum", "sum"), ("avg", "avg"), ("max", "max"), ("min", "min"), ("count", "count"),
])
def test_kind_falls_back_to_intent(intent, expected):
    """'carpet area' has no aggregation word — the kind must come from intent."""
    assert _kind("carpet area", intent=intent) == expected


def test_the_measured_failure_case():
    """The exact live extraction that was losing its measure."""
    assert _kind("carpet area", intent="sum") == "sum"


def test_phrase_still_wins_when_it_carries_the_word():
    """Unchanged behavior: an aggregation word in the phrase is authoritative, so a
    redundant extraction keeps working exactly as before."""
    assert _kind("total rent amount", intent="sum") == "sum"
    assert _kind("average carpet area", intent="avg") == "avg"


def test_phrase_wins_over_a_disagreeing_intent():
    """The phrase is the more specific signal; intent is only a FALLBACK. Pinned so the
    fallback can never silently override an explicit phrase."""
    assert _kind("average carpet area", intent="sum") == "avg"


# ══ non-aggregating intents must NOT manufacture a measure ══════════════════════

@pytest.mark.parametrize("intent", ["list", "rank", "compare", "refuse", "clarify"])
def test_non_aggregation_intents_do_not_become_a_kind(intent):
    """Only count/sum/avg/max/min are aggregations. 'list' must not turn a plain noun
    into a measure — that would invent an aggregation the user never asked for."""
    assert _kind("carpet area", intent=intent) is None


def test_no_concept_is_still_no_measure():
    """A plain list has no measure at all; the intent fallback must not fabricate one."""
    assert _kind(None, intent="sum") is None
    assert _kind("", intent="sum") is None


def test_unknown_intent_is_ignored():
    assert _kind("carpet area", intent="bogus_intent") is None
    assert _kind("carpet area", intent=None) is None


# ══ back-compat: the parameter is optional ═════════════════════════════════════

def test_intent_param_is_optional():
    """Existing callers that don't pass `intent` keep the previous behavior exactly."""
    assert ground_measure("carpet area", "assets_asset", TABLES, JUNC) is None
    m = ground_measure("total paid amount", "assets_asset", TABLES, JUNC)
    assert m is not None and m.kind == "sum"


def test_count_of_an_entity_still_resolves_its_table():
    """The count branch is unchanged: it grounds the COUNTED entity's table, not the
    anchor's."""
    m = ground_measure("number of payment transactions", "assets_asset", TABLES, JUNC)
    assert m is not None and m.kind == "count"
    assert m.table == "accounts_paymenttransaction"
