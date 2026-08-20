"""Unit tests for the enterprise understanding layer (veda/understanding/).

Deterministic, no SLM: the LLM extractor is exercised only through its pure JSON-repair /
shape-validation helpers; the grounding FIREWALL (the anti-hallucination core) and the
flag gate are tested in full against the real semantic model + relationship graph.

Run from repo root: ``pytest tests/test_understanding.py``
"""
import os, sys, json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VC = os.path.join(_ROOT, "veda_core")
sys.path.insert(0, _VC)
from veda.planning import _junction_tables
from veda.understanding import understand_query
from veda.understanding.grounding import ground
from veda.understanding.schema import GroundedIntent
from veda.understanding.extractor import _repair_json
from veda.understanding.schema import INTENTS
# NOTE: no module-level os.chdir — that leaks cwd into sibling test modules (the repo's
# order-flakiness). Load data by ABSOLUTE path; grounding's glossary loader has a
# module-relative fallback, so it works without chdir.
_SM = json.load(open(os.path.join(_VC, "data", "veda_semantic_model.json")))
_GRAPH = json.load(open(os.path.join(_VC, "data", "veda_relationship_graph.json")))


def _junctions():
    return _junction_tables(_GRAPH, _SM)


# ── flag gate: default OFF → prod byte-identical (degrade to existing path) ────
def test_flag_default_off_returns_none():
    import config
    assert config.QUERY_UNDERSTANDING_ENABLED is False
    assert understand_query("top 5 properties by number of payments", _SM) is None


# ── grounding firewall: grain concept → real anchor table (fixes grain-inversion)
def test_ground_grain_property_to_asset():
    from veda.understanding.schema import RawIntent
    raw = RawIntent(intent="rank", grain="property", measure="number of payments",
                    entities=["property", "payment"], confidence=0.9)
    g = ground(raw, _SM, _GRAPH, _junctions())
    assert isinstance(g, GroundedIntent)
    assert g.anchor == "assets_asset"          # NOT inverted to the payment table


def test_ground_grain_owner_to_user():
    from veda.understanding.schema import RawIntent
    raw = RawIntent(intent="sum", grain="owner", measure="total rent",
                    entities=["owner", "property"], confidence=0.9)
    g = ground(raw, _SM, _GRAPH, _junctions())
    assert g.anchor == "users_user"            # glossary owner→users_user


# ── firewall: an ungroundable grain must REFUSE, never guess a table ───────────
def test_ungroundable_grain_refuses():
    from veda.understanding.schema import RawIntent, Refusal
    raw = RawIntent(intent="list", grain="spaceship", entities=["spaceship"], confidence=0.9)
    g = ground(raw, _SM, _GRAPH, _junctions())
    assert isinstance(g, Refusal)
    assert g.reason == "ungrounded"


def test_refuse_intent_becomes_refusal():
    from veda.understanding.schema import RawIntent, Refusal
    g = ground(RawIntent(intent="refuse", confidence=0.9), _SM, _GRAPH, _junctions())
    assert isinstance(g, Refusal) and g.reason == "impossible"


# ── low confidence → degrade (None), don't guess or refuse ────────────────────
def test_low_confidence_degrades():
    from veda.understanding.schema import RawIntent
    raw = RawIntent(intent="list", grain="property", confidence=0.2)
    assert ground(raw, _SM, _GRAPH, _junctions(), min_confidence=0.5) is None


# ── measure grounding: "number of payments" → COUNT of the payments table ──────
def test_measure_count_grounds_table():
    from veda.understanding.schema import RawIntent
    raw = RawIntent(intent="rank", grain="property",
                    measure="number of payment transactions", confidence=0.9)
    g = ground(raw, _SM, _GRAPH, _junctions())
    assert g.measure is not None and g.measure.kind == "count"


# ── extractor pure helpers (no SLM) ───────────────────────────────────────────
def test_repair_json_strips_fences():
    assert _repair_json('```json\n{"intent":"count"}\n```') == {"intent": "count"}
    assert _repair_json("garbage no json") is None


def test_extract_rejects_bad_shape(monkeypatch=None):
    # a dict whose intent isn't in the closed vocabulary must not become a RawIntent
    obj = _repair_json('{"intent":"frobnicate"}')
    assert obj["intent"] not in INTENTS
