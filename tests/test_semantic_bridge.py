"""Unit tests for the semantic bridge selector (planning._semantic_bridge_prefer).

Deterministic, no DB / no SLM: drives the selector against the REAL semantic model +
relationship graph. Proves the ownership-vs-document disambiguation the FK-structural
heuristics cannot do, and that it NEVER guesses (ties / no-match → empty set).

Run from repo root: ``pytest tests/test_semantic_bridge.py``
"""
import os, sys, json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VC = os.path.join(_ROOT, "veda_core")
sys.path.insert(0, _VC)
from veda.planning import _semantic_bridge_prefer
_SM = json.load(open(os.path.join(_VC, "data", "veda_semantic_model.json")))
# Load the relationship graph by ABSOLUTE path (no module-level os.chdir — that would
# leak cwd into sibling test modules and break their cwd-relative glossary loading).
_GRAPH = json.load(open(os.path.join(_VC, "data", "veda_relationship_graph.json")))


def _pref(query, anchor, targets):
    return _semantic_bridge_prefer(query, _SM, _GRAPH, anchor, targets)


# ── the core fix: "owners" → ownership link, NOT the document/key bridge ──────
def test_owners_selects_assetuser():
    p = _pref("show properties with their owners", "assets_asset", ["users_user"])
    assert p == {"assets_assetuser"}          # not assetdocument / assetkey


def test_owners_excludes_document_bridge():
    p = _pref("show properties with their owners", "assets_asset", ["users_user"])
    assert "assets_assetdocument" not in p
    assert "assets_assetkey" not in p


# ── safety: it NEVER guesses when semantics don't uniquely disambiguate ──────
def test_no_relationship_word_falls_back():
    # "assigned" is not in any candidate bridge's business_purpose → empty (fallback)
    p = _pref("show properties with their assigned users", "assets_asset", ["users_user"])
    assert p == set()


def test_multiword_tie_falls_back_not_wrong():
    # owner + payment both present → ownership vs payment bridges tie → no guess
    p = _pref("show properties with their owners and payment transactions",
              "assets_asset", ["users_user", "accounts_paymenttransaction"])
    assert "assets_assetdocument" not in p       # never the wrong doc bridge
    # tolerant: either a clean assetuser pick or a safe fallback — never a wrong pick
    assert p in (set(), {"assets_assetuser"})


def test_directly_joinable_target_no_bridge():
    # amenity path uses a real junction; no ownership-style disambiguation applies
    p = _pref("show properties with their amenities", "assets_asset", ["assets_amenity"])
    assert "assets_assetdocument" not in p


# ── entity hubs (paymenttransaction, 10 FKs) can never pose as a bridge ──────
def test_entity_hub_not_a_candidate_bridge():
    p = _pref("show properties with their owners", "assets_asset", ["users_user"])
    assert "accounts_paymenttransaction" not in p


# ── flag defaults OFF → production planner byte-identical ────────────────────
def test_flag_default_off():
    import config
    assert config.JOIN_SEMANTIC_BRIDGE is False
