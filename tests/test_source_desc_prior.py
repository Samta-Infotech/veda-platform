import ingestion.m3_encoder as ENC
import numpy as np
import traceback
"""Tests for the source-description prior (flag-gated, default OFF).

The prior scores query ↔ sources_source.description, reusing the embed-once query vector, and blends
it into the source-aboutness signal (max with item-prior) BEFORE tiering — additive, only promoting a
source whose description matches. Pure-logic + provider tests use fakes so no live DB/model is needed;
routing integration (DB/DataLake/doc/MULTI/SLM) is covered by the live A/B in the implementation doc.
Run: `python tests/test_source_desc_prior.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

import config as _config  # noqa: E402
import query.source_coordinator as SC  # noqa: E402
from query.source_evidence import SourceEvidence  # noqa: E402


# ── Test 1 — flag OFF: provider is a no-op (no DB, no model) ──────────────────────────────────
def test_flag_off_provider_empty():
    _config.SOURCE_DESC_PRIOR_ENABLED = False
    assert SC._source_desc_prior_provider("what is the leave policy", (2, 3, 4)) == {}


# ── Test 2 — flag ON: prior attached to the correct source + blended ─────────────────────────
def test_apply_blends_and_records():
    ev = {"2": SourceEvidence(source_id="2", top_item_score=0.40),
          "3": SourceEvidence(source_id="3", top_item_score=0.30)}
    SC._apply_source_desc_prior(ev, {"3": 0.55})          # doc's description matches strongly
    assert abs(ev["3"].top_desc_score - 0.55) < 1e-9      # recorded (traceability)
    assert abs(ev["3"].top_item_score - 0.55) < 1e-9      # promoted (max(0.30, 0.55))
    assert abs(ev["2"].top_item_score - 0.40) < 1e-9      # untouched (no prior for src2)
    assert ev["2"].top_desc_score == 0.0


# ── Test 3 — query embedding reuse (embed ONCE) ──────────────────────────────────────────────
def test_query_embedding_reused_once():
    _config.SOURCE_DESC_PRIOR_ENABLED = True
    calls = {"n": 0}
    orig_qe = SC._query_embedding
    orig_ld = SC._load_source_descriptions
    orig_de = SC._desc_embedding
    SC._query_embedding = lambda q: (calls.__setitem__("n", calls["n"] + 1) or np.array([1.0, 0.0]))
    SC._load_source_descriptions = lambda sids: {"3": "hr handbook"}
    SC._desc_embedding = lambda t: np.array([1.0, 0.0])
    try:
        out = SC._source_desc_prior_provider("q", (2, 3))
        assert calls["n"] == 1                       # query embedded exactly once inside the provider
        assert abs(out["3"] - 1.0) < 1e-6            # identical vectors → cosine 1.0
    finally:
        SC._query_embedding, SC._load_source_descriptions, SC._desc_embedding = orig_qe, orig_ld, orig_de
        _config.SOURCE_DESC_PRIOR_ENABLED = False


# ── Test 4 — source scope: only in-scope sources participate ─────────────────────────────────
def test_only_in_scope_sources():
    _config.SOURCE_DESC_PRIOR_ENABLED = True
    orig_qe, orig_ld, orig_de = SC._query_embedding, SC._load_source_descriptions, SC._desc_embedding
    SC._query_embedding = lambda q: np.array([1.0, 0.0])
    # _load_source_descriptions is the scope gate (SELECT ... WHERE id IN scope); emulate it honoring sids
    SC._load_source_descriptions = lambda sids: {s: "desc" for s in ["3"] if s in {str(x) for x in sids}}
    SC._desc_embedding = lambda t: np.array([1.0, 0.0])
    try:
        assert set(SC._source_desc_prior_provider("q", (2, 3)).keys()) == {"3"}
        assert SC._source_desc_prior_provider("q", (2, 4)) == {}   # src3 out of scope → no signal
    finally:
        SC._query_embedding, SC._load_source_descriptions, SC._desc_embedding = orig_qe, orig_ld, orig_de
        _config.SOURCE_DESC_PRIOR_ENABLED = False


# ── Test 5 — missing/empty description → no signal, routing untouched ─────────────────────────
def test_missing_description_no_signal():
    assert SC._desc_embedding("") is None
    assert SC._desc_embedding(None) is None
    assert SC._desc_embedding("   ") is None
    ev = {"3": SourceEvidence(source_id="3", top_item_score=0.30)}
    SC._apply_source_desc_prior(ev, {})              # no prior for src3
    assert ev["3"].top_item_score == 0.30 and ev["3"].top_desc_score == 0.0


# ── Test 6 — embedding failure → no signal (never raises) ─────────────────────────────────────
def test_embedding_failure_safe():
    _config.SOURCE_DESC_PRIOR_ENABLED = True
    orig_qe, orig_ld, orig_de = SC._query_embedding, SC._load_source_descriptions, SC._desc_embedding
    SC._query_embedding = lambda q: np.array([1.0, 0.0])
    SC._load_source_descriptions = lambda sids: {"3": "hr"}
    SC._desc_embedding = lambda t: None              # embedding failed
    try:
        assert SC._source_desc_prior_provider("q", (2, 3)) == {}     # no signal, no crash
    finally:
        SC._query_embedding, SC._load_source_descriptions, SC._desc_embedding = orig_qe, orig_ld, orig_de
        _config.SOURCE_DESC_PRIOR_ENABLED = False


# ── Test 7 — changed description re-embeds (cache keyed on text, no stale vector) ─────────────
def test_desc_cache_keyed_on_text():
    seen = []
    orig = ENC.encode_dense
    ENC.encode_dense = lambda texts: (seen.append(texts[0]) or [[float(len(texts[0])), 0.0]])
    SC._DESC_EMB_CACHE.clear()
    try:
        SC._desc_embedding("v1 description"); SC._desc_embedding("v1 description")   # same → 1 embed
        assert seen == ["v1 description"]
        SC._desc_embedding("v2 changed description")                                 # changed → re-embed
        assert seen == ["v1 description", "v2 changed description"]
    finally:
        ENC.encode_dense = orig
        SC._DESC_EMB_CACHE.clear()


# ── Test 8 — apply only PROMOTES (never demotes an already-higher item-prior) ────────────────
def test_apply_never_demotes():
    ev = {"2": SourceEvidence(source_id="2", top_item_score=0.50)}
    SC._apply_source_desc_prior(ev, {"2": 0.20})     # weaker desc than the item-prior
    assert ev["2"].top_item_score == 0.50            # unchanged — max keeps the higher item-prior
    assert abs(ev["2"].top_desc_score - 0.20) < 1e-9 # still recorded for trace


# ── Test 9 — apply SEEDS a source that had no evidence (mirrors item-prior) ───────────────────
def test_apply_seeds_new_source():
    ev = {}
    SC._apply_source_desc_prior(ev, {"3": 0.44})
    assert "3" in ev and abs(ev["3"].top_item_score - 0.44) < 1e-9


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print("PASS", name)
        except Exception:
            failed += 1; print("FAIL", name); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
