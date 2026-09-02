import traceback
"""Tests for the chunk-into-tier blend (flag-gated, default OFF).

_dominance_retier tiers on the item-prior and discards a source's clean chunk evidence; this blend puts
the chunk back into the aboutness signal as max(item_prior, chunk) so a source whose actual CONTENT
matches wins its tier. Run: `python tests/test_chunk_tier_blend.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

import config as _config  # noqa: E402
import query.source_coordinator as SC  # noqa: E402
from query.source_evidence import SourceEvidence  # noqa: E402


def _ev():
    # doc: weak item-prior but STRONG chunk (real content match); homzhub: item-prior, no chunk.
    return {"2": SourceEvidence(source_id="2", top_item_score=0.458, top_chunk_score=0.0),
            "3": SourceEvidence(source_id="3", top_item_score=0.393, top_chunk_score=0.594)}


def test_flag_off_noop():
    _config.CHUNK_TIER_BLEND_ENABLED = False
    ev = _ev()
    SC._apply_chunk_tier_blend(ev)
    assert ev["3"].top_item_score == 0.393        # unchanged — chunk NOT blended
    assert ev["2"].top_item_score == 0.458


def test_flag_on_chunk_wins_for_doc():
    _config.CHUNK_TIER_BLEND_ENABLED = True
    try:
        ev = _ev()
        SC._apply_chunk_tier_blend(ev)
        assert abs(ev["3"].top_item_score - 0.594) < 1e-9   # doc promoted by its chunk (0.594 > 0.393)
        assert abs(ev["2"].top_item_score - 0.458) < 1e-9   # homzhub unchanged (no chunk)
        assert ev["3"].top_item_score > ev["2"].top_item_score   # doc now out-tiers homzhub
    finally:
        _config.CHUNK_TIER_BLEND_ENABLED = False


def test_never_demotes():
    # a strong item-prior with a weaker chunk keeps the item-prior (max).
    _config.CHUNK_TIER_BLEND_ENABLED = True
    try:
        ev = {"2": SourceEvidence(source_id="2", top_item_score=0.60, top_chunk_score=0.20)}
        SC._apply_chunk_tier_blend(ev)
        assert ev["2"].top_item_score == 0.60
    finally:
        _config.CHUNK_TIER_BLEND_ENABLED = False


def test_column_not_blended():
    # only the CHUNK is blended; a noisy column score must NOT lift the tier.
    _config.CHUNK_TIER_BLEND_ENABLED = True
    try:
        ev = {"2": SourceEvidence(source_id="2", top_item_score=0.30,
                                  top_column_score=0.99, top_chunk_score=0.0)}
        SC._apply_chunk_tier_blend(ev)
        assert ev["2"].top_item_score == 0.30      # column (0.99) ignored — only chunk blends
    finally:
        _config.CHUNK_TIER_BLEND_ENABLED = False


def test_dominant_only_weak_chunk_not_promoted():
    # DOMINANT-ONLY: a doc's WEAK chunk (0.453) BELOW the strongest DB column (homzhub 0.523) must NOT
    # promote the doc — so a real DB query ("lease transactions per project") is not dragged to a
    # false-ambiguous co-leader.
    _config.CHUNK_TIER_BLEND_ENABLED = True
    try:
        ev = {"2": SourceEvidence(source_id="2", top_item_score=0.466, top_column_score=0.523),
              "3": SourceEvidence(source_id="3", top_item_score=0.348, top_chunk_score=0.453)}
        SC._apply_chunk_tier_blend(ev)
        assert ev["3"].top_item_score == 0.348     # NOT promoted (0.453 < best column 0.523)
    finally:
        _config.CHUNK_TIER_BLEND_ENABLED = False


def test_dominant_chunk_still_promotes():
    # a DOMINANT doc chunk (0.594) above the strongest column (0.47) STILL promotes → doc wins.
    _config.CHUNK_TIER_BLEND_ENABLED = True
    try:
        ev = {"2": SourceEvidence(source_id="2", top_item_score=0.458, top_column_score=0.470),
              "3": SourceEvidence(source_id="3", top_item_score=0.393, top_chunk_score=0.594)}
        SC._apply_chunk_tier_blend(ev)
        assert abs(ev["3"].top_item_score - 0.594) < 1e-9   # promoted (0.594 > best column 0.47)
    finally:
        _config.CHUNK_TIER_BLEND_ENABLED = False


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
