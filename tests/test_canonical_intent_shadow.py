import traceback
"""Tests for the Canonical-QueryIntent SHADOW measurement spike (Phase 1, observe-only, flag-gated).

Covers: the preserve-on-decline ContextVar side-channel (capture/read/reset) and the field-level
agreement comparator (MATCH/MISMATCH/NOT_APPLICABLE/UNKNOWN). The shadow layer never changes SQL —
these prove the MEASUREMENT is correct and that it is inert when the flag is off.
Run: `python tests/test_canonical_intent_shadow.py`.
"""
import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

import config as _config  # noqa: E402
import query.fast_path as FP  # noqa: E402
from veda import canonical_intent_shadow as S  # noqa: E402


def _flt(col):
    return SimpleNamespace(column=col)


def _intent(query_type="count", subject_table="assets_asset", select_expr=None,
            group_col=None, group_col2=None, time_bucket=None, filters=()):
    return SimpleNamespace(query_type=query_type, subject_table=subject_table, select_expr=select_expr,
                           group_col=group_col, group_col2=group_col2, time_bucket=time_bucket,
                           filters=list(filters))


# ── ContextVar side-channel: capture / read / reset ────────────────────────────────────────────────

def test_capture_and_read():
    FP._FASTPATH_INTENT_QV.set(None)
    it = _intent()
    FP._capture_intent(it, "VALIDATION_DECLINE")
    got = FP.get_preserved_intent()
    assert got and got["intent"] is it and got["reason"] == "VALIDATION_DECLINE"


def test_reset_clears_stale():
    FP._capture_intent(_intent(), "QUALIFIER_GROUNDING_DECLINE")
    FP._FASTPATH_INTENT_QV.set(None)                     # what try_fast_path does at entry
    assert FP.get_preserved_intent() is None


# ── comparator: field-level agreement ──────────────────────────────────────────────────────────────

def test_agg_match_and_mismatch():
    it = _intent(query_type="count")
    assert S.shadow_agreement(it, "SELECT COUNT(*) FROM assets_asset", {})["aggregation"] == S.MATCH
    assert S.shadow_agreement(it, "SELECT AVG(rent) FROM assets_asset", {})["aggregation"] == S.MISMATCH


def test_entity_match_and_mismatch():
    it = _intent(subject_table="assets_asset")
    assert S.shadow_agreement(it, "SELECT COUNT(*) FROM assets_asset", {})["entity"] == S.MATCH
    assert S.shadow_agreement(it, "SELECT COUNT(*) FROM leads_lead", {})["entity"] == S.MISMATCH


def test_dimension_match_mismatch_na():
    it_g = _intent(group_col="city_name")
    assert S.shadow_agreement(it_g, "SELECT city_name, COUNT(*) FROM assets_asset GROUP BY city_name",
                              {})["dimension"] == S.MATCH
    assert S.shadow_agreement(it_g, "SELECT country, COUNT(*) FROM assets_asset GROUP BY country",
                              {})["dimension"] == S.MISMATCH
    it_none = _intent(group_col=None)
    assert S.shadow_agreement(it_none, "SELECT COUNT(*) FROM assets_asset", {})["dimension"] == S.NA


def test_temporal_match_and_na():
    sm = {"columns": {"leads_lead.created_at": {"table_name": "leads_lead", "semantic_type": "TEMPORAL"}}}
    it_t = _intent(query_type="count", time_bucket="month")
    assert S.shadow_agreement(it_t, "SELECT DATE_TRUNC('month', created_at), COUNT(*) FROM leads_lead GROUP BY 1",
                              sm)["temporal"] == S.MATCH
    it_nt = _intent(query_type="count", time_bucket=None)
    assert S.shadow_agreement(it_nt, "SELECT COUNT(*) FROM leads_lead", sm)["temporal"] == S.NA


def test_filter_unknown_on_grounding_decline():
    it = _intent(filters=[_flt("city_name")])
    # default reason (not a grounding-decline): intent has a filter, SQL has none → MISMATCH.
    assert S.shadow_agreement(it, "SELECT COUNT(*) FROM assets_asset", {})["filters"] == S.MISMATCH
    # on a QUALIFIER_GROUNDING_DECLINE the intent's filter was the ungrounded decline reason → UNKNOWN.
    assert S.shadow_agreement(it, "SELECT COUNT(*) FROM assets_asset", {}, "QUALIFIER_GROUNDING_DECLINE"
                              )["filters"] == S.UNKNOWN


# ── flag gating: record_shadow is inert when OFF ────────────────────────────────────────────────────

def test_record_shadow_flag_off_noop():
    _config.CANONICAL_INTENT_SHADOW_ENABLED = False
    FP._capture_intent(_intent(), "VALIDATION_DECLINE")
    assert S.record_shadow("how many assets", "SELECT COUNT(*) FROM assets_asset", {}) is None


def test_record_shadow_flag_on_records():
    _config.CANONICAL_INTENT_SHADOW_ENABLED = True
    try:
        FP._FASTPATH_INTENT_QV.set(None)
        FP._capture_intent(_intent(query_type="count", subject_table="assets_asset"),
                           "VALIDATION_DECLINE")
        rec = S.record_shadow("how many assets", "SELECT COUNT(*) FROM assets_asset", {})
        assert rec and rec["fastpath_decline_reason"] == "VALIDATION_DECLINE"
        assert rec["agreement"]["aggregation"] == S.MATCH and rec["agreement"]["entity"] == S.MATCH
    finally:
        _config.CANONICAL_INTENT_SHADOW_ENABLED = False


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
