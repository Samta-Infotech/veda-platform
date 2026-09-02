import traceback
"""Tests for the shared Intent↔SQL referent-alignment guard (Option B increment 1, flag-gated).

TEMPORAL: a per-time breakdown intent whose SQL groups by a non-temporal column → refuse.
ENTITY-ANCHOR: the query names a measure column on table T but the SQL measures/orders by a column not
on T → refuse. Both fire only on a clear mismatch (no over-refusal) and are no-ops when the flag is off.
Run: `python tests/test_intent_sql_alignment.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

import config as _config  # noqa: E402
from veda import intent_sql_alignment as A  # noqa: E402


def _col(table, sem, role):
    return {"table_name": table, "semantic_type": sem, "analytics_role": role}


# leads_lead: lead_stage (dimension), created_at (temporal); assets_asset: carpet_area (measure);
# assets_carpetareaunit: base_conversion_factor (measure on a reference table).
SM = {"columns": {
    "leads_lead.lead_stage": _col("leads_lead", "CATEGORY", "DIMENSION"),
    "leads_lead.created_at": _col("leads_lead", "TEMPORAL", "TIME_DIMENSION"),
    "assets_asset.carpet_area": _col("assets_asset", "METRIC", "MEASURE"),
    "assets_carpetareaunit.base_conversion_factor": _col("assets_carpetareaunit", "METRIC", "MEASURE"),
    "assets_leaselisting.expected_monthly_rent": _col("assets_leaselisting", "MONETARY", "MEASURE"),
}}


def _on():
    _config.INTENT_SQL_ALIGNMENT_ENABLED = True


def _off():
    _config.INTENT_SQL_ALIGNMENT_ENABLED = False


# ── TEMPORAL ─────────────────────────────────────────────────────────────────────────────────────

def test_temporal_non_temporal_group_refuses():
    _on()
    try:
        ok, _ = A.temporal_alignment_ok(
            "how many leads per month",
            'SELECT lead_stage, COUNT(*) FROM leads_lead GROUP BY lead_stage', SM)
        assert ok is False               # per-month intent, grouped by lead_stage → refuse
    finally:
        _off()


def test_temporal_with_temporal_group_ok():
    _on()
    try:
        ok, _ = A.temporal_alignment_ok(
            "how many leads per month",
            'SELECT created_at, COUNT(*) FROM leads_lead GROUP BY created_at', SM)
        assert ok is True                # grouped by a TEMPORAL column → aligned
    finally:
        _off()


def test_temporal_date_trunc_ok():
    _on()
    try:
        ok, _ = A.temporal_alignment_ok(
            "monthly leads",
            "SELECT DATE_TRUNC('month', created_at), COUNT(*) FROM leads_lead GROUP BY 1", SM)
        assert ok is True                # explicit date bucketing → aligned
    finally:
        _off()


def test_temporal_adverb_in_measure_name_not_refused():
    # "highest monthly rent" — "monthly" is part of the MEASURE expected_monthly_rent, NOT a per-month
    # breakdown → must NOT over-refuse a legitimate ranking/aggregate (the coverage-benchmark false-positive).
    _on()
    try:
        ok, _ = A.temporal_alignment_ok(
            "which lease listings have the highest monthly rent",
            'SELECT id, expected_monthly_rent FROM assets_leaselisting '
            'ORDER BY expected_monthly_rent DESC LIMIT 10', SM)
        assert ok is True                                    # no false temporal refusal
    finally:
        _off()


def test_temporal_per_unit_still_catches_despite_measure_name():
    # "leads per month" still fires via the "per <unit>" prep signal → refuse if grouped non-temporally.
    _on()
    try:
        ok, _ = A.temporal_alignment_ok(
            "how many leads per month",
            'SELECT lead_stage, COUNT(*) FROM leads_lead GROUP BY lead_stage', SM)
        assert ok is False
    finally:
        _off()


def test_no_temporal_intent_untouched():
    _on()
    try:
        # "this month" is a FILTER, not a per-time breakdown → guard silent.
        ok, _ = A.temporal_alignment_ok(
            "how many leads this month",
            'SELECT COUNT(*) FROM leads_lead WHERE created_at >= %s', SM)
        assert ok is True
    finally:
        _off()


# ── ENTITY-ANCHOR ────────────────────────────────────────────────────────────────────────────────

def test_anchor_wrong_table_refuses():
    _on()
    try:
        ok, _ = A.entity_anchor_ok(
            "which asset has the highest carpet area",
            'SELECT name, base_conversion_factor FROM assets_carpetareaunit '
            'ORDER BY base_conversion_factor DESC LIMIT 1', SM)
        assert ok is False               # names carpet_area (assets_asset) but orders by another table's col
    finally:
        _off()


def test_anchor_correct_table_ok():
    _on()
    try:
        ok, _ = A.entity_anchor_ok(
            "which asset has the highest carpet area",
            'SELECT project_name, carpet_area FROM assets_asset ORDER BY carpet_area DESC LIMIT 1', SM)
        assert ok is True                # orders by the named measure column → aligned
    finally:
        _off()


def test_anchor_no_named_measure_untouched():
    _on()
    try:
        # query names no specific measure column → guard cannot misalign → silent.
        ok, _ = A.entity_anchor_ok(
            "how many assets are there", 'SELECT COUNT(*) FROM assets_asset', SM)
        assert ok is True
    finally:
        _off()


# ── flag off + combined ──────────────────────────────────────────────────────────────────────────

def test_flag_off_always_ok():
    _off()
    ok1, _ = A.temporal_alignment_ok("leads per month",
                                     'SELECT lead_stage, COUNT(*) FROM leads_lead GROUP BY lead_stage', SM)
    ok2, _ = A.entity_anchor_ok("highest carpet area",
                                'SELECT base_conversion_factor FROM assets_carpetareaunit '
                                'ORDER BY base_conversion_factor DESC', SM)
    assert ok1 is True and ok2 is True


def test_alignment_ok_combined():
    _on()
    try:
        ok, why = A.alignment_ok("how many leads per month",
                                 'SELECT lead_stage, COUNT(*) FROM leads_lead GROUP BY lead_stage', SM)
        assert ok is False and why       # combined entry catches the temporal violation
    finally:
        _off()


# ── C. DIMENSION referent alignment (increment 2) ──────────────────────────────────────────────────
DSM = {"columns": {
    "leads_lead.furnishing_status": _col("leads_lead", "CATEGORY", "DIMENSION"),
    "leads_lead.loe_status": _col("leads_lead", "CATEGORY", "DIMENSION"),
    "assets_asset.city_name": _col("assets_asset", "CATEGORY", "DIMENSION"),
    "assets_asset.country": _col("assets_asset", "CATEGORY", "DIMENSION"),
}}


def _dim_on():
    _config.INTENT_SQL_DIMENSION_ALIGNMENT_ENABLED = True


def _dim_off():
    _config.INTENT_SQL_DIMENSION_ALIGNMENT_ENABLED = False


def test_dim_single_candidate_aligned():
    _dim_on()
    try:
        out, _ = A.dimension_alignment("how many assets by city",
                                       'SELECT city_name, COUNT(*) FROM assets_asset GROUP BY city_name', DSM)
        assert out == A.DIM_ALIGNED
    finally:
        _dim_off()


def test_dim_single_candidate_unrelated_refuses():
    _dim_on()
    try:
        out, _ = A.dimension_alignment("how many assets by city",
                                       'SELECT country, COUNT(*) FROM assets_asset GROUP BY country', DSM)
        assert out == A.DIM_REFUSE               # grouped by country, not the city candidate
    finally:
        _dim_off()


def test_dim_multiple_candidates_clarify():
    _dim_on()
    try:
        out, why = A.dimension_alignment(
            "how many leads by status",
            'SELECT furnishing_status, loe_status, COUNT(*) FROM leads_lead '
            'GROUP BY furnishing_status, loe_status', DSM)
        assert out == A.DIM_CLARIFY and "furnishing_status" in why and "loe_status" in why
    finally:
        _dim_off()


def test_dim_multiple_candidates_outside_refuses():
    _dim_on()
    try:
        # candidates {furnishing_status, loe_status} but SQL groups by city_name (outside) → refuse.
        out, _ = A.dimension_alignment(
            "how many leads by status",
            'SELECT city_name, COUNT(*) FROM leads_lead GROUP BY city_name', DSM)
        assert out == A.DIM_REFUSE
    finally:
        _dim_off()


def test_dim_explicit_phrase_disambiguates():
    _dim_on()
    try:
        # "by furnishing status" → phrase {furnishing, status} → only furnishing_status → ALIGNED.
        out, _ = A.dimension_alignment(
            "how many leads by furnishing status",
            'SELECT furnishing_status, COUNT(*) FROM leads_lead GROUP BY furnishing_status', DSM)
        assert out == A.DIM_ALIGNED
    finally:
        _dim_off()


def test_dim_empty_candidate_set_not_applicable():
    _dim_on()
    try:
        # "by region" — no column name-matches "region" → decline (never refuse).
        out, _ = A.dimension_alignment("how many assets by region",
                                       'SELECT country, COUNT(*) FROM assets_asset GROUP BY country', DSM)
        assert out == A.DIM_NOT_APPLICABLE
    finally:
        _dim_off()


def test_dim_no_group_by_not_applicable():
    _dim_on()
    try:
        out, _ = A.dimension_alignment("how many assets by city",
                                       'SELECT COUNT(*) FROM assets_asset', DSM)
        assert out == A.DIM_NOT_APPLICABLE       # no GROUP BY → grouped-shape guard's concern, not this
    finally:
        _dim_off()


def test_dim_flag_off_not_applicable():
    _dim_off()
    out, _ = A.dimension_alignment("how many leads by status",
                                   'SELECT city_name FROM leads_lead GROUP BY city_name', DSM)
    assert out == A.DIM_NOT_APPLICABLE           # flag off → never judges


# ── Aggregate-OMISSION guard (Increment 3A) ─────────────────────────────────────────────────────────
def _agg_on():
    _config.INTENT_SQL_AGG_PRESENCE_ENABLED = True


def _agg_off():
    _config.INTENT_SQL_AGG_PRESENCE_ENABLED = False


def test_agg_omission_blocks_projection():
    _agg_on()
    try:
        for q, sql in [("how many projects", "SELECT project_name FROM projects LIMIT 100"),
                       ("total rent", "SELECT rent FROM assets_leasetransaction LIMIT 100"),
                       ("average rent", "SELECT rent FROM assets_leasetransaction LIMIT 100"),
                       ("sum of monthly rent", "SELECT expected_monthly_rent FROM assets_leaselisting")]:
            ok, _ = A.aggregate_presence_ok(q, sql)
            assert ok is False, q                        # aggregate intent + no aggregate → refuse
    finally:
        _agg_off()


def test_agg_present_allows():
    _agg_on()
    try:
        for q, sql in [("how many projects", "SELECT COUNT(*) FROM projects"),
                       ("total rent", "SELECT SUM(rent) FROM assets_leasetransaction"),
                       ("average rent by city",
                        "SELECT city_name, AVG(rent) FROM assets_leasetransaction GROUP BY city_name")]:
            ok, _ = A.aggregate_presence_ok(q, sql)
            assert ok is True, q                          # SQL has an aggregate → allow
    finally:
        _agg_off()


def test_agg_for_each_not_over_refused():
    # the adversarial false-positives: "for each"/"of each" = per-entity attribute listing, a legit projection.
    _agg_on()
    try:
        assert A.aggregate_presence_ok("show number of bedrooms for each property",
                                       "SELECT project_name, bedrooms FROM assets_asset")[0] is True
        assert A.aggregate_presence_ok("what is the total area of each property",
                                       "SELECT project_name, total_area FROM assets_asset")[0] is True
    finally:
        _agg_off()


def test_agg_no_intent_untouched():
    _agg_on()
    try:
        assert A.aggregate_presence_ok("list all amenities",
                                       "SELECT amenity_name FROM catalog")[0] is True
        # underscore column name is space-bounded → not matched as intent
        assert A.aggregate_presence_ok("show total_area for properties",
                                       "SELECT total_area FROM assets_asset")[0] is True
    finally:
        _agg_off()


def test_agg_flag_off_always_ok():
    _agg_off()
    assert A.aggregate_presence_ok("how many projects",
                                   "SELECT project_name FROM projects LIMIT 100")[0] is True


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
