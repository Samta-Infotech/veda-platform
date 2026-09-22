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

import pytest  # noqa: E402

import config as _config  # noqa: E402
from veda import intent_sql_alignment as A  # noqa: E402

try:                      # the entity-coverage check resolves nouns through the planner's
    from veda.understanding.grounding import ground_entity as _ground_entity  # noqa: F401
    _GROUNDING_IMPORTABLE = True                     # own resolver, which pulls in veda.runtime
except Exception:                                    # (DB/embedding handles). A whole-suite run
    _GROUNDING_IMPORTABLE = False                    # without those deps can't exercise it at all —
                                                     # skip rather than assert a vacuous pass.
_needs_grounding = pytest.mark.skipif(
    not _GROUNDING_IMPORTABLE, reason="grounding stack (veda.runtime) not importable here")


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
        # Both candidates must still be NAMED, but in ordinary words: this string is
        # shown to the user verbatim as the reply, so it may not carry raw
        # identifiers (nor "column"/"table"/"SQL") — the same rule the
        # explainability projection enforces. Underscored names would leak the
        # schema's own spelling into a user-facing sentence.
        assert out == A.DIM_CLARIFY
        assert "furnishing status" in why and "loe status" in why, why
        assert "_" not in why, why
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


# ---------------------------------------------------------------------------
# C. ENTITY COVERAGE — the question named several entities, the SQL answers a
# subset. Never refuses; the caller reports it and caps the confidence.
# Regression: "audit of ticket updates, assignees and attachments by category"
# counted ticket updates alone and shipped at confidence 1.0 with
# "no requested filters were ignored" — the dropped entities aren't filters.
# ---------------------------------------------------------------------------

# Underscore-separated names: the concatenated Django-style names the real schema uses
# ("worklists_ticketattachment") are segmented by semantic/name_tokens against the ACTIVE
# scope's model, which a hermetic unit test has no business loading. The coverage logic
# under test is the same either way.
COVER_SM = {"tables": {
    "worklists_ticket": {"primary_entity": "A support ticket."},
    "worklists_ticket_update": {"primary_entity": "An update to a ticket."},
    "worklists_ticket_category": {"primary_entity": "A single category for tickets."},
    "worklists_ticket_attachment": {"primary_entity": "A single attachment for a ticket."},
    "assets_asset_attachment": {"primary_entity": "An attachment for an asset."},
}, "columns": {}}

COVER_SQL = (
    'SELECT "t2"."name" AS "category_name", COUNT("t0"."id") AS "activity_count" '
    'FROM "worklists_ticket_update" AS "t0" '
    'JOIN "worklists_ticket" AS "t1" ON "t1"."id" = "t0"."ticket_id" '
    'JOIN "worklists_ticket_category" AS "t2" ON "t2"."id" = "t1"."ticket_category_id" '
    'GROUP BY "t2"."name" ORDER BY "activity_count" DESC LIMIT 100'
)


def _neighbours(monkeypatch, tables, *, on=True):
    """Stand in for the ingested join-path artifact (no DB / no artifacts in tests), and
    set the flag on the guard itself. The flag is patched through `_coverage_enabled`
    rather than the config module because a whole-suite run imports the repo-ROOT
    `config/` package over veda_core's (see the EXPLAIN_TRACE_ENABLED failures in the
    same run), so assigning an attribute there reaches a different module object than
    the guard reads."""
    monkeypatch.setattr(A, "_coverage_enabled", lambda: on)
    monkeypatch.setattr(A, "_join_neighbourhood", lambda sql_tables: set(tables) - set(sql_tables))


@_needs_grounding
def test_entity_coverage_reports_an_entity_the_sql_never_reached(monkeypatch):
    _neighbours(monkeypatch, {"worklists_ticket_attachment"})
    ok, missing, terms = A.entity_coverage(
        "audit of ticket updates and attachments grouped by ticket category",
        COVER_SQL, COVER_SM)
    assert ok is False
    assert missing == ["worklists_ticket_attachment"]
    # the USER'S word for it — what the summariser must not claim to have measured
    assert terms == ["attachment"]


@_needs_grounding
def test_entity_coverage_silent_when_every_named_entity_is_in_the_sql(monkeypatch):
    _neighbours(monkeypatch, {"worklists_ticket_attachment"})
    assert A.entity_coverage("ticket updates grouped by ticket category",
                             COVER_SQL, COVER_SM) == (True, [], [])


@_needs_grounding
def test_entity_coverage_ignores_a_table_outside_the_join_neighbourhood(monkeypatch):
    """An unrelated same-named entity elsewhere in the schema is never reported —
    only what hangs directly off the tables being queried."""
    _neighbours(monkeypatch, set())            # nothing joinable
    assert A.entity_coverage("ticket updates and attachments by ticket category",
                             COVER_SQL, COVER_SM) == (True, [], [])


@_needs_grounding
def test_entity_coverage_needs_one_named_entity_actually_present(monkeypatch):
    """A SQL sharing NO entity with the question is a wrong-anchor problem, which the
    alignment guards above own — coverage must not also fire on it."""
    _neighbours(monkeypatch, {"worklists_ticket_attachment"})
    assert A.entity_coverage("attachments", COVER_SQL.replace(
        "worklists_ticket_update", "assets_asset_attachment"), COVER_SM)[0] is True


def test_entity_coverage_off_is_byte_identical(monkeypatch):
    _neighbours(monkeypatch, {"worklists_ticket_attachment"}, on=False)
    assert A.entity_coverage("ticket updates and attachments by category",
                             COVER_SQL, COVER_SM) == (True, [], [])


@_needs_grounding
def test_entity_coverage_resolves_a_noun_only_the_description_carries(monkeypatch):
    """"assignees" is nowhere in worklists_ticketuser's NAME — the table describes itself
    as "a single ticket assignment record", and assignee/assignment share a stem no
    substring test finds. A UNIQUE description match in the neighbourhood resolves it."""
    sm = {"tables": {**COVER_SM["tables"],
                     "worklists_ticket_user": {
                         "primary_entity": "A single ticket assignment record.",
                         "business_purpose": "Tracks ticket assignments."}},
          "columns": {}}
    _neighbours(monkeypatch, {"worklists_ticket_user"})
    ok, missing, terms = A.entity_coverage(
        "audit of ticket updates and assignees grouped by ticket category", COVER_SQL, sm)
    assert (ok, missing, terms) == (False, ["worklists_ticket_user"], ["assignee"])


@_needs_grounding
def test_description_match_must_be_unique_to_count(monkeypatch):
    """A word every neighbour's description carries ("ticket") names no single entity —
    ambiguity is ignored rather than guessed at, so it can never invent a coverage gap."""
    sm = {"tables": {**COVER_SM["tables"],
                     "worklists_ticket_review": {"primary_entity": "A ticket review record."},
                     "worklists_ticket_payment": {"primary_entity": "A ticket payment record."}},
          "columns": {}}
    _neighbours(monkeypatch, {"worklists_ticket_review", "worklists_ticket_payment"})
    assert A._description_referent("tickets", set(sm["tables"]), sm) is None
    # and a short word is never stemmed at all
    assert A._description_referent("pay", set(sm["tables"]), sm) is None
