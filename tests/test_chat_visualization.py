"""Tests for apps/chat/visualization.py's VisualizationRecommender — pure
dataclasses/enum, no Django settings needed. Regression coverage for the
production bug where an identifier column (id/asset_id) got charted as a
measure/dimension (e.g. a line_histogram plotting `id` against
`payment_attempt_count` with "None" labels)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _recommender():
    from apps.chat.visualization import VisualizationRecommender
    return VisualizationRecommender()


def test_no_chart_for_empty_input():
    r = _recommender()
    assert r.recommend([], []) == []
    assert r.recommend(["a"], []) == []


def test_identifier_columns_never_selected_as_axes():
    """Regression: a query with id/asset_id (identifier, numeric-looking) and
    payment_attempt_count (a real measure) plus processed_date (temporal)
    must chart processed_date/payment_attempt_count — never id vs anything.
    Temporal+numeric now yields [line, bar] (multi-viz) — line first (the
    original single-chart choice), never id/asset_id on either one."""
    r = _recommender()
    cols = ["id", "asset_id", "payment_attempt_count", "processed_date"]
    rows = [
        [1, 10, 0, "2021-01-01"],
        [2, 10, 1, "2021-01-02"],
        [3, 11, 0, "2021-01-03"],
    ]
    specs = r.recommend(cols, rows)
    assert len(specs) == 2
    assert specs[0].type.value == "line"
    assert specs[1].type.value == "bar"
    for spec in specs:
        assert spec.x_axis_title == "Processed Date"
        assert spec.y_axis_title == "Payment Attempt Count"


def test_identifier_only_numeric_columns_produce_no_chart():
    r = _recommender()
    cols = ["id", "customer_id", "order_id"]
    rows = [[1, 100, 5000], [2, 101, 5001], [3, 102, 5002]]
    assert r.recommend(cols, rows) == []


def test_category_numeric_pie_small_cardinality():
    """Small category count now yields [pie, bar] (multi-viz) — pie first,
    preserving today's single-chart choice for any caller that only reads
    specs[0]; bar is the new additive second chart, built from the SAME
    totals (same confidence, not a separately-justified guess)."""
    r = _recommender()
    cols = ["region", "revenue"]
    rows = [["west", 100], ["east", 200], ["north", 150]]
    specs = r.recommend(cols, rows)
    assert len(specs) == 2
    assert specs[0].type.value == "pie"
    assert specs[0].confidence >= 0.6
    assert specs[1].type.value == "bar"
    assert specs[1].confidence == specs[0].confidence
    assert specs[1].chart_data["labels"] == ["west", "east", "north"]
    assert specs[1].chart_data["values"] == [100, 200, 150]


def test_single_category_never_forces_a_chart():
    r = _recommender()
    cols = ["region", "revenue"]
    rows = [["west", 100], ["west", 200]]  # same category, totals to ONE slice
    assert r.recommend(cols, rows) == []


def test_temporal_numeric_line_chart():
    """Temporal+numeric now yields [line, bar] (multi-viz) — line first,
    preserving today's single-chart choice; bar is the new additive second
    chart, built from the SAME ordered (labels, values) (same confidence)."""
    r = _recommender()
    cols = ["month", "total"]
    rows = [["2026-01", 100], ["2026-02", 200], ["2026-03", 150]]
    specs = r.recommend(cols, rows)
    assert len(specs) == 2
    assert specs[0].type.value == "line"
    assert specs[0].confidence == 0.9
    assert specs[1].type.value == "bar"
    assert specs[1].confidence == 0.9
    assert specs[1].chart_data == specs[0].chart_data


def test_every_spec_carries_confidence_in_to_dict():
    r = _recommender()
    cols = ["region", "revenue"]
    rows = [["west", 100], ["east", 200]]
    specs = r.recommend(cols, rows)
    d = specs[0].to_dict()
    assert "confidence" in d
    assert 0.0 <= d["confidence"] <= 1.0


def test_confidence_threshold_drops_low_confidence_chart(monkeypatch):
    """A high-cardinality category breakdown that overflows into a bar chart
    (confidence 0.7, since it's a long-tail-bucketed fallback) is still
    returned at the default threshold, but dropped entirely when the
    threshold is raised above it — the gate genuinely suppresses low-
    confidence charts rather than always returning something."""
    import apps.chat.visualization as viz_mod
    r = _recommender()
    cols = ["category", "amount"]
    rows = [[f"cat{i}", i * 10] for i in range(20)]   # 20 distinct categories

    specs_default = r.recommend(cols, rows)
    assert len(specs_default) == 1
    assert specs_default[0].confidence == 0.7

    monkeypatch.setattr(viz_mod, "_CONFIDENCE_THRESHOLD", 0.8)
    assert r.recommend(cols, rows) == []


# ---------------------------------------------------------------------------
# Multi-visualization support (2026-07 architecture review): a result that
# naturally supports more than one EQUALLY VALID rendering of the SAME data
# now returns all of them, not just one. Never "synthesizes" unrelated
# charts — every additional spec reuses the exact (labels, values)/(slices)
# data and confidence the primary spec already computed.
# ---------------------------------------------------------------------------

def test_many_categories_stays_a_single_bar_chart():
    """The long-tail (>6 categories) case deliberately stays single-chart —
    a pie with this many slices is unreadable, so only bar is returned, same
    as before this feature (see the architecture review: "many categories ->
    bar only")."""
    r = _recommender()
    cols = ["category", "amount"]
    rows = [[f"cat{i}", i * 10] for i in range(20)]
    specs = r.recommend(cols, rows)
    assert len(specs) == 1
    assert specs[0].type.value == "bar"


def test_multi_viz_specs_are_independently_confidence_gated():
    """Even though pie/bar (or line/bar) share a confidence value today, the
    filter is applied per-spec, not as an all-or-nothing pair — verified by
    raising the threshold above the small-category pie/bar confidence (0.9)
    and confirming BOTH are dropped together (not just one silently kept)."""
    import apps.chat.visualization as viz_mod
    r = _recommender()
    cols = ["region", "revenue"]
    rows = [["west", 100], ["east", 200], ["north", 150]]

    monkeypatch_value = viz_mod._CONFIDENCE_THRESHOLD
    try:
        viz_mod._CONFIDENCE_THRESHOLD = 0.95
        assert r.recommend(cols, rows) == []
    finally:
        viz_mod._CONFIDENCE_THRESHOLD = monkeypatch_value


def test_multi_viz_does_not_apply_to_the_dual_measure_combo_chart():
    """A dimension + TWO measures (line_histogram) is inherently a specific,
    different shape from bar/pie/line — it stays single-chart on purpose
    (see the architecture review: this shape isn't naturally expressible as
    a second bar/pie/line without guessing which single measure to plot)."""
    r = _recommender()
    cols = ["month", "sales_volume", "conversion_rate"]
    rows = [["2026-01", 100, 0.5], ["2026-02", 200, 0.6], ["2026-03", 150, 0.55]]
    specs = r.recommend(cols, rows)
    assert len(specs) == 1
    assert specs[0].type.value == "line_histogram"


# ---------------------------------------------------------------------------
# Free-text column exclusion — regression: a free-text column (e.g. `notes`)
# structurally reads as "categorical" (non-numeric, non-date strings), so
# without this check it could outrank a real dimension like `label` for the
# chart's category axis purely because it happened to come first in `cols`.
# ---------------------------------------------------------------------------

def test_free_text_column_never_outranks_a_real_category_dimension():
    r = _recommender()
    cols = ["notes", "label", "amount"]
    rows = [
        ["Customer called to follow up on the overdue invoice and payment plan", "west", 100],
        ["Escalated to collections after repeated missed payment reminders", "east", 200],
        ["Resolved after partial payment was received and plan restructured", "north", 150],
    ]
    specs = r.recommend(cols, rows)
    assert specs, "expected a chart using 'label' as the dimension"
    for spec in specs:
        assert spec.title == "Amount by Label"


def test_free_text_name_hint_excludes_short_values_too():
    """Even when sampled values are short, a name hint (e.g. `email`) alone
    is enough to exclude a column from ever becoming the chart dimension."""
    r = _recommender()
    cols = ["email", "region", "revenue"]
    rows = [["a@x.com", "west", 100], ["b@x.com", "east", 200], ["c@x.com", "north", 150]]
    specs = r.recommend(cols, rows)
    assert specs
    for spec in specs:
        assert spec.title == "Revenue by Region"


def test_free_text_only_columns_produce_no_chart():
    r = _recommender()
    cols = ["notes", "description"]
    rows = [
        ["Customer called to follow up on the overdue invoice and payment plan", "Long form detail one here"],
        ["Escalated to collections after repeated missed payment reminders", "Long form detail two here"],
    ]
    assert r.recommend(cols, rows) == []


# ---------------------------------------------------------------------------
# Negative-value pie guard (2026-07-17) — a pie slice can't represent a
# negative share of a whole (profit/loss, net-change, refund data); bar
# handles negative fine, pie must never be offered when any total is negative.
# ---------------------------------------------------------------------------

def test_negative_values_skip_pie_small_category_count():
    r = _recommender()
    cols = ["category", "net_change"]
    rows = [["A", 500], ["B", -200], ["C", 100]]
    specs = r.recommend(cols, rows)
    assert specs
    assert all(s.type.value != "pie" for s in specs)
    assert any(s.type.value == "bar" for s in specs)


def test_negative_values_skip_pie_long_tail():
    """Same guard in the >MAX_PIE_SLICES branch (top-N + 'Other')."""
    r = _recommender()
    cols = ["category", "net_change"]
    rows = [[f"cat{i}", 100 - i * 20] for i in range(12)]   # last few go negative
    specs = r.recommend(cols, rows)
    assert specs
    assert all(s.type.value != "pie" for s in specs)


def test_all_positive_small_category_count_still_gets_pie():
    """Sanity: the guard only fires on an actual negative value — an
    all-positive result is unaffected (regression guard for the fix itself)."""
    r = _recommender()
    cols = ["category", "amount"]
    rows = [["A", 500], ["B", 200], ["C", 100]]
    specs = r.recommend(cols, rows)
    assert any(s.type.value == "pie" for s in specs)


# ---------------------------------------------------------------------------
# Humanized titles/axis labels (2026-07-17) — consistency with the table's
# own header humanization (apps/chat/table_rendering.py's fmt_header).
# ---------------------------------------------------------------------------

def test_line_chart_axis_titles_humanized():
    r = _recommender()
    cols = ["order_date", "total_revenue"]
    rows = [["2026-01-01", 100], ["2026-01-02", 200], ["2026-01-03", 150]]
    specs = r.recommend(cols, rows)
    assert specs
    for spec in specs:
        assert spec.x_axis_title == "Order Date"
        assert spec.y_axis_title == "Total Revenue"


# ---------------------------------------------------------------------------
# camelCase identifier detection (2026-07-17) — real gap: a non-Django source
# (NoSQL/federated/external schema) commonly names ids "AccountID"/"customerId"/
# "buildId" with no underscore. "accountid".endswith("_id") is False, so these
# slipped through and got charted as a category/measure ("ids in the label and
# value" bug report).
# ---------------------------------------------------------------------------

def test_camelcase_identifier_excluded_from_chart():
    r = _recommender()
    cols = ["AccountID", "region", "revenue"]
    rows = [["acc-1", "west", 100], ["acc-2", "east", 200], ["acc-3", "north", 150]]
    specs = r.recommend(cols, rows)
    assert specs
    for spec in specs:
        assert "AccountID" not in (spec.title or "")
        assert spec.x_axis_title != "Accountid" and spec.x_axis_title != "AccountID"


def test_camelcase_identifier_variants_detected():
    r = _recommender()
    for name in ("AccountID", "CustomerId", "orderID", "buildId"):
        assert r._is_identifier(name), f"{name!r} should be detected as an identifier"


def test_lowercase_english_words_ending_in_id_not_flagged():
    """Regression guard for the fix itself: ordinary lowercase words that
    happen to end in 'id' must never be treated as identifiers."""
    r = _recommender()
    for word in ("paid", "valid", "invalid", "grid", "hybrid", "android",
                "void", "avoid", "solid", "rapid", "fluid", "arid", "acid"):
        assert not r._is_identifier(word), f"{word!r} must NOT be flagged as an identifier"


# ---------------------------------------------------------------------------
# RANKING rescue (2026-07-19): id-labelled bar for engine-classified rankings
# ---------------------------------------------------------------------------

def test_ranking_rescue_charts_id_labelled_leaderboard():
    """An engine-classified RANKING with a measure but only identifier labels
    (id/reference-heavy schemas) now gets a bar leaderboard instead of nothing."""
    from apps.chat.visualization import VisualizationRecommender
    cols = ["payment_reference_number", "paid_amount"]
    rows = [[f"order_R{i}", 100000 - i * 1000] for i in range(10)]
    analytics = {"result_shape": "RANKING", "column_stats": [
        {"name": "payment_reference_number", "kind": "categorical", "role": "identifier"},
        {"name": "paid_amount", "kind": "numeric", "role": "measure"},
    ]}
    specs = VisualizationRecommender().recommend(cols, rows, analytics=analytics)
    assert specs and specs[0].type.value == "bar"
    d = specs[0].to_dict()
    assert d["chart_data"]["labels"][0] == "order_R0"
    assert d["chart_data"]["values"][0] == 100000


def test_ranking_rescue_needs_engine_shape():
    """Without the engine's own RANKING classification the identifier exclusion
    stands unchanged — same data, no shape → no chart (pinned behaviour)."""
    from apps.chat.visualization import VisualizationRecommender
    cols = ["payment_reference_number", "paid_amount"]
    rows = [[f"order_R{i}", 100000 - i * 1000] for i in range(10)]
    analytics = {"column_stats": [
        {"name": "payment_reference_number", "kind": "categorical", "role": "identifier"},
        {"name": "paid_amount", "kind": "numeric", "role": "measure"},
    ]}
    assert VisualizationRecommender().recommend(cols, rows, analytics=analytics) == []


def test_ranking_rescue_prefers_non_identifier_label():
    """When a real name column exists alongside the id, the rescue labels by the
    name, not the id."""
    from apps.chat.visualization import VisualizationRecommender
    cols = ["payment_id", "payer_note", "paid_amount"]
    rows = [[f"{i}", f"note-{i}", 500 - i] for i in range(5)]
    analytics = {"result_shape": "RANKING", "column_stats": [
        {"name": "payment_id", "kind": "categorical", "role": "identifier"},
        {"name": "payer_note", "kind": "categorical", "role": "text"},
        {"name": "paid_amount", "kind": "numeric", "role": "measure"},
    ]}
    specs = VisualizationRecommender().recommend(cols, rows, analytics=analytics)
    assert specs and specs[0].to_dict()["chart_data"]["labels"][0] == "note-0"


# --- listing charts: row-preserving, never aggregated (2026-09-11) ----------------
#
# Production bug: "Show me the cheapest properties currently on the market for sale"
# returned a correct table but a chart that (1) summed rows sharing a building name,
# (2) re-sorted by value DESC and swept the three cheapest rows into an "Other"
# bucket, and (3) picked its axes positionally, charting a column the user never saw
# in the table. A RANKING/DETAIL_TABLE is a listing, not a breakdown — one bar per
# row, in the result's own order.

_CHEAPEST_COLS = ["asset_name", "expected_price", "market_status"]
_CHEAPEST_ROWS = [
    ["Shri sai reality", 4, "For Sale"],
    ["Hanuman Road 150", 150, "For Sale"],
    ["Information Technology Park", 465, "For Sale"],
    ["Sumangal Vihar", 500, "For Sale"],
    ["Information Technology Park", 545, "For Sale"],   # same building, a SECOND listing
    ["Copy Quick", 1242, "For Sale"],
    ["Alpha", 2000, "For Sale"], ["Beta", 3000, "For Sale"],
    ["Gamma", 4000, "For Sale"], ["Delta", 5000, "For Sale"],
    ["Epsilon", 6000, "For Sale"], ["Zeta", 7000, "For Sale"],
]
_CHEAPEST_ANALYTICS = {
    "result_shape": "RANKING",
    "orderings": [["expected_price", True]],
    "column_stats": [{"name": "asset_name", "kind": "categorical", "role": "dimension"},
                     {"name": "expected_price", "kind": "numeric", "role": "measure"},
                     {"name": "market_status", "kind": "categorical", "role": "dimension"}],
}


def test_listing_preserves_result_order_and_never_buckets_other():
    specs = _recommender().recommend(_CHEAPEST_COLS, _CHEAPEST_ROWS, _CHEAPEST_ANALYTICS)
    assert len(specs) == 1 and specs[0].type.value == "bar"
    data = specs[0].chart_data
    assert "Other" not in data["labels"]                      # cheapest rows stay visible
    assert data["values"] == [r[1] for r in _CHEAPEST_ROWS]   # cheapest-first, not re-sorted


def test_listing_never_sums_rows_sharing_a_label():
    data = _recommender().recommend(
        _CHEAPEST_COLS, _CHEAPEST_ROWS, _CHEAPEST_ANALYTICS)[0].chart_data
    assert 465 in data["values"] and 545 in data["values"]    # two listings, two bars
    assert 1010 not in data["values"]                         # never the merged total
    assert len(data["labels"]) == len(set(data["labels"]))    # both stay distinguishable


def test_listing_charts_the_ordered_measure_and_the_identifying_label():
    """Axes come from the SQL's own ORDER BY + the most row-distinct label column —
    not from whatever the SELECT list happens to start with."""
    cols = ["sale_listing_id", "expected_price", "booking_amount_grace_period",
            "cancellation_reason_title", "building_name"]
    rows = [[i, 100 * (i + 1), 7, "Duplicate" if i % 2 else "Other", f"Bldg{i}"]
            for i in range(6)]
    spec = _recommender().recommend(
        cols, rows, {"result_shape": "DETAIL_TABLE",
                     "orderings": [["expected_price", True]]})[0]
    assert spec.x_axis_title == "Building Name"      # not Cancellation Reason Title
    assert spec.y_axis_title == "Expected Price"     # not the grace period
    assert spec.chart_data["values"] == [100, 200, 300, 400, 500, 600]


def test_listing_truncates_the_tail_instead_of_bucketing_it():
    cols = ["name", "price"]
    rows = [[f"P{i}", i] for i in range(40)]
    spec = _recommender().recommend(
        cols, rows, {"result_shape": "RANKING", "orderings": [["price", True]]})[0]
    assert spec.chart_data["values"] == list(range(25))   # the cheapest 25, in order
    assert "Other" not in spec.chart_data["labels"]
    assert "of 40 rows" in (spec.sub_title or "")         # truncation is disclosed


def test_grouped_breakdown_still_aggregates_unchanged():
    """The listing path must not change GROUPED/DISTRIBUTION, where summing rows
    per category and bucketing a long tail into 'Other' is the correct behavior."""
    specs = _recommender().recommend(
        _CHEAPEST_COLS, _CHEAPEST_ROWS, {"result_shape": "GROUPED"})
    labels = specs[0].chart_data["labels"]
    assert "Other" in labels                              # long-tail bucketing intact
    assert 1010 in specs[0].chart_data["values"]          # per-category totals intact


# --- payload-safety invariants across degenerate inputs (2026-09-11) --------------
#
# From an adversarial sweep of the recommender. These are not about which chart is
# chosen — they are the invariants that must hold for EVERY chart it ever emits,
# because apps/chat/services.py::_build_visualizations has no try/except around
# this call: anything raised here takes down the whole chat turn, not just the chart.

def test_nan_and_infinity_never_reach_chart_data():
    """json.dumps writes NaN/Infinity as bare tokens, which are not valid JSON — a
    strict frontend JSON.parse rejects the entire response. A NaN measure must be
    treated exactly like a NULL one (row skipped), on every chart path."""
    import json
    import math
    r = _recommender()
    rows = [["a", float("nan")], ["b", float("inf")], ["c", 1.0], ["d", 2.0]]
    for shape in ("RANKING", "DETAIL_TABLE", "GROUPED", "DISTRIBUTION"):
        for spec in r.recommend(["name", "price"], rows,
                                {"result_shape": shape, "orderings": [["price", False]]}):
            payload = spec.to_dict()
            json.dumps(payload, allow_nan=False)        # raises if NaN/Inf slipped in
            values = (payload["chart_data"].get("values")
                      or [s["value"] for s in payload["chart_data"].get("slices", [])])
            assert all(math.isfinite(v) for v in values)


def test_a_row_shorter_than_cols_is_dropped_not_raised():
    """Every builder indexes rows positionally against cols; one malformed row used
    to raise IndexError out of the recommender and fail the turn."""
    specs = _recommender().recommend(
        ["name", "price"], [["a", 1], ["b"], ["c", 3]],
        {"result_shape": "RANKING", "orderings": [["price", False]]})
    assert specs and specs[0].chart_data["values"] == [1, 3]


def test_listing_labels_stay_unique_even_against_a_preexisting_suffix():
    """The de-duplication suffix can itself collide with a label already in the data
    ("X", "X (2)", "X") — each row must still get its own distinct bar."""
    rows = [["X", 1], ["X (2)", 2], ["X", 3]]
    labels = _recommender().recommend(
        ["name", "price"], rows,
        {"result_shape": "RANKING", "orderings": [["price", False]]})[0].chart_data["labels"]
    assert len(labels) == len(set(labels)) == 3


def test_listing_label_prefers_the_identifying_column_over_a_repeating_one():
    """A status/flag column repeats across a listing; the name column identifies each
    row. Distinctness decides, so SELECT-list position can no longer win."""
    rows = [["For Sale", f"Bldg{i}", i] for i in range(6)]
    spec = _recommender().recommend(
        ["status", "building_name", "price"], rows,
        {"result_shape": "RANKING", "orderings": [["price", False]]})[0]
    assert spec.x_axis_title == "Building Name"


def test_listing_never_charts_an_identifier_or_text_column_as_the_measure():
    """Even when the SQL's ORDER BY names one — an id is not a quantity."""
    r = _recommender()
    spec = r.recommend(["asset_id", "name", "price"], [[9, "a", 1], [8, "b", 2]],
                       {"result_shape": "RANKING", "orderings": [["asset_id", False]]})[0]
    assert spec.y_axis_title == "Price"


# --- silent truncation: a page must never be charted as the whole (2026-09-22) ---
#
# assets_asset holds 7,814 rows; a question naming no row count comes back with
# 1,000. The listing sub_title used to read "First 25 of 1000 rows", which tells
# the reader 1,000 IS the total. The engine now flags the silent cut as
# analytics["result_truncated"]; a user-requested "top 5" is a complete answer
# and is excluded by the producer, so these tests only pin how the flag is honoured.

_PAGE_ANALYTICS = {"result_shape": "RANKING", "orderings": [["price", False]],
                   "result_truncated": True}


def _captions(specs):
    return [s.sub_title for s in specs]


def test_truncated_listing_does_not_present_the_page_as_the_total():
    cols = ["name", "price"]
    rows = [[f"P{i}", i] for i in range(1000)]
    spec = _recommender().recommend(cols, rows, _PAGE_ANALYTICS)[0]
    assert spec.sub_title == ("Partial data — first 25 of the 1,000 rows returned; "
                              "the full result is larger")
    # the old wording is what implied a total — it must be gone, not merely amended
    assert "of 1000 rows" not in spec.sub_title
    assert spec.chart_data["values"] == list(range(25))   # data itself unchanged


def test_untruncated_listing_keeps_the_exact_old_caption():
    """Pinned: without the flag, the bar cap is our own and "First N of M rows"
    is then completely true — M really is the row count."""
    cols = ["name", "price"]
    rows = [[f"P{i}", i] for i in range(40)]
    spec = _recommender().recommend(
        cols, rows, {"result_shape": "RANKING", "orderings": [["price", False]]})[0]
    assert spec.sub_title == "First 25 of 40 rows, in result order"


def test_truncated_listing_shorter_than_the_bar_cap_is_still_captioned():
    """Only 6 rows on screen, but they are 6 of a larger result — the chart is
    just as partial as a capped one."""
    rows = [[f"P{i}", i] for i in range(6)]
    spec = _recommender().recommend(["name", "price"], rows, _PAGE_ANALYTICS)[0]
    assert spec.sub_title == "Partial data — the 6 rows returned; the full result is larger"


def test_truncated_category_breakdown_drops_the_pie_and_captions_the_bar():
    """A pie's slices always sum to 100% — no caption undoes that claim, so on a
    truncated result the part-of-whole chart is suppressed and the bar (which
    asserts no denominator) carries the disclosure instead."""
    cols = ["region", "revenue"]
    rows = [["west", 100], ["east", 200], ["north", 150]]
    specs = _recommender().recommend(cols, rows, {"result_truncated": True})
    assert [s.type.value for s in specs] == ["bar"]
    assert all("Partial data" in (c or "") for c in _captions(specs))


def test_truncated_long_tail_breakdown_also_drops_the_pie():
    """The top-N + 'Other' branch can emit a pie too (<= 6 slices) — same guard."""
    cols = ["category", "amount"]
    rows = [[f"cat{i}", 10] for i in range(12)]
    specs = _recommender().recommend(cols, rows, {"result_truncated": True})
    assert specs and all(s.type.value != "pie" for s in specs)
    assert all("Partial data" in (s.sub_title or "") for s in specs)


def test_untruncated_category_breakdown_keeps_pie_and_has_no_caption():
    cols = ["region", "revenue"]
    rows = [["west", 100], ["east", 200], ["north", 150]]
    for analytics in (None, {}, {"result_truncated": False}):
        specs = _recommender().recommend(cols, rows, analytics)
        assert [s.type.value for s in specs] == ["pie", "bar"], analytics
        assert _captions(specs) == [None, None], analytics


def test_truncated_time_series_captions_every_spec_it_returns():
    """line + bar are two renderings of the same page — one captioned chart and
    one bare one beside it would be worse than none."""
    cols = ["month", "total"]
    rows = [["2026-01", 100], ["2026-02", 200], ["2026-03", 150]]
    specs = _recommender().recommend(cols, rows, {"result_truncated": True})
    assert [s.type.value for s in specs] == ["line", "bar"]
    assert all(s.sub_title == "Partial data — the 3 rows returned; the full result is larger"
               for s in specs)


def test_truncated_combo_chart_is_captioned():
    cols = ["month", "sales_volume", "conversion_rate"]
    rows = [["2026-01", 100, 0.5], ["2026-02", 200, 0.6], ["2026-03", 150, 0.55]]
    specs = _recommender().recommend(cols, rows, {"result_truncated": True})
    assert specs[0].type.value == "line_histogram"
    assert "Partial data" in (specs[0].sub_title or "")


def test_truncated_ranking_rescue_is_captioned():
    """The id-labelled leaderboard is its own return path in recommend() — the
    disclosure is applied at one choke point so it can't be missed there."""
    cols = ["payment_reference_number", "paid_amount"]
    rows = [[f"order_R{i}", 100000 - i * 1000] for i in range(10)]
    analytics = {"result_shape": "RANKING", "result_truncated": True, "column_stats": [
        {"name": "payment_reference_number", "kind": "categorical", "role": "identifier"},
        {"name": "paid_amount", "kind": "numeric", "role": "measure"},
    ]}
    specs = _recommender().recommend(cols, rows, analytics=analytics)
    assert specs and specs[0].type.value == "bar"
    assert "Partial data" in (specs[0].sub_title or "")


def test_truncation_caption_reaches_the_wire_payload():
    cols = ["month", "total"]
    rows = [["2026-01", 100], ["2026-02", 200], ["2026-03", 150]]
    d = _recommender().recommend(cols, rows, {"result_truncated": True})[0].to_dict()
    assert "Partial data" in d["sub_title"]


def test_a_missing_or_malformed_analytics_never_raises_or_captions():
    """Older payloads omit the key entirely, and other heads may ship something
    that isn't a dict at all. recommend() has no try/except around it in
    services._build_visualizations — a raise here fails the whole chat turn."""
    cols = ["month", "total"]
    rows = [["2026-01", 100], ["2026-02", 200], ["2026-03", 150]]
    for analytics in (None, {}, [], "truncated", 7, {"result_truncated": None},
                      {"column_stats": None}):
        specs = _recommender().recommend(cols, rows, analytics)
        assert specs, analytics
        assert _captions(specs) == [None] * len(specs), analytics


def test_public_builders_disclose_truncation_too():
    """services._spec_from_suggestion reaches these directly when the
    recommender itself found nothing — the query tier's suggested chart is built
    from the same page and needs the same caption (and the same pie guard)."""
    r = _recommender()
    cols = ["region", "revenue"]
    rows = [["west", 100], ["east", 200], ["north", 150]]
    specs = r.build_category_specs(cols, rows, 0, 1, analytics={"result_truncated": True})
    assert [s.type.value for s in specs] == ["bar"]
    assert "Partial data" in (specs[0].sub_title or "")

    line = r.build_line_spec(cols, rows, 0, 1, analytics={"result_truncated": True})
    assert "Partial data" in (line.sub_title or "")
    # default (no analytics) is byte-identical to the old behaviour
    assert r.build_line_spec(cols, rows, 0, 1).sub_title is None
    assert [s.type.value for s in r.build_category_specs(cols, rows, 0, 1)] == ["pie", "bar"]
