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


# ---------------------------------------------------------------------------
# Long-tail bucketing: the label must not collide with a real category, the cap
# must not fire on a readable result, and the bucket is a SUM — so it only
# exists for a measure that can be summed.
# Regression: a 12-category ticket breakdown charted a real category "Others"
# (6) beside a synthetic bucket "Other" (6) — two near-identical bars with the
# same value, three named categories hidden behind one of them.
# ---------------------------------------------------------------------------

_TICKET_COLS = ["category_name", "activity_count"]
_TICKET_ROWS = [[n, v] for n, v in
                [("Electrical Fittings", 22), ("Modular Kitchen", 14), ("Carpentry", 8),
                 ("Others", 6), ("Painting", 5), ("Deep Cleaning", 4), ("Seepage", 4),
                 ("Consumer Appliances", 2), ("Fixtures and Fittings", 2),
                 ("Renovation and Rectification", 2), ("Structural damage", 2),
                 ("Furniture", 2)]]
_COUNT_ANALYTICS = {"measure_aggregates": {"activity_count": "COUNT"}}


def test_twelve_categories_are_all_charted_not_bucketed():
    r = _recommender()
    spec = r.build_category_specs(_TICKET_COLS, _TICKET_ROWS, 0, 1, _COUNT_ANALYTICS)[0]
    labels = spec.chart_data["labels"]
    assert len(labels) == 12
    assert labels.count("Others") == 1 and "Other" not in labels
    assert sum(spec.chart_data["values"]) == sum(r[1] for r in _TICKET_ROWS)


def test_tail_bucket_never_collides_with_a_real_category_name():
    """A real catch-all category called "Others" must not be shadowed by an "Other"
    bucket — singular/plural counts as the same name."""
    r = _recommender()
    rows = [[f"C{i}", 30 - i] for i in range(24)] + [["Others", 99]]
    spec = r.build_category_specs(["cat", "n_count"], rows, 0, 1,
                                  {"measure_aggregates": {"n_count": "COUNT"}})[0]
    labels = spec.chart_data["labels"]
    assert len(labels) == 20                       # 19 + one bucket
    assert labels[-1] == "All other categories"
    assert "Others" in labels                      # the real category survives
    assert sum(spec.chart_data["values"]) == sum(r[1] for r in rows)   # nothing lost


def test_non_additive_measure_drops_the_tail_instead_of_summing_it():
    r = _recommender()
    rows = [[f"C{i}", 30 - i] for i in range(25)]
    spec = r.build_category_specs(["cat", "avg_rating"], rows, 0, 1,
                                  {"measure_aggregates": {"avg_rating": "AVG"}})[0]
    assert len(spec.chart_data["labels"]) == 19
    assert not any(l.startswith("Other") or l.startswith("All other")
                   for l in spec.chart_data["labels"])
    assert spec.title.endswith("(top 19)")         # the chart says it is a subset


def test_non_additive_measure_with_duplicate_rows_gets_no_chart():
    """Ungrouped rows + an average = no honest way to combine them. A SUM of the same
    shape still charts, exactly as before."""
    r = _recommender()
    rows = [["A", 4.0], ["A", 2.0], ["B", 5.0]]
    assert r.build_category_specs(["cat", "avg_score"], rows, 0, 1,
                                  {"measure_aggregates": {"avg_score": "AVG"}}) == []
    specs = r.build_category_specs(["cat", "total_amount"], rows, 0, 1,
                                   {"measure_aggregates": {"total_amount": "SUM"}})
    assert specs and specs[0].chart_data["slices"][0] == {"name": "A", "value": 6.0}


def test_additivity_falls_back_to_the_measure_name_without_analytics():
    """Federated results carry no measure_aggregates — the column's own name is then
    the signal, and anything unrecognised stays additive (today's behaviour)."""
    from apps.chat.visualization import _measure_is_additive
    assert _measure_is_additive("activity_count", None) is True
    assert _measure_is_additive("total_amount", None) is True
    assert _measure_is_additive("avg_rating", None) is False
    assert _measure_is_additive("completion_rate", None) is False
    assert _measure_is_additive("pct_overdue", None) is False
    # the engine's own aggregate outranks the name
    assert _measure_is_additive("avg_rating", {"measure_aggregates": {"avg_rating": "SUM"}}) is True
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
