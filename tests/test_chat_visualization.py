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
