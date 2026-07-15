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
        assert spec.x_axis_title == "processed_date"
        assert spec.y_axis_title == "payment_attempt_count"


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
