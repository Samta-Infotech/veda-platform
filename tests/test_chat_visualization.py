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
    must chart processed_date/payment_attempt_count — never id vs anything."""
    r = _recommender()
    cols = ["id", "asset_id", "payment_attempt_count", "processed_date"]
    rows = [
        [1, 10, 0, "2021-01-01"],
        [2, 10, 1, "2021-01-02"],
        [3, 11, 0, "2021-01-03"],
    ]
    specs = r.recommend(cols, rows)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.type.value == "line"
    assert spec.x_axis_title == "processed_date"
    assert spec.y_axis_title == "payment_attempt_count"


def test_identifier_only_numeric_columns_produce_no_chart():
    r = _recommender()
    cols = ["id", "customer_id", "order_id"]
    rows = [[1, 100, 5000], [2, 101, 5001], [3, 102, 5002]]
    assert r.recommend(cols, rows) == []


def test_category_numeric_pie_small_cardinality():
    r = _recommender()
    cols = ["region", "revenue"]
    rows = [["west", 100], ["east", 200], ["north", 150]]
    specs = r.recommend(cols, rows)
    assert len(specs) == 1
    assert specs[0].type.value == "pie"
    assert specs[0].confidence >= 0.6


def test_single_category_never_forces_a_chart():
    r = _recommender()
    cols = ["region", "revenue"]
    rows = [["west", 100], ["west", 200]]  # same category, totals to ONE slice
    assert r.recommend(cols, rows) == []


def test_temporal_numeric_line_chart():
    r = _recommender()
    cols = ["month", "total"]
    rows = [["2026-01", 100], ["2026-02", 200], ["2026-03", 150]]
    specs = r.recommend(cols, rows)
    assert len(specs) == 1
    assert specs[0].type.value == "line"
    assert specs[0].confidence == 0.9


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
