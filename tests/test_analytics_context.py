"""Tests for the centralized deterministic analytics layer (2026-07):
veda/result_analyzer.py's grounding fields, pattern detection, chart
candidates and analytics_summary, plus the consumers — follow-up grounding
in query/result_explainer.py and the api-tier VisualizationRecommender's
analytics-aware classification. Pure-python, no DB, no network, no SLM.
Run from the repo root: ``pytest tests/test_analytics_context.py``"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Pattern detection — rule-based, zero LLM
# ---------------------------------------------------------------------------

def _stats_for(columns, rows):
    from veda.result_analyzer import _column_stats
    return _column_stats(columns, rows, max_rows=200)


# ---------------------------------------------------------------------------
# camelCase identifier detection (2026-07-17) — engine-side copy, kept in sync
# with apps/chat/visualization.py's own _is_identifier. See that test file's
# camelCase tests for the full false-positive/false-negative rationale.
# ---------------------------------------------------------------------------

def test_engine_side_camelcase_identifier_detected():
    from veda.result_analyzer import _looks_like_identifier
    for name in ("AccountID", "CustomerId", "orderID", "buildId", "account_id", "id"):
        assert _looks_like_identifier(name), f"{name!r} should be an identifier"


def test_engine_side_lowercase_words_ending_in_id_not_flagged():
    from veda.result_analyzer import _looks_like_identifier
    for word in ("paid", "valid", "grid", "hybrid", "android", "avoid", "solid"):
        assert not _looks_like_identifier(word), f"{word!r} must NOT be an identifier"


def test_engine_side_camelcase_identifier_excluded_from_column_stats_role():
    """End-to-end through classify_column_role (no semantic_type available —
    the degraded/structural path federated/nosql results actually take)."""
    from veda.result_analyzer import classify_column_role
    values = [f"acc-{i}" for i in range(5)]
    role = classify_column_role("AccountID", values, kind="categorical")
    assert role == "identifier"


def test_dominance_suppressed_when_sql_already_filters_that_column():
    """The trivial-insight guard: 'DEBIT is the dominant entry type' must NOT
    be emitted when the executed SQL itself says WHERE entry_type = 'DEBIT'."""
    from veda.result_analyzer import detect_patterns
    rows = [{"entry_type": "DEBIT", "amount": i * 10} for i in range(9)] + \
           [{"entry_type": "CREDIT", "amount": 5}]
    stats = _stats_for(["entry_type", "amount"], rows)

    unfiltered = detect_patterns("DETAIL_TABLE", stats, rows, filters=[], measures=[])
    assert any(p.kind == "dominance" and p.column == "entry_type" for p in unfiltered)

    filtered = detect_patterns("DETAIL_TABLE", stats, rows,
                               filters=[("entry_type", "EQ", "DEBIT")], measures=[])
    assert not any(p.kind == "dominance" and p.column == "entry_type" for p in filtered)


def test_missing_values_pattern():
    from veda.result_analyzer import detect_patterns
    rows = [{"name": f"u{i}", "last_login": None if i < 6 else "2026-01-01"}
            for i in range(10)]
    stats = _stats_for(["name", "last_login"], rows)
    pats = detect_patterns("DETAIL_TABLE", stats, rows, filters=[], measures=[])
    missing = [p for p in pats if p.kind == "missing_values" and p.column == "last_login"]
    assert missing and "6 of 10" in missing[0].detail


def test_trend_growth_pattern():
    from veda.result_analyzer import detect_patterns
    rows = [{"month": f"2026-0{m}", "revenue": 1000 * m} for m in range(1, 5)]
    stats = _stats_for(["month", "revenue"], rows)
    pats = detect_patterns("TREND", stats, rows, filters=[], measures=["revenue"])
    assert any(p.kind == "growth" and p.column == "revenue" for p in pats)


def test_ranking_top_gap_pattern():
    from veda.result_analyzer import detect_patterns
    rows = [{"name": "a", "total": 1000}, {"name": "b", "total": 100},
            {"name": "c", "total": 90}]
    stats = _stats_for(["name", "total"], rows)
    pats = detect_patterns("RANKING", stats, rows, filters=[], measures=["total"])
    assert any(p.kind == "top_gap" for p in pats)


def test_identifier_columns_never_produce_patterns():
    from veda.result_analyzer import detect_patterns
    rows = [{"user_id": None} for _ in range(10)]
    stats = _stats_for(["user_id"], rows)
    assert detect_patterns("DETAIL_TABLE", stats, rows, filters=[], measures=[]) == []


# ---------------------------------------------------------------------------
# Grounding fields + chart candidates via analyze_result
# ---------------------------------------------------------------------------

_SM = {
    "tables": {
        "accounts_generalledger": {"primary_entity": "A single financial transaction."},
    },
    "columns": {
        "accounts_generalledger.amount":       {"analytics_role": "MEASURE",
                                                "importance_class": "HIGH",
                                                "semantic_type": "MONETARY"},
        "accounts_generalledger.entry_type":   {"analytics_role": "DIMENSION",
                                                "importance_class": "HIGH",
                                                "semantic_type": "CATEGORY"},
        "accounts_generalledger.created_at":   {"analytics_role": "TIME_DIMENSION",
                                                "importance_class": "MEDIUM",
                                                "semantic_type": "TEMPORAL"},
        "accounts_generalledger.id":           {"analytics_role": "IDENTIFIER",
                                                "semantic_type": "IDENTIFIER"},
    },
}


def _ctx(monkeypatch, sql="SELECT entry_type, SUM(amount) AS total FROM accounts_generalledger "
                          "GROUP BY entry_type LIMIT 100"):
    import veda.result_analyzer as ra
    monkeypatch.setattr(ra, "_related_entities",
                        lambda table, cap=8: ["accounts_paymenttransaction"] if table else [])
    rows = [{"entry_type": "DEBIT", "total": 900}, {"entry_type": "CREDIT", "total": 100},
            {"entry_type": "REFUND", "total": 50}]
    return ra.analyze_result("transaction totals by entry type", sql,
                             ["entry_type", "total"], rows,
                             sm=_SM, table="accounts_generalledger")


def test_grounding_fields_populated(monkeypatch):
    ctx = _ctx(monkeypatch)
    assert ctx.primary_entity == "A single financial transaction."
    assert ctx.related_entities == ["accounts_paymenttransaction"]
    assert "amount" in ctx.available_measures
    assert "entry_type" in ctx.available_dimensions and "created_at" in ctx.available_dimensions
    assert "id" not in ctx.available_measures and "id" not in ctx.available_dimensions


def test_chart_candidates_grouped_shape(monkeypatch):
    ctx = _ctx(monkeypatch)
    assert ctx.result_shape == "GROUPED"
    assert ctx.chart_candidates, "GROUPED shape must yield chart candidates"
    top = ctx.chart_candidates[0]
    assert top["type"] == "bar" and top["x_axis"] == "entry_type" and top["y_axis"] == "total"
    assert all(c["type"] in ("bar", "line", "pie") for c in ctx.chart_candidates)


def test_analytics_summary_is_json_safe(monkeypatch):
    from veda.result_analyzer import analytics_summary
    ctx = _ctx(monkeypatch)
    wire = analytics_summary(ctx)
    encoded = json.dumps(wire)   # must not raise
    assert wire["result_shape"] == "GROUPED"
    assert wire["display_columns"] == ["entry_type", "total"]
    assert {"name": "entry_type", "kind": "categorical", "role": "dimension"} in wire["column_stats"]
    assert "sample_rows" not in wire and "semantic_model" not in wire
    assert isinstance(encoded, str)


# ---------------------------------------------------------------------------
# Follow-up groundedness gate
# ---------------------------------------------------------------------------

def test_follow_up_validation_drops_invented_concepts(monkeypatch):
    from query.result_explainer import validate_follow_up_questions
    ctx = _ctx(monkeypatch)
    kept = validate_follow_up_questions([
        "Break down by entry type",                      # result column — grounded
        "Compare total amount by month",                 # available measure — grounded
        "Show customer churn by marketing campaign",     # invented concepts — dropped
    ], ctx)
    assert "Break down by entry type" in kept
    assert "Compare total amount by month" in kept
    assert all("churn" not in q for q in kept)


def test_follow_up_validation_empty_input():
    from query.result_explainer import validate_follow_up_questions

    class _Bare:
        columns, table, primary_entity = [], None, None
        available_measures, available_dimensions, related_entities = [], [], []

    assert validate_follow_up_questions([], _Bare()) == []


# ---------------------------------------------------------------------------
# API-tier visualization consumes the server-side classification
# ---------------------------------------------------------------------------

def test_recommender_prefers_server_analytics_roles():
    """'account' looks categorical/numeric structurally (name heuristics can't
    catch it) — but when the engine's analytics says role=identifier, the
    api-tier must exclude it instead of charting it."""
    from apps.chat.visualization import VisualizationRecommender
    cols = ["account", "amount"]
    rows = [[f"ACC-{i}", 100 + i] for i in range(1, 8)]

    baseline = VisualizationRecommender().recommend(cols, rows)
    assert baseline, "sanity: without analytics this shape produces a chart"

    analytics = {"column_stats": [
        {"name": "account", "kind": "categorical", "role": "identifier"},
        {"name": "amount", "kind": "numeric", "role": "measure"},
    ]}
    assert VisualizationRecommender().recommend(cols, rows, analytics=analytics) == []


def test_recommender_unchanged_without_analytics():
    from apps.chat.visualization import VisualizationRecommender
    cols = ["category", "value"]
    rows = [["a", 10], ["b", 20], ["c", 30]]
    specs = VisualizationRecommender().recommend(cols, rows)          # no analytics arg
    specs2 = VisualizationRecommender().recommend(cols, rows, analytics=None)
    assert [s.type for s in specs] == [s.type for s in specs2]
    assert specs and specs[0].type.value == "pie"
