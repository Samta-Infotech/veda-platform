"""Tests for veda/result_analyzer.py (Insight Engine — deterministic
ResultAnalyzer/InsightContext). Pure-python, no DB, no network. Run from the
repo root: ``pytest tests/test_result_analyzer.py``"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))


# ---------------------------------------------------------------------------
# classify_result_type
# ---------------------------------------------------------------------------

def test_classify_empty():
    from veda.result_analyzer import classify_result_type
    assert classify_result_type(0, ["count"]) == "empty"


def test_classify_scalar():
    from veda.result_analyzer import classify_result_type
    assert classify_result_type(1, ["count"]) == "scalar"


def test_classify_single_row():
    from veda.result_analyzer import classify_result_type
    assert classify_result_type(1, ["name", "age"]) == "single_row"


def test_classify_multi_row():
    from veda.result_analyzer import classify_result_type
    assert classify_result_type(3, ["name", "age"]) == "multi_row"


# ---------------------------------------------------------------------------
# infer_column_kind
# ---------------------------------------------------------------------------

def test_infer_kind_temporal_by_name():
    from veda.result_analyzer import infer_column_kind
    assert infer_column_kind("transaction_date", ["not-a-date"]) == "temporal"


def test_infer_kind_temporal_by_value():
    from veda.result_analyzer import infer_column_kind
    assert infer_column_kind("event_col", ["2026-01-01", "2026-02-01"]) == "temporal"


def test_infer_kind_numeric():
    from veda.result_analyzer import infer_column_kind
    assert infer_column_kind("amount", [1, 2, 3.5]) == "numeric"


def test_infer_kind_categorical():
    from veda.result_analyzer import infer_column_kind
    assert infer_column_kind("status", ["active", "inactive"]) == "categorical"


def test_infer_kind_empty_values_defaults_categorical():
    from veda.result_analyzer import infer_column_kind
    assert infer_column_kind("mystery", [None, None]) == "categorical"


# ---------------------------------------------------------------------------
# analyze_result — no LLM, no new SQL query, zero-network by construction
# ---------------------------------------------------------------------------

def _sql():
    return ('SELECT payer_name, SUM(amount) AS total FROM ledger '
            'GROUP BY payer_name ORDER BY total DESC LIMIT 5')


def test_analyze_result_extracts_dimensions_measures_limit():
    from veda.result_analyzer import analyze_result
    ctx = analyze_result(
        "who are the top spenders", _sql(), ["payer_name", "total"],
        [{"payer_name": "Alice", "total": 500}, {"payer_name": "Bob", "total": 300}],
        sm=None, table="ledger",
    )
    assert ctx.result_type == "multi_row"
    assert ctx.row_count == 2
    assert ctx.dimensions == ["payer_name"]
    assert ctx.measures == ["amount"]
    assert ctx.limit == 5
    assert ctx.entities == ["ledger"]


def test_analyze_result_column_stats():
    from veda.result_analyzer import analyze_result
    ctx = analyze_result(
        "who are the top spenders", _sql(), ["payer_name", "total"],
        [{"payer_name": "Alice", "total": 500}, {"payer_name": "Bob", "total": 300},
         {"payer_name": None, "total": 300}],
        sm=None, table="ledger",
    )
    by_name = {s.name: s for s in ctx.column_stats}
    assert by_name["payer_name"].kind == "categorical"
    assert by_name["payer_name"].null_count == 1
    assert by_name["total"].kind == "numeric"
    assert by_name["total"].min == 300
    assert by_name["total"].max == 500
    assert by_name["total"].distinct_count == 2  # values are 500, 300, 300 -> 2 distinct


def test_analyze_result_applies_semantic_types():
    from veda.result_analyzer import analyze_result
    sm = {"columns": {"ledger.total": {"semantic_type": "MONETARY"}}}
    ctx = analyze_result(
        "top spenders", _sql(), ["payer_name", "total"],
        [{"payer_name": "Alice", "total": 500}], sm=sm, table="ledger",
    )
    by_name = {s.name: s for s in ctx.column_stats}
    assert by_name["total"].semantic_type == "MONETARY"
    assert by_name["payer_name"].semantic_type is None


def test_analyze_result_empty_rows():
    from veda.result_analyzer import analyze_result
    ctx = analyze_result("nothing here", "SELECT * FROM ledger WHERE 1=0",
                         ["id"], [], sm=None, table="ledger")
    assert ctx.result_type == "empty"
    assert ctx.row_count == 0
    # column identity is still known even with zero rows — just no data to profile
    assert [s.name for s in ctx.column_stats] == ["id"]
    assert ctx.column_stats[0].distinct_count == 0


def test_analyze_result_max_rows_caps_sample_but_not_row_count():
    from veda.result_analyzer import analyze_result
    rows = [{"id": i} for i in range(10)]
    ctx = analyze_result("list ids", "SELECT id FROM ledger", ["id"], rows,
                         sm=None, table="ledger", max_rows=3)
    assert ctx.row_count == 10          # TRUE total, not truncated
    assert len(ctx.sample_rows) == 3    # capped sample for stats/prompt


# ---------------------------------------------------------------------------
# Phase 2 gap-fill: avg/median stats, connector_type/query_intent/
# confidence_inputs passthrough, result_shape detection
# ---------------------------------------------------------------------------

def test_column_stats_avg_and_median():
    from veda.result_analyzer import analyze_result
    ctx = analyze_result("amounts", "SELECT amount FROM ledger", ["amount"],
                         [{"amount": 100}, {"amount": 200}, {"amount": 300}],
                         sm=None, table="ledger")
    stat = ctx.column_stats[0]
    assert stat.avg == 200.0
    assert stat.median == 200


def test_analyze_result_passthrough_metadata_defaults():
    from veda.result_analyzer import analyze_result
    ctx = analyze_result("q", "SELECT 1", ["x"], [{"x": 1}], sm=None, table="t")
    assert ctx.connector_type == "relational"    # default when caller doesn't specify
    assert ctx.query_intent is None
    assert ctx.confidence_inputs == {}


def test_analyze_result_passthrough_metadata_supplied():
    from veda.result_analyzer import analyze_result
    ctx = analyze_result("q", "SELECT 1", ["x"], [{"x": 1}], sm=None, table="t",
                         connector_type="csv", query_intent="AGGREGATE",
                         confidence_inputs={"anchor": 0.8, "join": 0.6})
    assert ctx.connector_type == "csv"
    assert ctx.query_intent == "AGGREGATE"
    assert ctx.confidence_inputs == {"anchor": 0.8, "join": 0.6}


def test_result_shape_ranking():
    from veda.result_analyzer import analyze_result
    sql = 'SELECT id, amount FROM ledger ORDER BY amount DESC LIMIT 10'
    ctx = analyze_result("top 10", sql, ["id", "amount"],
                         [{"id": 1, "amount": 5}, {"id": 2, "amount": 3}], sm=None, table="ledger")
    assert ctx.result_shape == "RANKING"


def test_result_shape_trend():
    from veda.result_analyzer import analyze_result
    sql = 'SELECT month, SUM(amount) AS total FROM ledger GROUP BY month'
    ctx = analyze_result("revenue by month", sql, ["month", "total"],
                         [{"month": "2026-01", "total": 100}, {"month": "2026-02", "total": 200}],
                         sm=None, table="ledger")
    assert ctx.result_shape == "TREND"


def test_result_shape_pivot():
    from veda.result_analyzer import analyze_result
    sql = ('SELECT region, SUM(revenue) AS rev, SUM(cost) AS cost FROM sales '
           'GROUP BY region')
    ctx = analyze_result("revenue and cost by region", sql, ["region", "rev", "cost"],
                         [{"region": "west", "rev": 100, "cost": 40},
                          {"region": "east", "rev": 200, "cost": 90}], sm=None, table="sales")
    assert ctx.result_shape == "PIVOT"


def test_result_shape_detail_table():
    from veda.result_analyzer import analyze_result
    sql = 'SELECT id, label, notes FROM ledger'
    ctx = analyze_result("show me the ledger", sql, ["id", "label", "notes"],
                         [{"id": 1, "label": "a", "notes": "x"},
                          {"id": 2, "label": "b", "notes": "y"}], sm=None, table="ledger")
    assert ctx.result_shape == "DETAIL_TABLE"


def test_result_shape_scalar_for_non_multi_row():
    from veda.result_analyzer import analyze_result
    ctx = analyze_result("count", "SELECT COUNT(*) AS n FROM ledger", ["n"],
                         [{"n": 5}], sm=None, table="ledger")
    assert ctx.result_shape == "SCALAR"


def test_result_shape_distribution_vs_grouped():
    from veda.result_analyzer import analyze_result
    # COUNT-only aggregation per category -> DISTRIBUTION (a frequency breakdown)
    dist_sql = 'SELECT country, COUNT(*) AS n FROM users GROUP BY country'
    dist_ctx = analyze_result("how many users per country", dist_sql, ["country", "n"],
                              [{"country": "US", "n": 10}, {"country": "UK", "n": 5}],
                              sm=None, table="users")
    assert dist_ctx.result_shape == "DISTRIBUTION"

    # A real (non-count) measure per category -> GROUPED (a comparison, not a frequency)
    grouped_sql = 'SELECT country, SUM(revenue) AS total FROM sales GROUP BY country'
    grouped_ctx = analyze_result("revenue by country", grouped_sql, ["country", "total"],
                                 [{"country": "US", "total": 500}, {"country": "UK", "total": 300}],
                                 sm=None, table="sales")
    assert grouped_ctx.result_shape == "GROUPED"


# ---------------------------------------------------------------------------
# Semantic column role — identifier | dimension | measure | date | boolean | text
# ---------------------------------------------------------------------------

def test_role_identifier_by_name():
    from veda.result_analyzer import classify_column_role
    assert classify_column_role("id", [1, 2, 3], "numeric") == "identifier"
    assert classify_column_role("customer_id", [1, 2, 3], "numeric") == "identifier"
    assert classify_column_role("order_uuid", ["a-b-c"], "categorical") == "identifier"


def test_role_measure_by_kind():
    from veda.result_analyzer import classify_column_role
    assert classify_column_role("amount", [100, 200], "numeric") == "measure"
    assert classify_column_role("revenue", [1.5, 2.5], "numeric") == "measure"


def test_role_date_by_kind():
    from veda.result_analyzer import classify_column_role
    assert classify_column_role("created_at", ["2026-01-01"], "temporal") == "date"


def test_role_boolean_by_name_and_values():
    from veda.result_analyzer import classify_column_role
    assert classify_column_role("is_active", [True, False], "categorical") == "boolean"
    assert classify_column_role("flag", [True, False], "categorical") == "boolean"


def test_role_text_vs_dimension():
    from veda.result_analyzer import classify_column_role
    assert classify_column_role("email", ["a@b.com"], "categorical") == "text"
    assert classify_column_role("country", ["US", "UK"], "categorical") == "dimension"
    long_text = ["This is a long free-form note about the customer's request " * 2]
    assert classify_column_role("notes", long_text, "categorical") == "text"


def test_role_semantic_type_takes_priority():
    from veda.result_analyzer import classify_column_role
    # column NAMED like a measure but the ingested semantic model says IDENTIFIER
    assert classify_column_role("amount", [1, 2], "numeric", semantic_type="IDENTIFIER") == "identifier"
    assert classify_column_role("code", ["a"], "categorical", semantic_type="MONETARY") == "measure"


def test_column_stats_carry_role_via_analyze_result():
    from veda.result_analyzer import analyze_result
    ctx = analyze_result("show payments", "SELECT id, amount FROM payments", ["id", "amount"],
                         [{"id": 1, "amount": 100}, {"id": 2, "amount": 200}], sm=None, table="payments")
    by_name = {s.name: s for s in ctx.column_stats}
    assert by_name["id"].role == "identifier"
    assert by_name["amount"].role == "measure"


# ---------------------------------------------------------------------------
# Deterministic chart confidence
# ---------------------------------------------------------------------------

def test_chart_confidence_zero_for_identifier():
    from veda.result_analyzer import chart_confidence
    assert chart_confidence("RANKING", "bar", "identifier", "measure") == 0.0
    assert chart_confidence("RANKING", "bar", "dimension", "identifier") == 0.0


def test_chart_confidence_zero_for_non_chartable_shapes():
    from veda.result_analyzer import chart_confidence
    assert chart_confidence("SCALAR", "bar", "dimension", "measure") == 0.0
    assert chart_confidence("DETAIL_TABLE", "bar", "dimension", "measure") == 0.0
    assert chart_confidence("PIVOT", "bar", "dimension", "measure") == 0.0


def test_chart_confidence_high_for_canonical_pairing():
    from veda.result_analyzer import chart_confidence
    assert chart_confidence("TREND", "line", "date", "measure") >= 0.9
    assert chart_confidence("RANKING", "bar", "dimension", "measure") >= 0.9


def test_chart_confidence_lower_for_non_canonical():
    from veda.result_analyzer import chart_confidence
    canonical = chart_confidence("TREND", "line", "date", "measure")
    noncanonical = chart_confidence("TREND", "pie", "date", "measure")
    assert noncanonical < canonical
