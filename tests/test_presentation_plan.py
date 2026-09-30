"""Tests for apps.chat.presentation_plan — the formal PresentationPlan contract
(2026-09-29 consistency work). Pure-Python, no Django (matches table_rendering.py/
visualization.py's own zero-Django-dependency pattern)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps.chat.presentation_plan import (
    build_presentation_plan, AnalysisStatus, PresentationMode,
)


def test_scalar_result_gets_summary_mode():
    analytics = {"result_shape": "SCALAR", "display_columns": ["count"],
                "query_relevant_columns": []}
    plan = build_presentation_plan(["count"], [[42]], analytics, [])
    assert plan.analysis_status == AnalysisStatus.SUCCESS
    assert plan.presentation_mode == PresentationMode.SUMMARY


def test_empty_rows_is_not_applicable_not_a_failure():
    """A successful query with zero rows must never look like a failure."""
    analytics = {"result_shape": "DETAIL_TABLE"}
    plan = build_presentation_plan(["a", "b"], [], analytics, [])
    assert plan.analysis_status == AnalysisStatus.SUCCESS
    assert plan.presentation_mode == PresentationMode.NOT_APPLICABLE


def test_detail_table_with_no_chart_is_table_mode():
    analytics = {"result_shape": "DETAIL_TABLE", "display_columns": ["name", "amount"],
                "query_relevant_columns": ["name", "amount"]}
    plan = build_presentation_plan(["name", "amount"], [["A", 1], ["B", 2]], analytics, [])
    assert plan.presentation_mode == PresentationMode.TABLE
    assert plan.display_columns == ["name", "amount"]


def test_grouped_with_a_real_chart_is_table_and_chart_mode():
    analytics = {"result_shape": "GROUPED", "display_columns": ["city", "total"],
                "query_relevant_columns": ["city", "total"]}
    chart = {"type": "bar", "x_axis_title": "City", "y_axis_title": "Total",
             "chart_data": {"labels": ["A", "B"], "values": [1, 2]}}
    plan = build_presentation_plan(["city", "total"], [["A", 1], ["B", 2]], analytics, [chart])
    assert plan.presentation_mode == PresentationMode.TABLE_AND_CHART
    assert plan.chart_type == "bar"
    assert plan.x_axis == "City"
    assert plan.y_axis == "Total"


def test_missing_analytics_is_safe_fallback_with_raw_result():
    """analyze_result() didn't run / raised upstream — the validated (cols, rows)
    are still shown, never fabricated, never re-derived."""
    plan = build_presentation_plan(["a", "b"], [[1, 2]], None, [])
    assert plan.analysis_status == AnalysisStatus.SAFE_FALLBACK
    assert plan.presentation_mode == PresentationMode.RAW_RESULT
    assert plan.display_columns == ["a", "b"]


def test_missing_analytics_and_no_rows_is_safe_fallback_not_applicable():
    plan = build_presentation_plan([], [], None, [])
    assert plan.analysis_status == AnalysisStatus.SAFE_FALLBACK
    assert plan.presentation_mode == PresentationMode.NOT_APPLICABLE


def test_chart_referencing_a_real_column_survives():
    analytics = {"result_shape": "TREND", "display_columns": ["month", "revenue"],
                "query_relevant_columns": []}
    chart = {"type": "line", "x_axis_title": "Month", "y_axis_title": "Revenue"}
    plan = build_presentation_plan(["month", "revenue"], [["Jan", 1]], analytics, [chart])
    assert plan.presentation_mode == PresentationMode.TABLE_AND_CHART


def test_chart_referencing_a_nonexistent_column_is_dropped_defensively():
    """Chart-safety validation (defense in depth) — a spec whose axis title
    doesn't humanize-match any real column falls back to TABLE-only rather
    than rendering a chart that names data not actually in the result."""
    analytics = {"result_shape": "TREND", "display_columns": ["month", "revenue"],
                "query_relevant_columns": []}
    chart = {"type": "line", "x_axis_title": "Not A Real Column", "y_axis_title": "Revenue"}
    plan = build_presentation_plan(["month", "revenue"], [["Jan", 1]], analytics, [chart])
    assert plan.presentation_mode == PresentationMode.TABLE
    assert plan.chart_type is None


def test_multi_source_federated_result_uses_the_same_contract():
    """PresentationPlan must never assume one-source == one-table == one-chart —
    it reads only the FINAL validated (cols, rows, analytics), never a source_id
    or route type. A federated/cross-source result (join across sources, no
    single "table" name) goes through the identical decision path as a plain
    single-source one."""
    analytics = {
        "result_shape": "GROUPED",
        "display_columns": ["source_name", "total_revenue"],
        "query_relevant_columns": ["source_name", "total_revenue"],
        # federated results carry no single `table` — analytics.table is absent/None
    }
    chart = {"type": "bar", "x_axis_title": "Source Name", "y_axis_title": "Total Revenue"}
    plan = build_presentation_plan(
        ["source_name", "total_revenue"],
        [["homzhub", 100], ["crm", 50]],
        analytics, [chart],
    )
    assert plan.analysis_status == AnalysisStatus.SUCCESS
    assert plan.presentation_mode == PresentationMode.TABLE_AND_CHART
    assert plan.chart_type == "bar"


def test_multi_source_scalar_cross_source_aggregate_is_summary():
    """A cross-source scalar aggregate (e.g. total customer count across all
    sources) is still SUMMARY — the same rule as any other scalar, regardless
    of how many sources fed it."""
    analytics = {"result_shape": "SCALAR"}
    plan = build_presentation_plan(["total_customers"], [[9001]], analytics, [])
    assert plan.presentation_mode == PresentationMode.SUMMARY


def test_to_dict_is_json_safe():
    plan = build_presentation_plan(["a"], [[1]], {"result_shape": "SCALAR"}, [])
    d = plan.to_dict()
    assert d["analysis_status"] == "SUCCESS"
    assert d["presentation_mode"] == "SUMMARY"
    assert isinstance(d["display_columns"], list)
