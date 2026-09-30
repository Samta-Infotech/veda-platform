"""apps.chat.presentation_plan — formal PresentationPlan contract (2026-09-29).

Formalizes decisions the pipeline ALREADY makes — table_rendering.py's column
narrowing, visualization.py's chart recommendation, and veda_core's
analyze_result()/analytics_summary() — into one typed, observable object. This
module does not re-derive or re-decide anything: it is a pure wrapper over
already-computed (cols, rows, analytics, chart_specs), so it is purely additive
and cannot change what table/chart the user actually sees. Zero Django
dependency, same reasoning as table_rendering.py/visualization.py's own
separation (see their docstrings) — this module is safe to unit-test alone.

Why this exists: `analytics_summary()` already carries every field a formal
plan needs (result_shape, display_columns, query_relevant_columns,
chart_candidates), but scattered across a loose dict that table_rendering.py
and visualization.py each read independently. Wrapping it in one typed object
makes "why did this turn show/not show a chart" a single, testable, loggable
decision instead of an inference two call sites have to agree on by
convention.

NOT_APPLICABLE vs SAFE_FALLBACK — the one distinction this module exists to
make explicit:
- NOT_APPLICABLE: the query executed successfully and deterministic analysis
  ran fine; there is simply nothing meaningful to show as a table/chart (a
  scalar answer already covers it, or the result has zero rows). This is a
  SUCCESS, never logged/treated/counted as a failure.
- SAFE_FALLBACK: deterministic analysis (`analyze_result`) itself did not run
  or raised (caught by the engine's own best-effort try/except upstream — see
  veda_hybrid.py's "Analytics (skipped: ...)" sites). The user still gets
  whatever validated (cols, rows) survived — never fabricated, never re-run
  through an SLM to "recover" — just presented as a plain result with no
  narrowing/chart applied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from .table_rendering import fmt_header

class AnalysisStatus(str, Enum):
    SUCCESS = "SUCCESS"
    SAFE_FALLBACK = "SAFE_FALLBACK"


class PresentationMode(str, Enum):
    SUMMARY = "SUMMARY"
    TABLE = "TABLE"
    CHART = "CHART"
    TABLE_AND_CHART = "TABLE_AND_CHART"
    RAW_RESULT = "RAW_RESULT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class PresentationPlan:
    analysis_status: AnalysisStatus
    presentation_mode: PresentationMode
    display_columns: list = field(default_factory=list)
    query_relevant_columns: list = field(default_factory=list)
    chart_type: str | None = None
    x_axis: str | None = None
    y_axis: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "analysis_status": self.analysis_status.value,
            "presentation_mode": self.presentation_mode.value,
            "display_columns": list(self.display_columns or []),
            "query_relevant_columns": list(self.query_relevant_columns or []),
            "chart_type": self.chart_type,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "reason": self.reason,
        }


def build_presentation_plan(cols: list | None, rows: list | None,
                            analytics: dict | None,
                            chart_dicts: list | None) -> PresentationPlan:
    """`chart_dicts`: whatever `apps/chat/services.py::_build_visualizations`
    already produced (a list of `VisualizationSpec.to_dict()` dicts, or []) —
    never re-run here. `analytics`: `res0["analytics"]` — absent/None means the
    engine's own analyze_result() call didn't run or raised upstream (already
    logged there as "Analytics (skipped: ...)").
    """
    if not analytics:
        # Genuine analyzer failure/absence — SAFE_FALLBACK, not NOT_APPLICABLE.
        # The validated (cols, rows) are still shown as-is: no narrowing (no
        # query_relevant_columns signal exists to narrow with), no chart (no
        # column-kind/role signal exists to recommend one from either).
        has_result = bool(cols and rows)
        return PresentationPlan(
            analysis_status=AnalysisStatus.SAFE_FALLBACK,
            presentation_mode=(PresentationMode.RAW_RESULT if has_result
                               else PresentationMode.NOT_APPLICABLE),
            display_columns=list(cols or []),
            reason=("deterministic result analysis unavailable — showing the "
                    "validated result as-is" if has_result else
                    "deterministic result analysis unavailable and no rows to show"),
        )

    if not rows:
        # A successful query with zero rows is a SUCCESS with nothing to
        # present — not a failure, not a fallback (§4 of the request: a query
        # not needing a table/chart is not an error state).
        return PresentationPlan(
            analysis_status=AnalysisStatus.SUCCESS,
            presentation_mode=PresentationMode.NOT_APPLICABLE,
            reason="query executed successfully with no rows to present",
        )

    display_columns = list(analytics.get("display_columns") or cols or [])
    query_relevant_columns = list(analytics.get("query_relevant_columns") or [])

    if analytics.get("result_shape") == "SCALAR":
        return PresentationPlan(
            analysis_status=AnalysisStatus.SUCCESS,
            presentation_mode=PresentationMode.SUMMARY,
            display_columns=display_columns,
            query_relevant_columns=query_relevant_columns,
            reason="single scalar value — a table/chart would add nothing over the summary",
        )

    chart = (chart_dicts or [None])[0]
    # Chart-safety validation (§16 of the request): a chart spec only ever
    # names columns visualization.py itself read positionally off the REAL
    # `cols` list (see VisualizationRecommender — every axis title is
    # `_fmt_axis(cols[i])` for a real index i), so it cannot structurally
    # reference a column absent from the result. This check is defense-in-
    # depth, not a fix for an observed bug: it guards against a future spec
    # shape drifting from that invariant, not a known failure mode today.
    if chart and not _chart_columns_exist(chart, cols):
        chart = None

    if chart is None:
        return PresentationPlan(
            analysis_status=AnalysisStatus.SUCCESS,
            presentation_mode=PresentationMode.TABLE,
            display_columns=display_columns,
            query_relevant_columns=query_relevant_columns,
        )
    return PresentationPlan(
        analysis_status=AnalysisStatus.SUCCESS,
        presentation_mode=PresentationMode.TABLE_AND_CHART,
        display_columns=display_columns,
        query_relevant_columns=query_relevant_columns,
        chart_type=chart.get("type"),
        x_axis=chart.get("x_axis_title"),
        y_axis=chart.get("y_axis_title"),
    )


def _chart_columns_exist(chart: dict, cols: list | None) -> bool:
    """True unless the chart spec names an axis title that doesn't humanize-match
    ANY real column — see the docstring above on why this should never actually
    fire today. `_fmt_axis` (table_rendering.fmt_header) is applied to build axis
    titles from column names, so the comparison undoes that same humanization."""
    if not cols:
        return False    
    humanized = {fmt_header(c) for c in cols}
    for title in (chart.get("x_axis_title"), chart.get("y_axis_title"),
                 chart.get("histogram_title"), chart.get("line_title")):
        if title and title not in humanized:
            return False
    return True
