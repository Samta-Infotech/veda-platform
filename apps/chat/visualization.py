"""apps.chat.visualization — deterministic chart-type recommendation.

Operates purely on the (cols, rows) already produced by the existing query
pipeline — no new execution, no new LLM call, no semantic-layer dependency.
Kept isolated behind one ``VisualizationRecommender.recommend()`` entry point
so it can be lifted into a fuller analytics runtime later without any
caller-side changes.

Tabular presentation is intentionally NOT a chart type here — it's already
covered by the existing markdown-table content block (see
``services.py::_rows_to_markdown_table``). This module only ever recommends
the four frontend-contracted chart types (bar, line, pie, line_histogram),
or none at all.

Chart payload shapes follow the frontend Query API contract exactly:
- bar/line:  chart_data = {labels: str[], values: number[]}
- pie:       chart_data = {slices: [{name: str, value: number}]}
- line_histogram: a DUAL-SERIES combo chart (one dimension + two measures,
  e.g. sales volume as bars + conversion rate as a line) — NOT a single-
  column binned distribution. chart_data = {labels, histogram_values,
  line_values}.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any


class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    LINE_HISTOGRAM = "line_histogram"


@dataclass
class VisualizationSpec:
    type: ChartType
    title: str = ""
    sub_title: str | None = None
    x_axis_title: str | None = None
    y_axis_title: str | None = None
    histogram_title: str | None = None  # line_histogram only
    line_title: str | None = None       # line_histogram only
    chart_data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"type": self.type.value, "title": self.title, "chart_data": self.chart_data}
        for key in ("sub_title", "x_axis_title", "y_axis_title", "histogram_title", "line_title"):
            value = getattr(self, key)
            if value:
                d[key] = value
        return d


def _to_number(v: Any):
    """psycopg2 returns Decimal for NUMERIC/SUM/AVG columns — Decimal isn't a
    JSON number, so anything headed into chart_data is normalized to float."""
    return float(v) if isinstance(v, Decimal) else v


def _is_numeric(v: Any) -> bool:
    return isinstance(v, (int, float, Decimal)) and not isinstance(v, bool)


_DATE_RE = re.compile(r"^\d{4}-\d{2}(-\d{2})?")
_TEMPORAL_NAME_HINTS = ("date", "month", "year", "week", "day", "time", "period", "quarter")

_MAX_PIE_SLICES = 6
_TOP_N_CATEGORIES = 9  # + 1 "Other" bucket for the long tail = 10 slices/bars, still readable


class VisualizationRecommender:
    """Single responsibility: given the (cols, rows) already returned by the
    existing execution layer, recommend zero or more charts. Deterministic
    only — no LLM, no DB access.

    Column-count-agnostic on purpose: real query results routinely return
    more than 2 columns (e.g. region, month, revenue). Rather than requiring
    an exact 2-column shape, this scans every returned column for the best
    dimension/measure pairing(s) — any extra columns are simply not charted
    (they still appear in full in the existing markdown table).

    High-cardinality categories are bucketed into a top-N + "Other" slice
    instead of giving up — so bar/pie work for any category count. Genuinely
    un-chartable shapes (e.g. only text columns, or a single scalar with no
    dimension) correctly return no chart — the markdown table remains the
    fallback, not a guess.

    Returns a list (0+ specs) rather than a single Optional spec, matching
    the frontend contract's "response can include multiple chart blocks"
    shape. In practice a single query result almost always yields at most
    one meaningful chart — synthesizing multiple unrelated charts from one
    result set isn't attempted."""

    def recommend(self, cols: list, rows: list) -> list[VisualizationSpec]:
        if not cols or not rows:
            return []

        kinds = [self._infer_kind(cols[i], [row[i] for row in rows]) for i in range(len(cols))]
        numeric_idx = [i for i, k in enumerate(kinds) if k == "numeric"]
        temporal_idx = [i for i, k in enumerate(kinds) if k == "temporal"]
        categorical_idx = [i for i, k in enumerate(kinds) if k == "categorical"]
        dimension_idx = temporal_idx[:1] or categorical_idx[:1]

        # A dimension with TWO measures is the frontend's line_histogram: a
        # combo chart correlating two metrics over the same axis (e.g. sales
        # volume vs. conversion rate by month). Takes priority — it's a more
        # specific, more informative match than either measure charted alone.
        if dimension_idx and len(numeric_idx) >= 2:
            combo = self._combo(cols, rows, dimension_idx[0], numeric_idx[0], numeric_idx[1])
            if combo is not None:
                return [combo]

        if temporal_idx and numeric_idx and len(rows) > 1:
            return [self._line(cols, rows, temporal_idx[0], numeric_idx[0])]

        if categorical_idx and numeric_idx:
            spec = self._category_numeric(cols, rows, categorical_idx[0], numeric_idx[0])
            if spec is not None:
                return [spec]

        return []

    # --- kind inference ----------------------------------------------------

    def _infer_kind(self, col_name: str, values: list) -> str:
        name_lower = col_name.lower()
        if any(hint in name_lower for hint in _TEMPORAL_NAME_HINTS):
            return "temporal"
        sample = [v for v in values[:20] if v is not None]
        if not sample:
            return "categorical"
        if all(self._looks_like_date(v) for v in sample):
            return "temporal"
        if all(_is_numeric(v) for v in sample):
            return "numeric"
        return "categorical"

    @staticmethod
    def _looks_like_date(v: Any) -> bool:
        if hasattr(v, "isoformat"):  # datetime/date objects
            return True
        return isinstance(v, str) and bool(_DATE_RE.match(v))

    # --- chart builders ------------------------------------------------------

    def _category_numeric(self, cols: list, rows: list, cat_idx: int, val_idx: int) -> VisualizationSpec | None:
        pairs = [(str(row[cat_idx]), _to_number(row[val_idx])) for row in rows if _is_numeric(row[val_idx])]
        if not pairs:
            return None  # the "numeric" column turned out to be unusable for this row set

        title = f"{cols[val_idx]} by {cols[cat_idx]}"
        if len(pairs) <= _MAX_PIE_SLICES:
            slices = [{"name": name, "value": value} for name, value in pairs]
            return VisualizationSpec(type=ChartType.PIE, title=title, chart_data={"slices": slices})

        # Long tail: keep the top N by value, collapse the rest into "Other"
        # rather than dropping the chart entirely — this is what makes bar/pie
        # work for ANY category count, not just small ones.
        ranked = sorted(pairs, key=lambda p: p[1], reverse=True)
        top, rest = ranked[:_TOP_N_CATEGORIES], ranked[_TOP_N_CATEGORIES:]
        slices = [{"name": name, "value": value} for name, value in top]
        if rest:
            slices.append({"name": "Other", "value": sum(value for _, value in rest)})

        if len(slices) <= _MAX_PIE_SLICES:
            return VisualizationSpec(type=ChartType.PIE, title=title, chart_data={"slices": slices})

        labels = [s["name"] for s in slices]
        values = [s["value"] for s in slices]
        return VisualizationSpec(
            type=ChartType.BAR, title=title, x_axis_title=cols[cat_idx], y_axis_title=cols[val_idx],
            chart_data={"labels": labels, "values": values},
        )

    @staticmethod
    def _line(cols: list, rows: list, x_idx: int, y_idx: int) -> VisualizationSpec:
        return VisualizationSpec(
            type=ChartType.LINE, title=f"{cols[y_idx]} over {cols[x_idx]}",
            x_axis_title=cols[x_idx], y_axis_title=cols[y_idx],
            chart_data={"labels": [str(row[x_idx]) for row in rows],
                       "values": [_to_number(row[y_idx]) for row in rows]},
        )

    @staticmethod
    def _combo(cols: list, rows: list, dim_idx: int, hist_idx: int, line_idx: int) -> VisualizationSpec | None:
        triples = [
            (str(row[dim_idx]), _to_number(row[hist_idx]), _to_number(row[line_idx]))
            for row in rows if _is_numeric(row[hist_idx]) and _is_numeric(row[line_idx])
        ]
        if len(triples) < 2:
            return None
        return VisualizationSpec(
            type=ChartType.LINE_HISTOGRAM,
            title=f"{cols[hist_idx]} and {cols[line_idx]} by {cols[dim_idx]}",
            x_axis_title=cols[dim_idx], histogram_title=cols[hist_idx], line_title=cols[line_idx],
            chart_data={"labels": [t[0] for t in triples],
                       "histogram_values": [t[1] for t in triples],
                       "line_values": [t[2] for t in triples]},
        )
