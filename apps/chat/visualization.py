"""apps.chat.visualization — deterministic chart-type recommendation.

Operates purely on the (cols, rows) already produced by the existing query
pipeline — no new execution, no new LLM call, no semantic-layer dependency.
Kept isolated behind one ``VisualizationRecommender.recommend()`` entry point
so it can be lifted into a fuller analytics runtime later without any
caller-side changes.

Tabular presentation is intentionally NOT a chart type here — it's already
covered by the existing markdown-table content block (see
``services.py::_rows_to_markdown_table``). This module only ever recommends
one of the four supported chart types, or no visualization at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
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
    chart_data: dict = field(default_factory=dict)


_DATE_RE = re.compile(r"^\d{4}-\d{2}(-\d{2})?")
_TEMPORAL_NAME_HINTS = ("date", "month", "year", "week", "day", "time", "period", "quarter")

_MAX_PIE_SLICES = 6
_TOP_N_CATEGORIES = 9  # + 1 "Other" bucket for the long tail = 10 slices/bars, still readable
_MIN_HISTOGRAM_ROWS = 15
_HISTOGRAM_BINS = 8


class VisualizationRecommender:
    """Single responsibility: given the (cols, rows) already returned by the
    existing execution layer, recommend one chart type. Deterministic only —
    no LLM, no DB access.

    Column-count-agnostic on purpose: real query results routinely return
    more than 2 columns (e.g. region, month, revenue). Rather than requiring
    an exact 2-column shape, this scans every returned column for the best
    dimension/measure pair — any extra columns are simply not charted (they
    still appear in full in the existing markdown table).

    High-cardinality categories are bucketed into a top-N + "Other" slice
    instead of giving up — so bar/pie work for any category count, not just
    small ones. Genuinely un-chartable shapes (e.g. only text columns, or a
    single scalar with no other dimension) still correctly return None —
    the markdown table remains the fallback presentation, not a guess."""

    def recommend(self, cols: list, rows: list) -> VisualizationSpec | None:
        if not cols or not rows:
            return None

        kinds = [self._infer_kind(cols[i], [row[i] for row in rows]) for i in range(len(cols))]
        numeric_idx = [i for i, k in enumerate(kinds) if k == "numeric"]
        temporal_idx = [i for i, k in enumerate(kinds) if k == "temporal"]
        categorical_idx = [i for i, k in enumerate(kinds) if k == "categorical"]

        # Priority: a genuine time series beats a plain category breakdown,
        # which beats a single-column distribution. Only the FIRST matching
        # dimension + FIRST matching measure are charted when several exist —
        # documented, deterministic, no silent "pick a random one" behavior.
        if temporal_idx and numeric_idx and len(rows) > 1:
            return self._line(cols, rows, temporal_idx[0], numeric_idx[0])

        if categorical_idx and numeric_idx:
            spec = self._category_numeric(cols, rows, categorical_idx[0], numeric_idx[0])
            if spec is not None:
                return spec

        if len(numeric_idx) == 1 and not categorical_idx and not temporal_idx and len(rows) > _MIN_HISTOGRAM_ROWS:
            return self._line_histogram(cols, rows, numeric_idx[0])

        return None

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
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in sample):
            return "numeric"
        return "categorical"

    @staticmethod
    def _looks_like_date(v: Any) -> bool:
        if hasattr(v, "isoformat"):  # datetime/date objects
            return True
        return isinstance(v, str) and bool(_DATE_RE.match(v))

    # --- chart builders ------------------------------------------------------

    def _category_numeric(self, cols: list, rows: list, cat_idx: int, val_idx: int) -> VisualizationSpec | None:
        pairs = [(str(row[cat_idx]), row[val_idx]) for row in rows
                if isinstance(row[val_idx], (int, float)) and not isinstance(row[val_idx], bool)]
        if not pairs:
            return None  # the "numeric" column turned out to be unusable for this row set

        title = f"{cols[val_idx]} by {cols[cat_idx]}"
        if len(pairs) <= _MAX_PIE_SLICES:
            labels, values = zip(*pairs)
            return VisualizationSpec(type=ChartType.PIE, title=title,
                                     chart_data={"labels": list(labels), "values": list(values)})

        # Long tail: keep the top N by value, collapse the rest into "Other"
        # rather than dropping the chart entirely — this is what makes bar/pie
        # work for ANY category count, not just small ones.
        ranked = sorted(pairs, key=lambda p: p[1], reverse=True)
        top, rest = ranked[:_TOP_N_CATEGORIES], ranked[_TOP_N_CATEGORIES:]
        labels = [p[0] for p in top]
        values = [p[1] for p in top]
        if rest:
            labels.append("Other")
            values.append(sum(p[1] for p in rest))

        chart_type = ChartType.PIE if len(labels) <= _MAX_PIE_SLICES else ChartType.BAR
        return VisualizationSpec(type=chart_type, title=title,
                                 chart_data={"labels": labels, "values": values})

    @staticmethod
    def _line(cols: list, rows: list, x_idx: int, y_idx: int) -> VisualizationSpec:
        return VisualizationSpec(
            type=ChartType.LINE, title=f"{cols[y_idx]} over {cols[x_idx]}",
            chart_data={"labels": [str(row[x_idx]) for row in rows],
                       "values": [row[y_idx] for row in rows]},
        )

    def _line_histogram(self, cols: list, rows: list, val_idx: int) -> VisualizationSpec | None:
        values = [row[val_idx] for row in rows if isinstance(row[val_idx], (int, float))]
        if len(values) < _MIN_HISTOGRAM_ROWS:
            return None
        lo, hi = min(values), max(values)
        if lo == hi:
            return None  # constant column — nothing to bin
        width = (hi - lo) / _HISTOGRAM_BINS
        counts = [0] * _HISTOGRAM_BINS
        for v in values:
            idx = min(int((v - lo) / width), _HISTOGRAM_BINS - 1)
            counts[idx] += 1
        bins = [f"{lo + i * width:.1f}-{lo + (i + 1) * width:.1f}" for i in range(_HISTOGRAM_BINS)]
        return VisualizationSpec(
            type=ChartType.LINE_HISTOGRAM, title=f"Distribution of {cols[val_idx]}",
            chart_data={"bins": bins, "counts": counts},
        )
