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

from .table_rendering import fmt_header as _fmt_axis   # "customer_name" -> "Customer Name" —
# same humanization the table headers use, applied to chart titles/axis labels
# for consistency (2026-07-17); table_rendering.py has zero Django dependency,
# so importing it here doesn't pull anything heavier into this module.


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
    confidence: float = 1.0             # deterministic 0-1 — see _CONFIDENCE_THRESHOLD

    def to_dict(self) -> dict:
        d = {"type": self.type.value, "title": self.title, "chart_data": self.chart_data,
             "confidence": self.confidence}
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

# Identifier detection — a self-contained copy of the same heuristic
# veda_core/veda/result_analyzer.py's classify_column_role() uses (which
# itself mirrors config.IDENTIFIER_SUFFIXES). Not a shared import: this
# Django api-tier module never imports veda_core (apps/query/inference_client.py's
# own docstring) — kept in sync manually if either changes.
#
# WHY this exists: an id/uuid/code column is structurally "numeric" (or a
# short low-cardinality "categorical") by _infer_kind's rules alone, which
# previously let it get picked as a chart measure or dimension (e.g. a
# line_histogram plotting a primary-key `id` column as one of the two
# "measures" against payment_attempt_count) — a real, observed production bug.
# Identifiers are excluded from every candidate pool below; they may still
# appear in the accompanying markdown table, just never charted.
_IDENTIFIER_SUFFIXES = ("_id", "_uuid", "_key", "_no", "_number", "_num", "_code", "_ref")
_IDENTIFIER_NAMES = ("id", "uuid", "guid")
# camelCase identifier suffix (AccountID/customerId/buildId) — see _is_identifier.
_CAMEL_ID_SUFFIX_RE = re.compile(r"[a-z0-9](Id|ID)$")

# Free-text detection — a self-contained copy of the same heuristic
# veda_core/veda/result_analyzer.py uses to keep free-text columns (notes,
# descriptions, addresses...) out of chart axes. Structurally these are
# "categorical" by _infer_kind's other rules (non-numeric, non-date strings),
# so without this check a column like `notes` could outrank a real dimension
# like `label` for the chart's category axis — a real, observed bug (cols =
# ['notes', 'label', 'amount'] picked 'notes' over 'label').
_TEXT_NAME_HINTS = ("email", "notes", "description", "remarks", "comment", "address", "bio")
_TEXT_AVG_LEN_THRESHOLD = 40  # avg sampled value length above this reads as free text, not a category

_MAX_PIE_SLICES = 6
_TOP_N_CATEGORIES = 9  # + 1 "Other" bucket for the long tail = 10 slices/bars, still readable

# Below this, no chart is returned at all — a table-only response is always
# safer than a low-confidence or borderline-meaningless chart.
_CONFIDENCE_THRESHOLD = 0.6


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

    Returns a list (0+ specs), matching the frontend contract's "response can
    include multiple chart blocks" shape. Multi-viz support (2026-07): when a
    result naturally supports more than one EQUALLY VALID rendering of the
    SAME data — a small category breakdown (pie + bar) or a time series
    (line + bar) — both are returned, in the same order today's single-chart
    behavior already picked (so a caller that only reads specs[0] sees zero
    change). This is never "synthesizing" unrelated charts from one result;
    every additional spec reuses the exact same (labels, values)/(slices)
    data and confidence the primary spec already computed — no new
    recommendation logic, no guessing."""

    def recommend(self, cols: list, rows: list, analytics: dict | None = None) -> list[VisualizationSpec]:
        """`analytics` (optional): the engine's own deterministic analysis
        (veda_core result_analyzer's analytics_summary, riding the result dict
        across the HTTP boundary). When present, its per-column kind/role is
        AUTHORITATIVE — it was computed once, with semantic-model access this
        tier doesn't have — and this module's own structural heuristics run
        only for columns the engine didn't cover (or when analytics is absent
        entirely, e.g. federated results). Kills the double classification
        without breaking the api-tier's no-veda_core-import boundary: what
        crosses is plain data, not an import."""
        if not cols or not rows:
            return []

        stats_by_name = {}
        for s in (analytics or {}).get("column_stats") or []:
            if isinstance(s, dict) and s.get("name"):
                stats_by_name[s["name"]] = s

        def _kind(i: int) -> str:
            st = stats_by_name.get(cols[i])
            if st:
                # The engine's SEMANTIC role is authoritative and OUTRANKS the
                # structural kind — a CATEGORY dimension coded with numeric values
                # (year, rating, postal code) must chart as the category axis, not be
                # mistaken for a measure just because its values look numeric. Only
                # fall back to the structural `kind` when no role was resolved (e.g.
                # federated results with no semantic model). Naming/dtype heuristics
                # never override stronger metadata (Phase-5 invariant).
                role = st.get("role")
                kind = st.get("kind")
                if role == "text":
                    return "text"
                if role == "date" or kind == "temporal":
                    return "temporal"
                if role == "dimension":
                    return "categorical"
                if role == "measure":
                    return "numeric"
                if kind in ("temporal", "numeric", "categorical"):
                    return kind
            return self._infer_kind(cols[i], [row[i] for row in rows])

        def _ident(i: int) -> bool:
            st = stats_by_name.get(cols[i])
            if st and st.get("role"):
                return st["role"] == "identifier"
            return self._is_identifier(cols[i])

        kinds = [_kind(i) for i in range(len(cols))]
        is_id = [_ident(i) for i in range(len(cols))]
        # Identifiers are excluded from every candidate pool up front — an id/
        # uuid/code column never becomes a measure OR a dimension, regardless
        # of how numeric/categorical it structurally looks.
        numeric_idx = [i for i, k in enumerate(kinds) if k == "numeric" and not is_id[i]]
        temporal_idx = [i for i, k in enumerate(kinds) if k == "temporal" and not is_id[i]]
        categorical_idx = [i for i, k in enumerate(kinds) if k == "categorical" and not is_id[i]]
        dimension_idx = temporal_idx[:1] or categorical_idx[:1]

        # A dimension with TWO measures is the frontend's line_histogram: a
        # combo chart correlating two metrics over the same axis (e.g. sales
        # volume vs. conversion rate by month). Takes priority — it's a more
        # specific, more informative match than either measure charted alone.
        if dimension_idx and len(numeric_idx) >= 2:
            combo = self._combo(cols, rows, dimension_idx[0], numeric_idx[0], numeric_idx[1])
            if combo is not None and combo.confidence >= _CONFIDENCE_THRESHOLD:
                return [combo]

        if temporal_idx and numeric_idx and len(rows) > 1:
            line_spec = self._line(cols, rows, temporal_idx[0], numeric_idx[0])
            if line_spec.confidence >= _CONFIDENCE_THRESHOLD:
                # Bar-over-time is an equally valid read of the SAME (labels,
                # values) data as the line chart — same confidence, since it's
                # the identical underlying data, just a different chart
                # geometry, not a separately-justified guess. Line stays
                # first (today's single-chart behavior — any caller that
                # only reads specs[0] sees no change at all); bar is the new
                # additive second chart (multi-viz support).
                bar_spec = self._bar_over_time(cols, rows, temporal_idx[0], numeric_idx[0],
                                               line_spec.confidence)
                return [line_spec, bar_spec]

        if categorical_idx and numeric_idx:
            specs = self._category_numeric(cols, rows, categorical_idx[0], numeric_idx[0])
            # A RANKING (top/bottom-N) is NOT a part-of-whole: a pie of the top N
            # misrepresents proportions (the N don't sum to the whole). When the engine
            # classified the shape as RANKING, lead with the bar (its canonical chart,
            # matching result_analyzer.CANONICAL_CHART_FOR_SHAPE) and drop the pie.
            # Shape-driven, not name-driven; a GROUPED breakdown keeps pie+bar.
            if (analytics or {}).get("result_shape") == "RANKING":
                specs = [s for s in specs if s.type != ChartType.PIE] or specs
            confident = [s for s in specs if s.confidence >= _CONFIDENCE_THRESHOLD]
            if confident:
                return confident

        # RANKING rescue (2026-07-19): identifier-heavy schemas often leave a
        # ranking with a real measure but NO chartable dimension — every label-ish
        # column is an id/reference, and identifiers are (correctly) banned from
        # every pool above, so "top N by amount" got no chart at all. In a RANKING,
        # though, an id/reference IS the row's name — a bar leaderboard keyed by it
        # is meaningful (unlike an id-pie/aggregation, which stays banned). Fires
        # ONLY when the ENGINE itself classified the shape (result_analyzer's
        # RANKING, never guessed here) and every structural rule above found
        # nothing; picks a non-numeric label column, preferring non-identifiers.
        if ((analytics or {}).get("result_shape") == "RANKING"
                and numeric_idx and len(rows) > 1):
            non_numeric = [i for i, k in enumerate(kinds) if k != "numeric"]
            label_idx = next((i for i in non_numeric if not is_id[i]),
                             next(iter(non_numeric), None))
            if label_idx is not None:
                y = numeric_idx[0]
                spec = VisualizationSpec(
                    type=ChartType.BAR,
                    title=f"{_fmt_axis(cols[y])} by {_fmt_axis(cols[label_idx])}",
                    x_axis_title=_fmt_axis(cols[label_idx]),
                    y_axis_title=_fmt_axis(cols[y]),
                    chart_data={"labels": [str(row[label_idx]) for row in rows],
                                "values": [_to_number(row[y]) for row in rows]},
                    confidence=0.65,   # above threshold, below a canonical pairing
                )
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
        # "text" is deliberately distinct from "categorical" — recommend()'s
        # categorical_idx only collects kind == "categorical", so a free-text
        # column is naturally excluded from ever becoming the chart's
        # dimension, the same way an identifier is excluded via is_id. It may
        # still appear in the markdown table, just never charted.
        if self._looks_like_free_text(col_name, sample):
            return "text"
        return "categorical"

    @staticmethod
    def _is_identifier(col_name: str) -> bool:
        name = str(col_name)
        lname = name.lower()
        if lname in _IDENTIFIER_NAMES or lname.endswith(_IDENTIFIER_SUFFIXES):
            return True
        # camelCase identifier convention from non-Django sources (NoSQL/
        # federated/external schemas commonly use "AccountID"/"customerId"/
        # "buildId" with no underscore) — 2026-07-17, a real observed gap:
        # "accountid".endswith("_id") is False (no underscore), so these slipped
        # through and got charted as a category/measure. Checked on the
        # ORIGINAL case and requires the Id/ID segment to follow a lowercase/
        # digit boundary, so this never false-positives on ordinary lowercase
        # English words that happen to end in "id" (paid, valid, grid, hybrid,
        # android, ...) — those are never capitalized mid-word in real data.
        return bool(_CAMEL_ID_SUFFIX_RE.search(name))

    @staticmethod
    def _looks_like_date(v: Any) -> bool:
        if hasattr(v, "isoformat"):  # datetime/date objects
            return True
        return isinstance(v, str) and bool(_DATE_RE.match(v))

    @staticmethod
    def _looks_like_free_text(col_name: str, sample: list) -> bool:
        name_lower = col_name.lower()
        if any(hint in name_lower for hint in _TEXT_NAME_HINTS):
            return True
        str_values = [str(v) for v in sample]
        avg_len = sum(len(v) for v in str_values) / len(str_values)
        return avg_len > _TEXT_AVG_LEN_THRESHOLD

    # --- chart builders ------------------------------------------------------

    def _category_numeric(self, cols: list, rows: list, cat_idx: int, val_idx: int) -> list[VisualizationSpec]:
        # SQL upstream doesn't guarantee GROUP BY on the category column, so the
        # same category name can appear across multiple rows — sum them here
        # rather than plotting one slice/bar per raw row.
        totals: dict[str, float] = {}
        for row in rows:
            if not _is_numeric(row[val_idx]):
                continue
            name = str(row[cat_idx])
            totals[name] = totals.get(name, 0) + _to_number(row[val_idx])
        pairs = list(totals.items())
        if len(pairs) < 2:
            return []  # a single category isn't a chart — never force one

        # A pie slice can't represent a negative share of a whole (loss/refund/
        # net-change data) — bar handles negative values fine (a bar below the
        # axis), a pie cannot. Computed once, applied to every branch below.
        has_negative = any(value < 0 for _, value in pairs)

        title = f"{_fmt_axis(cols[val_idx])} by {_fmt_axis(cols[cat_idx])}"
        x_title, y_title = _fmt_axis(cols[cat_idx]), _fmt_axis(cols[val_idx])
        if len(pairs) <= _MAX_PIE_SLICES:
            labels = [name for name, _ in pairs]
            values = [value for _, value in pairs]
            bar = VisualizationSpec(
                type=ChartType.BAR, title=title, x_axis_title=x_title, y_axis_title=y_title,
                chart_data={"labels": labels, "values": values}, confidence=0.9,
            )
            if has_negative:
                return [bar]
            slices = [{"name": name, "value": value} for name, value in pairs]
            pie = VisualizationSpec(type=ChartType.PIE, title=title, chart_data={"slices": slices},
                                    confidence=0.9)
            # Bar is an equally valid read of the SAME totals — same
            # confidence as pie (identical data, different geometry), not a
            # separately-justified guess. Pie stays first (today's single-
            # chart behavior — any caller that only reads specs[0] sees no
            # change at all); bar is the new additive second chart.
            return [pie, bar]

        # Long tail: keep the top N by value, collapse the rest into "Other"
        # rather than dropping the chart entirely — this is what makes bar/pie
        # work for ANY category count, not just small ones. Bar-only here
        # (unchanged) — a pie with this many slices is genuinely unreadable,
        # so this stays a single-chart case on purpose (see the architecture
        # review: "many categories -> bar only").
        ranked = sorted(pairs, key=lambda p: p[1], reverse=True)
        top, rest = ranked[:_TOP_N_CATEGORIES], ranked[_TOP_N_CATEGORIES:]
        slices = [{"name": name, "value": value} for name, value in top]
        if rest:
            slices.append({"name": "Other", "value": sum(value for _, value in rest)})

        if len(slices) <= _MAX_PIE_SLICES and not has_negative:
            return [VisualizationSpec(type=ChartType.PIE, title=title, chart_data={"slices": slices},
                                      confidence=0.75)]

        labels = [s["name"] for s in slices]
        values = [s["value"] for s in slices]
        return [VisualizationSpec(
            type=ChartType.BAR, title=title, x_axis_title=x_title, y_axis_title=y_title,
            chart_data={"labels": labels, "values": values}, confidence=0.7,
        )]

    @staticmethod
    def _line(cols: list, rows: list, x_idx: int, y_idx: int) -> VisualizationSpec:
        # SQL upstream doesn't guarantee ORDER BY on the temporal column, so
        # rows can arrive in arbitrary DB order — sort here or the line zig-zags.
        ordered = sorted(rows, key=lambda row: (row[x_idx] is None, row[x_idx]))
        non_null_x = sum(1 for row in ordered if row[x_idx] is not None)
        confidence = 0.9 if non_null_x == len(ordered) and len(ordered) >= 3 else 0.7
        return VisualizationSpec(
            type=ChartType.LINE, title=f"{_fmt_axis(cols[y_idx])} over {_fmt_axis(cols[x_idx])}",
            x_axis_title=_fmt_axis(cols[x_idx]), y_axis_title=_fmt_axis(cols[y_idx]),
            chart_data={"labels": [str(row[x_idx]) for row in ordered],
                       "values": [_to_number(row[y_idx]) for row in ordered]},
            confidence=confidence,
        )

    @staticmethod
    def _bar_over_time(cols: list, rows: list, x_idx: int, y_idx: int, confidence: float) -> VisualizationSpec:
        """Bar-chart rendering of the SAME (labels, values) data _line() plots
        — an equally valid read of a temporal+numeric result, not a separate
        recommendation. `confidence` is passed in from the caller's already-
        computed line confidence rather than recomputed here, since it's the
        identical underlying data."""
        ordered = sorted(rows, key=lambda row: (row[x_idx] is None, row[x_idx]))
        return VisualizationSpec(
            type=ChartType.BAR, title=f"{_fmt_axis(cols[y_idx])} over {_fmt_axis(cols[x_idx])}",
            x_axis_title=_fmt_axis(cols[x_idx]), y_axis_title=_fmt_axis(cols[y_idx]),
            chart_data={"labels": [str(row[x_idx]) for row in ordered],
                       "values": [_to_number(row[y_idx]) for row in ordered]},
            confidence=confidence,
        )

    @staticmethod
    def _combo(cols: list, rows: list, dim_idx: int, hist_idx: int, line_idx: int) -> VisualizationSpec | None:
        triples = [
            (str(row[dim_idx]), _to_number(row[hist_idx]), _to_number(row[line_idx]))
            for row in rows if _is_numeric(row[hist_idx]) and _is_numeric(row[line_idx])
        ]
        if len(triples) < 2:
            return None
        # A combo chart is inherently a more specific/riskier read than a plain
        # bar/line — scale confidence by how much of the result set actually
        # produced a usable (dim, measure1, measure2) triple.
        coverage = len(triples) / len(rows) if rows else 0.0
        confidence = 0.8 if coverage >= 0.8 else 0.6
        return VisualizationSpec(
            type=ChartType.LINE_HISTOGRAM,
            title=f"{_fmt_axis(cols[hist_idx])} and {_fmt_axis(cols[line_idx])} by {_fmt_axis(cols[dim_idx])}",
            x_axis_title=_fmt_axis(cols[dim_idx]), histogram_title=_fmt_axis(cols[hist_idx]),
            line_title=_fmt_axis(cols[line_idx]),
            chart_data={"labels": [t[0] for t in triples],
                       "histogram_values": [t[1] for t in triples],
                       "line_values": [t[2] for t in triples]},
            confidence=confidence,
        )
