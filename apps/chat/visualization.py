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


_INF = float("inf")


def _to_number(v: Any):
    """psycopg2 returns Decimal for NUMERIC/SUM/AVG columns — Decimal isn't a
    JSON number, so anything headed into chart_data is normalized to float."""
    return float(v) if isinstance(v, Decimal) else v


def _is_numeric(v: Any) -> bool:
    """NaN/Infinity are excluded deliberately: they are unplottable, and
    json.dumps emits them as the bare tokens NaN/Infinity, which are NOT valid
    JSON — a strict frontend JSON.parse rejects the whole response. A NaN
    measure is therefore treated exactly like a NULL one (row skipped), on
    every chart path, rather than leaking into chart_data."""
    if isinstance(v, bool) or not isinstance(v, (int, float, Decimal)):
        return False
    return not (isinstance(v, (float, Decimal)) and (v != v or v in (_INF, -_INF)))


_DATE_RE = re.compile(r"^\d{4}-\d{2}(-\d{2})?")
_TEMPORAL_NAME_HINTS = ("date", "month", "year", "week", "day", "time", "period", "quarter")
_NAME_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _has_temporal_name_hint(name_lower: str) -> bool:
    """Whole-TOKEN match, not a raw substring search (2026-09-28 fix, mirrors
    result_analyzer.py's identical fix — see that copy's docstring): a plain
    `hint in name_lower` check treated "month" as present in "expected_MONTHly_
    rent" (a rent amount), misclassifying it as temporal. "monthly" tokenizes
    to ["monthly"] (no match); "created_date" still tokenizes to ["created",
    "date"] (still matches)."""
    return any(t in _TEMPORAL_NAME_HINTS for t in _NAME_TOKEN_RE.split(name_lower))


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
# Bar-chart category cap: beyond this the long tail collapses into one bucket (+1 bar).
# Was 9, which bucketed a 12-category result down to 10 bars — hiding three named
# categories to save two slots, on a chart that was perfectly readable at 12. A bar
# chart stays legible to roughly 20 bars; a PIE does not, which is what _MAX_PIE_SLICES
# is for. These are two different limits and must not be conflated.
_TOP_N_CATEGORIES = 19  # + 1 bucket for the long tail = 20 bars

# An aggregate whose values can be ADDED UP across rows. A long-tail bucket and the
# duplicate-row roll-up below are both sums, so they are only meaningful for these:
# summing averages, medians, rates or shares produces a number that means nothing.
_ADDITIVE_AGGS = frozenset({"COUNT", "SUM", "TOTAL"})
# Fallback when the engine didn't tell us the aggregate (federated results, un-aliased
# SQL): the measure column's OWN name. Deliberately narrow — a name that doesn't match
# is treated as additive, i.e. exactly today's behaviour.
_NON_ADDITIVE_NAME_HINTS = ("avg", "average", "mean", "median", "rate", "ratio",
                            "pct", "percent", "share")


def _agg_key(name: object) -> str:
    """Comparison key that treats a plural as the same word — "Others" (a real category
    in ticketing/CRM schemas) must not be considered distinct from the synthetic
    "Other" bucket just because of one letter."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower()).rstrip("s")


def _measure_is_additive(col_name: str, analytics: dict | None) -> bool:
    """Can this measure's values legitimately be summed across rows? The engine's own
    SQL-derived aggregate (analytics.measure_aggregates, keyed by SELECT alias = the
    result column name) is authoritative; the column name is the fallback."""
    agg = ((analytics or {}).get("measure_aggregates") or {}).get(col_name)
    if agg:
        return str(agg).upper() in _ADDITIVE_AGGS
    words = re.sub(r"[^a-z0-9]+", " ", str(col_name).lower())
    return not any(h in words for h in _NON_ADDITIVE_NAME_HINTS)


def _tail_label(existing_names: list, rest_count: int) -> str:
    """A name for the collapsed long tail that cannot be mistaken for a real category.
    A genuine catch-all category named "Other"/"Others" is common, and a chart carrying
    a real "Others" bar of 6 beside a synthetic "Other" bar of 6 is unreadable — which
    is exactly what shipped before this."""
    taken = {_agg_key(n) for n in existing_names}
    for cand in ("Other", "All other categories"):
        if _agg_key(cand) not in taken:
            return cand
    return f"All other ({rest_count} categories)"

# Row-preserving listing charts (RANKING / DETAIL_TABLE) — see _row_listing.
# A listing is NOT a breakdown: every row is its own entity, so it is charted
# one-bar-per-row in the SQL's own order. The cap truncates the tail (keeping
# the rows the ORDER BY put first — for "cheapest N" that IS the answer) rather
# than folding it into an "Other" bucket, which would hide individual results.
_MAX_ROW_BARS = 25
_LISTING_SHAPES = ("RANKING", "DETAIL_TABLE")

# Below this, no chart is returned at all — a table-only response is always
# safer than a low-confidence or borderline-meaningless chart.
_CONFIDENCE_THRESHOLD = 0.6


# --- silent truncation disclosure (2026-09-30, merged from a parallel branch) -
#
# `analytics["is_truncated"]` (result_analyzer.compute_result_completeness,
# same shared signal table_rendering.py's notice already reads) says `rows`
# here is itself only a PAGE of a larger result the backend/SQL cut off. Every
# chart branch below is built from that page; some already caption this
# themselves (the row-listing, which knows its own more precise "first N of M"
# count) — this is the SAFETY NET for every branch that doesn't: it only fills
# in a caption where one is still missing, so a branch with its own, more
# specific wording is never overwritten.
def _result_truncated(analytics: Any) -> bool:
    try:
        return bool(analytics.get("is_truncated"))
    except AttributeError:      # None, or an analytics payload that isn't a dict
        return False


def _partial_sub_title(shown: int, page: int) -> str:
    """Caption for a chart drawn from a silently-truncated result. Says the two
    things the reader needs: how much is on screen, and that the real result is
    bigger. Deliberately NOT "First N of M rows" — that phrasing is what made
    the page look like the population."""
    if shown < page:
        return (f"Partial data — first {shown} of the {page:,} rows returned; "
                "the full result is larger")
    return f"Partial data — the {page:,} rows returned; the full result is larger"


def _disclose_truncation(specs: list, page_rows: int, truncated: bool) -> list:
    """Caption every chart drawn from a truncated page that doesn't already
    carry its own (more precise) sub_title."""
    if not truncated:
        return specs
    for spec in specs:
        if not spec.sub_title:
            spec.sub_title = _partial_sub_title(page_rows, page_rows)
    return specs


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
        """Thin public wrapper around `_recommend_specs()` (2026-09-30, merged
        from a parallel branch's independent truncation-disclosure work): a
        defensive row-length guard (a row shorter than `cols` would raise
        IndexError deep inside a builder — `services._build_visualizations`
        has no try/except around this call, so one malformed row would take
        down the whole chat turn, not just the chart) and the ONE choke point
        that captions every returned chart drawn from a truncated page,
        including the branches that don't already caption themselves. Kept
        separate from `_recommend_specs` so that method's own early returns
        stay simple — the caption is applied once here, after, not threaded
        through every return point."""
        rows = [row for row in rows if len(row) >= len(cols)]
        if not rows:
            return []
        if not isinstance(analytics, dict):
            analytics = None
        truncated = _result_truncated(analytics)
        specs = self._recommend_specs(cols, rows, analytics)
        return _disclose_truncation(specs, len(rows), truncated)

    def _recommend_specs(self, cols: list, rows: list, analytics: dict | None = None) -> list[VisualizationSpec]:
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
                # role checked FIRST for every case it covers (2026-09-28 fix) —
                # previously `kind == "temporal"` was checked ahead of
                # role == "dimension"/"measure", so a structural false-positive
                # (infer_column_kind's temporal-name-hint substring match, e.g.
                # "expected_MONTHly_rent" containing "month") silently overrode a
                # CORRECT role == "measure" from the semantic model — the exact
                # contradiction this comment already claimed didn't happen. Live-
                # confirmed: "highest expected monthly rent" charted the measure
                # itself as the X-axis dimension against two unrelated columns.
                # `kind`/`date` are now only consulted once role has had its say.
                if role == "text":
                    return "text"
                if role == "dimension":
                    return "categorical"
                if role == "measure":
                    return "numeric"
                if role == "date" or kind == "temporal":
                    return "temporal"
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

        # Query-relevance reordering (2026-09-28, TABLE_QUERY_COLUMN_FILTER_ENABLED):
        # every `[0]`-picking branch below takes "the first candidate in SELECT
        # order" as the chart axis — which is exactly the same over-selection bug
        # the table view had (an unrelated column the SQL kept "just in case"
        # sorts ahead of the one the question actually named, e.g. cols=
        # [is_gated, project_name, corner_property, ...building_name, carpet_area]
        # for "show properties where is gated is true" charted on the WHERE
        # predicate's own boolean instead of the entity's display column). Reusing
        # result_analyzer's own `query_relevant_columns` (analyze_result's
        # recommended_projection recompute, same signal the table narrowing
        # uses) to REORDER — never filter — each pool: a query-relevant
        # candidate goes first when one exists, the full pool is kept
        # otherwise so an un-flagged/empty signal (flag off, federated result)
        # leaves this a no-op and every candidate remains chartable.
        _relevant = set((analytics or {}).get("query_relevant_columns") or ())

        def _prefer_relevant(idx_list: list) -> list:
            if not _relevant:
                return idx_list
            preferred = [i for i in idx_list if cols[i] in _relevant]
            return preferred + [i for i in idx_list if i not in preferred] if preferred else idx_list

        numeric_idx = _prefer_relevant(numeric_idx)
        temporal_idx = _prefer_relevant(temporal_idx)
        categorical_idx = _prefer_relevant(categorical_idx)

        # A MEASURE (the numeric axis specifically — what actually gets summed/
        # plotted) must be query-relevant when a relevance signal exists at all;
        # a dimension mismatch is more forgivable, but a chart whose VALUE axis
        # is something nobody asked about is the misleading case (2026-09-30,
        # no hardcoded column names — purely reusing query_relevant_columns).
        # Real observed bugs this closes, across every branch that reads
        # numeric_idx (combo/TREND/GROUPED all check it — one shared fix, not
        # four patched separately): "possession status of each project" (only
        # numeric column: longitude) charted "Longitude by Possession"; "show
        # properties that have power backup" (only numeric column: carpet_area)
        # charted "Carpet Area by Power Backup". Only fires when `_relevant` is
        # non-empty AND genuinely none of the candidates match it — a pool with
        # no signal at all (flag off, federated result) is completely
        # unaffected, same as `_prefer_relevant` above.
        if _relevant and numeric_idx and not any(cols[i] in _relevant for i in numeric_idx):
            numeric_idx = []

        # Temporal-over-categorical priority is only safe to keep UNCONDITIONAL
        # when there's no relevance signal to check it against (2026-09-30, real
        # observed bug — no hardcoded column names anywhere in this fix, purely
        # reusing the SAME query_relevant_columns signal the reordering above
        # already trusts): "What is the convenience fee for EACH PAYMENT TYPE?"
        # (cols: convenience_fee, name, id, created_by_id, created_at) charted
        # "Convenience Fee over Created At" — a temporal column NOBODY asked
        # about (created_at isn't query-relevant) outranked the categorical
        # column the query's own wording named (payment type -> "name", the
        # table's resolved display column, which IS query-relevant). A
        # query-relevant categorical only overrides a NON-relevant temporal —
        # when neither or both are relevant, or there's no signal at all
        # (flag off / federated), today's temporal-first default is unchanged.
        _temporal_not_relevant = bool(_relevant and temporal_idx and cols[temporal_idx[0]] not in _relevant)
        _categorical_is_relevant = bool(_relevant and categorical_idx and cols[categorical_idx[0]] in _relevant)
        _prefer_categorical_dimension = _temporal_not_relevant and _categorical_is_relevant

        dimension_idx = (categorical_idx[:1] if _prefer_categorical_dimension
                        else (temporal_idx[:1] or categorical_idx[:1]))

        # A listing (RANKING / DETAIL_TABLE) is charted row-per-row, BEFORE any of
        # the aggregating branches below can touch it. Those branches exist for
        # breakdowns (GROUPED/DISTRIBUTION/TREND), where summing per category is
        # correct; on a raw listing every row is a distinct entity, and summing
        # them, re-sorting them by value, or bucketing the tail into "Other"
        # produces a chart that contradicts the table it sits next to.
        if (analytics or {}).get("result_shape") in _LISTING_SHAPES:
            spec = self._row_listing(cols, rows, kinds, is_id, analytics)
            if spec is not None:
                return [spec]

        # A dimension with TWO measures is the frontend's line_histogram: a
        # combo chart correlating two metrics over the same axis (e.g. sales
        # volume vs. conversion rate by month). Takes priority — it's a more
        # specific, more informative match than either measure charted alone.
        if dimension_idx and len(numeric_idx) >= 2:
            combo = self._combo(cols, rows, dimension_idx[0], numeric_idx[0], numeric_idx[1])
            if combo is not None and combo.confidence >= _CONFIDENCE_THRESHOLD:
                return [combo]

        if temporal_idx and numeric_idx and len(rows) > 1 and not _prefer_categorical_dimension:
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
            # Try each (dimension, measure) pairing in pool order until one
            # actually charts (2026-09-28) — a pairing legitimately produces no
            # chart when the dimension has a single value/no real breakdown
            # (_category_numeric's own "never force a chart" rule). Previously
            # only categorical_idx[0]/numeric_idx[0] was ever tried, so an
            # unchartable first candidate silently gave up on a result that a
            # later candidate could chart fine. This never picks a WORSE chart
            # than before (the [0][0] pairing, when chartable, is still tried
            # first) — it only adds a fallback where there was none, which the
            # new query-relevance reordering above needs: a query-relevant
            # column now goes first, and must not regress a result that used
            # to chart via a later, non-relevant column.
            specs: list[VisualizationSpec] = []
            for _cat_i in categorical_idx:
                for _num_i in numeric_idx:
                    specs = self._category_numeric(cols, rows, _cat_i, _num_i, analytics)
                    if specs:
                        break
                if specs:
                    break
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
        if _has_temporal_name_hint(name_lower):
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

    # --- public chart builders ------------------------------------------------
    #
    # recommend() decides WHICH columns to chart; these build the chart_data for an
    # already-chosen (dimension, measure) pairing. They are public because
    # services.py's _spec_from_suggestion reuses them to render the query tier's
    # own validated column suggestion — the suggestion names the columns, this
    # module still owns how the payload is built, so the two paths can never drift.
    # (Previously services.py called the private `_line`/`_category_numeric`
    # directly, reaching across the module's encapsulation boundary.)
    #
    # The underscore-prefixed names remain as aliases: they are the long-standing
    # internal entry points and are exercised directly by the visualization tests.

    def build_category_specs(self, cols: list, rows: list, cat_idx: int,
                             val_idx: int, analytics: dict | None = None) -> list[VisualizationSpec]:
        """Chart(s) for a (category, measure) pairing — pie and/or bar. Returns a
        LIST because a small category breakdown supports two equally valid
        renderings of the same totals; empty when the data can't be charted (e.g.
        a single category).

        `analytics`: the engine's own analysis payload, same one `recommend` takes —
        its `measure_aggregates` is what tells this builder whether the measure may be
        summed, and its `is_truncated` drives the truncation caption below (2026-09-30,
        merged from a parallel branch) — the suggestion/candidate fallbacks in
        services.py reach this builder WITHOUT going through recommend()'s own
        wrapper, and a chart is no less misleading for having been picked by the
        query tier instead of by this module."""
        specs = self._category_numeric(cols, rows, cat_idx, val_idx, analytics)
        return _disclose_truncation(specs, len(rows), _result_truncated(analytics))

    def build_line_spec(self, cols: list, rows: list, x_idx: int, y_idx: int,
                        analytics: dict | None = None) -> VisualizationSpec:
        """The line chart for an (x, measure) pairing. Always returns one spec.
        `analytics` is optional — see build_category_specs's own docstring."""
        spec = self._line(cols, rows, x_idx, y_idx)
        return _disclose_truncation([spec], len(rows), _result_truncated(analytics))[0]

    # --- chart builders ------------------------------------------------------

    def _row_listing(self, cols: list, rows: list, kinds: list, is_id: list,
                     analytics: dict | None) -> VisualizationSpec | None:
        """One bar per RESULT ROW, in the result's own order — the chart for a
        listing (RANKING / DETAIL_TABLE), where each row is a separate entity
        rather than a bucket of a whole.

        Three things this deliberately does NOT do, each of which was a real
        observed bug on "cheapest properties on the market" (2026-09-11):
        - no summing of rows that share a label (two listings in the same
          building are two properties, not one at the combined price);
        - no re-sort by value and no "Other" bucket (the question asked for the
          cheapest — a value-DESC top-N leads with the most expensive and hides
          the actual answer inside "Other");
        - no positional column guessing (`numeric_idx[0]` / `categorical_idx[0]`
          picked whatever the SELECT list happened to start with — e.g.
          `cancellation_reason_title`, a column the user never saw in the table).
        """
        measure_idx = self._listing_measure(cols, kinds, is_id, analytics)
        if measure_idx is None:
            return None
        label_idx = self._listing_label(cols, rows, kinds, is_id, measure_idx)
        if label_idx is None:
            return None

        plotted = [row for row in rows if _is_numeric(row[measure_idx])]
        if len(plotted) < 2:
            return None
        # Respect an explicit user ask (2026-09-30, presentation-policy pass): the
        # SQL's own LIMIT (analytics["limit"], from the executed statement's own
        # AST) reflects what the query actually asked for — "top 30" must not get
        # silently cut to 25 bars just because that happens to be this module's
        # generic readability default. Bounded at 100 regardless (hundreds of bars
        # is unreadable no matter how explicit the ask), and never LOWERS today's
        # default — a query with no meaningful LIMIT (None, or one <= the default)
        # behaves exactly as before.
        _row_cap = _MAX_ROW_BARS
        _explicit_limit = (analytics or {}).get("limit")
        if isinstance(_explicit_limit, int) and _MAX_ROW_BARS < _explicit_limit <= 100:
            _row_cap = _explicit_limit
        truncated = len(plotted) > _row_cap
        plotted = plotted[:_row_cap]          # head, not top-N: keeps the ORDER BY's own answer

        # Two rows can legitimately carry the same label (same building, same
        # project). They stay SEPARATE bars; the suffix only keeps the axis
        # labels distinguishable instead of silently overlaying them.
        labels, used = [], set()
        for row in plotted:
            name = "—" if row[label_idx] is None else str(row[label_idx])
            label, n = name, 1
            while label in used:            # the suffixed form can itself already be in
                n += 1                      # the data ("X", "X (2)", "X") — keep going
                label = f"{name} ({n})"
            used.add(label)
            labels.append(label)

        title = f"{_fmt_axis(cols[measure_idx])} by {_fmt_axis(cols[label_idx])}"
        # `len(rows)` in the sub_title below is itself only the FETCHED count — if
        # the fetch was already capped (analytics["is_truncated"]), say so instead
        # of presenting that number as the true total (2026-09-30, shared
        # result-completeness — result_analyzer.compute_result_completeness).
        _fetch_capped = bool((analytics or {}).get("is_truncated"))
        _limit_exceeded = bool((analytics or {}).get("reason") == "FETCH_CAP_EXCEEDED")
        if _limit_exceeded:
            # A KNOWN, deterministic limitation ("top 1001" against a 1000-row
            # fetch cap) — distinct from the open-ended "more may exist" hedge
            # below (2026-09-30, requested-limit gap). Fires regardless of
            # whether this chart's own bar cap also truncated the render.
            _sub = (f"You asked for the top {(analytics or {}).get('requested_limit'):,}, "
                   f"but only up to {(analytics or {}).get('fetch_limit'):,} rows can be "
                   f"fetched in a single result")
        elif truncated and _fetch_capped:
            _sub = (f"First {len(plotted)} of {len(rows)} rows fetched, in result "
                   f"order — more rows may exist beyond this fetch")
        elif _fetch_capped:
            # The backend's own fetch was capped upstream of this chart — even
            # when the bar cap itself never kicks in (e.g. only 6 rows made it
            # through the fetch), those 6 are still a PAGE, not the population
            # (2026-09-30, shared result-completeness).
            _sub = (f"Partial data — the {len(plotted)} row"
                   f"{'s' if len(plotted) != 1 else ''} returned; the full result is larger")
        elif truncated:
            _sub = f"First {len(plotted)} of {len(rows)} rows, in result order"
        else:
            _sub = None
        return VisualizationSpec(
            type=ChartType.BAR, title=title,
            sub_title=_sub,
            x_axis_title=_fmt_axis(cols[label_idx]),
            y_axis_title=_fmt_axis(cols[measure_idx]),
            chart_data={"labels": labels,
                        "values": [_to_number(row[measure_idx]) for row in plotted]},
            confidence=0.85,
        )

    @staticmethod
    def _listing_measure(cols: list, kinds: list, is_id: list, analytics: dict | None) -> int | None:
        """The measure a listing is ABOUT is the one its ORDER BY ranked on — the
        engine ships the SQL's own orderings in analytics, so this is read, not
        guessed. Falls back to a query-relevant numeric column when the result
        carries no usable ordering; NEVER to an arbitrary first-numeric-column
        guess (2026-09-30, real observed bug, no hardcoded column names) —
        "Show properties that have power backup" (cols: power_backup,
        project_name, address, corner_property, building_name, carpet_area; no
        ORDER BY at all) charted "Carpet Area by Project Name": carpet_area is a
        real, structurally-numeric column, just one nobody asked about. A
        misleading chart is worse than no chart, so with no ordering AND no
        query-relevant numeric candidate, this returns None (no chart) rather
        than guessing. When `query_relevant_columns` itself is absent (flag off,
        federated result — no signal to check against at all), the old
        numeric[0] fallback is unchanged, so this never regresses a result that
        had nothing better to go on."""
        numeric = [i for i, k in enumerate(kinds) if k == "numeric" and not is_id[i]]
        if not numeric:
            return None
        by_name = {str(c).lower(): i for i, c in enumerate(cols)}
        for entry in (analytics or {}).get("orderings") or []:
            name = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
            i = by_name.get(str(name).lower().split(".")[-1])
            if i is not None and i in numeric:
                return i
        _relevant = set((analytics or {}).get("query_relevant_columns") or ())
        if not _relevant:
            return numeric[0]
        for i in numeric:
            if cols[i] in _relevant:
                return i
        return None

    @staticmethod
    def _listing_label(cols: list, rows: list, kinds: list, is_id: list,
                       measure_idx: int) -> int | None:
        """The column that NAMES each row. In a listing the identifying label is
        near-unique per row (building/project/property name) while a status,
        reason or type column repeats across rows — so the most distinct
        non-numeric column is the right axis, and a low-cardinality status column
        can no longer win just by appearing first in the SELECT list. Identifiers
        are the last resort (a listing keyed by id still beats no chart — the same
        allowance the RANKING rescue below already makes)."""
        def _distinct(i: int) -> int:
            return len({str(row[i]) for row in rows})

        def _pick(pool: list) -> int | None:
            # Most distinct wins; ties break on SELECT-list order (stable sort).
            return max(pool, key=_distinct) if pool else None

        named = [i for i, k in enumerate(kinds)
                 if i != measure_idx and k != "numeric" and not is_id[i]]
        picked = _pick(named)
        if picked is not None:            # `or` would discard a legitimate index 0
            return picked
        return _pick([i for i in range(len(cols))
                      if i != measure_idx and kinds[i] != "numeric"])

    def _category_numeric(self, cols: list, rows: list, cat_idx: int, val_idx: int,
                          analytics: dict | None = None) -> list[VisualizationSpec]:
        # Whether this measure may be ADDED across rows decides two things below: the
        # duplicate-row roll-up here, and the long-tail bucket further down. Both are
        # sums, and a sum of averages/rates/shares is not a number that means anything.
        additive = _measure_is_additive(cols[val_idx], analytics)
        # SQL upstream doesn't guarantee GROUP BY on the category column, so the
        # same category name can appear across multiple rows — sum them here
        # rather than plotting one slice/bar per raw row.
        totals: dict[str, float] = {}
        duplicates = False
        for row in rows:
            if not _is_numeric(row[val_idx]):
                continue
            name = str(row[cat_idx])
            if name in totals:
                duplicates = True
            totals[name] = totals.get(name, 0) + _to_number(row[val_idx])
        if duplicates and not additive:
            return []  # no honest way to combine an average/rate across rows — no chart
        # Never present a client-side roll-up of TRUNCATED raw data as a complete
        # breakdown (2026-09-30, presentation-policy pass §7 — the highest-priority
        # correctness rule in that spec). `duplicates` already means the SAME
        # category appeared on more than one raw row — i.e. the SQL did NOT group
        # by this category server-side, so this function is doing the summing
        # itself, over whatever rows the backend's fetch cap happened to return.
        # If that fetch was ALSO truncated (analytics["is_truncated"] — shared
        # result-completeness, result_analyzer.compute_result_completeness), these
        # per-category totals are a sum over an arbitrary PARTIAL slice of the true
        # data, not the real totals — e.g. 5,000 true rows, only the first 1,000
        # fetched, "Revenue by Category" quietly computed from just those 1,000.
        # Refusing the chart here (rather than rendering it with a caveat) matches
        # the spec's own preference: "prefer not rendering the chart if the
        # aggregation cannot be trusted." A properly GROUP-BY'd result (no
        # `duplicates` at all — the server already aggregated over the FULL data
        # before any fetch cap applied) is unaffected regardless of is_truncated.
        if duplicates and (analytics or {}).get("is_truncated"):
            return []
        pairs = list(totals.items())
        if len(pairs) < 2:
            return []  # a single category isn't a chart — never force one

        # A pie slice can't represent a negative share of a whole (loss/refund/
        # net-change data) — bar handles negative values fine (a bar below the
        # axis), a pie cannot. Computed once, applied to every branch below.
        has_negative = any(value < 0 for _, value in pairs)
        # A pie is a part-of-whole STATEMENT: its slices always sum to 100% of
        # what's shown. On a silently truncated result (analytics["is_truncated"]
        # — merged from a parallel branch, 2026-09-30) they sum to 100% of one
        # PAGE (e.g. 1,000 of 7,814 true rows) — not merely incomplete, actually
        # WRONG shares, and no caption undoes that the way it does for a bar (a
        # bar claims no denominator, so it stays honest with a caption). Same
        # suppression the negative-value guard already performs.
        no_pie = has_negative or bool((analytics or {}).get("is_truncated"))

        title = f"{_fmt_axis(cols[val_idx])} by {_fmt_axis(cols[cat_idx])}"
        x_title, y_title = _fmt_axis(cols[cat_idx]), _fmt_axis(cols[val_idx])
        if len(pairs) <= _MAX_PIE_SLICES:
            labels = [name for name, _ in pairs]
            values = [value for _, value in pairs]
            bar = VisualizationSpec(
                type=ChartType.BAR, title=title, x_axis_title=x_title, y_axis_title=y_title,
                chart_data={"labels": labels, "values": values}, confidence=0.9,
            )
            if no_pie:
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
        if rest and additive:
            slices.append({"name": _tail_label([n for n, _ in ranked], len(rest)),
                           "value": sum(value for _, value in rest)})
        elif rest:
            # A non-additive measure has no honest single value for the tail, so the
            # tail is DROPPED rather than faked — and the title then has to say that
            # the chart is a subset, or it reads as the whole distribution.
            title = f"{title} (top {len(top)})"

        if len(slices) <= _MAX_PIE_SLICES and not no_pie:
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
