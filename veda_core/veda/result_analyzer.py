# veda/result_analyzer.py
# VEDA — deterministic Result Analyzer (Insight Engine, Phase 2/3).
#
# Turns (sql, cols, rows) — already executed, no new query — into a canonical
# InsightContext: the single object everything downstream of SQL execution
# (Insight Engine, visualization validation, explainability) consumes. No
# LLM. No new SQL. Connector-agnostic by construction: it only ever sees the
# same (cols, rows) shape every connector already normalizes to.
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Column kind inference — a small, self-contained copy of
# apps/chat/visualization.py's VisualizationRecommender._infer_kind. NOT a
# shared import: apps/ (Django api tier) never imports veda_core directly
# (apps/query/inference_client.py's own docstring), so the two tiers each
# carry their own copy of this ~15-line structural helper rather than one
# importing across that boundary. Keep the two in sync if either changes.
# ---------------------------------------------------------------------------
_DATE_RE = re.compile(r"^\d{4}-\d{2}(-\d{2})?")
_TEMPORAL_NAME_HINTS = ("date", "month", "year", "week", "day", "time", "period", "quarter")


def _is_numeric(v: Any) -> bool:
    return isinstance(v, (int, float, Decimal)) and not isinstance(v, bool)


def _to_number(v: Any):
    """Decimal isn't directly comparable to float (raises TypeError) and isn't
    a JSON number — normalize before min/max or storage, same as
    apps/chat/visualization.py's _to_number."""
    return float(v) if isinstance(v, Decimal) else v


def _looks_like_date(v: Any) -> bool:
    if hasattr(v, "isoformat"):
        return True
    return isinstance(v, str) and bool(_DATE_RE.match(v))


def infer_column_kind(col_name: str, values: List[Any]) -> str:
    """'temporal' | 'numeric' | 'categorical' — structural inference only."""
    name_lower = str(col_name).lower()
    if any(hint in name_lower for hint in _TEMPORAL_NAME_HINTS):
        return "temporal"
    sample = [v for v in values[:20] if v is not None]
    if not sample:
        return "categorical"
    if all(_looks_like_date(v) for v in sample):
        return "temporal"
    if all(_is_numeric(v) for v in sample):
        return "numeric"
    return "categorical"


# ---------------------------------------------------------------------------
# Semantic column ROLE — Identifier | Dimension | Measure | Date | Boolean |
# Text. One layer above the structural `kind` above: `kind` says "this LOOKS
# numeric"; `role` says "and here's what that numeric-looking column actually
# IS" — the distinction that keeps an identifier (numeric-looking, but never
# a measure) out of a chart. Primary signal: the ingested semantic model's
# semantic_type (config.SEMANTIC_TYPES — already computed once, deterministically,
# by ingestion/semantic_type_inference.py; reused here, never recomputed).
# Falls back to structural/name heuristics only when no semantic-model entry
# exists for this result column (NoSQL, Tier-2 joins, aliased/computed columns).
# ---------------------------------------------------------------------------
_SEMANTIC_TYPE_TO_ROLE = {
    "IDENTIFIER": "identifier",
    "TEMPORAL":   "date",
    "CATEGORY":   "dimension",
    "METRIC":     "measure",
    "MONETARY":   "measure",
    "FREE_TEXT":  "text",
}

# Mirrors config.IDENTIFIER_SUFFIXES (the same heuristic
# ingestion/semantic_type_inference.py uses at ingestion time) — a small
# literal copy rather than an import so this module carries no ingestion-time
# dependency; keep the two in sync if either changes.
_IDENTIFIER_SUFFIXES = ("_id", "_uuid", "_key", "_no", "_number", "_num", "_code", "_ref")
_IDENTIFIER_NAMES = ("id", "uuid", "guid")
_BOOLEAN_NAME_PREFIXES = ("is_", "has_", "was_", "can_")
_TEXT_NAME_HINTS = ("email", "notes", "description", "remarks", "comment", "address", "bio")
_TEXT_AVG_LEN_THRESHOLD = 40   # avg sampled value length above this reads as free text, not a category


def _looks_like_identifier(col_name: str) -> bool:
    name = str(col_name).lower()
    return name in _IDENTIFIER_NAMES or name.endswith(_IDENTIFIER_SUFFIXES)


def _looks_like_boolean(col_name: str, values: List[Any]) -> bool:
    name = str(col_name).lower()
    if name.startswith(_BOOLEAN_NAME_PREFIXES):
        return True
    sample = [v for v in values[:20] if v is not None]
    return bool(sample) and all(isinstance(v, bool) for v in sample)


def classify_column_role(col_name: str, values: List[Any], kind: str,
                         semantic_type: Optional[str] = None) -> str:
    """Deterministic — identifier | dimension | measure | date | boolean | text.
    Never an LLM call."""
    if semantic_type:
        role = _SEMANTIC_TYPE_TO_ROLE.get(str(semantic_type).upper())
        if role:
            return role
    if _looks_like_identifier(col_name):
        return "identifier"
    if _looks_like_boolean(col_name, values):
        return "boolean"
    if kind == "temporal":
        return "date"
    if kind == "numeric":
        return "measure"
    # categorical — a short, low-cardinality DIMENSION vs. free-form TEXT
    if any(h in str(col_name).lower() for h in _TEXT_NAME_HINTS):
        return "text"
    sample = [str(v) for v in values[:20] if v is not None]
    if sample and (sum(len(s) for s in sample) / len(sample)) > _TEXT_AVG_LEN_THRESHOLD:
        return "text"
    return "dimension"


# ---------------------------------------------------------------------------
# Result-shape classification — same canonical-shape logic
# query/result_explainer.py's template_answer() already applies (empty /
# single scalar / single row / narrative). Exposed here as a named function
# for result_analyzer's own use; result_explainer keeps its own inline
# checks unchanged (no cross-import — same reasoning as infer_column_kind).
# ---------------------------------------------------------------------------
def classify_result_type(row_count: int, columns: List[str]) -> str:
    if row_count == 0:
        return "empty"
    if row_count == 1 and columns and len(columns) == 1:
        return "scalar"
    if row_count == 1:
        return "single_row"
    return "multi_row"


@dataclass
class ColumnStat:
    """Mirrors ingestion/data_profiler.ColumnProfile's shape for familiarity —
    computed in pure Python over already-fetched rows, not a new DB scan."""
    name:           str
    kind:           str                 # temporal | numeric | categorical
    role:           str = "dimension"   # identifier | dimension | measure | date | boolean | text
    semantic_type:  Optional[str] = None
    null_count:     int = 0
    distinct_count: int = 0
    min:            Optional[Any] = None
    max:            Optional[Any] = None
    avg:            Optional[float] = None
    median:         Optional[float] = None
    top_values:     List[Any] = field(default_factory=list)


# Result "shape" — a finer-grained label than result_type, used to steer both
# the Insight Engine's prompt (ranking/trend/distribution/concentration
# examples need to know which one this is) and visualization validation.
# Priority order matters: a result can technically match more than one signal
# (e.g. a ranked trend), so this picks the most SPECIFIC applicable label.
# SCALAR covers every non-multi_row shape (empty/scalar/single_row) — none of
# those are chartable or need shape-specific phrasing beyond "just answer it".
RESULT_SHAPES = ("SCALAR", "DETAIL_TABLE", "RANKING", "TREND", "GROUPED",
                 "DISTRIBUTION", "PIVOT")


def detect_result_shape(result_type: str, dimensions: List[str], measures: List[str],
                        column_stats: List["ColumnStat"], limit: Optional[int],
                        orderings: List[Tuple[str, bool]],
                        aggregations: Optional[List[Tuple[str, Optional[str]]]] = None) -> str:
    if result_type != "multi_row":
        return "SCALAR"
    kinds = {s.name: s.kind for s in column_stats}
    has_temporal = any(kinds.get(d) == "temporal" for d in dimensions) or \
        any(k == "temporal" for k in kinds.values())
    has_numeric_measure = bool(measures) or any(k == "numeric" for k in kinds.values())
    agg_funcs = {f for f, _ in (aggregations or [])}
    is_count_only = bool(agg_funcs) and agg_funcs <= {"COUNT"}

    if len(measures) >= 2 and dimensions:
        return "PIVOT"
    if has_temporal and has_numeric_measure:
        return "TREND"
    if limit is not None and orderings:
        return "RANKING"
    if dimensions and is_count_only:
        return "DISTRIBUTION"          # "how many X per category" — a frequency breakdown
    if dimensions and has_numeric_measure:
        return "GROUPED"               # "revenue by category" — a real measure per category
    return "DETAIL_TABLE"              # a raw listing — no grouping or aggregation at all


@dataclass
class InsightContext:
    """The canonical object everything after SQL execution consumes."""
    question:       str
    sql:            str
    table:          Optional[str]
    result_type:    str
    row_count:      int
    columns:        List[str]
    entities:       List[str]
    dimensions:     List[str]
    measures:       List[str]
    filters:        List[Tuple[str, str, Optional[str]]]
    orderings:      List[Tuple[str, bool]]
    limit:          Optional[int]
    distinct:       bool
    column_stats:   List[ColumnStat]
    sample_rows:    List[dict]
    semantic_model: Optional[dict] = None
    connector_type: str = "relational"
    query_intent:   Optional[str] = None          # SIMPLE | MULTI_TABLE | AGGREGATE (query_engine.IntentDetector)
    confidence_inputs: Dict[str, float] = field(default_factory=dict)
    result_shape:   str = "SCALAR"                 # see RESULT_SHAPES


def _column_stats(columns: List[str], rows: List[dict], max_rows: int,
                  table: Optional[str] = None, semantic_model: Optional[dict] = None) -> List[ColumnStat]:
    sample = rows[:max_rows]
    cols_meta = (semantic_model or {}).get("columns", {}) if (semantic_model and table) else {}
    stats = []
    for col in columns:
        values = [r.get(col) for r in sample]
        non_null = [v for v in values if v is not None]
        kind = infer_column_kind(col, non_null)
        counts = Counter(str(v) for v in non_null)
        top = [v for v, _ in counts.most_common(5)]
        numeric_vals = [_to_number(v) for v in non_null if _is_numeric(v)]
        meta = cols_meta.get(f"{table}.{col}") if table else None
        semantic_type = (meta.get("semantic_type") or meta.get("analytics_role")) if meta else None
        role = classify_column_role(col, non_null, kind, semantic_type)
        stats.append(ColumnStat(
            name=col, kind=kind, role=role, semantic_type=semantic_type,
            null_count=len(values) - len(non_null),
            distinct_count=len(counts),
            min=min(numeric_vals) if numeric_vals else None,
            max=max(numeric_vals) if numeric_vals else None,
            avg=round(statistics.fmean(numeric_vals), 4) if numeric_vals else None,
            median=statistics.median(numeric_vals) if numeric_vals else None,
            top_values=top,
        ))
    return stats


def analyze_result(
    question:          str,
    sql:                str,
    columns:            List[str],
    rows:               List[dict],
    sm:                 Optional[dict] = None,
    table:              Optional[str] = None,
    max_rows:           int = 200,
    connector_type:     str = "relational",
    query_intent:       Optional[str] = None,
    confidence_inputs:  Optional[Dict[str, float]] = None,
    params:             Optional[List[Any]] = None,
) -> InsightContext:
    """Build the canonical InsightContext from an already-executed result.
    No LLM call, no new SQL query — reuses veda.business_explain.extract_sql_facts
    for the AST-derived query_analysis facts (dimensions/measures/filters/etc.).

    `query_intent` / `confidence_inputs`: metadata the CALLER already computed
    upstream (e.g. veda/pipeline.py's IntentDetector classification, anchor/join
    selection confidence) — reused here, never recomputed. Both optional and
    default to None/{} so existing callers are unaffected.

    `params`: the bound values `sql`'s %s placeholders were parameterized to
    (veda/validation.py's validate_and_parameterize) — without these, filter
    values in the derived facts come back None (see business_explain._extract)."""
    from veda.business_explain import extract_sql_facts

    facts = extract_sql_facts(sql or "", params=params)
    row_count = len(rows)
    result_type = classify_result_type(row_count, columns)
    stats = _column_stats(columns, rows, max_rows, table=table, semantic_model=sm)

    measures = [col for _, col in facts["aggregations"] if col]
    dimensions = list(facts["groupings"])
    result_shape = detect_result_shape(result_type, dimensions, measures, stats,
                                       facts["limit"], facts["orderings"],
                                       aggregations=facts["aggregations"])

    return InsightContext(
        question=question, sql=sql or "", table=table,
        result_type=result_type, row_count=row_count, columns=list(columns),
        entities=facts["entities"], dimensions=dimensions, measures=measures,
        filters=facts["filters"], orderings=facts["orderings"],
        limit=facts["limit"], distinct=facts["distinct"],
        column_stats=stats, sample_rows=rows[:max_rows], semantic_model=sm,
        connector_type=connector_type, query_intent=query_intent,
        confidence_inputs=dict(confidence_inputs or {}), result_shape=result_shape,
    )


# ---------------------------------------------------------------------------
# Chart recommendation confidence — deterministic, reused by BOTH the query
# tier's SLM-suggestion validator (query/result_explainer.py's
# validate_visualization) and, in principle, any other consumer of an
# InsightContext. Never an LLM self-report.
#
# PIVOT deliberately has no canonical chart: the existing frontend contract
# (apps/chat/visualization.py's ChartType) only supports bar/line/pie/
# line_histogram — no heatmap. Recommending one would require a response-
# contract change the mandate forbids, so a PIVOT result gets table-only,
# never a wrong (or new-type) chart.
# ---------------------------------------------------------------------------
CANONICAL_CHART_FOR_SHAPE = {
    "RANKING":      "bar",
    "TREND":        "line",
    "DISTRIBUTION": "pie",
    "GROUPED":      "bar",
}


def chart_confidence(shape: str, chart_type: str, dim_role: Optional[str],
                     measure_role: Optional[str]) -> float:
    """0.0-1.0. Hard zero whenever an identifier is involved (an identifier is
    never a valid chart axis, regardless of shape) or the shape has no
    canonical chart (SCALAR/DETAIL_TABLE/PIVOT). Full confidence for the
    canonical shape->chart pairing with valid roles; a reduced but non-zero
    score for a plausible-but-not-canonical categorical pairing."""
    if dim_role == "identifier" or measure_role == "identifier":
        return 0.0
    canonical = CANONICAL_CHART_FOR_SHAPE.get(shape)
    if canonical is None:
        return 0.0
    if chart_type == canonical:
        return 0.95 if measure_role == "measure" else 0.8
    if chart_type in ("bar", "pie") and shape in ("GROUPED", "DISTRIBUTION", "RANKING"):
        return 0.6
    return 0.3
