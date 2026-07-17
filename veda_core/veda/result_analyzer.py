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
# camelCase identifier suffix (AccountID/customerId/buildId) — see _looks_like_identifier.
_CAMEL_ID_SUFFIX_RE = re.compile(r"[a-z0-9](Id|ID)$")
_BOOLEAN_NAME_PREFIXES = ("is_", "has_", "was_", "can_")
_TEXT_NAME_HINTS = ("email", "notes", "description", "remarks", "comment", "address", "bio")
_TEXT_AVG_LEN_THRESHOLD = 40   # avg sampled value length above this reads as free text, not a category


def _looks_like_identifier(col_name: str) -> bool:
    name = str(col_name)
    lname = name.lower()
    if lname in _IDENTIFIER_NAMES or lname.endswith(_IDENTIFIER_SUFFIXES):
        return True
    # camelCase identifier convention from non-Django sources (NoSQL/federated/
    # external schemas commonly use "AccountID"/"customerId"/"buildId" with no
    # underscore) — 2026-07-17, a real observed gap: "accountid".endswith("_id")
    # is False, so these columns slipped through and got charted as a category/
    # measure. Requires the Id/ID segment to follow a lowercase/digit boundary
    # so this never false-positives on lowercase English words ending in "id"
    # (paid, valid, grid, hybrid, android, ...) — never capitalized mid-word in
    # real data. Kept in sync with apps/chat/visualization.py's own copy.
    return bool(_CAMEL_ID_SUFFIX_RE.search(name))


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
class Pattern:
    """One deterministically-detected business pattern over the sampled result
    rows (detect_patterns) — rule-based, zero LLM. `detail` is a complete,
    human-readable sentence fragment the Insight Engine prompt can quote
    verbatim, so the SLM narrates precomputed facts instead of estimating."""
    kind:     str          # trend/growth | decline | dominance | concentration | top_gap | outlier | missing_values
    column:   Optional[str]
    detail:   str
    strength: float        # 0-1, used only for ranking which patterns to surface first


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
    # ── Grounding metadata (2026-07, Analytics centralization) ──────────────
    # All deterministic, all read from artifacts that already exist by the time
    # analyze_result runs (semantic model, FK relationship graph) — never a new
    # reasoning pass. These ground the Insight Engine's follow-up questions and
    # summaries in what actually exists, so the SLM can't invent business
    # concepts absent from the schema.
    primary_entity:       Optional[str] = None     # sm.tables[table].primary_entity ("A single financial transaction.")
    related_entities:     List[str] = field(default_factory=list)   # FK-adjacent tables (relationship graph)
    available_measures:   List[str] = field(default_factory=list)   # this table's MEASURE columns (sm analytics_role)
    available_dimensions: List[str] = field(default_factory=list)   # this table's DIMENSION/TIME_DIMENSION columns
    patterns:             List[Pattern] = field(default_factory=list)  # detect_patterns() output
    chart_candidates:     List[dict] = field(default_factory=list)  # compute_chart_candidates() output


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


# ---------------------------------------------------------------------------
# Grounding metadata — read (never recomputed) from artifacts that already
# exist by query time: the ingested semantic model's per-column analytics_role
# / importance_class (ingestion/deterministic_metadata.py) and the FK
# relationship graph (query/join_planner.load_graph, the same graph the join
# planner already uses). Zero LLM, zero new SQL.
# ---------------------------------------------------------------------------
def _available_columns(sm: Optional[dict], table: Optional[str]) -> Tuple[List[str], List[str]]:
    """(measures, dimensions) available on `table` per the semantic model —
    the FULL table, not just this query's SELECT list. HIGH-importance columns
    rank first (same importance_class signal routing.recommended_projection
    already trusts), capped at 10 each."""
    if not sm or not table:
        return [], []
    meas: List[Tuple[str, bool]] = []
    dims: List[Tuple[str, bool]] = []
    for key, meta in (sm.get("columns") or {}).items():
        t, _, c = key.partition(".")
        if t != table or not isinstance(meta, dict):
            continue
        role = str(meta.get("analytics_role") or "").upper()
        entry = (c, meta.get("importance_class") == "HIGH")
        if role == "MEASURE":
            meas.append(entry)
        elif role in ("DIMENSION", "TIME_DIMENSION"):
            dims.append(entry)
    _ranked = lambda lst: [c for c, _hi in sorted(lst, key=lambda e: not e[1])][:10]   # noqa: E731
    return _ranked(meas), _ranked(dims)


def _related_entities(table: Optional[str], cap: int = 8) -> List[str]:
    """Tables FK-adjacent to `table`, from the SAME relationship graph the join
    planner uses (query/join_planner._adjacency over load_graph()) — reused,
    never re-derived. Best-effort: any load failure returns [] rather than
    breaking analysis (the graph is an enrichment, not a dependency)."""
    if not table:
        return []
    try:
        from veda.runtime import get_graph
        from query.join_planner import _adjacency
        adj = _adjacency(get_graph())
        return list(dict.fromkeys(n for n, _e in adj.get(table, [])))[:cap]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Deterministic pattern detection — rule-based business patterns over the
# sampled rows. No SLM. Each rule is a plain threshold; `detail` strings are
# complete phrasings the Insight Engine can quote instead of estimating.
# ---------------------------------------------------------------------------
_EQUALITY_OPS        = ("EQ", "Is")
_MISSING_RATIO       = 0.30
_DOMINANCE_SHARE     = 0.60
_CONCENTRATION_SHARE = 0.50
_TREND_CHANGE        = 0.10
_GAP_RATIO           = 0.50
_OUTLIER_SIGMA       = 3.0
_MAX_PATTERNS        = 6


def detect_patterns(result_shape: str, column_stats: List[ColumnStat], rows: List[dict],
                    filters: List[Tuple[str, str, Optional[str]]],
                    measures: List[str]) -> List[Pattern]:
    """Deterministic pattern sweep over the sampled result rows.

    Trivial-insight guard: a dimension the query ALREADY pins with an equality
    filter is never reported as dominant — "DEBIT is the dominant entry type"
    is meaningless when the SQL says WHERE entry_type = 'DEBIT'. The filter
    list comes from the executed SQL's own AST (extract_sql_facts), so this
    guard is grounded in what actually ran, not in the question text."""
    pats: List[Pattern] = []
    n = len(rows)
    if n == 0:
        return pats
    filtered_eq = {c for c, op, _v in (filters or []) if op in _EQUALITY_OPS}

    for stat in column_stats:
        if stat.role == "identifier":
            continue
        col = stat.name
        vals = [r.get(col) for r in rows]
        non_null = [v for v in vals if v is not None]

        miss = len(vals) - len(non_null)
        if n >= 5 and miss / n >= _MISSING_RATIO:
            pats.append(Pattern("missing_values", col,
                f"{miss} of {n} sampled rows have no {col} value", round(miss / n, 3)))

        if stat.role in ("dimension", "boolean") and col not in filtered_eq and len(non_null) >= 5:
            counts = Counter(str(v) for v in non_null)
            if len(counts) >= 2:
                top_val, top_n = counts.most_common(1)[0]
                share = top_n / len(non_null)
                if share >= _DOMINANCE_SHARE:
                    pats.append(Pattern("dominance", col,
                        f"'{top_val}' accounts for {round(share * 100)}% of {col} values",
                        round(share, 3)))

        if stat.role == "measure":
            nums = [_to_number(v) for v in non_null if _is_numeric(v)]
            if len(nums) >= 8:
                mean, std = statistics.fmean(nums), statistics.pstdev(nums)
                if std > 0:
                    if max(nums) > mean + _OUTLIER_SIGMA * std:
                        pats.append(Pattern("outlier", col,
                            f"{col} has a high outlier ({max(nums)} vs average {round(mean, 2)})", 0.8))
                    if min(nums) < mean - _OUTLIER_SIGMA * std:
                        pats.append(Pattern("outlier", col,
                            f"{col} has a low outlier ({min(nums)} vs average {round(mean, 2)})", 0.8))

    if result_shape == "TREND":
        t_col = next((s.name for s in column_stats if s.kind == "temporal" and s.role != "identifier"), None)
        m_col = next((s.name for s in column_stats if s.role == "measure"), None) or \
                next((s.name for s in column_stats if s.kind == "numeric" and s.role != "identifier"), None)
        if t_col and m_col:
            pairs = [(r.get(t_col), _to_number(r.get(m_col))) for r in rows
                     if r.get(t_col) is not None and _is_numeric(r.get(m_col))]
            if len(pairs) >= 3:
                pairs.sort(key=lambda p: str(p[0]))
                first, last = pairs[0][1], pairs[-1][1]
                if first and abs(first) > 0:
                    change = (last - first) / abs(first)
                    if abs(change) >= _TREND_CHANGE:
                        word = "grew" if change > 0 else "declined"
                        pats.append(Pattern("growth" if change > 0 else "decline", m_col,
                            f"{m_col} {word} {abs(round(change * 100))}% from the first to the last period",
                            min(abs(change), 1.0)))

    if result_shape in ("RANKING", "GROUPED", "DISTRIBUTION"):
        stats_by = {s.name: s for s in column_stats}
        m_col = next((m for m in (measures or [])
                      if m in stats_by and stats_by[m].role != "identifier"), None) or \
                next((s.name for s in column_stats if s.role == "measure"), None)
        if m_col:
            nums = [_to_number(r.get(m_col)) for r in rows if _is_numeric(r.get(m_col))]
            if len(nums) >= 3:
                total = sum(nums)
                if total > 0 and max(nums) / total >= _CONCENTRATION_SHARE:
                    pats.append(Pattern("concentration", m_col,
                        f"the top {m_col} entry holds {round(max(nums) / total * 100)}% of the total",
                        round(max(nums) / total, 3)))
            if result_shape == "RANKING" and len(nums) >= 2 and nums[1] and abs(nums[1]) > 0:
                gap = (nums[0] - nums[1]) / abs(nums[1])
                if gap >= _GAP_RATIO:
                    pats.append(Pattern("top_gap", m_col,
                        f"the #1 entry leads #2 by {round(gap * 100)}% on {m_col}", min(gap, 1.0)))

    pats.sort(key=lambda p: -p.strength)
    return pats[:_MAX_PATTERNS]


def compute_chart_candidates(result_shape: str, column_stats: List[ColumnStat],
                             dimensions: List[str], measures: List[str]) -> List[dict]:
    """Deterministic chart candidates for this result — shape's canonical chart
    first (CANONICAL_CHART_FOR_SHAPE), scored by the same chart_confidence the
    SLM-suggestion validator already uses. Identifiers are never an axis.
    Consumers SELECT from these; they never invent chart types."""
    canonical = CANONICAL_CHART_FOR_SHAPE.get(result_shape)
    if canonical is None:
        return []
    stats_by = {s.name: s for s in column_stats}
    _usable = lambda name: name in stats_by and stats_by[name].role != "identifier"   # noqa: E731
    if result_shape == "TREND":
        dim = next((s.name for s in column_stats if s.kind == "temporal" and s.role != "identifier"), None)
    else:
        dim = next((d for d in (dimensions or []) if _usable(d)), None) or \
              next((s.name for s in column_stats if s.role in ("dimension", "boolean", "date")), None)
    meas = next((m for m in (measures or []) if _usable(m) and stats_by[m].kind == "numeric"), None) or \
           next((s.name for s in column_stats if s.role == "measure" and s.kind == "numeric"), None)
    if not dim or not meas:
        return []
    out = []
    for ctype in dict.fromkeys((canonical, "bar")):
        conf = chart_confidence(result_shape, ctype, stats_by[dim].role, stats_by[meas].role)
        if conf > 0:
            out.append({"type": ctype, "x_axis": dim, "y_axis": meas, "confidence": conf})
    return out


def analytics_summary(ctx: "InsightContext") -> dict:
    """JSON-safe projection of the deterministic analytics — the piece of the
    InsightContext that crosses the inference→api HTTP boundary (attached to
    the result dict, same channel `explain` already uses) so the Django tier
    (apps/chat) can consume ONE precomputed classification instead of
    re-deriving column kinds/roles from raw rows. Deliberately excludes
    sample_rows/semantic_model (bulky, already available elsewhere)."""
    return {
        "result_shape":         ctx.result_shape,
        "result_type":          ctx.result_type,
        "row_count":            ctx.row_count,
        "table":                ctx.table,
        "primary_entity":       ctx.primary_entity,
        "related_entities":     list(ctx.related_entities),
        "available_measures":   list(ctx.available_measures),
        "available_dimensions": list(ctx.available_dimensions),
        "display_columns":      [s.name for s in ctx.column_stats if s.role != "identifier"],
        "column_stats":         [{"name": s.name, "kind": s.kind, "role": s.role}
                                 for s in ctx.column_stats],
        "patterns":             [{"kind": p.kind, "column": p.column, "detail": p.detail,
                                  "strength": p.strength} for p in ctx.patterns],
        "chart_candidates":     list(ctx.chart_candidates),
    }


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

    sample = rows[:max_rows]
    # Grounding metadata + deterministic patterns/chart candidates — all read
    # from artifacts already computed by this point (semantic model, FK graph,
    # the stats above). Enrichment only: the result itself is never modified.
    primary_entity = ((sm or {}).get("tables", {}).get(table or "", {}) or {}).get("primary_entity")
    available_measures, available_dimensions = _available_columns(sm, table)
    related_entities = _related_entities(table)
    patterns = detect_patterns(result_shape, stats, sample, facts["filters"], measures)
    chart_candidates = compute_chart_candidates(result_shape, stats, dimensions, measures)

    return InsightContext(
        question=question, sql=sql or "", table=table,
        result_type=result_type, row_count=row_count, columns=list(columns),
        entities=facts["entities"], dimensions=dimensions, measures=measures,
        filters=facts["filters"], orderings=facts["orderings"],
        limit=facts["limit"], distinct=facts["distinct"],
        column_stats=stats, sample_rows=sample, semantic_model=sm,
        connector_type=connector_type, query_intent=query_intent,
        confidence_inputs=dict(confidence_inputs or {}), result_shape=result_shape,
        primary_entity=primary_entity, related_entities=related_entities,
        available_measures=available_measures, available_dimensions=available_dimensions,
        patterns=patterns, chart_candidates=chart_candidates,
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
