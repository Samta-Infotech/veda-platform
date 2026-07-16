# query/result_explainer.py
# VEDA — Result Explanation Layer (Step 7: query result -> NL answer)
# Gate: NL_ANSWER_ENABLED
#
# The small instruction SLM (NL_SUMMARY_MODEL) phrases every non-empty result —
# including the "simple" scalar/single-row shapes, which used to be answered by
# a canned string template with no SLM call at all. Template answers read
# robotic ("The count is 137."); routing them through the SLM instead gives a
# more natural sentence. To keep that cheap, the SLM never sees the raw rows —
# a deterministic extractor first PRECOMPUTES a small "facts" payload (the
# values that actually matter, nothing else), and that tiny payload is the
# entire prompt body, regardless of how many rows/columns the result has.
# template_answer()/deterministic_fallback_answer() remain as the safety net
# when the SLM is unavailable or times out.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import (
    NL_SUMMARY_MODEL,
    NL_SUMMARY_TIMEOUT_MS,
    NL_SUMMARY_MAX_TOKENS,
    INSIGHT_ENGINE_TIMEOUT_MS,
)
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NLAnswerResult:
    answer:      str
    row_count:   int
    duration_ms: float
    error:       Optional[str] = None


def _fmt_value(v):
    """Human-friendly scalar formatting (thousands separators for ints)."""
    from decimal import Decimal
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, Decimal):      # psycopg2 returns NUMERIC as Decimal
        v = float(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return str(v)


def _label_from_column(col: str) -> str:
    return str(col).replace("_", " ").strip().lower()


def template_answer(query: str, columns: List[str], rows: List[dict]) -> Optional[str]:
    """Q-7: deterministic NL answer for CANONICAL result shapes — no SLM call.

    Covers the empty set, single scalar (incl. count/aggregate), and single row.
    Returns None for multi-row narrative results, which still go to the SLM. This
    is the same phrasing the SLM would produce for these shapes, computed for free.
    """
    row_count = len(rows)
    if row_count == 0:
        return "No results found."

    if row_count == 1 and columns:
        row = rows[0]
        if len(columns) == 1:
            col = columns[0]
            val = _fmt_value(row.get(col))
            label = _label_from_column(col)
            # count/aggregate shapes: "count", "total", "n", "count(*)" …
            if any(k in label for k in ("count", "total", "number", "num ", "sum", "avg", "min", "max")):
                return f"The {label} is {val}."
            return f"{col}: {val}" if val != "" else "No results found."
        # single row, multiple columns — compact deterministic summary
        parts = [f"{_label_from_column(c)} {_fmt_value(row.get(c))}"
                 for c in columns[:6] if row.get(c) is not None]
        if parts:
            return "Result: " + ", ".join(parts) + "."
    return None


def deterministic_fallback_answer(query: str, columns: List[str], rows: List[dict]) -> str:
    """Row count + first rows summary — no SLM call. Used as the immediate answer
    on the F6 fast-return path, and as the SLM-failure fallback in run_nl_answer."""
    row_count = len(rows)
    if row_count == 0:
        return "No results found."
    first_vals = []
    if rows and columns:
        for c in columns[:3]:
            v = rows[0].get(c)
            if v is not None:
                first_vals.append(f"{c}={v}")
    return f"Returned {row_count} row(s)." + (
        f" First: {', '.join(first_vals)}." if first_vals else ""
    )


_FACTS_SAMPLE_ROWS = 5   # rows included in the precomputed facts payload, regardless of result size


def _extract_facts(columns: List[str], rows: List[dict], rank_column: Optional[str] = None) -> dict:
    """Precompute the compact 'facts' payload that is the ONLY data given to the
    SLM — never the raw rows/table. Cheap (no SLM call), deterministic, and
    constant-size: a 3-row result and a 3,000-row result produce a same-sized
    payload (a handful of sample rows + the true row_count), so prompt cost
    doesn't scale with result size.

    `rank_column`: when the caller resolved a ranking request ("top 10 X" /
    "latest N X") to a specific ORDER BY column, name it explicitly as
    "ranked_by" — otherwise the SLM has no way to know WHICH field made these
    "top"/"latest" and tends to narrate the wrong one (e.g. an id column).
    Always kept in the sampled fields even if outside the first 6 columns.
    """
    row_count = len(rows)
    cols = list(columns[:6])
    if rank_column and rank_column not in cols:
        cols.append(rank_column)
    if row_count == 1:
        row = rows[0]
        facts = {"row_count": 1,
                "fields": {c: row.get(c) for c in cols if row.get(c) is not None}}
    else:
        sample = [{c: r.get(c) for c in cols if r.get(c) is not None}
                  for r in rows[:_FACTS_SAMPLE_ROWS]]
        facts = {"row_count": row_count, "sample_rows": sample}
        if row_count > len(sample):
            facts["note"] = f"showing {len(sample)} of {row_count} rows"
    if rank_column:
        facts["ranked_by"] = rank_column
    return facts


def _column_glossary(columns: List[str], table: Optional[str], semantic_model: Optional[dict]) -> str:
    """Short 'business meaning' lines for the result's columns, pulled from the
    ingested semantic model (business_definition / analytics_role). Returns ""
    when no semantic model / table is available — purely additive context."""
    if not semantic_model or not table:
        return ""
    cols_meta = semantic_model.get("columns", {})
    lines = []
    for c in columns[:6]:
        meta = cols_meta.get(f"{table}.{c}")
        if not meta:
            continue
        definition = (meta.get("business_definition") or "").strip()
        role = meta.get("analytics_role") or meta.get("semantic_type")
        if not definition and not role:
            continue
        bits = [b for b in (role, definition[:80]) if b]
        lines.append(f"- {c}: {' — '.join(bits)}")
    return ("\n\nColumn meanings:\n" + "\n".join(lines)) if lines else ""


def run_nl_answer(
    query:          str,
    columns:        List[str],
    rows:           List[dict],
    verbose:        bool = False,
    timeout:        Optional[float] = None,
    table:          Optional[str] = None,
    semantic_model: Optional[dict] = None,
    rank_column:    Optional[str] = None,
) -> NLAnswerResult:
    """
    Converts result rows into a natural-language prose answer using a small local
    SLM (NL_SUMMARY_MODEL) — never the heavy code model used for SQL/IR generation.

    Every non-empty result is phrased by the SLM (including "simple" scalar/
    single-row shapes — a canned template reads robotically for those). To keep
    that cheap regardless of result size, the SLM never sees the raw rows: a
    deterministic, no-SLM extractor (_extract_facts) precomputes a small facts
    payload first, and ONLY that payload is sent — token cost stays flat whether
    the result is 1 row or 10,000. Falls back to the deterministic template/
    row-count answer if the SLM is unavailable, times out, or returns empty.

    `table` + `semantic_model` are optional: when both are given, the SLM prompt
    is enriched with each column's business definition / analytics role from the
    ingested semantic model (veda_semantic_model.json), so the answer can speak
    in business terms instead of raw column names.

    `rank_column`: the column a "top N"/"latest N" style request was actually
    ordered by (caller already resolved this for the SQL — see
    veda/pipeline.py's _resolve_rank_metric_column / canonical temporal column).
    Passed through to _extract_facts so the SLM narrates the right field instead
    of guessing (e.g. an id column) when asked to summarize a ranking.
    """
    import time
    t0 = time.time()

    row_count = len(rows)

    # Nothing to phrase — a fixed literal, calling the SLM here would be pure
    # waste (there is no data to summarize differently).
    if row_count == 0:
        return NLAnswerResult(answer="No results found.", row_count=0,
                              duration_ms=round((time.time() - t0) * 1000, 2))

    facts = _extract_facts(columns, rows, rank_column=rank_column)
    glossary = _column_glossary(columns, table, semantic_model)

    rank_line = (f"\n\nThese rows are already ordered by \"{rank_column}\" — "
                f"refer to that field's values, not any id column, when describing rank/order."
                ) if rank_column else ""

    prompt = (
        f"User question: {query}\n\n"
        f"Extracted data: {json.dumps(facts, default=str)}"
        f"{glossary}{rank_line}\n\n"
        f"Write ONE short sentence (max ~30 words) that directly answers the "
        f"question using this data and mentions any important numbers. "
        f"Summarize the pattern — do NOT list every row's full details one by one. "
        f"Do not repeat the column names verbatim, do not explain SQL, no markdown."
    )

    _slm_timeout = NL_SUMMARY_TIMEOUT_MS / 1000.0 if timeout is None else timeout
    try:
        from slm import call_slm
        answer = call_slm(
            prompt,
            purpose="nl_answer",
            temperature=0.1,
            num_predict=NL_SUMMARY_MAX_TOKENS,
            endpoint="generate",
            timeout=_slm_timeout,
            model=NL_SUMMARY_MODEL,
        ).strip()
        if not answer:
            raise ValueError("Empty response from SLM")
    except Exception as e:
        answer = template_answer(query, columns, rows) or \
            deterministic_fallback_answer(query, columns, rows)
        # Unconditional (not gated behind verbose=True, which neither pipeline.py's
        # L7b nor veda_hybrid.py's _tier2_finish ever pass) — previously a raw/
        # generic answer could reach the user with NO record anywhere of why the
        # SLM call didn't produce one (timeout vs. connection error vs. empty
        # response vs. bad JSON), making "sometimes the summary is raw" undiagnosable.
        logger.warning("run_nl_answer: SLM unavailable/failed (%s: %s) — using fallback answer",
                       type(e).__name__, e)
        if verbose:
            print(f"  [ResultExplainer] SLM unavailable ({e}) — using fallback answer")

    duration_ms = round((time.time() - t0) * 1000, 2)
    return NLAnswerResult(answer=answer, row_count=row_count, duration_ms=duration_ms)


# =============================================================================
# Insight Engine (Phase 4) — extends the summary SLM above with insights /
# visualization suggestion / follow-up questions, produced by ONE combined
# call, not a second SLM round trip on top of run_nl_answer's. When
# INSIGHT_ENGINE_ENABLED, this call REPLACES run_nl_answer's own SLM call for
# the same query (veda/pipeline.py's _done() picks one or the other) — never
# both, so "only one post-query SLM call" holds regardless of which is on.
# =============================================================================

_MAX_FOLLOW_UPS = 3
_VIZ_TYPES = ("bar", "line", "pie", "none")


@dataclass
class InsightResult:
    answer:              str
    row_count:            int
    duration_ms:          float
    insights:             List[str] = field(default_factory=list)
    visualization:        Optional[Dict[str, Any]] = None
    follow_up_questions:  List[str] = field(default_factory=list)
    confidence:           float = 1.0
    error:                Optional[str] = None


_SHAPE_HINTS = {
    "PIVOT":        "This result has multiple measures broken out by a dimension (a pivot-style breakdown).",
    "TREND":        "This result is a TREND over time — describe the direction/change, not just a snapshot.",
    "RANKING":      "This result is a RANKING (ordered, limited to the top/bottom N) — describe rank order.",
    "GROUPED":      "This result compares a real measure across categories — describe how they compare.",
    "DISTRIBUTION": "This result is a DISTRIBUTION (a count/frequency per category) — describe concentration/skew if notable.",
    "DETAIL_TABLE": "This is a raw detail listing with no grouping or aggregation.",
}


def _shape_line(ctx) -> str:
    hint = _SHAPE_HINTS.get(getattr(ctx, "result_shape", "SCALAR"))
    return f"\n\n{hint}" if hint else ""


def synthesize_confidence(confidence_inputs: Optional[Dict[str, float]]) -> float:
    """Weakest-link confidence from whatever gating signals the caller already
    computed upstream (e.g. veda/pipeline.py's anchor-selection + join-plan
    confidence) — never invented and never the SLM's own self-report. 1.0
    (fully confident) when the caller supplied nothing, matching the engine's
    existing default-confidence convention. Public so callers that don't run
    the Insight Engine (INSIGHT_ENGINE_ENABLED=False) can still get a
    deterministic confidence for every answered query."""
    vals = list((confidence_inputs or {}).values())
    return round(min(vals), 3) if vals else 1.0


def _synthesize_confidence(ctx) -> float:
    return synthesize_confidence(getattr(ctx, "confidence_inputs", None))


def _fallback_summary(ctx) -> str:
    """Same deterministic phrasing template_answer/deterministic_fallback_answer
    already produce, but keyed off ctx.row_count (the TRUE total) rather than
    len(ctx.sample_rows) — sample_rows is capped (RESULT_ANALYZER_MAX_ROWS) and
    would silently under-report the count for a large result set otherwise."""
    if ctx.row_count == 0:
        return "No results found."
    if ctx.row_count == 1:
        tmpl = template_answer(ctx.question, ctx.columns, ctx.sample_rows[:1])
        if tmpl:
            return tmpl
    first_vals = []
    if ctx.sample_rows and ctx.columns:
        row0 = ctx.sample_rows[0]
        for c in ctx.columns[:3]:
            v = row0.get(c)
            if v is not None:
                first_vals.append(f"{c}={v}")
    return f"Returned {ctx.row_count} row(s)." + (
        f" First: {', '.join(first_vals)}." if first_vals else ""
    )


def validate_visualization(viz: Optional[dict], ctx) -> Optional[Dict[str, Any]]:
    """Never trust the SLM's visualization suggestion blindly:
    - referenced columns must exist in the result
    - neither axis may be an IDENTIFIER column (veda/result_analyzer.classify_column_role)
      — an id/uuid/code column is never a valid measure or chart axis, regardless
      of how "numeric" it structurally looks (this is the exact bug class behind
      the id-vs-payment_attempt_count chart seen in production)
    - the result's shape (ctx.result_shape) must have a canonical chart at all —
      SCALAR/DETAIL_TABLE/PIVOT never get one (see result_analyzer.CANONICAL_CHART_FOR_SHAPE)
    - the suggested type is coerced to the shape's canonical chart when they
      disagree (a deterministic correction, not a second SLM call)
    - a deterministic confidence (veda/result_analyzer.chart_confidence) must
      clear VISUALIZATION_CONFIDENCE_THRESHOLD, or no chart is returned at all
    """
    if not viz or not isinstance(viz, dict):
        return None
    if getattr(ctx, "result_type", None) != "multi_row":
        return None
    from veda.result_analyzer import CANONICAL_CHART_FOR_SHAPE, chart_confidence

    shape = getattr(ctx, "result_shape", "SCALAR")
    canonical = CANONICAL_CHART_FOR_SHAPE.get(shape)
    if canonical is None:
        return None   # SCALAR / DETAIL_TABLE / PIVOT — no chart exists for this shape

    vtype = str(viz.get("type") or "none").strip().lower()
    if vtype not in _VIZ_TYPES or vtype == "none":
        vtype = canonical
    elif vtype != canonical:
        vtype = canonical   # shape-driven correction takes precedence (Phase 4)

    x_axis, y_axis = viz.get("x_axis"), viz.get("y_axis")
    col_names = set(ctx.columns or [])
    if x_axis is not None and x_axis not in col_names:
        return None
    if y_axis is not None and y_axis not in col_names:
        return None

    stats_by_name = {s.name: s for s in (ctx.column_stats or [])}
    x_stat, y_stat = stats_by_name.get(x_axis), stats_by_name.get(y_axis)
    if x_stat and x_stat.role == "identifier":
        return None
    if y_stat and y_stat.role == "identifier":
        return None
    if vtype == "line" and x_stat and x_stat.kind != "temporal":
        return None
    if vtype in ("bar", "pie") and y_stat and y_stat.kind != "numeric":
        return None

    try:
        from config import VISUALIZATION_CONFIDENCE_THRESHOLD
    except Exception:
        VISUALIZATION_CONFIDENCE_THRESHOLD = 0.6
    confidence = chart_confidence(
        shape, vtype,
        dim_role=x_stat.role if x_stat else None,
        measure_role=y_stat.role if y_stat else None,
    )
    if confidence < VISUALIZATION_CONFIDENCE_THRESHOLD:
        return None

    return {"type": vtype, "x_axis": x_axis, "y_axis": y_axis,
            "reason": viz.get("reason"), "confidence": confidence}


def _stats_block(ctx) -> str:
    """Precomputed per-column statistics (min/max/avg/median/nulls/distinct/top
    values, from veda/result_analyzer.py's ColumnStat) — grounds insights in
    ACTUAL numbers the backend already computed, instead of the model having
    to estimate patterns from a handful of sample rows. Identifiers are
    excluded (never a meaningful subject for an insight)."""
    lines = []
    for stat in (ctx.column_stats or []):
        if stat.role == "identifier":
            continue
        bits = []
        if stat.min is not None and stat.max is not None:
            bits.append(f"range {stat.min}-{stat.max}")
        if stat.avg is not None:
            bits.append(f"avg {stat.avg}")
        if stat.median is not None:
            bits.append(f"median {stat.median}")
        if stat.distinct_count:
            bits.append(f"{stat.distinct_count} distinct")
        if stat.null_count:
            bits.append(f"{stat.null_count} missing")
        if stat.top_values and stat.role in ("dimension", "boolean"):
            bits.append(f"most common: {', '.join(str(v) for v in stat.top_values[:3])}")
        if bits:
            lines.append(f"- {stat.name} ({stat.role}): " + ", ".join(bits))
    if not lines:
        return ""
    return "\n\nStatistics (already computed — ground insights in these, don't estimate):\n" + "\n".join(lines)


def run_insight_engine(ctx, verbose: bool = False, timeout: Optional[float] = None,
                       rank_column: Optional[str] = None) -> InsightResult:
    """The one post-query SLM call when INSIGHT_ENGINE_ENABLED — extends
    run_nl_answer's summary with insights/visualization/follow-ups in a SINGLE
    json_format call, reusing the same facts-extraction (_extract_facts) and
    semantic-metadata enrichment (_column_glossary) run_nl_answer already uses.
    Never sends raw rows to the model — same constant-size facts payload
    regardless of result size. Deterministic-safe on any failure: `answer`
    always comes back populated (from the same fallback template_answer/
    deterministic_fallback_answer phrasing), insights/visualization/
    follow_up_questions simply come back empty."""
    import time
    t0 = time.time()

    confidence = _synthesize_confidence(ctx)

    if ctx.row_count == 0:
        return InsightResult(answer="No results found.", row_count=0,
                             duration_ms=round((time.time() - t0) * 1000, 2),
                             confidence=confidence)

    facts = _extract_facts(ctx.columns, ctx.sample_rows, rank_column=rank_column)
    glossary = _column_glossary(ctx.columns, ctx.table, ctx.semantic_model)
    stats_block = _stats_block(ctx)
    rank_line = (f"\n\nThese rows are already ordered by \"{rank_column}\" — "
                f"refer to that field's values, not any id column, when describing rank/order."
                ) if rank_column else ""
    shape_line = _shape_line(ctx)

    prompt = (
        f"User question: {ctx.question}\n\n"
        f"Extracted data: {json.dumps(facts, default=str)}"
        f"{stats_block}{glossary}{rank_line}{shape_line}\n\n"
        "Return ONLY a JSON object with this exact shape (no markdown, no commentary):\n"
        '{"summary": "ONE analytical sentence (max ~30 words) using the statistics above — '
        'note a range, concentration, gap, or notable pattern, not just the row count. '
        'Bad: \'The system returned 100 users.\' Good: \'Most users have no recorded '
        'last login timestamp.\' Never restate LIMIT/COUNT/SQL mechanics; never repeat '
        'column names verbatim", '
        '"insights": ["0-3 short factual observations grounded ONLY in the statistics/data '
        'above — e.g. largest value, lowest value, an outlier, the dominant category, a '
        'large increase/decrease, high concentration in one category, or notable missing '
        'values. [] if nothing notable. Never fabricate a number not shown above"], '
        '"visualization": {"type": "bar|line|pie|none", "x_axis": "a column name from '
        'the data above, or null", "y_axis": "a column name from the data above, or null", '
        '"reason": "why this chart fits the data (e.g. \'compares a measure across '
        'discrete categories\'), or null"}, '
        '"follow_up_questions": ["0-3 natural follow-up questions — must reference ONLY '
        'column names shown above, must read as an executable data question (e.g. '
        '\'Show only active users\', \'Compare by month\', \'Break down by department\'), '
        'never a vague topic, never an invented field"]}\n'
        "Never invent columns or values not present in the data above."
    )

    _slm_timeout = INSIGHT_ENGINE_TIMEOUT_MS / 1000.0 if timeout is None else timeout
    try:
        from slm import call_slm
        raw = call_slm(
            prompt,
            purpose="insight_engine",
            temperature=0.1,
            num_predict=NL_SUMMARY_MAX_TOKENS + 120,   # room for insights/follow-ups beyond the summary
            json_format=True,
            timeout=_slm_timeout,
            model=NL_SUMMARY_MODEL,
        ).strip()
        parsed = json.loads(raw)
        summary = (parsed.get("summary") or "").strip()
        if not summary:
            raise ValueError("Empty summary from SLM")
        insights = [str(i) for i in (parsed.get("insights") or []) if str(i).strip()][:_MAX_FOLLOW_UPS]
        follow_ups = [str(q) for q in (parsed.get("follow_up_questions") or [])
                     if str(q).strip()][:_MAX_FOLLOW_UPS]
        visualization = validate_visualization(parsed.get("visualization"), ctx)
    except Exception as e:
        summary = _fallback_summary(ctx)
        insights, follow_ups, visualization = [], [], None
        # Unconditional — see run_nl_answer's identical logging fix above for why.
        logger.warning("run_insight_engine: SLM unavailable/invalid (%s: %s) — using fallback answer",
                       type(e).__name__, e)
        if verbose:
            print(f"  [InsightEngine] SLM unavailable/invalid ({e}) — using fallback answer")

    duration_ms = round((time.time() - t0) * 1000, 2)
    return InsightResult(answer=summary, row_count=ctx.row_count, duration_ms=duration_ms,
                         insights=insights, visualization=visualization,
                         follow_up_questions=follow_ups, confidence=confidence)
