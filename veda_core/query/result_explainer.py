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
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import (
    NL_SUMMARY_MODEL,
    NL_SUMMARY_TIMEOUT_MS,
    NL_SUMMARY_MAX_TOKENS,
    NL_SUMMARY_NUMERIC_GUARD,
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
    # True when the SLM produced the prose (and therefore already wove in any
    # `patterns` it was handed); False when the deterministic template/row-count
    # fallback was used instead. Lets the caller decide whether to blend the
    # detected patterns deterministically (fallback) or leave them alone (the SLM
    # already phrased them) — avoids the double-statement the old unconditional
    # "Analysis:" suffix produced on SLM answers. See veda/pipeline.py L7b.
    slm_used:    bool = False


def blend_patterns(answer: str, patterns: Optional[List[str]]) -> str:
    """Fold the deterministic detected-pattern details into an answer as ONE
    natural clause instead of a mechanical "Analysis: …" suffix (product call,
    2026-07-17). Used as the SLM-failure fallback and by veda_hybrid.py's non-Tier-1
    heads, so every path blends identically. Top 2 only — a sentence, not a report.

        answer="Total is ₹3.2L across 5 payments", patterns=["DEBIT dominates (4 of 5)",
        "top value is 40% above the median"]
        → "Total is ₹3.2L across 5 payments — DEBIT dominates (4 of 5), and top value
           is 40% above the median."
    """
    pats = [str(p).strip().rstrip(".") for p in (patterns or []) if str(p).strip()][:2]
    if not pats:
        return answer or ""
    tail = pats[0] if len(pats) == 1 else f"{pats[0]}, and {pats[1]}"
    base = (answer or "").rstrip()
    if not base:
        # No prose at all — lead with the finding rather than a dangling clause.
        return f"{tail[:1].upper()}{tail[1:]}."
    return f"{base.rstrip('.')} — {tail}."


# ── Summary shaping — result-shape-aware guidance + a style exemplar ─────────────
# Lever #4: tell the SLM WHAT KIND of result it is looking at, so a ranking reads
# like a ranking and a trend like a trend, instead of one generic phrasing for all.
# Keyed by veda/result_analyzer.py's RESULT_SHAPES. Empty string for unknown/None.
_SHAPE_GUIDANCE = {
    "RANKING":      "This is a ranking — name who leads (and, if useful, who trails) and by how much.",
    "TREND":        "This is a time trend — say whether it rose or fell over the period and the overall direction.",
    "GROUPED":      "This is a measure broken down by category — name the largest (and smallest) group and any concentration.",
    "DISTRIBUTION": "This is a frequency breakdown — name the most and least common categories.",
    "PIVOT":        "This is a cross-tab — call out the standout cell(s), not every combination.",
    "SCALAR":       "This is a single figure — state it directly and what it represents.",
    "DETAIL_TABLE": "This is a list of records — summarize the overall picture, not row by row.",
}

# Lever #3: one schema-agnostic exemplar pins the desired 2-3 sentence business
# format/tone for the instruct model (few-shot). Its NUMBERS are fictional and must
# not leak into a real answer — the prompt says so. Deliberately CURRENCY-NEUTRAL
# (no $/₹): the exemplar must not bias the model's currency — the real answer takes
# its currency/units from the data, not from this sample.
_STYLE_EXEMPLAR = (
    "\n\nStyle example (illustrative only — never reuse its numbers, entities, or units):\n"
    "Q: Top 3 regions by revenue\n"
    "A: North leads with 1.2M in revenue, ahead of West (900K) and South (600K). "
    "Revenue is concentrated at the top — North alone is nearly half the combined total."
)


# ── Numeric anti-hallucination guardrail (config: NL_SUMMARY_NUMERIC_GUARD) ──────
_MAGNITUDE = {"k": 1e3, "l": 1e5, "lakh": 1e5, "lakhs": 1e5,
              "m": 1e6, "mn": 1e6, "million": 1e6, "cr": 1e7, "crore": 1e7,
              "crores": 1e7, "b": 1e9, "bn": 1e9, "billion": 1e9}
_NUM_TOKEN = re.compile(
    r'(?<![\w.])(?:[₹$]|rs\.?\s*)?(\d[\d,]*(?:\.\d+)?)\s*'
    # magnitude suffix must be a whole token (\b) so "2 matching" is NOT read as
    # "2 million" — the alpha suffix can't glom onto the next word's first letter.
    r'(%|(?:k|l|cr|m|b|lakhs?|crores?|million|billion|bn|mn)\b)?',
    re.IGNORECASE)


def _parse_numbers_from_text(text: str) -> List[float]:
    """Every number a piece of text asserts, normalized to a float. Understands
    thousands commas, currency prefixes, a trailing '%' (kept as its face value —
    40% → 40) and magnitude suffixes (K/L/M/Cr/lakh/crore/…). Used both to collect
    the ALLOWED numbers (from the precomputed facts + patterns) and to pull the
    numbers a generated summary states, so the two can be compared."""
    out: List[float] = []
    for m in _NUM_TOKEN.finditer(str(text or "")):
        raw, suffix = m.group(1), (m.group(2) or "").lower()
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        if suffix and suffix != "%":
            val *= _MAGNITUDE.get(suffix, 1.0)
        out.append(val)
    return out


def _collect_allowed_numbers(facts: dict, patterns: Optional[List[str]]) -> List[float]:
    """The set of numbers the summary is ALLOWED to state — every numeric value in
    the precomputed facts payload (sample cells, row_count, the exact metrics) plus
    every number named in the detected patterns. Walked recursively so nested
    metrics/sample dicts are covered."""
    allowed: List[float] = []

    def _walk(v):
        from decimal import Decimal
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float, Decimal)):
            allowed.append(float(v))
        elif isinstance(v, str):
            allowed.extend(_parse_numbers_from_text(v))
        elif isinstance(v, dict):
            for x in v.values():
                _walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                _walk(x)

    _walk(facts)
    for p in (patterns or []):
        allowed.extend(_parse_numbers_from_text(str(p)))
    return allowed


def _answer_numbers_grounded(answer: str, facts: dict, patterns: Optional[List[str]]) -> bool:
    """True when EVERY number the summary states is traceable to the precomputed
    facts/metrics/patterns (within ±2%, floor ±2) — or is a small count/position
    (≤ row_count, ≤ 12) the model may legitimately mention ("4 of the 5"). A single
    ungrounded figure ⇒ False, so the caller can fall back to the deterministic
    answer rather than ship a confident wrong number. Deliberately lenient (large
    allowed set + tolerance) so it only trips on genuine invention, not rounding."""
    allowed = _collect_allowed_numbers(facts, patterns)
    ceiling = max(int(facts.get("row_count", 0) or 0), 12)
    for n in _parse_numbers_from_text(answer):
        if float(n).is_integer() and abs(n) <= ceiling:
            continue   # a count / rank / ordinal — always fair game
        if any(abs(n - a) <= max(2.0, 0.02 * abs(a)) for a in allowed):
            continue
        return False   # stated a figure that appears nowhere in the grounded inputs
    return True


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


def _as_number(v):
    """Coerce a cell to float if it is (or looks like) a number, else None."""
    from decimal import Decimal
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, Decimal):
        return float(v)
    return None


def _numeric_aggregates(columns: List[str], rows: List[dict], max_cols: int = 6) -> dict:
    """Deterministically precompute per-column aggregates (count/sum/min/max/mean/
    median) for every numeric column, so the summary SLM is HANDED the exact
    totals/extremes it would otherwise have to compute itself (2026-07-17,
    anti-hallucination lever #1). It never computes numbers on its own → the
    prompt tells it to use only these. Rounded to 2 dp; empty when no numeric
    column. Constant cost — reads at most the first RESULT rows already in hand."""
    metrics: dict = {}
    for c in list(columns)[:max_cols]:
        # Skip identifier columns by name — summing/averaging ids is meaningless
        # (and pollutes the allowed-number set). No semantic model here, so this is
        # a deterministic name heuristic; measures/amounts/counts are unaffected.
        _cl = str(c).lower()
        if _cl == "id" or _cl.endswith("_id"):
            continue
        vals = [n for n in (_as_number(r.get(c)) for r in rows) if n is not None]
        if len(vals) < 1:
            continue
        m = {"count": len(vals),
             "min": round(min(vals), 2), "max": round(max(vals), 2),
             "sum": round(sum(vals), 2), "mean": round(statistics.fmean(vals), 2)}
        if len(vals) >= 2:
            m["median"] = round(statistics.median(vals), 2)
        metrics[c] = m
    return metrics


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
    # Exact aggregates over the FULL result (not just the sampled rows) so the SLM
    # states real totals/extremes instead of summing the sample by eye.
    _metrics = _numeric_aggregates(cols, rows)
    if _metrics:
        facts["metrics"] = _metrics
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
    patterns:       Optional[List[str]] = None,
    result_shape:   Optional[str] = None,
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

    # Deterministic findings (result_analyzer detected these — precomputed, not
    # for the model to recompute or second-guess). Handed in so the SLM WEAVES
    # them into the prose naturally, instead of the caller bolting on a separate
    # "Analysis:" suffix afterwards. Top 2 only.
    _pats = [str(p).strip().rstrip(".") for p in (patterns or []) if str(p).strip()][:2]
    findings_line = ("\n\nKey findings already computed (weave these into the answer as "
                     "insight, do not restate as a list): " + "; ".join(_pats)) if _pats else ""

    shape_hint = _SHAPE_GUIDANCE.get(result_shape or "", "")
    shape_line = f"\n{shape_hint}" if shape_hint else ""

    prompt = (
        f"User question: {query}\n\n"
        f"Extracted data: {json.dumps(facts, default=str)}"
        f"{glossary}{rank_line}{findings_line}{_STYLE_EXEMPLAR}\n\n"
        f"You are a business analyst. Write a SHORT answer of 1-3 sentences (max ~55 "
        f"words total):\n"
        f"1. First sentence: directly answer the question, leading with the single most "
        f"important number.\n"
        f"2. ONLY IF there is a notable pattern or a finding listed above, add one short "
        f"sentence on what it means. If there is no real pattern, stop after the direct "
        f"answer — do NOT invent an insight or pad the response.{shape_line}\n"
        f"Speak in business terms using the column meanings above (name the entity, not "
        f"an id). Use the data's OWN currency/units/dates exactly as shown — never "
        f"introduce a currency symbol or unit that isn't in the data. Do not repeat "
        f"column names verbatim, do not explain SQL, no markdown, no bullet points.\n"
        f"IMPORTANT: use ONLY numbers that appear above (the data, the metrics, and the "
        f"findings — totals/averages/min/max are already computed for you). Never "
        f"calculate, sum, or estimate a new figure of your own."
    )

    _slm_timeout = NL_SUMMARY_TIMEOUT_MS / 1000.0 if timeout is None else timeout
    slm_used = False
    try:
        from slm import call_slm
        answer = call_slm(
            prompt,
            purpose="nl_answer",
            temperature=0.1,
            # 2-3 business sentences need more room than the old one-liner budget;
            # +50 covers the extra sentence + woven findings without unbounding it.
            num_predict=NL_SUMMARY_MAX_TOKENS + 50,
            endpoint="generate",
            timeout=_slm_timeout,
            model=NL_SUMMARY_MODEL,
        ).strip()
        if not answer:
            raise ValueError("Empty response from SLM")
        # Anti-hallucination guardrail: a summary that invents a number is worse
        # than a plainer correct one — fall back to the deterministic blend if any
        # stated figure isn't grounded in the facts/metrics/findings.
        if NL_SUMMARY_NUMERIC_GUARD and not _answer_numbers_grounded(answer, facts, _pats):
            logger.warning("run_nl_answer: summary stated an ungrounded number — "
                           "falling back to deterministic answer. summary=%r", answer)
            raise ValueError("ungrounded number in SLM summary")
        slm_used = True   # the SLM wove the findings into its prose — caller must NOT re-append
    except Exception as e:
        # Deterministic fallback: blend the findings in ourselves (naturally, not a
        # bolted-on "Analysis:" suffix) since the SLM prose that would have woven
        # them never arrived.
        answer = template_answer(query, columns, rows) or \
            deterministic_fallback_answer(query, columns, rows)
        answer = blend_patterns(answer, _pats)
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
    return NLAnswerResult(answer=answer, row_count=row_count, duration_ms=duration_ms,
                          slm_used=slm_used)


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
    would silently under-report the count for a large result set otherwise.
    Shape decided by result_analyzer.classify_result_type — the one canonical
    shape classifier — instead of a third inline row-count re-derivation."""
    from veda.result_analyzer import classify_result_type
    result_type = classify_result_type(ctx.row_count, ctx.columns)
    if result_type == "empty":
        return "No results found."
    if result_type in ("scalar", "single_row"):
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


def _patterns_block(ctx) -> str:
    """Deterministically-detected business patterns (result_analyzer.detect_patterns)
    rendered for the prompt — the SLM narrates these precomputed facts instead of
    estimating patterns from a handful of sample rows. Trivial patterns (e.g. a
    dominance fact about a dimension the SQL already filters on) were already
    suppressed at detection time, grounded in the executed SQL's own AST."""
    pats = getattr(ctx, "patterns", None) or []
    if not pats:
        return ""
    lines = [f"- {p.detail}" for p in pats[:6]]
    return ("\n\nDetected patterns (deterministic, precomputed — base insights on these, "
            "never invent others):\n" + "\n".join(lines))


def _grounding_block(ctx) -> str:
    """The business vocabulary that ACTUALLY exists for this result — primary
    entity, this table's available measures/dimensions (full table, not just
    this query's SELECT list), FK-adjacent related entities, and the filters
    the query already applied. All read from InsightContext fields that
    result_analyzer populated deterministically (semantic model + FK graph).
    This is what keeps follow-up questions grounded: the prompt explicitly
    scopes them to this vocabulary, and validate_follow_up_questions drops
    anything that references none of it."""
    lines = []
    if getattr(ctx, "primary_entity", None):
        lines.append(f"- Each row is: {ctx.primary_entity}")
    if getattr(ctx, "available_measures", None):
        lines.append(f"- Measures available on this data: {', '.join(ctx.available_measures[:8])}")
    if getattr(ctx, "available_dimensions", None):
        lines.append(f"- Dimensions available for grouping/filtering: {', '.join(ctx.available_dimensions[:8])}")
    if getattr(ctx, "related_entities", None):
        lines.append(f"- Related entities (joinable): {', '.join(ctx.related_entities[:6])}")
    if getattr(ctx, "filters", None):
        applied = ", ".join(f"{c} {op} {v}" for c, op, v in ctx.filters[:5])
        lines.append(f"- Filters ALREADY applied by this query: {applied} — never restate "
                     f"these as findings, and never suggest re-applying them")
    if not lines:
        return ""
    return ("\n\nBusiness context (the ONLY entities, measures and dimensions that exist "
            "— use nothing outside this list):\n" + "\n".join(lines))


def validate_follow_up_questions(follow_ups: List[str], ctx) -> List[str]:
    """Deterministic groundedness gate for the SLM's follow-up suggestions —
    the follow-up counterpart of validate_visualization. A suggestion survives
    only if it references at least one term that actually exists for this
    result: a result column, an available measure/dimension of the table, a
    related (FK-adjacent) entity, the table itself, or a word from the primary-
    entity description. Anything referencing none of these is an invented
    business concept and is dropped. Deliberately lenient in matching
    (underscores→spaces, substring), strict in principle: a dropped-but-valid
    follow-up costs little; an unanswerable one erodes trust."""
    if not follow_ups:
        return []
    vocab = set()

    def _add(term):
        t = str(term or "").strip().lower()
        if len(t) > 2:
            vocab.add(t.replace("_", " "))

    for c in (getattr(ctx, "columns", None) or []):
        _add(c)
    for m in (getattr(ctx, "available_measures", None) or []):
        _add(m)
    for d in (getattr(ctx, "available_dimensions", None) or []):
        _add(d)
    for r in (getattr(ctx, "related_entities", None) or []):
        _add(r)
        for part in str(r).split("_"):
            _add(part)
    if getattr(ctx, "table", None):
        _add(ctx.table)
        for part in str(ctx.table).split("_"):
            _add(part)
    for w in str(getattr(ctx, "primary_entity", "") or "").lower().split():
        _add(w.strip(".,"))

    kept = []
    for q in follow_ups:
        ql = str(q).lower().replace("_", " ")
        if any(term in ql for term in vocab):
            kept.append(q)
        else:
            logger.warning("validate_follow_up_questions: dropped ungrounded follow-up %r", q)
    return kept[:_MAX_FOLLOW_UPS]


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
    patterns_block = _patterns_block(ctx)
    grounding_block = _grounding_block(ctx)
    rank_line = (f"\n\nThese rows are already ordered by \"{rank_column}\" — "
                f"refer to that field's values, not any id column, when describing rank/order."
                ) if rank_column else ""
    shape_line = _shape_line(ctx)

    prompt = (
        f"User question: {ctx.question}\n\n"
        f"Extracted data: {json.dumps(facts, default=str)}"
        f"{stats_block}{patterns_block}{glossary}{grounding_block}{rank_line}{shape_line}\n\n"
        "Return ONLY a JSON object with this exact shape (no markdown, no commentary):\n"
        '{"summary": "ONE analytical sentence (max ~30 words) using the statistics and '
        'detected patterns above — note a range, concentration, gap, or notable pattern, '
        'not just the row count. '
        'Bad: \'The system returned 100 users.\' Good: \'Most users have no recorded '
        'last login timestamp.\' Never restate LIMIT/COUNT/SQL mechanics; never repeat '
        'column names verbatim", '
        '"insights": ["0-3 short factual observations — prefer rephrasing the detected '
        'patterns above in business language; otherwise ground strictly in the statistics/'
        'data shown. Never restate a filter the query already applied as a finding. '
        '[] if nothing notable. Never fabricate a number not shown above"], '
        '"visualization": {"type": "bar|line|pie|none", "x_axis": "a column name from '
        'the data above, or null", "y_axis": "a column name from the data above, or null", '
        '"reason": "why this chart fits the data (e.g. \'compares a measure across '
        'discrete categories\'), or null"}, '
        '"follow_up_questions": ["0-3 natural follow-up questions — must use ONLY the '
        'columns, measures, dimensions, or related entities listed in the business '
        'context above, must read as an executable data question (e.g. '
        '\'Show only active users\', \'Compare by month\'), '
        'never a vague topic, never an invented field or business concept"]}\n'
        "Never invent columns, values, or business concepts not present in the data above."
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
        follow_ups = validate_follow_up_questions(
            [str(q) for q in (parsed.get("follow_up_questions") or []) if str(q).strip()], ctx)
        # Parked (INSIGHT_FOLLOW_UPS_ENABLED, default off — see config.py): read
        # at call time (not module import) so a per-deployment/env flip needs no
        # process restart of THIS module's import chain and tests can monkeypatch.
        import config as _cfg
        if not getattr(_cfg, "INSIGHT_FOLLOW_UPS_ENABLED", False):
            follow_ups = []
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
