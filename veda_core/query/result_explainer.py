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
    NL_SUMMARY_ANALYTICAL_MAX_TOKENS,
    NL_SUMMARY_MAX_FINDINGS,
    NL_SUMMARY_NUMERIC_GUARD,
    INSIGHT_ENGINE_TIMEOUT_MS,
)
from utils.logger import get_logger
from decimal import Decimal
import time
from veda.result_analyzer import classify_result_type
from veda.result_analyzer import CANONICAL_CHART_FOR_SHAPE, chart_confidence

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


_CURRENCY_SYMS = "$₹€£¥₩₽"


_GROUPED_NUMBER_RE = __import__("re").compile(
    r"(?<![\d,.])(\d{1,4}(?:,\d+)+)(\.\d+)?(?![\d,])")
_WESTERN_GROUPING_RE = __import__("re").compile(r"^\d{1,3}(?:,\d{3})+$")
_INDIAN_GROUPING_RE = __import__("re").compile(r"^\d{1,2}(?:,\d{2})*,\d{3}$")


def _regroup_numbers(answer: str) -> str:
    """Re-group any number whose thousands separators are in the wrong places.

    The summary SLM is handed raw values (1222441350.14) and inserts the commas ITSELF —
    measured 2026-09-25 on the live engine: "The average expected price of sale listings
    is 122,244,1350.14" and "The total security deposit … is 23,493,131,4590.000". The
    VALUE is right (the numeric guard strips commas, so it passed), the rendering is not,
    and it is the first thing a reader sees. A grouping that is valid in EITHER the
    Western (1,222,441,350) or the Indian (12,22,44,135) convention is left exactly as
    written; only an impossible grouping is rewritten, Western-style, digits unchanged.
    """
    if not answer or "," not in answer:
        return answer

    def _fix(m):
        whole, frac = m.group(1), m.group(2) or ""
        if _WESTERN_GROUPING_RE.match(whole) or _INDIAN_GROUPING_RE.match(whole):
            return m.group(0)
        groups = whole.split(",")
        # Only a group that is TOO LONG proves a mis-grouped number ("…,1350", "…,4590",
        # "1234,567"). Groups that are too SHORT ("1,2,3", "10,20") are far more likely a
        # list of separate figures, and rewriting them would merge values — left alone.
        if max(len(g) for g in groups[1:]) <= 3 and len(groups[0]) <= 3:
            return m.group(0)
        if any(len(g) < 3 for g in groups[1:]):
            return m.group(0)
        return f"{int(''.join(groups)):,}{frac}"

    return _GROUPED_NUMBER_RE.sub(_fix, answer)


def _strip_invented_currency(answer: str, facts: dict) -> str:
    """Remove any currency symbol the SLM prefixed that ISN'T present in the data
    (2026-07-17). The platform is multi-source/multi-tenant with currency itself a
    data column (see apps/chat/table_rendering.py's same rule), so a guessed symbol
    ($ on INR data, etc.) is actively misleading. Only strips a symbol absent from
    the facts payload — if the data genuinely carries one, it's kept. Deterministic
    backstop for the prompt's 'never introduce a currency not in the data' rule,
    which a 7B model doesn't always obey."""
    if not answer:
        return answer
    # ensure_ascii=False is REQUIRED: the default escapes non-ASCII (₹ → "₹"),
    # which would make the "symbol present in data?" check fail for every non-$
    # currency and wrongly strip a ₹/€/£/¥ the data genuinely carries.
    facts_text = json.dumps(facts, default=str, ensure_ascii=False)
    for sym in _CURRENCY_SYMS:
        if sym not in facts_text:
            answer = answer.replace(sym, "")
    return answer


_PROPORTION_RE = __import__("re").compile(
    r"(\d+(?:\.\d+)?\s*%)"                             # "86%"
    r"|(\b\d+(?:\.\d+)?\s*percent\b)"                 # "86 percent"
    r"|(\b\d[\d,]*\s+of\s+\d[\d,]*\b)"                # "8 of 100"
    # "79 assets out of 100" — the counted NOUN sits between the number and "out of",
    # which the adjacent-only form above missed. Measured 2026-09-22: that exact
    # sentence passed the first version of this guard on a truncated page. Intervening
    # words are allowed only for the unambiguous "out of"; a bare "of" stays adjacent,
    # since "1,190.81 square meters of ..." is not a proportion.
    r"|(\b\d[\d,]*(?:\s+\w+){0,3}\s+out\s+of\s+\d[\d,]*\b)",
    __import__("re").I)


def _states_a_proportion(answer: str) -> bool:
    """Does this summary express a share of a whole — a percentage, or "N of M"?

    Only ever consulted for a TRUNCATED result, where the page is not the population
    and every such figure is therefore unsupported by the rows the narrator saw.

    The prompt already forbids this in as many words ("never state a percentage or
    proportion, never say 'out of N'"), and the model says it anyway. Measured
    2026-09-22 on one query whose WHERE clause already restricted the rows to Pune —
    so the true share is 100% — three runs of the SAME SQL produced "86% of the assets
    listed are in Pune", "8 of 100 rows are in Pune", and a third with no figure at
    all. Two contradictory invented numbers, both delivered confidently, and the
    truncation caveat was appended to them rather than preventing them.

    So this is a deterministic guard rather than more prompt text, the same choice
    this module already made for invented currency symbols and ungrounded numbers
    above: a hard guarantee, not a lower probability.
    """
    return bool(_PROPORTION_RE.search(answer or ""))


_AGGREGATE_WORD_RE = __import__("re").compile(
    r"\b(average|mean|median|typical)\b", __import__("re").I)


def _states_page_derived_total(answer: str, facts: dict) -> bool:
    """Does this summary present a figure derived from the PAGE as a fact about the data?

    Two shapes, both measured on the live engine 2026-09-22 against assets_asset, which
    holds 7,814 rows:

        "There are 1000 assets listed ..."                  the page size as the total
        "The average carpet area is 1628.46 square meters"  a mean over one page

    Neither is a truncation artefact the reader can discount — both read as statements
    about the table. Only consulted when the result was SILENTLY cut; a limit the user
    asked for ("top 5") is excluded upstream by _user_asked_for_n_rows, so a correct
    "the top 5 transactions total 45,648,588" is never touched.
    """
    if not answer:
        return False
    if _AGGREGATE_WORD_RE.search(answer):
        return True
    shown = facts.get("rows_shown") or facts.get("row_count")
    if not shown:
        return False
    # The page size quoted back as a quantity — "1000 assets", "There are 1,000 ...".
    pattern = __import__("re").compile(
        r"\b" + f"{int(shown):,}".replace(",", "[,]?") + r"\b")
    return bool(pattern.search(answer))


def _answer_numbers_grounded(answer: str, facts: dict, patterns: Optional[List[str]]) -> bool:
    """True when EVERY number the summary states is traceable to the precomputed
    facts/metrics/patterns (within ±2%, floor ±2) — or is a small count/position
    (≤ row_count, ≤ 12) the model may legitimately mention ("4 of the 5"). A single
    ungrounded figure ⇒ False, so the caller can fall back to the deterministic
    answer rather than ship a confident wrong number. Deliberately lenient (large
    allowed set + tolerance) so it only trips on genuine invention, not rounding."""
    allowed = _collect_allowed_numbers(facts, patterns)
    # A truncated result must NOT license every integer up to the page size: that whitelist is
    # exactly what let "9 out of 100 … Over 95% …" through the guard on a LIMIT-100 page.
    ceiling = 12 if facts.get("result_truncated") else max(int(facts.get("row_count", 0) or 0), 12)
    for n in _parse_numbers_from_text(answer):
        if float(n).is_integer() and abs(n) <= ceiling:
            continue   # a count / rank / ordinal — always fair game
        if any(abs(n - a) <= max(2.0, 0.02 * abs(a)) for a in allowed):
            continue
        return False   # stated a figure that appears nowhere in the grounded inputs
    return True


def _fmt_value(v):
    """Human-friendly scalar formatting (thousands separators for ints)."""
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

_LIMIT_RE = __import__("re").compile(r"\bLIMIT\s+(\d+)\s*$", __import__("re").I)


def _user_asked_for_n_rows(query: Optional[str], n_rows: int) -> bool:
    """Did the USER ask for exactly this many rows ("top 5", "first 20")?

    A limit the user chose is not a truncation: an answer about their top 5 is an
    answer about what they asked for, and statistics over those 5 are legitimate. Only
    rows WE silently removed make a derived figure misleading. Without this the guards
    below would fire on every "top N" question and strip correct summaries.
    """
    if not query or n_rows <= 0:
        return False
    try:
        from query.ranking_parser import parse_ranking
        top_n = parse_ranking(query).top_n
    except Exception:
        return False
    return top_n is not None and n_rows <= int(top_n)


def _sql_truncated(sql: Optional[str], n_rows: int, query: Optional[str] = None) -> bool:
    """True when the executed SQL's trailing LIMIT is exactly filled — i.e. these rows are ONE PAGE
    of a larger result and `len(rows)` is NOT the population size. Deliberately conservative: it
    only fires on a filled limit, so a 1-row aggregate under `LIMIT 100` is never flagged.
    Flag-gated; off → always False and every caller behaves exactly as before."""
    try:
        from config import SUMMARY_TRUNCATION_AWARE_ENABLED as _on
    except Exception:
        return False
    if not _on or n_rows <= 0:
        return False
    if _user_asked_for_n_rows(query, n_rows):
        return False
    # The EXECUTOR's own cap, which applies whether or not the SQL carries a LIMIT:
    # veda/execution.py fetches at most EXECUTION_RESULT_LIMIT rows. Checking only the
    # SQL text was correct while every query was given a LIMIT; once the invented
    # default was removed (veda/validation.py, 2026-09-22) a result could be cut by the
    # fetch alone and this returned False — silently disabling every truncation
    # protection downstream, including the proportion guard. Detection must not depend
    # on the cap being visible in the SQL string.
    try:
        from config import EXECUTION_RESULT_LIMIT as _exec_cap
    except Exception:
        _exec_cap = None
    if _exec_cap and n_rows >= int(_exec_cap):
        return True
    if not sql:
        return False
    m = _LIMIT_RE.search(sql.strip().rstrip(";"))
    return bool(m) and n_rows >= int(m.group(1))


def _as_number(v):
    """Coerce a cell to float if it is (or looks like) a number, else None."""
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
    # Phase-7 scalability + RC-4 consistency: scan at most ANALYSIS_MAX_ROWS — the
    # SAME bound result_analyzer applies to the pattern sweep — so the headline mean
    # here and any pattern's "vs average X" are always over one population, and neither
    # does an unbounded O(rows) pass on a very large enterprise result.
    try:
        from config import ANALYSIS_MAX_ROWS as _ANALYSIS_MAX
    except Exception:
        _ANALYSIS_MAX = 50000
    if len(rows) > _ANALYSIS_MAX:
        rows = rows[:_ANALYSIS_MAX]
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


def _extract_facts(columns: List[str], rows: List[dict], rank_column: Optional[str] = None,
                   truncated: bool = False) -> dict:
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
        if truncated:
            # row_count is ONE PAGE, not the population — say so inside the payload itself, since
            # that payload is the only thing the narrator sees.
            facts["result_truncated"] = True
            facts["rows_shown"] = row_count
            facts["note"] = (f"showing {len(sample)} of {row_count} returned rows; {row_count} is "
                             f"a truncated page, the true total is UNKNOWN")
    if rank_column:
        facts["ranked_by"] = rank_column
    # Aggregates over the result, bounded to ANALYSIS_MAX_ROWS (Phase-7 scalability).
    # When the result is LARGER than that bound the metrics cover only the first N rows,
    # so they are a PARTIAL-population statistic — flag it explicitly (metrics_partial /
    # metrics_scanned) so the narrator does NOT present a partial SUM/COUNT as a
    # full-population total. Under the bound (the common case) metrics are exact and no
    # flag is set. row_count above is always the TRUE total.
    from config import ANALYSIS_MAX_ROWS as _ANALYSIS_MAX
    _metrics = _numeric_aggregates(cols, rows)
    if _metrics:
        facts["metrics"] = _metrics
        if row_count > _ANALYSIS_MAX:
            facts["metrics_partial"] = True
            facts["metrics_scanned"] = _ANALYSIS_MAX
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


# ── Evidence-adaptive summary modes ─────────────────────────────────────────────
# The summary is NOT one-size-fits-all: a scalar answer is one line; a grouped/ranked/
# trend result with several VERIFIED findings earns a short analytical narrative.
# Depth is driven by result SHAPE + how much verified evidence exists — never by
# padding to a length target. Findings are precomputed deterministically
# (result_analyzer); the SLM only narrates them.
_ANALYTICAL_SHAPES = {"RANKING", "GROUPED", "DISTRIBUTION", "TREND", "PIVOT"}


def _summary_mode(result_shape: Optional[str], row_count: int) -> str:
    """'brief' | 'analytical'. Analytical only when the shape is genuinely analytical
    AND there is more than one group/row to discuss — otherwise brief."""
    if (result_shape in _ANALYTICAL_SHAPES) and row_count and row_count > 1:
        return "analytical"
    return "brief"


def _analytical_context_block(ctx: Optional[dict]) -> str:
    """Render the RESOLVED analytical context (decided UPSTREAM — operation, measure,
    dimension, display, ranking, temporal, explicit-id) so the narrator speaks to the
    user's real intent. Reused, never re-derived here. Empty when nothing supplied."""
    if not ctx:
        return ""
    order = [("intent", "intent"), ("operation", "operation"), ("measure", "measure"),
             ("dimension", "dimension"), ("display", "display field"),
             ("ranking", "ranking"), ("temporal", "time range"),
             ("explicit_identifier", "explicit id requested")]
    bits = [f"{label}={ctx.get(k)}" for k, label in order if ctx.get(k)]
    return ("\n\nResolved analytical context: " + "; ".join(bits)) if bits else ""


# ── Provenance, and the ungrounded-business-term guard ─────────────────────────
#
# Measured silent-wrong answer (evaluation/drilldown_l7, turn 21, 2026-09-24):
#
#   Q: "Who is the listing agent?"
#   SQL: SELECT DISTINCT t.first_name FROM assets_salenegotiation a
#        JOIN users_user t ON a.negotiator_id = t.id
#   A: "There are 4 listing agents named Ashutosh, Deepa, Demo, and Pritam."
#
# There is no agent concept anywhere in that schema. The summariser was handed
# the user's question and a bag of first names, and did the one thing this layer
# must never do: it restated the user's BUSINESS TERM as though the data had
# confirmed it, while never naming the column it actually read. The SQL's
# wrongness was completely invisible in the prose — no hedge, no provenance.
#
# Two mechanisms here, both deterministic (no SLM, no DB):
#   1. `_provenance_block` tells the model, in the prompt, exactly which
#      table.column each value came from and that it may not rename them.
#   2. `ungrounded_answer_term` is the backstop that does not depend on a 7B
#      model obeying an instruction: if the prose adopts a salient word from the
#      question that is grounded NOWHERE in the schema and appears nowhere in the
#      executed SQL, the fluent sentence is replaced by `ungrounded_term_answer`,
#      which says so and names the fields actually read.
#
# Fail-open by construction — every unknown (no SQL, unparseable SQL, resolver
# unavailable) means "no finding". This guard may only ever fire on positive
# evidence that a term is absent from the schema, never on absence of evidence.
# The upstream refusal gate (veda/validation.py::qualifier_completeness) does not
# catch this case on purpose: an unaccounted token that names no column of the
# queried tables and is no stored value is classified as FILLER there, to avoid
# false refusals on words like "database"/"system". "agent" lands in exactly that
# bucket. Widening that classification is a hot-path change to the refusal gate;
# this layer instead refuses to LAUNDER the term into a confident claim.

_UNGROUNDED_MIN_LEN = 4
_PROV_MAX_FIELDS = 6
_VALUES_IN_HONEST_ANSWER = 8


def _prov_of(sql: Optional[str]) -> dict:
    """sql_provenance(), never raising, empty dict-shape when unavailable."""
    try:
        from veda.business_explain import sql_provenance
        return sql_provenance(sql or "")
    except Exception:
        return {"projections": [], "tables": [], "join_keys": [], "ordered": False,
                "vocabulary": []}


def _provenance_block(prov: Optional[dict]) -> str:
    """Prompt block naming the fields the values actually came from. Empty when the
    provenance is unknown — an empty block is honest, an invented one is not."""
    if not prov or not prov.get("projections"):
        return ""
    line = ("\n\nData provenance — every value above was read from these fields and no "
            "others: " + ", ".join(prov["projections"][:_PROV_MAX_FIELDS]))
    if prov.get("join_keys"):
        line += " (reached through " + "; ".join(prov["join_keys"][:2]) + ")"
    line += (". Describe the values as what THOSE fields are. If the question names a "
             "business concept that is not one of those fields, do not answer as though "
             "the data confirmed it — say the data has no such field.")
    return line


def _words(text: Any) -> List[str]:
    return [w for w in re.findall(r"[a-z]+", str(text or "").lower()) if len(w) > 2]


def _singular(word: str) -> str:
    try:
        from retrieval.query_enrichment import _singularize
        return _singularize(word)
    except Exception:
        return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _operation_words() -> set:
    """Words that name an ANALYTICS OPERATION rather than a business entity —
    "average", "total", "minimum", "maximum". Read from veda/business_explain.py's
    existing SQL-function -> business-word map (`_AGG_WORD`, the one place this
    translation lives), never re-listed here. Measured need: "What is the average
    expected price of sale listings?" ran a correct AVG(expected_price), and
    'average' has no schema referent — without this the guard flagged a perfectly
    good answer. An operation word is not a claim about what the rows ARE."""
    try:
        from veda.business_explain import _AGG_WORD
        words = {str(k).lower() for k in _AGG_WORD} | {str(v).lower() for v in _AGG_WORD.values()}
    except Exception:
        words = set()
    # Comparison words ("above 1000", "at least 2k") name an OPERATION too — read from the
    # one list that grounds them (query/qualifier_grounding.py::_COMPARATORS). Measured
    # 2026-09-26: "carpet area above 1000" ran with the filter applied and was then refused
    # here on 'above'.
    try:
        from query.qualifier_grounding import _COMPARATORS
        words |= {w for phrase, _op in _COMPARATORS for w in phrase}
    except Exception:
        pass
    return words


def _query_content_words(query: str) -> List[str]:
    """The user's CONTENT words — what they asked FOR — using the same centralized
    query-LANGUAGE vocabulary the refusal gate uses (config.QUERY_GRAMMAR /
    QUERY_LANGUAGE via veda/validation.py::_gate_strip). No word list lives here.
    Empty on any failure, which disables the guard (fail-open)."""
    try:
        from veda.validation import _gate_strip
        strip = _gate_strip()
    except Exception:
        # veda.validation pulls in the runtime (DB/context) — unavailable in some
        # import contexts. Fall back to the SAME centralized vocabulary it reads,
        # straight from config; never to a word list written here.
        try:
            from config import QUERY_GRAMMAR, QUERY_LANGUAGE
            strip = set()
            for ops in QUERY_GRAMMAR.values():
                for w in ops:
                    strip.update(w.split())
            for cls in QUERY_LANGUAGE.values():
                strip.update(cls)
        except Exception:
            return []
    ops = _operation_words()
    # Pointer words the conversation layer already resolved into a row of the previous
    # answer ("second" of "show the second one") are not what was asked FOR — measured
    # 2026-09-25, a correctly resolved row was replaced by "this data has no 'second'".
    # One definition, shared with the ranking parser.
    try:
        from query.ranking_parser import _resolved_pointer_words
        pointer = _resolved_pointer_words()
    except Exception:
        pointer = set()
    out = []
    for w in _words(query):
        s = _singular(w)
        if w in strip or s in strip or len(s) < _UNGROUNDED_MIN_LEN:
            continue
        if w in ops or s in ops or w in pointer or s in pointer:
            continue
        if s not in out:
            out.append(s)
    return out


def _accounted_in(token: str, vocabulary: List[str]) -> bool:
    """Same substring-tolerant accounting the refusal gate uses (>=4 chars either
    way), so morphology (listing/listed, negotiator/negotiation) still matches."""
    if token in vocabulary:
        return True
    return any(len(v) >= 4 and (token in v or v in token) for v in vocabulary)


def _schema_referent(token: str, sm: Optional[dict]) -> Optional[bool]:
    """True / False / None(unknown) — does this token refer to ANYTHING in this
    scope's schema (a sampled value, an entity table, a column name)? Reuses
    query/resolution.py::has_schema_referent, the existing schema-derived
    referent test; None whenever it cannot be consulted, and None never fires."""
    try:
        from query.resolution import has_schema_referent
    except Exception:
        return None
    try:
        return bool(has_schema_referent(token, sm))
    except Exception:
        return None


def ungrounded_answer_term(question: str, answer: str, columns: List[str],
                           rows: List[dict], prov: Optional[dict],
                           sm: Optional[dict] = None) -> Optional[str]:
    """The business term the ANSWER adopted from the QUESTION that this data cannot
    support — or None (the overwhelmingly common case).

    All four conditions must hold, and any unknown means None:
      1. provenance is KNOWN (the SQL parsed and projects real fields),
      2. the term is a content word of the question that appears nowhere in the
         executed SQL, the result's columns, or the returned values,
      3. the term refers to nothing anywhere in the schema (has_schema_referent),
      4. the prose actually USES the term — i.e. it was laundered into the answer.
    """
    if not prov or not prov.get("projections") or not answer:
        return None
    vocab = list(prov.get("vocabulary") or [])
    for c in columns or []:
        for w in _words(c):
            if w not in vocab:
                vocab.append(w)
    for r in (rows or [])[:_FACTS_SAMPLE_ROWS]:
        for v in (r.values() if isinstance(r, dict) else []):
            if isinstance(v, str):
                for w in _words(v):
                    if w not in vocab:
                        vocab.append(w)
    vocab = [_singular(v) for v in vocab]
    answer_words = {_singular(w) for w in _words(answer)}
    content = _query_content_words(question)
    # Only the HEAD of what was asked for — the last content word of the question
    # ("the listing AGENT"). A modifier earlier in the phrase is describing a noun
    # that IS in the data ("the average expected PRICE", "the cheapest PROPERTY"),
    # and flagging those produced a measured false positive. Deliberately narrow:
    # this trades recall (a laundered term buried mid-question is missed) for never
    # refusing a question whose subject the data really does hold. Fail-open again.
    for token in content[-1:]:
        if _accounted_in(token, vocab):
            continue
        if token not in answer_words and not any(
                len(a) >= 4 and (token in a or a in token) for a in answer_words):
            continue
        if _schema_referent(token, sm) is False:
            return token
    return None


def ungrounded_term_answer(term: str, prov: dict, columns: List[str],
                           rows: List[dict]) -> str:
    """The honest replacement for a fluent answer that asserted `term`. Names the
    term as unsupported FIRST, then reports what was actually read, with the
    fields it came from — the provenance the wrong answer was missing."""
    fields = ", ".join((prov.get("projections") or [])[:_PROV_MAX_FIELDS])
    via = (prov.get("join_keys") or [])
    n = len(rows or [])
    head = (f"I can't answer that: this data has no \"{term}\" — no table, column or "
            f"value anywhere in it corresponds to that.")
    body = f" What the query actually returned is {n} row(s) of {fields}"
    if via:
        body += ", reached through " + "; ".join(via[:2])
    body += "."
    tail = ""
    if len(columns or []) == 1 and 0 < n <= _VALUES_IN_HONEST_ANSWER:
        col = columns[0]
        vals = [str(r.get(col)) for r in rows if isinstance(r, dict) and r.get(col) is not None]
        if vals:
            tail = f" Those values are: {', '.join(vals)}."
    return head + body + tail + (" That is not the same thing — please rephrase using a "
                                 "field that exists in this data.")


def _unordered_note(query: str, prov: Optional[dict], row_count: int) -> str:
    """A multi-row result with NO ORDER BY identifies no 'first'/'cheapest' row, yet
    the sample rows handed to the model look exactly like a ranked answer — measured
    on drilldown_l7 turn 1, where an unordered 373-row result was narrated as
    "First: status=APPROVED, expected_price=5500000". Says so in the prompt when the
    question asked for a specific one. Ranking vocabulary comes from the single
    source of truth (query/ranking_parser.py), never a list here."""
    if not prov or prov.get("ordered") or row_count < 2:
        return ""
    try:
        from query.ranking_parser import parse_ranking
        spec = parse_ranking(query or "")
        asked = bool(getattr(spec, "ranked", False) or getattr(spec, "top_n", None))
    except Exception:
        return ""
    if not asked:
        return ""
    return ("\nNOTE: these rows came back in NO particular order (the query has no "
            "ORDER BY), so the result does not identify a first/top/cheapest/largest "
            "one. Do not present any row as the one asked for — say the result is "
            "unordered instead.")


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
    analytical_context: Optional[dict] = None,
    sql:            Optional[str] = None,
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
    t0 = time.time()

    row_count = len(rows)

    # Nothing to phrase — a fixed literal, calling the SLM here would be pure
    # waste (there is no data to summarize differently).
    if row_count == 0:
        return NLAnswerResult(answer="No results found.", row_count=0,
                              duration_ms=round((time.time() - t0) * 1000, 2))

    facts = _extract_facts(columns, rows, rank_column=rank_column,
                           truncated=_sql_truncated(sql, len(rows), query))
    glossary = _column_glossary(columns, table, semantic_model)
    # What the values were ACTUALLY read from — put in front of the model (so it
    # describes those fields rather than the question's wording) and kept for the
    # deterministic ungrounded-term check after the call. Empty when `sql` is
    # absent or unparseable, which disables both.
    prov = _prov_of(sql)
    prov_line = _provenance_block(prov)
    unordered_line = _unordered_note(query, prov, row_count)

    rank_line = (f"\n\nThese rows are already ordered by \"{rank_column}\" — "
                f"refer to that field's values, not any id column, when describing rank/order."
                ) if rank_column else ""

    # Evidence-adaptive depth: a scalar/detail answer stays brief; a genuinely
    # analytical shape with multiple groups earns a short analytical narrative. Depth
    # comes from VERIFIED evidence, never from padding.
    _mode = _summary_mode(result_shape, row_count)
    _max_findings = NL_SUMMARY_MAX_FINDINGS if _mode == "analytical" else 2

    # Deterministic findings (result_analyzer detected these — precomputed, not for the
    # model to recompute or second-guess). Handed in so the SLM NARRATES the decision-
    # relevant ones. Analytical mode gets several; brief mode at most two.
    _pats = [str(p).strip().rstrip(".") for p in (patterns or []) if str(p).strip()]
    if facts.get("result_truncated"):
        # result_analyzer computes its findings over the ROWS IT WAS GIVEN, and has no
        # notion of truncation — so on a truncated page it emits shares of that page as
        # if they were shares of the data ("'False' accounts for 74% of corner_property
        # values", "79 of 100 rows have no facing value").
        #
        # They were then handed to the model as "Verified findings", one line below a
        # prompt telling it never to state a proportion. Measured 2026-09-22: the model
        # resolved that contradiction by narrating the proportion anyway ("86% of the
        # assets listed are in Pune", on a query whose WHERE already restricted every
        # row to Pune), and the deterministic fallback re-appended the same claim
        # through blend_patterns — so rejecting the model's prose alone did not remove
        # it. Dropped at the source instead: not shown to the model, not blended into
        # the fallback. Non-proportional findings (outliers, ranges) still pass.
        # Averages as well as shares. Measured 2026-09-22: after the proportion filter
        # landed, the guarded answer still carried "carpet_area has a high outlier
        # (34234.0 vs average 1628.46)" — the mean of one page of 7,814 rows, blended
        # back in from the findings after the model's own prose had been rejected. A
        # page mean is exactly as misleading as a page share; the outlier itself is a
        # value that genuinely occurs, so only findings that STATE one of these
        # derived figures are dropped.
        _kept = [p for p in _pats
                 if not _states_a_proportion(p) and not _AGGREGATE_WORD_RE.search(p)]
        if len(_kept) != len(_pats):
            logger.info("run_nl_answer: dropped %d page-derived finding(s) (share or "
                        "average) computed over a truncated page",
                        len(_pats) - len(_kept))
        _pats = _kept
    _pats = _pats[:_max_findings]
    findings_line = ("\n\nVerified findings already computed (narrate the decision-relevant "
                     "ones as insight; do not restate as a bare list): "
                     + "; ".join(_pats)) if _pats else ""

    ctx_line = _analytical_context_block(analytical_context)
    shape_hint = _SHAPE_GUIDANCE.get(result_shape or "", "")
    shape_line = f"\n{shape_hint}" if shape_hint else ""
    # When the metrics cover only the first N of a larger result, tell the narrator so
    # a partial SUM/COUNT is described as a sample, never as the full-population total.
    partial_line = ""
    if facts.get("metrics_partial"):
        partial_line = (
            f"\nNOTE: the metrics summarize only the first {facts.get('metrics_scanned')} "
            f"of {facts['row_count']} rows — describe any total/count/average from them as "
            f"based on that sample, and do NOT state a partial sum or count as the "
            f"full-result total.")
    if facts.get("result_truncated"):
        partial_line += (
            f"\nNOTE: the result was TRUNCATED — these {facts.get('rows_shown')} rows are one page "
            f"of a larger result and the true total is UNKNOWN. Never say 'out of "
            f"{facts.get('rows_shown')}', never state a percentage or proportion, and never "
            f"describe this page as 'the properties'/'the records'/'this dataset'. Answer only "
            f"about the rows shown.")

    if _mode == "analytical":
        role_line = (
            "You are a business analyst turning VERIFIED analytics into clear business "
            "language. Write a grounded analytical summary — about 3-5 concise sentences, "
            "as many as the verified findings justify and no more (do not pad):\n"
            "1. Directly answer the question first, leading with the most decision-relevant "
            "verified result (not a mechanical first metric).\n"
            "2. Then synthesize the VERIFIED findings above that matter — leaders/laggards, "
            "comparisons and gaps, spread/concentration, trends, outliers — using only those "
            "actually provided and relevant. Prioritize insight over listing every metric.\n"
        )
    else:
        role_line = (
            "You are a business analyst. Write a SHORT, direct answer (1-2 sentences):\n"
            "1. Directly answer the question, leading with the single most important number.\n"
            "2. Add a second sentence ONLY if a verified finding above is decision-relevant; "
            "otherwise stop — do NOT invent an insight or pad the response.\n"
        )

    prompt = (
        f"User question: {query}\n\n"
        f"Extracted data: {json.dumps(facts, default=str)}"
        f"{glossary}{ctx_line}{prov_line}{rank_line}{findings_line}{partial_line}"
        f"{unordered_line}{_STYLE_EXEMPLAR}\n\n"
        + role_line
        + f"{shape_line}\n"
        f"Speak in business terms using the column meanings above — name each entity by its "
        f"display value, not an id, UNLESS the question explicitly asked for an id/code/key "
        f"(then keep it). Use the data's OWN currency/units/dates exactly as shown — never "
        f"introduce a currency symbol or unit that isn't in the data. Do not repeat column "
        f"names verbatim, do not explain SQL, no markdown, no bullet points.\n"
        f"IMPORTANT: use ONLY numbers that appear above (data, metrics, findings — "
        f"totals/averages/min/max/rankings are already computed for you). Never calculate, "
        f"sum, average, or estimate a new figure, percentage, difference, or growth rate of "
        f"your own.\n"
        f"State ONLY what the data and findings show. Do NOT infer causes, reasons, or "
        f"business meaning that isn't given (never explain WHY a value is high/low/missing), "
        f"and avoid generic filler ('strong performance', 'positive momentum', 'significant "
        f"variation') unless a specific verified finding supports it."
    )

    _slm_timeout = NL_SUMMARY_TIMEOUT_MS / 1000.0 if timeout is None else timeout
    slm_used = False
    try:
        from slm import call_slm
        answer = call_slm(
            prompt,
            purpose="nl_answer",
            temperature=0.1,
            # Mode-aware budget: an analytical narrative (3-5 sentences weaving several
            # verified findings) needs more room; a brief answer keeps the tighter cap.
            # Still bounded — this prevents essays, it does not license unbounded output.
            num_predict=(NL_SUMMARY_ANALYTICAL_MAX_TOKENS if _mode == "analytical"
                         else NL_SUMMARY_MAX_TOKENS + 50),
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
        if facts.get("result_truncated") and _states_page_derived_total(answer, facts):
            # Same fallback as an ungrounded number: the deterministic answer says
            # "Returned N row(s)", which is true of what came back and claims nothing
            # about the table.
            logger.warning("run_nl_answer: summary presented a PAGE-derived figure as a "
                           "fact about the data (result was silently truncated) — "
                           "falling back. summary=%r", answer)
            raise ValueError("page-derived total on a truncated result")
        if facts.get("result_truncated") and _states_a_proportion(answer):
            # A share computed over one page is a statement about the page presented as
            # a statement about the data. Same fallback as an ungrounded number: the
            # deterministic answer below describes what was actually returned.
            logger.warning("run_nl_answer: summary stated a proportion over a TRUNCATED "
                           "page — falling back to deterministic answer. summary=%r",
                           answer)
            raise ValueError("proportion stated over a truncated result")
        # Deterministic backstop: drop any currency symbol the model prefixed that
        # the data doesn't actually carry (7B doesn't always obey the prompt rule).
        # Ungrounded-business-term backstop: the prose may not assert a concept the
        # schema has no referent for (measured: "listing agents" from
        # users_user.first_name). Replaced, not just logged — a fluent wrong answer
        # is the failure mode this whole layer is supposed to prevent.
        _bad_term = None
        try:
            _bad_term = ungrounded_answer_term(query, answer, columns, rows, prov,
                                               semantic_model)
        except Exception as _ug:          # never let the guard break summarisation
            logger.debug("run_nl_answer: ungrounded-term check skipped (%s)", _ug)
        if _bad_term:
            logger.warning("run_nl_answer: summary asserted the business term %r, which has "
                           "no referent in this schema and appears nowhere in the executed "
                           "SQL — replacing with the provenance-named answer. summary=%r",
                           _bad_term, answer)
            return NLAnswerResult(
                answer=ungrounded_term_answer(_bad_term, prov, columns, rows),
                row_count=row_count,
                duration_ms=round((time.time() - t0) * 1000, 2),
                slm_used=False)
        answer = _strip_invented_currency(answer, facts)
        answer = _regroup_numbers(answer)
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
    # Same provenance/ungrounded-term discipline as run_nl_answer — this is the
    # other summariser (INSIGHT_ENGINE_ENABLED picks one or the other), and a
    # guard that only covers one of them isn't a guard.
    prov = _prov_of(ctx.sql)
    prov_line = _provenance_block(prov)

    prompt = (
        f"User question: {ctx.question}\n\n"
        f"Extracted data: {json.dumps(facts, default=str)}"
        f"{stats_block}{patterns_block}{glossary}{grounding_block}{prov_line}"
        f"{rank_line}{shape_line}\n\n"
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
        try:
            _bad_term = ungrounded_answer_term(ctx.question, summary, ctx.columns,
                                               ctx.sample_rows, prov, ctx.semantic_model)
        except Exception:
            _bad_term = None
        if _bad_term:
            logger.warning("run_insight_engine: summary asserted the business term %r, which "
                           "has no referent in this schema — replacing with the "
                           "provenance-named answer. summary=%r", _bad_term, summary)
            summary = ungrounded_term_answer(_bad_term, prov, ctx.columns, ctx.sample_rows)
            insights = []          # they were narrated about the same ungrounded concept
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
