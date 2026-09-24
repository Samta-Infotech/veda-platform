"""Deterministic numeric-predicate grounder (D.1, 2026-09-23).

The value arbiter grounds CATEGORICAL spans ("open", "critical") against the sampled
value store. Nothing grounded NUMERIC comparisons, so "payments between 100 and 50,000"
and "properties priced above 10,000" reached SQL generation with the range carried only
in the English, and came back as an unfiltered dump: the live run of `question.txt` on
2026-09-23 answered Q10 with

    SELECT ... FROM "accounts_paymenttransaction" LIMIT 100

— no WHERE, no amount column even projected, explainability reporting
`"filters": {"applied": [], "summary": "No filters applied."}`, and the answer still
rendered at confidence 0.635 with an invented trend claim. That is the single most
damaging failure shape in the set: a reader cannot tell it from a real answer.

This module turns those phrases into grounded `IRFilter`-shaped predicates on a REAL
measure column of the anchor, or, when the anchor has several equally-plausible measures
and the question does not say which, into a clarify that NAMES them. It never guesses:
no measure column -> no filter -> the caller refuses rather than answering broadly.

Deterministic, no LLM, no word lists beyond the closed comparator grammar below.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# A number as people write it: 10,000 / 10000 / 10000.50 / .5
_NUM = r"\d[\d,]*(?:\.\d+)?|\.\d+"

# (regex, op) — longest / most specific first. Two-sided shapes come first so
# "between 100 and 50,000" is never read as a bare "100".
_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(rf"\bbetween\s+({_NUM})\s+and\s+({_NUM})", re.I), "BETWEEN"),
    (re.compile(rf"\bfrom\s+({_NUM})\s+to\s+({_NUM})", re.I), "BETWEEN"),
    (re.compile(rf"\bin\s+the\s+range\s+(?:of\s+)?({_NUM})\s*(?:-|to|and)\s*({_NUM})", re.I), "BETWEEN"),
    (re.compile(rf"\b(?:at\s+least|no\s+less\s+than|not\s+less\s+than|minimum\s+of)\s+({_NUM})", re.I), ">="),
    (re.compile(rf"\b(?:at\s+most|no\s+more\s+than|not\s+more\s+than|maximum\s+of|up\s+to)\s+({_NUM})", re.I), "<="),
    (re.compile(rf"\b(?:more\s+than|greater\s+than|larger\s+than|higher\s+than|above|over|exceeding|exceeds)\s+({_NUM})", re.I), ">"),
    (re.compile(rf"\b(?:less\s+than|fewer\s+than|lower\s+than|smaller\s+than|below|under)\s+({_NUM})", re.I), "<"),
    (re.compile(rf">=\s*({_NUM})"), ">="),
    (re.compile(rf"<=\s*({_NUM})"), "<="),
    (re.compile(rf">\s*({_NUM})"), ">"),
    (re.compile(rf"<\s*({_NUM})"), "<"),
]

#: Semantic types that can carry a magnitude comparison.
_MEASURE_TYPES = {"MONETARY", "METRIC", "MEASURE", "NUMERIC", "QUANTITY"}

#: Words that mean "this comparison is about money", used only to RANK real columns.
_MONEY_WORDS = {"price", "priced", "cost", "amount", "paid", "pay", "payment", "fee",
                "charge", "value", "worth", "expensive", "cheap", "budget", "rent",
                "salary", "revenue", "total", "balance"}


@dataclass
class NumericPredicate:
    op: str                      # ">" ">=" "<" "<=" "BETWEEN"
    low: float
    high: Optional[float]        # BETWEEN only
    phrase: str                  # the matched text, for explanation


def _to_num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_numeric_predicates(query: str) -> List[NumericPredicate]:
    """Every numeric comparison the query states, left to right, non-overlapping.

    A span already consumed by an earlier (more specific) pattern is not re-matched, so
    "between 100 and 50,000" yields ONE BETWEEN and not also "> 100"."""
    if not query:
        return []
    ql = query
    taken: List[Tuple[int, int]] = []
    out: List[NumericPredicate] = []

    def _overlaps(a, b):
        return any(not (b <= s or a >= e) for s, e in taken)

    for rx, op in _PATTERNS:
        for m in rx.finditer(ql):
            if _overlaps(m.start(), m.end()):
                continue
            try:
                if op == "BETWEEN":
                    lo, hi = _to_num(m.group(1)), _to_num(m.group(2))
                    if lo > hi:
                        lo, hi = hi, lo
                    out.append(NumericPredicate("BETWEEN", lo, hi, m.group(0)))
                else:
                    out.append(NumericPredicate(op, _to_num(m.group(1)), None, m.group(0)))
            except (ValueError, IndexError):
                continue
            taken.append((m.start(), m.end()))
    return out


def _typed_columns(anchor: str, sm: dict, types: set) -> List[str]:
    out = []
    for key, m in (sm.get("columns", {}) or {}).items():
        tbl, _, cn = key.partition(".")
        if tbl != anchor:
            continue
        if str((m or {}).get("semantic_type", "")).upper() in types:
            out.append(cn)
    return sorted(out)


def measure_columns(anchor: str, sm: dict, query: str = "") -> List[str]:
    """Columns a numeric comparison in `query` may bind to on `anchor`.

    THE NOUN BINDS THE TYPE. When the comparator carries a money noun — "PRICED above
    10,000", "amount over 500", "fee under 50" — the only admissible targets are
    MONETARY columns. Falling back to "any measure" there is how "priced above 10,000"
    reached `assets_asset.construction_year > 10000` (2026-09-24, Q9): that table has no
    monetary column at all, only METRIC ones (carpet_area, latitude, floor_number,
    construction_year), and a year happily accepts a 10,000 comparison while meaning
    nothing. With no monetary column the honest outcome is to decline — the caller then
    clarifies — never to compare against a plausible-looking number.

    With no money noun present, the historical behaviour stands: the model's curated
    measure list, else any measure-typed column."""
    money_noun = bool({w for w in re.findall(r"[a-z]+", (query or "").lower())} & _MONEY_WORDS)
    if money_noun:
        monetary = _typed_columns(anchor, sm, {"MONETARY"})
        return monetary            # possibly [] -> decline, by design
    tmeta = (sm.get("tables", {}) or {}).get(anchor, {}) or {}
    curated = [c for c in (tmeta.get("candidate_measure_columns") or [])]
    if curated:
        return curated
    return _typed_columns(anchor, sm, _MEASURE_TYPES)


def _col_tokens(anchor: str, col: str, sm: dict) -> set:
    toks = {w for w in re.findall(r"[a-z]+", col.lower()) if len(w) > 2}
    m = (sm.get("columns", {}) or {}).get(f"{anchor}.{col}", {}) or {}
    for field in ("business_role", "business_definition"):
        toks |= {w for w in re.findall(r"[a-z]+", str(m.get(field, "") or "").lower())
                 if len(w) > 2}
    for al in (m.get("aliases") or []):
        toks |= {w for w in re.findall(r"[a-z]+", str(al).lower()) if len(w) > 2}
    return toks


def resolve_column(query: str, anchor: str, sm: dict) -> Tuple[Optional[str], List[str]]:
    """The measure column a numeric comparison in `query` is about.

    Returns (column, candidates). `column` is None when the anchor has SEVERAL measures
    and the question does not lean towards any one of them — the caller must then clarify
    and name `candidates`, never pick one. Sole measure -> use it."""
    cands = measure_columns(anchor, sm, query)
    if not cands:
        return None, []
    if len(cands) == 1:
        return cands[0], cands

    qtoks = {w for w in re.findall(r"[a-z]+", (query or "").lower()) if len(w) > 2}
    # "priced"/"pricing" should reach a column named "price"
    qstems = qtoks | {w.rstrip("d") for w in qtoks} | {re.sub(r"ing$", "", w) for w in qtoks}

    scored = []
    for c in cands:
        ctoks = _col_tokens(anchor, c, sm)
        overlap = len(qstems & ctoks)
        money = len(qstems & _MONEY_WORDS & ctoks)
        scored.append((overlap + 0.5 * money, c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1], cands
    return None, cands


def build_filters(query: str, anchor: str, sm: dict
                  ) -> Tuple[List[dict], Optional[str], List[str]]:
    """(filters, clarify_msg, candidates).

    `filters` are dicts shaped like `query.value_arbiter.anchor_filters` output so the
    deterministic branch composes them the same way, with kind="numeric":
        {"column", "op", "value", "value2", "kind": "numeric", "reason"}
    `clarify_msg` is set ONLY when the question states a comparison but the anchor offers
    several measures and nothing says which — refuse-over-guess, with the real names."""
    preds = parse_numeric_predicates(query)
    if not preds:
        return [], None, []

    col, cands = resolve_column(query, anchor, sm)
    if col is None:
        if not cands:
            return [], None, []
        human = ", ".join(cands)
        return [], (f"I can apply that numeric range, but {anchor} has more than one "
                    f"amount it could apply to — {human}. Which one did you mean?"), cands

    out = []
    for p in preds:
        out.append({"column": col, "op": p.op, "value": p.low,
                    "value2": p.high, "kind": "numeric",
                    "reason": f"{p.phrase.strip()} -> {col} {p.op}"})
    return out, None, cands


def where_clause(filters: List[dict], alias: Optional[str] = None) -> str:
    """AND-composed numeric predicates. Literals are numbers this module parsed, never
    user text, so they are emitted directly; `validate_and_parameterize` downstream
    parameterises them exactly like every other deterministic-path literal."""
    prefix = f"{alias}." if alias else ""
    parts = []
    for f in filters:
        col = f'{prefix}"{f["column"]}"'
        if f["op"] == "BETWEEN":
            parts.append(f'{col} BETWEEN {_fmt(f["value"])} AND {_fmt(f["value2"])}')
        else:
            parts.append(f'{col} {f["op"]} {_fmt(f["value"])}')
    return " AND ".join(parts)


def _fmt(n) -> str:
    if n is None:
        return "NULL"
    return str(int(n)) if float(n).is_integer() else repr(float(n))


def explain(filters: List[dict]) -> str:
    return "; ".join(f["reason"] for f in filters)
