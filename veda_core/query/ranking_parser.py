# query/ranking_parser.py
# VEDA — shared ranking / top-N extraction.
#
# Single source of truth for "top 10", "latest 10", "10 latest", "bottom 5",
# "highest three", "oldest 20 records" style requests. Previously this was two
# independent, narrower regexes (veda/generation.py, veda/planning.py) that only
# recognized the literal word "top" before a digit — "latest 10", "last 20",
# "10 most recent" etc. silently lost the requested count. One parser now backs
# every SQL-construction path (single-table deterministic branches, the LLM
# single-table prompt, and the per-anchor aggregate planner).
import re
from dataclasses import dataclass
from typing import Optional

# Spelled-out counts, shared with every caller that used to keep its own copy.
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

_NUM_PATTERN = r"(\d+|" + "|".join(NUM_WORDS) + r")"

# (phrase, direction, basis) — checked in this order, so multi-word phrases
# ("most recent") are tried before the bare words they contain ("most").
# basis="temporal"  → rank by a date/time column (recency)
# basis="metric"    → rank by a measure column (magnitude)
_RANKING_WORDS = [
    ("most recent", "desc", "temporal"),
    ("latest",      "desc", "temporal"),
    ("newest",      "desc", "temporal"),
    ("last",        "desc", "temporal"),
    ("oldest",      "asc",  "temporal"),
    ("earliest",    "asc",  "temporal"),
    ("first",       "asc",  "temporal"),
    ("top",         "desc", "metric"),
    ("highest",     "desc", "metric"),
    ("biggest",     "desc", "metric"),
    ("largest",     "desc", "metric"),
    ("greatest",    "desc", "metric"),
    ("most",        "desc", "metric"),
    ("maximum",     "desc", "metric"),
    ("max",         "desc", "metric"),
    ("bottom",      "asc",  "metric"),
    ("lowest",      "asc",  "metric"),
    ("smallest",    "asc",  "metric"),
    ("least",       "asc",  "metric"),
    ("minimum",     "asc",  "metric"),
    ("min",         "asc",  "metric"),
    ("fewest",      "asc",  "metric"),
    ("fewer",       "asc",  "metric"),
]


@dataclass
class RankingSpec:
    top_n:     Optional[int]  # explicit row count the query named, or None
    ranked:    bool           # True if ANY ranking language was detected
    direction: str            # "desc" | "asc" — which end of the order
    basis:     Optional[str]  # "temporal" | "metric" | None — what to sort by


def _to_int(tok: str) -> int:
    return int(tok) if tok.isdigit() else NUM_WORDS[tok.lower()]


def parse_ranking(query: str) -> RankingSpec:
    """Detect a ranking request and its explicit count, if any.

    Recognizes the count either BEFORE or AFTER the ranking word/phrase
    ("latest 10", "10 latest", "top of 5", "5 of the top"), digit or
    spelled-out ("ten"), across a wide vocabulary (recency: latest/newest/
    last/oldest/earliest/first; magnitude: top/highest/bottom/lowest/...).

    Returns `ranked=False, top_n=None` for a query with no ranking language at
    all — callers gate any behavior change on `top_n is not None` (or `ranked`
    for the softer "is this a ranking-shaped query" signal), so an ordinary
    query is completely unaffected.
    """
    ql = f" {query.lower()} "
    ranked = False
    direction, basis, top_n = "desc", None, None

    for phrase, d, b in _RANKING_WORDS:
        p = re.escape(phrase)
        if not re.search(rf"\b{p}\b", ql):
            continue
        ranked = True
        if basis is None:   # first (i.e. highest-priority / longest-phrase) match wins
            direction, basis = d, b
        if top_n is None:
            m = (re.search(rf"\b{p}\s+(?:of\s+)?{_NUM_PATTERN}\b", ql)
                 or re.search(rf"\b{_NUM_PATTERN}\s+(?:of\s+(?:the\s+)?)?{p}\b", ql))
            if m:
                top_n = _to_int(m.group(1))

    return RankingSpec(top_n=top_n, ranked=ranked, direction=direction, basis=basis)
