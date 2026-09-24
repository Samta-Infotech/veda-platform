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
# TEMPORAL ENTRIES COME FIRST AND THAT IS LOAD-BEARING: `basis` is set by the first
# match, so a recency phrase outranks "top" when a query carries both. "Show me the top
# 5 MOST RECENTLY dated accounting entries" must rank by date, not by a measure — and
# before 2026-09-23 it ranked by neither, because "most recent" is wrapped in \b...\b
# and `\bmost recent\b` does not match "most recentLY" (the char after "recent" is a
# word char). "top" then won, basis became "metric", and the ORDER BY was dropped
# entirely. The -ly / bare forms are spelled out rather than made optional in the regex
# so the matching stays one literal phrase per row.
# (phrase, direction, basis, PRIORITY) — priority, not list order, decides which basis
# wins when a query carries several ranking words, because neither pure ordering works:
#
#   "top 5 MOST RECENTLY dated entries"  -> must rank by DATE  (temporal beats "top")
#   "SMALLEST payments ... RECENTLY"     -> must rank by AMOUNT (a measure beats vague
#                                           recency; "recently" is a qualifier there)
#
# so specificity is the tiebreak, lowest number wins:
#   0  an explicit MEASURE superlative — names a magnitude direction outright
#   1  an explicit TEMPORAL phrase     — names a date direction outright
#   2  a GENERIC ranking word          — "top"/"last"/"most": a direction, no subject
#   3  VAGUE RECENCY                   — "recent"/"recently": weakest; a time qualifier
#                                        that only becomes the ranking basis alone
#
# Before 2026-09-23 "most recent" also silently MISSED on "most recentLY" — it is
# matched as \bmost recent\b and the following char is a word char — so "top" won by
# default and the ORDER BY was dropped. The -ly and bare forms are spelled out below.
_RANKING_WORDS = [
    # --- explicit measure superlatives (priority 0) ---
    ("most expensive",  "desc", "metric",   0),
    ("least expensive", "asc",  "metric",   0),
    ("cheapest",    "asc",  "metric",   0),
    ("dearest",     "desc", "metric",   0),
    ("highest",     "desc", "metric",   0),
    ("biggest",     "desc", "metric",   0),
    ("largest",     "desc", "metric",   0),
    ("greatest",    "desc", "metric",   0),
    ("maximum",     "desc", "metric",   0),
    ("max",         "desc", "metric",   0),
    ("lowest",      "asc",  "metric",   0),
    ("smallest",    "asc",  "metric",   0),
    ("least",       "asc",  "metric",   0),
    ("minimum",     "asc",  "metric",   0),
    ("min",         "asc",  "metric",   0),
    ("fewest",      "asc",  "metric",   0),
    ("fewer",       "asc",  "metric",   0),
    # --- explicit temporal phrases (priority 1) ---
    ("most recently", "desc", "temporal", 1),
    ("most recent", "desc", "temporal", 1),
    ("latest",      "desc", "temporal", 1),
    ("newest",      "desc", "temporal", 1),
    ("oldest",      "asc",  "temporal", 1),
    ("earliest",    "asc",  "temporal", 1),
    # --- generic ranking words (priority 2) ---
    ("top",         "desc", "metric",   2),
    ("bottom",      "asc",  "metric",   2),
    ("most",        "desc", "metric",   2),
    ("last",        "desc", "temporal", 2),
    ("first",       "asc",  "temporal", 2),
    # --- vague recency (priority 3) ---
    ("recently",    "desc", "temporal", 3),
    ("recent",      "desc", "temporal", 3),
]


# Explicit SORT verbs. Not ranking words — they name no end of the order and no count
# ("sorted by price" is neither "top" nor "5") — but they DO ask for an ORDER BY, which is
# what veda/ir_equivalence.py's unrequested-ordering rule needs to know. They live here so
# that rule has no second vocabulary of its own: this module exists precisely because the
# ranking words were once duplicated across generation.py and planning.py, each copy
# narrower than the last.
_SORT_VERBS = ["sorted", "sort", "rank", "ranked", "order", "ordered", "arrange", "arranged"]


@dataclass
class RankingSpec:
    top_n:     Optional[int]  # explicit row count the query named, or None
    ranked:    bool           # True if ANY ranking language was detected
    direction: str            # "desc" | "asc" — which end of the order
    basis:     Optional[str]  # "temporal" | "metric" | None — what to sort by
    sort_requested: bool = False     # the query asked for SOME ordering — a ranking word, or a
    #                                  bare sort verb ("sorted by price") that names no end and
    #                                  no count. `ranked` stays the narrower ranking-only signal.
    subject:   Optional[str] = None  # the noun the ranking is OVER ("top 5 PROPERTIES
    #                                  by number of payments" → "properties"). This is the
    #                                  GRAIN of a "top N X by <measure>" query; callers may
    #                                  resolve it to a table so the grain doesn't invert to
    #                                  the measured entity. None when no subject noun follows.


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
    direction, basis, top_n, subject = "desc", None, None, None

    best_prio = 99
    for phrase, d, b, prio in _RANKING_WORDS:
        p = re.escape(phrase)
        if not re.search(rf"\b{p}\b", ql):
            continue
        ranked = True
        if prio < best_prio:        # most SPECIFIC match wins, not the first one seen
            direction, basis, best_prio = d, b, prio
        if top_n is None:
            m = (re.search(rf"\b{p}\s+(?:of\s+)?{_NUM_PATTERN}\b", ql)
                 or re.search(rf"\b{_NUM_PATTERN}\s+(?:of\s+(?:the\s+)?)?{p}\b", ql))
            if m:
                top_n = _to_int(m.group(1))
        if subject is None and b == "metric":
            # The noun the ranking is OVER: the first content word after the magnitude
            # word (+ optional count / "of the"), stopping at a by/per/with/of/from
            # boundary. Reuses the SAME ranking vocabulary — no new keywords. Empty when
            # the ranking word is followed directly by a measure ("highest paid amount").
            sm = re.search(rf"\b{p}\s+(?:{_NUM_PATTERN}\s+)?(?:of\s+(?:the\s+)?)?"
                           rf"(?P<subj>[a-z]+(?:\s+[a-z]+){{0,2}}?)"
                           rf"\s+(?:by|per|with|of|from|across|and)\b", ql)
            if sm:
                subject = sm.group("subj")

    sort_requested = ranked or any(re.search(rf"\b{v}\b", ql) for v in _SORT_VERBS)

    return RankingSpec(top_n=top_n, ranked=ranked, direction=direction,
                       basis=basis, sort_requested=sort_requested, subject=subject)
