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
    # Price superlatives. These already existed in THREE other copies — config.py's
    # `superlative_min`, and fast_path's _SUPERLATIVE_ASC / _SUP_ASC — but not here, which
    # is the fragmentation this module was written to end. Measured 2026-09-24: "Show me
    # the 5 cheapest ones" parsed to nothing, so the SQL got no ORDER BY *and* no LIMIT and
    # returned all 373 rows with the summariser narrating an arbitrary first one. The
    # identical "5 lowest ones" was correct. Multi-word forms lead, so "most expensive"
    # is tried before the bare "most" below.
    ("most expensive",  "desc", "metric"),
    ("least expensive", "asc",  "metric"),
    ("most affordable", "asc",  "metric"),
    ("lowest priced",   "asc",  "metric"),
    ("highest priced",  "desc", "metric"),
    ("cheapest",    "asc",  "metric"),
    ("priciest",    "desc", "metric"),
    ("costliest",   "desc", "metric"),
    ("dearest",     "desc", "metric"),
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


def _resolved_pointer_words() -> set:
    """Words of THIS turn's message the conversation layer already resolved into a row of
    the previous answer ("first" / "one" of "the first one" — now an id or group filter,
    ConversationContext.resolved_terms). Ranking must not read them a second time:
    "the first one" is a position the user saw, not "earliest, LIMIT 1" — measured
    2026-09-25, a group pick came back as the oldest single record of that group. Empty
    for every caller with no conversation context, which leaves parsing unchanged."""
    try:
        from veda_core.context import current_conversation_context
        ctx = current_conversation_context() or {}
    except Exception:
        return set()
    return {t.lower() for t in (ctx.get("resolved_terms") or [])
            if isinstance(t, str) and t.strip()}


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
    _consumed = _resolved_pointer_words()
    if _consumed:
        query = re.sub(r"\b(?:" + "|".join(re.escape(w) for w in sorted(_consumed)) + r")\b",
                       " ", query or "", flags=re.IGNORECASE)
    ql = f" {query.lower()} "
    ranked = False
    direction, basis, top_n, subject = "desc", None, None, None

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
