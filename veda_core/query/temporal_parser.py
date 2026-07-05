# =============================================================================
# query/temporal_parser.py
# VEDA POC — Query Pipeline Layer 1: Temporal Parser
#
# Extracts and normalises temporal expressions from a raw NL query.
# Returns a TemporalParserResult with:
#   - temporal_filter  : {start, end} as ISO 8601 strings, or None
#   - cleaned_query    : raw query with temporal tokens stripped (fed to L2)
#   - raw_expressions  : matched temporal text fragments (for debug)
#
# Handles:
#   Q1/Q2/Q3/Q4 YYYY  |  last N days/weeks/months/years
#   last/past/previous week/month/quarter/year
#   this/current week/month/quarter/year
#   yesterday / today
#   since/after <date>  |  before <date>
#   between <date> and <date>  |  from <date> to <date>
#   in/during/for YYYY  |  Month YYYY
#   recently / lately / latest / newest / most recent  → last 30 days
#   recently created/updated/modified                  → last 30 days
#   N days/hours/minutes ago                           → point-in-time to now
#   last N hours/minutes                               → rolling window
# =============================================================================

import re
import sys
import os
import calendar
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import dateparser as _dateparser_lib
    _DATEPARSER_AVAILABLE = True
except ImportError:
    _DATEPARSER_AVAILABLE = False


# =============================================================================
# Output structures
# =============================================================================

@dataclass
class TemporalFilter:
    start: Optional[str]   # ISO 8601  "2024-01-01T00:00:00"  or None
    end:   Optional[str]   # ISO 8601  "2024-01-31T23:59:59"  or None


@dataclass
class TemporalParserResult:
    temporal_filter:  Optional[TemporalFilter]
    cleaned_query:    str
    raw_expressions:  List[str] = field(default_factory=list)
    parser_available: bool      = True


# =============================================================================
# Internal helpers
# =============================================================================

def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _now() -> datetime:
    return datetime.now().replace(microsecond=0)


def _sod(dt: datetime) -> datetime:   # start-of-day
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _eod(dt: datetime) -> datetime:   # end-of-day
    return dt.replace(hour=23, minute=59, second=59, microsecond=0)


def _month_range(year: int, month: int) -> Tuple[datetime, datetime]:
    _, last_day = calendar.monthrange(year, month)
    return datetime(year, month, 1), datetime(year, month, last_day, 23, 59, 59)


def _year_range(year: int) -> Tuple[datetime, datetime]:
    return datetime(year, 1, 1), datetime(year, 12, 31, 23, 59, 59)


def _quarter_range(year: int, q: int) -> Tuple[datetime, datetime]:
    first_month = (q - 1) * 3 + 1
    last_month  = first_month + 2
    _, last_day = calendar.monthrange(year, last_month)
    return datetime(year, first_month, 1), datetime(year, last_month, last_day, 23, 59, 59)


def _last_n_units(n: int, unit: str) -> Tuple[datetime, datetime]:
    now  = _now()
    unit = unit.lower().rstrip("s")   # normalise plural
    if unit == "day":
        return _sod(now - timedelta(days=n)), now
    if unit in ("hour", "hr"):
        return now - timedelta(hours=n), now
    if unit in ("minute", "min"):
        return now - timedelta(minutes=n), now
    if unit == "week":
        return _sod(now - timedelta(weeks=n)), now
    if unit == "month":
        m, y = now.month - n, now.year
        while m <= 0:
            m += 12; y -= 1
        return datetime(y, m, 1), now
    if unit == "year":
        return datetime(now.year - n, now.month, now.day), now
    return _sod(now - timedelta(days=30 * n)), now


def _last_unit(unit: str) -> Tuple[datetime, datetime]:
    now   = _now()
    unit  = unit.lower()
    if unit in ("day", "yesterday"):
        d = now - timedelta(days=1); return _sod(d), _eod(d)
    if unit == "week":
        monday = _sod(now - timedelta(days=now.weekday() + 7))
        return monday, _eod(monday + timedelta(days=6))
    if unit == "month":
        fom = now.replace(day=1)
        last_prev = fom - timedelta(days=1)
        return datetime(last_prev.year, last_prev.month, 1), _eod(last_prev)
    if unit == "quarter":
        cq   = (now.month - 1) // 3 + 1
        pq   = cq - 1 if cq > 1 else 4
        yr   = now.year if cq > 1 else now.year - 1
        return _quarter_range(yr, pq)
    if unit == "year":
        return _year_range(now.year - 1)
    return _sod(now - timedelta(days=30)), now


def _this_unit(unit: str) -> Tuple[datetime, datetime]:
    now  = _now()
    unit = unit.lower()
    if unit == "week":
        monday = _sod(now - timedelta(days=now.weekday()))
        return monday, _eod(monday + timedelta(days=6))
    if unit == "month":
        return _month_range(now.year, now.month)
    if unit == "quarter":
        return _quarter_range(now.year, (now.month - 1) // 3 + 1)
    if unit == "year":
        return _year_range(now.year)
    return _sod(now), _eod(now)


_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3,   "apr": 4, "april": 4,
    "may": 5, "jun": 6,     "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8,  "sep": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _parse_date_str(text: str) -> Optional[datetime]:
    """Best-effort parse of a date string; dateparser as fallback."""
    text = text.strip()
    if not text:
        return None
    # Month YYYY
    m = re.match(r'^([A-Za-z]+)\s+(20\d{2}|19\d{2})$', text)
    if m:
        mnum = _MONTH_MAP.get(m.group(1)[:3].lower())
        if mnum:
            return datetime(int(m.group(2)), mnum, 1)
    # YYYY-MM-DD
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        pass
    # Bare year
    if re.match(r'^(20\d{2}|19\d{2})$', text):
        return datetime(int(text), 1, 1)
    # Fallback to dateparser
    if _DATEPARSER_AVAILABLE:
        parsed = _dateparser_lib.parse(
            text,
            settings={"RETURN_AS_TIMEZONE_AWARE": False, "PREFER_DAY_OF_MONTH": "first"},
        )
        return parsed
    return None


# =============================================================================
# Pattern registry — ordered from most specific to least specific
# =============================================================================

_PATTERNS = [
    # recently / lately / latest / newest / most recent — vague recency → last 30 days
    # Placed first (highest priority) so they don't conflict with other patterns.
    (re.compile(
        r'\b(?:recently|lately|just\s+(?:now|added|created|updated|modified)|'
        r'newly\s+(?:created|added|updated)|latest|newest|most\s+recent)\b',
        re.IGNORECASE,
    ), "recently"),
    # N days/hours/minutes ago — point-in-time to now
    (re.compile(
        r'\b(\d+)\s+(days?|hours?|hrs?|minutes?|mins?)\s+ago\b',
        re.IGNORECASE,
    ), "n_ago"),
    # between <date> and <date>  — captures "Jan 2023" or "2023-01-01" as a unit
    (re.compile(
        r'\bbetween\s+(\S+(?:\s+\d{2,4})?)\s+and\s+(\S+(?:\s+\d{2,4})?)(?=\s|$|[,.])',
        re.IGNORECASE,
    ), "between"),
    # from <date> to <date>
    (re.compile(
        r'\bfrom\s+(\S+(?:\s+\d{2,4})?)\s+to\s+(\S+(?:\s+\d{2,4})?)(?=\s|$|[,.])',
        re.IGNORECASE,
    ), "from_to"),
    # Q1–Q4 YYYY
    (re.compile(r'\bQ([1-4])\s*(20\d{2}|19\d{2})\b', re.IGNORECASE), "quarter"),
    # last N unit(s) — extended to hours and minutes
    (re.compile(r'\b(?:last|past)\s+(\d+)\s+(days?|hours?|hrs?|minutes?|mins?|weeks?|months?|years?)\b', re.IGNORECASE), "last_n"),
    # last/past/previous unit
    (re.compile(r'\b(?:last|past|previous)\s+(day|week|month|quarter|year)\b', re.IGNORECASE), "last_unit"),
    # this/current unit
    (re.compile(r'\b(?:this|current)\s+(week|month|quarter|year)\b', re.IGNORECASE), "this_unit"),
    # yesterday
    (re.compile(r'\byesterday\b', re.IGNORECASE), "yesterday"),
    # today
    (re.compile(r'\btoday\b', re.IGNORECASE), "today"),
    # since/after <date>
    (re.compile(
        r'\b(?:since|after)\s+([A-Za-z]+\s+(?:20|19)\d{2}|(?:20|19)\d{2}-\d{2}-\d{2}|(?:20|19)\d{2})\b',
        re.IGNORECASE,
    ), "since"),
    # before <date>
    (re.compile(
        r'\bbefore\s+([A-Za-z]+\s+(?:20|19)\d{2}|(?:20|19)\d{2}-\d{2}-\d{2})\b',
        re.IGNORECASE,
    ), "before"),
    # in/during/for YYYY
    (re.compile(r'\b(?:in|during|for)\s+(20\d{2}|19\d{2})\b', re.IGNORECASE), "year"),
    # Month YYYY
    (re.compile(
        r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        r'\s+(20\d{2}|19\d{2})\b',
        re.IGNORECASE,
    ), "month_year"),
    # standalone year (last resort)
    (re.compile(r'\b(20\d{2}|19\d{2})\b'), "year_bare"),
]


def _resolve_pattern(m: re.Match, handler: str) -> Optional[Tuple[datetime, datetime]]:
    """Returns (start, end) or None if pattern cannot be resolved."""
    now = _now()
    try:
        if handler == "recently":
            # Vague recency — default window: last 30 days ending now.
            # Generic across any domain — no schema-specific knowledge needed.
            return _sod(_now() - timedelta(days=30)), _now()
        if handler == "n_ago":
            n    = int(m.group(1))
            unit = m.group(2).lower().rstrip("s")
            if unit in ("hour", "hr"):
                point = _now() - timedelta(hours=n)
            elif unit in ("minute", "min"):
                point = _now() - timedelta(minutes=n)
            else:
                point = _sod(_now() - timedelta(days=n))
            return point, _now()
        if handler == "quarter":
            return _quarter_range(int(m.group(2)), int(m.group(1)))
        if handler == "last_n":
            return _last_n_units(int(m.group(1)), m.group(2))
        if handler == "last_unit":
            return _last_unit(m.group(1))
        if handler == "this_unit":
            return _this_unit(m.group(1))
        if handler == "yesterday":
            d = now - timedelta(days=1); return _sod(d), _eod(d)
        if handler == "today":
            return _sod(now), _eod(now)
        if handler == "since":
            p = _parse_date_str(m.group(1))
            return (p, now) if p else None
        if handler == "before":
            p = _parse_date_str(m.group(1))
            return (datetime(2000, 1, 1), p) if p else None
        if handler in ("year", "year_bare"):
            return _year_range(int(m.group(1)))
        if handler == "month_year":
            mnum = _MONTH_MAP.get(m.group(1)[:3].lower())
            return _month_range(int(m.group(2)), mnum) if mnum else None
        if handler in ("between", "from_to"):
            p1 = _parse_date_str(m.group(1))
            p2 = _parse_date_str(m.group(2))
            if p1 and p2:
                return (p1, p2)
            if p1:
                return (p1, now)
    except (ValueError, TypeError):
        pass
    return None


def _spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _strip_spans(query: str, spans: List[Tuple[int, int]]) -> str:
    """Remove matched spans (sorted descending) from query string."""
    result = query
    for start, end in sorted(spans, key=lambda x: x[0], reverse=True):
        result = result[:start] + result[end:]
    # Remove stray possessives left after stripping (e.g. "yesterday's" → "")
    result = re.sub(r"\s*'s?\b", " ", result)
    return re.sub(r"\s{2,}", " ", result).strip()


# =============================================================================
# Public entry point
# =============================================================================

def run_temporal_parser(query: str) -> TemporalParserResult:
    """
    Layer 1 entry point. Extracts temporal context from the raw NL query.

    Returns
    -------
    TemporalParserResult
        temporal_filter  — ISO start/end or None
        cleaned_query    — query with temporal tokens stripped (passed to L2)
        raw_expressions  — list of matched temporal text fragments
    """
    if not query or not query.strip():
        return TemporalParserResult(
            temporal_filter  = None,
            cleaned_query    = query or "",
            parser_available = _DATEPARSER_AVAILABLE,
        )

    raw_expressions: List[str]            = []
    spans:           List[Tuple[int,int]] = []
    start_dt:        Optional[datetime]   = None
    end_dt:          Optional[datetime]   = None

    for pattern, handler in _PATTERNS:
        for m in pattern.finditer(query):
            match_span = (m.start(), m.end())
            # Skip if this span overlaps with a previously accepted match.
            # Higher-priority patterns (earlier in _PATTERNS) win — prevents
            # "Q2 2024" from also matching the bare "2024" inside it.
            if any(_spans_overlap(match_span, s) for s in spans):
                continue
            result = _resolve_pattern(m, handler)
            if result is None:
                continue
            s, e = result
            # Union range across all temporal matches
            if start_dt is None or s < start_dt:
                start_dt = s
            if end_dt is None or e > end_dt:
                end_dt = e
            raw_expressions.append(m.group(0))
            spans.append(match_span)

    if start_dt is None and end_dt is None:
        return TemporalParserResult(
            temporal_filter  = None,
            cleaned_query    = query,
            raw_expressions  = [],
            parser_available = _DATEPARSER_AVAILABLE,
        )

    temporal_filter = TemporalFilter(
        start = _iso(start_dt) if start_dt else None,
        end   = _iso(end_dt)   if end_dt   else None,
    )
    cleaned = _strip_spans(query, spans)
    if not cleaned.strip():
        cleaned = query   # guard against fully stripping the query

    logger.debug("L1 temporal: matched %s → start=%s end=%s",
                 raw_expressions, temporal_filter.start, temporal_filter.end)

    return TemporalParserResult(
        temporal_filter  = temporal_filter,
        cleaned_query    = cleaned,
        raw_expressions  = raw_expressions,
        parser_available = _DATEPARSER_AVAILABLE,
    )


# =============================================================================
# Smoke test — python query/temporal_parser.py
# =============================================================================

if __name__ == "__main__":
    _tests = [
        "show all AML alerts flagged last month",
        "transactions in Q2 2024",
        "cases assigned to analyst since January 2024",
        "risk scores between Jan 2023 and June 2024",
        "total revenue last 30 days",
        "open incidents this quarter",
        "alerts before March 2024",
        "flagged cases in 2023",
        "yesterday's pending reviews",
        # new vague/relative patterns
        "show incidents created recently",
        "list the latest audit log entries",
        "show newly created users",
        "find records updated 3 days ago",
        "show changes from last 2 hours",
        "most recent workflow history",
        "show me everything",   # no temporal
    ]
    print("Layer 1 — Temporal Parser smoke test")
    print("=" * 60)
    for q in _tests:
        r = run_temporal_parser(q)
        tf = r.temporal_filter
        print(f"  Query   : {q!r}")
        if tf:
            print(f"  Filter  : {tf.start}  →  {tf.end}")
            print(f"  Matched : {r.raw_expressions}")
            print(f"  Cleaned : {r.cleaned_query!r}")
        else:
            print(f"  Filter  : None")
        print()