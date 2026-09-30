"""Ground FLAG and YEAR qualifiers of a drill-down follow-up onto the anchor table.

The value arbiter grounds words that are sampled VALUES ("Nagpur", "FULL"). Two qualifier
kinds are not values and slipped through it, silently:

  * a FLAG — "only the gated ones" names the boolean column `is_gated`; its values are
    true/false, never "gated";
  * a YEAR — "only the ones built in 2026" names a number, and numeric columns are not
    sampled.

Measured 2026-09-24: both came back with the PREVIOUS turn's SQL unchanged — the word was
dropped and the answer looked right. (qualifier_completeness did not catch it either: it
only reads alphabetic tokens, so a year is invisible to it, and "gated" is absorbed by the
table's business description.)

Everything here is data-driven: flag columns are the anchor's columns whose sampled values
are exactly {true, false}; year columns are the anchor's columns whose name or aliases say
"year". No word lists of domain terms. When a year cannot be pinned to ONE column the
caller refuses rather than guesses.

Pure functions; the caller supplies the flag columns and the semantic model.
"""
import re
from typing import Dict, Iterable, List, Optional, Tuple

_FLAG_PREFIXES = ("is_", "has_", "can_", "allows_", "allow_")
_NEGATORS = ("not", "non", "without", "no")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2}|2100)\b")
_COMPARISON_RE = re.compile(r"\b(vs\.?|versus|compared?\s+(to|with)|or\s+not)\b", re.I)


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def flag_phrase(column: str) -> List[str]:
    """`is_gated` → ["gated"]; `power_backup` → ["power", "backup"]."""
    name = column.lower()
    for p in _FLAG_PREFIXES:
        if name.startswith(p):
            name = name[len(p):]
            break
    return [w for w in name.split("_") if w]


def _find_phrase(words: List[str], phrase: List[str]) -> Optional[int]:
    """Index where `phrase` starts in `words` (whole words, plural-tolerant on the last)."""
    n = len(phrase)
    last = phrase[-1]
    forms = {last, last + "s", last + "es"}
    if last.endswith("y"):
        forms.add(last[:-1] + "ies")
    for i in range(len(words) - n + 1):
        window = words[i:i + n]
        if window[:-1] == phrase[:-1] and window[-1] in forms:
            return i
    return None


def ground_flags(message: str, flag_columns: Iterable[str]) -> List[dict]:
    """FLAG filters the message names, as value-arbiter filter dicts.

    "only the gated ones" → is_gated = true;  "only the non-gated ones" / "not gated" →
    is_gated = false. A message that names a flag in BOTH senses, or phrases a comparison
    ("gated vs non-gated"), is a comparison, not a filter — nothing is grounded.
    """
    if _COMPARISON_RE.search(message or ""):
        return []
    words = _words(message)
    out: List[dict] = []
    for col in sorted(flag_columns):
        phrase = flag_phrase(col)
        if not phrase:
            continue
        i = _find_phrase(words, phrase)
        if i is None:
            continue
        negated = i > 0 and words[i - 1] in _NEGATORS
        # the same flag named again later in the other sense → a comparison
        rest = words[i + len(phrase):]
        j = _find_phrase(rest, phrase)
        if j is not None and (j > 0 and rest[j - 1] in _NEGATORS) != negated:
            continue
        val = "false" if negated else "true"
        out.append({"column": col, "op": "=", "value": val, "value_norm": val,
                    "grounded_as": "flag",
                    # the negating word is now represented BY the predicate's value; the
                    # caller tells the qualifier gate so it does not refuse on "non"
                    "consumed": [words[i - 1]] if negated else []})
    return out


def year_columns(table: str, sm: dict) -> List[Tuple[str, List[str]]]:
    """[(column, alias_words)] for the anchor's columns whose name or aliases say 'year'."""
    out = []
    for key, meta in (sm.get("columns") or {}).items():
        if not key.startswith(table + "."):
            continue
        col = key.split(".", 1)[1]
        aliases = [str(a).lower() for a in (meta or {}).get("aliases") or []]
        if "year" in col.lower().split("_") or any("year" in a.split() for a in aliases):
            words = set(col.lower().split("_"))
            for a in aliases:
                words |= set(a.split())
            out.append((col, sorted(words)))
    return out


_GENERIC_YEAR_WORDS = {"year", "years", "date", "dates", "the", "in", "of", "on", "for",
                       "ones", "only", "just"}


def named_year_column(message: str, table: str, sm: dict) -> Optional[str]:
    """The ONE year column the user's own words name ("built" ↔ alias "year built"), or
    None. Generic words ("year", "date", "in") name no particular column. Used when L1
    already read the year as a date range: the range lands on the table's canonical
    timestamp (created_at), which answers "created in 2026", not "built in 2026"."""
    msg_words = set(_words(message)) - _GENERIC_YEAR_WORDS
    hits = [c for c, ws in year_columns(table, sm)
            if msg_words & (set(ws) - _GENERIC_YEAR_WORDS)]
    return hits[0] if len(hits) == 1 else None


def ground_year(message: str, table: str, sm: dict) -> Tuple[List[dict], Optional[str]]:
    """(filters, refusal). A 4-digit year in the message → `<year column> = <year>`.

    One year column on the anchor → use it. Several → the one whose name/aliases share a
    word with the message ("built" ↔ alias "year built"); still several or none → refusal
    text, never a guess. No year in the message → ([], None).
    """
    years = _YEAR_RE.findall(message or "")
    if not years:
        return [], None
    if len(set(years)) > 1:
        return [], (f"I can't apply more than one year ({', '.join(sorted(set(years)))}) "
                    f"as a filter here — please ask for one year at a time.")
    year = years[0]
    cands = year_columns(table, sm)
    if len(cands) > 1:
        named = named_year_column(message, table, sm)
        cands = [(c, ws) for c, ws in cands if c == named]
    if len(cands) != 1:
        return [], (f"I couldn't tell which column the year {year} applies to on this "
                    f"table — please name it (e.g. 'built in {year}').")
    col = cands[0][0]
    return [{"column": col, "op": "=", "value": year, "value_norm": year,
             "grounded_as": "year"}], None


# ── NUMERIC comparisons ──────────────────────────────────────────────────────────────
# "only the ones with carpet area above 1000" — the arbiter grounds sampled VALUES, and a
# threshold is not one, so the completeness gate refused on "above" (measured 2026-09-26,
# audit D1 level 5). A comparator is closed English grammar; the COLUMN comes from the
# semantic model: a column of the anchor whose allowed aggregations make it numeric
# (SUM/AVG), named in the message by its own words or one of its aliases.
_COMPARATORS = [  # longest first, so "greater than or equal to" wins over "greater than"
    (("greater", "than", "or", "equal", "to"), ">="), (("less", "than", "or", "equal", "to"), "<="),
    (("at", "least"), ">="), (("at", "most"), "<="), (("no", "more", "than"), "<="),
    (("no", "less", "than"), ">="), (("more", "than"), ">"), (("greater", "than"), ">"),
    (("less", "than"), "<"), (("fewer", "than"), "<"), (("above",), ">"), (("over",), ">"),
    (("exceeding",), ">"), (("below",), "<"), (("under",), "<"),
]
_NUMBER_RE = re.compile(r"^\d[\d,]*(?:\.\d+)?k?$", re.I)


def _to_number(tok: str) -> Optional[str]:
    t = tok.lower().replace(",", "")
    mult = 1000 if t.endswith("k") else 1
    t = t[:-1] if mult != 1 else t
    try:
        n = float(t) * mult
    except ValueError:
        return None
    return str(int(n)) if n == int(n) else str(n)


def numeric_columns(table: str, sm: dict) -> List[Tuple[str, List[List[str]]]]:
    """(column, [its name words, and each alias as words]) for the anchor's numeric columns."""
    out = []
    for key, meta in ((sm or {}).get("columns") or {}).items():
        t, _, col = str(key).partition(".")
        if t != table or not isinstance(meta, dict):
            continue
        aggs = {str(a).upper() for a in (meta.get("allowed_aggregations") or [])}
        if not aggs & {"SUM", "AVG"}:
            continue
        names = [_words(col.replace("_", " "))]
        names += [_words(a) for a in (meta.get("aliases") or []) if _words(a)]
        out.append((col, names))
    return out


def ground_numeric(message: str, table: str, sm: dict) -> List[dict]:
    """`<column> <comparator> <number>` → [{column, op, value}]. The column must be the ONE
    numeric column of the anchor whose name or alias appears in the words before the
    comparator (the longest such name decides between columns that share a word); none or
    several → [] and the turn is handled exactly as before this existed."""
    # Numbers kept whole ("1,000", "2.5k"); _words would split "1,000" into "1", "000".
    words = re.findall(r"\d[\d,]*(?:\.\d+)?k?\b|[a-z]+", (message or "").lower())
    for i in range(len(words)):
        for phrase, op in _COMPARATORS:
            n = len(phrase)
            if tuple(words[i:i + n]) != phrase or i + n >= len(words):
                continue
            num_tok = words[i + n]
            if not _NUMBER_RE.match(num_tok):
                continue
            value = _to_number(num_tok)
            before = words[:i]
            best, best_len, tie = None, 0, False
            for col, names in numeric_columns(table, sm):
                for nm in names:
                    if nm and _find_phrase(before, nm) is not None:
                        if len(nm) > best_len:
                            best, best_len, tie = (col, nm), len(nm), False
                        elif len(nm) == best_len and best and best[0] != col:
                            tie = True
            if value is None or best is None or tie:
                return []
            return [{"column": best[0], "op": op, "value": value, "value_norm": value,
                     "grounded_as": "numeric",
                     "consumed": list(best[1]) + list(phrase) + [num_tok]}]
    return []
