"""Result reference memory — "the second one", "its price", "those".

A follow-up that points at a row of the PREVIOUS answer needs to know which row. The
frame cannot say: it records the question (entity, filters, shape), not the rows the user
saw. Until this existed the only record of a result was `last_result` — the whole engine
result in the checkpoint, unbounded, with no notion of which column identifies a row — so
"the second one" had nothing to resolve against and was answered as a fresh question.

What is kept, and why only this:
  · the IDENTITY of each displayed row, in display order, capped at _MAX_ITEMS. A row of
    records is identified by its primary key; a row of a GROUP BY result by its group
    values. The engine states which result column that is (explain["result_key"],
    veda/business_explain.py::result_key) — this layer never guesses it from names.
  · never the row VALUES. A referenced row is always re-queried by its identity, so the
    answer is current data under the CURRENT turn's authorisation. Memory is context,
    never a cache and never a grant.

Resolution is deterministic and fails closed:
  · an ordinal ("the second one", "the 3rd", "the last one") selects by display position;
    out of range is an honest refusal, never the nearest row;
  · a reference to a result that had exactly ONE row selects that row;
  · anything else — several rows and no position — is NOT resolved here, and the turn
    proceeds exactly as it did before this module existed.
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

# How many displayed rows are remembered. A reference is to what the user can SEE; nobody
# says "the 400th one", and a bound keeps the slot small whatever the result size.
_MAX_ITEMS = 50
_MAX_VALUE_LEN = 200

# ── ordinal grammar ──────────────────────────────────────────────────────────────────────
# A closed grammar (like ranking_parser's NUM_WORDS), not a phrase list: every ordinal of
# English up to the retained bound, in word, suffix and "number N" forms. It is entity-,
# source- and customer-independent. "last" is a position, not a count.
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
    "last": -1,
}
_ORDINAL_RE = re.compile(
    r"(?:\b(?P<word>" + "|".join(_ORDINAL_WORDS) + r")\b"
    r"|\b(?P<num>\d{1,3})(?:st|nd|rd|th)\b"
    # "#" is not a word character, so it cannot sit after a \b — its own alternative.
    r"|(?:\b(?:number|no\.?)|#)\s*(?P<numbered>\d{1,3})\b)",
    re.IGNORECASE,
)
# A COUNT after the word ("first 10", "last 5 payments") makes it a ranking, not a
# position — that belongs to query/ranking_parser on the engine side. "one" is deliberately
# not a count here: "the first one" / "the last one" point at a single displayed row.
_COUNT_AFTER_RE = re.compile(r"^\s+(?:\d+|two|three|four|five|six|seven|eight|nine|ten)\b",
                             re.IGNORECASE)


# Nouns that name a ROW OF A RESULT rather than anything in the data. Generic by
# construction — no entity, source or customer word belongs here.
_POINTER_NOUNS = frozenset({"one", "ones", "row", "rows", "item", "items", "result",
                            "results", "entry", "entries", "record", "records"})
_NEXT_WORD_RE = re.compile(r"^\W*([A-Za-z0-9]+)")


def _ordinal(message: str) -> Optional[Tuple[int, str, Optional[str], bool]]:
    """(position, the ordinal token as typed, the word right after it, numbered form?)."""
    for m in _ORDINAL_RE.finditer(message or ""):
        tail = (message or "")[m.end():]
        nxt = _NEXT_WORD_RE.match(tail)
        nxt = nxt.group(1).lower() if nxt else None
        if m.group("word"):
            # "the first 10", "last 5 entries": a count follows → a ranking, skip it.
            if _COUNT_AFTER_RE.match(tail):
                continue
            return (_ORDINAL_WORDS[m.group("word").lower()], m.group("word"), nxt, False)
        n = int(m.group("num") or m.group("numbered"))
        if n >= 1:
            token = m.group("numbered") or m.group(0).strip()
            return (n, token, nxt, bool(m.group("numbered")))
    return None


# The singular POSSESSIVE only — "its price", "its location" point back at a thing by
# construction. Bare "it" does not ("is it possible to see all vendors?"), nor do
# "this"/"that", which as often introduce a new subject ("this month", "that city").
_SINGULAR_PRONOUN_RE = re.compile(r"\bits\b", re.IGNORECASE)


def points_at_the_one_row(ref: Optional[Dict[str, Any]], message: str,
                          frame: Optional[Dict[str, Any]]) -> bool:
    """"what is its amount?" when the answer on screen is exactly one record: "its" can
    only mean that record, so this is evidence of a continuation in itself."""
    return bool(ref and ref.get("kind") == "rows" and ref.get("complete")
                and len(ref.get("items") or []) == 1
                and frame and str(frame.get("entity") or "") == str(ref.get("entity") or "")
                and _SINGULAR_PRONOUN_RE.search(message or ""))


def _positions(message: str) -> List[int]:
    """EVERY position the message points at ("the first and third", "1st, 2nd and 5th"),
    in order, de-duplicated. Counts ("first 10") are skipped exactly as in _ordinal."""
    out: List[int] = []
    for m in _ORDINAL_RE.finditer(message or ""):
        tail = (message or "")[m.end():]
        if m.group("word"):
            if _COUNT_AFTER_RE.match(tail):
                continue
            p = _ORDINAL_WORDS[m.group("word").lower()]
        else:
            p = int(m.group("num") or m.group("numbered"))
            if p < 1:
                continue
        if p not in out:
            out.append(p)
    return out


def parse_ordinal(message: str) -> Optional[int]:
    """1-based display position the message points at, -1 for "last", None for none."""
    hit = _ordinal(message)
    return hit[0] if hit else None


def result_pointer(message: str) -> Optional[Tuple[int, List[str]]]:
    """A message that points at a ROW OF A RESULT by position — and names nothing else it
    could be about: "the second one", "the 3rd row", "number 2", "#4", "and the last?".

    Returned as (position, words-it-used-to-point). Such a message has no subject of its
    own, so it is evidence of a continuation in itself, and with no result to point at it
    cannot be answered at all. An ordinal followed by any OTHER word ("the first
    transaction of 2024", "the second listing") is not a pointer here: it may name its own
    subject, and only a turn already placed as a continuation resolves it."""
    hit = _ordinal(message)
    if not hit:
        return None
    pos, token, nxt, numbered = hit
    if not (numbered or nxt is None or nxt in _POINTER_NOUNS):
        return None
    terms = [token] + ([nxt] if nxt in _POINTER_NOUNS else [])
    if numbered:
        terms += [w for w in re.findall(r"[A-Za-z]+", message or "")
                  if w.lower() in ("number", "no")]
    return pos, terms


# ── building ─────────────────────────────────────────────────────────────────────────────
def _column_index(cols: List[Any], name: str) -> Optional[int]:
    names = [str(c) for c in cols]
    if name in names:
        return names.index(name)
    low = [n.lower() for n in names]
    return low.index(name.lower()) if name.lower() in low else None


def _clean(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v)
    return s[:_MAX_VALUE_LEN] if s.strip() else None


def build_reference(engine_result: Dict[str, Any],
                    source_id: Optional[Any]) -> Optional[Dict[str, Any]]:
    """The reference for an ANSWERED result, or None when its rows are not referable.

    None is the common case and is not a failure: a scalar total, a result without its
    key in the projection, a document answer — there is nothing to point at, and the
    caller then clears any older reference rather than leave one describing a result the
    user is no longer looking at."""
    # Stated by the engine beside `explain` (veda/business_explain.py::result_key).
    key = engine_result.get("result_key") or {}
    cols = engine_result.get("cols") or []
    rows = engine_result.get("rows") or []
    kind = key.get("kind")
    if kind not in ("rows", "groups") or not cols or not rows:
        return None

    items: List[Any] = []
    if kind == "rows" and key.get("pinned") is not None and not key.get("result_column"):
        # The query pinned the key (`WHERE id = %s`) without projecting it. That identifies
        # the row only when there IS exactly one — anything else is not referable.
        if len(rows) != 1 or not key.get("column"):
            return None
        items = [_clean(key["pinned"])]
        if items[0] is None:
            return None
        extra = {"key_column": str(key["column"])}
    elif kind == "rows":
        idx = _column_index(cols, str(key.get("result_column") or ""))
        if idx is None or not key.get("column"):
            return None
        for r in rows[:_MAX_ITEMS]:
            v = _clean(r.get(cols[idx]) if isinstance(r, dict) else
                       (r[idx] if idx < len(r) else None))
            if v is None:
                return None                     # a row with no identity: not referable
            items.append(v)
        extra = {"key_column": str(key["column"]),
                 **_selectors(cols, rows[:_MAX_ITEMS], skip=cols[idx])}
    else:
        spec = [(str(c.get("column") or ""), _column_index(cols, str(c.get("result_column") or "")))
                for c in (key.get("columns") or [])]
        if not spec or any(not c or i is None for c, i in spec):
            return None
        for r in rows[:_MAX_ITEMS]:
            group = {}
            for c, i in spec:
                v = _clean(r.get(cols[i]) if isinstance(r, dict) else
                           (r[i] if i < len(r) else None))
                if v is None:
                    group = {}                  # a NULL group value cannot be re-selected
                    break                       # by equality — keep the position, mark it
                group[c] = v
            items.append(group or None)
        extra = {"group_columns": [c for c, _ in spec]}

    return {
        "result_id": uuid.uuid4().hex[:12],
        "kind": kind,
        "entity": str(key.get("table") or engine_result.get("table") or ""),
        "source_id": source_id,
        "items": items,
        # The user saw every row only when the result was not cut at the bound. An ordinal
        # past the bound on an incomplete result is "not remembered", not "does not exist".
        "complete": len(rows) <= _MAX_ITEMS and not engine_result.get("truncated"),
        "row_count": len(rows),
        "created_at": int(time.time()),
        **extra,
    }


# ── selectors: what else a displayed record can be picked BY ─────────────────────────────
# A record is still re-queried by its KEY; these only decide WHICH key. Two kinds, both read
# off the result the user saw, never off a word list:
#   · labels — the values of the result's naming column ("One & Only House"), so "details of
#     One & Only House" selects that row. The naming column is the first text column of the
#     projection (other than the key) whose values tell the rows apart;
#   · orderable values — each numeric or date column, so "the cheapest one" / "the latest
#     one" can be answered from what was shown instead of asking a model to compute it.
_MAX_TEXT_COLUMNS = 8
_MAX_TEXT_LEN = 80
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def _cell(r: Any, cols: List[Any], i: int) -> Any:
    return r.get(cols[i]) if isinstance(r, dict) else (r[i] if i < len(r) else None)


def _selectors(cols: List[Any], rows: List[Any], *, skip: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not rows:
        return out
    values: Dict[str, List[Optional[float]]] = {}
    dates: Dict[str, List[Optional[str]]] = {}
    label_col = None
    for i, c in enumerate(cols):
        if c == skip:
            continue
        name = str(c).lower()
        if name == "id" or name.endswith("_id"):
            continue                           # an identifier is neither a label nor a measure
        cells = [_cell(r, cols, i) for r in rows]
        present = [x for x in cells if x is not None and str(x).strip()]
        if not present:
            continue
        if all(isinstance(x, (int, float)) and not isinstance(x, bool)
               or (isinstance(x, str) and _NUM_RE.match(x.strip())) for x in present):
            values[str(c)] = [float(x) if x is not None and str(x).strip() else None
                              for x in cells]
        elif all(_DATE_RE.match(str(x)) for x in present):
            dates[str(c)] = [str(x)[:32] if x is not None else None for x in cells]
        elif all(isinstance(x, str) for x in present):
            # Every shown TEXT column is kept (bounded), so a record can be named by any
            # value the user saw in it — not only by its naming column.
            texts = out.setdefault("texts", {})
            if len(texts) < _MAX_TEXT_COLUMNS:
                texts[str(c)] = [(_clean(x) or "")[:_MAX_TEXT_LEN] or None for x in cells]
            if (label_col is None
                    and len({x.strip().lower() for x in present}) >= max(2, int(0.8 * len(rows)))
                    and sum(len(x.strip()) for x in present) / len(present) >= 3):
                label_col = str(c)
                out["label_column"] = label_col
                out["labels"] = [_clean(x) for x in cells]
    if values:
        out["values"] = values
    if dates:
        out["dates"] = dates
    return out


def _words_of(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _find_words(hay: List[str], needle: List[str]) -> bool:
    n = len(needle)
    return n > 0 and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def _nameable(v: Any) -> bool:
    """A shown value long and specific enough to be NAMED: 4+ characters, not a number,
    not a boolean. Short or generic cells ("A", "12", "True") are never matched."""
    t = str(v or "").strip()
    return (len(t) >= 4 and not _NUM_RE.match(t)
            and t.lower() not in ("true", "false", "none", "null"))


def _shown_value_hits(ref: Dict[str, Any], message: str) -> List[Tuple[int, str, str]]:
    """(row/group index, column, value) for every SHOWN value the message names as a whole
    phrase — only the longest phrase(s) named, so "Green Valley Phase 2" beats "Green
    Valley". Rows: every remembered text column; groups: their group values."""
    msg = _words_of(message)
    hits: List[Tuple[int, str, str]] = []
    if ref.get("kind") == "rows":
        for col, vals in (ref.get("texts") or {}).items():
            for i, v in enumerate(vals or []):
                if v and _nameable(v) and _find_words(msg, _words_of(v)):
                    hits.append((i, col, v))
    elif ref.get("kind") == "groups":
        for i, item in enumerate(ref.get("items") or []):
            for col, v in (item or {}).items():
                if v and _nameable(v) and _find_words(msg, _words_of(v)):
                    hits.append((i, col, v))
    if not hits:
        return []
    longest = max(len(_words_of(v)) for _, _, v in hits)
    return [h for h in hits if len(_words_of(h[2])) == longest]


def names_a_row(ref: Optional[Dict[str, Any]], message: str,
                frame: Optional[Dict[str, Any]]) -> bool:
    """Does the message name a displayed record by its label ("details of One & Only
    House")? Evidence of a continuation in itself — the label came from the result on
    screen — but only for a label of 2+ words or 6+ characters, so a common short word
    that happens to be a label is not mistaken for one."""
    # Record listings only. On a GROUPED result a named value ("only the EAST ones") is a
    # drill the engine grounds itself, and treating it as a pick would pin the reference to
    # the old list and could steal a clarification answer ("EAST").
    if not (ref and ref.get("kind") == "rows" and frame
            and str(frame.get("entity") or "") == str(ref.get("entity") or "")):
        return False
    return bool(_shown_value_hits(ref, message))


_NUM_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10}
_SET_EDGE_RE = re.compile(
    r"\b(?P<edge>first|last|top|bottom)\s+(?P<n>\d{1,2}|" + "|".join(_NUM_WORDS) + r")\b"
    r"(?P<tail>(?:\s+(?:ones?|rows?|items?|results?|entries|records))?)\s*(?:of them)?"
    r"\s*[.?!]*\s*$", re.IGNORECASE)
_SET_ALL_RE = re.compile(
    r"\b(?:those|these|all|the)\s+(?P<n>\d{1,2}|" + "|".join(_NUM_WORDS) + r")\b"
    r"(?:\s+(?:ones?|rows?|items?|results?|entries|records|of them))?\s*[.?!]*\s*$",
    re.IGNORECASE)


def _count(tok: str) -> int:
    t = tok.lower()
    return _NUM_WORDS.get(t) or int(t)


def _key_filters(ref: Dict[str, Any], picked: List[int]) -> List[Dict[str, Any]]:
    col = ref.get("key_column")
    return [{"field": col, "column": col, "operator": "equals",
             "value": str(ref["items"][i]), "source": "result_reference"} for i in picked]


def _set_reference(ref: Dict[str, Any], message: str) -> Optional[Tuple[Any, ...]]:
    """"the first three", "the last two rows", "those three" — a SET of displayed records,
    kept as their keys. A count that is not what was shown ("those five" after three rows)
    is refused, never trimmed or padded."""
    n = len(ref["items"])
    m = _SET_EDGE_RE.search(message or "")
    if m:
        k = _count(m.group("n"))
        if k < 1 or k > n:
            return ("refuse", f"The previous answer showed {n} row{'s' if n != 1 else ''}, so "
                              f"I can't pick {k} of them from the {m.group('edge').lower()}.")
        picked = list(range(k)) if m.group("edge").lower() in ("first", "top") \
            else list(range(n - k, n))
        terms = [m.group("edge"), m.group("n")] + _words_of(m.group("tail"))
        return ("filters", _key_filters(ref, picked), terms)
    m = _SET_ALL_RE.search(message or "")
    if m:
        k = _count(m.group("n"))
        if k != n:
            return ("refuse", f"The previous answer showed {n} row{'s' if n != 1 else ''}, not "
                              f"{k} — which ones do you mean?")
        return ("filters", _key_filters(ref, list(range(n))), [m.group("n")])
    return None


def selects_rows(ref: Optional[Dict[str, Any]], message: str,
                 frame: Optional[Dict[str, Any]]) -> bool:
    """Does the message pick rows out of the current listing — a set ("the first three",
    "those four") or a record by name? Used to keep such a message from being taken as a
    request to redraw the whole result."""
    if not (ref and ref.get("kind") == "rows" and frame
            and str(frame.get("entity") or "") == str(ref.get("entity") or "")):
        return False
    if _SET_EDGE_RE.search(message or "") or _SET_ALL_RE.search(message or ""):
        return True
    if len(_positions(message)) >= 2:
        return True                          # "the first and third"
    if names_a_row(ref, message, frame) or _partial_label_hits(ref, message):
        return True
    # "which one is the cheapest?", "the most expensive one": a superlative over the rows on
    # screen, pointed at with "one(s)" and naming no other subject ("which CITY is the
    # cheapest" does not qualify). Only when the extreme can actually be read off a shown
    # column — otherwise it is left to the engine as before.
    return bool(re.search(r"\bones?\b", message or "", re.I)
                and _ranking_reference(ref, message, frame) is not None)


_MIN_WORDS = frozenset({"cheapest", "lowest", "smallest", "least", "minimum", "fewest",
                        "shortest"})
_MAX_WORDS = frozenset({"highest", "largest", "biggest", "greatest", "maximum", "most",
                        "costliest", "longest"})
_DATE_MAX = frozenset({"latest", "newest", "recent"})
_DATE_MIN = frozenset({"oldest", "earliest"})


def _ranking_reference(ref: Dict[str, Any], message: str,
                       frame: Optional[Dict[str, Any]]) -> Optional[Tuple[Any, ...]]:
    """"which one is the cheapest", "the most expensive one", "the latest one" — the
    extreme of a column the user SAW, picked here from the remembered values. The column
    is the one the message names; else the one the result was ranked by; else the result's
    only numeric (or date) column. Several candidates and no way to choose, or a tie at the
    extreme → not resolved / refused, never guessed."""
    words = _words_of(message)
    wset = set(words)
    if wset & (_DATE_MAX | _DATE_MIN):
        series, highest, word = ref.get("dates") or {}, bool(wset & _DATE_MAX), \
            next(w for w in words if w in _DATE_MAX | _DATE_MIN)
    elif wset & (_MIN_WORDS | _MAX_WORDS):
        series, highest = ref.get("values") or {}, None
        for w in words:                      # the first superlative decides ("most" > "least")
            if w in _MIN_WORDS or w in _MAX_WORDS:
                highest, word = w in _MAX_WORDS, w
                break
    else:
        return None
    if not series:
        return None
    # What the superlative is ABOUT: the words after it, up to "one(s)" or the end
    # ("the lowest amount one" → amount). "most"/"least" take one quality word first
    # ("most expensive", "least costly") that names no column. Anything left must name a
    # SHOWN column — "the highest rating" over rows that never showed a rating is not
    # answered from some other column.
    i = words.index(word)
    about = []
    for w in words[i + 1:]:
        if w in ("one", "ones"):
            break
        if w not in ("the", "a", "an", "of", "in", "is", "by", "with", "from", "them", "these",
                     "those", "all"):
            about.append(w)
    if word in ("most", "least") and about:
        about = about[1:]
    named = [c for c in series if _find_words(words, _words_of(c.replace("_", " ")))]
    if about and not any(set(_words_of(c.replace("_", " "))) & set(about) for c in named):
        return None
    ranked = [str(o.get("field")) for o in ((frame or {}).get("order_by") or [])
              if isinstance(o, dict) and str(o.get("field")) in series]
    col = (named[0] if len(named) == 1 else
           ranked[0] if not named and ranked else
           next(iter(series)) if not named and len(series) == 1 else None)
    if col is None:
        return None
    vals = [(i, v) for i, v in enumerate(series[col]) if v is not None]
    if not vals:
        return None
    best = (max if highest else min)(v for _, v in vals)
    picked = [i for i, v in vals if v == best]
    if len(picked) != 1:
        return ("refuse", f"{len(picked)} of the rows shown share the "
                          f"{'highest' if highest else 'lowest'} {col.replace('_', ' ')} — "
                          f"which one do you mean?")
    terms = [word] + words[i + 1:i + 2] if word in ("most", "least") else [word]
    return ("filters", _key_filters(ref, picked), terms + (["one"] if "one" in wset else []))


def _fold_word(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _partial_label_hits(ref: Optional[Dict[str, Any]], message: str) -> List[Tuple[int, str]]:
    """Rows whose NAMING-column value contains, as a contiguous run of whole words, the
    longest run the message shares with any of them — "infotech", "Friends Colony",
    "Infotech Towers" (plural folded). Only a run of 2+ words, or one word of 7+
    characters, counts, so a common word inside a long address never does."""
    if not ref or ref.get("kind") != "rows" or not ref.get("labels"):
        return []
    msg = [_fold_word(w) for w in _words_of(message)]
    best, hits = 0, []
    for i, lab in enumerate(ref.get("labels") or []):
        if not lab:
            continue
        lw = [_fold_word(w) for w in _words_of(lab)]
        run = 0
        for a in range(len(msg)):
            for b in range(len(lw)):
                k = 0
                while a + k < len(msg) and b + k < len(lw) and msg[a + k] == lw[b + k]:
                    k += 1
                if k > run and (k >= 2 or len(msg[a]) >= 7):
                    run = k
        if run == 0:
            continue
        if run > best:
            best, hits = run, [(i, lab)]
        elif run == best:
            hits.append((i, lab))
    return hits


def _partial_label_reference(ref: Dict[str, Any], message: str) -> Optional[Tuple[Any, ...]]:
    hits = _partial_label_hits(ref, message)
    if not hits:
        return None
    if len(hits) == 1:
        lab_words = {_fold_word(w) for w in _words_of(hits[0][1])}
        return ("filters", _key_filters(ref, [hits[0][0]]),
                [w for w in _words_of(message) if _fold_word(w) in lab_words])
    names = "; or ".join(f"\"{lab}\"" for _, lab in hits[:4])
    return ("refuse", f"More than one of the rows shown matches that — do you mean {names}?")


def _nth(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _name_reference(ref: Dict[str, Any], message: str) -> Optional[Tuple[Any, ...]]:
    """"details of One & Only House", "what about EAST?" — the ONE shown record (or group)
    whose value the message names. A value several rows share is not a pick but a filter
    (the engine grounds it as before) — except on the naming column, where two records
    with the same name are a question, never the first one."""
    hits = _shown_value_hits(ref, message)
    if not hits:
        return _partial_label_reference(ref, message)
    rows = sorted({i for i, _, _ in hits})
    terms = _words_of(hits[0][2])
    # Several DIFFERENT values named, each on exactly one row ("compare Infotech Tower and
    # Shivsai Apartment") → those rows as a set. One shared value is a filter, not a pick.
    by_value: Dict[str, set] = {}
    for i, _c, v in hits:
        by_value.setdefault(v.strip().lower(), set()).add(i)
    if (ref.get("kind") == "rows" and len(by_value) >= 2
            and all(len(ix) == 1 for ix in by_value.values())):
        picked = sorted({next(iter(ix)) for ix in by_value.values()})
        return ("filters", _key_filters(ref, picked),
                [w for v in by_value for w in _words_of(v)] + ["and"])
    if len(rows) == 1:
        if ref.get("kind") == "rows":
            return ("filters", _key_filters(ref, rows), terms)
        item = (ref.get("items") or [None])[rows[0]]
        if not item:
            return None
        return ("filters", [{"field": c, "column": c, "operator": "equals", "value": str(v),
                             "source": "result_reference"} for c, v in item.items()], terms)
    if (ref.get("kind") == "rows" and all(c == ref.get("label_column") for _, c, _ in hits)
            and len({v.strip().lower() for _, _, v in hits}) == 1):
        return ("refuse", f"{len(rows)} of the rows shown are called \"{hits[0][2]}\" — "
                          f"which one do you mean? (for example \"the {_nth(rows[1] + 1)} "
                          f"one\")")
    return None


# ── resolving ────────────────────────────────────────────────────────────────────────────
def resolve_reference(ref: Optional[Dict[str, Any]], message: str, *,
                      frame: Optional[Dict[str, Any]],
                      referential: bool) -> Optional[Tuple[Any, ...]]:
    """What a follow-up's reference selects from the previous result.

    Returns
      ("filters", [ {field, column, operator, value}, ... ], [pointer words])
                                                              — the row(s) selected
      ("refuse",  "<honest reason>")                          — a position that is not there
      None                                                    — not a reference this layer
                                                                 resolves; carry on as before
    `referential` is the caller's own anaphora signal; a message with neither that nor an
    ordinal never touches the reference, so a self-contained question is never narrowed
    to a row of the previous answer.
    """
    if not ref or not ref.get("items"):
        return None
    # The reference must describe the result the conversation is ON. A frame that moved to
    # another entity or source (a topic switch, a revoked source dropped by the RBAC check)
    # makes the reference stale — it is not resolved against, whatever the words say.
    if not frame or str(frame.get("entity") or "") != str(ref.get("entity") or ""):
        return None
    if (frame.get("source_id") is not None and ref.get("source_id") is not None
            and str(frame["source_id"]) != str(ref["source_id"])):
        return None

    items = ref["items"]
    _many = _positions(message)
    if len(_many) >= 2 and ref.get("kind") == "rows":
        n = len(items)
        idx = [n - 1 if p == -1 else p - 1 for p in _many]
        bad = [p for p, i in zip(_many, idx) if i < 0 or i >= n]
        if bad:
            return ("refuse", f"The previous answer had {ref.get('row_count', n)} rows, so "
                              f"there is no row {bad[0]} to pick. Which ones did you mean?")
        return ("filters", _key_filters(ref, idx),
                [t for t in re.findall(r"[A-Za-z0-9#]+", message or "")
                 if _ORDINAL_RE.fullmatch(t) or t.lower() in ("and",)])
    hit = _ordinal(message)
    pos = hit[0] if hit else None
    # The words the user POINTED with — they name a row they saw, not data, and the engine
    # is told so (ConversationContext.resolved_terms) rather than asked to find them in SQL.
    terms: List[str] = []
    if hit:
        terms = [hit[1]] + ([hit[2]] if hit[2] in _POINTER_NOUNS else [])
    if pos is None and referential and ref.get("kind") == "rows":
        # A record picked by something other than its position — a set, the extreme of a
        # shown column, or its name. Each returns None when it does not apply, so a message
        # none of them recognises continues exactly as before.
        for picker in (lambda: _set_reference(ref, message),
                       lambda: _ranking_reference(ref, message, frame),
                       lambda: _name_reference(ref, message)):
            hit2 = picker()
            if hit2 is not None:
                return hit2
    if pos is None:
        if not referential:
            return None
        # Only the POSSESSIVE points at the one row ("what is its price?"). Being a follow-up
        # is not enough: measured 2026-09-25 (demo drill chains C5, C7), a "go back" after a
        # level that returned exactly one row was pinned to that row — the row's group values
        # (corner_property=true, facing=east) were added as filters and the drill-up came back
        # wrong. "go back", "only the X ones" and every other follow-up are about the result
        # set, not about a single row of it.
        if not _SINGULAR_PRONOUN_RE.search(message or ""):
            return None
        if len(items) == 1 and ref.get("complete"):
            pos = 1                              # "its price" about a one-row answer
        elif ref.get("kind") == "rows" and len(items) > 1:
            # "what is its city?" with several records on screen: "its" means ONE of them
            # and nothing says which. Asked, not guessed — measured 2026-09-26, sent on as
            # a fresh question it came back as an unrelated federated answer.
            return ("refuse", f"The previous answer has {len(items)} rows — which one do "
                              f"you mean? (for example \"the 2nd one\", or its name)")
        else:
            return None                          # several groups, no position: not resolved
    n = len(items)
    index = n - 1 if pos == -1 else pos - 1
    if index < 0 or index >= n:
        if ref.get("complete"):
            return ("refuse", f"The previous answer had {ref.get('row_count', n)} "
                              f"row{'s' if ref.get('row_count', n) != 1 else ''}, so there is no "
                              f"row {pos} to pick. Which one did you mean?")
        return ("refuse", f"I only keep the first {n} rows of the previous answer, so I "
                          f"can't pick row {pos} from it. Could you narrow the question?")
    item = items[index]
    if item is None:
        return ("refuse", "That row's grouping value is empty, so it can't be selected on "
                          "its own. Could you describe it instead?")
    if ref.get("kind") == "rows":
        col = ref.get("key_column")
        return ("filters", [{"field": col, "column": col, "operator": "equals",
                             "value": str(item), "source": "result_reference"}], terms)
    return ("filters", [{"field": c, "column": c, "operator": "equals", "value": str(v),
                         "source": "result_reference"} for c, v in item.items()], terms)


# ── earlier results: "the 1st one from the price list" ───────────────────────────────────
# Every answered result the user could point at is remembered in a short, bounded history
# with the words of the question that produced it. A message that QUALIFIES its reference
# by one of those results ("from the price list", "in the earlier result") is resolved
# against that result instead of the current one. The qualifier is matched on the
# remembered question's own words — never a word list — and two matches are a question to
# the user, never a guess.
_MAX_RESULTS = 5
_FROM_RESULT_RE = re.compile(
    r"\b(?:from|in|of|on)\s+(?:the|that|those)\s+(?P<q>(?:[a-z0-9]+\s+){0,5}?)"
    r"(?P<noun>list|lists|result|results|answer|table|ranking)\b", re.IGNORECASE)
_EARLIER_WORDS = frozenset({"previous", "earlier", "other", "last", "prior", "before"})
_FIRST_WORDS = frozenset({"first", "original", "initial"})
_QUALIFIER_FILLER = frozenset({"the", "a", "an", "one", "ones", "that", "this", "those",
                               "these", "we", "saw", "you", "showed", "shown", "by"})


def _fold(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def remember_result(history: Optional[List[Dict[str, Any]]], ref: Optional[Dict[str, Any]],
                    question: str) -> List[Dict[str, Any]]:
    """`ref` at the front of the history, tagged with the question that produced it."""
    kept = [h for h in (history or []) if isinstance(h, dict) and h.get("items")]
    if not ref or not ref.get("items"):
        return kept[:_MAX_RESULTS]
    entry = {**ref, "question": str(question or "")[:300]}
    return ([entry] + [h for h in kept if h.get("result_id") != ref.get("result_id")]
            )[:_MAX_RESULTS]


def _describes(entry: Dict[str, Any], words: List[str]) -> bool:
    # The QUESTION's words (and the table's): results of one table share every column, so
    # columns cannot tell "the price list" from "the area list" — the question can.
    vocab = {_fold(w) for w in _words_of(entry.get("question"))}
    vocab |= {_fold(w) for w in _words_of(str(entry.get("entity") or "").replace("_", " "))}
    return all(_fold(w) in vocab for w in words)


def earlier_result(history: Optional[List[Dict[str, Any]]], message: str,
                   current: Optional[Dict[str, Any]]) -> Optional[Tuple[Any, ...]]:
    """("ref", entry, qualifier words) | ("refuse", reason) | None (no qualifier: resolve
    against the current result exactly as before)."""
    m = _FROM_RESULT_RE.search(message or "")
    if not m:
        return None
    q = [w for w in _words_of(m.group("q")) if w not in _QUALIFIER_FILLER]
    terms = _words_of(m.group(0))
    entries = [h for h in (history or []) if isinstance(h, dict) and h.get("items")]
    if not entries:
        return None
    cur_id = (current or {}).get("result_id")
    if q and set(q) <= _EARLIER_WORDS:
        older = [h for h in entries if h.get("result_id") != cur_id]
        if not older:
            return ("refuse", "There's no earlier list in this conversation to pick from.")
        return ("ref", older[0], terms)
    if q and set(q) <= _FIRST_WORDS:
        return ("ref", entries[-1], terms)
    if not q:
        return None                                   # "from the list": the current one
    hits = [h for h in entries if _describes(h, q)]
    if len(hits) == 1:
        return ("ref", hits[0], terms)
    if not hits:
        return ("refuse", f"I don't have an earlier list about \"{' '.join(q)}\" in this "
                          f"conversation. Which list do you mean?")
    asked = "; or ".join(f"\"{h.get('question')}\"" for h in hits[:3])
    return ("refuse", f"More than one earlier list fits \"{' '.join(q)}\" — do you mean {asked}?")
