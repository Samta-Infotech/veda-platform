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
        extra = {"key_column": str(key["column"])}
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
    hit = _ordinal(message)
    pos = hit[0] if hit else None
    # The words the user POINTED with — they name a row they saw, not data, and the engine
    # is told so (ConversationContext.resolved_terms) rather than asked to find them in SQL.
    terms: List[str] = []
    if hit:
        terms = [hit[1]] + ([hit[2]] if hit[2] in _POINTER_NOUNS else [])
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
        else:
            return None                          # several rows, no position: not resolved
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
