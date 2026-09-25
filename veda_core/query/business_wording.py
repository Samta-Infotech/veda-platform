"""Business wording for the answer text — no raw table or column names in front of the user.

Measured 2026-09-26 on "What is the distribution of properties by facing?": the summary
read "The corner_property field shows that 64% of assets are in corners ... The
assets_asset_count ranges widely from 1 to 471". Two sources put those names there: the
deterministic findings in veda/result_analyzer.py are templated on the raw column name
("{m_col} ranges from ..."), and the summary model echoes the column names it was shown.
Either way the reader sees schema, not business language.

This is the last step on the answer text, applied once at the engine's front door
(veda_hybrid.run_hybrid_query) so every head — Tier-1, Tier-2, federated, hybrid — is
covered by the same rule. It replaces ONLY identifiers that are provably schema:
  · the result's own columns and table, and the tables/columns the semantic model knows;
  · and only when they contain an underscore — a one-word column ("facing", "status") is
    already an English word and is left exactly as written.
Everything else in the text is untouched, numbers included. The labels are derived, never
looked up in a hand-written list:
  · a column             -> its words:                corner_property  -> corner property
  · a COUNT alias        -> "number of <entities>":   assets_asset_count -> number of assets
  · a SUM/AVG/MIN/MAX alias -> "<total|average|..> <column words>"
  · a table              -> the semantic model's business name (business_explain)
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

_IDENT_CHARS = r"A-Za-z0-9_"
_AGG_PREFIX = {"total": "total", "sum": "total", "avg": "average", "average": "average",
               "mean": "average", "min": "minimum", "max": "maximum"}
_AGG_SUFFIX = {"total": "total", "sum": "total", "avg": "average", "average": "average",
               "min": "minimum", "max": "maximum"}


def _words(name: str) -> str:
    return " ".join(w for w in str(name).split("_") if w).lower()


_ARTICLE_RE = re.compile(r"^(?:an?|the|one|single|each)\s+", re.I)
# Where a descriptive sentence stops naming the thing and starts qualifying it: "A lease
# listing FOR an asset", "A user OF the system". English grammar, not a vocabulary list.
_QUALIFIER_RE = re.compile(r"\s+(?:for|of|in|on|with|within|between|among|under|across|about|per|that|which|who|whose|from|by|to|at)\s+.*$",
                           re.I)


def _plural(phrase: str) -> str:
    words = phrase.split()
    if not words:
        return phrase
    last = words[-1]
    if not last.endswith("s"):
        if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
            last = last[:-1] + "ies"
        elif last.endswith(("x", "z", "ch", "sh")):
            last += "es"
        else:
            last += "s"
    return " ".join(words[:-1] + [last])


def _table_label(table: str, sm: Optional[dict]) -> str:
    """The entity a table holds, plural and lower-case: the semantic model's own
    primary_entity sentence cut to its noun phrase ("A lease listing for an asset." ->
    "lease listings"); the table name's words when the model has none."""
    entity = str((((sm or {}).get("tables") or {}).get(table) or {}).get("primary_entity")
                 or "").strip().rstrip(".")
    phrase = entity
    while True:
        cut = _ARTICLE_RE.sub("", phrase)
        if cut == phrase:
            break
        phrase = cut
    phrase = _QUALIFIER_RE.sub("", phrase).strip().lower()
    if not phrase:
        phrase = _words(table.split("_", 1)[1] if "_" in table else table)
    return _plural(phrase)


def _entity_label(prefix: str, tables: Iterable[str], sm: Optional[dict]) -> str:
    """What a COUNT of `prefix` counts: a table's business name when it is a table,
    else the prefix's own words made plural."""
    if prefix in set(tables):
        return _table_label(prefix, sm)
    return _plural(_words(prefix))


def column_label(col: str, tables: Iterable[str] = (), sm: Optional[dict] = None) -> str:
    """Plain-language name for a result column (see module docstring)."""
    c = str(col)
    low = c.lower()
    tables = list(tables)
    if low.endswith("_count") and len(low) > len("_count"):
        return "number of " + _entity_label(c[: -len("_count")], tables, sm)
    for pre in ("count_of_", "number_of_", "num_"):
        if low.startswith(pre) and len(low) > len(pre):
            return "number of " + _entity_label(c[len(pre):], tables, sm)
    head, _, rest = low.partition("_")
    if rest and head in _AGG_PREFIX:
        return f"{_AGG_PREFIX[head]} {_words(rest)}"
    base, _, tail = low.rpartition("_")
    if base and tail in _AGG_SUFFIX:
        return f"{_AGG_SUFFIX[tail]} {_words(base)}"
    return _words(c)


def _schema_names(cols: List[str], table: Optional[str], sm: Optional[dict]) -> Dict[str, str]:
    tables = set((sm or {}).get("tables", {}) or {})
    if table:
        tables.add(str(table))
    names: Dict[str, str] = {}
    for t in tables:
        if "_" in t:
            names[t] = _table_label(t, sm)
    # Columns the model knows for the result's table (the summary model may name one that
    # is not in the SELECT list), then the result's own columns, which win.
    for key in ((sm or {}).get("columns", {}) or {}):
        t, _, c = str(key).partition(".")
        if t == table and "_" in c:
            names[c] = column_label(c, tables, sm)
    for c in cols or []:
        if c and "_" in str(c):
            names[str(c)] = column_label(str(c), tables, sm)
    return names


def business_wording(text: Any, cols: List[str], table: Optional[str],
                     sm: Optional[dict] = None) -> Any:
    """`text` with schema identifiers replaced by their plain-language names. Returns the
    input unchanged when it is not a string, is empty, or names no schema identifier."""
    if not isinstance(text, str) or "_" not in text:
        return text
    names = _schema_names(list(cols or []), table, sm)
    if not names:
        return text
    # Longest first, so "assets_asset_count" is replaced before "assets_asset".
    alts = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    pat = re.compile(rf"[`'\"]?(?<![{_IDENT_CHARS}])({alts})(?![{_IDENT_CHARS}])[`'\"]?")

    def _sub(m):
        whole, ident = m.group(0), m.group(1)
        label = names[ident]
        # Keep a quote that is not part of a matched pair (e.g. an apostrophe after it).
        lead = whole[0] if whole[0] in "`'\"" else ""
        trail = whole[-1] if whole[-1] in "`'\"" and len(whole) > len(ident) + len(lead) else ""
        if lead and trail and lead == trail:
            return label
        return lead + label + trail
    return pat.sub(_sub, text)
