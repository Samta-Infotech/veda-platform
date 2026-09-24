"""Lifecycle / status phrase grounding (D.5–D.6, 2026-09-23).

`query/value_arbiter` grounds a span by EXACT match against the sampled value store, so
it can resolve "cancelled" (a real value of `assets_salelisting.status`) but is blind to
"currently on the market" — a phrase that means `status = 'APPROVED'` and appears
nowhere in the data. The qualifier was therefore dropped in silence: the 2026-09-23 run
answered "the CHEAPEST properties CURRENTLY ON THE MARKET" with 74 of its 100 rows in
DRAFT or CANCELLED, the three cheapest all DRAFT.

Two behaviours, in order:

1. **Mapped phrase -> grounded filter.** The curated per-source artifact
   `veda_value_aliases.json` (published from a TRACKED seed by
   `ingestion/entity_alias_seeder.publish_values`) maps a business phrase to a concrete
   value of a concrete column. A hit becomes a normal grounded filter.

2. **Unmapped lifecycle phrase -> typed clarify, never a silent unfiltered answer.**
   When the question carries a residual adjective/state phrase that is NOT a value of
   anything, and the anchor HAS a status/category column, the honest answer names that
   column's real domain. This is what "which properties have an ACTIVE status" needs:
   `status` is APPROVED / DRAFT / CANCELLED and there is no ACTIVE, so the only correct
   reply says so and offers the real values — not a content-free "could you clarify".
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

ARTIFACT = "veda_value_aliases.json"
SEED_NAME = "veda_value_aliases.seed.json"

#: Column names that carry a lifecycle/state, used only to decide whether an unmapped
#: state phrase deserves a grounded clarify (never to invent a filter).
_STATE_COL_HINTS = ("status", "state", "stage", "phase", "lifecycle", "condition",
                    "category", "type", "kind")

#: Adjective/state phrases a user attaches to an entity. Presence alone does nothing;
#: they only matter when NOTHING in the data grounded them and the anchor has a state
#: column — then they are the reason to ask rather than answer.
#: "current"/"currently" are deliberately NOT here. They are determiners far more often
#: than status values — "and what is their CURRENT STATUS?" merely asks to SHOW the
#: column — so including them turned a perfectly answerable question into a clarify.
_STATE_WORDS = {"active", "inactive", "live", "open", "closed",
                "available", "pending", "approved", "rejected", "cancelled", "canceled",
                "published", "unpublished", "draft", "listed", "delisted", "expired",
                "valid", "invalid", "enabled", "disabled", "archived", "ongoing"}

_CACHE: Dict[tuple, dict] = {}


def _seed_path(source_id) -> str:
    here = os.path.dirname(os.path.abspath(__file__))              # .../veda_core/query
    return os.path.join(os.path.dirname(here), "data", "seeds", str(source_id), SEED_NAME)


def load_glossary() -> dict:
    """{"table.column": {"VALUE": [phrase, ...]}} for the ambient source.

    Artifact first, tracked seed as the fallback — the artifact lives under the
    gitignored, reingest-cleared tree, so "missing" is a normal post-ingest state and
    must degrade to stale-but-correct, never to empty (same reasoning as
    query/entity_resolver._entity_glossary)."""
    try:
        from veda_core import context as _ctx
        c = _ctx.try_current()
        key = (str(c.tenant), str(c.source_id)) if c is not None else ("", "")
    except Exception:
        key = ("", "")
    if key in _CACHE:
        return _CACHE[key]

    g = {}
    try:
        from config import resolve_source_artifact
        p = resolve_source_artifact(ARTIFACT)
        if p and os.path.exists(p):
            g = {k: v for k, v in json.load(open(p)).items() if not str(k).startswith("_")}
    except Exception:
        g = {}
    if not g and key[1]:
        try:
            sp = _seed_path(key[1])
            if os.path.exists(sp):
                g = {k: v for k, v in json.load(open(sp)).items()
                     if not str(k).startswith("_")}
        except Exception:
            g = {}
    _CACHE[key] = g
    return g


def phrase_filters(query: str, anchor: str, sm: dict) -> Tuple[List[dict], List[str]]:
    """(filters, matched_phrases) for phrases the curated glossary maps onto `anchor`.

    Longest phrase first, so "currently on the market" is consumed before "on the
    market"; a column already filtered is not filtered twice."""
    if not query or not anchor:
        return [], []
    g = load_glossary()
    if not g:
        return [], []
    ql = f" {query.lower()} "
    out, matched, used_cols = [], [], set()

    entries = []
    for colkey, mapping in g.items():
        tbl, _, col = str(colkey).partition(".")
        if tbl != anchor or not col:
            continue
        for value, phrases in (mapping or {}).items():
            for ph in (phrases or []):
                entries.append((len(str(ph)), str(ph).lower(), col, value))
    entries.sort(reverse=True)

    for _, ph, col, value in entries:
        if col in used_cols:
            continue
        if re.search(rf"(?<![a-z]){re.escape(ph)}(?![a-z])", ql):
            out.append({"column": col, "op": "=", "value": value,
                        "value_norm": str(value).lower(), "kind": "value",
                        "reason": f"'{ph}' -> {col} = {value}"})
            used_cols.add(col)
            matched.append(ph)
    return out, matched


def state_columns(anchor: str, sm: dict) -> List[str]:
    """Status/category-ish columns of `anchor`, by semantic role then by name."""
    out = []
    for key, m in (sm.get("columns", {}) or {}).items():
        tbl, _, cn = key.partition(".")
        if tbl != anchor:
            continue
        role = f"{(m or {}).get('semantic_type', '')} {(m or {}).get('business_role', '')}".upper()
        if "STATUS" in role or "CATEGORY" in role or "DIMENSION" in role \
                or any(h in cn.lower() for h in _STATE_COL_HINTS):
            out.append(cn)
    # name-hint columns first, then the rest, each alphabetically — deterministic
    named = sorted(c for c in out if any(h in c.lower() for h in _STATE_COL_HINTS))
    rest = sorted(c for c in out if c not in named)
    return named + rest


def unmapped_state_clarify(query: str, anchor: str, sm: dict,
                           already_matched: Optional[List[str]] = None) -> Optional[str]:
    """A grounded clarify when the query names a STATE the data does not have.

    Returns the message, or None when there is nothing to ask about. Deliberately
    conservative: it fires only when a state word is present, no glossary phrase
    consumed it, and the anchor really has a state column with a readable domain — so a
    question with no lifecycle wording is never interrupted."""
    if not query or not anchor:
        return None
    ql = f" {query.lower()} "
    consumed = " ".join(already_matched or []).lower()
    # ADJACENCY, not mere presence. A state word only asks to be treated as a VALUE when
    # it sits directly against a state noun — "an ACTIVE STATUS", "status is CANCELLED".
    # Presence alone is far too loose: "the cheapest properties currently on the market,
    # along with their CURRENT market STATUS" contains both "current" and "status" and is
    # a perfectly answerable question, so firing on presence would turn a correct answer
    # into a clarify. One intervening word is not allowed, which is exactly what keeps
    # "current MARKET status" out while letting "active status" in.
    _nouns = "status|state|stage|phase|condition"
    present = []
    for w in sorted(_STATE_WORDS):
        if w in consumed:
            continue
        e = re.escape(w)
        if re.search(rf"\b{e}\s+(?:{_nouns})\b", ql) or \
           re.search(rf"\b(?:{_nouns})\s+(?:is|are|of|=)?\s*{e}\b", ql):
            present.append(w)
    if not present:
        return None

    cols = state_columns(anchor, sm)
    if not cols:
        return None

    try:
        from query.resolution import domain_for
    except Exception:
        return None

    # rank the candidate columns by name overlap with the question ("status" in the
    # question -> the column called status), then take the first with a real domain
    qtoks = {w for w in re.findall(r"[a-z]+", ql) if len(w) > 2}
    cols = sorted(cols, key=lambda c: (-len(qtoks & set(re.findall(r"[a-z]+", c.lower()))), c))
    for col in cols[:3]:
        vals = [v for v in (domain_for(anchor, col) or []) if str(v).strip()]
        if not vals:
            continue
        low = {str(v).strip().lower() for v in vals}
        if any(w in low for w in present):
            return None                      # the state IS a real value — not our case
        named = ", ".join(str(v) for v in vals[:6])
        word = present[0]
        return (f"'{word}' isn't one of the values {col} actually takes here — "
                f"it holds {named}. Which of those did you mean?")
    return None


def where_clause(filters: List[dict], alias: Optional[str] = None) -> str:
    """Case-insensitive equality, matching value_arbiter's composition."""
    prefix = f"{alias}." if alias else ""
    parts = []
    for f in filters:
        v = str(f["value"]).replace("'", "''")
        parts.append(f'lower({prefix}"{f["column"]}") = lower(\'{v}\')')
    return " AND ".join(parts)


def explain(filters: List[dict]) -> str:
    return "; ".join(f["reason"] for f in filters)
