"""veda.understanding.grounding — the DETERMINISTIC anti-hallucination firewall.

Turns an (untrusted) RawIntent's CONCEPTS into a (trusted) GroundedIntent whose every
table/column is a real, schema-validated artifact — or a Refusal when a required
concept can't be grounded. NO LLM here: pure, testable, reproducible. This is the layer
that guarantees "never invent a table/column" no matter what the LLM said.

Grounding sources, in priority order (all existing VEDA infra, reused not rebuilt):
  1. exact/subset name-token match against real graph tables (most specific wins)
  2. the curated entity glossary (business noun → table)
  3. retrieval evidence (optional, when `results` supplied)
A concept that matches none of these does NOT get a guessed table — it goes to
`unresolved`, and the orchestrator turns a required unresolved into a Refusal.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from veda.understanding.schema import (
    RawIntent, GroundedIntent, GroundedMeasure, GroundedFilter, Refusal,
)

# language connectives that are NEVER entities/filters (schema-agnostic)
_STOP = frozenset({
    "the", "a", "an", "their", "them", "they", "with", "and", "or", "of", "for", "by",
    "per", "each", "all", "to", "from", "in", "on", "across", "relate", "related",
    "show", "list", "give", "get", "number", "count", "total", "amount", "sum",
    "average", "avg", "how", "many", "much", "value", "values",
})

_AGG_WORDS = {  # measure phrase → aggregation kind
    "count": "count", "number": "count", "how many": "count",
    "total": "sum", "sum": "sum",
    "average": "avg", "avg": "avg", "mean": "avg",
    "highest": "max", "max": "max", "maximum": "max", "largest": "max",
    "lowest": "min", "min": "min", "minimum": "min", "smallest": "min",
}


def _singularize(w: str) -> str:
    """Singular form of a token. Delegates to the shared enrichment singularizer; if that
    module is unavailable, falls back to a naive trailing-'s' strip (only for words > 3
    chars, so short tokens like 'is'/'as' are left intact)."""
    try:
        from retrieval.query_enrichment import _singularize as _s
        return _s(w)
    except ImportError:
        return w[:-1] if len(w) > 3 and w.endswith("s") else w


def _name_toks(t: str, sm=None):
    try:
        from veda.routing import _name_toks as _nt
        return _nt(t, sm)
    except Exception:
        return {p for p in t.split("_") if len(p) > 2}


def _glossary() -> Dict[str, str]:
    try:
        from query.entity_resolver import _entity_glossary
        return _entity_glossary() or {}
    except Exception:
        return {}


def _concept_tokens(concept: str) -> List[str]:
    return [_singularize(w) for w in re.findall(r"[a-z]+", (concept or "").lower())
            if len(w) > 2 and w not in _STOP]


def ground_entity(concept: str, graph_tables, junctions, sm=None,
                  retrieval_scores: Optional[Dict[str, float]] = None) -> Optional[str]:
    """Concept → the single best REAL table, or None if it can't be grounded.
    Deterministic: name-token match (most specific), then glossary, then retrieval.
    Never invents — None means 'the firewall must decide (refuse if required)'."""
    # 0. exact real-table match — the extractor's entity_catalog contains real table
    #    names (when a business_name is absent), so the LLM often echoes one verbatim.
    #    An exact hit on a real table IS valid grounding (not a guess) — and it sidesteps
    #    the compound-token mismatch ('accounts_paymenttransaction' tokenizes to
    #    {account, paymenttransaction} but the table's name_toks are {account, payment,
    #    transaction}). Case/space-insensitive.
    if concept:
        _c = concept.strip().lower().replace(" ", "_")
        for t in graph_tables:
            if t.lower() == _c and t not in junctions:
                return t
    toks = _concept_tokens(concept)
    if not toks:
        return None
    concat = "".join(toks)
    # 1. curated glossary FIRST (business noun → table). It is a deliberate, human-verified
    #    per-source mapping, so it must OUT-PRIORITIZE a coincidental name-token match:
    #    "tenant" → users_user (glossary), NOT assets_leasetenant (which merely shares the
    #    'tenant' token). Checked before name-tokens for exactly this class.
    gl = _glossary()
    for key in (concat, *toks):
        if key in gl and gl[key] in graph_tables and gl[key] not in junctions:
            return gl[key]
    # 2a. EXACT name-token equality wins — even for a table the junction heuristic
    #     flagged. When the concept's tokens EXACTLY equal a table's name tokens, the user
    #     named that entity precisely (it's a query TARGET, not a fuzzy intermediate
    #     bridge): "asset type" {asset,type} → assets_assettype {asset,type}, even though
    #     assettype is (mis)classified as a junction. Junction-exclusion only applies to
    #     the fuzzy subset match below, never to an exact name.
    _toks = set(toks)
    exact = [t for t in graph_tables if _name_toks(t, sm) == _toks]
    if exact:
        return min(exact, key=len)
    # 2b. fuzzy name-token match (subset / concat) — junctions excluded here (they're
    #     bridges, not the named target). Prefer the FEWEST name tokens (most exact:
    #     "user" → users_user {user}, not users_userrole {user,role}).
    matches = [t for t in graph_tables
               if t not in junctions and (set(toks) <= _name_toks(t, sm)
                                          or concat in _name_toks(t, sm))]
    if matches:
        return min(matches, key=lambda t: (len(_name_toks(t, sm)), len(t)))
    # 3. retrieval evidence (optional): highest-scored table whose name shares a token
    if retrieval_scores:
        cand = [(s, t) for t, s in retrieval_scores.items()
                if t in graph_tables and t not in junctions and (set(toks) & _name_toks(t, sm))]
        if cand:
            return max(cand)[1]
    return None


def ground_measure(concept: Optional[str], anchor: Optional[str], graph_tables, junctions,
                   sm=None) -> Optional[GroundedMeasure]:
    """Measure concept → GroundedMeasure. "number of payment transactions" → COUNT of
    the payments table; "total paid amount" → SUM (column grounded downstream). Returns
    None when there's no measure (a plain list) or it can't be grounded to a kind."""
    if not concept:
        return None
    low = concept.lower()
    kind = None
    for w, k in _AGG_WORDS.items():
        if w in low:
            kind = k
            break
    if kind is None:
        return None
    # "number/count of <entity>" → count that entity's table
    if kind == "count":
        tbl = ground_entity(concept, graph_tables, junctions, sm)
        return GroundedMeasure(kind="count", table=tbl, column=None, concept=concept)
    # sum/avg/max/min of a column — column resolution is done later against `anchor`'s
    # columns by the planner/generator; here we record the kind + concept (no guess).
    return GroundedMeasure(kind=kind, table=anchor, column=None, concept=concept)


def ground(raw: RawIntent, sm, graph, junctions,
           retrieval_scores: Optional[Dict[str, float]] = None,
           min_confidence: float = 0.5):
    """RawIntent → GroundedIntent | Refusal | None.

    None  = degrade (low confidence / nothing to ground) → caller uses existing path.
    Refusal = a REQUIRED concept couldn't be grounded, or intent is refuse/clarify.
    GroundedIntent = every artifact validated real.
    """
    graph_tables = set(graph.get("tables", []))
    if not raw or not raw.is_valid_shape():
        return None
    if raw.intent in ("refuse", "clarify"):
        return Refusal(reason="impossible" if raw.intent == "refuse" else "ambiguous",
                       message=("This question can't be answered from the available data."
                                if raw.intent == "refuse"
                                else "This question is ambiguous — please specify."),
                       evidence={"llm_intent": raw.intent})
    if raw.confidence and raw.confidence < min_confidence:
        return None                          # not confident enough → degrade, don't guess

    unresolved: List[str] = []
    ev: Dict[str, Any] = {"grounded": {}}

    # grain → anchor (the single most important grounding — fixes grain-inversion)
    anchor = ground_entity(raw.grain, graph_tables, junctions, sm, retrieval_scores) \
        if raw.grain else None
    if raw.grain and not anchor:
        unresolved.append(f"grain:{raw.grain}")
    else:
        ev["grounded"]["grain"] = anchor

    # other entities → secondaries (distinct, real, not the anchor)
    secondaries: List[str] = []
    for e in (raw.entities or []):
        t = ground_entity(e, graph_tables, junctions, sm, retrieval_scores)
        if t and t != anchor and t not in secondaries:
            secondaries.append(t)
        elif not t and e:
            unresolved.append(f"entity:{e}")

    # anchor fallback: if grain didn't ground but an entity did, use the first entity
    if not anchor and secondaries:
        anchor = secondaries.pop(0)
        ev["grounded"]["grain"] = anchor

    measure = ground_measure(raw.measure, anchor, graph_tables, junctions, sm)

    # FIREWALL: an answer intent with NO grounded anchor at all cannot proceed → refuse
    if anchor is None:
        return Refusal(reason="ungrounded",
                       message=("Couldn't identify which data this question is about. "
                                "Please name the entity (e.g. properties, payments, users)."),
                       unresolved=unresolved or ([f"grain:{raw.grain}"] if raw.grain else []),
                       evidence=ev)

    return GroundedIntent(
        intent=raw.intent, anchor=anchor, secondaries=secondaries, measure=measure,
        dimensions=[], filters=[],           # dim/filter column grounding: progressive
        confidence=raw.confidence,
        evidence={**ev, "unresolved_nonfatal": unresolved},
    )
