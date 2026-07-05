"""VEDA · Answer-Entity Discovery.

When a query asks WHO (the answer is a person), the system should project that person's
display column reached over a foreign key — not the raw FK id. "who owns critical
incidents" → project users.<name> via incident.assigned_to_id, never `assigned_to_id`.

This is a deterministic FINDER, not a new subsystem. It reuses what already exists:
  • concept_graph["PERSON"]  — which tables/columns ARE people (was dormant until now)
  • the FK graph (get_graph) — the same edges the join planner uses (NOT a graph DB)
  • _resolve_display_column  — the existing "what label to project" resolver

Given (query, anchor, graph, semantic_model) it returns a descriptor:
    {anchor, fk_col, target_table, target_pk, display_col, relation, reason}
or None when there is no person-answer ask, no person-FK on the anchor, or the relation is
ambiguous (several person-FKs and none/many named) — refuse-over-guess, like value_resolver.

Pure and DB-free: the graph, semantic model, and display resolver are all injected, so the
finder is unit-testable with no DB and no Ollama.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

try:
    from retrieval.query_enrichment import _singularize as _sing
except Exception:                                    # pragma: no cover
    def _sing(w: str) -> str:
        return w[:-1] if len(w) > 3 and w.endswith("s") else w

# wh-words that request a PERSON as the answer. Closed-class function words — grammar,
# not a domain list. ("by whom" is covered: "whom" is in the set.)
_WH_PERSON = {"who", "whom", "whose", "whos"}

# Possessive / projection cues — "incidents and THEIR handler", "show ITS owner". Grammar,
# not a domain list. These ask to SHOW the related person as a column (projection), as opposed
# to a value-filter ("incidents assigned to raj" has no possessive cue → stays a filter).
_PROJECTION_CUE = {"their", "its", "theirs"}

DisplayResolver = Callable[[str, dict], Optional[str]]


def _default_display_resolver(table: str, sm: dict) -> Optional[str]:
    from veda.generation import _resolve_display_column
    return _resolve_display_column(table, sm)


def _person_tables(sm: dict) -> set:
    """Tables that ARE people: any table with >=1 column in concept_graph['PERSON']."""
    return {m["table"] for m in (sm.get("concept_graph", {}) or {}).get("PERSON", [])}


def _relation_words(anchor: str, fk_col: str, sm: dict) -> set:
    """Vocabulary that names this FK relation: the column-name parts + its aliases
    (data-driven, from the semantic model — e.g. assigned_to_id → {assigned, owner,
    handler, responsible}). `_id`/`_by` connectors are dropped."""
    words = {p for p in fk_col.split("_") if len(p) > 2 and p not in ("id", "by")}
    meta = (sm.get("columns", {}) or {}).get(f"{anchor}.{fk_col}", {}) or {}
    for al in (meta.get("aliases") or []):
        words |= {w for w in re.findall(r"[a-z]+", al.lower()) if len(w) > 2}
    return {_sing(w) for w in words}


def _relation_named(qtoks: List[str], rel_words: set) -> bool:
    """Does the query name this relation? token intersection, with short-substring
    tolerance so owns↔owner, handles↔handler match (mirrors the qualifier gate)."""
    q = {_sing(t) for t in qtoks}
    if q & rel_words:
        return True
    for a in q:
        for b in rel_words:
            if len(a) >= 3 and len(b) >= 3 and (a in b or b in a):
                return True
            # shared stem (handled↔handler, created↔creator): same 4-char prefix
            if len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4]:
                return True
    return False


def find_answer_entity(
    query: str,
    anchor: str,
    graph: dict,
    sm: dict,
    display_resolver: DisplayResolver = _default_display_resolver,
) -> Optional[Dict]:
    """Return an answer-entity projection descriptor, or None. See module docstring."""
    qtoks = [w for w in re.findall(r"[a-z']+", query.lower())]
    qset = set(t.replace("'", "") for t in qtoks)
    is_wh = bool(qset & _WH_PERSON)
    is_projection = bool(qset & _PROJECTION_CUE)
    if not (is_wh or is_projection):
        return None

    person_tables = _person_tables(sm)
    if not person_tables:
        return None

    # candidate person-FKs: anchor FK columns whose target table is a PERSON table
    cands = [(e["source_column"], e["target_table"], e["target_column"])
             for e in graph.get("edges", [])
             if e["source_table"] == anchor and e["target_table"] in person_tables]
    if not cands:
        return None

    named = [c for c in cands
             if _relation_named(qtoks, _relation_words(anchor, c[0], sm))]
    if is_wh:
        # WHO question: a single named person-FK, else the sole person-FK.
        if len(named) == 1:
            chosen, rel, mode = named[0], "named relation", "who"
        elif len(cands) == 1:
            chosen, rel, mode = cands[0], "sole person-FK", "who"
        else:
            return None                   # ambiguous → refuse
    else:
        # Possessive projection ("incidents and their handler"): the relation MUST be named
        # unambiguously — a bare "their" with several person-FKs is too vague to guess.
        if len(named) == 1:
            chosen, rel, mode = named[0], "named projection", "projection"
        else:
            return None

    fk_col, target_table, target_pk = chosen
    display = display_resolver(target_table, sm) or target_pk
    rel_label = next((p for p in fk_col.split("_") if len(p) > 2 and p not in ("id", "by")),
                     target_table)
    return {
        "anchor": anchor, "fk_col": fk_col, "target_table": target_table,
        "target_pk": target_pk, "display_col": display, "relation": rel, "mode": mode,
        "rel_label": rel_label,
        "reason": (f"PERSON {mode} ({rel}): {anchor}.{fk_col} → "
                   f"{target_table}.{display}"),
    }
