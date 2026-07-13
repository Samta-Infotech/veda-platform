"""VEDA · Multi-hop FK value resolution (junction-aware, refuse-on-ambiguity).

value_resolver.resolve_value_filter handles 1-hop entity filters. This module generalises
to N hops over the FK graph so values reachable only through a JUNCTION table resolve:

    "tags on document X"  (anchor = document, X ∈ tag)
        path : document ← document_tags → tag
        SQL  : WHERE document_id IN (
                 SELECT document_id FROM document_tags WHERE tag_id IN (
                   SELECT tag_id FROM tag WHERE lower(name) = lower('X')))

CRITICAL — it only fires when the membership path is UNAMBIGUOUS. FK topology does NOT
encode relationship *meaning*: the same two tables are often linked by several paths that
mean different things ("owned by" via owner_id vs "shared with" via a junction; RBAC
"assigned" = direct permissions vs role-derived). No structural rule (shortest / union /
first-hop) can pick the intended one — so when >1 membership path exists, this REFUSES
(returns None) and the query falls to the LLM, exactly like value_resolver on ambiguity.
Never guesses, never unions. (RBAC effective-set semantics, if needed, is a declared
per-relationship config — not inferred from topology.)

A MEMBERSHIP path (vs a spurious bridge) (a) leaves the anchor via a junction that
REFERENCES the anchor (first hop to a child) and (b) enters the value table from a junction
that REFERENCES it (value entered as a parent). This excludes shared-DIMENSION bridges both
endpoints reference (permission → organizations ← user) and provenance/audit FKs
(created_by/updated_by → user). The value is EXACT-grounded (no fuzzy/LIKE). Graph + lookup
injected → unit-testable with no DB and no Ollama. NOT a graph DB/embedding/new store —
plain FK path-finding over the graph the join planner already loads.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

# lookup(token) -> [(table, column, value_raw), ...] — the column_values exact-grounding
Lookup = Callable[[str], List[Tuple[str, str, str]]]

_PATH_SCAN_CAP = 64                       # bound path enumeration (refuse beyond this)


def _sql_lit(s: str) -> str:
    return str(s).replace("'", "''")


def _is_provenance(col: str) -> bool:
    """Audit/provenance FK (created_by_id, updated_by_id, owned_by, ...). These shortcut
    almost every table directly to `user`; "who created it" is a provenance relation for
    the 1-hop relation-named resolver, NOT membership filtering. Structural naming
    convention (`*_by`/`*_by_id`) — the same one planning.py special-cases; not a word
    list. `assigned_to_id` is NOT `_by`, so it's kept."""
    c = col.lower()
    return c.endswith("_by") or c.endswith("_by_id")


def _adjacency(graph: dict, skip_provenance: bool = True):
    """table -> [(neighbour, this_side_col, neighbour_side_col, neighbour_is_source), ...].
    neighbour_is_source = True → neighbour REFERENCES this table (a child/junction of it);
    False → this table references the neighbour (a parent/dimension)."""
    adj = defaultdict(list)
    for e in graph.get("edges", []):
        s, sc = e["source_table"], e["source_column"]
        t, tc = e["target_table"], e["target_column"]
        if skip_provenance and _is_provenance(sc):
            continue
        adj[s].append((t, sc, tc, False))     # from s: neighbour t is the PARENT
        adj[t].append((s, tc, sc, True))       # from t: neighbour s is the CHILD
    return adj


def _membership_paths(anchor: str, target: str, adj, max_hops: int) -> List[list]:
    """All junction-membership paths anchor→target (each [(table, left_col, right_col), …],
    anchor's cols None). Rules: ≥2 hops; first hop to a CHILD junction (references anchor);
    target entered as a PARENT (a junction references it); simple paths; ≤max_hops."""
    out: List[list] = []
    stack = [[(anchor, None, None)]]
    scanned = 0
    while stack and scanned < _PATH_SCAN_CAP:
        path = stack.pop()
        scanned += 1
        cur = path[-1][0]
        depth = len(path) - 1
        if cur == target and depth >= 2:
            out.append(path)
            continue
        if depth >= max_hops:
            continue
        seen = {p[0] for p in path}
        for nb, lcol, rcol, nb_is_source in adj.get(cur, []):
            if nb in seen:
                continue                               # simple paths only
            if depth == 0 and not nb_is_source:
                continue                               # first hop must be a child junction
            if nb == target and nb_is_source:
                continue                               # value must be entered as a parent
            stack.append(path + [(nb, lcol, rcol)])
    return out


def _nested_subquery(path: list, pairs: List[Tuple[str, str]]) -> Tuple[str, str]:
    """(anchor_col, nested_IN_subquery) for the value predicate at the path's far end."""
    k = len(path) - 1
    vp = " OR ".join(f'''lower("{c}"::text) = lower('{_sql_lit(v)}')''' for c, v in pairs)
    tk, _, rk = path[k]
    inner = f'SELECT "{rk}" FROM "{tk}" WHERE {vp}'
    for i in range(k - 1, 0, -1):
        ti, _, ri = path[i]
        l_next = path[i + 1][1]
        inner = f'SELECT "{ri}" FROM "{ti}" WHERE "{l_next}" IN ({inner})'
    return path[1][1], inner                           # (anchor_col, subquery)


def resolve_fk_path(
    anchor: str,
    qtoks: List[str],
    graph: dict,
    lookup: Lookup,
    max_hops: int = 4,
    anchor_col_toks: Optional[set] = None,
) -> Optional[Dict]:
    """Single UNAMBIGUOUS multi-hop entity filter, or None (refuse → LLM).

    Returns {"kind": "multihop_subquery", "anchor_col", "subquery", "value_table",
             "path", "pairs"} only when exactly one grounded table is reachable by exactly
    one membership path. Otherwise None (ungrounded, ambiguous target, or >1 path).

    anchor_col_toks (optional, injected): singularized WORD-PIECES of the anchor table's
    column names (e.g. "workflow_state" → {"workflow", "state"}). A token that NAMES an
    attribute of the anchor ("state" → workflow_state, "updated" → updated_date) is a
    column to PROJECT, not a cross-table value filter — "incidents with their workflow
    state changes" stays single-table. Without this guard a common word coincidentally
    stored as a value in a distant table (signal_levels) hijacks the query into a
    fabricated multi-hop join. Mirrors the 1-hop resolve_value_filter guard."""
    from retrieval.query_enrichment import _singularize
    _skip = anchor_col_toks or set()

    # 1. EXACT-ground every token (same as value_resolver — never fuzzy/LIKE)
    hits, seen = [], set()
    for tok in qtoks:
        if tok in _skip or _singularize(tok) in _skip:
            continue
        try:
            found = lookup(tok) or []
        except Exception:
            found = []
        # A token that ALSO grounds to a value on the anchor's own columns is a single-table
        # value filter (handled by resolve_value_filter), never a multi-hop join trigger —
        # skip the whole token so its coincidental match in another table can't collapse an
        # ambiguous-refuse into a wrong join.
        if any(tbl == anchor for tbl, _c, _v in found):
            continue
        for tbl, col, val in found:
            if (tbl, col) not in seen:
                seen.add((tbl, col))
                hits.append((tbl, col, val))
    if not hits:
        return None

    # 2. Reachability scoping: noun/relation tokens ground to unrelated tables
    #    ("permissions"→module, "assigned"→state). Keep only tables reachable from the
    #    anchor by a junction-membership path; exactly one such table must remain, with
    #    exactly one path (else the membership relation is ambiguous → refuse → LLM).
    adj = _adjacency(graph)
    by_table: Dict[str, list] = defaultdict(list)
    for tbl, col, val in hits:
        if tbl != anchor:
            by_table[tbl].append((col, val))
    reachable = {t: paths for t in by_table
                 if (paths := _membership_paths(anchor, t, adj, max_hops))}
    if len(reachable) != 1:
        return None                                    # 0 reachable, or ambiguous target
    value_table, paths = next(iter(reachable.items()))
    if len(paths) != 1:
        return None                                    # >1 membership path → ambiguous → refuse
    path = paths[0]

    pairs, pseen = [], set()                           # OR across same-entity columns
    for c, v in by_table[value_table]:
        if c not in pseen:
            pseen.add(c)
            pairs.append((c, v))
    pairs.sort()

    anchor_col, subquery = _nested_subquery(path, pairs)
    return {
        "kind": "multihop_subquery", "anchor_col": anchor_col, "subquery": subquery,
        "value_table": value_table, "path": [p[0] for p in path], "pairs": pairs,
    }
