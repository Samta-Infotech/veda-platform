# =============================================================================
# query/join_planner.py
# VEDA V2 — deterministic join planner (runtime).
#
# Reads data/veda_relationship_graph.json (built offline). The LLM NEVER touches
# any of this — join keys, paths, polymorphic predicates are all computed here.
#
#   select_anchor()  → the query's subject/grain table
#   plan_joins()     → weighted shortest path anchor↔each target + confidence
#   build_skeleton() → deterministic FROM/JOIN/ON (+ polymorphic predicate)
#
# Failure-safe: no path → unreachable (caller refuses); >1 edge between a pair
# with no disambiguation → ambiguous (caller clarifies).
# =============================================================================

import os
import re
import json
import heapq
from collections import defaultdict

RELATIONSHIP_GRAPH_FILE = "data/veda_relationship_graph.json"

# confidence penalties (multiplicative)
_HOP_PENALTY = 0.92
_POLY_PENALTY = 0.95
_INFERRED_PENALTY = 0.9


_JOIN_PATHS_MAP = None   # Q-9: lazily-loaded precomputed reachability map


def _is_reachable(adj, t, o, max_hops=4):
    """Q-9: consult the precompiled join-path map first (deterministic, O(1)); fall back
    to live shortest-path traversal for unmapped pairs or when the flag/map is absent.
    Reachability only — the actual join edges are still computed by _shortest_path."""
    global _JOIN_PATHS_MAP
    # WP7: consult the precompiled join-path map first (unconditional), falling back to
    # live shortest-path traversal for unmapped pairs or schema drift between ingestions.
    try:
        if _JOIN_PATHS_MAP is None:
            from ingestion.join_paths import load_join_paths
            _JOIN_PATHS_MAP = load_join_paths() or {}
        if _JOIN_PATHS_MAP:
            return f"{t}|{o}" in _JOIN_PATHS_MAP
    except Exception:
        pass
    return _shortest_path(adj, t, o, max_hops=max_hops) is not None


def load_graph(path=RELATIONSHIP_GRAPH_FILE):
    if not os.path.exists(path):
        return {"tables": [], "edges": []}
    return json.load(open(path))


def _adjacency(graph):
    """Undirected traversal adjacency: node -> list of (neighbor, edge)."""
    adj = defaultdict(list)
    for e in graph.get("edges", []):
        adj[e["source_table"]].append((e["target_table"], e))
        adj[e["target_table"]].append((e["source_table"], e))
    return adj


_ANCHOR_CONNECTIVES = {"and", "or", "of", "to", "by"}


from dataclasses import dataclass


@dataclass
class AnchorCandidate:
    table: str
    score: float                 # composite 0..1
    signals: dict                # per-signal breakdown (lexical/position/retrieval/graph)


def score_anchors(query, candidate_tables, scores=None, graph=None, adj=None, sm=None):
    """Rank candidate subject/grain tables by a MULTI-SIGNAL composite — the
    replacement for a single winner-take-all heuristic.

    Signals (each normalized 0..1, weights in config.ANCHOR_SCORING):
      lexical   : matched name tokens / table tokens (how fully the query names it)
      position  : 1 - first_word_index / len  (word ORDER — subject prior, no vocab)
      retrieval : normalized retrieval score
      graph     : fraction of the OTHER candidates this table can reach (query's hub)

    Position is ONE feature, never the decider. The caller reads the top-two MARGIN
    as a confidence: a small margin means the grain is AMBIGUOUS and the planner
    should clarify/refuse rather than silently emit SQL at the wrong grain.

    "for each X / per X" is an EXPLICIT grain declaration — that table is pinned to
    score 1.0 (a confident top), since the user named the grain outright.

    Returns AnchorCandidate list sorted by score desc. WHOLE-token singularized
    matching (not substring), so "counterparties" matches counterparty_details, not
    investigation_and_research_counter_party."""
    import re
    from retrieval.query_enrichment import _singularize
    from config import ANCHOR_SCORING as W

    # Schema-vocabulary segmentation so concatenated names ("paymenttransaction")
    # expose their entity words to the lexical/position signals; plain split on
    # any failure — identical to the legacy tokens.
    def _tab_toks(t):
        try:
            from semantic.name_tokens import table_tokens
            return {w for w in table_tokens(t, sm) if w not in _ANCHOR_CONNECTIVES}
        except Exception:
            return {_singularize(w) for w in t.split("_")
                    if len(w) > 2 and w not in _ANCHOR_CONNECTIVES}

    # Table-token IDF: a matched name token shared by many tables ("value", "type",
    # an app prefix) is weak evidence; one carried by few ("payment") is strong.
    # Schema-derived — the generic-word discount without any stopword list. Without
    # this, subword segmentation would let generic tokens mis-anchor wide families
    # (query "…value…" → every valuebundle table). Empty dict → weights of 1.0
    # everywhere, i.e. exactly the unweighted legacy arithmetic.
    try:
        from semantic.name_tokens import token_table_idf
        _idf = token_table_idf(sm) or {}
    except Exception:
        _idf = {}

    def _w(tok):
        return _idf.get(tok, 1.0)

    scores = scores or {}
    words = [_singularize(w) for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 2]
    qtoks = set(words)
    nwords = len(words) or 1
    maxret = max(scores.values()) if scores else 0.0
    if adj is None and graph is not None:
        adj = _adjacency(graph)

    gm = re.search(r"\b(?:for each|for every|for all|per|each|every)\s+([a-z]+)", query.lower())
    grain = _singularize(gm.group(1)) if gm else None

    def _first_pos(toks):
        for i, w in enumerate(words):
            if w in toks:
                return i
        return None

    # specificity is relative to the most-named candidate: among tables sharing a
    # first token (incident vs incident_signal_score, rfi_objects vs
    # rfi_instance_questions), the one the query names with MORE tokens is the more
    # specific subject — the old picker's primary key, kept here as a weighted signal.
    _matched_weights = []
    for t in candidate_tables:
        _matched_weights.append(sum(_w(x) for x in (qtoks & _tab_toks(t))))
    max_matched = max(_matched_weights) if _matched_weights else 0.0

    rows = []
    for t in candidate_tables:
        toks = _tab_toks(t)
        matched = qtoks & toks
        mw = sum(_w(x) for x in matched)
        tw = sum(_w(x) for x in toks)
        coverage = mw / tw if tw else 0.0
        specificity = (mw / max_matched) if max_matched else 0.0
        # A SINGLE name-token match is often coincidental ("investigation" hits both
        # incident-related intent and the table investigation_..._party) and must NOT let
        # lexical override the learned reranker/retrieval signal. Mirrors select_primary_table:
        # one token contributes only coverage; 2+ matched tokens keep full lexical weight.
        lexical = (0.5 * coverage + 0.5 * specificity) if len(matched) >= 2 else 0.5 * coverage
        fp = _first_pos(toks)
        # Subject prior, discounted by the DISTINCTIVENESS of the word that claimed it:
        # an early generic token ("value") is a weak subject claim, an early rare one
        # ("payment") a strong one. IDF absent → weight 1.0 = legacy behavior.
        position = (1.0 - fp / nwords) * _w(words[fp]) if fp is not None else 0.0
        retrieval = (scores.get(t, 0.0) / maxret) if maxret else 0.0
        graph_sig = 0.0
        if adj is not None and len(candidate_tables) > 1:
            others = [o for o in candidate_tables if o != t]
            reach = sum(1 for o in others if _is_reachable(adj, t, o))
            graph_sig = reach / len(others) if others else 0.0
        composite = (W["lexical"] * lexical + W["position"] * position
                     + W["retrieval"] * retrieval + W["graph"] * graph_sig)
        # explicit grain declaration ("for each incident") pins that entity
        if grain is not None and toks == {grain}:
            composite = 1.0
        rows.append(AnchorCandidate(
            table=t, score=round(composite, 4),
            signals={"lexical": round(lexical, 3), "position": round(position, 3),
                     "retrieval": round(retrieval, 3), "graph": round(graph_sig, 3)}))
    rows.sort(key=lambda r: r.score, reverse=True)
    return rows


def select_anchor(query, candidate_tables, scores=None, graph=None):
    """Backward-compatible single-table pick: the top of score_anchors(). Callers that
    need the confidence margin / abstention should call score_anchors() directly."""
    ranked = score_anchors(query, candidate_tables, scores, graph=graph)
    return ranked[0].table if ranked else None


def _edge_tokens(edge):
    """Lowercase word tokens of an edge's join columns (for semantic tie-breaking)."""
    toks = set()
    for col in (edge.get("source_column", ""), edge.get("target_column", "")):
        toks |= {t for t in col.lower().split("_") if len(t) > 2}
    return toks


def _shortest_path(adj, src, dst, max_hops=3, allowed=None, max_cost=None, qtoks=None,
                   prefer_tables=None):
    """Weighted Dijkstra over edge weights. Returns list of edges or None.

    allowed (optional): the only tables that may appear as INTERMEDIATE hops. The
    necessity constraint is enforced during search (not after), so the planner finds
    the best path that routes only through requested/junction/named tables and never
    proposes a tunnel through an unrelated content table (e.g. role → organizations →
    permission). Endpoints src/dst are always permitted.

    max_cost (optional): cost ceiling on the whole path. With weights business=1,
    reference=2, audit=10, a ceiling expresses policy more naturally than a hop
    count — deep business chains pass, any audit chain is priced out.

    qtoks (optional): query tokens. Used ONLY as a secondary heap key, so among
    EXACT-equal-cost paths the one whose join columns the query names wins
    ("owned by" → owner_id edge, not created_by_id). It is a pure tie-breaker:
    it can never override the weight ordering, so an audit edge can never beat a
    business edge because of wording."""
    if src == dst:
        return []
    # A monotonic counter is the heap tiebreaker: when two entries tie on (cost,
    # -sem, node) the heap must NOT fall through to comparing `path` (a list of
    # dicts, which raises TypeError). Ties are common once the graph has many
    # equal-weight edges, so the counter keeps pushes strictly ordered.
    counter = 0
    pq = [(0, 0, counter, src, [])]
    seen = set()
    while pq:
        cost, neg_sem, _, node, path = heapq.heappop(pq)
        if node == dst:
            return path
        if node in seen or len(path) >= max_hops:
            continue
        seen.add(node)
        # Audit/ownership edges are TERMINAL: you may join a row to its owner
        # ("incident and its assigned user"), but you must never traverse THROUGH
        # the users hub to reach a third table ("incident → user → workflow" is not
        # a real relationship). So if we arrived here via an audit edge and this is
        # not the destination, it's a dead end — don't expand.
        if path and path[-1].get("relationship_type") == "audit":
            continue
        for nbr, edge in adj.get(node, []):
            if nbr in seen:
                continue
            # Necessity: never route THROUGH a table that isn't allowed as an
            # intermediate (it may still be the final destination).
            if allowed is not None and nbr != dst and nbr not in allowed:
                continue
            ncost = cost + edge.get("weight", 3)
            if max_cost is not None and ncost > max_cost:
                continue
            sem_gain = len(qtoks & _edge_tokens(edge)) if qtoks else 0
            # Junction preference (tie-break only): between equal-cost routes,
            # one through a pure bridge table IS the modeled relationship
            # (role→role_permissions→permission); one through a content table
            # (role→organizations→permission) encodes a different semantic
            # ("same organization") and must not win by heap order.
            if prefer_tables and (edge["source_table"] in prefer_tables
                                  or edge["target_table"] in prefer_tables):
                sem_gain += 1
            nsem = -neg_sem + sem_gain
            counter += 1
            heapq.heappush(pq, (ncost, -nsem, counter, nbr, path + [edge]))
    return None


def _edges_between(graph, a, b):
    out = []
    for e in graph.get("edges", []):
        if {e["source_table"], e["target_table"]} == {a, b}:
            out.append(e)
    return out


def _path_intermediate_tables(path, endpoints):
    """Tables a path passes THROUGH that are neither the anchor nor the target."""
    tbls = set()
    for e in path:
        tbls.add(e["source_table"]); tbls.add(e["target_table"])
    return tbls - set(endpoints)


def plan_joins(anchor, targets, graph, query="", allowed_intermediates=None):
    """Plan joins from anchor to each target. Returns a dict with the ordered
    edge path, confidence, why, and any unreachable/ambiguous targets.

    JOIN NECESSITY: a table may only appear in the join tree if it is the anchor, a
    requested target, a pure junction/bridge, or a table explicitly named in the
    query (a hub like `incident` in "signals per incident"). allowed_intermediates is
    that permitted set; it's enforced DURING the path search, so a path that could
    only connect two tables by tunnelling through an unrelated content table (e.g.
    role → organizations → permission, document_category → documents → investigation)
    is simply never found → the target is unreachable → the caller refuses."""
    adj = _adjacency(graph)
    path_edges, why, unreachable, ambiguous = [], [], [], []
    seen_pairs = set()
    allowed = (set(allowed_intermediates) if allowed_intermediates else set()) | {anchor} | set(targets)

    for tgt in targets:
        if tgt == anchor:
            continue
        # direct multi-edge ambiguity (e.g. created_by_id vs updated_by_id → users)
        direct = _edges_between(graph, anchor, tgt)
        if len(direct) > 1:
            # Disambiguate by the relation the query NAMES: match meaningful FK-column
            # parts (assigned_to_id → "assigned") against query WORDS. Word-set match,
            # not substring, and drop generic parts ("to"/"by"/"id"/"of") so they can't
            # spuriously match every edge ("to" in "total", "id" inside other words).
            qwords = set(re.findall(r"[a-z0-9]+", query.lower()))
            disamb = [e for e in direct
                      if {p for p in e["source_column"].lower().split("_") if len(p) > 2} & qwords]
            if len(disamb) != 1:
                ambiguous.append({"target": tgt, "options": [e["source_column"] for e in direct]})
                continue
            chosen = disamb
        else:
            chosen = _shortest_path(adj, anchor, tgt, allowed=allowed)

        if chosen is None:
            unreachable.append(tgt)
            continue
        for e in chosen:
            key = (e["source_table"], e["source_column"], e["target_table"], e["target_column"])
            if key not in seen_pairs:
                seen_pairs.add(key)
                path_edges.append(e)
                why.append(f"{e['source_table']}.{e['source_column']} → "
                           f"{e['target_table']}.{e['target_column']} "
                           f"({e['relationship_type']}, {e['cardinality']}, w{e['weight']})")

    # confidence
    conf = 1.0
    for e in path_edges:
        conf *= e.get("confidence", 1.0) * _HOP_PENALTY
        if e.get("polymorphic"):
            conf *= _POLY_PENALTY
        if e.get("discovery") == "data_inferred" and not e.get("polymorphic"):
            conf *= _INFERRED_PENALTY

    return {
        "anchor": anchor,
        "join_path": path_edges,
        "confidence": round(conf, 3) if path_edges else 1.0,
        "why": why,
        "unreachable": unreachable,
        "ambiguous": ambiguous,
        "max_fanout": _fanout_level(path_edges),
    }


def _fanout_level(edges):
    """Worst-case fan-out across the path (drives aggregation pre-agg decisions)."""
    if any(e["cardinality"] == "N:M" for e in edges):
        return "high"
    if any(e["cardinality"] in ("1:N", "N:1") for e in edges):
        return "medium"
    return "low"


def _ekey(e):
    return (e["source_table"], e["source_column"], e["target_table"], e["target_column"])


def _singular(w):
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _query_tokens(query):
    """Raw + singularized query tokens, so 'roles' matches edge/table token 'role'."""
    import re as _re
    raw = {w for w in _re.findall(r"[a-z0-9]+", (query or "").lower()) if len(w) > 2}
    return raw | {_singular(w) for w in raw}


def _table_tokens(table):
    """Delegates to veda.routing._name_toks — the canonical table-name tokenizer,
    which additionally segments Django's concatenated compound model names
    (assets_assetverificationdocument -> also 'asset'/'verification'/'document').
    A local import avoids a module-load-time cycle (routing.py only imports this
    module's score_anchors lazily, inside a function body)."""
    from veda.routing import _name_toks
    return _name_toks(table)


def plan_join_tree(anchor, targets, graph, query="", allowed_intermediates=None,
                   junctions=None):
    """Steiner-style join planner: ONE minimal tree connecting anchor + ALL targets,
    instead of independent pairwise paths. Same contract as plan_joins (same return
    shape, same edge rules: audit-terminal, necessity via `allowed`, weights), plus:

      • arbitrary target count (metric closure over terminals → MST → path union)
      • necessity by construction — every leaf of a Steiner tree is a terminal, so
        an unrequested table can never dangle off the tree
      • parallel direct edges that the query NAMES are ALL included as separate
        occurrences ("transitions with from state and to state" → both FK edges)
        instead of triggering a clarification
      • self-joins: a target equal to the anchor resolves through the anchor's
        self-edges (organizations.parent_id → organizations)
    """
    try:
        from config import JOIN_MAX_HOPS as _MH, JOIN_MAX_PATH_COST as _MC
    except Exception:
        _MH, _MC = 6, 8
    adj   = _adjacency(graph)
    qtoks = _query_tokens(query)

    terminals = [anchor] + [t for t in dict.fromkeys(targets) if t]
    allowed   = (set(allowed_intermediates) if allowed_intermediates else set()) | set(terminals)

    required, why = [], []          # forced occurrence edges (parallel / self)
    ambiguous, unreachable = [], []

    def _why(e):
        why.append(f"{e['source_table']}.{e['source_column']} → "
                   f"{e['target_table']}.{e['target_column']} "
                   f"({e['relationship_type']}, {e['cardinality']}, w{e['weight']})")

    # ── self-join targets (target == anchor) via the anchor's self-edges ──────
    real_targets = []
    for tgt in terminals[1:]:
        if tgt != anchor:
            real_targets.append(tgt)
            continue
        self_edges = [e for e in graph.get("edges", [])
                      if e["source_table"] == anchor and e["target_table"] == anchor]
        matched = [e for e in self_edges if qtoks & _edge_tokens(e)]
        pick = matched or (self_edges if len(self_edges) == 1 else [])
        if not pick and self_edges:
            ambiguous.append({"target": tgt,
                              "options": [e["source_column"] for e in self_edges]})
        elif not self_edges:
            unreachable.append(tgt)
        for e in pick:
            if _ekey(e) not in {_ekey(r) for r in required}:
                required.append(e)

    # ── direct parallel edges anchor→target: disambiguate or include ALL named ─
    pinned = set()
    for tgt in list(real_targets):
        # AUDIT edges (created_by/updated_by/assigned_to → user) are metadata, NOT the
        # entity relationship — exclude them from the parallel-key ambiguity test so
        # "permissions per user" routes via the user_permissions junction instead of
        # clarifying on created_by_id vs updated_by_id. The shortest-path step below
        # still uses an audit edge when it's genuinely the only route (and prefers the
        # cheaper junction path when one exists).
        direct = [e for e in _edges_between(graph, anchor, tgt)
                  if e.get("relationship_type") != "audit"]
        if len(direct) <= 1:
            continue
        ql = (query or "").lower()
        matched = [e for e in direct
                   if any(tok in ql for tok in e["source_column"].lower().split("_"))]
        if len(matched) == 0:
            ambiguous.append({"target": tgt,
                              "options": [e["source_column"] for e in direct]})
            real_targets.remove(tgt)
        else:
            # one named edge → that one; several named → ALL, as occurrences
            for e in matched:
                if _ekey(e) not in {_ekey(r) for r in required}:
                    required.append(e)
            pinned.add(tgt)
            real_targets.remove(tgt)

    # ── anchor-rooted attachment: one shortest path anchor→each target ────────
    # Deliberately NOT a free Steiner MST over all terminal pairs: "roles with
    # THEIR permissions" must attach permission via the role's own relationship
    # (role → role_permissions → permission), even when a target-target shortcut
    # is cheaper (permission.organization_id → organizations, cost 2 < 3). The
    # query's targets are the ANCHOR's related entities, so every target attaches
    # through a path rooted at the anchor; shared prefixes merge via dedup below,
    # so the union is still a minimal anchor-rooted tree.
    chosen = []
    for tgt in real_targets:
        # Tie-break tokens for THIS target's path exclude the name tokens of the
        # OTHER terminals: when role→permission ties between the role_permissions
        # route and an organizations route, the word "organization" in the query
        # belongs to the organizations TARGET and must not pull this path toward it.
        other_toks = set()
        for o in terminals:
            if o not in (tgt,):
                other_toks |= _table_tokens(o)
        other_toks -= _table_tokens(tgt) | _table_tokens(anchor)
        qt_t = qtoks - other_toks
        p = _shortest_path(adj, anchor, tgt, max_hops=_MH, allowed=allowed,
                           max_cost=_MC, qtoks=qt_t, prefer_tables=junctions)
        if p is None:
            unreachable.append(tgt)
        else:
            chosen.append(p)

    # ── union (dedup) + cycle-break: required edges are exempt (intentional
    #    parallels / self-joins); everything else must keep the union a tree ────
    union, seen_keys = [], set()
    for e in required:
        if _ekey(e) not in seen_keys:
            seen_keys.add(_ekey(e))
            union.append((e, True))
    for p in chosen:
        for e in p:
            if _ekey(e) not in seen_keys:
                seen_keys.add(_ekey(e))
                union.append((e, False))

    parent = {}
    def _find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    final = []
    for e, is_required in sorted(union, key=lambda t: (not t[1], t[0].get("weight", 3))):
        if is_required:
            _union(e["source_table"], e["target_table"])
            final.append(e)
        elif _union(e["source_table"], e["target_table"]):
            final.append(e)
    for e in final:
        _why(e)

    conf = 1.0
    for e in final:
        conf *= e.get("confidence", 1.0) * _HOP_PENALTY
        if e.get("polymorphic"):
            conf *= _POLY_PENALTY
        if e.get("discovery") == "data_inferred" and not e.get("polymorphic"):
            conf *= _INFERRED_PENALTY

    return {
        "anchor": anchor,
        "join_path": final,
        "confidence": round(conf, 3) if final else 1.0,
        "why": why,
        "unreachable": sorted(set(unreachable)),
        "ambiguous": ambiguous,
        "max_fanout": _fanout_level(final),
        "planner": "tree",
    }


def build_skeleton(plan):
    """Deterministic FROM/JOIN/ON skeleton with OCCURRENCE-based aliases. The LLM
    fills SELECT/WHERE. Returns (from_clause, alias_map). alias_map: alias -> table
    (occurrence-keyed: the same table may appear under several aliases, which is
    what makes same-table-twice and self-joins expressible —
      transition t0 JOIN state t1 ON t1.id=t0.from_state_id
                    JOIN state t2 ON t2.id=t0.to_state_id
      organizations t0 JOIN organizations t1 ON t1.org_id=t0.parent_id)."""
    anchor = plan["anchor"]
    alias_map = {"t0": anchor}          # alias -> table (occurrence-keyed)
    primary = {anchor: "t0"}            # table -> FIRST alias (driving-side reuse)
    lines = [f'FROM "{anchor}" t0']
    emitted = set()

    pending = list(plan["join_path"])
    guard = 0
    while pending and guard < 200:
        guard += 1
        progressed = False
        for e in list(pending):
            s, tt = e["source_table"], e["target_table"]
            key = (s, e["source_column"], tt, e["target_column"])
            if key in emitted:
                pending.remove(e); progressed = True; continue
            s_in, t_in = s in primary, tt in primary
            if not (s_in or t_in):
                continue                # neither side anchored yet; try later
            if s_in and t_in:
                # Both sides already aliased and this edge not yet emitted →
                # SAME-TABLE-TWICE or SELF-JOIN occurrence. The new alias goes to
                # the TARGET side (the looked-up entity); the source's primary
                # alias drives the ON.
                known_t, known_col = s, e["source_column"]
                new_t,   new_col   = tt, e["target_column"]
            elif s_in:
                known_t, known_col = s, e["source_column"]
                new_t,   new_col   = tt, e["target_column"]
            else:
                known_t, known_col = tt, e["target_column"]
                new_t,   new_col   = s, e["source_column"]
            ka = primary[known_t]
            na = f"t{len(alias_map)}"
            alias_map[na] = new_t
            primary.setdefault(new_t, na)      # first alias of a table stays primary
            on = f'{na}."{new_col}" = {ka}."{known_col}"'
            if e.get("requires_predicate"):
                # rewrite "table.col" → alias.col using THIS join's two occurrences
                # (a polymorphic predicate names the discriminator's own table)
                pred = e["requires_predicate"]
                for tbl, al in ((known_t, ka), (new_t, na)):
                    pred = pred.replace(f"{tbl}.", f"{al}.")
                on += f" AND {pred}"
            lines.append(f'JOIN "{new_t}" {na} ON {on}')
            emitted.add(key)
            pending.remove(e)
            progressed = True
        if not progressed:
            break
    return "\n".join(lines), alias_map


if __name__ == "__main__":
    g = load_graph()
    print(f"graph: {g.get('stats', {})}\n")

    # Test 1: the real polymorphic join (annotation anchor → counterparty)
    p = plan_joins("annotation_record", ["counterparty_details"], g,
                   query="annotations for counterparties")
    print("TEST 1  annotation_record + counterparty_details")
    print("  confidence:", p["confidence"], "| fanout:", p["max_fanout"])
    print("  why:", p["why"])
    skel, aliases = build_skeleton(p)
    print("  skeleton:\n   " + skel.replace("\n", "\n   "))
    print()

    # Test 2: unreachable (no FK path)
    p2 = plan_joins("dashboards", ["counterparty_details"], g, query="dashboards and counterparties")
    print("TEST 2  dashboards + counterparty_details (expect unreachable)")
    print("  unreachable:", p2["unreachable"])
    print()

    # Test 3: declared FK join
    p3 = plan_joins("dashboard_items", ["dashboards"], g, query="dashboard items with their dashboards")
    print("TEST 3  dashboard_items + dashboards")
    skel3, _ = build_skeleton(p3)
    print("  skeleton:\n   " + skel3.replace("\n", "\n   "))
