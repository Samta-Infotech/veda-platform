"""VEDA · author-agnostic graph guards (Tier-2 keystone).

Validate ANY built SQL against the relationship graph, with no deterministic plan:

  verify_joins_against_graph(sql)   — every base-table JOIN ON a.x=b.y must be a REAL
                                      FK edge (declared or discovered). Closes the
                                      wrong-join class for LLM-IR SQL by verification.
  check_connectivity(sql)           — all FROM/JOIN aliases must form ONE connected
                                      component via ON equalities (no cartesian / no
                                      disconnected sub-structures).
  fanout_parent_aliases(sql)        — derive parent(1)-side aliases from graph
                                      cardinality so the existing fan-out guard works
                                      on plan-less SQL.

These run AFTER the deterministic SQL builder (LLM authors IR, builder authors SQL)
and BEFORE execution — alongside validate_and_parameterize. All name-based, so they
work regardless of who/what produced the SQL.
"""
import os
import json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GRAPH = {"v": None, "edges": None, "card": None}


def _load_graph():
    if _GRAPH["v"] is None:
        path = os.path.join(_ROOT, "data", "veda_relationship_graph.json")
        g = json.load(open(path)) if os.path.exists(path) else {"edges": []}
        edge_set, card = set(), {}
        for e in g.get("edges", []):
            a = (e["source_table"].lower(), e["source_column"].lower())
            b = (e["target_table"].lower(), e["target_column"].lower())
            edge_set.add(frozenset((a, b)))
            # cardinality keyed by the unordered table pair → parent-side table(s)
            c = (e.get("cardinality") or "").upper()
            st, tt = e["source_table"].lower(), e["target_table"].lower()
            if c == "N:1":
                parents = {tt}
            elif c == "1:N":
                parents = {st}
            elif c == "N:M":
                parents = {st, tt}
            else:                       # 1:1 or unknown → no fan-out
                parents = set()
            card[frozenset((a, b))] = parents
        _GRAPH.update(v=g, edges=edge_set, card=card)
    return _GRAPH


def _parse(sql):
    import sqlglot
    from sqlglot import exp
    try:
        return sqlglot.parse_one(sql, read="postgres"), exp
    except Exception:
        return None, None


def _alias_map(tree, exp):
    """alias/name (lower) → real table name (lower). CTE names map to themselves but
    are tracked separately so join verification can skip derived-table joins."""
    amap, ctes = {}, set()
    for cte in (getattr(tree, "ctes", None) or []):
        if cte.alias:
            ctes.add(cte.alias.lower())
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        amap[(t.alias or t.name).lower()] = name
        amap[name] = name
    return amap, ctes


def _join_eq_pairs(tree, exp):
    """[(left_alias, left_col, right_alias, right_col)] for every column=column
    equality inside a JOIN ... ON. column=literal predicates are ignored."""
    out = []
    for join in tree.find_all(exp.Join):
        on = join.args.get("on")
        if on is None:
            continue
        for eq in on.find_all(exp.EQ):
            l, r = eq.this, eq.expression
            if isinstance(l, exp.Column) and isinstance(r, exp.Column):
                out.append(((l.table or "").lower(), l.name.lower(),
                            (r.table or "").lower(), r.name.lower()))
    return out


def verify_joins_against_graph(sql, graph=None):
    """(ok, reason|None). Every base-table↔base-table JOIN key must be a real FK edge.
    Joins involving a CTE/derived alias are skipped (not entity FKs). An unresolved
    alias on a base join → reject (can't verify = unsafe)."""
    g = graph or _load_graph()
    tree, exp = _parse(sql)
    if tree is None:
        return True, None                      # parse errors handled by the AST validator
    amap, ctes = _alias_map(tree, exp)
    for la, lc, ra, rc in _join_eq_pairs(tree, exp):
        lt, rt = amap.get(la), amap.get(ra)
        # derived-table join (e.g. pre-agg CTE): the resolved table IS a CTE name.
        # Check the resolved table, not the alias — `JOIN agg_0 a0` resolves a0→agg_0.
        if (lt in ctes) or (rt in ctes):
            continue
        if not lt or not rt:
            return False, f"unverifiable join key (unresolved alias): {la}.{lc} = {ra}.{rc}"
        pair = frozenset(((lt, lc), (rt, rc)))
        if pair not in g["edges"]:
            return False, f"join not backed by a real FK edge: {lt}.{lc} = {rt}.{rc}"
    return True, None


def check_connectivity(sql, graph=None):
    """(ok, reason|None). The TOP-LEVEL FROM/JOIN tables must form ONE component via
    their JOIN ON equalities — catches a real cartesian (FROM a, b with nothing
    connecting them). Tables INSIDE subqueries / CTEs / EXISTS are separate scopes
    (correlated or derived) and are deliberately NOT counted — they aren't cartesians."""
    tree, exp = _parse(sql)
    if tree is None:
        return True, None

    # A table is TOP-LEVEL iff its enclosing SELECT is the outermost one. Tables inside
    # a subquery / CTE definition / EXISTS have a different enclosing SELECT and are a
    # separate scope (correlated/derived) — not part of a cartesian. This also captures
    # comma-cartesians (FROM a, b) which both sit under the outer SELECT.
    nodes = set()
    for t in tree.find_all(exp.Table):
        if t.find_ancestor(exp.Select) is tree:
            nodes.add((t.alias or t.name).lower())

    # ON pairs from top-level JOINs only (joins whose enclosing SELECT is the outer one)
    pairs = []
    for j in tree.find_all(exp.Join):
        if j.find_ancestor(exp.Select) is not tree:
            continue
        on = j.args.get("on")
        if on is not None:
            for eq in on.find_all(exp.EQ):
                l, r = eq.this, eq.expression
                if isinstance(l, exp.Column) and isinstance(r, exp.Column):
                    pairs.append(((l.table or "").lower(), (r.table or "").lower()))

    if len(nodes) <= 1:
        return True, None
    parent = {n: n for n in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in pairs:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    roots = {find(n) for n in nodes}
    if len(roots) > 1:
        return False, f"disconnected query: {len(roots)} table groups not joined (cartesian risk)"
    return True, None


def fanout_parent_aliases(sql, graph=None):
    """Set of parent(1)-side aliases, derived from graph cardinality, for the existing
    fan-out guard — so COUNT/SUM/AVG over a parent column is caught on plan-less SQL."""
    g = graph or _load_graph()
    tree, exp = _parse(sql)
    if tree is None:
        return set()
    amap, ctes = _alias_map(tree, exp)
    # table -> set of aliases in this query (a table may appear under several)
    table_aliases = {}
    for t in tree.find_all(exp.Table):
        table_aliases.setdefault(t.name.lower(), set()).add((t.alias or t.name).lower())
    parents = set()
    for la, lc, ra, rc in _join_eq_pairs(tree, exp):
        if la in ctes or ra in ctes:
            continue
        lt, rt = amap.get(la), amap.get(ra)
        if not lt or not rt:
            continue
        pair = frozenset(((lt, lc), (rt, rc)))
        for ptable in g["card"].get(pair, set()):
            parents |= table_aliases.get(ptable, set())
    return parents
