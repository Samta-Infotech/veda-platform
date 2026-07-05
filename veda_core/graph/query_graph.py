#!/usr/bin/env python3
"""
graph/query_graph.py — Phase 3: in-memory query engine over the unified graph.

Loads data/veda_unified_graph.json ONCE into adjacency dicts (module-level cache) and
answers traversal queries in well under 20ms. Pure stdlib — no networkx — so it adds no
dependency and is trivially fast at this scale (~4k nodes / ~7.7k edges).

Public API:
    get_table_node(name)              get_column_node(table, col)
    get_concept_node(name)            get_metric_node(id)   get_dimension_node(id)
    get_neighbors(node_id, edge_types=None, direction="both")
    get_synonyms(term)                → ["priority", "urgency", ...]
    get_related_tables(table)         → tables reachable via FK_TO
    get_related_columns(table_or_col) → columns linked by REFERENCES / HAS_COLUMN
    shortest_path(a, b)               → [node_id, ...]  (BFS, undirected)
    resolve_term(term)                → column node ids a free term maps to (synonym/alias/name)

Reuses the node-id helpers from the builder so ids stay consistent across the codebase.
"""

from __future__ import annotations

import os
import sys
import json
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.unified_graph_builder import (
    table_id, col_id, concept_id, metric_id, dim_id, syn_id,
)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    import config as _cfg
    _GRAPH_FILE = os.path.join(_ROOT, getattr(_cfg, "UNIFIED_GRAPH_FILE", "data/veda_unified_graph.json"))
except Exception:
    _GRAPH_FILE = os.path.join(_ROOT, "data", "veda_unified_graph.json")


class UnifiedGraph:
    """Adjacency-indexed view of the unified graph. Cheap lookups, BFS traversal."""

    def __init__(self, graph: Dict[str, Any]):
        self.nodes: Dict[str, dict] = {n["id"]: n for n in graph.get("nodes", [])}
        # directed adjacency: out[src] = [(tgt, type)], in_[tgt] = [(src, type)]
        self.out: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.in_: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for e in graph.get("edges", []):
            s, t, ty = e["source"], e["target"], e["type"]
            self.out[s].append((t, ty))
            self.in_[t].append((s, ty))
        self.stats: Dict[str, Any] = graph.get("stats", {})

    # ── node lookups ─────────────────────────────────────────────────────────
    def node(self, node_id: str) -> Optional[dict]:
        return self.nodes.get(node_id)

    def get_table_node(self, name: str) -> Optional[dict]:
        return self.nodes.get(table_id(name))

    def get_column_node(self, table: str, col: str) -> Optional[dict]:
        return self.nodes.get(col_id(table, col))

    def get_concept_node(self, name: str) -> Optional[dict]:
        return self.nodes.get(concept_id(name))

    def get_metric_node(self, mid: str) -> Optional[dict]:
        return self.nodes.get(metric_id(mid))

    def get_dimension_node(self, did: str) -> Optional[dict]:
        return self.nodes.get(dim_id(did))

    # ── neighbourhood ────────────────────────────────────────────────────────
    def get_neighbors(self, node_id: str, edge_types: Optional[Set[str]] = None,
                      direction: str = "both") -> List[Tuple[str, str]]:
        """[(neighbor_id, edge_type)] filtered by edge_types and direction
        (out | in | both)."""
        res: List[Tuple[str, str]] = []
        if direction in ("out", "both"):
            for t, ty in self.out.get(node_id, []):
                if edge_types is None or ty in edge_types:
                    res.append((t, ty))
        if direction in ("in", "both"):
            for s, ty in self.in_.get(node_id, []):
                if edge_types is None or ty in edge_types:
                    res.append((s, ty))
        return res

    # ── term resolution (synonym/alias/exact column, plural-aware) ───────────
    def resolve_term(self, term: str) -> List[str]:
        """Column node ids a free-text term maps to: via SYNONYM/ALIAS edges, or a
        direct column-name match. Plural-aware (reuses the project singularizer so
        'handlers' matches the alias 'handler'). Ordered, deduped, deterministic."""
        from retrieval.query_enrichment import _singularize
        out: List[str] = []
        t = term.strip().lower()
        # exact first, then singular form (only if different) — never invent variants here.
        for var in [t, _singularize(t)]:
            sid = syn_id(var)
            if sid in self.nodes:
                for tgt, ty in self.out.get(sid, []):
                    if ty in ("SYNONYM_OF", "ALIAS_OF") and self.nodes.get(tgt, {}).get("type") == "COLUMN":
                        out.append(tgt)
            for nid, n in self.nodes.items():
                if n["type"] == "COLUMN" and n["name"].split(".", 1)[-1].lower() == var:
                    out.append(nid)
            if out:
                break
        seen, uniq = set(), []
        for x in out:
            if x not in seen:
                seen.add(x); uniq.append(x)
        return uniq

    def get_synonyms(self, term: str) -> List[str]:
        """OTHER surface terms (synonyms/aliases) for the column(s) a term resolves to.
        get_synonyms('severity') → ['priority', 'urgency', ...]."""
        cols = self.resolve_term(term)
        terms: List[str] = []
        term_l = term.strip().lower()
        for cid in cols:
            for src, ty in self.in_.get(cid, []):
                if ty in ("SYNONYM_OF", "ALIAS_OF") and self.nodes.get(src, {}).get("type") == "SYNONYM":
                    name = self.nodes[src]["name"]
                    if name != term_l:
                        terms.append(name)
        seen, uniq = set(), []
        for x in terms:
            if x not in seen:
                seen.add(x); uniq.append(x)
        return uniq

    # ── relationships ────────────────────────────────────────────────────────
    def get_related_tables(self, table: str) -> List[str]:
        """Table names connected to `table` via FK_TO (either direction)."""
        tid = table_id(table)
        out: List[str] = []
        for nb, ty in self.get_neighbors(tid, edge_types={"FK_TO"}, direction="both"):
            n = self.nodes.get(nb)
            if n and n["type"] == "TABLE":
                out.append(n["name"])
        seen, uniq = set(), []
        for x in out:
            if x not in seen:
                seen.add(x); uniq.append(x)
        return uniq

    def get_related_columns(self, node_id: str) -> List[str]:
        """Columns related to a table (HAS_COLUMN) or a column (REFERENCES, both ways)."""
        n = self.nodes.get(node_id)
        if not n:
            # accept a bare table name too
            if table_id(node_id) in self.nodes:
                node_id = table_id(node_id)
                n = self.nodes[node_id]
            else:
                return []
        out: List[str] = []
        if n["type"] == "TABLE":
            for nb, ty in self.get_neighbors(node_id, edge_types={"HAS_COLUMN"}, direction="out"):
                out.append(nb)
        elif n["type"] == "COLUMN":
            for nb, ty in self.get_neighbors(node_id, edge_types={"REFERENCES"}, direction="both"):
                out.append(nb)
        return out

    # ── retrieval expansion (engine-agnostic; returns plain 'table.col' names) ──
    def suggest_expansions(self, query: str, have_cols: Set[str], have_tables: Set[str],
                           max_add: int = 12):
        """Graph-suggested column names ('table.col') to ADD to a retrieval candidate set.
        Pure names (no engine-specific result type) so any caller can wrap them. Returns
        (seed_terms, added_names, synonyms_map). Deterministic; never raises here."""
        import re as _re
        seen = set(have_cols)
        seeds: List[str] = []
        added: List[str] = []
        resolved_cids: List[str] = []
        # 1) synonym/alias/name resolution of query tokens → columns (the high-signal part)
        for tok in [t for t in _re.findall(r"[a-zA-Z_]+", query.lower()) if len(t) > 2]:
            cids = self.resolve_term(tok)
            if cids:
                seeds.append(tok)
            for cid in cids:
                resolved_cids.append(cid)
                n = self.nodes.get(cid)
                if n and "." in n["name"] and n["name"] not in seen:
                    seen.add(n["name"]); added.append(n["name"])
        # 2) FK JOIN-KEY reach — ONLY the specific column a resolved column REFERENCES (the
        #    join key on the other side), e.g. assigned_to_id → user.user_id. NOT every column
        #    of the neighbour table (that flooded results with user.email etc.).
        for cid in resolved_cids:
            for nb, ty in self.get_neighbors(cid, {"REFERENCES"}, "both"):
                n = self.nodes.get(nb)
                if n and "." in n["name"] and n["name"] not in seen:
                    seen.add(n["name"]); added.append(n["name"])
        added = added[:max_add]
        synonyms = {s: self.get_synonyms(s)[:6] for s in seeds}
        return seeds, added, synonyms

    # ── pathfinding ──────────────────────────────────────────────────────────
    def shortest_path(self, a: str, b: str, max_hops: int = 8) -> List[str]:
        """BFS shortest path (undirected) between two node ids. [] if none/too far."""
        if a not in self.nodes or b not in self.nodes:
            return []
        if a == b:
            return [a]
        prev: Dict[str, Optional[str]] = {a: None}
        q = deque([(a, 0)])
        while q:
            cur, d = q.popleft()
            if d >= max_hops:
                continue
            for nb, _ty in (self.out.get(cur, []) + self.in_.get(cur, [])):
                if nb not in prev:
                    prev[nb] = cur
                    if nb == b:
                        path = [nb]
                        while prev[path[-1]] is not None:
                            path.append(prev[path[-1]])
                        return list(reversed(path))
                    q.append((nb, d + 1))
        return []


# ── module-level singleton (load once per process) ───────────────────────────
_GRAPH: Optional[UnifiedGraph] = None


def get_graph(path: str = _GRAPH_FILE, force_reload: bool = False) -> Optional[UnifiedGraph]:
    """Return the cached UnifiedGraph; None if the artifact is missing/unreadable."""
    global _GRAPH
    if _GRAPH is not None and not force_reload:
        return _GRAPH
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    _GRAPH = UnifiedGraph(data)
    return _GRAPH


# ── thin module-level wrappers (the names the spec lists) ─────────────────────
def get_table_node(name: str):  g = get_graph();  return g.get_table_node(name) if g else None
def get_column_node(t: str, c: str):  g = get_graph();  return g.get_column_node(t, c) if g else None
def get_concept_node(name: str):  g = get_graph();  return g.get_concept_node(name) if g else None
def get_neighbors(node_id: str, edge_types=None, direction="both"):
    g = get_graph();  return g.get_neighbors(node_id, edge_types, direction) if g else []
def get_synonyms(term: str):  g = get_graph();  return g.get_synonyms(term) if g else []
def get_related_tables(table: str):  g = get_graph();  return g.get_related_tables(table) if g else []
def get_related_columns(node_id: str):  g = get_graph();  return g.get_related_columns(node_id) if g else []
def shortest_path(a: str, b: str):  g = get_graph();  return g.shortest_path(a, b) if g else []
def resolve_term(term: str):  g = get_graph();  return g.resolve_term(term) if g else []
def suggest_expansions(query, have_cols, have_tables, max_add=12):
    g = get_graph()
    return g.suggest_expansions(query, have_cols, have_tables, max_add) if g else ([], [], {})


if __name__ == "__main__":
    import time
    g = get_graph()
    if not g:
        print("no unified graph — run: python3 ingestion/unified_graph_builder.py")
        raise SystemExit(1)
    print(f"loaded {len(g.nodes)} nodes")
    for term in ("severity", "priority", "status", "editor", "author"):
        t0 = time.perf_counter()
        syns = g.get_synonyms(term)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  get_synonyms({term!r}) = {syns[:6]}   [{dt:.2f}ms]")
