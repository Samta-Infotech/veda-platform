#!/usr/bin/env python3
"""
ingestion/unified_graph_builder.py — Phase 1 of the Unified Knowledge Graph.

Fuses VEDA's separate graph-like artifacts into ONE node/edge graph:
    data/veda_semantic_model.json      → TABLE, COLUMN nodes + HAS_COLUMN
    data/veda_relationship_graph.json  → FK_TO (table↔table) + REFERENCES (col↔col)
    data/veda_concept_graph.json       → CONCEPT nodes + IS_CONCEPT
    data/veda_domain_synonyms.json     → SYNONYM nodes + SYNONYM_OF
    semantic/metrics.json              → METRIC nodes + IS_METRIC
    semantic/dimensions.json           → DIMENSION nodes + IS_DIMENSION
    (column aliases in semantic model)  → ALIAS_OF

This does NOT replace any existing artifact — it is a derived, additive view that the
existing builders keep feeding. Output: data/veda_unified_graph.json.

Design constraints honoured:
  • Zero new dependencies (pure stdlib) — meets <5min build / reasonable memory trivially.
  • Generic — no table/column/business names in code; everything derives from the artifacts.
  • Idempotent — deterministic ordering (sorted) → same inputs always yield the same graph.
  • Graceful — any missing/unreadable artifact is skipped with a warning, never crashes.

Node id scheme (stable, joinable to existing code):
  table:{table}   col:{table}.{col}   concept:{NAME}   metric:{id}   dim:{id}   syn:{term}

Usage:
    python3 ingestion/unified_graph_builder.py            # build + write + print stats
    python3 ingestion/unified_graph_builder.py --quiet
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _p(rel: str) -> str:
    return os.path.join(_ROOT, rel)


# Resolve artifact paths from config when available, else fall back to defaults.
try:
    import config as _cfg
    _SEMANTIC_MODEL = _p(getattr(_cfg, "SEMANTIC_MODEL_FILE", "data/veda_semantic_model.json"))
    _REL_GRAPH      = _p(getattr(_cfg, "RELATIONSHIP_GRAPH_FILE", "data/veda_relationship_graph.json"))
    _CONCEPT_GRAPH  = _p(getattr(_cfg, "CONCEPT_GRAPH_FILE", "data/veda_concept_graph.json"))
    _DOMAIN_SYN     = _p(getattr(_cfg, "DOMAIN_SYNONYMS_FILE", "data/veda_domain_synonyms.json"))
    _OUT_FILE       = _p(getattr(_cfg, "UNIFIED_GRAPH_FILE", "data/veda_unified_graph.json"))
except Exception:
    _SEMANTIC_MODEL = _p("data/veda_semantic_model.json")
    _REL_GRAPH      = _p("data/veda_relationship_graph.json")
    _CONCEPT_GRAPH  = _p("data/veda_concept_graph.json")
    _DOMAIN_SYN     = _p("data/veda_domain_synonyms.json")
    _OUT_FILE       = _p("data/veda_unified_graph.json")

_METRICS    = _p("semantic/metrics.json")
_DIMENSIONS = _p("semantic/dimensions.json")

GRAPH_VERSION = "1.0"


# ─────────────────────────────────────────────────────────────────────────────
def _load(path: str) -> Optional[Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# Node id helpers — single source of truth, reused by query_graph.py.
def table_id(t: str) -> str:  return f"table:{t}"
def col_id(t: str, c: str) -> str:  return f"col:{t}.{c}"
def concept_id(name: str) -> str:  return f"concept:{name}"
def metric_id(mid: str) -> str:  return f"metric:{mid}"
def dim_id(did: str) -> str:  return f"dim:{did}"
def syn_id(term: str) -> str:  return f"syn:{term.strip().lower()}"


class _GraphAccumulator:
    """Collects nodes (deduped by id) and edges (deduped by (src,tgt,type))."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: Dict[tuple, Dict[str, Any]] = {}

    def node(self, nid: str, ntype: str, name: str, **metadata) -> str:
        if nid not in self._nodes:
            self._nodes[nid] = {"id": nid, "type": ntype, "name": name,
                                "metadata": {k: v for k, v in metadata.items() if v is not None}}
        return nid

    def edge(self, source: str, target: str, etype: str, **metadata) -> None:
        # Both endpoints must exist as nodes — never emit an edge to a phantom (grounded-only).
        if source not in self._nodes or target not in self._nodes:
            return
        key = (source, target, etype)
        if key not in self._edges:
            e: Dict[str, Any] = {"source": source, "target": target, "type": etype}
            if metadata:
                e["metadata"] = {k: v for k, v in metadata.items() if v is not None}
            self._edges[key] = e

    def to_graph(self) -> Dict[str, Any]:
        nodes = sorted(self._nodes.values(), key=lambda n: (n["type"], n["id"]))
        edges = sorted(self._edges.values(), key=lambda e: (e["type"], e["source"], e["target"]))
        type_counts: Dict[str, int] = {}
        for n in nodes:
            type_counts[n["type"]] = type_counts.get(n["type"], 0) + 1
        edge_counts: Dict[str, int] = {}
        for e in edges:
            edge_counts[e["type"]] = edge_counts.get(e["type"], 0) + 1
        return {
            "version": GRAPH_VERSION,
            "stats": {
                "nodes": len(nodes),
                "edges": len(edges),
                "node_types": type_counts,
                "edge_types": edge_counts,
            },
            "nodes": nodes,
            "edges": edges,
        }


def build_unified_graph() -> Dict[str, Any]:
    """Build the unified graph from the on-disk artifacts. Always returns a valid graph
    dict (possibly small) — missing artifacts are skipped, never fatal."""
    g = _GraphAccumulator()
    warnings: List[str] = []

    sm = _load(_SEMANTIC_MODEL)
    if not sm:
        warnings.append(f"semantic model missing/unreadable: {_SEMANTIC_MODEL}")
        sm = {}

    tables: Dict[str, Any] = sm.get("tables", {}) or {}
    columns: Dict[str, Any] = sm.get("columns", {}) or {}

    # ── TABLE nodes ──────────────────────────────────────────────────────────
    for tname, trec in sorted(tables.items()):
        g.node(table_id(tname), "TABLE", tname,
               business_purpose=(trec or {}).get("business_purpose"),
               primary_entity=(trec or {}).get("primary_entity"),
               table_type=(trec or {}).get("table_type"))

    # ── COLUMN nodes + HAS_COLUMN + ALIAS_OF ─────────────────────────────────
    for ckey, crec in sorted(columns.items()):
        t = crec.get("table_name")
        c = crec.get("col_name")
        if not t or not c:
            continue
        # ensure the owning table exists even if it wasn't in tables{}
        g.node(table_id(t), "TABLE", t)
        cid = g.node(col_id(t, c), "COLUMN", f"{t}.{c}",
                     semantic_type=crec.get("semantic_type"),
                     analytics_role=crec.get("analytics_role"),
                     business_role=crec.get("business_role"),
                     business_definition=crec.get("business_definition"),
                     importance_class=crec.get("importance_class"),
                     contains_pii=crec.get("contains_pii"))
        g.edge(table_id(t), cid, "HAS_COLUMN")
        for alias in sorted(set(crec.get("aliases") or [])):
            a = alias.strip().lower()
            if not a:
                continue
            sid = g.node(syn_id(a), "SYNONYM", a)
            g.edge(sid, cid, "ALIAS_OF")

    # ── FK_TO (table↔table) + REFERENCES (col↔col) ───────────────────────────
    rg = _load(_REL_GRAPH)
    if rg:
        for e in rg.get("edges", []) or []:
            st, sc = e.get("source_table"), e.get("source_column")
            tt, tc = e.get("target_table"), e.get("target_column")
            if not (st and tt):
                continue
            g.node(table_id(st), "TABLE", st)
            g.node(table_id(tt), "TABLE", tt)
            g.edge(table_id(st), table_id(tt), "FK_TO",
                   cardinality=e.get("cardinality"),
                   relationship_type=e.get("relationship_type"),
                   discovery=e.get("discovery"),
                   polymorphic=e.get("polymorphic"))
            if sc and tc:
                src_c = g.node(col_id(st, sc), "COLUMN", f"{st}.{sc}")
                tgt_c = g.node(col_id(tt, tc), "COLUMN", f"{tt}.{tc}")
                g.edge(src_c, tgt_c, "REFERENCES", cardinality=e.get("cardinality"))
    else:
        warnings.append(f"relationship graph missing: {_REL_GRAPH}")

    # ── CONCEPT nodes + IS_CONCEPT ───────────────────────────────────────────
    cg = _load(_CONCEPT_GRAPH)
    if cg:
        for cname, members in sorted(cg.items()):
            kid = g.node(concept_id(cname), "CONCEPT", cname)
            for m in members or []:
                t, c = m.get("table"), m.get("column")
                if not (t and c):
                    continue
                cid = g.node(col_id(t, c), "COLUMN", f"{t}.{c}")
                g.edge(cid, kid, "IS_CONCEPT", role=m.get("role"))
    else:
        warnings.append(f"concept graph missing: {_CONCEPT_GRAPH}")

    # ── METRIC nodes + IS_METRIC ─────────────────────────────────────────────
    met = _load(_METRICS)
    if met and isinstance(met.get("items"), dict):
        for mid, mrec in sorted(met["items"].items()):
            owner = (mrec or {}).get("source_table") or (mrec or {}).get("owner_table")
            nid = g.node(metric_id(mid), "METRIC", mid,
                         kind=(mrec or {}).get("kind"), owner_table=owner,
                         expression=(mrec or {}).get("expression"))
            if owner:
                g.node(table_id(owner), "TABLE", owner)
                g.edge(nid, table_id(owner), "IS_METRIC")
            for lbl in sorted(set((mrec or {}).get("labels") or [])):
                sid = g.node(syn_id(lbl), "SYNONYM", lbl.strip().lower())
                g.edge(sid, nid, "SYNONYM_OF")

    # ── DIMENSION nodes + IS_DIMENSION ───────────────────────────────────────
    dim = _load(_DIMENSIONS)
    if dim and isinstance(dim.get("items"), dict):
        for did, drec in sorted(dim["items"].items()):
            owner = (drec or {}).get("owner_table")
            colname = (drec or {}).get("col_name")
            nid = g.node(dim_id(did), "DIMENSION", did, owner_table=owner)
            if owner and colname:
                cid = g.node(col_id(owner, colname), "COLUMN", f"{owner}.{colname}")
                g.edge(nid, cid, "IS_DIMENSION")
            for lbl in sorted(set((drec or {}).get("labels") or [])):
                sid = g.node(syn_id(lbl), "SYNONYM", lbl.strip().lower())
                g.edge(sid, nid, "SYNONYM_OF")

    # ── SYNONYM_OF (domain synonyms term → column) ───────────────────────────
    ds = _load(_DOMAIN_SYN)
    if ds:
        for term, targets in sorted(ds.items()):
            sid = g.node(syn_id(term), "SYNONYM", term.strip().lower())
            for tgt in targets or []:
                if "." not in tgt:
                    continue
                t, c = tgt.split(".", 1)
                cid = g.node(col_id(t, c), "COLUMN", f"{t}.{c}")
                g.edge(sid, cid, "SYNONYM_OF")
    else:
        warnings.append(f"domain synonyms missing: {_DOMAIN_SYN}")

    graph = g.to_graph()
    graph["stats"]["warnings"] = warnings
    return graph


def write_unified_graph(graph: Optional[Dict[str, Any]] = None) -> str:
    if graph is None:
        graph = build_unified_graph()
    os.makedirs(os.path.dirname(_OUT_FILE), exist_ok=True)
    with open(_OUT_FILE, "w") as f:
        json.dump(graph, f, indent=2)
    return _OUT_FILE


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the VEDA unified knowledge graph.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    graph = build_unified_graph()
    path = write_unified_graph(graph)
    s = graph["stats"]
    if not args.quiet:
        print(f"Unified graph → {os.path.relpath(path, _ROOT)}")
        print(f"  nodes: {s['nodes']}   edges: {s['edges']}")
        print("  node types: " + "  ".join(f"{k}={v}" for k, v in sorted(s["node_types"].items())))
        print("  edge types: " + "  ".join(f"{k}={v}" for k, v in sorted(s["edge_types"].items())))
        if s.get("warnings"):
            print("  warnings:")
            for w in s["warnings"]:
                print(f"    - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
