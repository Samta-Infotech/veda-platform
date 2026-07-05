#!/usr/bin/env python3
"""
graph/graph_validator.py — Phase 2: integrity checks for the unified knowledge graph.

Validates data/veda_unified_graph.json and emits a report:
  • orphan nodes          — nodes with no incident edges
  • duplicate edges       — same (source, target, type) appearing twice
  • invalid references    — edges whose source/target node id does not exist
  • disconnected components — count of weakly-connected components (1 = fully connected)

Pure stdlib (BFS over an adjacency dict). Returns a dict report; CLI prints it.

Usage:
    python3 graph/graph_validator.py
    python3 graph/graph_validator.py --strict      # exit 1 if any integrity error
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from collections import deque, Counter
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    import config as _cfg
    _GRAPH_FILE = os.path.join(_ROOT, getattr(_cfg, "UNIFIED_GRAPH_FILE", "data/veda_unified_graph.json"))
except Exception:
    _GRAPH_FILE = os.path.join(_ROOT, "data", "veda_unified_graph.json")


def validate(graph: Dict[str, Any]) -> Dict[str, Any]:
    nodes: List[dict] = graph.get("nodes", []) or []
    edges: List[dict] = graph.get("edges", []) or []

    node_ids = {n["id"] for n in nodes}
    node_type_counts = Counter(n["type"] for n in nodes)

    # invalid references + duplicate edges
    seen = set()
    duplicate_edges: List[dict] = []
    invalid_references: List[dict] = []
    adj: Dict[str, set] = {nid: set() for nid in node_ids}
    incident = set()

    for e in edges:
        s, t, ty = e.get("source"), e.get("target"), e.get("type")
        if s not in node_ids or t not in node_ids:
            invalid_references.append(e)
            continue
        key = (s, t, ty)
        if key in seen:
            duplicate_edges.append(e)
        else:
            seen.add(key)
        adj[s].add(t)
        adj[t].add(s)          # undirected view for connectivity
        incident.add(s)
        incident.add(t)

    orphan_nodes = sorted(node_ids - incident)

    # weakly-connected components (BFS over undirected adjacency)
    visited = set()
    components = 0
    largest = 0
    for start in node_ids:
        if start in visited:
            continue
        components += 1
        size = 0
        q = deque([start])
        visited.add(start)
        while q:
            cur = q.popleft()
            size += 1
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        largest = max(largest, size)

    report: Dict[str, Any] = {
        # headline counts (matches the requested report shape)
        "tables":     node_type_counts.get("TABLE", 0),
        "columns":    node_type_counts.get("COLUMN", 0),
        "concepts":   node_type_counts.get("CONCEPT", 0),
        "metrics":    node_type_counts.get("METRIC", 0),
        "dimensions": node_type_counts.get("DIMENSION", 0),
        "synonyms":   node_type_counts.get("SYNONYM", 0),
        "nodes":      len(nodes),
        "edges":      len(edges),
        # integrity
        "integrity": {
            "orphan_node_count":      len(orphan_nodes),
            "orphan_nodes_sample":    orphan_nodes[:20],
            "duplicate_edge_count":   len(duplicate_edges),
            "invalid_reference_count": len(invalid_references),
            "invalid_references_sample": invalid_references[:20],
            "connected_components":   components,
            "largest_component_size": largest,
            "fully_connected":        components <= 1,
        },
    }
    report["ok"] = (len(invalid_references) == 0 and len(duplicate_edges) == 0)
    return report


def load_and_validate(path: str = _GRAPH_FILE) -> Dict[str, Any]:
    try:
        with open(path) as f:
            graph = json.load(f)
    except (OSError, ValueError) as e:
        return {"ok": False, "error": f"cannot read graph: {e}"}
    return validate(graph)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the VEDA unified graph.")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any integrity error (invalid refs / dup edges)")
    args = ap.parse_args()

    rep = load_and_validate()
    print(json.dumps({k: v for k, v in rep.items() if k != "integrity"}, indent=2))
    if "integrity" in rep:
        intg = rep["integrity"]
        print("\nintegrity:")
        print(f"  orphan nodes        : {intg['orphan_node_count']}")
        print(f"  duplicate edges     : {intg['duplicate_edge_count']}")
        print(f"  invalid references  : {intg['invalid_reference_count']}")
        print(f"  connected components: {intg['connected_components']} "
              f"(largest {intg['largest_component_size']})")
        if intg["orphan_nodes_sample"]:
            print(f"  orphan sample       : {intg['orphan_nodes_sample'][:8]}")
    if args.strict and not rep.get("ok", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
