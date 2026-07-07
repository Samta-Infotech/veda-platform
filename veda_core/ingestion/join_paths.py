"""L2 ANALYZE · precompiled pairwise join paths (Q-9).

Precomputes the shortest FK join path (key pairs + fan-out direction) between every
pair of tables within a hop limit, at ingestion, from the scanned FK edges. The
query-time ``join_planner`` consults this static map first and only falls back to
live graph traversal for unmapped pairs — deterministic, versioned with the schema.

Pure transform of scan_result → no source-DB touch, non-fatal.
"""
from __future__ import annotations

import json
import os
from collections import deque
from typing import Dict, List

MAX_HOPS = 4


def _index_path() -> str:
    from config import artifact_path
    return artifact_path("veda_join_paths.json")


def _edges_from_scan(scan_result) -> List[dict]:
    edges = []
    for e in getattr(scan_result, "fk_edges", []) or []:
        ft, tt = e.get("from_table"), e.get("to_table")
        if not ft or not tt:
            continue
        edges.append({
            "from_table": ft, "to_table": tt,
            "from_col": e.get("from_col_name"), "to_col": e.get("to_col_name"),
        })
    return edges


def build_join_paths(scan_result, source_id: str = "", verbose: bool = False,
                     max_hops: int = MAX_HOPS) -> Dict[str, dict]:
    """BFS over the undirected FK graph → shortest path per table pair (<= max_hops)."""
    edges = _edges_from_scan(scan_result)

    # adjacency: table -> [(neighbour, from_col, to_col, direction)]
    adj: Dict[str, list] = {}
    for e in edges:
        adj.setdefault(e["from_table"], []).append(
            (e["to_table"], e["from_col"], e["to_col"], "many_to_one"))
        adj.setdefault(e["to_table"], []).append(
            (e["from_table"], e["to_col"], e["from_col"], "one_to_many"))

    tables = list(adj.keys())
    paths: Dict[str, dict] = {}

    for start in tables:
        # BFS shortest path from `start` to every reachable table
        visited = {start}
        q = deque([(start, [])])
        while q:
            node, path = q.popleft()
            if len(path) >= max_hops:
                continue
            for nbr, fcol, tcol, direction in adj.get(node, []):
                if nbr in visited:
                    continue
                visited.add(nbr)
                hop = {"from_table": node, "to_table": nbr,
                       "from_col": fcol, "to_col": tcol, "direction": direction}
                new_path = path + [hop]
                key = f"{start}|{nbr}"
                if key not in paths:
                    paths[key] = {"from": start, "to": nbr,
                                  "hops": len(new_path), "path": new_path}
                q.append((nbr, new_path))

    out = {"pairs": paths, "max_hops": max_hops, "tables": len(tables)}
    path_file = _index_path()
    os.makedirs(os.path.dirname(path_file) or ".", exist_ok=True)
    with open(path_file, "w") as f:
        json.dump(out, f)
    if verbose:
        print(f"  [join_paths] {len(paths)} pairs over {len(tables)} tables → {path_file}")
    return paths


def load_join_paths() -> Dict[str, dict]:
    """Query-tier loader for join_planner: {"<from>|<to>": {...path...}} or {} if absent."""
    path = _index_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f).get("pairs", {})
    except Exception:
        return {}
