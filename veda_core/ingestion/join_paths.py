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


def _index_path(source_id=None, tenant: str = "default") -> str:
    """Per-(tenant, source) when source_id is given (P0-5, 2026-09-10) — unconditionally,
    via config.source_artifact_path(), not gated behind VEDA_ARTIFACT_SCOPE (P0-7)."""
    if source_id is not None:
        from config import source_artifact_path
        return source_artifact_path("veda_join_paths.json", source_id, tenant)
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


def build_join_paths(scan_result, source_id: str = "", tenant: str = "default",
                     verbose: bool = False, max_hops: int = MAX_HOPS) -> Dict[str, dict]:
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
    path_file = _index_path(source_id or None, tenant)
    os.makedirs(os.path.dirname(path_file) or ".", exist_ok=True)
    with open(path_file, "w") as f:
        json.dump(out, f)
    if verbose:
        print(f"  [join_paths] {len(paths)} pairs over {len(tables)} tables → {path_file}")

    # P0-6: drop this source's cached map so the next planner read sees the fresh one.
    try:
        invalidate_join_paths_cache(source_id or None)
    except Exception:
        pass

    return paths


# (tenant, str(source_id) | "") -> {"<from>|<to>": {...}} — per-source (P0-5, 2026-09-10).
_JOIN_PATHS_CACHE: dict = {}


def load_join_paths(source_id=None, tenant=None) -> Dict[str, dict]:
    """Query-tier loader for join_planner: {"<from>|<to>": {...path...}} or {} if absent.
    Resolves source_id/tenant from the ambient request context when not given."""
    if source_id is None:
        from veda_core import context
        ctx = context.try_current()
        if ctx is not None:
            source_id = ctx.source_id
            tenant = ctx.tenant or tenant
    key = (tenant or "default", str(source_id) if source_id is not None else "")
    if key in _JOIN_PATHS_CACHE:
        return _JOIN_PATHS_CACHE[key]
    path = _index_path(source_id, tenant or "default")
    if not os.path.exists(path):
        _JOIN_PATHS_CACHE[key] = {}
        return {}
    try:
        with open(path) as f:
            data = json.load(f).get("pairs", {})
        _JOIN_PATHS_CACHE[key] = data
        return data
    except Exception:
        return {}


def invalidate_join_paths_cache(source_id=None):
    """Drop the cached join-paths map for one source, or every source when source_id
    is None. Call after build_join_paths() rewrites the artifact and from both
    rehydrate paths (P0-6)."""
    if source_id is None:
        _JOIN_PATHS_CACHE.clear()
        return
    sid = str(source_id)
    for key in [k for k in _JOIN_PATHS_CACHE if k[1] == sid]:
        del _JOIN_PATHS_CACHE[key]
