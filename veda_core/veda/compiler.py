"""Task 10 — Graph SQL Compiler: join inference (ARCHITECTURE_HYBRID.md §2 L6, directive).

The compiler is the SOLE author of joins and SQL. Its core invariant is COMPILE-or-REFUSE,
never GUESS. This module implements the join-inference half — the part that must REFUSE on
ambiguity (the diamond + the abhijit direct-vs-junction gaps that `plan_join_tree` guesses on).

Additive + safe: this does NOT modify the live `query.join_planner.plan_join_tree`. It is the
join engine of the new IR→SQL compiler path, gated behind COMPILER_STRICT_JOINS. The full
IR→dialect-SQL emission reuses the existing clause assembly; this module owns the decision of
WHICH joins (or refusal), which is where correctness lives.

Pure graph logic. No DB, no Ollama.

Return shape mirrors plan_join_tree so existing consumers/tests read it unchanged:
    {join_path, confidence, unreachable, ambiguous, refused, reason}
"""
from typing import Any, Dict, List, Optional, Tuple


def _singular(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _query_tokens(query: str) -> set:
    import re
    raw = {w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(w) > 2}
    return raw | {_singular(w) for w in raw}


def _edge_tokens(e: dict) -> set:
    toks = set()
    for col in (e.get("source_column", ""), e.get("target_column", "")):
        for p in str(col).lower().split("_"):
            if len(p) > 2:
                toks.add(p)
                toks.add(_singular(p))
    return toks


def _table_tokens(table: str) -> set:
    toks = set()
    for p in str(table).lower().split("_"):
        if len(p) > 2:
            toks.add(p)
            toks.add(_singular(p))
    return toks


def _ekey(e: dict) -> Tuple:
    return (e["source_table"], e["source_column"], e["target_table"], e["target_column"])


def _adjacency(graph: dict) -> Dict[str, List[Tuple[str, dict]]]:
    adj: Dict[str, List[Tuple[str, dict]]] = {}
    for e in graph.get("edges", []):
        adj.setdefault(e["source_table"], []).append((e["target_table"], e))
        adj.setdefault(e["target_table"], []).append((e["source_table"], e))
    return adj


def _all_simple_paths(adj, src, dst, allowed, max_hops=4) -> List[List[dict]]:
    """Every DISTINCT simple path src→dst whose intermediate nodes are all `allowed`."""
    out: List[List[dict]] = []
    seen_sigs = set()

    def dfs(node, visited, path):
        if len(path) > max_hops:
            return
        if node == dst and path:
            sig = tuple(_ekey(e) for e in path)
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                out.append(list(path))
            return
        for nbr, e in adj.get(node, []):
            if nbr in visited:
                continue
            # intermediate (non-terminal) nodes must be allowed
            if nbr != dst and nbr != src and allowed is not None and nbr not in allowed:
                continue
            dfs(nbr, visited | {nbr}, path + [e])

    dfs(src, {src}, [])
    return out


def infer_join_path(anchor: str, targets: List[str], graph: dict, query: str = "",
                    allowed_intermediates: Optional[List[str]] = None,
                    max_hops: int = 4) -> Dict[str, Any]:
    """COMPILE-or-REFUSE join inference. Refuses (ambiguous) when >1 distinct path exists
    between anchor and a target and the query does not uniquely disambiguate one of them."""
    adj = _adjacency(graph)
    qtoks = _query_tokens(query)
    terminals = [t for t in dict.fromkeys(targets) if t]
    allowed = (set(allowed_intermediates) if allowed_intermediates else set()) | {anchor} | set(terminals)

    join_path: List[dict] = []
    seen_keys = set()
    ambiguous: List[dict] = []
    unreachable: List[str] = []

    for tgt in terminals:
        if tgt == anchor:
            # self-join: the anchor's self-edges
            self_edges = [e for e in graph.get("edges", [])
                          if e["source_table"] == anchor and e["target_table"] == anchor]
            paths = [[e] for e in self_edges]
        else:
            paths = _all_simple_paths(adj, anchor, tgt, allowed, max_hops=max_hops)

        if not paths:
            unreachable.append(tgt)
            continue
        if len(paths) == 1:
            chosen = paths[0]
        else:
            # disambiguate by query tokens, but EXCLUDE the anchor/target entity names:
            # those trivially appear in junction FK columns (role_id, user_id) and would
            # falsely "disambiguate" — only a relationship qualifier should select a path.
            disambig = qtoks - _table_tokens(anchor) - _table_tokens(tgt)
            named = [p for p in paths if any(disambig & _edge_tokens(e) for e in p)]
            if len(named) == 1:
                chosen = named[0]
            else:
                # 0 named (no disambiguator) or >1 named (still ambiguous) → REFUSE
                ambiguous.append({
                    "target": tgt,
                    "n_paths": len(paths),
                    "options": [[_ekey(e) for e in p] for p in paths[:4]],
                })
                continue
        for e in chosen:
            if _ekey(e) not in seen_keys:
                seen_keys.add(_ekey(e))
                join_path.append(e)

    refused = bool(ambiguous or unreachable)
    reason = ""
    if unreachable:
        reason = f"not reachable from {anchor}: {sorted(set(unreachable))}"
    elif ambiguous:
        t = ambiguous[0]
        reason = (f"{t['n_paths']} distinct relationship paths {anchor}→{t['target']}; "
                  f"query does not specify which — refusing rather than guessing")

    conf = 1.0
    for e in join_path:
        conf *= e.get("confidence", 1.0)

    return {
        "anchor": anchor,
        "join_path": join_path if not refused else [],
        "confidence": round(conf, 3),
        "unreachable": sorted(set(unreachable)),
        "ambiguous": ambiguous,
        "refused": refused,
        "reason": reason,
    }
