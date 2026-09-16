#!/usr/bin/env python3
"""
ingestion/unified_graph_builder.py — Phase 1 of the Unified Knowledge Graph.

Fuses VEDA's separate graph-like artifacts (each resolved PER SOURCE via
config.resolve_source_artifact) into ONE node/edge graph:
    veda_semantic_model.json      → TABLE, COLUMN nodes + HAS_COLUMN
    veda_relationship_graph.json  → FK_TO (table↔table) + REFERENCES (col↔col)
    veda_concept_graph.json       → CONCEPT nodes + IS_CONCEPT
    veda_domain_synonyms.json     → SYNONYM nodes + SYNONYM_OF
    metrics.json                  → METRIC nodes + IS_METRIC
    dimensions.json               → DIMENSION nodes + IS_DIMENSION
    (column aliases in semantic model)  → ALIAS_OF

This does NOT replace any existing artifact — it is a derived, additive view that the
existing builders keep feeding. Output: the per-source veda_unified_graph.json artifact.

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


# Legacy FLAT input/output paths — used ONLY by the ctx-less dev-CLI path
# (`source_id=None`). Every real build resolves per source via
# config.resolve_source_artifact() (see _resolve_input_paths). M1 close-out
# (2026-09-15): no literal paths here; config.artifact_path() is the single
# definition of the flat location.
import config as _cfg
_SEMANTIC_MODEL = _p(_cfg.SEMANTIC_MODEL_FILE)
_REL_GRAPH      = _p(_cfg.RELATIONSHIP_GRAPH_FILE)
_CONCEPT_GRAPH  = _p(_cfg.CONCEPT_GRAPH_FILE)
_DOMAIN_SYN     = _p(_cfg.DOMAIN_SYNONYMS_FILE)
_OUT_FILE       = _p(_cfg.UNIFIED_GRAPH_FILE)
_METRICS        = _p(_cfg.METRICS_FILE)
_DIMENSIONS     = _p(_cfg.DIMENSIONS_FILE)

GRAPH_VERSION = "1.0"


# ─────────────────────────────────────────────────────────────────────────────
def _load(path: Optional[str]) -> Optional[Any]:
    if not path:                      # resolver returned None (no scope) → absent input
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ── freshness fingerprint ────────────────────────────────────────────────────
# GRAPH_VERSION is a static schema-format version, so it can never answer "does
# this graph reflect the CURRENT artifacts?". Every build therefore stamps the
# (mtime, size) of each input it fused, letting a consumer detect that an input
# has since been regenerated — previously a stale graph was served silently and
# indefinitely (its siblings under DERIVED_ARTIFACTS_ENABLED promise freshness;
# this one had no such contract).
def _resolve_input_paths(source_id=None, tenant: str = "default") -> Dict[str, str]:
    """The 6 input artifact paths this builder fuses.

    P0-5 (2026-09-10/11): when `source_id` is given, prefer the per-source path for
    each input via `config.resolve_source_artifact()` — but ONLY when that
    per-source file actually exists on disk (that helper's own contract). A
    source with no per-source copy of a given input yet keeps reading the shared
    flat file for THAT input, unaffected by the others — a graceful, non-breaking
    mix as each of the 6 gets migrated on its own schedule. `source_id=None` (the
    legacy dev-CLI / ctx-less call) is unchanged — always the flat paths."""
    flat = {
        "semantic_model":     _SEMANTIC_MODEL,
        "relationship_graph": _REL_GRAPH,
        "concept_graph":      _CONCEPT_GRAPH,
        "domain_synonyms":    _DOMAIN_SYN,
        "metrics":            _METRICS,
        "dimensions":         _DIMENSIONS,
    }
    if source_id is None:
        return flat
    try:
        from config import resolve_source_artifact
    except Exception:
        return flat
    names = {
        "semantic_model":     "veda_semantic_model.json",
        "relationship_graph": "veda_relationship_graph.json",
        "concept_graph":      "veda_concept_graph.json",
        "domain_synonyms":    "veda_domain_synonyms.json",
        "metrics":            "metrics.json",
        "dimensions":         "dimensions.json",
    }
    return {key: resolve_source_artifact(names[key], source_id, tenant, flat_default=flat_path)
            for key, flat_path in flat.items()}


def _input_paths(source_id=None, tenant: str = "default") -> Dict[str, str]:
    return _resolve_input_paths(source_id, tenant)


def _fingerprint(source_id=None, tenant: str = "default") -> Dict[str, Any]:
    """{name: {mtime, size}} for each input artifact. Missing inputs record None so
    an input APPEARING later also registers as a change."""
    fp: Dict[str, Any] = {}
    for name, path in _input_paths(source_id, tenant).items():
        try:
            st = os.stat(path)
            fp[name] = {"mtime": round(st.st_mtime, 3), "size": st.st_size}
        except (OSError, TypeError):      # missing file, or None (no scope)
            fp[name] = None
    return fp


def stale_inputs(graph: Dict[str, Any], source_id=None, tenant: str = "default") -> List[str]:
    """Names of input artifacts that changed since `graph` was built.
    [] means fresh (also [] for a graph with no recorded fingerprint — an older
    artifact predating this field, which we cannot judge and must not cry wolf on).
    `source_id`/`tenant` (P0-5, 2026-09-10) must match what built `graph`, or this
    compares against the wrong resolved paths — query_graph.py's caller passes the
    same source it resolved the graph itself from."""
    recorded = (graph or {}).get("inputs")
    if not isinstance(recorded, dict) or not recorded:
        return []
    current = _fingerprint(source_id, tenant)
    return sorted(k for k in recorded if current.get(k) != recorded.get(k))


# Node id helpers — single source of truth, reused by query_graph.py.
def table_id(t: str) -> str:  return f"table:{t}"
def col_id(t: str, c: str) -> str:  return f"col:{t}.{c}"
def concept_id(name: str) -> str:  return f"concept:{name}"
def metric_id(mid: str) -> str:  return f"metric:{mid}"
def dim_id(did: str) -> str:  return f"dim:{did}"
def syn_id(term: str) -> str:  return f"syn:{term.strip().lower()}"


def is_prose_label(term: str) -> bool:
    """True for a label that is a sentence/description rather than a search term.

    The registry generators phrase every metric/dimension as "{agg} {label}". When the
    label is a column's business_definition rather than its name, the result is prose —
    "avg a sequential number assigned to each worklist quote." — which no user ever
    types, so it only adds SYNONYM nodes that dilute term resolution.

    Detection is a trailing period ONLY, deliberately. Measured against the live
    registries: that flags 1781 of 1849 prose labels with ZERO false positives, while
    a word-count or article-based rule would wrongly drop legitimate long phrasings
    ("count of role module permission mapping", "is approved by owner", "has full
    access"). The longest non-prose label is 50 chars and no truncated description
    reaches the graph without its period, so length adds no coverage here.
    """
    return (term or "").strip().endswith(".")


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


def build_unified_graph(source_id=None, tenant: str = "default") -> Dict[str, Any]:
    """Build the unified graph from the on-disk artifacts. Always returns a valid graph
    dict (possibly small) — missing artifacts are skipped, never fatal.

    `source_id`/`tenant` (P0-5, 2026-09-10): resolves each of the 6 input paths via
    `_resolve_input_paths()` — per-source where that artifact is already migrated
    (the relationship graph, post-P0-1), the legacy flat path otherwise. These local
    names deliberately SHADOW the module-level `_SEMANTIC_MODEL`/`_REL_GRAPH`/etc.
    constants for the rest of this function, so the body below (unchanged) reads
    the resolved paths without needing every one of its ~10 `_load(...)` call sites
    edited individually."""
    _paths = _resolve_input_paths(source_id, tenant)
    _SEMANTIC_MODEL = _paths["semantic_model"]
    _REL_GRAPH = _paths["relationship_graph"]
    _CONCEPT_GRAPH = _paths["concept_graph"]
    _DOMAIN_SYN = _paths["domain_synonyms"]
    _METRICS = _paths["metrics"]
    _DIMENSIONS = _paths["dimensions"]

    g = _GraphAccumulator()
    warnings: List[str] = []
    prose_skipped = 0        # description-shaped labels rejected as SYNONYM terms

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
                if is_prose_label(lbl):
                    prose_skipped += 1
                    continue
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
                if is_prose_label(lbl):
                    prose_skipped += 1
                    continue
                sid = g.node(syn_id(lbl), "SYNONYM", lbl.strip().lower())
                g.edge(sid, nid, "SYNONYM_OF")

    # ── SYNONYM_OF (domain synonyms term → column) ───────────────────────────
    ds = _load(_DOMAIN_SYN)
    if ds:
        for term, targets in sorted(ds.items()):
            if is_prose_label(term):
                prose_skipped += 1
                continue
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
    graph["stats"]["prose_labels_skipped"] = prose_skipped
    # Stamp the inputs this build fused, so a consumer can detect staleness later.
    # Captured AFTER reading them: any input rewritten mid-build shows as changed
    # on the next check rather than being wrongly certified fresh.
    graph["inputs"] = _fingerprint(source_id, tenant)
    return graph


def write_unified_graph(graph: Optional[Dict[str, Any]] = None, source_id=None,
                        tenant: str = "default") -> str:
    """`source_id`/`tenant` (P0-5, 2026-09-10): write to the per-source output path
    unconditionally via config.source_artifact_path() — unlike the INPUTS (see
    build_unified_graph()), the output is entirely under this writer's control, so
    there's no "does it exist yet" ambiguity. `source_id=None` keeps the legacy flat
    `_OUT_FILE` path (dev-CLI / ctx-less callers)."""
    if graph is None:
        graph = build_unified_graph(source_id, tenant)
    if source_id is not None:
        from config import source_artifact_path
        out_path = source_artifact_path("veda_unified_graph.json", source_id, tenant)
    else:
        out_path = _OUT_FILE
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(graph, f, indent=2)

    # P0-6: drop this source's cached unified graph so the next read sees the fresh
    # one (get_graph() also self-invalidates via mtime/size, so this is belt-and-braces).
    try:
        from graph.query_graph import invalidate_unified_graph_cache
        invalidate_unified_graph_cache(source_id)
    except Exception:
        pass

    return out_path


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
