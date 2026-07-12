# =============================================================================
# ingestion/value_referents.py
# VEDA — derived artifact: value → typed referent closure (QSR, Phase A).
#
# Post-ingestion, LLM-free transform (DERIVED_ARTIFACTS family, like
# relationship-graph / join-paths / registry): for every sampled value in the
# `column_values` store, emit the REFERENTS that value is evidence for:
#
#   direct : the (table, column) the value was sampled from.
#   closed : every table that REFERENCES the value's table through a declared
#            N:1 foreign key — the lookup/LOV closure. 'captured' is stored as a
#            row of a list-of-values table, but as query evidence it points at
#            accounts_paymenttransaction.payment_status_id (and any other
#            referencing FK whose rows REALLY reach that label). Without this
#            closure, value evidence can never vote for the transaction table
#            the user is actually asking about.
#
# Precision: shared lookup tables (one LOV table serving many FK domains) would
# make a broad closure claim that 'captured' is evidence for mode_of_payment_id,
# floor_type_id, … The builder therefore runs, per FK edge, ONE bounded query on
# the SOURCE database —
#     SELECT DISTINCT <label> FROM <R> JOIN <T> ON R.fk = T.pk
# — the exact label set reachable through that FK — and emits a closed referent
# only when the value is in that set. Purely structural + data-derived; no
# table, column, or value names appear in this module. When the source DB is
# unreachable (no request context), the builder falls back to the broad closure
# with a warning, so the artifact still exists.
#
# SAFETY: closed referents are ANCHOR/ROUTING evidence only. They must never be
# rendered as direct SQL predicates (fk_col = 'captured' would be wrong — the
# label lives one join away); consumers get the distinction explicitly via
# kind=direct|closed and the via_* join fields.
#
# Build (in-container; internal store + source DB):
#   python3 -m ingestion.value_referents --source-id 2 --tenant default
# =============================================================================
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

MAX_REFERENTS_PER_VALUE = 24      # cap fan-out for values reused across many columns
MAX_CLOSURE_EDGES_PER_TABLE = 32  # cap FK fan-in per lookup table
MAX_LABELS_PER_EDGE = 5000        # bound the per-edge reachable-label query


def _artifact() -> str:
    from config import artifact_path
    return artifact_path("veda_value_referents.json")


def _norm(v) -> str:
    """Match the column_values value_norm convention (lowercased word runs)."""
    return " ".join(re.findall(r"[a-z0-9]+", str(v or "").lower()))


def _load_graph_edges() -> List[dict]:
    from query.join_planner import load_graph
    from config import artifact_path
    g = load_graph(artifact_path("veda_relationship_graph.json"))
    return g.get("edges", [])


def _table_types(sm: dict) -> Dict[str, str]:
    return {t: (m or {}).get("table_type", "") for t, m in (sm.get("tables") or {}).items()}


def _q(ident: str) -> str:
    return '"' + str(ident).replace('"', '""') + '"'


def _reachable_labels(source_conn, edge: Tuple[str, str, str], lookup_table: str,
                      label_col: str, cache: dict) -> Optional[Set[str]]:
    """Normalized labels of `lookup_table.label_col` actually referenced through
    edge (R.fk → T.pk). One bounded DISTINCT query per (edge, label_col), cached.
    None → could not determine (treat as reachable: fail open to the broad closure)."""
    if source_conn is None:
        return None
    rt, rc, tpk = edge
    key = (rt, rc, lookup_table, label_col)
    if key in cache:
        return cache[key]
    try:
        cur = source_conn.cursor()
        cur.execute(
            f"SELECT DISTINCT t.{_q(label_col)} FROM {_q(rt)} r "
            f"JOIN {_q(lookup_table)} t ON r.{_q(rc)} = t.{_q(tpk)} "
            f"LIMIT {MAX_LABELS_PER_EDGE}")
        labels = {_norm(r[0]) for r in cur.fetchall() if r[0] is not None}
        cache[key] = labels
        return labels
    except Exception:
        try:
            source_conn.rollback()
        except Exception:
            pass
        cache[key] = None
        return None


def build_value_referents(conn, sm: dict, source_conn=None, verbose: bool = False) -> dict:
    """Compute the artifact dict from the column_values store + relationship graph.

    `conn`        : open connection to the INTERNAL store (column_values).
    `source_conn` : optional connection to the SOURCE database — enables the
                    precise per-edge reachable-label closure."""
    from config import COLUMN_VALUES_TABLE_NAME as TBL

    ttypes = _table_types(sm)

    # FK closure index: lookup_table -> [(referencing_table, fk_column, target_pk)]
    # Structural filter: N:1 edge, target not a TRANSACTION table.
    fan_in: Dict[str, List] = defaultdict(list)
    for e in _load_graph_edges():
        if e.get("cardinality") != "N:1":
            continue
        tgt = e.get("target_table")
        if ttypes.get(tgt, "").upper() == "TRANSACTION":
            continue
        if len(fan_in[tgt]) < MAX_CLOSURE_EDGES_PER_TABLE:
            fan_in[tgt].append((e.get("source_table"), e.get("source_column"),
                                e.get("target_column")))

    cur = conn.cursor()
    cur.execute(f"SELECT value_norm, table_name, col_name, semantic_type, value_raw "
                f"FROM {TBL}")
    rows = cur.fetchall()

    referents: Dict[str, List[dict]] = defaultdict(list)
    seen: Dict[str, set] = defaultdict(set)
    label_cache: dict = {}
    n_direct = n_closed = n_pruned = 0

    def _add(value_norm: str, ref: dict) -> None:
        nonlocal n_direct, n_closed
        # one entry per referent column — a value reachable via two label columns
        # of the same lookup (code + list_label) is still ONE piece of evidence
        key = (ref["kind"], ref["table"], ref["column"], ref.get("via_table"))
        if key in seen[value_norm] or len(referents[value_norm]) >= MAX_REFERENTS_PER_VALUE:
            return
        seen[value_norm].add(key)
        referents[value_norm].append(ref)
        if ref["kind"] == "direct":
            n_direct += 1
        else:
            n_closed += 1

    for value_norm, t, c, st, raw in rows:
        if not value_norm:
            continue
        _add(value_norm, {"kind": "direct", "table": t, "column": c,
                          "type": st, "value_raw": raw})
        for edge in fan_in.get(t, ()):
            labels = _reachable_labels(source_conn, edge, t, c, label_cache)
            if labels is not None and value_norm not in labels:
                n_pruned += 1
                continue
            rt, rc, tpk = edge
            _add(value_norm, {"kind": "closed", "table": rt, "column": rc,
                              "via_table": t, "via_column": c, "via_pk": tpk,
                              "type": st, "value_raw": raw})

    # Per-FK-edge reachable-label domains (already computed for pruning): the exact
    # value vocabulary of `R.fk_col` seen THROUGH the lookup — the payload of a
    # grounded clarify ("payment statuses here are captured / authorized / …").
    # Keyed "R.fk_col|via_column" — kept SEPARATE per label column, because shared
    # lookups also carry meta columns (a table_name/column_name column of an LOV)
    # whose "labels" would pollute a merged domain. Consumers scope by the
    # via_column their resolved values actually came through.
    edge_domains: Dict[str, List[str]] = {}
    for (rt, rc, _lt, lc), labels in label_cache.items():
        if labels:
            edge_domains[f"{rt}.{rc}|{lc}"] = sorted(labels)[:50]

    art = {"version": 3, "precise": source_conn is not None,
           "n_values": len(referents), "n_direct": n_direct, "n_closed": n_closed,
           "n_pruned": n_pruned, "referents": dict(referents),
           "edge_domains": edge_domains}
    if verbose:
        mode = "precise" if source_conn is not None else "broad"
        print(f"[value_referents] {mode}: {len(referents)} values, {n_direct} direct, "
              f"{n_closed} closed, {n_pruned} pruned (unreachable label)")
    return art


def write_value_referents(conn, sm: dict, source_conn=None, verbose: bool = True) -> str:
    art = build_value_referents(conn, sm, source_conn=source_conn, verbose=verbose)
    path = _artifact()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(art, f)
    if verbose:
        print(f"[value_referents] wrote {path}")
    return path


_CACHE = {"path": None, "art": None}


def load_value_referents() -> dict:
    """Runtime loader (cached per process per artifact path). Missing artifact →
    empty referent map; consumers fall back to the live value store."""
    path = _artifact()
    if _CACHE["path"] == path and _CACHE["art"] is not None:
        return _CACHE["art"]
    art = {"version": 0, "referents": {}}
    try:
        if os.path.exists(path):
            art = json.load(open(path))
    except Exception:
        pass
    _CACHE["path"], _CACHE["art"] = path, art
    return art


def main() -> None:
    import argparse
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id", type=int, default=None)
    ap.add_argument("--tenant", default="default")
    args = ap.parse_args()

    source_conn = None
    if args.source_id is not None:
        try:
            from veda_core import context
            from veda_core.context import RequestContext
            context.set_context(RequestContext(source_id=args.source_id, tenant=args.tenant))
            from veda.runtime import _pg
            source_conn = _pg()
        except Exception as e:
            print(f"[value_referents] source DB unavailable ({e}) — broad closure")

    from veda.runtime import _internal_db_config, _load_scoped_sm
    import psycopg2
    db = _internal_db_config()
    conn = psycopg2.connect(host=db["host"], port=db["port"], dbname=db["database"],
                            user=db["user"], password=db["password"])
    try:
        write_value_referents(conn, _load_scoped_sm(), source_conn=source_conn, verbose=True)
    finally:
        conn.close()
        if source_conn is not None:
            source_conn.close()


if __name__ == "__main__":
    main()
