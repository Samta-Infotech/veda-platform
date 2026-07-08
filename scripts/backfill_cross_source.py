#!/usr/bin/env python3
"""Backfill cross-source sketches + links for ALREADY-INGESTED sources.

Cross-source plan (docs/CROSSSOURCE_GRAPH.md): normally the MinHash sketch pass and
the cross_source_fk discovery run as ingestion stages, so a FRESH ingestion links
sources automatically. This script covers the other case — a source that was
ingested BEFORE those stages existed (e.g. the large Postgres source) and that you
do NOT want to re-ingest. It:

  1. Reads the source's column nodes from graph_nodes (no source re-scan).
  2. Samples DISTINCT values per join-key-shaped column — crucially INCLUDING the
     PK/FK/id columns the value sampler deliberately skips (those are the join keys
     cross-source discovery needs) — straight from the source DB, read-only.
  3. Computes 128-perm MinHash sketches and persists them to column_sketches.
  4. Runs ingestion.cross_source_graph over the whole tenant to emit cross_source_fk
     edges connecting this source to the others.

Idempotent: sketches upsert per column; discovery deletes + re-emits the tenant's
cross_source_fk edges. Re-run any time you add a source.

Usage:
    python scripts/backfill_cross_source.py --source-ids 1,2 --tenant default
    python scripts/backfill_cross_source.py --all --tenant default        # every source in graph_nodes
    python scripts/backfill_cross_source.py --source-ids 1 --sketch-only  # sketch, don't link yet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
# veda_core LAST so it wins position 0: this script speaks the ENGINE `config`
# (veda_core/config.py), not the Django settings package (repo-root config/).
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "veda_core"))

from config import (  # noqa: E402
    CROSS_SOURCE_SKETCH_SAMPLE_SIZE, SENSITIVE_PATTERNS, GRAPH_NODES_TABLE,
)
from ingestion import column_sketches as CS  # noqa: E402
from ingestion import cross_source_graph as XS  # noqa: E402
from ingestion.db_abstraction import (  # noqa: E402
    get_internal_connection, release_internal_connection, get_client_connection,
)


def _q(name: str) -> str:
    return '"' + str(name).replace('"', "") + '"'


def read_columns(source_id: str) -> list:
    """Column nodes for a source from graph_nodes (internal store):
    [{col_id, table_name, col_name, semantic_type, data_type, is_pk, is_fk}]."""
    conn = get_internal_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT ref_id, table_name, name, semantic_type, data_type, is_pk, is_fk "
                f"FROM {GRAPH_NODES_TABLE} WHERE node_type = 'column' AND source_id = %s",
                [str(source_id)])
            rows = cur.fetchall()
    finally:
        release_internal_connection(conn)
    return [{"col_id": r[0], "table_name": r[1], "col_name": r[2],
             "semantic_type": r[3] or "", "data_type": r[4] or "",
             "is_pk": bool(r[5]), "is_fk": bool(r[6])} for r in rows]


def _is_join_key_shaped(col: dict) -> bool:
    """Sketch a column if it is a join key (PK/FK/id) OR a sketchable value type,
    excluding anything whose name looks sensitive (PII never gets a value sketch)."""
    name = (col["col_name"] or "").lower()
    if any(p in name for p in SENSITIVE_PATTERNS):
        return False
    if col["is_pk"] or col["is_fk"]:
        return True
    return (col["semantic_type"] or "").upper() in CS.SKETCHABLE_TYPES


def _value_class(col: dict) -> str:
    if col["is_pk"] or col["is_fk"]:
        return "id"          # join keys compare as ids regardless of storage type
    return CS.value_class(col["semantic_type"], col["data_type"])


def sample_distinct(cur, table_name: str, col_name: str, n: int) -> list:
    """Read up to n DISTINCT non-null values (read-only). Returns [] on any error
    so one bad column never aborts the backfill."""
    try:
        cur.execute(
            f"SELECT DISTINCT {_q(col_name)} FROM {_q(table_name)} "
            f"WHERE {_q(col_name)} IS NOT NULL LIMIT %s", (n,))
        return [r[0] for r in cur.fetchall()]
    except Exception:
        try:
            cur.connection.rollback()
        except Exception:
            pass
        return []


def backfill_source(source_id: str, tenant: str, sample_size: int, verbose: bool) -> int:
    cols = [c for c in read_columns(source_id) if _is_join_key_shaped(c)]
    if not cols:
        print(f"  source {source_id}: no join-key-shaped columns found in graph_nodes")
        return 0
    conn = get_client_connection(source_id)
    rows = []
    try:
        with conn.cursor() as cur:
            for c in cols:
                vals = sample_distinct(cur, c["table_name"], c["col_name"], sample_size)
                sketch, n = CS.compute_sketch(vals)
                if sketch is None:
                    continue
                rows.append({"col_id": c["col_id"], "table_name": c["table_name"],
                             "col_name": c["col_name"], "n_distinct": n,
                             "value_class": _value_class(c), "sketch": sketch})
                if verbose:
                    print(f"    {c['table_name']}.{c['col_name']} "
                          f"[{_value_class(c)}] n_distinct(sampled)={n}")
    finally:
        try:
            conn.close()
        except Exception:
            pass
    written = CS.persist_sketches(rows, source_id=source_id, tenant=tenant)
    print(f"  source {source_id}: sketched {written}/{len(cols)} join-key columns")
    return written


def all_source_ids() -> list:
    conn = get_internal_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT source_id FROM {GRAPH_NODES_TABLE} "
                        f"WHERE node_type = 'column'")
            return sorted(str(r[0]) for r in cur.fetchall())
    finally:
        release_internal_connection(conn)


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill cross-source sketches + links")
    ap.add_argument("--source-ids", default="", help="comma-separated source ids")
    ap.add_argument("--all", action="store_true", help="every source in graph_nodes")
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--sample-size", type=int, default=CROSS_SOURCE_SKETCH_SAMPLE_SIZE)
    ap.add_argument("--sketch-only", action="store_true",
                    help="compute+persist sketches but skip cross_source_fk discovery")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not CS.sketches_available():
        print("ERROR: datasketch not installed — cannot compute MinHash sketches.")
        return 2

    if args.all:
        source_ids = all_source_ids()
    else:
        source_ids = [s.strip() for s in args.source_ids.split(",") if s.strip()]
    if not source_ids:
        print("Nothing to do: pass --source-ids or --all")
        return 1

    print(f"Backfilling sketches for sources {source_ids} (tenant={args.tenant}, "
          f"sample_size={args.sample_size})")
    total = 0
    for sid in source_ids:
        total += backfill_source(sid, args.tenant, args.sample_size, args.verbose)
    print(f"Sketched {total} columns across {len(source_ids)} source(s).")

    if args.sketch_only:
        print("--sketch-only: skipping cross_source_fk discovery.")
        return 0

    print("Running cross-source join discovery...")
    stats = XS.discover_and_persist(args.tenant, source_ids=None, verbose=args.verbose)
    print(f"Cross-source discovery: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
