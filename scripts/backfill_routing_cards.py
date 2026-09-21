#!/usr/bin/env python
# =============================================================================
# scripts/backfill_routing_cards.py
# VEDA — build the per-source ROUTING CARD for already-ingested sources
# (Checkpoint B.1; see veda_core/ingestion/routing_card.py for what a card is)
#
# WHY THIS EXISTS
#   The card is produced by the `routing_card` stage in L5 (ingestion/layers/
#   l5_publish.py), so any source ingested from now on gets one for free. Every
#   source already in this deployment was ingested before that stage existed, and
#   re-running a full ingestion just to emit a derived artifact would re-do the
#   expensive LLM stages (L3 alone is hours for source 2's 178 tables) to produce
#   byte-identical inputs. The card is a PURE transform of artifacts that already
#   exist on disk and in the engine store, so it can simply be rebuilt.
#
# WHAT IT READS (nothing is recomputed)
#   - veda_semantic_model.json  (per-source artifact)      → entities, roles, naming
#   - column_values             (engine store)             → real sample values
#   - graph_edges/cross_source_fk (engine store)           → joins_to
#   - row counts                (the source itself, --with-row-counts; optional)
#
#   Row counts are the ONE field a full ingestion has that this does not: they live
#   on the in-memory scan result, not in any artifact or engine table. By default
#   they are omitted (the card's other fields carry the routing signal); pass
#   --with-row-counts to open a read-only connection to each source and fetch them.
#
# USAGE (inside the ingest-worker or inference container, engine config on the path):
#   cd /app/veda_core && python /app/scripts/backfill_routing_cards.py
#   Flags: --sources 2,4,5   --tenant default   --with-row-counts   --print
#
# Idempotent: rewrites each card atomically. Exit 1 if no card could be built.
# =============================================================================
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

# Ensure the ENGINE config wins over the Django `config/` package regardless of cwd —
# `import config` must resolve to veda_core/config.py, not /app/config/__init__.py.
# Order matters: /app goes on FIRST so the engine dir ends up ahead of it at sys.path[0]
# (same fix as scripts/backfill_semantic_bridge.py).
sys.path.insert(0, "/app")
_ENGINE_DIR = "/app/veda_core"
if os.path.isdir(_ENGINE_DIR):
    if _ENGINE_DIR in sys.path:
        sys.path.remove(_ENGINE_DIR)
    sys.path.insert(0, _ENGINE_DIR)


def _discover_sources(tenant: str) -> list:
    """Every source that has been ingested, from the ENGINE STORE — not from which
    artifacts happen to be on disk.

    Discovering by `veda_semantic_model.json` finds only relational sources: a datalake
    source builds its lite model at query time and never writes one, and a document
    source has no tables at all. That would have given a card to source 2 alone and left
    the router comparing a rich card against bare column matches for everything else.
    Avoids needing Django (same rationale as backfill_semantic_bridge.py)."""
    try:
        from ingestion.db_abstraction import internal_connection
        with internal_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT source_id FROM column_embeddings_v2")
                ids = {str(r[0]) for r in (cur.fetchall() or []) if r[0]}
                cur.execute("SELECT DISTINCT source_id FROM doc_chunks")
                ids |= {str(r[0]) for r in (cur.fetchall() or []) if r[0]}
        return sorted(ids, key=lambda x: (len(x), x))
    except Exception as exc:
        print(f"store discovery failed ({type(exc).__name__}: {exc})")
        return []


def _kind(source_id: str) -> str:
    """Structural kind of an already-ingested source, from what it actually has in the
    engine store. A real ingestion gets this from SourceContext.type; a backfill has no
    context, and the semantic model carries no such field — so it is read off the
    stores: chunks ⇒ document, columns ⇒ tabular. Empty string when neither (the card
    then simply omits the claim rather than asserting a wrong one)."""
    try:
        from ingestion.db_abstraction import internal_connection
        with internal_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM doc_chunks WHERE source_id = %s", (str(source_id),))
                docs = int((cur.fetchone() or [0])[0])
                cur.execute("SELECT count(*) FROM column_embeddings_v2 WHERE source_id = %s", (str(source_id),))
                cols = int((cur.fetchone() or [0])[0])
        if docs and not cols:
            return "document"
        # A tabular source could be relational OR datalake and the engine store does not
        # distinguish them (both populate column_embeddings_v2). Rather than guess — a
        # datalake source labelled "relational" on its own card is a false statement the
        # router would read as fact — return nothing and let render_card take the
        # authoritative `source_type` from the live registry profile, which it already
        # prefers over the card's stored kind.
        return ""
    except Exception:
        return ""


def _row_counts(source_id: str) -> dict:
    """Read-only row counts straight from the source. Only used with
    --with-row-counts; any failure degrades to {} and the card is still written."""
    try:
        from config import get_source
        from connectors import build_connector
        conn = build_connector(get_source(source_id))
        counts = {}
        for t in (conn.list_tables() or []):
            try:
                counts[t] = int(conn.get_row_count(t))
            except Exception:
                continue
        return counts
    except Exception as exc:
        print(f"    row counts unavailable ({type(exc).__name__}: {exc})")
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="", help="comma-separated source ids (default: all with a semantic model)")
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--with-row-counts", action="store_true",
                    help="open a read-only connection per source to fetch row counts")
    ap.add_argument("--print", dest="show", action="store_true", help="print each card")
    args = ap.parse_args()

    from config import source_artifact_path
    from ingestion.routing_card import write_routing_card

    sources = ([s.strip() for s in args.sources.split(",") if s.strip()]
               or _discover_sources(args.tenant))
    if not sources:
        print(f"no sources with a semantic model under tenant {args.tenant!r}")
        return 1

    built = 0
    for sid in sources:
        sm_path = source_artifact_path("veda_semantic_model.json", sid, args.tenant)
        # No L3 model → write_routing_card falls back to the engine store (datalake /
        # document sources). state stays empty and the builder does the right thing.
        sm = {}
        if os.path.exists(sm_path):
            with open(sm_path) as f:
                sm = json.load(f)

        counts = _row_counts(sid) if (args.with_row_counts and sm) else {}
        # The card builder reads row counts off `state["scan_result"].tables[]`; a
        # backfill has no scan result, so present the same shape from whatever counts
        # we have (empty is fine — row_count is then simply absent from the card).
        scan = SimpleNamespace(tables=[SimpleNamespace(table_name=t, row_count=c)
                                       for t, c in counts.items()])
        ctx = SimpleNamespace(source_id=sid, tenant=args.tenant, type=_kind(sid))
        state = {"semantic_model": sm, "scan_result": scan}

        path = write_routing_card(ctx, state, verbose=False)
        with open(path) as f:
            card = json.load(f)
        print(f"  [{sid}] kind={card.get('kind') or '?'} "
              f"{len(card['entities'])}/{card['entity_count_total']} entities, "
              f"{len(card['joins_to'])} linked source(s), "
              f"{len(card['example_questions'])} example questions"
              f"{' [from store]' if card.get('built_from') else ''} -> {path}")
        if args.show:
            print(json.dumps(card, indent=2)[:4000])
        built += 1

    print(f"\nbuilt {built} routing card(s)")
    return 0 if built else 1


if __name__ == "__main__":
    sys.exit(main())
