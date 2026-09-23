#!/usr/bin/env python
# =============================================================================
# scripts/embed_source_items.py
# VEDA — build the per-item routing prior (source_item_embeddings) WITHOUT a
#        remote embedding server, using the engine's own in-process BGE-M3.
#
# WHY THIS EXISTS
#   apps/sources/item_profiler.py embeds each SourceItem by POSTing to METAL_EMBED_URL.
#   That host only exists on one network, so on any deployment without it the routing
#   prior simply never gets built — and the failure is invisible, because profile_items()
#   swallows the exception as a logged warning after having already saved the summary.
#   The engine encoder (ingestion/m3_encoder.py) produces the SAME vectors in-process on
#   CPU, so the prior can always be built; it is only slower.
#
#   It also creates source_item_embeddings if absent. Nothing in the tree did: every other
#   engine table self-creates in its writer, this one never did, so the INSERT always
#   raised "relation does not exist" (fixed in item_profiler.py too, 2026-09-23).
#
# WHAT IT DOES NOT DO
#   It does not call the SLM. Items must already carry the `summary` that
#   `manage.py build_source_items` writes; this embeds "<name>. <summary>" exactly as
#   item_profiler does, so the vectors are interchangeable with the remote path's.
#
# USAGE (inside ingest-worker or inference — needs the engine on the path):
#   docker compose exec -T -w /app/veda_core ingest-worker python /app/scripts/embed_source_items.py
#   Flags: --sources 2,3   --batch 16   --force   --dry-run
#
# Idempotent: upserts on (source_id, item_type, item_key). Re-run any time.
# =============================================================================
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, "/app/veda_core")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veda_core"))

import psycopg2  # noqa: E402

EMBED_DIM = 1024

DDL = f"""
CREATE TABLE IF NOT EXISTS source_item_embeddings (
    source_id  TEXT        NOT NULL,
    item_type  TEXT        NOT NULL,
    item_key   TEXT        NOT NULL,
    name       TEXT,
    summary    TEXT,
    embedding  VECTOR({EMBED_DIM}),
    updated_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (source_id, item_type, item_key)
);
"""


def _conn(dbname: str):
    """Both databases live on the same server; only the name differs (CLAUDE.md: `veda`
    holds the Django tables, `veda_engine` the pgvector ones)."""
    return psycopg2.connect(
        host=os.environ.get("VEDA_INTERNAL_HOST", "pgbouncer"),
        port=int(os.environ.get("VEDA_INTERNAL_PORT", "6432")),
        dbname=dbname,
        user=os.environ.get("VEDA_INTERNAL_USER", "veda"),
        password=os.environ.get("VEDA_INTERNAL_PASSWORD", ""),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="", help="comma-separated source ids (default: all)")
    ap.add_argument("--batch", type=int, default=16, help="items per encode call")
    ap.add_argument("--force", action="store_true",
                    help="re-embed items that already have a vector")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    django_db = os.environ.get("POSTGRES_DB", "veda")
    engine_db = os.environ.get("VEDA_INTERNAL_DBNAME", "veda_engine")

    where, params = "summary <> ''", []
    if args.sources:
        ids = [s.strip() for s in args.sources.split(",") if s.strip()]
        where += " AND source_id = ANY(%s)"
        params.append([int(i) for i in ids])

    dconn = _conn(django_db)
    with dconn.cursor() as cur:
        cur.execute(f"SELECT source_id, item_type, item_key, name, summary "
                    f"FROM sources_sourceitem WHERE {where} ORDER BY source_id, item_key", params)
        items = cur.fetchall()
    dconn.close()

    if not items:
        print("no items with a summary — run `manage.py build_source_items` first")
        return 1

    econn = _conn(engine_db)
    econn.autocommit = False
    with econn.cursor() as cur:
        cur.execute(DDL)
    econn.commit()

    if not args.force:
        with econn.cursor() as cur:
            cur.execute("SELECT source_id, item_type, item_key FROM source_item_embeddings "
                        "WHERE embedding IS NOT NULL")
            have = {(r[0], r[1], r[2]) for r in cur.fetchall()}
        before = len(items)
        items = [i for i in items if (str(i[0]), i[1], i[2]) not in have]
        if before != len(items):
            print(f"skipping {before - len(items)} already-embedded item(s) (--force to redo)")

    print(f"embedding {len(items)} item(s) into {engine_db}.source_item_embeddings")
    if args.dry_run:
        for sid, itype, ikey, name, _ in items[:10]:
            print(f"  [dry-run] source {sid} {itype} {ikey} — {name}")
        return 0

    # Imported here, not at module scope: loading BGE-M3 costs ~20s and a --dry-run or an
    # empty run should not pay it.
    from ingestion.m3_encoder import encode_dense, get_embed_backend

    done = 0
    for start in range(0, len(items), args.batch):
        chunk = items[start:start + args.batch]
        texts = [f"{name}. {summary}" for _s, _t, _k, name, summary in chunk]
        vecs = encode_dense(texts)
        with econn.cursor() as cur:
            for (sid, itype, ikey, name, summary), vec in zip(chunk, vecs):
                cur.execute(
                    "INSERT INTO source_item_embeddings "
                    "(source_id, item_type, item_key, name, summary, embedding, updated_at) "
                    "VALUES (%s,%s,%s,%s,%s,%s,now()) "
                    "ON CONFLICT (source_id, item_type, item_key) DO UPDATE SET "
                    "name=EXCLUDED.name, summary=EXCLUDED.summary, "
                    "embedding=EXCLUDED.embedding, updated_at=now()",
                    [str(sid), itype, ikey, name, summary,
                     "[" + ",".join(f"{float(v):.8f}" for v in vec) + "]"])
        econn.commit()
        done += len(chunk)
        print(f"  {done}/{len(items)} (backend={get_embed_backend()})", flush=True)

    with econn.cursor() as cur:
        cur.execute("SELECT source_id, count(*) FROM source_item_embeddings "
                    "WHERE embedding IS NOT NULL GROUP BY 1 ORDER BY 1")
        for sid, n in cur.fetchall():
            print(f"  source {sid}: {n} item vector(s)")
    econn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
