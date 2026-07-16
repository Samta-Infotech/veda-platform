#!/usr/bin/env python
# =============================================================================
# scripts/backfill_semantic_bridge.py
# VEDA — One-time (idempotent) backfill for the semantic bridge
# (docs/SEMANTIC_ENTITY_BRIDGE.md — Tier A `semantic_about` + Tier B `semantic_value_of`)
#
# WHY THIS EXISTS
#   The semantic bridge links unstructured chunks to the structured semantic layer
#   at INGEST time. To light it up on an already-ingested database (no re-ingest of
#   sources), three things must be backfilled — each idempotent, each safe to re-run:
#
#     Phase EMBED   — embed every structured source's column/table nodes into
#                     graph_node_embeddings. THIS IS THE PIECE THAT WAS MISSING:
#                     relational sources whose columns were never embedded there have
#                     nothing for the bridge (or PPR seeding) to match against.
#     Phase VALUES  — embed eligible sampled DISPLAY values into entity_value_embeddings
#                     (Tier B index). Reconstructs value rows from the existing
#                     column_values store, so no source connection is needed.
#     Phase RELINK  — re-run entity linking for every document source using the chunks
#                     already in doc_chunks (no re-parse, no re-embed) so the exact +
#                     semantic_about + semantic_value_of edges get created.
#     Phase VERIFY  — print edge/embedding counts and assert the safety invariant
#                     (no semantic edge is column<->column, so none can drive a SQL join).
#
# SAFETY / IDEMPOTENCY
#   - Every writer scoped-deletes its own rows before re-inserting, so re-running is safe.
#   - Django is NOT required: sources are discovered from the internal graph store
#     (column/table nodes ⇒ structured source; chunk nodes ⇒ document source).
#   - Read-only against the tenant's source databases (it only reads the internal store
#     + re-embeds text); it never mutates a customer source.
#
# USAGE (inside the ingest-worker container; run with the ENGINE config on the path):
#   cd /app/veda_core && python /app/scripts/backfill_semantic_bridge.py --phase all
#   Flags: --phase all|embed|values|relink|verify   --sources 2,3,4   --tenant default
# =============================================================================

from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

# Ensure the ENGINE config wins over the Django `config/` package regardless of cwd —
# `import config` must resolve to veda_core/config.py. (Two config modules exist; see
# docs. Inserting the engine dir at sys.path[0] is the robust fix.)
_ENGINE_DIR = "/app/veda_core"
if os.path.isdir(_ENGINE_DIR) and _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)
else:
    # local/dev fallback: this file is <repo>/scripts/…; the engine is <repo>/veda_core
    _here = os.path.dirname(os.path.abspath(__file__))
    _guess = os.path.join(os.path.dirname(_here), "veda_core")
    if os.path.isdir(_guess):
        sys.path.insert(0, _guess)

from ingestion.db_abstraction import (           # noqa: E402
    INTERNAL_DB_AVAILABLE, get_internal_connection, release_internal_connection,
)


def _log(msg: str) -> None:
    print(f"[backfill] {msg}", flush=True)


# --------------------------------------------------------------------------- discovery
def _discover_sources() -> Dict[str, List[str]]:
    """{'structured': [source_id...], 'document': [source_id...]} from the internal
    graph store — no Django needed. Structured = has column/table nodes; document =
    has chunk nodes."""
    conn = get_internal_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT source_id FROM graph_nodes "
                    "WHERE node_type IN ('column','table') ORDER BY source_id")
        structured = [str(r[0]) for r in cur.fetchall()]
        cur.execute("SELECT DISTINCT source_id FROM graph_nodes "
                    "WHERE node_type = 'chunk' ORDER BY source_id")
        document = [str(r[0]) for r in cur.fetchall()]
        cur.close()
        return {"structured": structured, "document": document}
    finally:
        release_internal_connection(conn)


def _filter(ids: List[str], keep: Optional[set]) -> List[str]:
    return [i for i in ids if (keep is None or i in keep)] if ids else []


# --------------------------------------------------------------------------- phase EMBED
def phase_embed(structured: List[str]) -> None:
    """Embed column/table nodes into graph_node_embeddings for each structured source
    (the missing piece). Idempotent: embed_graph_nodes scoped-deletes per source."""
    from ingestion.graph_embedder import embed_graph_nodes
    _log(f"EMBED: {len(structured)} structured source(s): {structured}")
    for sid in structured:
        t = time.time()
        try:
            r = embed_graph_nodes(source_id=sid, verbose=False)
            _log(f"EMBED src{sid}: {r.nodes_embedded} node embeddings ({time.time()-t:.1f}s)")
        except Exception as e:
            _log(f"EMBED src{sid}: FAILED ({e})")


# --------------------------------------------------------------------------- phase VALUES
def _reconstruct_sampled_columns(source_id: str) -> List[SimpleNamespace]:
    """Rebuild value_sampler.SampledColumn-shaped records for one source from the existing
    column_values store (joined to graph_nodes for source_id + data_type). Only eligible
    types are needed; value_embedder re-filters, so we pass CATEGORY/FREE_TEXT through."""
    conn = get_internal_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT cv.col_id, cv.col_name, cv.table_id, cv.table_name, cv.semantic_type,
                   cv.value_norm, cv.value_raw, gn.data_type
            FROM column_values cv
            JOIN graph_nodes gn
              ON gn.ref_id = cv.col_id AND gn.node_type = 'column' AND gn.source_id = %s
            WHERE cv.semantic_type IN ('CATEGORY','FREE_TEXT')
            """, [str(source_id)])
        rows = cur.fetchall()
        cur.close()
    finally:
        release_internal_connection(conn)
    cols: Dict[str, SimpleNamespace] = {}
    for col_id, col_name, table_id, table_name, st, vnorm, vraw, dtype in rows:
        o = cols.get(col_id)
        if o is None:
            o = SimpleNamespace(col_id=col_id, col_name=col_name, table_id=table_id,
                                table_name=table_name, semantic_type=st,
                                data_type=dtype or "", values=[], raw_values=[])
            cols[col_id] = o
        o.values.append(vnorm)
        o.raw_values.append(vraw)
    return list(cols.values())


def phase_values(structured: List[str], tenant: str) -> None:
    """Build the Tier B value index (entity_value_embeddings) for each structured source
    that has sampled CATEGORY/FREE_TEXT values in column_values. Idempotent:
    embed_source_values scoped-deletes per source."""
    from ingestion.value_embedder import embed_source_values
    _log(f"VALUES: building value index for {len(structured)} structured source(s)")
    for sid in structured:
        sampled = _reconstruct_sampled_columns(sid)
        if not sampled:
            _log(f"VALUES src{sid}: no CATEGORY/FREE_TEXT values in column_values — skipped")
            continue
        t = time.time()
        try:
            r = embed_source_values(sid, sampled, tenant=tenant, verbose=False)
            _log(f"VALUES src{sid}: {r.values_embedded} value vectors / {r.columns} cols "
                 f"({time.time()-t:.1f}s)")
        except Exception as e:
            _log(f"VALUES src{sid}: FAILED ({e})")


# --------------------------------------------------------------------------- phase RELINK
def _reconstruct_chunks(source_id: str) -> List[SimpleNamespace]:
    """Rebuild DocumentChunk-shaped records for one document source from doc_chunks
    (chunk_id, text, doc metadata) — enough for link_entities + both semantic bridges;
    no re-parse or re-embed needed (chunk vectors already live in doc_chunks)."""
    from config import DOC_CHUNKS_TABLE_NAME
    conn = get_internal_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT chunk_id, doc_id, doc_name, chunk_index, text, page_num "
            f"FROM {DOC_CHUNKS_TABLE_NAME} WHERE source_id = %s ORDER BY chunk_index",
            [str(source_id)])
        rows = cur.fetchall()
        cur.close()
    finally:
        release_internal_connection(conn)
    out: List[SimpleNamespace] = []
    for chunk_id, doc_id, doc_name, chunk_index, text, page_num in rows:
        out.append(SimpleNamespace(
            chunk_id=chunk_id, doc_id=doc_id or "", doc_name=doc_name or "",
            chunk_index=chunk_index or 0, text=text or "", page_num=page_num,
            metadata={}))
    return out


def phase_relink(document: List[str], tenant: str) -> None:
    """Re-run entity linking for each document source over its existing chunks — creates
    the exact + semantic_about + semantic_value_of edges. Idempotent: link_entities
    scoped-deletes the source's chunk/entity edges first."""
    from ingestion.entity_linker import link_entities
    _log(f"RELINK: {len(document)} document source(s): {document}")
    for sid in document:
        chunks = _reconstruct_chunks(sid)
        if not chunks:
            _log(f"RELINK src{sid}: no chunks in doc_chunks — skipped")
            continue
        t = time.time()
        try:
            r = link_entities(chunks, sid, tenant=tenant, verbose=False)
            sa = r.stats.get("semantic_about", 0)
            sv = r.stats.get("semantic_value_of", 0)
            _log(f"RELINK src{sid}: {r.chunk_nodes} chunks, {r.entity_nodes} entities, "
                 f"{r.value_of} value_of, {sa} semantic_about, {sv} semantic_value_of "
                 f"({time.time()-t:.1f}s)")
        except Exception as e:
            _log(f"RELINK src{sid}: FAILED ({e})")


# --------------------------------------------------------------------------- phase VERIFY
def phase_verify() -> bool:
    """Print counts and assert the safety invariant. Returns True when safe."""
    conn = get_internal_connection()
    ok = True
    try:
        cur = conn.cursor()

        def one(sql, p=None):
            cur.execute(sql, p or []); return cur.fetchone()[0]

        _log("VERIFY: graph_node_embeddings per source:")
        cur.execute("SELECT source_id, node_type, count(*) FROM graph_node_embeddings "
                    "GROUP BY 1,2 ORDER BY 1,2")
        for r in cur.fetchall():
            _log(f"   src{r[0]} {r[1]}: {r[2]}")

        try:
            n_val = one("SELECT count(*) FROM entity_value_embeddings")
        except Exception:
            n_val = 0
            try: conn.rollback()
            except Exception: pass
        _log(f"VERIFY: entity_value_embeddings rows = {n_val}")

        sa = one("SELECT count(*) FROM graph_edges WHERE edge_type='semantic_about'")
        sv = one("SELECT count(*) FROM graph_edges WHERE edge_type='semantic_value_of'")
        _log(f"VERIFY: semantic_about edges = {sa}, semantic_value_of edges = {sv}")

        # dangling check: every semantic_value_of edge must resolve to an entity src node
        cur.execute("""SELECT count(*) FROM graph_edges e
                       WHERE e.edge_type='semantic_value_of'
                         AND NOT EXISTS (SELECT 1 FROM graph_nodes n
                                         WHERE n.node_id=e.src_node_id AND n.node_type='entity')""")
        dangling = cur.fetchone()[0]
        if dangling:
            ok = False
            _log(f"VERIFY: !! {dangling} dangling semantic_value_of edge(s) (entity pruned) — NOT OK")
        else:
            _log("VERIFY: no dangling semantic_value_of edges (OK)")

        # SAFETY INVARIANT: no semantic edge may connect two column nodes (only column<->
        # column can be a SQL join). This must be zero.
        cur.execute("""SELECT count(*) FROM graph_edges e
                       JOIN graph_nodes s ON s.node_id=e.src_node_id AND s.node_type='column'
                       JOIN graph_nodes d ON d.node_id=e.dst_node_id AND d.node_type='column'
                       WHERE e.edge_type IN ('semantic_about','semantic_value_of')""")
        col_col = cur.fetchone()[0]
        if col_col:
            ok = False
            _log(f"VERIFY: !! {col_col} column<->column semantic edge(s) — SAFETY VIOLATION")
        else:
            _log("VERIFY: 0 column<->column semantic edges — safety invariant holds (OK)")

        cur.close()
    finally:
        release_internal_connection(conn)
    _log("VERIFY: " + ("ALL CHECKS PASSED" if ok else "CHECKS FAILED — see above"))
    return ok


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill the VEDA semantic bridge (idempotent).")
    ap.add_argument("--phase", default="all",
                    choices=["all", "embed", "values", "relink", "verify"])
    ap.add_argument("--sources", default="", help="comma-separated source ids to limit to")
    ap.add_argument("--tenant", default=os.environ.get("VEDA_TENANT", "default"))
    args = ap.parse_args()

    if not INTERNAL_DB_AVAILABLE:
        _log("internal store not available (INTERNAL_DB_AVAILABLE=False) — aborting")
        return 2

    keep = set(s.strip() for s in args.sources.split(",") if s.strip()) or None
    disc = _discover_sources()
    structured = _filter(disc["structured"], keep)
    document = _filter(disc["document"], keep)
    _log(f"discovered structured={disc['structured']} document={disc['document']}"
         + (f"  (filtered to {sorted(keep)})" if keep else ""))

    t0 = time.time()
    if args.phase in ("all", "embed"):
        phase_embed(structured)
    if args.phase in ("all", "values"):
        phase_values(structured, args.tenant)
    if args.phase in ("all", "relink"):
        phase_relink(document, args.tenant)

    ok = True
    if args.phase in ("all", "verify"):
        ok = phase_verify()

    _log(f"done in {time.time()-t0:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
