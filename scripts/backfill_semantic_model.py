"""Backfill per-source semantic models for tabular/doc sources (MULTI_SOURCE_SERVING.md MS-2).

The tabular/doc ingestion never built a per-source semantic model, so the global homzhub
model got persisted + published under sources 3/4/5. This rebuilds a DETERMINISTIC semantic
model for each given source FROM its own column nodes in graph_nodes (which are per-source
correct), then persists it to the Sm* substrate and republishes veda:sm:{sid}:{tenant} — with
no re-ingest. Homzhub (source 2, real semantic_layer_v2 output) is never touched.

Run inside the api/ingest container (needs Django + storage_adapters):
    python scripts/backfill_semantic_model.py --source-ids 4,5,3 --tenant default
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import django

_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _tokens(*names: str) -> list:
    out: list = []
    for n in names:
        for t in _TOKEN_RE.split((n or "").lower()):
            if t and len(t) >= 2 and t not in out:
                out.append(t)
    return out


def _agg_for(sem_type: str, data_type: str) -> list:
    st = (sem_type or "").upper()
    if st in ("METRIC", "MEASURE") or (data_type or "").lower() in (
            "integer", "bigint", "numeric", "double", "smallint", "real", "money"):
        return ["SUM", "AVG", "MIN", "MAX", "COUNT"]
    return ["COUNT", "GROUP_BY"]


def _internal_conn():
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get("VEDA_INTERNAL_HOST", "pgbouncer"),
        port=int(os.environ.get("VEDA_INTERNAL_PORT", "6432")),
        dbname=os.environ.get("VEDA_INTERNAL_DBNAME", "veda_engine"),
        user=os.environ.get("VEDA_INTERNAL_USER", "veda"),
        password=os.environ.get("VEDA_INTERNAL_PASSWORD", os.environ.get("POSTGRES_PASSWORD", "change-me")),
    )


def _read_columns(cur, source_id: str) -> list:
    cur.execute(
        "SELECT table_name, name, semantic_type, data_type, is_pk, is_fk "
        "FROM graph_nodes WHERE node_type='column' AND source_id=%s ORDER BY table_name, name",
        [str(source_id)])
    return [dict(table_name=r[0], col_name=r[1], semantic_type=(r[2] or "FREE_TEXT"),
                 data_type=(r[3] or ""), is_pk=bool(r[4]), is_fk=bool(r[5]))
            for r in cur.fetchall()]


def build_sm(cols: list) -> dict:
    """Deterministic semantic model from column rows — structurally compatible with the
    homzhub semantic_layer_v2 output (tables/columns/retrieval_documents)."""
    tables: dict = {}
    columns: dict = {}
    retrieval_documents: dict = {}
    by_table: dict = {}
    for c in cols:
        by_table.setdefault(c["table_name"], []).append(c)

    for tname, tcols in by_table.items():
        measures = [c["col_name"] for c in tcols
                    if (c["data_type"] or "").lower() in ("integer", "bigint", "numeric", "double", "real", "money")
                    and not c["is_pk"] and not c["is_fk"]]
        tables[tname] = {
            "table_name": tname,
            "business_purpose": f"Records from the {tname} source table.",
            "primary_entity": f"A single {tname} row.",
            "table_type": "TRANSACTION" if measures else "REFERENCE",
            "candidate_temporal_columns": [c["col_name"] for c in tcols
                                           if "date" in c["col_name"].lower() or "time" in c["col_name"].lower()],
            "candidate_measure_columns": measures,
        }
        for c in tcols:
            key = f"{tname}.{c['col_name']}"
            role = "IDENTIFIER" if (c["is_pk"] or c["is_fk"] or c["semantic_type"] == "IDENTIFIER") else \
                   ("METRIC" if c["col_name"] in measures else "DIMENSION")
            aliases = _tokens(c["col_name"])
            columns[key] = {
                "col_name": c["col_name"], "table_name": tname,
                "semantic_type": c["semantic_type"], "analytics_role": role,
                "business_definition": f"{c['col_name'].replace('_', ' ')} of {tname}.",
                "aliases": aliases, "allowed_aggregations": _agg_for(c["semantic_type"], c["data_type"]),
                "confidence": 0.6, "business_role": role.title(),
                "column_domain": tname, "sql_usage": "", "contains_pii": False,
                "sample_values": [], "importance_class": "NORMAL",
                "data_type": c["data_type"],
            }
            retrieval_documents[key] = (
                f"COLUMN: {c['col_name']} | ROLE: {role} | DEFINITION: "
                f"{c['col_name'].replace('_', ' ')} of {tname} | TERMS: {', '.join(aliases)}")
    return {"version": "2.0-lite", "tables": tables, "columns": columns,
            "retrieval_documents": retrieval_documents, "domain_synonyms": {}, "concept_graph": {}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-ids", default="4,5,3")
    ap.add_argument("--tenant", default="default")
    args = ap.parse_args()

    django.setup()   # DJANGO_SETTINGS_MODULE comes from the container env
    from storage_adapters import assembler

    sids = [s.strip() for s in args.source_ids.split(",") if s.strip()]
    conn = _internal_conn()
    try:
        with conn.cursor() as cur:
            for sid in sids:
                cols = _read_columns(cur, sid)
                sm = build_sm(cols)
                # persist/publish take source_id explicitly — no ambient context needed.
                assembler.persist(sm, source_id=int(sid), tenant=args.tenant, version="2.0-lite")
                n = assembler.publish_sm(int(sid), args.tenant)
                print(f"source {sid}: {len(sm['tables'])} tables, {len(sm['columns'])} cols "
                      f"→ Sm* persisted + published ({n} bytes)")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
