"""Backfill the uniform SourceItem layer from the existing kind-specific stores.

Populates one `SourceItem` per top-level thing a source holds — a table (from `column_embeddings_v2`)
or a document (from `doc_chunks`) — so a filesystem source's documents become first-class items exactly
like a datalake's tables (docs/multisource_routing/SOURCE_ITEM_METADATA_DESIGN.md). Idempotent
(upsert on the natural key). This is a DATA operation, deliberately separate from the schema migration.

Reads the engine store on the SAME Django DB connection (postgres in the container). Safe to re-run.

    python manage.py backfill_source_items            # all ready sources
    python manage.py backfill_source_items --source 3 # one source
"""
from django.core.management.base import BaseCommand
from django.db import connection

from apps.sources.models import Source, SourceItem, SourceItemType


def _rows(sql, params):
    with connection.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


class Command(BaseCommand):
    help = "Backfill SourceItem rows from column_embeddings_v2 (tables) and doc_chunks (documents)."

    def add_arguments(self, parser):
        parser.add_argument("--source", type=int, default=None, help="only this source id")

    def handle(self, *args, **opts):
        qs = Source.objects.all()
        if opts["source"]:
            qs = qs.filter(pk=opts["source"])
        total = 0
        for s in qs:
            sid = str(s.pk)
            kind = s.source_kind()  # relational|datalake|document|nosql
            made = 0
            # ── tabular: one item per table (from column_embeddings_v2) ──────────────────────
            try:
                tabs = _rows(
                    "SELECT table_name, count(*) FROM column_embeddings_v2 "
                    "WHERE source_id=%s GROUP BY table_name", [sid])
            except Exception as e:
                tabs = []
                self.stderr.write(f"  src {sid}: column store read failed ({e})")
            item_type = SourceItemType.DATASET if kind == "datalake" else (
                SourceItemType.COLLECTION if kind == "nosql" else SourceItemType.TABLE)
            for table_name, ncols in tabs:
                cols = [r[0] for r in _rows(
                    "SELECT col_name FROM column_embeddings_v2 WHERE source_id=%s AND table_name=%s "
                    "LIMIT 50", [sid, table_name])]
                SourceItem.objects.update_or_create(
                    source=s, item_type=item_type, item_key=table_name,
                    defaults=dict(name=table_name, child_count=ncols,
                                  item_metadata={"columns": cols, "column_count": ncols}))
                made += 1
            # ── documents: one item per file (from doc_chunks) ───────────────────────────────
            try:
                docs = _rows(
                    "SELECT doc_name, count(*) FROM doc_chunks WHERE source_id=%s GROUP BY doc_name",
                    [sid])
            except Exception as e:
                docs = []
                self.stderr.write(f"  src {sid}: chunk store read failed ({e})")
            for doc_name, nchunks in docs:
                ftype = (doc_name.rsplit(".", 1)[-1].lower() if "." in doc_name else "")
                SourceItem.objects.update_or_create(
                    source=s, item_type=SourceItemType.DOCUMENT, item_key=doc_name,
                    defaults=dict(name=doc_name, child_count=nchunks,
                                  item_metadata={"file_type": ftype, "chunk_count": nchunks}))
                made += 1
            total += made
            self.stdout.write(f"  src {sid} ({s.name}, {kind}): {made} items")
        self.stdout.write(self.style.SUCCESS(f"Backfilled {total} SourceItem rows."))
