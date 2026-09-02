"""Build + profile the uniform SourceItem layer for sources (multi-source routing).

Backfills SourceItem rows from the kind-specific stores AND profiles each item (SLM summary +
BGE-M3 embedding into source_item_embeddings, the routing prior). Delegates to
`apps.sources.item_profiler.build_source_items` — the same service the post-ingestion hook uses,
so a manual run and an automatic ingestion produce identical results. Idempotent.

    python manage.py build_source_items            # all sources
    python manage.py build_source_items --source 3 # one source
    python manage.py build_source_items --force    # re-profile already-summarised items
"""
from django.core.management.base import BaseCommand

from apps.sources.models import Source
from apps.sources.item_profiler import build_source_items


class Command(BaseCommand):
    help = "Backfill + profile SourceItem rows (structural + SLM summary + embedding)."

    def add_arguments(self, parser):
        parser.add_argument("--source", type=int, default=None)
        parser.add_argument("--force", action="store_true")

    def handle(self, *args, **opts):
        qs = Source.objects.all()
        if opts["source"]:
            qs = qs.filter(pk=opts["source"])
        total_items = total_prof = 0
        for s in qs:
            r = build_source_items(s.pk, force=opts["force"])
            total_items += r["items"]
            total_prof += r["profiled"]
            self.stdout.write(f"  src {s.pk} ({s.name}): {r['items']} items, {r['profiled']} profiled")
        self.stdout.write(self.style.SUCCESS(
            f"Built {total_items} items, profiled {total_prof}."))
