"""``manage.py sync_catalog`` — reconcile the catalog projection.

Discovery is correctness-critical: because ``CatalogResource`` has no foreign key to
the substrate (it is deleted and recreated on every re-ingestion), reconciliation is
the only thing keeping the projection honest. Between "substrate recreated" and
"catalog re-synced", every resource of that source is absent — and absent means denied.

This command is the operations surface. The durable fix is to call
``CatalogDiscoveryService`` from ``apps/ingestion/tasks.py`` where a source is marked
ready; that is a deliberate follow-up, kept out of this change because touching the
ingestion pipeline carries its own risk.
"""
from django.core.management.base import BaseCommand

from apps.access_management.services import CatalogDiscoveryService


class Command(BaseCommand):
    help = "Rebuild the CatalogResource projection from Source/SchemaTable/SchemaColumn."

    def add_arguments(self, parser):
        parser.add_argument(
            "--source-id", type=int, default=None,
            help="Reconcile only this source. Omit to reconcile every source.")

    def handle(self, *args, **options):
        from apps.sources.models import Source

        source_id = options["source_id"]
        sources = (Source.objects.filter(pk=source_id) if source_id is not None
                   else Source.objects.all().order_by("pk"))
        if source_id is not None and not sources.exists():
            self.stderr.write(self.style.ERROR(f"no source with id {source_id}"))
            return

        reports = CatalogDiscoveryService().sync_all(sources)

        for pk, report in reports.items():
            summary = report.as_dict()
            skipped = summary.pop("skipped")
            self.stdout.write(f"source {pk}: " + ", ".join(
                f"{k}={v}" for k, v in summary.items()))
            for entry in skipped:
                # Unaddressable resources are the one thing an operator must act on:
                # nobody can grant access to a resource that has no name.
                self.stderr.write(self.style.WARNING(f"  unaddressable: {entry}"))
        if not reports:
            self.stdout.write("no sources reconciled")
