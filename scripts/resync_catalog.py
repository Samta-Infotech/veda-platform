"""Production Catalog Cleanup and Resync Script.

Run on production or dev server via:
    python scripts/resync_catalog.py

Or via manage.py shell:
    python manage.py shell < scripts/resync_catalog.py

Does not depend on any specific IDs; dynamically cleans stale catalog resources
for non-relational sources (e.g., file_system / datalake) and re-syncs the catalog.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")

import django
django.setup()

from apps.access_management.models import CatalogResource
from apps.access_management.services.catalog import CatalogDiscoveryService
from apps.sources.models import Source


def main():
    print("Starting Catalog Cleanup & Resync...")

    # 1. Clean up stale sub-resource children for non-DB sources (where dialect is not a relational DB)
    non_db_sources = Source.objects.exclude(dialect__in=["postgres", "postgresql", "mysql", "sqlite"])
    cleaned_count = 0

    for source in non_db_sources:
        # Delete children catalog resources under non-DB sources
        prefix = f"{source.name}:"
        deleted, _ = CatalogResource.objects.filter(path__icontains=prefix).exclude(parent_path="").delete()
        cleaned_count += deleted
        print(f"Cleaned {deleted} stale child resource(s) for source '{source.name}'")

    # 2. Re-sync catalog for all registered sources
    print("Running CatalogDiscoveryService().sync_all()...")
    reports = CatalogDiscoveryService().sync_all()

    for source_id, report in reports.items():
        print(f"Synced Source ID {source_id}: {report.as_dict()}")

    print("Catalog Cleanup and Resync completed successfully!")


if __name__ == "__main__":
    main()
