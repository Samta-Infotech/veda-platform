"""Catalog routes — read-only.

No write routes: the catalog is a projection rebuilt by discovery from `Source` /
`SchemaTable` / `SchemaColumn`. Re-sync is an operations action
(`manage.py sync_catalog`), not an authoring one.
"""
from django.urls import path

from ..views import CatalogDetailView, CatalogListView

urlpatterns = [
    path("catalog/detail", CatalogDetailView.as_view(), name="catalog-detail"),
    path("catalog/list", CatalogListView.as_view(), name="catalog-list"),
]
