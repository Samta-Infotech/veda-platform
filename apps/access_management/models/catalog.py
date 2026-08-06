"""``CatalogResource`` — the addressable projection of VEDA's catalog.

PURPOSE
    Gives grants one uniform thing to point at. Every resource a permission can be
    granted on — a source, a table, a column, a document — has exactly one row here,
    identified by its canonical path (ADR-0001).

    This is *addressing*, not metadata. `Source`, `SchemaTable` and `SchemaColumn`
    remain the authoritative description of what exists; this table records only how
    to *name* those things and how they nest. Copying row counts, data types or
    sensitivity flags in here would make it a competing catalog — the exact mixing of
    "catalog is metadata, permission is authorization" the architecture forbids.

WHY NOT FOREIGN KEYS TO THE SUBSTRATE — the central design point
    ``storage_adapters/writer.py:137-139`` deletes and recreates **every**
    ``SchemaTable``/``SchemaColumn`` row for a source on **every re-ingestion**::

        SchemaColumn.objects.all_tenants().filter(source_id=..., tenant=...).delete()
        SchemaTable.objects.all_tenants().filter(source_id=..., tenant=...).delete()

    So a substrate FK is unsafe at any setting:
      * ``CASCADE``  — re-ingesting a source would delete its catalog rows and, with
        them, every grant referencing them. Silent mass revocation.
      * ``PROTECT``  — re-ingestion raises ``ProtectedError``; the AI pipeline breaks.
      * ``SET_NULL`` — the row survives with a null FK, so the FK guarantees nothing.

    Those rows are a *rebuildable projection* of an upstream database, not durable
    entities. Their identity is stable (ids are ``uuid5``, see
    ``veda_core/connectors/relational.py:405``) but their lifetime is not.

    Identity here is therefore the natural key — the path — and referential integrity
    to the substrate is replaced by reconciliation in
    ``services/catalog.py::CatalogDiscoveryService``.

    ``source`` IS a foreign key, with ``PROTECT``: ``Source`` rows are durable, and
    blocking the deletion of a source that still carries grants is exactly right.

EXTENSION POINTS (deliberately not built)
    ``RolePermission`` will reference ``path``. New resource kinds need no schema
    change — only a ``KIND_BY_DIALECT`` entry. Resource groups, wildcards and policies
    all attach to the path without touching this table.

SECURITY
    ``is_active`` is authorization-relevant: discovery *deactivates* resources that
    have vanished rather than deleting them, so grants survive a transient
    re-ingestion window and the audit trail stays readable. A resolver must treat an
    inactive or missing resource as **denied**.

PERFORMANCE
    Two access patterns, two indexes. ``path`` (unique) answers "does this resource
    exist / what is granted on it". ``parent_path`` answers "list this node's
    children" for the admin tree. Prefix resolution never needs a ``LIKE`` scan:
    ``resource_path.prefixes()`` expands to at most ``MAX_SEGMENTS`` exact paths, so
    the resolver reads with ``path__in=[...]`` against the unique index.
"""
from __future__ import annotations

from django.db import models

from apps.core.models import TimeStampedModel


class CatalogResource(TimeStampedModel):
    """One addressable resource. See the module docstring for the design notes."""

    #: Canonical path (ADR-0001) — the identity grants reference. Unique because a
    #: resource must have exactly one name; two rows for one path would let a grant
    #: match one and a check match the other.
    path = models.CharField(max_length=512, unique=True)

    #: Coarse family from ``resource_path.KIND_BY_DIALECT``. Denormalized from the
    #: path's first segment so the admin UI can filter by kind without parsing.
    kind = models.CharField(max_length=20)

    #: Immediate parent path, or "" for a source-level resource. Indexed: this is the
    #: whole tree-navigation access pattern, and computing it from ``path`` at query
    #: time would force a scan.
    parent_path = models.CharField(max_length=512, blank=True, db_index=True)

    #: The owning source. PROTECT because deleting a source that still has catalog
    #: rows (and therefore possibly grants) must be a deliberate, blocked-by-default act.
    source = models.ForeignKey(
        "sources.Source", on_delete=models.PROTECT, related_name="catalog_resources")

    #: The ``SchemaTable``/``SchemaColumn`` UUID this projects, when there is one.
    #: A plain column, NOT a foreign key — see the module docstring. It carries no
    #: integrity guarantee; it exists so retrieval can join back to the substrate
    #: without re-deriving the id.
    substrate_id = models.UUIDField(null=True, blank=True, db_index=True)

    #: False once discovery no longer finds the resource upstream. Never deleted:
    #: deletion would silently drop the grants that reference it.
    is_active = models.BooleanField(default=True)

    class Meta:
        indexes = [
            # "everything under this source, currently live" — the admin root listing
            # and the resolver's per-source narrowing.
            models.Index(fields=["source", "is_active"],
                         name="catalog_source_active_idx"),
        ]
        ordering = ("path",)
        verbose_name = "Catalog resource"
        verbose_name_plural = "Catalog resources"

    def __str__(self) -> str:
        return self.path
