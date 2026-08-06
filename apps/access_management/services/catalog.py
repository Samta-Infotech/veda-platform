"""Catalog discovery and read access.

Two responsibilities, deliberately in one module because they are two views of the
same projection:

  * ``CatalogDiscoveryService`` — rebuild ``CatalogResource`` from the authoritative
    catalog (``Source`` / ``SchemaTable`` / ``SchemaColumn``).
  * ``CatalogService`` — read it, for the admin tree and (later) the resolver.

RECONCILIATION IS THE INTEGRITY MECHANISM
    There is no foreign key to the substrate — it is deleted and recreated on every
    re-ingestion (see ``models/catalog.py``). So this service *is* the guard, and that
    makes it correctness-critical rather than a convenience:

      * upstream but not projected  -> insert
      * projected and still upstream -> reactivate if it had been deactivated
      * projected but gone upstream  -> **deactivate, never delete**

    Deleting would silently drop the grants that reference the row. Deactivating keeps
    the audit trail and makes the resolver deny — the correct failure direction.

    **Discovery must run after ingestion.** Between "substrate recreated" and
    "catalog re-synced" every resource of that source is absent, and absent means
    denied. The hook point is ``apps/ingestion/tasks.py:256``, where a source is marked
    ready; wiring it there is the deliberate next step and is NOT done here, because
    touching the ingestion pipeline is a separate, riskier change.

TENANCY
    ``SchemaTable``/``SchemaColumn`` are ``(source, tenant)``-scoped, but a resource
    path carries no tenant (ADR-0001 §7, option 1). Two tenants sharing a source would
    therefore produce the same path twice. Discovery collapses them to one row and logs
    it. Correct while VEDA is single-tenant; revisit with the multi-tenancy decision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from django.db import transaction
from django.db.models import Q
from django.utils import timezone

from .. import resource_path
from ..models import CatalogResource
from .base import NotFoundError, paginate

logger = logging.getLogger(__name__)

CODE_RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
MSG_RESOURCE_NOT_FOUND = "No such catalog resource."

#: Exactly the columns ``views/catalog.py::public_fields`` renders.
CATALOG_LIST_FIELDS = (
    "id", "path", "kind", "parent_path", "source_id", "substrate_id",
    "is_active", "created_at", "updated_at",
)


class ResourceNotFound(NotFoundError):
    """No catalog resource with that path. Inherits its 404 from ``NotFoundError``."""

    code = CODE_RESOURCE_NOT_FOUND
    message = MSG_RESOURCE_NOT_FOUND


@dataclass
class DiscoveryReport:
    """What one discovery run changed — returned rather than only logged, so a caller
    (a command, a task, a test) can assert on it and an operator can see drift."""

    created: int = 0
    reactivated: int = 0
    deactivated: int = 0
    unchanged: int = 0
    #: Resources that could not be addressed (e.g. a name containing ':'). Reported
    #: loudly rather than skipped silently — an unaddressable table is one nobody can
    #: grant access to, which an operator needs to know about.
    skipped: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "created": self.created, "reactivated": self.reactivated,
            "deactivated": self.deactivated, "unchanged": self.unchanged,
            "skipped": list(self.skipped),
        }


class CatalogDiscoveryService:
    """Rebuilds the addressable projection from the authoritative catalog."""

    def sync_source(self, source) -> DiscoveryReport:
        """Reconcile one source's resources. Idempotent.

        Args:
            source: A ``sources.Source`` instance.

        Returns:
            A ``DiscoveryReport``.

        Raises:
            resource_path.UnknownDialect: the source's dialect has no mapped kind, so
                nothing under it is addressable. Fails closed rather than guessing a
                kind and projecting resources into the wrong namespace.

        The whole reconciliation runs in one transaction: a partially-synced catalog
        would deny access to resources that do exist, and half a projection is worse
        than a stale one.
        """
        kind = resource_path.kind_for_dialect(source.dialect)
        report = DiscoveryReport()
        expected = self._expected_resources(source, kind, report)

        with transaction.atomic():
            existing = {
                row.path: row
                for row in CatalogResource.objects.filter(source=source)
            }
            self._upsert(source, expected, existing, report)
            self._deactivate_vanished(expected, existing, report)

        logger.info("catalog discovery source_id=%s name=%s %s",
                    source.pk, source.name, report.as_dict())
        return report

    def sync_all(self, sources=None) -> dict[int, DiscoveryReport]:
        """Reconcile every source (or the given ones).

        A source whose dialect cannot be mapped is reported and skipped rather than
        aborting the run — one unmappable source must not stop the rest of the catalog
        from being correct.
        """
        if sources is None:
            from apps.sources.models import Source

            sources = Source.objects.all().order_by("pk")

        reports: dict[int, DiscoveryReport] = {}
        for source in sources:
            try:
                reports[source.pk] = self.sync_source(source)
            except resource_path.UnknownDialect as exc:
                logger.error("catalog discovery skipped source_id=%s name=%s: %s",
                             source.pk, source.name, exc)
        return reports

    # -- expected state -----------------------------------------------------

    def _expected_resources(self, source, kind: str, report: DiscoveryReport) -> dict:
        """``{path: (parent_path, substrate_id)}`` for everything under one source.

        Reads the substrate with ``all_tenants()``: ``TenantScopedModel``'s default
        manager filters by the ambient ``veda_core.context``, which a management
        command or Celery task does not set — so the default manager would silently
        see nothing. Same precedent as ``storage_adapters/writer.py``.
        """
        from apps.substrate.models import SchemaColumn, SchemaTable

        expected: dict[str, tuple[str, object]] = {}

        source_path = self._safe_path(report, kind, source.name)
        if source_path is None:
            return expected
        expected[source_path] = ("", None)

        tables = (SchemaTable.objects.all_tenants()
                  .filter(source_id=source.pk).only("id", "name"))
        table_paths: dict[object, str] = {}
        for table in tables:
            table_path = self._safe_path(report, kind, source.name, table.name)
            if table_path is None:
                continue
            # First writer wins on a tenant collision — one path, one row (see the
            # module docstring on tenancy).
            if table_path not in expected:
                expected[table_path] = (source_path, table.id)
            table_paths[table.id] = table_path

        columns = (SchemaColumn.objects.all_tenants()
                   .filter(source_id=source.pk).only("id", "name", "table_id"))
        for column in columns:
            table_path = table_paths.get(column.table_id)
            if table_path is None:
                continue  # orphan column, or its table was unaddressable
            column_path = self._safe_path(
                report, *resource_path.segments(table_path), column.name)
            if column_path is None:
                continue
            if column_path not in expected:
                expected[column_path] = (table_path, column.id)

        return expected

    @staticmethod
    def _safe_path(report: DiscoveryReport, *parts: str) -> str | None:
        """Build a path, recording rather than raising when a name is unaddressable.

        One table with a ``:`` in its name must not make the whole source
        undiscoverable — but it must not vanish silently either, because nobody can
        grant access to a resource that has no name.
        """
        try:
            return resource_path.build(*parts)
        except resource_path.InvalidResourcePath as exc:
            report.skipped.append(f"{'.'.join(str(p) for p in parts)}: {exc}")
            return None

    # -- reconciliation -----------------------------------------------------

    @staticmethod
    def _upsert(source, expected: dict, existing: dict, report: DiscoveryReport) -> None:
        # bulk_update does NOT call Field.pre_save, so `auto_now` never fires and
        # updated_at would be written back at its stale in-memory value — the row
        # would silently claim it had not changed. (Model.save(update_fields=...) is
        # different: it DOES call pre_save, which is why roles/users can just name the
        # field.) Stamping one timestamp for the whole run is also more truthful: these
        # rows changed as one reconciliation.
        now = timezone.now()
        to_create = []
        to_reactivate = []
        for path, (parent, substrate_id) in expected.items():
            row = existing.get(path)
            if row is None:
                to_create.append(CatalogResource(
                    path=path, kind=path.split(":")[0], parent_path=parent,
                    source=source, substrate_id=substrate_id, is_active=True))
            elif not row.is_active or row.substrate_id != substrate_id:
                row.is_active = True
                row.substrate_id = substrate_id
                row.updated_at = now
                to_reactivate.append(row)
            else:
                report.unchanged += 1

        if to_create:
            # ignore_conflicts: a concurrent discovery run for the same source may have
            # inserted the same path. The unique constraint arbitrates; losing the race
            # is not an error because both runs compute the same rows.
            CatalogResource.objects.bulk_create(to_create, ignore_conflicts=True)
            report.created += len(to_create)
        if to_reactivate:
            CatalogResource.objects.bulk_update(
                to_reactivate, ["is_active", "substrate_id", "updated_at"])
            report.reactivated += len(to_reactivate)

    @staticmethod
    def _deactivate_vanished(expected: dict, existing: dict,
                             report: DiscoveryReport) -> None:
        vanished = [row.pk for path, row in existing.items()
                    if path not in expected and row.is_active]
        if vanished:
            # Deactivate, never delete: deleting would silently drop every grant that
            # references these paths.
            report.deactivated += CatalogResource.objects.filter(
                pk__in=vanished).update(is_active=False)


class CatalogService:
    """Read access to the catalog projection.

    ``request`` is accepted for symmetry with the other services (and so a future
    audit hook has the actor).
    """

    def __init__(self, request=None):
        self._request = request

    def list_resources(self, *, page: int, page_size: int, search: str = "",
                       is_active=None, ordering: str = "path",
                       kind: str = "", parent_path: str | None = None,
                       source_id=None) -> tuple[list, int]:
        """One page of catalog resources, plus the total matching count.

        ``parent_path`` is tri-state, like ``is_active``:
          * omitted / ``None`` -> no level filter, every resource
          * ``""``             -> the source-level roots (their parent is "")
          * a path             -> exactly that node's children (the lazy-tree call)
        """
        queryset = CatalogResource.objects.all()
        if search:
            queryset = queryset.filter(Q(path__icontains=search))
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active)
        if kind:
            queryset = queryset.filter(kind=kind)
        if parent_path is not None:
            # "" is meaningful here — it selects the roots — so only None disables
            # the filter.
            queryset = queryset.filter(parent_path=parent_path)
        if source_id is not None:
            queryset = queryset.filter(source_id=source_id)

        return paginate(queryset, page=page, page_size=page_size, ordering=ordering,
                        only_fields=CATALOG_LIST_FIELDS)

    def get_resource(self, path: str) -> CatalogResource:
        """One resource by its canonical path.

        The path is canonicalised before lookup, so a caller sending ``DB:CRM`` finds
        the same row as one sending ``db:crm`` — otherwise the same resource would be
        addressable under two strings, only one of which matches its grants.

        Raises:
            ResourceNotFound: unknown path, or one that is not even expressible.
        """
        try:
            canonical = resource_path.validate(path)
        except resource_path.InvalidResourcePath:
            # An unexpressible path can name nothing. Reported as "not found" rather
            # than as a validation error: whether a path exists is not information an
            # unaddressable string should be able to probe for.
            raise ResourceNotFound() from None

        resource = (CatalogResource.objects.filter(path=canonical)
                    .only(*CATALOG_LIST_FIELDS).first())
        if resource is None:
            raise ResourceNotFound()
        return resource
