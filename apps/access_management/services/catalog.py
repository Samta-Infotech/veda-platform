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
from apps.core.messages import MESSAGES
from ..models import RolePermission

from ..models import CatalogResource, Effect
from .base import NotFoundError, paginate

logger = logging.getLogger(__name__)

CODE_RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"

#: Exactly the columns ``views/catalog.py::public_fields`` renders.
CATALOG_LIST_FIELDS = (
    "id", "path", "kind", "parent_path", "source_id", "substrate_id",
    "is_active", "created_at", "updated_at",
)


class ResourceNotFound(NotFoundError):
    """No catalog resource with that path. Inherits its 404 from ``NotFoundError``."""

    code = CODE_RESOURCE_NOT_FOUND
    message = MESSAGES["catalog"]["not_found"]


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

        # Only relational databases have SchemaTable / SchemaColumn substrate rows
        if kind == "db":
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

        # Document sources (files) have no SchemaTable/SchemaColumn rows — their
        # content lives in the engine's own doc_chunks store (veda_engine), one row
        # per ingested document. Read it directly (same raw-psycopg2-to-veda_engine
        # pattern as storage_adapters.writer._build_lite_sm_from_graph) rather than
        # mirroring it into a Django table nobody else needs.
        elif kind == "files":
            for doc_id, doc_name in self._document_rows(source.pk):
                doc_path = self._safe_path(report, kind, source.name, doc_name)
                if doc_path is None:
                    continue
                if doc_path not in expected:
                    expected[doc_path] = (source_path, doc_id)

        return expected

    @staticmethod
    def _document_rows(source_id: int) -> list[tuple]:
        """(doc_id, doc_name) for every document ingested for this source, read live
        from the engine's doc_chunks store (veda_engine) — Django owns no mirrored
        copy of this. Best-effort: an unreachable engine store degrades to "no
        document children this run" rather than failing the whole catalog sync.
        """
        import os

        import psycopg2

        dsn = dict(
            host=os.environ.get("VEDA_INTERNAL_HOST", "pgbouncer"),
            port=int(os.environ.get("VEDA_INTERNAL_PORT", "6432")),
            dbname=os.environ.get("VEDA_INTERNAL_DBNAME", "veda_engine"),
            user=os.environ.get("VEDA_INTERNAL_USER", "veda"),
            password=os.environ.get(
                "VEDA_INTERNAL_PASSWORD", os.environ.get("POSTGRES_PASSWORD", "change-me")),
        )
        try:
            conn = psycopg2.connect(**dsn)
        except Exception:
            logger.warning("catalog discovery: doc_chunks unreachable for source_id=%s",
                           source_id)
            return []
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT doc_id, doc_name FROM doc_chunks WHERE source_id = %s",
                    [str(source_id)],
                )
                return cur.fetchall()
        finally:
            conn.close()

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

    #: category -> the CatalogResource.kind values it covers. Both the canonical
    #: names (validated by CatalogTreeSerializer) and their internal aliases map here,
    #: since the serializer normalises "db"/"lake"/"files"/"nosql" through too.
    _CATEGORY_KINDS = {
        "database": ("db", "nosql"), "db": ("db", "nosql"), "nosql": ("db", "nosql"),
        "datalake": ("lake",), "lake": ("lake",),
        "file_system": ("files",), "files": ("files",),
    }

    def get_tree(self, *, role_id: int | None = None, category: str = "",
                 parent_path: str | None = None, search: str = "") -> dict:
        """Dynamic hierarchical resource catalog tree projection.

        Supports grouped tabs (database, datalake, file_system) or level-by-level
        tree navigation with optional permission resolution for a role.
        """
        qs = CatalogResource.objects.filter(is_active=True)
        if search:
            qs = qs.filter(path__icontains=search)
        if category in self._CATEGORY_KINDS:
            qs = qs.filter(kind__in=self._CATEGORY_KINDS[category])

        grants = self._resource_grants(role_id) if role_id is not None else None
        has_children_of = self._has_children_index()

        if parent_path is not None:
            items = qs.filter(parent_path=parent_path).order_by("path")
            resources = [self._node(item, has_children_of, role_id, grants)
                        for item in items]
            return {
                "role_id": role_id,
                "parent_path": parent_path,
                "category": category or None,
                "resources": resources,
            }

        # Grouped tabs: every level when searching (a match can be at any depth),
        # otherwise just the roots — the entry points each tab starts from.
        scoped = qs if search else qs.filter(parent_path="")
        return {
            "role_id": role_id,
            "parent_path": "",
            "database": self._nodes(scoped, ("db", "nosql"), has_children_of, role_id, grants),
            "datalake": self._nodes(scoped, ("lake",), has_children_of, role_id, grants),
            "file_system": self._nodes(scoped, ("files",), has_children_of, role_id, grants),
        }

    @staticmethod
    def _resource_grants(role_id: int) -> dict[str, str]:
        """One role's resource-scoped grants, ``{resource_path: effect}``.

        Global grants (``resource_path == ""``) are excluded — they answer "may
        this role use permission X at all", not "is this catalog node allowed",
        so they have no place in a tree overlay.
        """

        return {
            rp.resource_path: rp.effect
            for rp in RolePermission.objects.filter(role_id=role_id).exclude(resource_path="")
        }

    @staticmethod
    def _has_children_index() -> set[str]:
        """Every path that is *someone's* parent, for an O(1) ``has_children`` check."""
        return set(
            CatalogResource.objects.filter(is_active=True)
            .values_list("parent_path", flat=True).distinct()
        )

    def _nodes(self, qs, kinds: tuple[str, ...], has_children_of: set[str],
               role_id: int | None, grants: dict | None) -> list[dict]:
        items = qs.filter(kind__in=kinds).order_by("path")
        return [self._node(item, has_children_of, role_id, grants) for item in items]

    def _node(self, item, has_children_of: set[str], role_id: int | None,
              grants: dict | None) -> dict:
        node = {
            "path": item.path,
            "name": resource_path.segments(item.path)[-1],
            "kind": item.kind,
            "parent_path": item.parent_path,
            "source_id": item.source_id,
            "has_children": item.path in has_children_of,
        }
        if role_id is not None:
            effect = self._resolve_effect(item.path, grants)
            node["effect"] = effect.upper() if effect else None
            node["is_allowed"] = (effect == Effect.ALLOW)
        return node

    @staticmethod
    def _resolve_effect(path: str, grants: dict[str, str]) -> str | None:
        """The grant that governs ``path``: itself, else its nearest granted ancestor.

        A grant on a parent covers every descendant (ADR §3.2's SELF_AND_DESCENDANTS)
        until a more specific grant overrides it closer to the leaf — so this walks
        from ``path`` outward and stops at the first match, most specific first.
        """
        for candidate in reversed(resource_path.prefixes(path)):
            if candidate in grants:
                return grants[candidate]
        return None

