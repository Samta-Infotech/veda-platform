"""apps.access_management.services.data_scope — the table/column allow-payload for
Gate 1's data-level narrowing (User Story 3, Task 14).

Task 13 (``apps.query.scope.permitted_source_ids``) answers "which sources can this
user reach at all" — a coarse, source-level filter. This module answers the next,
finer question for those already-permitted sources: "within a source, which tables
and columns does the user's ``data.read`` actually reach", so a later phase (Task
15/16) can forward that answer across the Django-to-inference process boundary and
have the engine filter ``sm['tables']``/``sm['columns']`` against it.

DESIGN: "fully open" vs "restricted", per source
    The overwhelmingly common grant shape in this programme's own recommended
    pattern is "whole-source ALLOW + specific DENY carve-outs" (see
    ``RBAC_PROGRESS_LOG.md``). Enumerating every table/column for a source that is
    simply wide open would make the cross-process payload scale with schema size for
    no reason — a real risk given HTTP header size limits. So a source is reported as
    ``open: True`` (no table/column list at all) when it has a source-level ALLOW and
    no narrower DENY exists anywhere under it; only when that's not the case does this
    walk the catalog and enumerate exactly what's reachable, at table granularity
    first (same "open" shortcut per table) and column granularity only when a table
    itself is partially restricted.

    This mirrors ``EffectivePermissions.allows()``'s own deny-wins prefix semantics
    exactly (a table/column is included only if ``allows()`` on its own path is
    True) — it does not reimplement authorization, it just enumerates it.

NAMES, NOT PATHS
    The engine's semantic model keys tables/columns by their real (as-scanned) name,
    which can have casing ``resource_path`` normalizes away (see the case-mismatch
    bug fixed in ``apps.query.scope.permitted_source_ids``). So this module resolves
    each ``CatalogResource.substrate_id`` back to the real ``SchemaTable``/
    ``SchemaColumn.name`` rather than trusting the lowercased path segment — reading
    with ``all_tenants()`` for the same reason ``CatalogDiscoveryService`` does: a
    resource path carries no tenant, and there is no ambient request context here.

CONTRACT
    ``compute_data_scope(user, source_ids)`` returns ``None`` for "no restriction
    at all" (RBAC off only — ``user.is_staff`` is Django-admin-panel login, not a
    data-scope bypass) — the caller then must not forward any scope payload,
    exactly as it must not narrow sources in that case. Otherwise it returns one
    entry per ``source_id`` (assumed already source-permitted by the caller;
    this function does not re-check source-level reachability).

RESOLVE ONCE PER REQUEST (User Story 3, Task 17 — centralized enforcement)
    ``resolve_effective_permissions(user)`` is the ONE place that decides "is
    this request even subject to RBAC" (off / staff bypass / real user) and, if
    so, pays for the one resolver query. Both ``apps.query.scope``'s
    source-level narrowing and this module's table/column narrowing take an
    already-resolved ``effective`` (or ``_UNRESOLVED``, the "resolve it
    yourself" default for a caller that hasn't centralized yet) instead of each
    calling ``PermissionResolver().resolve(user)`` independently — the
    literal "never rebuild permissions multiple times in different layers"
    requirement. A view calls this exactly once and threads the result through
    both checks; see ``apps.query.views``/``apps.chat.views``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import resource_path as rp
from ..codes import PermissionCode
from ..models import CatalogResource, Effect
from .resolver import EffectivePermissions, PermissionResolver
from apps.substrate.models import SchemaColumn, SchemaTable


@dataclass(frozen=True)
class TableScope:
    """One table's reachable shape. ``columns=None`` means "every column of this
    table" — the per-table mirror of a source-level ``open``."""

    name: str
    columns: tuple[str, ...] | None


@dataclass(frozen=True)
class SourceDataScope:
    """One source's reachable shape. ``tables=()`` when ``open`` is True — there is
    nothing to enumerate when the whole source is unrestricted."""

    open: bool
    tables: tuple[TableScope, ...] = ()


#: Sentinel default for the ``effective`` parameter below — distinct from ``None``,
#: which is itself a meaningful value ("no restriction"). A caller that omits
#: ``effective`` gets it resolved for them (unchanged behaviour for every caller
#: that predates Task 17's centralization); a caller that already resolved it once
#: (a view, per request) passes it through instead of paying for a second query.
UNRESOLVED = object()


def resolve_effective_permissions(user):
    """The ONE per-request resolution point (User Story 3, Task 17): decides
    whether this user is even subject to RBAC narrowing at all, and if so, pays
    for the one resolver query. Returns ``None`` for "no restriction anywhere" —
    no user, or RBAC off — else the resolved ``EffectivePermissions``.
    ``is_staff`` grants Django-admin-panel login only; it is NOT a data-scope
    bypass, so it is intentionally not checked here — a staff user is scoped
    by their granted role permissions like anyone else. Both the source-level
    check (``apps.query.scope.permitted_source_ids``) and this module's
    table/column check consume this SAME value when a caller centralizes it,
    instead of each re-resolving independently."""
    if user is None:
        return None

    from ..gate import MODE_OFF, rbac_mode

    if rbac_mode() == MODE_OFF:
        return None
    return PermissionResolver().resolve(user)


def serialize_data_scope(scope: dict[int, SourceDataScope] | None) -> dict | None:
    """``compute_data_scope``'s result as a JSON-safe wire payload (Task 15: this is
    what actually crosses the Django-to-inference HTTP boundary as the
    ``X-Veda-Data-Scope`` header), or ``None`` when there is nothing to restrict —
    the caller must then omit the header entirely rather than send an empty one, so
    "no header" and "no restriction" stay the same signal on both sides.

    Keys are stringified (``source_id`` -> ``str``): JSON object keys are always
    strings, and forcing that here — rather than leaving it to ``json.dumps`` to do
    silently — keeps the round-trip shape explicit at the one place it is produced.
    """
    if scope is None:
        return None
    return {
        str(source_id): {
            "open": source_scope.open,
            "tables": {
                table.name: list(table.columns) if table.columns is not None else None
                for table in source_scope.tables
            },
        }
        for source_id, source_scope in scope.items()
    }


def compute_data_scope(user, source_ids, effective=UNRESOLVED
                       ) -> dict[int, SourceDataScope] | None:
    """Per-source table/column reachability for ``user``, or ``None`` for "no
    restriction" (RBAC off, or staff). See the module docstring.

    ``effective``: an already-resolved ``EffectivePermissions`` (Task 17 — resolve
    once per request, reuse everywhere), or the default sentinel ``UNRESOLVED``,
    which resolves it internally exactly as before Task 17 — every caller that
    predates centralization keeps its existing behaviour unchanged."""
    if effective is UNRESOLVED:
        effective = resolve_effective_permissions(user)
    if effective is None:
        return None

    return {
        source_id: _source_scope(effective, source_id)
        for source_id in source_ids
    }


def _source_scope(effective: EffectivePermissions, source_id: int) -> SourceDataScope:
    root = (CatalogResource.objects
            .filter(source_id=source_id, parent_path="", is_active=True)
            .only("path", "kind").first())
    if root is None:
        # No catalog projection for this source (discovery never ran / found
        # nothing) — nothing is addressable, so nothing can be enumerated. Fail
        # closed: report it as restricted-with-nothing, not open.
        return SourceDataScope(open=False, tables=())

    if _fully_open(effective, root.path):
        return SourceDataScope(open=True)

    tables = list(CatalogResource.objects
                  .filter(source_id=source_id, parent_path=root.path, is_active=True)
                  .only("path", "substrate_id"))
    if not tables:
        return SourceDataScope(open=False, tables=())

    if root.kind == "files":
        # Document sources: each child IS a leaf (no columns beneath it, unlike a
        # db table) — and a document has no SchemaTable/SchemaColumn row of its own
        # (see apps.access_management.services.catalog._document_rows, which reads
        # doc names live from the engine's doc_chunks store instead of mirroring
        # them into a Django table). So the allowed name comes straight off the
        # resource path's own last segment, not a substrate lookup.
        allowed_docs = [
            rp.segments(t.path)[-1] for t in tables
            if effective.allows(PermissionCode.DATA_READ, t.path)
        ]
        if not allowed_docs:
            return SourceDataScope(open=False, tables=())
        return SourceDataScope(
            open=False,
            tables=tuple(TableScope(name=name, columns=None) for name in allowed_docs))

    table_names = _substrate_names("SchemaTable", [t.substrate_id for t in tables])

    open_tables = [t for t in tables if _fully_open(effective, t.path)]
    other_tables = [t for t in tables if t not in open_tables]

    scopes = [
        TableScope(name=table_names[t.substrate_id], columns=None)
        for t in open_tables if t.substrate_id in table_names
    ]

    # Under strict hierarchy the source is already allowed here (we are past the
    # _fully_open source check), so every column inherits ALLOW from it unless a
    # DENY carves it out — each non-open table's columns are checked directly via
    # allows() so a per-column DENY narrows the table to exactly its surviving
    # columns (and a table with none left is omitted below).
    if other_tables:
        columns = list(CatalogResource.objects
                       .filter(parent_path__in=[t.path for t in other_tables],
                               is_active=True)
                       .only("path", "parent_path", "substrate_id"))
        reachable_columns = [
            c for c in columns if effective.allows(PermissionCode.DATA_READ, c.path)]
        column_names = _substrate_names(
            "SchemaColumn", [c.substrate_id for c in reachable_columns])

        columns_by_table: dict[str, list[str]] = {}
        for c in reachable_columns:
            name = column_names.get(c.substrate_id)
            if name is not None:
                columns_by_table.setdefault(c.parent_path, []).append(name)

        for t in other_tables:
            if t.substrate_id not in table_names:
                continue
            table_columns = columns_by_table.get(t.path, [])
            if table_columns:  # a table with zero reachable columns is not
                scopes.append(  # addressable at all — omit it entirely.
                    TableScope(name=table_names[t.substrate_id], columns=tuple(table_columns)))

    return SourceDataScope(open=False, tables=tuple(scopes))


def _fully_open(effective: EffectivePermissions, path: str) -> bool:
    """Whether ``path`` is granted with no narrower DENY carving out an exception —
    see the module docstring's "open" definition. A DENY at ``path`` itself is
    already covered by ``allows()`` (deny-wins at the matched path), so only
    strictly-deeper grants need the extra scan here."""
    if not effective.allows(PermissionCode.DATA_READ, path):
        return False
    prefix = path + rp.SEPARATOR
    return not any(
        g.permission_code == PermissionCode.DATA_READ and g.effect == Effect.DENY
        and g.resource_path.startswith(prefix)
        for g in effective.grants
    )


def _substrate_names(model_name: str, substrate_ids) -> dict:
    """``{substrate_id: real_name}`` for the given ``SchemaTable``/``SchemaColumn``
    ids, read via ``all_tenants()`` — a resource path carries no tenant (ADR-0001
    §7), and there is no ambient request context in this service."""
    ids = [i for i in substrate_ids if i is not None]
    if not ids:
        return {}

    model = SchemaTable if model_name == "SchemaTable" else SchemaColumn
    return dict(model.objects.all_tenants().filter(id__in=ids).values_list("id", "name"))
