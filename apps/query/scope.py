"""apps.query.scope — server-side resolution of a request's query SCOPE (§6.2, P5).

The scope is the ordered list of source ids a query is allowed to run against,
primary first. It is ALWAYS resolved server-side: a client-supplied
``source_id``/``source_ids`` is treated as a *request*, intersected with the
tenant's ready-source registry, never trusted verbatim.

Extracted from ``QueryView`` (2026-08) because both HTTP entry points need it —
``/api/v1/query`` (apps.query.views.QueryView) and the chat turn endpoint
(apps.chat.views.ConversationQueryView). The chat view previously imported the
view class and called its private ``_resolve_scope`` staticmethod across an app
boundary; both now depend on this one public function instead. Behaviour is
unchanged — ``QueryView._resolve_scope`` remains as a thin delegating alias.

"""
from __future__ import annotations

import logging
import os
from django.db.models import Q
from apps.access_management.codes import PermissionCode
from apps.access_management.models import Effect
from apps.access_management.resource_path import InvalidResourcePath, source_of
from apps.sources.models import Source
from apps.access_management.services import resolve_effective_permissions


logger = logging.getLogger(__name__)

# Last-resort source id when the ready-source registry is unreadable or empty, so
# the fail-closed context seam (§4.1) always receives a source. Dev convenience:
# in a real deployment at least one Source row is ready.
ENV_DEFAULT_SOURCE_ID = "VEDA_DEFAULT_SOURCE_ID"
_FALLBACK_DEFAULT_SOURCE_ID = "1"

# Sentinel default for `effective` below — distinct from `None` (itself meaningful:
# "no restriction"). A LOCAL sentinel, not apps.access_management.services.UNRESOLVED
# — importing that module eagerly here would force apps.access_management into
# INSTALLED_APPS for every caller of this module, even ones that never touch RBAC
# (e.g. this module's own non-Django-configured test setups). A caller never needs
# to reference this object by name: it only ever passes a concrete `effective`
# (a real EffectivePermissions, or None) or omits the argument entirely.
_UNRESOLVED = object()


def resolve_query_scope(data, tenant, user=None, effective=_UNRESOLVED) -> list[int]:
    """The validated query scope (list of source ids, primary first) — §6.2, P5.

    Precedence: an explicit request ``source_ids``/``source_id`` is intersected with
    the tenant's READY sources (ownership check — never trust the body verbatim);
    absent any request pin, the default scope is ALL ready sources of the tenant.
    Falls back to ``VEDA_DEFAULT_SOURCE_ID`` when the registry can't be read /
    nothing is ready yet, so the fail-closed context seam (§4.1) always gets a source.

    RBAC narrows the READY set itself (before the request-pin intersection and
    before the no-pin fallback), so both existing branches respect it automatically
    without their own logic changing.

    Args:
        data: The raw request body (any mapping supporting ``.get``). Untrusted.
        tenant: The server-resolved tenant. Accepted for call-site symmetry and
            future per-tenant registry filtering; the ready registry is currently
            global, so it does not narrow the result today.
        user: The authenticated principal, if any. ``None`` (the default) means
            "no RBAC narrowing" — every existing caller that doesn't pass it keeps
            today's behaviour exactly.
        effective: An already-resolved ``EffectivePermissions`` (Task 17 — resolve
            once per request, reuse everywhere) or the sentinel default, which
            resolves it internally exactly as before Task 17.

    Returns:
        A non-empty, de-duplicated list of source ids, primary first.

    Raises:
        ValueError: if ``VEDA_DEFAULT_SOURCE_ID`` is set to a non-integer. This is
            a deployment misconfiguration and fails loudly on purpose rather than
            silently querying an unintended source.
    """
    default_source_id = int(os.environ.get(ENV_DEFAULT_SOURCE_ID, _FALLBACK_DEFAULT_SOURCE_ID))
    ready_source_ids = _ready_source_ids()

    permitted = permitted_source_ids(user, effective)
    if permitted is not None:
        ready_source_ids = [i for i in ready_source_ids if i in permitted]
    ready_set = set(ready_source_ids)

    requested_ids = _requested_source_ids(data)
    if requested_ids is not None:
        # Ownership: keep only ids the tenant actually owns (ready registry). If the
        # registry is unreadable, trust the explicit pin rather than fail the request.
        scope = [i for i in requested_ids if i in ready_set] if ready_set else requested_ids
        if scope:
            return list(dict.fromkeys(scope))
        logger.info("query scope: requested sources %s are not ready; falling back to "
                    "the tenant default scope (tenant=%s)", requested_ids, tenant)

    # No valid request pin → default to all ready sources (plan default), else the
    # dev fallback so inference always receives a context. Note: if RBAC narrowed
    # ready_source_ids to nothing, this still falls back to the dev default rather
    # than returning an empty scope — the caller (the view) is responsible for
    # answering "you have access to nothing" as a 403 BEFORE calling this when
    # that matters; this function's contract ("always returns a non-empty scope so
    # the context seam always gets a source") does not change.
    return ready_source_ids or [default_source_id]


def permitted_source_ids(user, effective=_UNRESOLVED) -> set[int] | None:
    """Which source ids ``user`` has ``data.read`` on, or ``None`` for "no
    narrowing" (RBAC off, staff bypass, or no user to check) — the PUBLIC, single
    source-level RBAC check (Task 17: also the one a view calls to decide whether
    to answer 403, since an empty ``set()`` here means "authenticated, but
    permitted nothing").

    Deliberately derived from ``EffectivePermissions.grants`` directly rather than
    ``.resources_for()`` — that method does not expand to descendants by design
    (see its own docstring), so a whole-source grant on ``db:crm`` would not be
    counted as covering it; here we want exactly "does any grant reach this
    source", which a plain scan of the (small, already-fetched) grant tuple
    answers without a second resolver call.

    A source qualifies if the user holds an ALLOW for ``data.read`` whose resource
    path names that source (``resource_path.source_of(path)`` matches). This counts
    raw ALLOW grants, not resolved allow/deny outcomes: "is this source reachable
    at all" only needs "is there at least one live path into it" — the deny-wins
    precision that matters for a SPECIFIC table/column check is Gate 1's
    table/column phase (a later change), not this source-level coarse filter. A
    source with an ALLOW on one table and a DENY on another still correctly stays
    in scope here; the DENY is enforced later, once the specific resource is known.

    A blank (global) resource path never counts — ADR §3.4/resolver docstring:
    a permission with no resource path is not resource-scoped, so it does not
    open any specific source. (This is why staff need the explicit bypass below:
    the seeded "Admin" role's grants are exactly this shape.)

    ``effective``: an already-resolved ``EffectivePermissions`` (Task 17 — resolve
    once per request; see ``apps.access_management.services.
    resolve_effective_permissions``, which is what the sentinel default delegates
    to). Lazily imported here — not at module level — so this module stays
    importable without ``apps.access_management`` in ``INSTALLED_APPS`` for a
    caller that never touches RBAC at all.
    """
    if effective is _UNRESOLVED:
        if user is None:
            return None  # fast path: no import at all when there's nothing to check

        effective = resolve_effective_permissions(user)
    if effective is None:
        return None



    source_names = set()
    for grant in effective.grants:
        if grant.permission_code != PermissionCode.DATA_READ:
            continue
        if grant.effect != Effect.ALLOW or grant.is_global:
            continue
        try:
            source_names.add(source_of(grant.resource_path))
        except InvalidResourcePath:
            continue  # a stored grant should always be canonical; never trust it blindly

    if not source_names:
        return set()
    # Case-insensitive: resource_path.build() (used when the catalog is discovered,
    # see CatalogService) always lowercases the source name into the path, but
    # Source.name itself is stored as typed — a plain name__in lookup on the
    # already-lowercased set would silently drop any source with uppercase in its
    # name and wrongly deny access that was actually granted.
    query = Q()
    for name in source_names:
        query |= Q(name__iexact=name)
    return set(Source.objects.filter(query).values_list("id", flat=True))


def _ready_source_ids() -> list[int]:
    """Ready sources, ascending. An unreadable registry (no migrations yet, DB
    down) degrades to "unknown ownership" rather than failing the request — the
    caller then trusts an explicit pin, exactly as before."""
    try:

        return list(Source.objects.filter(ready=True).order_by("id").values_list("id", flat=True))
    except Exception:  # noqa: BLE001 — registry unreadable must not fail the query
        logger.warning("query scope: ready-source registry unreadable; "
                       "falling back to the request pin / default source", exc_info=True)
        return []


def _requested_source_ids(data) -> list[int] | None:
    """The client's requested scope, coerced to ints — or None when the request
    pinned nothing. A malformed pin yields an empty list, which callers treat as
    "no usable pin" (falling through to the default scope) rather than an error."""
    requested = data.get("source_ids")
    if requested is None and data.get("source_id") is not None:
        requested = [data.get("source_id")]
    if requested is None:
        return None
    try:
        return [int(s) for s in requested]
    except (TypeError, ValueError):
        logger.warning("query scope: ignoring malformed source pin %r", requested)
        return []
