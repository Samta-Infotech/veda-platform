"""Permission resolution — turning the RBAC graph into an answer.

    resolve(user)  ->  EffectivePermissions   (immutable)

RESPONSIBILITY, AND WHAT IS DELIBERATELY NOT HERE
    This module *resolves*. It does not enforce. Nothing here inspects a request,
    raises 403, or knows what an endpoint is — that is Gate 2's job, in a later phase.
    Keeping the two apart is what lets the resolver be deployed and observed (and
    cached) before a single request's outcome changes.

THE RULES IT IMPLEMENTS (ADR-0001 §3.5, + strict hierarchy 2026-08)
    For a permission on a resource:

      1. Collect every grant whose path is a prefix-or-equal of the requested resource.
      2. If ANY is deny  -> DENY.
      3. Else if the SOURCE-level ancestor (2-segment ``db:<source>`` prefix) is
         itself allowed -> ALLOW.
      4. Else -> DENY.

    Two independent fail-closed rules: DENY is unpierceable at any depth, and the
    absence of a *source-level* grant is a denial.

    STRICT HIERARCHY (user's call): the source is the gate. Rule 3 means an ALLOW
    on a table/column with no source-level allow above it grants NOTHING — the
    model is "allow the source, refine DOWN with denies", not "allow-list
    individual tables from a denied/ungranted source". ``allows()`` carries the
    full reasoning; ``permitted_source_ids`` and the catalog tree's
    ``_resolve_effect`` mirror it and MUST change with it.

WHAT MAKES SOMETHING GRANT NOTHING
    A grant is only counted when the whole chain is live — an inactive **user**,
    **role** or **permission** contributes nothing. Filtered in the query rather than
    in Python, so a disabled capability cannot leak through a code path that forgot to
    check it.

GLOBAL GRANTS DO NOT COVER RESOURCES
    ``resource_path=""`` means "this permission is not resource-scoped"
    (``user.manage``), NOT "every resource". A check for ``data.read`` on
    ``db:crm:employee`` is **not** satisfied by a blank-path grant. This follows from
    ADR §3.4 — a blank path has zero segments and is never a prefix of a real one —
    and it is the fail-closed reading: an administrator who grants ``data.read`` with
    no resource must not silently open every table in the platform.

PERFORMANCE
    Resolution is **one query** (user -> roles -> grants, joined and filtered in the
    database) plus O(depth) work per check: ``prefixes()`` yields at most
    ``MAX_SEGMENTS`` candidates, matched against a per-code index built once. There is
    no N+1 and no per-check query.

    It is still one query *per resolution*, which is why Phase 7 caches the result.
    ``EffectivePermissions`` is frozen and self-describing precisely so it can be
    cached whole; the cache key will need a version counter (``PermissionVersion``),
    which does not exist yet — noted rather than half-built.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .. import resource_path as rp
from ..models import Effect, RolePermission


@dataclass(frozen=True, slots=True)
class Grant:
    """One resolved decision. Frozen: a caller must never be able to edit its own
    permissions after the fact."""

    permission_code: str
    resource_path: str
    effect: str

    @property
    def is_global(self) -> bool:
        """Whether this grant is not resource-scoped."""
        return self.resource_path == ""


@dataclass(frozen=True)
class EffectivePermissions:
    """Everything a user may (and may not) do, as one immutable value.

    Immutable by design: this object is the answer to an authorization question, it
    will be cached and shared, and a mutable answer is one a later caller can quietly
    change. ``grants`` is a tuple and ``_by_code`` a read-only mapping.
    """

    user_id: int
    grants: tuple[Grant, ...]
    #: permission_code -> that code's grants. Built once so ``allows`` is O(depth),
    #: not O(total grants).
    _by_code: Mapping[str, tuple[Grant, ...]]

    def allows(self, permission_code: str, resource_path: str = "") -> bool:
        """Whether this user may exercise ``permission_code`` on ``resource_path``.

        Args:
            permission_code: e.g. ``"data.read"``.
            resource_path: The target. Omit for a permission that is not
                resource-scoped; a blank-path grant will NOT satisfy a check that
                names a resource (see the module docstring).

        Returns:
            True only if the deny/allow rules below are satisfied. Every other
            outcome — no grants, unknown permission, unaddressable path — is False.

        STRICT HIERARCHY (user's call, 2026-08): an ALLOW only takes effect if the
        SOURCE-level ancestor (the 2-segment ``db:<source>`` prefix) is itself
        allowed — the source is the gate. A grant deeper in the tree
        (``db:src:table``) with no source-level allow above it grants NOTHING.
        This makes the model "allow the source, then refine DOWN with denies",
        NOT "allow-list individual tables from nothing". DENY is unchanged:
        deny-wins at any depth, unpierceable.

        Consistency: ``apps.query.scope.permitted_source_ids`` (the coarse
        source gate) and ``apps.access_management.services.catalog._resolve_effect``
        (the admin-tree display) implement the SAME rule — change all three
        together or they drift and the tree lies about effective access.
        """
        matched = self._matching(permission_code, resource_path)
        if any(grant.effect == Effect.DENY for grant in matched):
            return False
        if not resource_path:
            # A non-resource-scoped permission (user.manage, query.execute, ...):
            # there is no hierarchy to gate, any global ALLOW suffices. Unchanged.
            return any(grant.effect == Effect.ALLOW for grant in matched)
        # Resource-scoped: the source-level ancestor must be explicitly allowed.
        try:
            source_prefix = rp.prefixes(resource_path)[0]
        except (rp.InvalidResourcePath, IndexError):
            return False
        return any(grant.resource_path == source_prefix and grant.effect == Effect.ALLOW
                   for grant in matched)

    def denies(self, permission_code: str, resource_path: str = "") -> bool:
        """Whether an explicit DENY matches.

        Distinct from ``not allows(...)``: that is also true when nothing was granted
        at all. Separating them lets an admin screen show "explicitly denied" apart
        from "never granted", which are very different things to an operator.
        """
        return any(grant.effect == Effect.DENY
                   for grant in self._matching(permission_code, resource_path))

    def resources_for(self, permission_code: str) -> tuple[str, ...]:
        """Every resource path explicitly ALLOWED for a permission, broadest first.

        Not an expansion into concrete resources: a grant on ``db:crm`` is returned as
        ``db:crm``, not as every table beneath it. Callers that need the closure ask
        the catalog — this object only knows what was granted.
        """
        allowed = [g.resource_path for g in self._by_code.get(permission_code, ())
                   if g.effect == Effect.ALLOW]
        return tuple(sorted(set(allowed), key=lambda p: (p.count(rp.SEPARATOR), p)))

    @property
    def permission_codes(self) -> tuple[str, ...]:
        """Every permission code with at least one ALLOW, sorted.

        Convenience for a UI that only needs "can this user do X at all". A code
        appears here even if a specific resource is denied, so it must NOT be used as
        the authorization decision — that is what ``allows`` is for.
        """
        return tuple(sorted(
            code for code, grants in self._by_code.items()
            if any(g.effect == Effect.ALLOW for g in grants)))

    def as_dict(self) -> dict:
        """A JSON-safe projection, for an API response or a cache entry."""
        return {
            "user_id": self.user_id,
            "permissions": [
                {"permission_code": g.permission_code,
                 "resource_path": g.resource_path,
                 "effect": g.effect}
                for g in self.grants
            ],
        }

    def _matching(self, permission_code: str, resource_path: str) -> tuple[Grant, ...]:
        """The grants that bear on one question.

        A blank ``resource_path`` matches only blank-path grants; a real path matches
        grants at any of its prefixes. An unaddressable path matches nothing — it
        names no resource, so it can be granted nothing (fail closed).
        """
        candidates = self._by_code.get(permission_code, ())
        if not candidates:
            return ()

        if not resource_path:
            return tuple(g for g in candidates if g.is_global)

        try:
            scope = frozenset(rp.prefixes(resource_path))
        except rp.InvalidResourcePath:
            return ()
        return tuple(g for g in candidates if g.resource_path in scope)


#: The answer for a user who can do nothing. A single shared value rather than a fresh
#: empty object per call — and the thing every failure path returns, so "something went
#: wrong" and "denied" are the same outcome.
NO_PERMISSIONS = EffectivePermissions(
    user_id=0, grants=(), _by_code=MappingProxyType({}))


class PermissionResolver:
    """Reads the RBAC graph and returns an effective permission set.

    Stateless and side-effect free: it never writes, never caches, and never decides
    what to do with the answer. ``request`` is accepted for symmetry with the other
    services.
    """

    def __init__(self, request=None):
        self._request = request

    def resolve(self, user) -> EffectivePermissions:
        """Everything ``user`` may do, right now.

        Args:
            user: A user instance, or None.

        Returns:
            ``EffectivePermissions`` — empty for an anonymous, missing or **inactive**
            user. A deactivated account resolves to nothing, so disabling a user takes
            effect at the next resolution without touching their grants.

        One query. Inactive roles and permissions are excluded in the database, not in
        Python, so a disabled capability cannot slip through a caller that forgot to
        filter.
        """
        user_id = getattr(user, "pk", None)
        if user_id is None or not getattr(user, "is_active", False):
            return NO_PERMISSIONS

        return self.resolve_for_user_id(user_id)

    def resolve_for_user_id(self, user_id: int) -> EffectivePermissions:
        """As ``resolve``, addressed by id.

        Used by the admin endpoint, which asks about *another* user and has already
        confirmed they exist. Does NOT re-check ``is_active`` — the caller decides
        whether it wants "what would this user get if enabled" (an admin screen does)
        or "what does this user get" (``resolve`` does).
        """
        rows = (RolePermission.objects
                .filter(role__user_assignments__user_id=user_id,
                        role__is_active=True,
                        permission__is_active=True)
                .values_list("permission__code", "resource_path", "effect"))

        grants = tuple(
            Grant(permission_code=code, resource_path=path, effect=effect)
            for code, path, effect in rows
        )
        return self._freeze(user_id, grants)

    @staticmethod
    def _freeze(user_id: int, grants: tuple[Grant, ...]) -> EffectivePermissions:
        """Build the per-code index once, then seal the whole thing."""
        by_code: dict[str, list[Grant]] = {}
        for grant in grants:
            by_code.setdefault(grant.permission_code, []).append(grant)

        return EffectivePermissions(
            user_id=user_id,
            grants=grants,
            _by_code=MappingProxyType(
                {code: tuple(items) for code, items in by_code.items()}),
        )
