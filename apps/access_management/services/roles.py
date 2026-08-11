"""Role administration business logic.

Views stay thin: validate a body, call in here, render the outcome. Everything that
decides whether a role can be created or changed lives in this module.

Mirrors ``services/users.py`` deliberately — same typed-error contract, same
"let the database arbitrate uniqueness" stance, same transaction discipline. Two
domains that behave the same way are easier to reason about than two that each
invented their own approach.

Deliberately absent: anything about permissions or assignment. A role is a named
record here and nothing more, until those phases land.
"""
from __future__ import annotations

import logging

from django.db import IntegrityError, transaction
from django.db.models import Q
from django.utils import timezone

from apps.core.messages import MESSAGES

from ..models import Role
from .base import AccessManagementError, ConflictError, NotFoundError, paginate
from .. import resource_path as rp
from ..codes import PermissionCode
from ..models import Effect, Permission, RolePermission
logger = logging.getLogger(__name__)

CODE_ROLE_NAME_TAKEN = "ROLE_NAME_TAKEN"
CODE_ROLE_NOT_FOUND = "ROLE_NOT_FOUND"
CODE_INVALID_GRANT = "INVALID_GRANT"

#: Exactly the columns ``views/roles.py::public_fields`` renders. Passed to
#: ``.only()`` so the set fetched and the set projected cannot drift: adding a column
#: to the response without adding it here would cost a deferred-field query per row.
ROLE_LIST_FIELDS = ("id", "name", "description", "is_active", "created_at", "updated_at",
                    "deleted_at")


class RoleNotFound(NotFoundError):
    """No role with that id. Inherits its 404 from ``NotFoundError``."""

    code = CODE_ROLE_NOT_FOUND
    message = MESSAGES["role"]["not_found"]


class RoleNameTaken(ConflictError):
    """The name is already used by another role, compared case-insensitively.

    Inherits its 409 from ``ConflictError``.
    """

    code = CODE_ROLE_NAME_TAKEN
    message = MESSAGES["role"]["name_taken"]


class InvalidGrant(AccessManagementError):
    """``permission_ids``/``resource_grants`` named something that cannot be
    granted — an unknown permission id, an unaddressable resource path, or an
    effect that is neither ``allow`` nor ``deny``.

    Rendered as 400 (the ``AccessManagementError`` fallback status): the request
    itself was malformed, not a conflict with existing state and not a missing
    record. The originating ``ValueError`` detail from ``_sync_grants`` is logged,
    never rendered — same doctrine as every other typed error here, so a caller
    cannot fish debugging detail out of the response.
    """

    code = CODE_INVALID_GRANT
    message = MESSAGES["role"]["invalid_grant"]


class RoleService:
    """Role administration for one request.

    ``request`` is optional so the service is usable (and testable) outside HTTP; it
    is read only to tag log lines with the ambient request id and the acting admin.
    """

    def __init__(self, request=None):
        self._request = request

    def create_role(self, *, name: str, description: str = "",
                    permission_ids: list[int] | None = None,
                    resource_grants: list[dict] | None = None) -> Role:
        """Create one active role with optional permissions and resource grants.

        Raises:
            RoleNameTaken: another role already holds ``name``.
            InvalidGrant: ``permission_ids``/``resource_grants`` named something
                that cannot be granted — see ``_sync_grants``. The role is not
                created; the whole call is one transaction.
        """
        try:
            with transaction.atomic():
                role = Role.objects.create(name=name, description=description)
                self._sync_grants(role, permission_ids, resource_grants)
        except IntegrityError as exc:
            conflict = self._classify_conflict(name)
            if conflict is None:
                logger.exception("role creation failed on an unattributable integrity "
                                 "error name=%s %s", name, self._log_context())
                raise
            raise conflict from exc
        except ValueError as exc:
            logger.warning("role creation rejected an invalid grant name=%s %s: %s",
                           name, self._log_context(), exc)
            raise InvalidGrant() from exc

        logger.info("role created role_id=%s name=%s %s",
                    role.pk, role.name, self._log_context())
        return role

    def list_roles(self, *, page: int, page_size: int, search: str = "",
                   is_active=None, ordering: str = "name") -> tuple[list, int]:
        """One page of roles, plus the total matching count.

        Owns only what "search" and "active" MEAN for a role; the paging mechanics are
        shared with every other list endpoint in the app (``base.paginate``).

        Args:
            search: Case-insensitive substring matched against name OR description.
            is_active: Tri-state — None means "no filter", not "False".
        """
        queryset = Role.objects.all()
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) | Q(description__icontains=search))
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active)

        return paginate(queryset, page=page, page_size=page_size, ordering=ordering,
                        only_fields=ROLE_LIST_FIELDS)

    def list_active_roles(self) -> list[Role]:
        """Every active role — id and name only. For a picker/dropdown, not the
        admin table.
        """
        return list(Role.objects.filter(is_active=True).order_by("name")
                    .only("id", "name"))

    def get_role(self, role_id: int) -> Role:
        """One role by id.

        Raises:
            RoleNotFound: no role with that id.
        """
        role = Role.objects.filter(pk=role_id).only(*ROLE_LIST_FIELDS).first()
        if role is None:
            raise RoleNotFound()
        return role

    def update_role(self, role_id: int, **fields) -> Role:
        """Apply changes to one role.

        Raises:
            RoleNotFound: no role with ``role_id``.
            RoleNameTaken: the new name belongs to a different role.
            InvalidGrant: ``permission_ids``/``resource_grants`` named something
                that cannot be granted — see ``_sync_grants``. Nothing is
                written; the whole call is one transaction.
        """
        permission_ids = fields.pop("permission_ids", None)
        resource_grants = fields.pop("resource_grants", None)

        if not fields and permission_ids is None and resource_grants is None:
            # the serializer rejects this too; belt-and-braces for direct calls
            raise ValueError("update_role requires at least one field")

        try:
            with transaction.atomic():
                role = Role.objects.select_for_update().filter(pk=role_id).first()
                if role is None:
                    raise RoleNotFound()

                if fields:
                    touched = list(fields)
                    if "is_active" in fields and fields["is_active"] != role.is_active:
                        role.deleted_at = None if fields["is_active"] else timezone.now()
                        touched.append("deleted_at")
                    for name, value in fields.items():
                        setattr(role, name, value)
                    role.save(update_fields=[*touched, "updated_at"])
                elif permission_ids is not None or resource_grants is not None:
                    # No plain field changed, but a grants-only request still changes
                    # the role — updated_at must reflect that too, or "last updated"
                    # sorting/display silently lies about the most recent write.
                    role.save(update_fields=["updated_at"])

                self._sync_grants(role, permission_ids, resource_grants)
        except IntegrityError as exc:
            conflict = self._classify_conflict(fields.get("name", ""), exclude_pk=role_id)
            if conflict is None:
                logger.exception("role update failed on an unattributable integrity "
                                 "error role_id=%s %s", role_id, self._log_context())
                raise
            raise conflict from exc
        except ValueError as exc:
            logger.warning("role update rejected an invalid grant role_id=%s %s: %s",
                           role_id, self._log_context(), exc)
            raise InvalidGrant() from exc

        logger.info("role updated role_id=%s %s", role.pk, self._log_context())
        return role

    def _sync_grants(self, role: Role, permission_ids: list[int] | None,
                     resource_grants: list[dict] | None) -> None:
        """Synchronize global system permissions and resource-level ``data.read``
        grants for ``role``.

        Both ``permission_ids`` and ``resource_grants`` are a FULL desired-state
        replacement, not an add-only patch: a permission/path that existed before
        but is absent from the new list is revoked. ``None`` (the field was
        omitted from the request) means "leave these grants untouched" — the one
        way to distinguish "sync to nothing" (an explicit ``[]``) from "don't
        touch this at all".

        Raises:
            ValueError: an unknown ``permission_id``, a resource path that fails
                canonicalization, or an ``effect`` that is neither ``allow`` nor
                ``deny`` — fails loudly rather than silently dropping/misapplying
                a caller's grant.
        """


        actor = getattr(self._request, "user", None)
        if actor is not None and not getattr(actor, "is_authenticated", False):
            actor = None

        # 1. Sync global system permissions (resource_path="") — full replace.
        if permission_ids is not None:
            deduped_ids = list(dict.fromkeys(permission_ids))
            valid_perms = {p.id: p for p in Permission.objects.filter(id__in=deduped_ids)}
            unknown = [pid for pid in deduped_ids if pid not in valid_perms]
            if unknown:
                raise ValueError(f"unknown permission_ids: {unknown}")

            RolePermission.objects.filter(role=role, resource_path="").delete()
            to_create = [
                RolePermission(role=role, permission=valid_perms[pid], resource_path="",
                               effect=Effect.ALLOW, granted_by=actor)
                for pid in deduped_ids
            ]
            if to_create:
                try:
                    RolePermission.objects.bulk_create(to_create)
                except IntegrityError as exc:
                    # Only the (role, permission, resource_path) constraint applies
                    # here — a concurrent sync for the same role lost the race. Not a
                    # role-name conflict, so it must not reach _classify_conflict,
                    # which only knows how to attribute THAT constraint; surfacing it
                    # as ValueError routes it through the caller's own translation
                    # into a client-safe InvalidGrant instead of an unattributed 500.
                    raise ValueError(
                        "permission_ids sync collided with a concurrent change to "
                        "this role — retry the request") from exc

        # 2. Sync resource-level data.read grants — also a full replace, mirroring
        # the global-permission sync above (an admin removing a row in the UI
        # must actually revoke it, not just fail to add a duplicate).
        if resource_grants is not None:
            data_read_perm = Permission.objects.filter(code=PermissionCode.DATA_READ).first()
            if data_read_perm is None:
                # Never fall back to "whichever permission happens to be first" —
                # that would silently grant/deny an unrelated system permission
                # (e.g. user.manage) on a data resource path, which is meaningless
                # and dangerous. A missing seeded permission is a deployment bug
                # that must fail loudly, not misapply grants.
                raise RuntimeError(
                    f"the {PermissionCode.DATA_READ!r} permission is not seeded — "
                    "cannot sync resource grants")

            canonical: dict[str, str] = {}
            for grant in resource_grants:
                raw_path = (grant.get("resource_path") or "").strip()
                if not raw_path:
                    continue
                try:
                    res_path = rp.validate(raw_path)
                except rp.InvalidResourcePath as exc:
                    raise ValueError(f"invalid resource_path {raw_path!r}: {exc}") from exc

                eff_raw = (grant.get("effect") or "allow").strip().lower()
                if eff_raw not in ("allow", "deny"):
                    raise ValueError(
                        f"invalid effect {grant.get('effect')!r} for {raw_path!r} — "
                        "must be 'allow' or 'deny'")
                canonical[res_path] = Effect.DENY if eff_raw == "deny" else Effect.ALLOW

            # Revoke any existing resource-scoped data.read grant not in the new
            # set — global (resource_path="") data.read grants are untouched;
            # those belong to the permission_ids sync above, not this one.
            (RolePermission.objects.filter(role=role, permission=data_read_perm)
             .exclude(resource_path="")
             .exclude(resource_path__in=canonical)
             .delete())

            for res_path, effect in canonical.items():
                try:
                    RolePermission.objects.update_or_create(
                        role=role, permission=data_read_perm, resource_path=res_path,
                        defaults={"effect": effect, "granted_by": actor})
                except IntegrityError as exc:
                    # Same reasoning as the permission_ids sync above: a concurrent
                    # writer raced this exact (role, permission, resource_path) row.
                    raise ValueError(
                        f"resource_grants sync collided with a concurrent change to "
                        f"{res_path!r} on this role — retry the request") from exc


    @staticmethod
    def _classify_conflict(name: str,
                           exclude_pk: int | None = None) -> ConflictError | None:
        """Which uniqueness rule the write violated, or None if it was not one.

        A pure classifier — it queries and returns, it never raises. The caller owns
        the control flow, matching ``UserService._classify_conflict`` exactly; a
        function named "classify" that secretly re-raises is a hidden side effect.

        Determined by querying, not by parsing the driver's error text: constraint
        names and message formats differ between Postgres and sqlite, and a string
        match that silently stopped matching would misreport every conflict. The query
        runs ONLY on the failure path, so the happy path pays nothing.

        ``exclude_pk`` is the row being updated: without it, a role keeping its own
        name would look like a conflict with itself.

        Returns None when the conflict cannot be attributed: an unexplained constraint
        failure is a server problem, and reporting it as "name already taken" would
        send an administrator chasing a role that is not there.
        """
        others = Role.objects.all()
        if exclude_pk is not None:
            others = others.exclude(pk=exclude_pk)
        if name and others.filter(name__iexact=name).exists():
            return RoleNameTaken()
        return None

    def _log_context(self) -> str:
        """``request_id``/``actor`` suffix for the audit line — who changed what.

        There is no queryable audit trail yet (tracked as M8 in
        ``AUTH_ISSUES_BACKLOG.md``); when one lands, this is a call site that should
        write to it.
        """
        actor = getattr(getattr(self._request, "user", None), "pk", None)
        return (f"request_id={getattr(self._request, 'request_id', '')} "
                f"actor={actor if actor is not None else '-'}")
