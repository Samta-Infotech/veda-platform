"""Granting business logic — role assignment and permission grants.

Two services, one module, mirroring ``models/grants.py``: the same concept
(an audited edge) with the same idempotency rules.

IDEMPOTENCY IS THE CONTRACT
    Both operations describe a desired *state*, not an event:

      assign   -> "this user holds this role"       (repeat = same state, success)
      revoke   -> "this user does not hold it"      (repeat = same state, success)

    So neither is an error when repeated. An administrator scripting "make sure alice
    is an analyst" must be able to run it twice. The response distinguishes *created*
    from *already present* so a UI can still say something useful, but both are 2xx.

    This differs deliberately from ``users/create`` and ``roles/create``, which 409 on
    a duplicate: creating implies newness, granting implies membership.

RE-GRANTING WITH THE OPPOSITE EFFECT UPDATES, IT DOES NOT DUPLICATE
    ``(role, permission, resource_path)`` is unique *without* ``effect``. Granting
    ALLOW where a DENY exists flips the existing decision. The alternative — two rows
    disagreeing about the same triple — would make the outcome depend on row order,
    which is a non-deterministic authorization result and the worst possible bug class
    here.

NOTHING IS ENFORCED BY THESE ROWS YET.
"""
from __future__ import annotations

import logging

from django.contrib.auth import get_user_model
from django.db import transaction

from .. import resource_path as rp
from apps.core.messages import MESSAGES

from ..models import CatalogResource, Effect, Permission, Role, RolePermission, UserRole
from .base import ConflictError, paginate
from .permissions import PermissionNotFound
from .roles import RoleNotFound
from .users import UserNotFound

logger = logging.getLogger(__name__)

CODE_ROLE_INACTIVE = "ROLE_INACTIVE"
CODE_PERMISSION_INACTIVE = "PERMISSION_INACTIVE"
CODE_INVALID_RESOURCE = "INVALID_RESOURCE_PATH"

USER_ROLE_LIST_FIELDS = ("id", "user_id", "role_id", "granted_by_id",
                         "created_at", "updated_at")
ROLE_PERMISSION_LIST_FIELDS = ("id", "role_id", "permission_id", "resource_path",
                               "effect", "granted_by_id", "created_at", "updated_at")


class RoleInactive(ConflictError):
    """Assigning a retired role would confer authority that is switched off.

    Rejected rather than allowed-but-inert: silently assigning something that grants
    nothing is the same "authority that does not exist" problem the permission
    catalogue is read-only to avoid.
    """

    code = CODE_ROLE_INACTIVE
    message = MESSAGES["grant"]["role_inactive"]


class PermissionInactive(ConflictError):
    """Granting a disabled permission — same reasoning as ``RoleInactive``."""

    code = CODE_PERMISSION_INACTIVE
    message = MESSAGES["grant"]["permission_inactive"]


class InvalidResourcePath(ConflictError):
    """The resource path is not expressible in the canonical grammar.

    A grant stored in a non-canonical shape would never match the resolver's lookups —
    it would look granted in the admin UI and deny in practice.
    """

    code = CODE_INVALID_RESOURCE
    message = MESSAGES["grant"]["invalid_resource"]


class _GrantServiceBase:
    """Shared plumbing: the actor, the audit suffix, and typed lookups.

    Lookups raise the SAME typed errors the user/role/permission services already
    define (``UserNotFound``, ``RoleNotFound``, ``PermissionNotFound``), so a missing
    target produces one 404 shape across the whole app rather than a new one per
    endpoint.
    """

    def __init__(self, request=None):
        self._request = request

    @property
    def _actor(self):
        user = getattr(self._request, "user", None)
        return user if (user is not None and user.is_authenticated) else None

    def _log_context(self) -> str:
        actor = getattr(self._actor, "pk", None)
        return (f"request_id={getattr(self._request, 'request_id', '')} "
                f"actor={actor if actor is not None else '-'}")

    @staticmethod
    def _get_user(user_id):
        user = get_user_model().objects.filter(pk=user_id).first()
        if user is None:
            raise UserNotFound()
        return user

    @staticmethod
    def _get_role(role_id, *, require_active: bool):
        role = Role.objects.filter(pk=role_id).first()
        if role is None:
            raise RoleNotFound()
        if require_active and not role.is_active:
            raise RoleInactive()
        return role

    @staticmethod
    def _get_permission(permission_id, *, require_active: bool):
        permission = Permission.objects.filter(pk=permission_id).first()
        if permission is None:
            raise PermissionNotFound()
        if require_active and not permission.is_active:
            raise PermissionInactive()
        return permission


class UserRoleService(_GrantServiceBase):
    """Assign and revoke roles."""

    def assign(self, *, user_id: int, role_id: int) -> tuple[UserRole, bool]:
        """Ensure a user holds a role.

        Returns:
            ``(assignment, created)`` — ``created`` False when it already held.

        Raises:
            UserNotFound / RoleNotFound: unknown target.
            RoleInactive: the role is retired.

        A retired role is refused, but an *inactive user* is not: pre-provisioning
        access for an account that is not yet enabled is a legitimate workflow, and
        the assignment grants nothing until the account is active anyway.
        """
        user = self._get_user(user_id)
        role = self._get_role(role_id, require_active=True)

        with transaction.atomic():
            # get_or_create, not check-then-insert: the unique constraint arbitrates,
            # so two concurrent assignments cannot both insert.
            assignment, created = UserRole.objects.get_or_create(
                user=user, role=role, defaults={"granted_by": self._actor})

        logger.info("role %s user_id=%s role_id=%s %s",
                    "assigned" if created else "already assigned",
                    user_id, role_id, self._log_context())
        return assignment, created

    def revoke(self, *, user_id: int, role_id: int) -> bool:
        """Ensure a user does not hold a role. Idempotent.

        Returns:
            True if an assignment was removed, False if there was nothing to remove.

        Deliberately does NOT verify the user or role exists: the desired end state —
        "this user does not hold this role" — is already true for a target that does
        not exist, and raising 404 would make a revoke script fail on exactly the rows
        it has nothing to do.
        """
        deleted, _ = UserRole.objects.filter(user_id=user_id, role_id=role_id).delete()
        removed = bool(deleted)
        logger.info("role %s user_id=%s role_id=%s %s",
                    "revoked" if removed else "revoke no-op",
                    user_id, role_id, self._log_context())
        return removed

    def list_assignments(self, *, page: int, page_size: int, search: str = "",
                         is_active=None, ordering: str = "id",
                         user_id=None, role_id=None) -> tuple[list, int]:
        """One page of assignments, optionally narrowed to a user or a role.

        ``search``/``is_active`` come from the shared paging contract and are not
        meaningful for an edge with no name and no active flag; they are accepted and
        ignored rather than rejected, so one list contract holds across the app.
        """
        queryset = UserRole.objects.all()
        if user_id is not None:
            queryset = queryset.filter(user_id=user_id)
        if role_id is not None:
            queryset = queryset.filter(role_id=role_id)

        return paginate(queryset, page=page, page_size=page_size, ordering=ordering,
                        only_fields=USER_ROLE_LIST_FIELDS)


class RolePermissionService(_GrantServiceBase):
    """Grant and revoke permissions on resources."""

    def grant(self, *, role_id: int, permission_id: int, resource_path: str = "",
              effect: str = Effect.ALLOW) -> tuple[RolePermission, bool]:
        """Ensure a role has the given decision for a permission on a resource.

        Returns:
            ``(grant, created)`` — ``created`` False when the row already existed
            (its ``effect`` is then updated in place, never duplicated).

        Raises:
            RoleNotFound / PermissionNotFound: unknown target.
            RoleInactive / PermissionInactive: the target is switched off.
            InvalidResourcePath: the path is not canonical.

        ``resource_path=""`` grants the permission globally, which is correct for
        capabilities that are not resource-scoped (``user.manage``).
        """
        role = self._get_role(role_id, require_active=True)
        permission = self._get_permission(permission_id, require_active=True)
        path = self._canonical_path(resource_path)

        with transaction.atomic():
            grant, created = RolePermission.objects.update_or_create(
                role=role, permission=permission, resource_path=path,
                defaults={"effect": effect, "granted_by": self._actor})

        logger.info("permission %s role_id=%s permission_id=%s effect=%s path=%s %s",
                    "granted" if created else "grant updated",
                    role_id, permission_id, effect, path or "(global)",
                    self._log_context())
        return grant, created

    def revoke(self, *, role_id: int, permission_id: int,
               resource_path: str = "") -> bool:
        """Remove a decision. Idempotent, and does not 404 on unknown targets —
        same reasoning as ``UserRoleService.revoke``.

        Removing a DENY does not create an ALLOW: with nothing left matching, the
        default-deny in ADR §3.5 applies.
        """
        path = self._canonical_path(resource_path)
        deleted, _ = RolePermission.objects.filter(
            role_id=role_id, permission_id=permission_id, resource_path=path).delete()
        removed = bool(deleted)
        logger.info("permission %s role_id=%s permission_id=%s path=%s %s",
                    "revoked" if removed else "revoke no-op",
                    role_id, permission_id, path or "(global)", self._log_context())
        return removed

    def list_grants(self, *, page: int, page_size: int, search: str = "",
                    is_active=None, ordering: str = "id",
                    role_id=None, permission_id=None,
                    resource_path=None) -> tuple[list, int]:
        """One page of grants, optionally narrowed."""
        queryset = RolePermission.objects.all()
        if role_id is not None:
            queryset = queryset.filter(role_id=role_id)
        if permission_id is not None:
            queryset = queryset.filter(permission_id=permission_id)
        if resource_path is not None:
            queryset = queryset.filter(resource_path=self._canonical_path(resource_path))

        return paginate(queryset, page=page, page_size=page_size, ordering=ordering,
                        only_fields=ROLE_PERMISSION_LIST_FIELDS)

    @staticmethod
    def known_resource_paths(paths) -> set[str]:
        """Which of these paths the catalog currently knows and has active.

        ONE query for a whole page, so rendering "is this grant pointing at something
        real?" costs no N+1. A grant on an unknown path is legal — pre-provisioning is
        deliberate (see ``models/grants.py``) — but an admin UI needs to show it.
        """
        wanted = {p for p in paths if p}
        if not wanted:
            return set()
        return set(CatalogResource.objects
                   .filter(path__in=wanted, is_active=True)
                   .values_list("path", flat=True))

    @staticmethod
    def _canonical_path(resource_path: str) -> str:
        """"" stays "" (a global grant); anything else must be canonical.

        Validated here rather than only in the serializer because the service is also
        called directly — and a non-canonical path stored once would look granted in
        the UI while never matching a resolver lookup.
        """
        if not resource_path:
            return ""
        try:
            return rp.validate(resource_path)
        except rp.InvalidResourcePath as exc:
            raise InvalidResourcePath() from exc
