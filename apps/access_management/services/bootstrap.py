"""First-Admin bootstrap — backend-only, never a public endpoint.

WHY THIS CANNOT BE A REGULAR ``users/create`` CALL
    ``UserCreateView`` is staff-only (``IsAdminUser``). On a fresh install there is
    no staff user yet, so nothing can ever call it — the classic bootstrap paradox.
    The fix is NOT to relax that endpoint's auth for "the zero-users case": that
    would make it a conditional public self-registration endpoint in disguise, which
    is exactly the surface this platform has deliberately never built. Bootstrap
    instead runs from a place a network request can never reach:
    ``manage.py bootstrap_admin`` (see ``management/commands/bootstrap_admin.py``).

WHAT THE FIRST ADMIN GETS, AND WHY BOTH
    ``is_staff=True`` — the flag that actually gates every admin endpoint today.
    The RBAC "Admin" role (seeded by migration 0007, granted every permission that
    exists) — decorative until RBAC enforcement is on, but assigned now so the
    platform is never left with zero holders of it the day enforcement does turn on.

RACE SAFETY
    "If no users exist, become admin" is a classic check-then-act race: two
    concurrent invocations can both observe zero users before either commits, and
    both then create an admin. Fixed the same way every other race in this app is —
    not with a new lock table, but by taking a row lock on something that already,
    deterministically exists: the seeded Admin role. ``select_for_update()`` on that
    one row inside a single transaction means the second concurrent call blocks
    until the first commits, then re-reads "does any user exist" and correctly sees
    the first admin — no window where both succeed.

    ``select_for_update`` is a Postgres-only guarantee (SQLite silently no-ops it,
    same caveat as ``RoleService.update_role``) — the concurrency test in
    ``tests/test_admin_bootstrap.py`` runs against Postgres for this reason; the
    local SQLite suite exercises the code path but does not prove the lock.
"""
from __future__ import annotations

import logging

from django.contrib.auth import get_user_model
from django.db import transaction

from apps.core.messages import MESSAGES

from ..models import Role, UserRole
from .admin_guard import ADMIN_ROLE_NAME
from .base import ConflictError
from .users import UserService

logger = logging.getLogger(__name__)


class AlreadyBootstrapped(ConflictError):
    """Refused: at least one user already exists. Bootstrap is a one-time action.

    No importable ``CODE_*`` constant — nothing outside this class currently needs
    to reference the code by name (unlike ``CODE_ROLE_NOT_FOUND`` etc., which tests
    assert on directly). Add one back the moment something does.
    """

    code = "ALREADY_BOOTSTRAPPED"
    message = MESSAGES["auth"]["already_bootstrapped"]


class AdminBootstrapService:
    """Creates the platform's first administrator. Backend-only — see module
    docstring for why this is never wired to an HTTP route."""

    def __init__(self, request=None):
        self._request = request

    def bootstrap(self, *, username: str, email: str, password: str,
                  first_name: str = "", last_name: str = ""):
        """Create the first user, made an admin unconditionally.

        Returns:
            The created ``User``.

        Raises:
            AlreadyBootstrapped: at least one user already exists.
            UsernameTaken / EmailTaken: reused from ``UserService`` — vanishingly
                unlikely against an empty table, but the table is not locked, only
                the Admin role row is, so a name collision is still theoretically
                reachable and must still be reported precisely, not as a 500.
        """
        with transaction.atomic():
            admin_role = Role.objects.select_for_update().get(name__iexact=ADMIN_ROLE_NAME)
            if get_user_model().objects.exists():
                raise AlreadyBootstrapped()

            user = UserService(self._request).create_user(
                username=username, email=email, password=password,
                first_name=first_name, last_name=last_name)
            user.is_staff = True
            user.save(update_fields=["is_staff"])
            UserRole.objects.create(user=user, role=admin_role, granted_by=None)

        logger.warning("admin bootstrap completed user_id=%s username=%s",
                       user.pk, user.username)
        return user
