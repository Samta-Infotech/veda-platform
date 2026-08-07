"""The "at least one Admin, always" invariant — one guard, every place that could
break it.

WHAT "ADMIN" MEANS HERE
    Two things, deliberately kept in step rather than picking one:

      * ``User.is_staff`` — the flag that ACTUALLY gates every admin endpoint today
        (``AdminView.permission_classes`` includes ``IsAdminUser``, which checks
        exactly this). This is the operative definition.
      * The RBAC ``Role`` named "Admin" (seeded by migration 0007, granted every
        permission that exists) — decorative until ``VEDA_RBAC_MODE=enforce``, but
        assigned alongside ``is_staff`` at bootstrap so the platform is not left with
        zero holders of either the moment RBAC enforcement does turn on.

    Guarding only ``is_staff`` and letting the Admin role assignment be stripped
    freely would leave a real admin (by access) with no RBAC record of ever having
    been one — correct today, silently wrong the day Gate 2 flips to enforce.

WHY NOT A DB CONSTRAINT
    "At least one row where is_staff=True" is not expressible as a column
    constraint — it is a property of the whole table, checked at the moment of a
    write that could reduce the count. Application-level invariants like this one
    are exactly what a guard function centralises: every mutation that could violate
    it calls the same predicate, so there is one definition of "last admin" instead
    of one per call site re-deriving it slightly differently.
"""
from __future__ import annotations

from django.contrib.auth import get_user_model
from django.db.models import QuerySet

from apps.core.messages import MESSAGES

from ..models import Role
from .base import ConflictError

#: Natural key the Admin role is looked up by everywhere at runtime (bootstrap,
#: this guard). Migration 0007 owns its OWN copy of this literal — migrations must
#: not import app code that can change shape after they are written.
ADMIN_ROLE_NAME = "Admin"

CODE_LAST_ADMIN_PROTECTED = "LAST_ADMIN_PROTECTED"
CODE_LAST_ADMIN_ROLE_PROTECTED = "LAST_ADMIN_ROLE_PROTECTED"


class LastAdminProtected(ConflictError):
    """Refused: this write would leave the platform with zero active admins."""

    code = CODE_LAST_ADMIN_PROTECTED
    message = MESSAGES["user"]["last_admin_protected"]


class LastAdminRoleProtected(ConflictError):
    """Refused: this would strip the Admin role from its last active holder."""

    code = CODE_LAST_ADMIN_ROLE_PROTECTED
    message = MESSAGES["grant"]["last_admin_role_protected"]


def active_admins() -> QuerySet:
    """Every currently-active, currently-staff user — the set "last admin" counts."""
    return get_user_model().objects.filter(is_staff=True, is_active=True)


def is_last_active_admin(user, *, exclude_pk=None) -> bool:
    """Whether ``user`` is the only entry in ``active_admins()``.

    ``exclude_pk`` lets a caller ask "if I removed this row, would ``user`` still
    not be the last one" without first performing the removal — used by the role-
    revoke guard, where the row being tested for removal is the assignment, not the
    user themselves.

    A user who is not currently an active admin can never be "the last" one by
    definition — asking whether to protect them from a change that does not concern
    admin status at all is a meaningless question, so this returns False rather than
    raising.
    """
    if not (user.is_staff and user.is_active):
        return False
    others = active_admins().exclude(pk=user.pk)
    if exclude_pk is not None:
        others = others.exclude(pk=exclude_pk)
    return not others.exists()


def get_admin_role() -> Role:
    """The seeded "Admin" role.

    Raises ``Role.DoesNotExist`` if migration 0007 has not been applied — a
    deployment/config problem, not a request-time failure any endpoint should
    translate into a client-facing error.
    """
    return Role.objects.get(name__iexact=ADMIN_ROLE_NAME)
