"""The ``Role`` aggregate — a named bundle of authority.

PURPOSE
    A role is the unit an administrator grants. Today it is only a named, describable
    record; permissions attach to it in a later phase, and users attach to it in the
    phase after that. It exists now so that role administration can be built, audited
    and reviewed before anything depends on it.

WHY NOT ``django.contrib.auth.Group``
    Group carries exactly two things: a unique ``name`` and an M2M to
    ``auth.Permission``. It has no description, no active flag and no timestamps, so
    an administration UI would need a parallel profile table beside it — two tables
    for one concept. More decisively, ``auth.Permission`` is *model-level*
    (add/change/delete/view per Django model), while VEDA's resources are data
    sources, schemas, tables and columns. Binding roles to Group would leave
    ``Group.permissions`` present, unused and misleading for every future reader.

RESPONSIBILITY
    Identity and lifecycle of a role, nothing else. No permission logic, no
    assignment logic, no authorization decisions — those belong to their own models
    and to the resolver, and putting any of them here would make this the god-object
    of the RBAC domain.

EXTENSION POINTS (deliberately not built)
    ``UserRole`` and ``RolePermission`` will foreign-key to ``Role.id``. That is why
    a role is *deactivated* rather than deleted (see below): those rows must keep a
    valid target, and an audit trail that says "granted role #7" must still be able
    to resolve #7 years later. No ``is_system``, no tenant column, no permission
    field is present — each would be a guess about a design still under discussion.

SECURITY
    ``is_active`` is authorization-relevant the moment the resolver exists: an
    inactive role must grant nothing. It is modelled here, and enforced there —
    modelling it now costs nothing and avoids a migration on a live table later.

PERFORMANCE
    Role cardinality is tens-to-hundreds, not millions. The one index is the
    case-insensitive unique constraint on ``name``, which the create/update conflict
    checks also use for lookups. An index on ``is_active`` is deliberately NOT added:
    on a table this small it would cost writes and save no measurable read time, and
    a filtered scan of a few hundred rows is free.
"""
from __future__ import annotations

from django.db import models
from django.db.models.functions import Lower

from apps.core.models import TimeStampedModel


class Role(TimeStampedModel):
    """A named bundle of authority. See the module docstring for the design notes."""

    #: Human-facing, and the natural key an administrator thinks in. Unique
    #: case-insensitively — "Admin" and "admin" are the same role, and allowing both
    #: would make grants ambiguous to every human who reads them.
    name = models.CharField(max_length=150)

    #: What the role is for, in the administrator's words. The field
    #: ``auth.Group`` cannot carry, and the main reason this model exists.
    description = models.TextField(blank=True)

    #: Retirement, not deletion. A role that is no longer granted stays queryable so
    #: history remains readable and future assignment rows keep a valid FK target.
    is_active = models.BooleanField(default=True)

    class Meta:
        constraints = [
            # Expression constraint rather than ``unique=True``: the rule is
            # case-insensitive. This is expressible natively because we own the model
            # — contrast migration 0001, which needed raw SQL for the same rule on
            # ``auth_user.email`` precisely because that model belongs to Django.
            #
            # Enforced by the DATABASE, not by a preceding SELECT: two concurrent
            # creates of the same name both pass any check-then-insert, and only a
            # constraint can reject the loser.
            models.UniqueConstraint(
                Lower("name"), name="access_management_role_name_ci_uniq"),
        ]
        # Default ordering for the admin and for any unordered queryset. The API
        # never relies on it — ``services.base.paginate`` always orders explicitly,
        # because an implicit order is not a stable one to paginate against.
        ordering = ("name",)
        verbose_name = "Role"
        verbose_name_plural = "Roles"

    def __str__(self) -> str:
        return self.name
