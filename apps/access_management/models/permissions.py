"""The ``Permission`` aggregate — the vocabulary of things VEDA can authorize.

PURPOSE
    A permission names an ACTION ("read data", "run ingestion"). It is the verb of
    the authorization sentence; the noun — which source, which table — is bound at
    grant time, against the catalog that already exists. Keeping them apart is your
    CATALOG rule made structural: catalog is metadata, permission is authorization,
    and neither model knows the other's business.

WHY CODE-DEFINED, NOT ADMIN-CREATED
    Only code can enforce a permission. A row an administrator invents at runtime is
    one no gate will ever check, so the UI would present authority that does not
    exist — an authorization system that silently does nothing is worse than one that
    refuses. Permissions are therefore seeded by migration and exposed read-only;
    ROLES are the layer administrators compose freely.

    The practical consequence: adding a permission is a code change (a seed migration
    plus the gate that checks it), and that is the intended friction.

NAME COLLISION
    ``django.contrib.auth.models.Permission`` also exists and is unrelated — it is
    model-level (add/change/delete/view per Django model) and describes Django's own
    admin. This one describes VEDA's actions. They never mix; import with an alias if
    a module ever needs both.

EXTENSION POINTS (deliberately not built)
    ``RolePermission`` will foreign-key here and carry the resource reference. That
    reference is genuinely hard — ``Source`` has an int PK while ``SchemaTable`` and
    ``SchemaColumn`` have UUIDs, in a different app — and it is deliberately NOT
    guessed at here. No ``resource_type`` column, no scope field: the grant phase
    will decide, with the grant requirements in hand.

SECURITY
    ``is_active`` is authorization-relevant the moment the resolver exists: an
    inactive permission must grant nothing, so a capability can be switched off
    without deleting rows that history references. Modelled now, enforced there.

PERFORMANCE
    Cardinality is a handful of rows that every permission check will read. The one
    index is the case-insensitive unique constraint on ``code``, which is also the
    lookup path — a resolver asking "does this role have data.read?" hits it.
"""
from __future__ import annotations

from django.db import models
from django.db.models.functions import Lower

from apps.core.models import TimeStampedModel


class Permission(TimeStampedModel):
    """A single authorizable action. See the module docstring for the design notes."""

    #: Stable machine key, dotted ``domain.action`` (e.g. ``data.read``). This is what
    #: gates compare against, so it is effectively an API: renaming one silently
    #: revokes every grant that referenced it by code. Unique case-insensitively so
    #: "Data.Read" cannot shadow "data.read".
    code = models.CharField(max_length=100)

    #: Human label for an administration screen.
    name = models.CharField(max_length=150)

    #: What granting this actually allows, in the administrator's words. The field
    #: that stops a role screen being a list of opaque dotted strings.
    description = models.TextField(blank=True)

    #: Switch a capability off platform-wide without deleting rows that grants and
    #: audit history reference.
    is_active = models.BooleanField(default=True)

    class Meta:
        constraints = [
            # Expression constraint rather than ``unique=True``: the rule is
            # case-insensitive. Enforced by the DATABASE, so a concurrent seed cannot
            # produce two rows for the same capability.
            models.UniqueConstraint(
                Lower("code"), name="access_management_permission_code_ci_uniq"),
        ]
        ordering = ("code",)
        verbose_name = "Permission"
        verbose_name_plural = "Permissions"

    def __str__(self) -> str:
        return self.code
