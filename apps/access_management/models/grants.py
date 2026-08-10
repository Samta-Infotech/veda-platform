"""The two grant edges — who holds a role, and what a role may do.

PURPOSE
    ``UserRole``       user  ── holds ──>  role
    ``RolePermission`` role  ── may (allow|deny) <permission> on <resource> ──>

    Together with ``Role``, ``Permission`` and ``CatalogResource`` these complete the
    RBAC graph. **Nothing enforces them yet** — the resolver reads them, and the gates
    act on the resolver, both later phases.

WHY ONE MODULE FOR TWO MODELS
    The rest of this package is one module per aggregate. These two are grouped
    because they are the same concept — a directed, audited edge between two nodes,
    with identical ``granted_by`` semantics and identical idempotency rules. Splitting
    them would produce two near-identical files whose shared conventions could drift.

AUDIT
    Both carry ``granted_by``. Until a real audit trail exists (M8 in
    ``AUTH_ISSUES_BACKLOG.md``), this is the only durable record of *who* conferred
    authority — the first question asked in any access incident. It is ``SET_NULL``,
    never ``CASCADE``: removing the administrator who made a grant must not silently
    remove the grant.

CASCADE RULES — each chosen deliberately, not defaulted
    ``UserRole.user``            CASCADE — an assignment without its user is meaningless.
    ``UserRole.role``            PROTECT — a role that is still held cannot be deleted.
    ``RolePermission.role``      CASCADE — a role's grants are part of the role.
    ``RolePermission.permission``PROTECT — the catalogue is seeded and never deleted;
                                 PROTECT makes any attempt fail loudly instead of
                                 silently revoking every grant of that capability.

WHY ``resource_path`` IS A STRING, NOT A FOREIGN KEY
    Same reasoning as ``CatalogResource.substrate_id``, one level up. A grant must
    survive catalog churn, and an operator must be able to grant on a resource *before*
    discovery has seen it (pre-provisioning a source that is still ingesting). The
    resolver denies unknown resources anyway, so the failure direction is already safe.
    The path is validated for canonical form on write (``resource_path.validate``), so
    it can never be stored in a shape the resolver would fail to match.
"""
from __future__ import annotations

from django.conf import settings
from django.db import models

from apps.core.models import TimeStampedModel


class Effect(models.TextChoices):
    """ALLOW or DENY, per ADR-0001 §3.5.

    DENY wins globally at any depth, and the absence of a grant is a denial. Both
    rules fail closed.
    """

    ALLOW = "allow", "Allow"
    DENY = "deny", "Deny"


class UserRole(TimeStampedModel):
    """A user holds a role."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="role_assignments")
    role = models.ForeignKey(
        "access_management.Role", on_delete=models.PROTECT,
        related_name="user_assignments")

    #: Who made this assignment. Null once that administrator is deleted — the grant
    #: survives, because losing an audit detail must not silently revoke access.
    granted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="+")

    class Meta:
        constraints = [
            # Holding a role twice is meaningless. Enforced by the database so two
            # concurrent assignments cannot both insert — the service relies on this
            # rather than on a preceding SELECT.
            models.UniqueConstraint(
                fields=["user", "role"], name="access_management_userrole_uniq"),
        ]
        indexes = [
            # The resolver's entry point: "which roles does this user hold?"
            models.Index(fields=["user"], name="userrole_user_idx"),
            # The admin's reverse question: "who holds this role?"
            models.Index(fields=["role"], name="userrole_role_idx"),
        ]
        ordering = ("user_id", "role_id")
        verbose_name = "User role"
        verbose_name_plural = "User roles"

    def __str__(self) -> str:
        return f"user#{self.user_id} holds role#{self.role_id}"


class RolePermission(TimeStampedModel):
    """A role may (or may not) exercise a permission on a resource."""

    role = models.ForeignKey(
        "access_management.Role", on_delete=models.CASCADE, related_name="grants")
    permission = models.ForeignKey(
        "access_management.Permission", on_delete=models.PROTECT,
        related_name="grants")

    #: Canonical resource path (ADR-0001), or "" for a permission that is not
    #: resource-scoped (``user.manage`` applies to the platform, not to a table).
    #: Deliberately not a foreign key — see the module docstring.
    resource_path = models.CharField(max_length=512, blank=True)

    effect = models.CharField(
        max_length=5, choices=Effect.choices, default=Effect.ALLOW)

    granted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="+")

    class Meta:
        constraints = [
            # One decision per (role, permission, resource). Without this a role could
            # hold both an ALLOW and a DENY for the same triple, and which one applied
            # would depend on row order — a non-deterministic authorization outcome.
            #
            # NOTE: `effect` is deliberately NOT part of the key. Re-granting with the
            # opposite effect must UPDATE the existing decision, not add a second,
            # contradictory one.
            models.UniqueConstraint(
                fields=["role", "permission", "resource_path"],
                name="access_management_rolepermission_uniq"),
        ]
        indexes = [
            # The resolver's second step: "what does this set of roles grant?"
            models.Index(fields=["role"], name="rolepermission_role_idx"),
            # Reverse lookup for the admin UI: "who can read this resource?"
            models.Index(fields=["resource_path"], name="rolepermission_path_idx"),
        ]
        ordering = ("role_id", "permission_id", "resource_path")
        verbose_name = "Role permission"
        verbose_name_plural = "Role permissions"

    def __str__(self) -> str:
        target = self.resource_path or "(global)"
        return f"role#{self.role_id} {self.effect} permission#{self.permission_id} on {target}"
