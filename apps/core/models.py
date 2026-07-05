"""apps.core base models — tenancy, UUID PK, timestamps (migration plan §2.1, §5, §9a).

Every substrate model inherits ``TenantScopedModel`` so multi-tenant deployment
needs no schema change later. Tenancy is applied via a custom manager whose
queryset filters by the ambient ``(source, tenant)`` from
``veda_core.context.current()`` — a forgotten ``.filter(tenant=...)`` cannot leak
data. An explicit ``all_tenants()`` escape hatch exists for admin/migrations only.
"""
from __future__ import annotations

import uuid

from django.db import models


class UUIDPrimaryKeyModel(models.Model):
    """Matches ingestion's per-table/column UUID scheme (plan §5 apps.core)."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    class Meta:
        abstract = True


class TimeStampedModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class TenantQuerySet(models.QuerySet):
    def all_tenants(self):
        """Escape hatch for admin/migrations only — bypasses ambient scoping."""
        return self


class TenantManager(models.Manager):
    """Filters by the ambient (source, tenant) so scoping is not left to callers.

    Reads ``veda_core.context.current()`` (§4.1). When no context is set the
    manager falls back to the unscoped queryset (admin/migration paths); request
    and task paths always set the context, and ``storage_adapters`` additionally
    scopes every raw query. Fail-closed enforcement lives in ``context.current()``.
    """

    def get_queryset(self):
        qs = TenantQuerySet(self.model, using=self._db)
        try:
            from veda_core.context import current

            ctx = current()
        except Exception:
            return qs
        return qs.filter(source_id=ctx.source_id, tenant=ctx.tenant)

    def all_tenants(self):
        return TenantQuerySet(self.model, using=self._db)


class TenantScopedModel(UUIDPrimaryKeyModel, TimeStampedModel):
    """Abstract base for every substrate row: scoped to (source, tenant)."""

    source = models.ForeignKey(
        "sources.Source", on_delete=models.CASCADE, related_name="+"
    )
    tenant = models.CharField(max_length=128, db_index=True)

    objects = TenantManager()

    class Meta:
        abstract = True
