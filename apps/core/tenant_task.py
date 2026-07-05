"""apps.core.tenant_task — Celery base that sets the ambient (source, tenant) context.

Migration plan §4.1 worker-tier: "a small task base class / decorator sets the
context from the task's (source, tenant) args before any veda_core ingestion
function runs." Every ingestion task uses this base so `storage_adapters` reads a
correct `(source, tenant)` from `veda_core.context.current()` — never an unset or
leaked one (fail-closed).
"""
from __future__ import annotations

from contextvars import copy_context

try:
    from celery import Task
except ImportError:  # keep importable without celery in the thin test env
    Task = object

from veda_core.context import RequestContext, set_context


class TenantTask(Task):
    """Celery Task base that binds `(source_id, tenant)` for the task's duration.

    Reads `source_id` / `tenant` from the task kwargs (or the first two positional
    args) and sets the context inside a copied contextvars context so it cannot
    leak across the worker's task boundary.
    """

    abstract = True

    def __call__(self, *args, **kwargs):
        source_id = kwargs.get("source_id")
        tenant = kwargs.get("tenant")
        if source_id is None and len(args) >= 1:
            source_id = args[0]
        if tenant is None and len(args) >= 2:
            tenant = args[1]

        def _run():
            if source_id is not None and tenant is not None:
                set_context(RequestContext(source_id=int(source_id), tenant=str(tenant)))
            return super(TenantTask, self).__call__(*args, **kwargs)

        # Copied context: the binding is scoped to this task run, never leaked.
        return copy_context().run(_run)
