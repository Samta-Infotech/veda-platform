"""veda_core/context.py

Migration plan §4.1 — Tenant & request context (the ambient-context pattern).

The engine's public functions are frozen and carry no tenant/source argument
(`run_query(query, sm, all_cols)`, `verified_cache_lookup(query)`,
`get_engine().retrieve(query, ...)`). So `(source, tenant)` travel out of band
through a contextvars.ContextVar, set at the edge (api/inference/worker tiers)
and read only by `storage_adapters`. Fail-closed: an unset context raises,
never defaults to a tenant, so a missing-context bug is a loud error on that
one request rather than a silent cross-tenant read.

No Django import here — this module is imported by both veda_core (library)
and the Django tiers, and must stay import-light.
"""

from contextvars import ContextVar, copy_context
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    source_id: int
    tenant: str


_ctx: ContextVar["RequestContext | None"] = ContextVar("veda_ctx", default=None)


def set_context(ctx: RequestContext):
    return _ctx.set(ctx)


def current() -> RequestContext:
    c = _ctx.get()
    if c is None:                       # fail-closed: never silently default a tenant
        raise RuntimeError("no VEDA request context set")
    return c


def try_current() -> "RequestContext | None":
    """Non-raising read — returns None when unset. For offload sites that must
    carry the PARENT context into worker threads (which start with an empty
    contextvars context): capture in the parent, `set_context` in each child."""
    return _ctx.get()


def with_context(ctx, fn):
    """Wrap `fn` so it re-binds `ctx` (a captured RequestContext, or None) at the
    start of the call. Used to carry the ambient (source, tenant) into
    ThreadPoolExecutor workers, which start with an empty contextvars context.
    Safe under concurrency: each worker sets its own thread-local contextvar."""
    def _wrapped(*args, **kwargs):
        if ctx is not None:
            set_context(ctx)
        return fn(*args, **kwargs)
    return _wrapped
