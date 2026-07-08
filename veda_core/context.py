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
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RequestContext:
    """Ambient (source, tenant) for one request.

    ``source_id`` is the PRIMARY source — the single-source ingest/publish path
    (dispatcher, writer, ingestion layers) reads it and its meaning is unchanged.
    ``source_ids`` is the query-time source *SET* (P5 / cross-source): the scope a
    served query traverses. It defaults to ``(source_id,)`` so every existing
    single-source construction (all of ingestion) behaves byte-identically — the
    set is only wider when the query tier explicitly passes a validated subset.
    Frozen + a tuple field so the context stays hashable (engine-cache key)."""
    source_id: int
    tenant: str
    source_ids: tuple = ()

    def __post_init__(self):
        # Normalize the set: default to the primary, dedupe preserving order, and
        # guarantee the primary is a member (so `source_id` is always in scope).
        ids = tuple(dict.fromkeys(int(s) for s in (self.source_ids or (self.source_id,))))
        if int(self.source_id) not in ids:
            ids = (int(self.source_id), *ids)
        object.__setattr__(self, "source_ids", ids)


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
