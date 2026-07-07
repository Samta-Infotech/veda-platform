"""inference/main.py — ASGI app + lifespan warm-load (migration_plan.md §8, §8.2, §3.5).

FastAPI/Uvicorn app. The lifespan handler calls ``loaders.hydrate()`` once per
worker (one warm engine per process). An ASGI middleware sets the ambient
``(source, tenant)`` context per request from the request body/headers (§3.5), so
``storage_adapters`` can scope every query without the engine signatures changing.

Endpoints (§8.2): /v1/run_hybrid_query, /v1/retrieve, /v1/rehydrate, /healthz, /readyz.

# LINT: raw run_in_threadpool / ThreadPoolExecutor.submit is banned here —
# use inference.concurrency.run_in_threadpool_with_context (§4.1)
"""
from __future__ import annotations

from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, Request
    _HAVE_FASTAPI = True
except ImportError:  # keep importable without FastAPI in this environment
    FastAPI = None
    Request = object
    _HAVE_FASTAPI = False

from veda_core.context import RequestContext, set_context


def _start_rehydrate_subscriber():
    """Subscribe to the redis-cache rehydrate channel (§8.4). On any broadcast, drop
    the in-process sm cache so the next query reloads the Django-assembled sm — this is
    how a re-ingestion on the worker reaches EVERY inference replica, not just one."""
    import os
    import threading

    def _run():
        try:
            import redis as _redis
            url = os.environ.get("REDIS_CACHE_URL", "redis://redis-cache:6379/0")
            pubsub = _redis.Redis.from_url(url).pubsub()
            pubsub.psubscribe("veda:rehydrate:*")
            import veda_hybrid
            for msg in pubsub.listen():
                if msg.get("type") == "pmessage":
                    veda_hybrid._SM.clear()   # scope-keyed dict (P5) — drop all scopes
                    try:                       # fast-path registries are scope-keyed too (P5)
                        from semantic import registry as _reg
                        _reg.clear()
                    except Exception:
                        pass
                    try:                       # rebuild per-source engines from the fresh model (P5)
                        from veda.runtime import clear_engines
                        clear_engines()
                    except Exception:
                        pass
                    try:                       # re-ingest may have retuned ef_search (Q-10)
                        from storage_adapters import reader as _reader
                        _reader.clear_ef_search_cache()
                    except Exception:
                        pass
                    try:                       # re-ingest rebuilt the graph (WP5 PPR matrix)
                        from query import graph_retriever as _gr
                        _gr.clear_ppr_cache()
                    except Exception:
                        pass
        except Exception:
            return  # non-fatal: a replica catches up on its next lifespan warm-load

    t = threading.Thread(target=_run, name="veda-rehydrate-sub", daemon=True)
    t.start()


@asynccontextmanager
async def lifespan(app):
    from inference import loaders

    app.state.versions = await loaders.hydrate()  # Phase 5: warm-load §8.1
    _start_rehydrate_subscriber()                  # §8.4 fan-out subscriber
    yield


def create_app():
    if not _HAVE_FASTAPI:  # pragma: no cover - fastapi ships in the inference image
        raise RuntimeError("fastapi is required to build the inference app")

    app = FastAPI(title="VEDA inference", lifespan=lifespan)

    @app.middleware("http")
    async def _tenant_context(request: Request, call_next):
        # api tier forwards server-resolved {source_id, tenant}; never client-supplied (§6.2).
        source_id = request.headers.get("x-veda-source-id")
        tenant = request.headers.get("x-veda-tenant")
        if source_id is not None and tenant is not None:
            set_context(RequestContext(source_id=int(source_id), tenant=tenant))
        return await call_next(request)

    from inference.routes import health, hybrid, retrieve

    app.include_router(health.router)
    app.include_router(retrieve.router)
    app.include_router(hybrid.router)
    return app


app = create_app() if _HAVE_FASTAPI else None
