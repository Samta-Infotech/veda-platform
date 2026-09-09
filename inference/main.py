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

import json
import logging

from veda_core.context import (RequestContext, parse_allowed_resources, set_context,
                               set_source_profiles)

logger = logging.getLogger(__name__)


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
                    try:                       # re-ingest changes the FK graph — drop the cached edge set
                        from storage_adapters import reader as _reader
                        _reader.clear_fk_adjacency_cache()
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
        # api tier forwards server-resolved {source_id, source_ids, tenant}; never
        # client-supplied (§6.2). X-Veda-Source-Ids is the validated query SCOPE (P5);
        # X-Veda-Source-Id is the primary member (single-source exec/audit path).
        source_id = request.headers.get("x-veda-source-id")
        source_ids_hdr = request.headers.get("x-veda-source-ids")
        tenant = request.headers.get("x-veda-tenant")
        if source_id is not None and tenant is not None:
            source_ids = tuple(int(s) for s in source_ids_hdr.split(",") if s.strip()) \
                if source_ids_hdr else ()
            # Gate 1 (User Story 3, Task 15) — the api tier's precomputed RBAC data
            # scope, if any. Absent header = no restriction (§ RequestContext
            # docstring); a header present but malformed fails CLOSED (empty tuple =
            # "nothing addressable"), never silently falls through to "no
            # restriction" — a bug in the sender must never widen access.
            data_scope_hdr = request.headers.get("x-veda-data-scope")
            allowed_resources = None
            if data_scope_hdr:
                try:
                    allowed_resources = parse_allowed_resources(data_scope_hdr)
                except Exception:
                    logger.exception(
                        "malformed X-Veda-Data-Scope header; failing closed to no access")
                    allowed_resources = ()
            set_context(RequestContext(source_id=int(source_id), tenant=tenant,
                                       source_ids=source_ids,
                                       allowed_resources=allowed_resources))
            # Multi-source routing profiles (source_type/is_canonical/domain_tags/description),
            # server-resolved by the api tier from the Source registry (apps/query/scope.py::
            # source_profiles_for) and sent as X-Veda-Source-Profiles. This was the one forwarded
            # header nothing here consumed, so set_source_profiles() was only ever called by
            # in-process callers (the bench scripts) — every HTTP request, on BOTH the plain and
            # the streaming route, reached the engine with NO profiles. That is why the same
            # question answered correctly in-process and failed through the API: with no profile,
            # veda_hybrid._is_datalake_source() is False for a datalake source, the
            # datalake-isolated semantic model is never loaded, and the SQL planner is handed the
            # primary source's 178-table homzhub schema instead — a vendor question then had only
            # irrelevant homzhub tables to choose between and refused with "ambiguous subject —
            # should rows be per reviews_pillar or reviews_pillarrating?".
            # Absent/malformed header = {} = exactly the previous behaviour (route on evidence
            # alone); profiles are routing METADATA, never an access decision, so unlike the
            # data-scope header above there is nothing to fail closed on.
            profiles_hdr = request.headers.get("x-veda-source-profiles")
            if profiles_hdr:
                try:
                    set_source_profiles(json.loads(profiles_hdr))
                except Exception:
                    logger.warning("malformed X-Veda-Source-Profiles header; ignoring")
                    set_source_profiles({})
            else:
                set_source_profiles({})
        return await call_next(request)

    from inference.routes import health, hybrid, retrieve

    app.include_router(health.router)
    app.include_router(retrieve.router)
    app.include_router(hybrid.router)
    return app


app = create_app() if _HAVE_FASTAPI else None
