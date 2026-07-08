"""inference retrieve + rehydrate routes (migration_plan.md §8.2, §8.4).

/v1/retrieve runs the warm 5-signal engine (get_engine().retrieve) under the
request's ambient context and returns ranked columns + scores. /v1/rehydrate
reloads the sm read-model and fans out to every replica via redis-cache pub/sub
(§8.4). Heavy sync work offloads through run_in_threadpool_with_context (§4.1).

# LINT: raw run_in_threadpool / ThreadPoolExecutor.submit is banned here —
# use inference.concurrency.run_in_threadpool_with_context (§4.1)
"""
from __future__ import annotations

try:
    from fastapi import APIRouter
    from pydantic import BaseModel
except ImportError:
    APIRouter = None
    BaseModel = object


def _serialize_results(results):
    out = []
    for r in results:
        out.append({
            "col_id": getattr(r, "col_id", None),
            "col_name": getattr(r, "col_name", None),
            "table_name": getattr(r, "table_name", None),
            "score": getattr(r, "final_score", getattr(r, "similarity", None)),
        })
    return out


if APIRouter is not None:
    router = APIRouter(prefix="/v1")

    class RetrieveRequest(BaseModel):
        query: str
        source_id: int | None = None
        tenant: str | None = None
        intent: str = "SIMPLE"
        top_k: int = 15

    @router.post("/retrieve")
    async def retrieve_route(req: "RetrieveRequest"):
        from inference.concurrency import run_in_threadpool_with_context

        def _run():
            from veda.runtime import get_engine
            engine = get_engine()
            results = engine.retrieve(req.query, req.intent, req.top_k)
            return _serialize_results(results)

        cols = await run_in_threadpool_with_context(_run)
        return {"query": req.query, "top_k": req.top_k, "columns": cols}

    class RehydrateRequest(BaseModel):
        source_id: int
        tenant: str
        scope: str = "all"

    @router.post("/rehydrate")
    async def rehydrate_route(req: "RehydrateRequest"):
        # Local reload (invalidate the in-process sm cache so the next query reloads
        # the Django-assembled sm from redis), then fan out to peers via pub/sub (§8.4).
        import veda_hybrid
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
        published = 0
        try:
            import json
            import os
            import redis as _redis
            url = os.environ.get("REDIS_CACHE_URL", "redis://redis-cache:6379/0")
            ch = f"veda:rehydrate:{req.source_id}:{req.tenant}:{req.scope}"
            published = _redis.Redis.from_url(url).publish(
                ch, json.dumps({"source_id": req.source_id, "tenant": req.tenant, "scope": req.scope})
            )
        except Exception:
            pass
        return {"reloaded": True, "peers_notified": published}
else:  # pragma: no cover
    router = None
