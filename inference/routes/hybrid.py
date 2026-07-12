"""inference hybrid route — POST /v1/run_hybrid_query (migration_plan.md §8.2).

Calls ``veda_core.veda_hybrid.run_hybrid_query`` VERBATIM (the single front door:
decompose / route / fan-out / firewall) and returns the MultiResult with the
terminal ``status`` preserved exactly (§19 item 1). The heavy sync flow runs in a
thread pool via ``run_in_threadpool_with_context`` so the event loop stays
responsive and the tenant context is carried in (§4.1, §5.3).

Also exposes POST /v1/run_hybrid_query/stream: the SAME pipeline, but the
sync call's ``on_event`` hook is wired to an SSE stream so a caller sees real
stage-progress (classify / decompose / route / answer) AS the pipeline
advances, instead of blocking silently for the whole call. The sync pipeline
runs on its own daemon thread (not the thread pool — it must emit while still
running, which a pool call that awaits completion can't do); the ambient
(source, tenant) context is captured in the request coroutine and re-bound in
that thread via ``veda_core.context.with_context`` (§4.1).

# LINT: raw run_in_threadpool / ThreadPoolExecutor.submit is banned here —
# use inference.concurrency.run_in_threadpool_with_context (§4.1)
"""
from __future__ import annotations

import dataclasses
import json
import threading
from typing import Any

try:
    from fastapi import APIRouter
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError:
    APIRouter = None
    StreamingResponse = None
    BaseModel = object


# Internal-only keys that must never reach an HTTP caller. "context" is
# veda.execution_state.ExecutionState (Tier1→Tier2 propagation — explicitly
# internal-only, never an API field). "trace" is the full debug trace (already
# deliberately excluded from the chat/SSE path — see apps/chat/services.py's own
# "never sent over SSE" comment); stripped here too so EVERY caller of this route
# (not just the chat path) gets the same guarantee, not just the ones that happen
# to allowlist their own fields.
_INTERNAL_ONLY_KEYS = frozenset({"context", "trace"})


def _verbose() -> bool:
    """Container-log verbosity for the query pipeline, controlled by env.

    VEDA_INFERENCE_VERBOSE=1 (default) → run_hybrid_query(verbose=True): the full
    stage-by-stage detail (classify, routing, tier decisions, reuse logging) prints
    to stdout where `docker logs inference` captures it. Set 0 to quiet it down.
    Read per-request (not at import) so it can be flipped without a code change —
    just restart the container with the new env value."""
    import os
    return os.environ.get("VEDA_INFERENCE_VERBOSE", "1") not in ("0", "false", "False")


def _serialize(obj: Any) -> Any:
    """Best-effort JSON-safe conversion that preserves the MultiResult shape.
    Strips _INTERNAL_ONLY_KEYS from any dict encountered, at any nesting depth —
    this is the ONE place every head result (SQL/Tier-2/RAG/hybrid/NoSQL) passes
    through before crossing the wire, so it's the correct place to enforce this."""
    if dataclasses.is_dataclass(obj):
        return _serialize(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items() if k not in _INTERNAL_ONLY_KEYS}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


if APIRouter is not None:
    router = APIRouter(prefix="/v1")

    class HybridRequest(BaseModel):
        query: str
        source_id: int | None = None
        tenant: str | None = None
        flags: dict | None = None

    @router.post("/run_hybrid_query")
    async def run_hybrid_query_route(req: "HybridRequest"):
        from inference.concurrency import run_in_threadpool_with_context
        from veda_core.veda_hybrid import run_hybrid_query

        result = await run_in_threadpool_with_context(run_hybrid_query, req.query,
                                                      verbose=_verbose())
        payload = _serialize(result)
        # Surface a top-level status for callers that don't walk items (§19 item 1).
        items = payload.get("items") if isinstance(payload, dict) else None
        top_status = (
            items[0].get("status") if items and isinstance(items[0], dict) else "unknown"
        )
        return {"status": top_status, "result": payload}

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    @router.post("/run_hybrid_query/stream")
    async def run_hybrid_query_stream_route(req: "HybridRequest"):
        import asyncio

        from veda_core.context import try_current, with_context
        from veda_core.veda_hybrid import run_hybrid_query

        loop = asyncio.get_event_loop()
        events: "asyncio.Queue[tuple[str, dict] | None]" = asyncio.Queue()
        parent_ctx = try_current()  # snapshot: the worker thread starts with no context (§4.1)

        def on_event(phase: str, message: str, extra: dict):
            loop.call_soon_threadsafe(
                events.put_nowait, ("progress", {"phase": phase, "message": message, **extra})
            )

        def _run():
            try:
                result = run_hybrid_query(req.query, verbose=_verbose(), on_event=on_event)
                payload = _serialize(result)
                items = payload.get("items") if isinstance(payload, dict) else None
                top_status = (
                    items[0].get("status") if items and isinstance(items[0], dict) else "unknown"
                )
                loop.call_soon_threadsafe(
                    events.put_nowait, ("result", {"status": top_status, "result": payload})
                )
            except Exception as exc:  # never leave the stream hanging on a crash
                loop.call_soon_threadsafe(
                    events.put_nowait, ("error", {"message": f"{type(exc).__name__}: {exc}"})
                )
            finally:
                loop.call_soon_threadsafe(events.put_nowait, None)

        threading.Thread(target=with_context(parent_ctx, _run), daemon=True).start()

        async def gen():
            while True:
                item = await events.get()
                if item is None:
                    return
                yield _sse(*item)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache",
                                           "X-Accel-Buffering": "no"})
else:  # pragma: no cover
    router = None