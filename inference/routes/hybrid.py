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
from decimal import Decimal
from typing import Any

try:
    from fastapi import APIRouter, Request
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError:
    APIRouter = None
    Request = None
    StreamingResponse = None
    BaseModel = object


def _incoming_trace_id(request) -> "str | None":
    """Reuse the api-tier correlation id (apps/core/middleware.RequestIdMiddleware
    sets X-Request-Id, forwarded by apps/query/inference_client.py) as the ONE
    query trace_id — §1 "don't introduce a redundant identifier". None → the engine
    mints its own, so direct/CLI callers still get a trace_id."""
    try:
        return request.headers.get("x-request-id") or None
    except Exception:
        return None


# Internal-only keys that must never reach an HTTP caller. "context" is
# veda.execution_state.ExecutionState (Tier1→Tier2 propagation — explicitly
# internal-only, never an API field). "trace" is the full debug trace (already
# deliberately excluded from the chat/SSE path — see apps/chat/services.py's own
# "never sent over SSE" comment); stripped here too so EVERY caller of this route
# (not just the chat path) gets the same guarantee, not just the ones that happen
# to allowlist their own fields. "_debug" is the same idea for paths that have no
# ExplainTrace to record onto (e.g. Tier-2's _tier2_finish() noting an Insight
# Engine failure so a zero-token usage reading is distinguishable from "no LLM
# call was needed" — veda_hybrid.py).
_INTERNAL_ONLY_KEYS = frozenset({"context", "trace", "_debug"})


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
    if isinstance(obj, Decimal):
        # psycopg2 returns Decimal for NUMERIC/SUM/AVG columns (e.g. monetary
        # "amount" fields) — falling through to the generic str(obj) below
        # turned every such value into a STRING on the wire (e.g. "423.000"),
        # which silently broke every downstream numeric check that expects a
        # real number: apps/chat/visualization.py's _is_numeric()/_to_number()
        # (its own comment already assumes it receives a Decimal to convert,
        # not a pre-stringified one) — no chart was ever produced for a query
        # whose measure was a NUMERIC/DECIMAL column, only INTEGER aggregates
        # (e.g. COUNT(*), which survive as native JSON ints) worked. float()
        # matches what that downstream code already does with a real Decimal.
        return float(obj)
    return str(obj)



#: The engine is imported under BOTH names — bare ``context`` (cwd=veda_core) and
#: ``veda_core.context`` (inference tier, PYTHONPATH=/app) — and Python loads those
#: as TWO module objects. Verified in the running container: their
#: ``_source_profiles`` ContextVars are DIFFERENT objects, so a value set through
#: one name is invisible to a reader going through the other.
#:
#:     copy_context() snapshot, profiles set via veda_core.context:
#:         veda_core.context.current_source_profiles()  -> {'2': {...}}
#:         context.current_source_profiles()            -> {}          <-- lost
#:
#: `copy_context()` carries whatever the writer actually set, which is why it fixed
#: the measured "every source resolved to 'a data source'" bug. These two helpers
#: are the belt for the other half of the problem: they re-bind the captured scope
#: through EVERY name, so a reader on either one sees it. `veda_hybrid._current_ctx`
#: already does the equivalent for reads of the RequestContext; this covers the
#: writes, and source profiles, which have no such fallback.
_CONTEXT_MODULES = ("veda_core.context", "context")


def _capture_scope() -> dict:
    """The request scope, read through whichever module name holds it. Never raises."""
    import importlib
    out = {"ctx": None, "profiles": None}
    for name in _CONTEXT_MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        try:
            if out["ctx"] is None:
                out["ctx"] = mod.try_current()
        except Exception:
            pass
        try:
            if not out["profiles"]:
                out["profiles"] = mod.current_source_profiles() or None
        except Exception:
            pass
    return out


def _rebind_scope(scope: dict) -> None:
    """Re-bind a captured scope through EVERY context module name.

    Idempotent — re-setting the value a copy_context() snapshot already carries
    changes nothing. Best-effort by design: a failure here must never cost the
    request, because the snapshot is the primary mechanism and this is the belt.
    """
    import importlib
    for name in _CONTEXT_MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if scope.get("ctx") is not None:
            try:
                mod.set_context(scope["ctx"])
            except Exception:
                pass
        if scope.get("profiles"):
            try:
                mod.set_source_profiles(scope["profiles"])
            except Exception:
                pass

if APIRouter is not None:
    router = APIRouter(prefix="/v1")

    class HybridRequest(BaseModel):
        query: str
        source_id: int | None = None
        tenant: str | None = None
        flags: dict | None = None

    @router.post("/run_hybrid_query")
    async def run_hybrid_query_route(req: "HybridRequest", request: Request):
        from inference.concurrency import run_in_threadpool_with_context
        from veda_core.veda_hybrid import run_hybrid_query

        _tid = _incoming_trace_id(request)
        result = await run_in_threadpool_with_context(run_hybrid_query, req.query,
                                                      verbose=_verbose(), trace_id=_tid)
        payload = _serialize(result)
        # Surface a top-level status for callers that don't walk items (§19 item 1).
        items = payload.get("items") if isinstance(payload, dict) else None
        top_status = (
            items[0].get("status") if items and isinstance(items[0], dict) else "unknown"
        )
        # trace_id surfaced top-level so a client can grep the full trace by it.
        _trace_id = payload.get("trace_id") if isinstance(payload, dict) else None
        return {"status": top_status, "trace_id": _trace_id, "result": payload}

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    @router.post("/run_hybrid_query/stream")
    async def run_hybrid_query_stream_route(req: "HybridRequest", request: Request):
        import asyncio
        from contextvars import copy_context

        from veda_core.veda_hybrid import run_hybrid_query

        loop = asyncio.get_event_loop()
        events: "asyncio.Queue[tuple[str, dict] | None]" = asyncio.Queue()
        # Snapshot the WHOLE context, not just the RequestContext.
        #
        # This used to be `with_context(try_current(), _run)`, which re-binds only
        # the (source, tenant) RequestContext. `_source_profiles` is a SEPARATE
        # ContextVar, so it was silently dropped in the worker thread — measured on
        # the real chat path: every source resolved to the generic "a data source"
        # label (`known: false`) even though the api tier had sent correct names,
        # and the routing coordinator's canonical tie-break saw an empty profile map.
        #
        # copy_context() carries EVERY ContextVar, which is what
        # inference.concurrency.run_in_threadpool_with_context already does for the
        # non-streaming route — so the two paths now behave identically, and adding
        # a third request-scoped ContextVar cannot reintroduce this class of bug.
        parent_ctx = copy_context()
        # Captured separately from the snapshot: see _capture_scope for why the
        # snapshot alone can leave one module view of the scope empty.
        _scope = _capture_scope()
        _tid = _incoming_trace_id(request)

        def on_event(phase: str, message: str, extra: dict):
            loop.call_soon_threadsafe(
                events.put_nowait, ("progress", {"phase": phase, "message": message, **extra})
            )

        def _run():
            _rebind_scope(_scope)          # belt; the snapshot is the primary path
            try:
                result = run_hybrid_query(req.query, verbose=_verbose(),
                                          on_event=on_event, trace_id=_tid)
                payload = _serialize(result)
                items = payload.get("items") if isinstance(payload, dict) else None
                top_status = (
                    items[0].get("status") if items and isinstance(items[0], dict) else "unknown"
                )
                _trace_id = payload.get("trace_id") if isinstance(payload, dict) else None
                loop.call_soon_threadsafe(
                    events.put_nowait, ("result", {"status": top_status,
                                                   "trace_id": _trace_id, "result": payload})
                )
            except Exception as exc:  # never leave the stream hanging on a crash
                loop.call_soon_threadsafe(
                    events.put_nowait, ("error", {"message": f"{type(exc).__name__}: {exc}"})
                )
            finally:
                loop.call_soon_threadsafe(events.put_nowait, None)

        # MERGE RESOLUTION (2026-09-10) — both sides were fixing the SAME bug and
        # both diagnoses were right; this keeps the working mechanism from one and
        # the concern from the other.
        #
        # The bug: this used to be `with_context(try_current(), _run)`, which
        # re-binds ONLY the (source, tenant) RequestContext. `source_profiles` is a
        # SEPARATE ContextVar and was silently dropped in the worker. Measured
        # consequences on the real chat path: every source resolved to the generic
        # "a data source" label (`known: false`) though the api tier had sent the
        # names; the routing coordinator's canonical tie-break saw an empty profile
        # map; and `_is_datalake_source()` returned False for a datalake source, so
        # the datalake-isolated semantic model was never loaded and a vendor
        # question got planned against the primary source's 178-table schema.
        #
        # `copy_context()` carries EVERY ContextVar, which is what
        # inference/concurrency.py::run_in_threadpool_with_context already does for
        # the non-streaming route — so both routes now behave identically and a
        # ContextVar added later propagates without touching this line again.
        #
        # The other side ALSO wrapped `with_context(...)` inside the snapshot, for
        # the dual-import case (bare `context` vs `veda_core.context` hold separate
        # vars — veda_hybrid.py:71-81). That concern is real and verified, but the
        # wrapper is not the way to address it here: `with_context` is not imported
        # in this module, and `parent_ctx` is a contextvars.Context rather than the
        # RequestContext that `set_context` expects. `_rebind_scope()` at the top of
        # `_run` covers the same ground correctly, for profiles as well as the
        # RequestContext, and is idempotent when the snapshot already carried them.
        threading.Thread(target=lambda: parent_ctx.run(_run), daemon=True).start()

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