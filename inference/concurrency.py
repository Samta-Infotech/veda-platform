"""inference/concurrency.py — the ONE offload primitive (migration_plan.md §4.1).

Heavy sync inference (encoders, reranker) must run in a thread pool so the ASGI
event loop stays responsive, but a thread-pool call that runs under a leaked
context silently reads the wrong tenant. So this is the single wrapped primitive:
it snapshots the current context with ``copy_context()`` and runs ``fn`` inside it.

Raw ``run_in_threadpool`` / bare ``ThreadPoolExecutor.submit`` are BANNED in
``inference/`` and ``veda_core/`` (lint-enforced, §4.1) — everything offloads here.
"""
from __future__ import annotations

from contextvars import copy_context

try:
    import anyio
except ImportError:  # keep importable without anyio in this environment
    anyio = None


async def run_in_threadpool_with_context(fn, *args, **kwargs):
    """Offload ``fn`` to a worker thread under a copy of the current context."""
    if anyio is None:  # pragma: no cover - dependency present in the inference image
        raise RuntimeError("anyio is required for run_in_threadpool_with_context")
    ctx = copy_context()
    return await anyio.to_thread.run_sync(lambda: ctx.run(fn, *args, **kwargs))
