"""Tenant-context isolation + fail-closed tests (migration plan §4.1, Phase 3 exit 5).

Run: python tests/test_context_isolation.py  (from repo root, veda_core importable).
Asserts:
  1. current() fails closed (raises) when no context is set.
  2. Interleaved contexts on concurrent threads never cross-read.
  3. The offload primitive runs fn under a COPY of the caller's context.
"""
import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import veda_core  # noqa: F401 — path shim
from veda_core.context import RequestContext, current, set_context


def test_fail_closed():
    # Fresh contextvars context with nothing set → current() must raise.
    def _probe():
        try:
            current()
            return "NO_RAISE"
        except RuntimeError:
            return "RAISED"
    from contextvars import copy_context
    assert copy_context().run(_probe) == "RAISED", "current() must fail closed when unset"
    print("[1] fail-closed: current() raises when unset  ✓")


def test_interleaved_no_cross_read():
    """Two threads bind different (source, tenant); each must read only its own."""
    results = {}
    barrier = threading.Barrier(2)

    def worker(sid, tenant):
        from contextvars import copy_context

        def _run():
            set_context(RequestContext(source_id=sid, tenant=tenant))
            barrier.wait(timeout=5)            # force interleave after both set
            ctx = current()
            return (ctx.source_id, ctx.tenant)
        results[(sid, tenant)] = copy_context().run(_run)

    t1 = threading.Thread(target=worker, args=(1, "alpha"))
    t2 = threading.Thread(target=worker, args=(2, "beta"))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert results[(1, "alpha")] == (1, "alpha"), results
    assert results[(2, "beta")] == (2, "beta"), results
    print("[2] interleaved contexts: no cross-tenant read  ✓")


def test_offload_carries_context():
    """run_in_threadpool_with_context must run fn under a copy of the set context."""
    from inference.concurrency import run_in_threadpool_with_context

    async def _drive():
        set_context(RequestContext(source_id=7, tenant="gamma"))

        def _in_thread():
            c = current()               # must see (7, gamma), not raise
            return (c.source_id, c.tenant)
        return await run_in_threadpool_with_context(_in_thread)

    got = asyncio.run(_drive())
    assert got == (7, "gamma"), got
    print("[3] offload primitive carries context into the thread pool  ✓")


if __name__ == "__main__":
    test_fail_closed()
    test_interleaved_no_cross_read()
    test_offload_carries_context()
    print("ALL CONTEXT-ISOLATION TESTS PASSED")
