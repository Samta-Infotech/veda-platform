"""inference/engine.py — the warm get_engine() singleton (migration_plan.md §8, §8.1).

This service IS ``veda/runtime.get_engine()``: one warm engine per process,
warm-loaded once at startup and held in memory. The warm engine object and its
``retrieve()`` behaviour come straight from
``veda_core.retrieval.retrieval_engine_phase3`` — we wrap it in a lifespan, we do
NOT rewrite it (§5.1 PRESERVE callout).

# LINT: raw run_in_threadpool / ThreadPoolExecutor.submit is banned here —
# use inference.concurrency.run_in_threadpool_with_context (§4.1)
"""
from __future__ import annotations

_ENGINE = None


def get_engine():
    """Return the process-wide warm engine, loading it once (§8.1)."""
    global _ENGINE
    if _ENGINE is None:
        raise NotImplementedError(
            "Phase 5.1: warm-load veda_core.retrieval.retrieval_engine_phase3.get_engine() "
            "into module-level _ENGINE during lifespan startup (one per process)."
        )
    return _ENGINE
