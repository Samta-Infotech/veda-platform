"""inference/loaders.py — startup warm-load / memory hydration (migration_plan.md §8.1).

The preserved engine lazy-initialises its heavy pieces (BGE encoder, reranker,
5-signal engine, semantic model) on first use inside ``veda_hybrid.run_hybrid_query``.
This hydrate() warms what it safely can at lifespan startup so the first request
isn't cold, and records a versions/readiness dict for /readyz — without duplicating
or second-guessing the engine's own initialisation (PRESERVE, §5.1).

Full §8.1 hydration (FK map / glossary / KG / verified-cache warm set / assembled
``sm`` from Redis+pgvector) lands with the storage_adapters seam (Phase 3 rest);
until then the engine reads its own internal store directly, which is behaviourally
identical.

# LINT: raw run_in_threadpool / ThreadPoolExecutor.submit is banned here —
# use inference.concurrency.run_in_threadpool_with_context (§4.1)
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("inference.loaders")

_STATE: dict = {"ready": False, "semantic_model": False, "engine_warm": False}


async def hydrate() -> dict:
    """Warm the engine best-effort; never crash startup on a not-yet-ingested store."""
    import veda_core  # noqa: F401 — activates the path shim
    from veda_core import config

    # 1) Semantic model — present only after an ingestion run has completed.
    sm_path = config.SEMANTIC_MODEL_FILE
    _STATE["semantic_model"] = os.path.exists(sm_path)
    if not _STATE["semantic_model"]:
        logger.warning("semantic model not found at %s — run ingestion first", sm_path)

    # 2) Warm the retrieval engine / encoders so the first query isn't cold.
    try:
        from veda_core.retrieval import retrieval_engine_phase3 as _rep

        if hasattr(_rep, "get_engine"):
            _rep.get_engine()
            _STATE["engine_warm"] = True
    except Exception as exc:  # non-fatal: engine also lazy-loads on first query
        logger.warning("engine warm-load deferred to first query: %s", exc)

    _STATE["ready"] = _STATE["semantic_model"]
    logger.info("hydrate complete: %s", _STATE)
    return dict(_STATE)


def readiness() -> dict:
    return dict(_STATE)
