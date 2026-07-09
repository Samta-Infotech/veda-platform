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

    # 3) Explicitly warm the heavy per-query models (BGE-M3 dense+sparse, the cross-encoder
    #    reranker, and the SLM). These lazy-init on first use, and cold BGE-M3 load alone is
    #    ~22s on CPU — paying it at startup keeps the first real query inside the SLA.
    def _p(msg):  # print so it lands in docker logs (the module logger isn't wired to stdout)
        print(f"  [warmup] {msg}", flush=True)

    try:
        from veda_core.ingestion import m3_encoder
        m3_encoder.encode_query("warm up the dense and sparse encoders")   # dense + sparse
        m3_encoder.encode_sparse(["warm up the sparse index encoder"])
        _p("✓ BGE-M3 (dense+sparse)")
    except Exception as exc:
        _p(f"BGE-M3 warm deferred: {exc}")
    try:
        from veda_core.query import reranker as _rr
        _r = _rr._get_reranker()
        if _r is not None:
            # CrossEncoder → .predict; FlagReranker → .compute_score. Support both.
            _score = getattr(_r, "predict", None) or getattr(_r, "compute_score", None)
            if _score is not None:
                _score([["warm up", "cross encoder reranker"]])
                _p("✓ cross-encoder reranker")
    except Exception as exc:
        _p(f"reranker warm deferred: {exc}")
    try:
        from veda_core.slm._call_slm import prewarm
        prewarm()               # loads + pins the SLM on the (host Metal) backend
        _p("✓ SLM")
    except Exception as exc:
        _p(f"SLM warm deferred: {exc}")

    _STATE["ready"] = _STATE["semantic_model"]
    _p(f"hydrate complete: {_STATE}")
    return dict(_STATE)


def readiness() -> dict:
    return dict(_STATE)
