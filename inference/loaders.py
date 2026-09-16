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

_STATE: dict = {"ready": False, "semantic_model": False, "engine_warm": False,
                "nl_summary_model_warm": False, "degraded": []}
# "degraded" (P1-4, 2026-09-10): names of components that warmed in a reduced-quality
# mode, surfaced non-gating in /readyz — a missing reranker model or unreachable SLM
# used to be discoverable only by grepping container logs.


async def hydrate() -> dict:
    """Warm the engine best-effort; never crash startup on a not-yet-ingested store."""
    import veda_core  # noqa: F401 — activates the path shim
    from veda_core import config

    # 1) Semantic model — present only after an ingestion run has completed. Per-source
    #    since the M1 close-out (2026-09-15): "warm" means at least one source has its own
    #    scoped model under <ARTIFACT_ROOT>/<tenant>/<source>/ (the flat file is retired).
    import glob as _glob
    _root = config.ARTIFACT_ROOT
    if not os.path.isabs(_root):
        _root = os.path.join(os.path.dirname(os.path.abspath(config.__file__)), _root)
    _scoped = _glob.glob(os.path.join(_root, "*", "*", "veda_semantic_model.json"))
    _STATE["semantic_model"] = bool(_scoped)
    if not _STATE["semantic_model"]:
        logger.warning("no per-source semantic model under %s — run ingestion first", _root)
        _STATE["degraded"].append("semantic_model_missing")

    # 2) Warm the retrieval engine / encoders so the first query isn't cold.
    try:
        from veda_core.retrieval import retrieval_engine_phase3 as _rep

        if hasattr(_rep, "get_engine"):
            _rep.get_engine()
            _STATE["engine_warm"] = True
    except Exception as exc:  # non-fatal: engine also lazy-loads on first query
        logger.warning("engine warm-load deferred to first query: %s", exc)
        _STATE["degraded"].append("engine_cold_at_startup")

    # 3) Explicitly warm the heavy per-query models (BGE-M3 dense+sparse, the cross-encoder
    #    reranker, and the SLM). These lazy-init on first use, and cold BGE-M3 load alone is
    #    ~22s on CPU — paying it at startup keeps the first real query inside the SLA.
    def _p(msg):  # print so it lands in docker logs (the module logger isn't wired to stdout)
        print(f"  [warmup] {msg}", flush=True)

    try:
        from veda_core.ingestion import m3_encoder
        m3_encoder.encode_query("warm up the dense and sparse encoders")   # dense + sparse
        m3_encoder.encode_sparse(["warm up the sparse index encoder"])
        _p(f"✓ BGE-M3 (dense+sparse) [{m3_encoder.get_embed_backend()}]")
        if m3_encoder.get_embed_backend() == "cpu" and os.environ.get("METAL_EMBED_URL", "").strip():
            # METAL_EMBED_URL is set but the very first call already fell back to CPU —
            # a real perf degrade (see m3_encoder.py's own module comment: ~28s/retrieval
            # on CPU vs. Metal-offloaded), not just "unset, using CPU by design".
            _STATE["degraded"].append("embed_backend_cpu_fallback")
    except Exception as exc:
        _p(f"BGE-M3 warm deferred: {exc}")
        _STATE["degraded"].append("bge_m3_warm_failed")
    try:
        from veda_core.query import reranker as _rr
        _r = _rr._get_reranker()
        if _r is not None:
            # CrossEncoder → .predict; FlagReranker → .compute_score. Support both.
            _score = getattr(_r, "predict", None) or getattr(_r, "compute_score", None)
            if _score is not None:
                _score([["warm up", "cross encoder reranker"]])
                _p("✓ cross-encoder reranker")
        else:
            # Not an exception — _get_reranker() returns None on a load failure
            # (already logged at ERROR there); this is where that fact becomes
            # visible outside container logs (P1-4).
            _p("reranker unavailable — retrieval will be pure RRF (degraded) for every query")
            _STATE["degraded"].append("reranker_unavailable")
    except Exception as exc:
        _p(f"reranker warm deferred: {exc}")
        _STATE["degraded"].append("reranker_warm_failed")
    try:
        from veda_core.slm._call_slm import prewarm
        prewarm()               # loads + pins the SLM on the (host Metal) backend
        _p("✓ SLM")
    except Exception as exc:
        _p(f"SLM warm deferred: {exc}")
        _STATE["degraded"].append("slm_unreachable")

    try:
        from veda_core.slm._call_slm import prewarm
        from veda_core.config import NL_SUMMARY_MODEL
        # Distinct small model for result summarization (query/result_explainer.py) —
        # its own cold Ollama load easily exceeds the NL_ANSWER_FAST_TIMEOUT_MS budget
        # (~800ms), which silently degrades every early answer to the generic
        # deterministic fallback until something else happens to warm it. Prewarming
        # here means the first real query already finds it resident (keep_alive 24h).
        prewarm(model=NL_SUMMARY_MODEL)
        _STATE["nl_summary_model_warm"] = True
        _p("✓ NL summary SLM")
    except Exception as exc:
        # Non-fatal by design (query/result_explainer.py falls back to deterministic
        # template answers) — but surfaced in /readyz so a missing/unpulled
        # NL_SUMMARY_MODEL is an observable operational fact, not a silent one.
        _p(f"NL summary SLM warm deferred: {exc}")
        _STATE["degraded"].append("nl_summary_slm_unreachable")

    _STATE["ready"] = _STATE["semantic_model"]
    _p(f"hydrate complete: {_STATE}")
    return dict(_STATE)


def readiness() -> dict:
    return dict(_STATE)
