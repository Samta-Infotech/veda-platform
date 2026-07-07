"""VEDA · Shared resources: DB/BGE/engine/graph handles, constants, warm-up."""
import os, re, sys, time, json, logging, threading
import numpy as np
from config import SLM_MODEL_NAME, SLM_OLLAMA_BASE_URL, BIENCODER_TABLE_TABLE


# L7 execution target = the client source DB. Resolved LAZILY from the DB `Source`
# table — the single source of truth (§3.1). NO hardcoded host/credentials and NO
# static .env source values: a missing source is a hard fail at request time, never
# a silent localhost fallback. Kept lazy so importing this module stays side-effect free.
def get_db_config() -> dict:
    """The client source DB connection for L7 execution.

    Served queries (a request context is set) resolve the connection FROM the
    `Source` table for the request's source_id via storage_adapters.reader — so one
    warm engine serves N sources, connecting to whichever the ambient context selects.
    Outside a request (ingestion subprocess / dev CLI, where no context is set) it
    falls back to the injected source (config.get_primary_relational_source), whose
    VEDA_SOURCE_* env the ingesting worker itself populated from the same Source row.
    Either way the Source table is the origin — no hardcoded creds, no static .env."""
    from veda_core import context
    if context.try_current() is not None:
        from storage_adapters import reader
        return reader.source_connection()
    from config import get_primary_relational_source
    src = get_primary_relational_source()
    return {"host": src["host"], "port": src["port"], "database": src["dbname"],
            "user": src["user"], "password": src["password"]}

# Word-boundary match so column names like `updated_datetime` / `created_by_id`
# don't trip the DML/DDL guard.


_FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "truncate",
              "create", "grant", "revoke")


# Single-sourced from config so it can't drift from the table the biencoder writes.
# (Was hardcoded "table_embeddings" — singular, never created in ensemble mode → the
# semantic table-routing signal silently returned {} on every query.)
TABLE_EMB_TABLE = BIENCODER_TABLE_TABLE   # "table_embeddings_v2"


IMPORTANCE_WEIGHTS = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.25}


_BGE = None
# Per-process memo of query→normalized vector. The same query string is encoded by
# both table routing and the verified-cache lookup; encoding once and reusing it
# avoids 1–2 redundant ~100–300ms BGE passes per request. Bounded so it can't grow
# without limit across a long-lived server process.


_QVEC_CACHE = {}


def _get_bge():
    """Lazy-load one BGE-M3 (offline). Reused for table routing + verified cache.

    Delegates to the ingestion biencoder singleton when it's the SAME model/device — the query
    path preloads bge-large via `ingestion.biencoder._get_biencoder`, so funnelling here means
    the whole process holds ONE ~1.3GB copy instead of two."""
    global _BGE
    if _BGE is None:
        from config import BGE_MODEL_NAME, BIENCODER_MODEL, BGE_DEVICE
        # Share the ingestion biencoder singleton whenever it's the SAME model — one ~1.3GB
        # copy instead of two. (Was gated on device=="cpu"; the shared instance already sits
        # on the resolved device, so the gate is unnecessary and blocked GPU sharing.)
        if BGE_MODEL_NAME == BIENCODER_MODEL:
            try:
                from ingestion.biencoder import _get_biencoder
                shared = _get_biencoder()
                if shared is not None:
                    _BGE = shared
                    return _BGE
            except Exception:
                pass   # fall back to an own load
        from sentence_transformers import SentenceTransformer
        _BGE = SentenceTransformer(BGE_MODEL_NAME, device=BGE_DEVICE, local_files_only=True)
    return _BGE


def _encode_query(query):
    """BGE-M3 encode + normalize, memoized per process (see _QVEC_CACHE)."""
    v = _QVEC_CACHE.get(query)
    if v is None:
        v = _get_bge().encode(query, normalize_embeddings=True)
        if len(_QVEC_CACHE) < 512:
            _QVEC_CACHE[query] = v
    return v


def _pg():
    import psycopg2
    cfg = get_db_config()
    return psycopg2.connect(host=cfg["host"], port=cfg["port"],
                            dbname=cfg["database"], user=cfg["user"],
                            password=cfg["password"])


_ENGINES = {}
# ONE BGE searcher shared across all per-source engines. Signal-1's query embedding is
# source-INDEPENDENT (the embedding STORE is source-scoped via storage_adapters.ann_search),
# so the ~1.3GB model loads once, not once per source.
_SEARCHER = None


_GRAPH = None


JOIN_CONFIDENCE_FLOOR = 0.55


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        from query.join_planner import load_graph
        _GRAPH = load_graph()
    return _GRAPH


def _internal_db_config() -> dict:
    """VEDA's INTERNAL embeddings store (config.VEDA_INTERNAL_DB) — where the engine's
    Signal-1 tables (column_embeddings_v2) live. This is the CORRECT store for the
    engine's semantic search; the client SOURCE DB (get_db_config) is only the L7 SQL
    EXECUTION target and never holds embeddings. For served queries Signal 1 goes
    through the source-scoped storage_adapters adapter; this connection backs only the
    no-context (dev CLI) direct-store fallback."""
    from config import VEDA_INTERNAL_DB as v
    return {"host": v["host"], "port": v["port"], "database": v["dbname"],
            "user": v["user"], "password": v["password"]}


def _shared_searcher():
    """The ONE BGE SemanticSearchEngine, shared by every per-source engine (Signal-1
    query embedding is source-independent). Also publishes its BGE model to the
    table-routing / verified-cache singleton so the whole process holds one copy.

    Tolerant: returns None if it can't build (e.g. internal store briefly unreachable at
    warm) so the worker still warms — each engine then falls back to its own searcher
    (or None → Signal 1 disabled), exactly as before this was shared."""
    global _SEARCHER, _BGE
    if _SEARCHER is None:
        try:
            from retrieval.semantic_search import SemanticSearchEngine
            _SEARCHER = SemanticSearchEngine(_internal_db_config())
            try:
                _BGE = _SEARCHER.searcher.model
            except Exception:
                pass
        except Exception:
            return None
    return _SEARCHER


def _engine_scope():
    """The (source, tenant) key for the per-source engine cache, or a single "_global"
    key when no request context is set (dev CLI / warm-load)."""
    from veda_core import context
    ctx = context.try_current()
    return (str(ctx.source_id), str(ctx.tenant)) if ctx is not None else ("_global", "_global")


def _load_scoped_sm():
    """This scope's semantic model for building the engine's BM25/signals. Redis-first
    (the Django assembler publishes `veda:sm:{source}:{tenant}`, §3.6) so one warm
    worker serves N sources; on-disk `SEMANTIC_MODEL_FILE` fallback (dev / cache miss)."""
    from veda_core import context
    ctx = context.try_current()
    if ctx is not None and os.environ.get("VEDA_SM_REDIS", "").strip().lower() in ("1", "true", "yes", "on"):
        try:
            import redis as _redis
            url = os.environ.get("REDIS_CACHE_URL", "redis://redis-cache:6379/0")
            raw = _redis.Redis.from_url(url).get(f"veda:sm:{ctx.source_id}:{ctx.tenant}")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    from config import SEMANTIC_MODEL_FILE
    with open(SEMANTIC_MODEL_FILE) as f:
        return json.load(f)


def get_engine(sm=None):
    """The Phase-3 retrieval engine for the CURRENT (source, tenant) scope (P5).

    Built lazily once per scope and reused, so one warm worker serves N ready sources
    — each engine carries THAT source's semantic model + BM25/signals, and all engines
    share the one BGE searcher (Signal-1 store is source-scoped via storage_adapters).
    `sm` (the caller's already-loaded per-source model, e.g. from veda_hybrid) is used
    when given; otherwise it's loaded for this scope (Redis-first)."""
    scope = _engine_scope()
    eng = _ENGINES.get(scope)
    if eng is None:
        from retrieval.retrieval_engine_phase3 import RetrievalEnginePhase3
        try:
            from config import RETRIEVAL_CACHE_ENABLED
        except Exception:
            RETRIEVAL_CACHE_ENABLED = False
        eng = RetrievalEnginePhase3(
            semantic_model=sm if sm is not None else _load_scoped_sm(),
            db_config=_internal_db_config(),
            semantic_searcher=_shared_searcher(),
            use_cache=RETRIEVAL_CACHE_ENABLED,
        )
        _ENGINES[scope] = eng
    return eng


def clear_engines():
    """Drop all per-source engines (rehydrate/re-ingest hook, §8.4): the next query
    for a scope rebuilds its engine from the freshly assembled semantic model."""
    _ENGINES.clear()


def _prewarm_ollama():
    """Best-effort: load the SQL model into the SLM backend's memory (Ollama:
    keep_alive pin; vLLM: 1-token completion) so the first real generate doesn't
    pay a cold model load. Safe in a background thread — never raises."""
    try:
        from slm import prewarm
        prewarm(timeout=120)
    except Exception:
        pass


def warm_up(verbose: bool = True, prewarm_llm: bool = True) -> float:
    """Load every per-process singleton ONCE — semantic registries, the retrieval
    engine (+ shared BGE-M3), and (concurrently) the Ollama SQL model. Idempotent.
    This is the single call a long-lived process makes so per-query work stays warm."""
    t0 = time.time()
    if prewarm_llm:                # load the LLM in parallel with the engine build
        threading.Thread(target=_prewarm_ollama, daemon=True).start()
    try:
        from semantic import registry as _reg
        _reg.load()
    except Exception as e:
        if verbose:
            print(f"  [warm] registry warning: {e}")
    _shared_searcher()             # preload the one BGE searcher (source-independent)
    try:                           # per-source engines build lazily per request; warm the
        get_engine()               # "_global" one when a flat model exists (dev/single-source)
    except Exception as e:
        if verbose:
            print(f"  [warm] global engine deferred (per-source built on first query): {e}")
    _get_bge()
    if verbose:
        print(f"  ready in {time.time() - t0:.1f}s")
    return time.time() - t0
