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
    """The ONE dense encoder (BGE-M3), reused for table routing + verified cache.

    WP3: returns the shared m3_encoder dense facade — the SAME underlying model that
    produces the stored column/table/graph/chunk vectors AND the sparse weights, so the
    whole process holds exactly one copy and query/passage vectors share one space."""
    global _BGE
    if _BGE is None:
        from ingestion import m3_encoder
        _BGE = m3_encoder.get_dense_encoder()
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


_ENGINES = {}   # scope -> engine, insertion-ordered → used as an LRU (see ENGINE_CACHE_MAX)
# Per-worker cap on distinct scope engines. An engine entry is index/signal state, NOT
# model weights (the ~1.3GB BGE and the SLM are shared singletons), so the marginal RSS
# of one entry is the per-source sparse/signal maps — bounded, but a multi-source tenant
# can spawn one engine per requested subset ({A}, {B}, {A,B}, …), so cap + LRU-evict the
# least-recently-used scope. Tune against measured worker RSS.
ENGINE_CACHE_MAX = int(os.environ.get("ENGINE_CACHE_MAX", "4"))
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
    """The (tenant, source-SET) key for the per-scope engine cache, or a single
    "_global" key when no request context is set (dev CLI / warm-load).

    Keyed by `frozenset(source_ids)` (P5 / cross-source): one engine per distinct
    request scope, so `{A}`, `{B}` and `{A,B}` are three separate engines — each
    carrying that scope's merged semantic model + sparse/signal state. The models
    (BGE-M3 etc.) are shared singletons, so an engine entry is index state, not
    model weights (see ENGINE_CACHE_MAX)."""
    from veda_core import context
    ctx = context.try_current()
    if ctx is None:
        return ("_global", frozenset())
    return (str(ctx.tenant), frozenset(int(s) for s in ctx.source_ids))


def _load_one_sm(source_id, tenant):
    """One source's semantic model. Redis-first (the Django assembler publishes
    `veda:sm:{source}:{tenant}`, §3.6) so one warm worker serves N sources; on-disk
    `SEMANTIC_MODEL_FILE` fallback (dev / cache miss)."""
    if os.environ.get("VEDA_SM_REDIS", "").strip().lower() in ("1", "true", "yes", "on"):
        try:
            import redis as _redis
            url = os.environ.get("REDIS_CACHE_URL", "redis://redis-cache:6379/0")
            raw = _redis.Redis.from_url(url).get(f"veda:sm:{source_id}:{tenant}")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    from config import SEMANTIC_MODEL_FILE
    with open(SEMANTIC_MODEL_FILE) as f:
        return json.load(f)


def _merge_scoped_sms(pairs):
    """Merge per-source semantic models into ONE namespace for a multi-source scope
    (§Phase 1.4). Table keys stay bare when unique across the set and are source-
    qualified (`src{ID}.{table}`) only on collision — column keys follow their table.
    Every table/column entry is tagged with `_source_id` so the SQL/federation tier
    (Phase 5) can resolve the owning source. Single-source scopes never reach here."""
    # Which bare table names collide across sources → those get qualified.
    seen: dict = {}
    for sid, sm in pairs:
        for t in (sm.get("tables") or {}):
            seen.setdefault(t, set()).add(sid)
    ambiguous = {t for t, sids in seen.items() if len(sids) > 1}

    def tkey(sid, t):
        return f"src{sid}.{t}" if t in ambiguous else t

    merged: dict = {"version": next((sm.get("version") for _, sm in pairs), "2.0"),
                    "tables": {}, "columns": {}, "retrieval_documents": {},
                    "domain_synonyms": {}, "concept_graph": {}}
    for sid, sm in pairs:
        remap: dict = {}  # bare table -> namespaced table (for this source)
        for t, entry in (sm.get("tables") or {}).items():
            nk = tkey(sid, t); remap[t] = nk
            merged["tables"][nk] = {**entry, "_source_id": sid}
        for ck, entry in (sm.get("columns") or {}).items():
            t, _, col = ck.partition(".")
            nck = f"{remap.get(t, t)}.{col}" if col else ck
            merged["columns"][nck] = {**entry, "_source_id": sid}
        for ck, doc in (sm.get("retrieval_documents") or {}).items():
            t, _, col = ck.partition(".")
            nck = f"{remap.get(t, t)}.{col}" if col else ck
            merged["retrieval_documents"][nck] = doc
        for term, v in (sm.get("domain_synonyms") or {}).items():
            merged["domain_synonyms"].setdefault(term, v)
        for concept, v in (sm.get("concept_graph") or {}).items():
            merged["concept_graph"].setdefault(concept, v)
    return merged


def _load_scoped_sm():
    """This scope's semantic model for building the engine's BM25/signals. A single-
    source scope returns that source's model unchanged (byte-identical to the pre-P5
    path); a multi-source scope returns the merged namespace (`_merge_scoped_sms`)."""
    from veda_core import context
    ctx = context.try_current()
    if ctx is None:
        from config import SEMANTIC_MODEL_FILE
        with open(SEMANTIC_MODEL_FILE) as f:
            return json.load(f)
    ids = list(ctx.source_ids)
    if len(ids) == 1:
        return _load_one_sm(ids[0], ctx.tenant)
    return _merge_scoped_sms([(sid, _load_one_sm(sid, ctx.tenant)) for sid in ids])


def get_engine(sm=None):
    """The Phase-3 retrieval engine for the CURRENT (source, tenant) scope (P5).

    Built lazily once per scope and reused, so one warm worker serves N ready sources
    — each engine carries THAT source's semantic model + BM25/signals, and all engines
    share the one BGE searcher (Signal-1 store is source-scoped via storage_adapters).
    `sm` (the caller's already-loaded per-source model, e.g. from veda_hybrid) is used
    when given; otherwise it's loaded for this scope (Redis-first)."""
    scope = _engine_scope()
    eng = _ENGINES.get(scope)
    if eng is not None:
        _ENGINES[scope] = _ENGINES.pop(scope)   # touch → most-recently-used (move to end)
        return eng
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
    while len(_ENGINES) > ENGINE_CACHE_MAX:        # evict least-recently-used scope(s)
        _ENGINES.pop(next(iter(_ENGINES)))
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
