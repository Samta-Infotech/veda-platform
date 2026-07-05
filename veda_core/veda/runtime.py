"""VEDA · Shared resources: DB/BGE/engine/graph handles, constants, warm-up."""
import os, re, sys, time, json, logging, threading
import numpy as np
from config import SLM_MODEL_NAME, SLM_OLLAMA_BASE_URL, BIENCODER_TABLE_TABLE


# L7 execution target = the client source DB. Env-overridable (§9) so the
# inference container reaches launchpad @ host.docker.internal:5433.
DB_CONFIG = {"host": os.environ.get("VEDA_SOURCE_HOST", "localhost"),
             "port": int(os.environ.get("VEDA_SOURCE_PORT", "5433")),
             "database": os.environ.get("VEDA_SOURCE_DBNAME", "default_db"),
             "user": os.environ.get("VEDA_SOURCE_USER", "postgres"),
             "password": os.environ.get("VEDA_SOURCE_PASSWORD", "admin")}

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
    return psycopg2.connect(host=DB_CONFIG["host"], port=DB_CONFIG["port"],
                            dbname=DB_CONFIG["database"], user=DB_CONFIG["user"],
                            password=DB_CONFIG["password"])


_ENGINE = None


_GRAPH = None


JOIN_CONFIDENCE_FLOOR = 0.55


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        from query.join_planner import load_graph
        _GRAPH = load_graph()
    return _GRAPH


def get_engine():
    """Lazily build the Phase 3 engine ONCE and reuse it. Also shares the engine's
    BGE-M3 with table-routing / verified-cache so the model loads only once."""
    global _ENGINE, _BGE
    if _ENGINE is None:
        from retrieval.retrieval_engine_phase3 import RetrievalEnginePhase3
        try:
            from config import RETRIEVAL_CACHE_ENABLED
        except Exception:
            RETRIEVAL_CACHE_ENABLED = False
        _ENGINE = RetrievalEnginePhase3(db_config=DB_CONFIG, use_cache=RETRIEVAL_CACHE_ENABLED)
        try:                       # reuse the engine's loaded BGE-M3 (kills double load)
            _BGE = _ENGINE.semantic_searcher.searcher.model
        except Exception:
            pass
    return _ENGINE


def _prewarm_ollama():
    """Best-effort: load the SQL model into Ollama memory (keep_alive) so the first
    real generate doesn't pay a cold model load. Safe to run in a background thread —
    never raises, and keep_alive holds the model resident for subsequent queries."""
    try:
        import urllib.request
        payload = {"model": SLM_MODEL_NAME, "stream": False, "keep_alive": "24h",
                   "messages": [{"role": "user", "content": "ok"}],
                   "options": {"num_predict": 1}}
        req = urllib.request.Request(f"{SLM_OLLAMA_BASE_URL}/api/chat",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=120).read()
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
    get_engine()                   # RetrievalEnginePhase3 + shared BGE-M3
    _get_bge()
    if verbose:
        print(f"  ready in {time.time() - t0:.1f}s")
    return time.time() - t0
