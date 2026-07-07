"""VEDA · Verified-query cache (file-based, cosine ≥ 0.85)."""
import os, re, sys, time, json, logging, threading
import numpy as np
from veda.runtime import _encode_query, _get_bge


# Absolute (repo-root) path, not CWD-relative — else the verified-query store silently
# reads/writes the wrong file (or none) when VEDA runs from a different working directory.
VERIFIED_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "veda_verified_queries.json")
# Guards the read-modify-write of the verified-query JSON store (concurrent writes
# under a threaded server would otherwise interleave and corrupt the file).


_VERIFIED_WRITE_LOCK = threading.Lock()

# Ranking-layer map: importance_class → score weight. Retunable here without
# re-ingesting (the metadata only stores the class).


_VERIFIED_EMB = {"queries": [], "mat": None}   # stored-query strings + their stacked embeddings


def verified_cache_lookup(query, threshold=0.85):
    """Return (sql, similarity) if a near-identical verified query exists, else (None, sim)."""
    # Phase 6.6 rewire: when a request/task context is set (platform), route through the
    # storage_adapters seam → Django VerifiedQueryCache + pgvector cosine, tenant-scoped.
    # Falls back to the legacy file store when no context (standalone/dev). Same return
    # shape (sql, similarity). Skip rules stay in the caller (pipeline.py).
    _nq = " ".join(str(query).lower().split())
    try:
        from veda_core.context import try_current
        if try_current() is not None:
            from storage_adapters import reader as _reader
            # Q-8: exact-hash short-circuit — one indexed lookup BEFORE the BGE encode.
            _exact = getattr(_reader, "verified_cache_exact", None)
            if _exact is not None:
                hit = _exact(query)
                if hit:
                    return hit["sql"], hit["similarity"]
            qv = _encode_query(query)
            hit = _reader.verified_cache_lookup(list(qv), threshold)
            return (hit["sql"], hit["similarity"]) if hit else (None, 0.0)
    except Exception:
        pass

    if not os.path.exists(VERIFIED_FILE):
        return None, 0.0
    try:
        import numpy as np
        store = json.load(open(VERIFIED_FILE))
        if not store:
            return None, 0.0
        # Q-8 (file store): normalized exact-string short-circuit before any encode.
        for _e in store:
            if " ".join(str(_e["query"]).lower().split()) == _nq:
                return _e["sql"], 1.0
        bge = _get_bge()
        stored_queries = [e["query"] for e in store]
        # Embed each stored query at most ONCE. Previously the whole store was
        # re-encoded on every lookup (O(N) BGE passes per query, unbounded growth as
        # the cache filled). Now: reuse the cached matrix, encode only appended rows,
        # and rebuild only if the store was edited/reordered.
        cached_qs = _VERIFIED_EMB["queries"]
        if _VERIFIED_EMB["mat"] is None or stored_queries[:len(cached_qs)] != cached_qs:
            _VERIFIED_EMB["mat"] = bge.encode(stored_queries, normalize_embeddings=True)
            _VERIFIED_EMB["queries"] = list(stored_queries)
        elif len(stored_queries) > len(cached_qs):
            new_mat = bge.encode(stored_queries[len(cached_qs):], normalize_embeddings=True)
            _VERIFIED_EMB["mat"] = np.vstack([_VERIFIED_EMB["mat"], new_mat])
            _VERIFIED_EMB["queries"] = list(stored_queries)
        mat = _VERIFIED_EMB["mat"]
        qv = _encode_query(query)
        sims = mat @ qv
        i = int(sims.argmax())
        s = float(sims[i])
        return (store[i]["sql"], s) if s >= threshold else (None, s)
    except Exception:
        return None, 0.0


def save_verified_query(query, sql):
    """Record a successfully-executed query so identical/similar ones skip the SLM."""
    # Phase 6.6 rewire: when a context is set, write to Django VerifiedQueryCache via the
    # adapter (idempotent ON CONFLICT, off no read-modify-write; §6.6). Falls back to the
    # legacy file store when standalone. The caller already applied the skip rules.
    try:
        from veda_core.context import try_current
        if try_current() is not None:
            from storage_adapters.reader import save_verified_query as _adapter_save
            qv = _encode_query(query)
            _adapter_save(query, list(qv), sql)
            return
    except Exception:
        pass

    try:
        with _VERIFIED_WRITE_LOCK:
            store = json.load(open(VERIFIED_FILE)) if os.path.exists(VERIFIED_FILE) else []
            if not any(e["query"] == query for e in store):
                store.append({"query": query, "sql": sql})
                os.makedirs(os.path.dirname(VERIFIED_FILE) or ".", exist_ok=True)
                json.dump(store, open(VERIFIED_FILE, "w"), indent=2)
    except Exception:
        pass
