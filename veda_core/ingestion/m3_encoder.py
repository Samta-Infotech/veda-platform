"""ingestion/m3_encoder.py — process-wide BGE-M3 encoder singleton (WP3).

ONE model — BAAI/bge-m3 — produces the 1024-dim dense vectors for columns, tables,
graph nodes AND document chunks, plus the learned sparse (lexical) weights that replace
BM25. This module is the single entry point every ingestion + query site uses so the
whole process holds exactly one copy of the model.

Deliberately NOT implemented: M3's ColBERT / multi-vector head. The bge-reranker-v2-m3
cross-encoder already supplies late-interaction quality on the top candidates, and storing
per-token vectors per column is cost without measured benefit (see RETRIEVAL_UPGRADE_PLAN
WP3). We request return_colbert_vecs=False everywhere.

Zero-egress: weights are baked into the image at build time; we force offline mode so a
missing local copy fails loud instead of reaching out to the hub.

FlagEmbedding is imported lazily inside the loader so this module imports cleanly in the
thin api image (which has no ML stack) — callers that never encode never pay for it.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import logging
import os
import threading
from typing import Dict, List, Tuple

import numpy as np

from config import BGE_MODEL_NAME, BIENCODER_BATCH_SIZE, resolve_device
import json as _json
import urllib.request as _u
from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)

_MODEL = None
_LOCK = threading.Lock()

# One user query is independently re-embedded by Tier1's 5-signal engine AND Tier2's
# retrieve_v2 (first-stage bi-encoder + table-prior) within the SAME request — same
# text, same model, same vector space, computed from scratch each time. This cache
# collapses those repeat encodes to one BGE-M3 forward pass per distinct query text.
# Bounded + immutable-tuple values so callers can't corrupt a shared numpy array by
# mutating what they get back. Algorithm-agnostic: every caller gets byte-identical
# output to an uncached call — only WHICH candidates each tier selects is untouched.
_QUERY_ENCODE_CACHE_SIZE = 64


# Optional Metal (host-GPU) offload: when METAL_EMBED_URL is set the encode calls are
# proxied to scripts/metal_embed_server.py running on the host (device=mps), mirroring how
# the SLM uses host Ollama. Docker-on-macOS has no GPU passthrough, so in-container BGE-M3
# is CPU-bound (~28s/retrieval); offloading to Metal is the main query-latency lever. Any
# transport error falls back to the in-process CPU model, so it never fails a query.
_METAL_URL = os.environ.get("METAL_EMBED_URL", "").strip()

# P1-4 (2026-09-10): which backend actually served the LAST encode call. A Metal
# transport failure falls back to CPU silently (only a stdout print, above) — this
# makes that fact readable by a caller (the query explain trace, /readyz) instead of
# only existing in container logs. "unset" until the first encode call in this process.
_LAST_EMBED_BACKEND = {"v": "unset" if _METAL_URL else "cpu"}


def get_embed_backend() -> str:
    """"metal" | "cpu" | "unset" — whichever backend served the most recent encode_*
    call in this process. Best-effort observability, not a per-call audit log."""
    return _LAST_EMBED_BACKEND["v"]


def _metal_post_uncached(path: str, payload: dict) -> dict:
    req = _u.Request(_METAL_URL.rstrip("/") + path, data=_json.dumps(payload).encode(),
                     headers={"Content-Type": "application/json"}, method="POST")
    with _u.urlopen(req, timeout=float(os.environ.get("METAL_EMBED_TIMEOUT", "60"))) as r:
        return _json.loads(r.read())


# ---------------------------------------------------------------------------
# Per-REQUEST single-flight memo for the Metal embed round-trips.
# (Benchmark finding §6.1, 2026-09-23.)
#
# Measured: every query issued each encode EXACTLY TWICE with a byte-identical
# payload. On "users created last month" that was /encode_sparse over 409 texts
# (the column catalog) twice at ~1.8s each — ~35% of the query's whole wall clock.
#
# The duplication has TWO different shapes, which is why a plain dict memo is not
# enough (measured with a thread-annotated probe):
#   /encode_dense   — same thread, sequential (355ms then 223ms) → a memo suffices;
#   /encode_sparse  — DIFFERENT threads, CONCURRENT (1823ms and 1827ms, overlapping)
#                     → both callers miss a plain memo and both still compute.
# So this is single-flight: the first caller for a key computes while the others
# BLOCK on an Event and then reuse its result.
#
# Scope is a ContextVar set once per request by veda_hybrid.run_hybrid_query. Two
# properties this buys, both verified rather than assumed:
#   • worker threads DO see it — veda_core/context.with_context carries the parent
#     context into the pool, confirmed by probe (ctxvar visible in
#     ThreadPoolExecutor-0_1);
#   • concurrent requests cannot share entries, and nothing survives the turn.
# Outside a request scope (ingestion, CLI) the cache is None and this is a plain
# pass-through — ingestion encodes thousands of distinct texts and must not
# accumulate them.
#
# Failures are never cached: the owner removes the key and every waiter re-raises,
# so each caller still takes its own documented CPU fallback.
# ---------------------------------------------------------------------------
_REQUEST_EMBED_CACHE: "contextvars.ContextVar[dict | None]" = contextvars.ContextVar(
    "veda_m3_request_embed_cache", default=None)


class _InFlight:
    """One key's slot: an Event the waiters block on, plus the result or error."""
    __slots__ = ("event", "value", "error")

    def __init__(self):
        self.event = threading.Event()
        self.value = None
        self.error = None


@contextlib.contextmanager
def request_embed_cache():
    """Open a per-request single-flight scope. Re-entrant: an inner `with` is a
    no-op pass-through so the outermost scope owns the cache for the whole turn."""
    if _REQUEST_EMBED_CACHE.get() is not None:
        yield
        return
    token = _REQUEST_EMBED_CACHE.set({"map": {}, "lock": threading.Lock()})
    try:
        yield
    finally:
        _REQUEST_EMBED_CACHE.reset(token)


def _payload_key(path: str, payload: dict) -> str:
    return path + ":" + hashlib.sha1(
        _json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _metal_post(path: str, payload: dict) -> dict:
    """Single-flight wrapper. Identical (path, payload) inside ONE request is sent
    to the Metal server exactly once; byte-identical behaviour otherwise."""
    cache = _REQUEST_EMBED_CACHE.get()
    if cache is None:
        return _metal_post_uncached(path, payload)

    key = _payload_key(path, payload)
    with cache["lock"]:
        slot = cache["map"].get(key)
        owner = slot is None
        if owner:
            slot = cache["map"][key] = _InFlight()

    if not owner:
        slot.event.wait()
        if slot.error is not None:
            raise slot.error
        return slot.value

    try:
        slot.value = _metal_post_uncached(path, payload)
    except BaseException as exc:
        slot.error = exc
        with cache["lock"]:          # never cache a failure
            cache["map"].pop(key, None)
        slot.event.set()
        raise
    slot.event.set()
    return slot.value


def _get_model():
    """Load (once) and return the shared BGEM3FlagModel. Thread-safe, lazy, offline."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOCK:
        if _MODEL is not None:
            return _MODEL
        # Zero-egress: never hit the hub — weights are baked into the image.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        device = resolve_device()
        use_fp16 = device == "cuda"   # fp16 only helps on GPU; CPU/MPS stay fp32
        logger.info(f"Loading BGE-M3 ({BGE_MODEL_NAME}) on {device} (fp16={use_fp16})...")
        _MODEL = BGEM3FlagModel(BGE_MODEL_NAME, use_fp16=use_fp16)
        logger.info("✓ BGE-M3 loaded")
    return _MODEL


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _clean_sparse(lw: dict) -> Dict[str, float]:
    """Normalize one M3 lexical_weights entry to a plain {token_id_str: float} dict,
    dropping zero/negative weights. FlagEmbedding returns a defaultdict keyed by token
    id (int or str) — we stringify keys so the inverted index is JSON-serializable and
    the query/passage token spaces line up (same tokenizer)."""
    out: Dict[str, float] = {}
    for tok, w in dict(lw).items():
        wf = float(w)
        if wf > 0.0:
            out[str(tok)] = wf
    return out


def encode_dense(texts: List[str]) -> np.ndarray:
    """Encode texts → (n, 1024) L2-normalized float32 dense matrix."""
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    if _METAL_URL:
        try:
            vecs = np.asarray(_metal_post("/encode_dense", {"texts": list(texts)})["vecs"],
                              dtype=np.float32)
            _LAST_EMBED_BACKEND["v"] = "metal"
            return vecs
        except Exception as _e:
            print(f"  [m3] metal encode_dense failed ({_e}) — CPU fallback", flush=True)
            _LAST_EMBED_BACKEND["v"] = "cpu"
    out = _get_model().encode(
        texts, batch_size=BIENCODER_BATCH_SIZE, max_length=512,
        return_dense=True, return_sparse=False, return_colbert_vecs=False,
    )
    _LAST_EMBED_BACKEND["v"] = "cpu"
    dense = np.asarray(out["dense_vecs"], dtype=np.float32)
    return _l2_normalize(dense)


def encode_sparse(texts: List[str]) -> List[Dict[str, float]]:
    """Encode texts → list of {token_id_str: weight} learned-sparse dicts."""
    if not texts:
        return []
    if _METAL_URL:
        try:
            result = [{str(k): float(v) for k, v in d.items()}
                      for d in _metal_post("/encode_sparse", {"texts": list(texts)})["sparse"]]
            _LAST_EMBED_BACKEND["v"] = "metal"
            return result
        except Exception as _e:
            print(f"  [m3] metal encode_sparse failed ({_e}) — CPU fallback", flush=True)
            _LAST_EMBED_BACKEND["v"] = "cpu"
    out = _get_model().encode(
        texts, batch_size=BIENCODER_BATCH_SIZE, max_length=512,
        return_dense=False, return_sparse=True, return_colbert_vecs=False,
    )
    _LAST_EMBED_BACKEND["v"] = "cpu"
    return [_clean_sparse(lw) for lw in out["lexical_weights"]]


@functools.lru_cache(maxsize=_QUERY_ENCODE_CACHE_SIZE)
def _encode_single_cached(text: str) -> Tuple[Tuple[float, ...], Tuple[Tuple[str, float], ...]]:
    """Single-text dense+sparse encode, memoized by exact text. Returns plain
    immutable tuples (not numpy/dict) so the cached entry can't be mutated by a
    caller holding a reference to a previous result."""
    if _METAL_URL:
        try:
            r = _metal_post("/encode_query", {"text": text})
            _LAST_EMBED_BACKEND["v"] = "metal"
            return (tuple(float(x) for x in r["dense"]),
                    tuple((str(k), float(v)) for k, v in r["sparse"].items()))
        except Exception as _e:
            print(f"  [m3] metal encode_query failed ({_e}) — CPU fallback", flush=True)
            _LAST_EMBED_BACKEND["v"] = "cpu"
    out = _get_model().encode(
        [text], batch_size=1, max_length=512,
        return_dense=True, return_sparse=True, return_colbert_vecs=False,
    )
    _LAST_EMBED_BACKEND["v"] = "cpu"
    dense = _l2_normalize(np.asarray(out["dense_vecs"], dtype=np.float32))[0]
    sparse = _clean_sparse(out["lexical_weights"][0])
    return tuple(dense.tolist()), tuple(sparse.items())


def encode_query(text: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """Encode one query → (1024-dim normalized dense vector, sparse weight dict) in a
    SINGLE forward pass (dense + sparse share the encode call). Memoized per exact
    query text (see _encode_single_cached) — a repeat call for the same text within
    or across requests returns the cached vector instead of re-running BGE-M3."""
    dense_t, sparse_t = _encode_single_cached(text)
    return np.asarray(dense_t, dtype=np.float32), dict(sparse_t)


class _DenseEncoder:
    """SentenceTransformer-compatible facade over the ONE BGE-M3 singleton.

    Lets the query-side call sites (veda.runtime._get_bge, retrieval_v2, semantic_search)
    keep calling ``.encode(text, normalize_embeddings=True)`` while sharing the exact same
    model + dense pooling as the stored embeddings — so query and passage vectors live in
    an identical space (WP3). Mirrors ST semantics: a str → 1-D vector, a list → 2-D."""

    def encode(self, sentences, normalize_embeddings=True, convert_to_numpy=True,
               show_progress_bar=False, batch_size=None, device=None, **_kw):
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        if len(texts) == 1:
            # Single query text (bare string OR a 1-element list, e.g. retrieve_v2's
            # per-query calls) — route through the shared memoized encode so a query
            # re-embedded by another retrieval path in the same request is free.
            dense_t, _sparse_t = _encode_single_cached(texts[0])
            out = np.asarray([dense_t], dtype=np.float32)
        else:
            out = encode_dense(texts)  # already L2-normalized (n, 1024)
        return out[0] if single else out

    def get_sentence_embedding_dimension(self) -> int:
        return 1024


_DENSE_ENCODER = _DenseEncoder()


def get_dense_encoder() -> "_DenseEncoder":
    """The shared dense-encoder facade (one instance, one underlying model)."""
    return _DENSE_ENCODER
