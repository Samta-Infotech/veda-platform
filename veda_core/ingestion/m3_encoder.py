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

import functools
import logging
import os
import threading
from typing import Dict, List, Tuple

import numpy as np

from config import BGE_MODEL_NAME, BIENCODER_BATCH_SIZE, resolve_device

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
        from FlagEmbedding import BGEM3FlagModel
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
    out = _get_model().encode(
        texts, batch_size=BIENCODER_BATCH_SIZE, max_length=512,
        return_dense=True, return_sparse=False, return_colbert_vecs=False,
    )
    dense = np.asarray(out["dense_vecs"], dtype=np.float32)
    return _l2_normalize(dense)


def encode_sparse(texts: List[str]) -> List[Dict[str, float]]:
    """Encode texts → list of {token_id_str: weight} learned-sparse dicts."""
    if not texts:
        return []
    out = _get_model().encode(
        texts, batch_size=BIENCODER_BATCH_SIZE, max_length=512,
        return_dense=False, return_sparse=True, return_colbert_vecs=False,
    )
    return [_clean_sparse(lw) for lw in out["lexical_weights"]]


@functools.lru_cache(maxsize=_QUERY_ENCODE_CACHE_SIZE)
def _encode_single_cached(text: str) -> Tuple[Tuple[float, ...], Tuple[Tuple[str, float], ...]]:
    """Single-text dense+sparse encode, memoized by exact text. Returns plain
    immutable tuples (not numpy/dict) so the cached entry can't be mutated by a
    caller holding a reference to a previous result."""
    out = _get_model().encode(
        [text], batch_size=1, max_length=512,
        return_dense=True, return_sparse=True, return_colbert_vecs=False,
    )
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
