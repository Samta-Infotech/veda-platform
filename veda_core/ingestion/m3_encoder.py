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

Ollama fallback: when FlagEmbedding isn't installed, dense encoding falls back to the
docker-internal `ollama` service's /api/embeddings with the "bge-m3" tag (`ollama pull
bge-m3` was run against that container specifically — NOT the host-native Ollama used for
the Qwen SLM, which only has qwen2.5-coder:7b). OLLAMA_EMBED_URL overrides the target if
needed. This only covers the DENSE vector — Ollama's embeddings endpoint has no learned-
sparse output, so encode_sparse()/encode_query()'s sparse half stays FlagEmbedding-only and
degrades to an empty sparse dict under the fallback (logged, not silent).
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Dict, List, Tuple

import numpy as np

from config import BGE_MODEL_NAME, BIENCODER_BATCH_SIZE, resolve_device

logger = logging.getLogger(__name__)

_MODEL = None
_LOCK = threading.Lock()
_FLAG_EMBEDDING_AVAILABLE = None   # tri-state cache: None=unchecked, True/False after first try

# Docker-internal ollama service (has "bge-m3" pulled) — deliberately independent of
# SLM_OLLAMA_BASE_URL/OLLAMA_URL, which may point at the host-native Ollama for Qwen instead.
OLLAMA_EMBED_URL = os.environ.get("OLLAMA_EMBED_URL", "http://ollama:11434")
OLLAMA_EMBED_MODEL = "bge-m3"


def _flag_embedding_available() -> bool:
    global _FLAG_EMBEDDING_AVAILABLE
    if _FLAG_EMBEDDING_AVAILABLE is None:
        try:
            import FlagEmbedding  # noqa: F401
            _FLAG_EMBEDDING_AVAILABLE = True
        except ImportError:
            logger.warning(
                "FlagEmbedding not installed — falling back to ollama (%s, model=%s) for "
                "dense encoding only; learned-sparse output is unavailable in this mode.",
                OLLAMA_EMBED_URL, OLLAMA_EMBED_MODEL,
            )
            _FLAG_EMBEDDING_AVAILABLE = False
    return _FLAG_EMBEDDING_AVAILABLE


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


def _ollama_embed_batch(texts: List[str]) -> np.ndarray:
    """Dense-only fallback via the docker ollama service's /api/embeddings — one call per
    text (that endpoint has no batch mode). Fine for ingestion throughput."""
    import requests

    vectors = []
    for t in texts:
        r = requests.post(
            f"{OLLAMA_EMBED_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": t},
            timeout=60,
        )
        r.raise_for_status()
        vectors.append(r.json()["embedding"])
    return np.asarray(vectors, dtype=np.float32)


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
    if not _flag_embedding_available():
        return _l2_normalize(_ollama_embed_batch(texts))
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
    if not _flag_embedding_available():
        # No learned-sparse equivalent via Ollama — degrade to empty dicts rather than crash.
        return [{} for _ in texts]
    out = _get_model().encode(
        texts, batch_size=BIENCODER_BATCH_SIZE, max_length=512,
        return_dense=False, return_sparse=True, return_colbert_vecs=False,
    )
    return [_clean_sparse(lw) for lw in out["lexical_weights"]]


def encode_query(text: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """Encode one query → (1024-dim normalized dense vector, sparse weight dict) in a
    SINGLE forward pass (dense + sparse share the encode call)."""
    if not _flag_embedding_available():
        dense = _l2_normalize(_ollama_embed_batch([text]))[0]
        return dense, {}
    out = _get_model().encode(
        [text], batch_size=1, max_length=512,
        return_dense=True, return_sparse=True, return_colbert_vecs=False,
    )
    dense = _l2_normalize(np.asarray(out["dense_vecs"], dtype=np.float32))[0]
    sparse = _clean_sparse(out["lexical_weights"][0])
    return dense, sparse


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
        out = encode_dense(texts)  # already L2-normalized (n, 1024)
        return out[0] if single else out

    def get_sentence_embedding_dimension(self) -> int:
        return 1024


_DENSE_ENCODER = _DenseEncoder()


def get_dense_encoder() -> "_DenseEncoder":
    """The shared dense-encoder facade (one instance, one underlying model)."""
    return _DENSE_ENCODER
