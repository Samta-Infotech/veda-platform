"""L4 INDEX · persisted BM25 index (Q-2).

Fits the BM25 corpus (tokenization + IDF) ONCE at ingestion and persists it, so the
query tier loads a ready-made index at warm instead of re-fitting per process
(``bm25_ranker.fit`` survives only inside ingestion). Scores are identical.

Pure transform of the on-disk semantic model → no source-DB touch, non-fatal.
"""
from __future__ import annotations

import json
import os
from typing import Dict


def _index_path() -> str:
    from config import artifact_path
    return artifact_path("veda_bm25_index.json")


def build_bm25_index(source_id: str = "", verbose: bool = False) -> Dict:
    """Fit BM25 on the semantic model's retrieval_documents and persist the index."""
    from config import SEMANTIC_MODEL_FILE
    from retrieval.bm25_ranker import BM25Ranker

    if not os.path.exists(SEMANTIC_MODEL_FILE):
        raise FileNotFoundError(f"semantic model not found: {SEMANTIC_MODEL_FILE}")
    with open(SEMANTIC_MODEL_FILE) as f:
        semantic_model = json.load(f)

    ranker = BM25Ranker()
    ranker.fit(semantic_model)
    state = ranker.to_dict()

    path = _index_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f)
    if verbose:
        print(f"  [bm25_index] {state['N']} docs, {len(state['idf'])} tokens → {path}")
    return {"docs": state["N"], "tokens": len(state["idf"]), "path": path}


def load_bm25_index():
    """Query-tier warm loader: return a BM25Ranker restored from the persisted index,
    or None if it was never built (caller falls back to a live fit())."""
    from retrieval.bm25_ranker import BM25Ranker
    path = _index_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            state = json.load(f)
        return BM25Ranker().load(state)
    except Exception:
        return None
