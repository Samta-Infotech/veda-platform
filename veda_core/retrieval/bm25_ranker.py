# =============================================================================
# retrieval/bm25_ranker.py
# VEDA Phase 2 - BM25 Keyword Ranking
#
# Purpose:
#   Rank columns by keyword relevance using BM25 algorithm.
#   Captures lexical similarity independent of semantic embeddings.
# =============================================================================

import sys
import os
import json
import logging
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import get_logger

logger = get_logger(__name__)


class BM25Ranker:
    """BM25 keyword-based column ranking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 ranker.

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.documents = {}
        self.idf = {}
        self.avgdl = 0
        self.N = 0

    def fit(self, semantic_model: Dict):
        """
        Fit BM25 on retrieval documents.

        Args:
            semantic_model: Output from semantic_layer_v2.py
        """
        retrieval_docs = semantic_model.get("retrieval_documents", {})

        logger.info(f"Fitting BM25 on {len(retrieval_docs)} documents...")

        # Tokenize documents
        self.documents = {}
        all_tokens = []

        for col_id, doc in retrieval_docs.items():
            tokens = self._tokenize(doc)
            self.documents[col_id] = tokens
            all_tokens.extend(tokens)

        # Compute IDF
        self.N = len(self.documents)
        doc_freq = {}

        for tokens in self.documents.values():
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        import math
        self.idf = {
            token: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)
            for token, freq in doc_freq.items()
        }

        # Compute average document length
        self.avgdl = sum(len(tokens) for tokens in self.documents.values()) / max(
            len(self.documents), 1
        )

        logger.info(f"✓ BM25 fitted ({len(self.idf)} unique tokens)")

    def rank(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Rank columns by BM25 score for query.

        Args:
            query: User query string
            top_k: Number of results

        Returns:
            [(column_id, bm25_score), ...]
        """
        query_tokens = self._tokenize(query)

        scores = {}
        for col_id, doc_tokens in self.documents.items():
            score = self._bm25_score(query_tokens, doc_tokens)
            scores[col_id] = score

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _bm25_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Compute BM25 score for query against document."""
        doc_len = len(doc_tokens)
        score = 0.0

        for token in query_tokens:
            if token not in self.idf:
                continue

            tf = doc_tokens.count(token)
            idf = self.idf[token]

            # BM25 formula
            numerator = idf * tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(self.avgdl, 1)))

            score += numerator / denominator

        return score

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (lowercase, split, remove short tokens)."""
        return [
            token.lower()
            for token in text.split()
            if len(token) > 2 and token.isalnum()
        ]
