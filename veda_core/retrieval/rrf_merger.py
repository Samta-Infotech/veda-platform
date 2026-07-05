# =============================================================================
# retrieval/rrf_merger.py
# VEDA Phase 2 - Reciprocal Rank Fusion (RRF)
#
# Purpose:
#   Merge 4 ranking signals (semantic, BM25, FK, subgraph) via RRF.
#   RRF is robust to signal variance and doesn't require weight tuning.
#
# Formula: RRF(d) = Σ 1/(k + rank(d))  for each signal
#   where k=60 (standard), rank is 1-indexed
# =============================================================================

import sys
import os
import logging
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import get_logger

logger = get_logger(__name__)


class RRFMerger:
    """Reciprocal Rank Fusion for merging multiple ranking signals."""

    def __init__(self, k: int = 60):
        """
        Initialize RRF merger.

        Args:
            k: Constant in RRF formula (higher = smoother)
        """
        self.k = k

    def merge(
        self,
        semantic_ranking: List[Tuple[str, float]],
        bm25_ranking: List[Tuple[str, float]],
        fk_signals: Dict[str, float],
        subgraph_signals: Dict[str, float],
        value_signals: Dict[str, float] = None,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Merge 5 signals using RRF (Phase 3 enhanced).

        Args:
            semantic_ranking: [(col_id, similarity), ...] from BGE-M3
            bm25_ranking: [(col_id, score), ...] from BM25 ranker
            fk_signals: {col_id: signal_value} from signal builder
            subgraph_signals: {col_id: signal_value} from signal builder
            value_signals: {col_id: signal_value} from value index (NEW - Phase 3)
            top_k: Number of results to return

        Returns:
            [(col_id, rrf_score), ...]
        """
        if value_signals is None:
            value_signals = {}

        logger.info("Merging 5 signals via RRF (Phase 3)...")

        # Create rank dictionaries (rank is 1-indexed)
        semantic_ranks = self._create_rank_dict(semantic_ranking)
        bm25_ranks = self._create_rank_dict(bm25_ranking)
        value_ranks = self._create_rank_dict([(col_id, score) for col_id, score in value_signals.items()])

        # Collect all candidates
        all_candidates = set()
        all_candidates.update(semantic_ranks.keys())
        all_candidates.update(bm25_ranks.keys())
        all_candidates.update(fk_signals.keys())
        all_candidates.update(subgraph_signals.keys())
        all_candidates.update(value_ranks.keys())

        # Compute RRF score for each candidate
        rrf_scores = {}

        for col_id in all_candidates:
            score = 0.0

            # Signal 1: Semantic search (BGE-M3, Phase 3)
            if col_id in semantic_ranks:
                rank = semantic_ranks[col_id]
                score += 1 / (self.k + rank)

            # Signal 2: BM25 keyword matching
            if col_id in bm25_ranks:
                rank = bm25_ranks[col_id]
                score += 1 / (self.k + rank)

            # Signal 3: FK subgraph proximity
            sg_score = subgraph_signals.get(col_id, 0.0)
            if sg_score > 0:
                virtual_rank = max(1, int((1 - sg_score) * self.k))
                score += 1 / (self.k + virtual_rank)

            # Signal 4: FK path bridges
            fk_score = fk_signals.get(col_id, 0.0)
            if fk_score > 0:
                virtual_rank = max(1, int((1 - fk_score) * self.k))
                score += 1 / (self.k + virtual_rank)

            # Signal 5: Value index direct lookups (NEW - Phase 3)
            if col_id in value_ranks:
                rank = value_ranks[col_id]
                score += 1 / (self.k + rank)

            rrf_scores[col_id] = score

        # Sort by RRF score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        logger.info(f"✓ RRF merged {len(all_candidates)} candidates (5-signal)")

        return ranked[:top_k]

    def _create_rank_dict(self, ranking: List[Tuple[str, float]]) -> Dict[str, int]:
        """Convert ranking list to rank dictionary (1-indexed)."""
        return {col_id: i + 1 for i, (col_id, _) in enumerate(ranking)}
