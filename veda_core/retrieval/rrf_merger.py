# =============================================================================
# retrieval/rrf_merger.py
# VEDA Phase 2 - Reciprocal Rank Fusion (RRF)
#
# Purpose:
#   Merge ranking signals (dense, sparse, FK, subgraph, value) via RRF.
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
        sparse_ranking: List[Tuple[str, float]],
        fk_signals: Dict[str, float],
        subgraph_signals: Dict[str, float],
        value_signals: Dict[str, float] = None,
        table_prior_signals: Dict[str, float] = None,
        weights: Dict[str, float] = None,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Merge 5 signals using RRF (Phase 3 enhanced).

        Args:
            semantic_ranking: [(col_id, similarity), ...] from BGE-M3
            sparse_ranking: [(col_id, score), ...] from the sparse ranker
            fk_signals: {col_id: signal_value} from signal builder
            subgraph_signals: {col_id: signal_value} from signal builder
            value_signals: {col_id: signal_value} from value index (NEW - Phase 3)
            top_k: Number of results to return

        Returns:
            [(col_id, rrf_score), ...]
        """
        if value_signals is None:
            value_signals = {}

        # WP6: weighted RRF — score(d) = Σ_s w_s / (k + rank_s(d)). Identity weights
        # (all 1.0) reproduce the pre-WP6 unweighted ranking bit-for-bit. Weights are
        # loaded from config.FUSION_WEIGHTS unless the caller passes an override (the
        # tuning harness sweeps them).
        if weights is None:
            try:
                from config import FUSION_WEIGHTS
                weights = FUSION_WEIGHTS
            except Exception:
                weights = {}
        w_dense  = float(weights.get("dense", 1.0))
        w_sparse = float(weights.get("sparse", 1.0))
        w_sub    = float(weights.get("subgraph", 1.0))
        w_fk     = float(weights.get("fk", 1.0))
        w_value  = float(weights.get("value", 1.0))
        w_tprior = float(weights.get("table_prior", 1.0))

        logger.info("Merging 6 signals via weighted RRF (WP6)...")

        # Create rank dictionaries (rank is 1-indexed)
        semantic_ranks = self._create_rank_dict(semantic_ranking)
        sparse_ranks = self._create_rank_dict(sparse_ranking)
        value_ranks = self._create_rank_dict([(col_id, score) for col_id, score in value_signals.items()])

        # Collect all candidates
        all_candidates = set()
        all_candidates.update(semantic_ranks.keys())
        all_candidates.update(sparse_ranks.keys())
        all_candidates.update(fk_signals.keys())
        all_candidates.update(subgraph_signals.keys())
        all_candidates.update(value_ranks.keys())

        # Compute RRF score for each candidate
        rrf_scores = {}

        for col_id in all_candidates:
            score = 0.0

            # Signal 1: Semantic search (BGE-M3 dense)
            if col_id in semantic_ranks:
                rank = semantic_ranks[col_id]
                score += w_dense * (1 / (self.k + rank))

            # Signal 2: learned-sparse (M3) matching
            if col_id in sparse_ranks:
                rank = sparse_ranks[col_id]
                score += w_sparse * (1 / (self.k + rank))

            # Signal 3: FK subgraph proximity
            sg_score = subgraph_signals.get(col_id, 0.0)
            if sg_score > 0:
                virtual_rank = max(1, int((1 - sg_score) * self.k))
                score += w_sub * (1 / (self.k + virtual_rank))

            # Signal 4: FK path bridges
            fk_score = fk_signals.get(col_id, 0.0)
            if fk_score > 0:
                virtual_rank = max(1, int((1 - fk_score) * self.k))
                score += w_fk * (1 / (self.k + virtual_rank))

            # Signal 5: Value index direct lookups
            if col_id in value_ranks:
                rank = value_ranks[col_id]
                score += w_value * (1 / (self.k + rank))

            # Signal 6: Table-first prior (WP4) — each column inherits its table's
            # affinity (dense ⊕ sparse, keyed by table_name). Soft: boosts existing
            # candidates only, never filters. col_id is "table.col".
            if table_prior_signals:
                tname = col_id.rsplit(".", 1)[0] if "." in col_id else ""
                tp = table_prior_signals.get(tname, 0.0)
                if tp > 0:
                    virtual_rank = max(1, int((1 - tp) * self.k))
                    score += w_tprior * (1 / (self.k + virtual_rank))

            rrf_scores[col_id] = score

        # Sort by RRF score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        logger.info(f"✓ RRF merged {len(all_candidates)} candidates (6-signal weighted)")

        return ranked[:top_k]

    def _create_rank_dict(self, ranking: List[Tuple[str, float]]) -> Dict[str, int]:
        """Convert ranking list to rank dictionary (1-indexed)."""
        return {col_id: i + 1 for i, (col_id, _) in enumerate(ranking)}
