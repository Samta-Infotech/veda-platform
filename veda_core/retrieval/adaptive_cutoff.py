# =============================================================================
# retrieval/adaptive_cutoff.py
# VEDA Phase 3e - Adaptive Cutoff via Semantic Cliff Detection
#
# Purpose:
#   Smart cutoff instead of forced top-K
#   Finds biggest score gap (semantic cliff) and cuts there
#   Prevents low-confidence results from contaminating top-15
#
# Input: Ranked results with scores
# Output: Adaptive top-K results (usually 12-18)
#
# Status: Phase 3e
# =============================================================================

import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class AdaptiveCutoff:
    """Adaptive cutoff via semantic cliff detection."""

    def __init__(
        self,
        gap_threshold: float = 0.28,
        min_k: int = 5,
        max_k: int = 20,
        hard_limit: int = 15
    ):
        """
        Initialize adaptive cutoff.

        Args:
            gap_threshold: Minimum gap to be considered a "cliff" (0.0-1.0)
            min_k: Minimum results to return
            max_k: Maximum results to consider
            hard_limit: Default cutoff if no cliff found
        """
        self.gap_threshold = gap_threshold
        self.min_k = min_k
        self.max_k = max_k
        self.hard_limit = hard_limit

    def find_semantic_cliff(self, scores: List[float]) -> Optional[Tuple[int, float]]:
        """
        Find the biggest score gap (semantic cliff).

        A semantic cliff is a large drop in score between consecutive results,
        indicating a transition from high-confidence to low-confidence results.

        Args:
            scores: List of relevance scores (descending)

        Returns:
            Tuple of (cliff_index, gap_size) or None if no cliff found
        """
        if len(scores) <= self.min_k:
            return None

        max_gap = 0.0
        cliff_idx = None

        # Check for gaps in the scoring range [min_k, max_k)
        for i in range(self.min_k - 1, min(len(scores) - 1, self.max_k)):
            gap = scores[i] - scores[i + 1]

            if gap > max_gap:
                max_gap = gap
                cliff_idx = i + 1

        logger.debug(f"Found gaps:")
        logger.debug(f"  Max gap: {max_gap:.4f} at position {cliff_idx}")

        # Return cliff only if gap exceeds threshold
        if max_gap > self.gap_threshold:
            logger.debug(f"  ✓ Cliff detected (gap {max_gap:.4f} > threshold {self.gap_threshold})")
            return (cliff_idx, max_gap)
        else:
            logger.debug(f"  ✗ No cliff (gap {max_gap:.4f} ≤ threshold {self.gap_threshold})")
            return None

    def visualize_scores(self, scores: List[float], cliff_idx: Optional[int] = None):
        """
        Visualize score distribution and cliff (for debugging).

        Args:
            scores: List of scores
            cliff_idx: Position of cliff (if any)
        """
        # Create simple ASCII chart
        import io
        chart = io.StringIO()

        chart.write("\nScore distribution:\n")

        for i, score in enumerate(scores[:self.max_k]):
            # Scale to 40 chars
            bars = int(score * 40)
            bar = "█" * bars + "░" * (40 - bars)

            marker = ""
            if cliff_idx and i == cliff_idx:
                marker = " ← CLIFF"

            chart.write(f"  {i:2d}: {bar} {score:.3f}{marker}\n")

        logger.debug(chart.getvalue())

    def cutoff(self, ranked_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Apply adaptive cutoff to ranked results.

        Args:
            ranked_results: List of (col_id, score) tuples (sorted descending)

        Returns:
            Adaptively cut results
        """
        if len(ranked_results) <= self.min_k:
            logger.info(f"✓ Results ({len(ranked_results)}) ≤ min_k ({self.min_k}), returning all")
            return ranked_results

        # Extract scores
        scores = [score for col_id, score in ranked_results]

        # Find semantic cliff
        cliff = self.find_semantic_cliff(scores)

        # Visualize for debugging
        self.visualize_scores(scores, cliff[0] if cliff else None)

        # Determine cutoff
        if cliff:
            cutoff_k = cliff[0]
            logger.info(f"✓ Semantic cliff found at position {cutoff_k} (gap {cliff[1]:.4f})")
        else:
            cutoff_k = self.hard_limit
            logger.info(f"✓ No cliff found, using hard limit (top-{self.hard_limit})")

        # Apply bounds
        cutoff_k = max(self.min_k, min(cutoff_k, self.max_k))

        logger.info(f"✓ Adaptive cutoff: top-{cutoff_k}")

        return ranked_results[:cutoff_k]


def adaptive_cutoff(
    ranked_results: List[Tuple[str, float]],
    gap_threshold: float = 0.28,
    min_k: int = 5,
    max_k: int = 20,
    hard_limit: int = 15
) -> List[Tuple[str, float]]:
    """
    Standalone function for adaptive cutoff.

    Args:
        ranked_results: List of (col_id, score) tuples
        gap_threshold: Minimum gap to detect cliff
        min_k: Minimum results to return
        max_k: Maximum results to consider
        hard_limit: Default cutoff if no cliff

    Returns:
        Cutoff results
    """
    cutter = AdaptiveCutoff(gap_threshold, min_k, max_k, hard_limit)
    return cutter.cutoff(ranked_results)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Example 1: Clear semantic cliff
    print("\n" + "="*60)
    print("Example 1: Clear semantic cliff at position 3")
    print("="*60)
    results1 = [
        ("payment.amount", 0.95),
        ("payment.fee", 0.92),
        ("payment.total", 0.88),
        ("payment.description", 0.15),  # ← BIG GAP here (0.73)
        ("payment.id", 0.12),
        ("user.name", 0.10),
    ]
    cut1 = adaptive_cutoff(results1)
    print(f"\nInput: {len(results1)} results")
    print(f"Output: {len(cut1)} results")
    for col_id, score in cut1:
        print(f"  {col_id:30} {score:.3f}")

    # Example 2: No clear cliff (gradual decline)
    print("\n" + "="*60)
    print("Example 2: No clear cliff (gradual decline)")
    print("="*60)
    results2 = [
        ("payment.amount", 0.95),
        ("payment.fee", 0.92),
        ("payment.total", 0.88),
        ("payment.description", 0.80),  # ← Small gap (0.08)
        ("payment.date", 0.78),
        ("payment.status", 0.75),
        ("user.name", 0.72),
    ]
    cut2 = adaptive_cutoff(results2)
    print(f"\nInput: {len(results2)} results")
    print(f"Output: {len(cut2)} results (hard_limit applied)")
    for col_id, score in cut2:
        print(f"  {col_id:30} {score:.3f}")

    # Example 3: Very high confidence results
    print("\n" + "="*60)
    print("Example 3: All high confidence (no cliff)")
    print("="*60)
    results3 = [
        ("payment.amount", 0.99),
        ("payment.total", 0.98),
        ("payment.sum", 0.97),
        ("payment.value", 0.96),
    ]
    cut3 = adaptive_cutoff(results3)
    print(f"\nInput: {len(results3)} results")
    print(f"Output: {len(cut3)} results")
    for col_id, score in cut3:
        print(f"  {col_id:30} {score:.3f}")
