# =============================================================================
# retrieval/intent_boosting.py
# VEDA Phase 3d - Intent-Aware Score Boosting
#
# Purpose:
#   Adjust retrieval scores based on detected query intent
#   AGGREGATE → boost MEASURE columns
#   TEMPORAL → boost TIME_DIMENSION columns
#   MULTI_TABLE → boost FK bridge columns
#
# Input: Fused scores + intent + semantic model
# Output: Intent-adjusted scores
#
# Status: Phase 3d
# =============================================================================

import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class IntentBooster:
    """Boosts scores based on query intent and column metadata."""

    def __init__(self, semantic_model: Dict):
        """
        Initialize intent booster.

        Args:
            semantic_model: L2 semantic model (veda_semantic_model.json)
        """
        self.semantic_model = semantic_model

    def _get_column_metadata(self, col_id: str) -> Dict:
        """
        Get column metadata from semantic model.

        Args:
            col_id: Column ID (table.column)

        Returns:
            Column metadata dict
        """
        # Try to find in semantic model
        tables = self.semantic_model.get("tables", {})

        for table_name, table_meta in tables.items():
            columns = table_meta.get("columns", {})
            for col_name, col_meta in columns.items():
                if f"{table_name}.{col_name}" == col_id:
                    return col_meta

        return {}

    def boost_aggregate(self, scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Boost scores for AGGREGATE intent.

        Strategy:
        - MEASURE columns: +0.30 (what user wants to sum/avg)
        - IDENTIFIER columns: -0.40 (don't aggregate IDs)
        - ATTRIBUTE columns: -0.20 (descriptive only)
        - TIME_DIMENSION: +0.10 (useful for GROUP BY time buckets)

        Args:
            scores: List of (col_id, score) tuples

        Returns:
            Boosted scores
        """
        boosted = []

        for col_id, score in scores:
            col_meta = self._get_column_metadata(col_id)
            role = col_meta.get("analytics_role", "").upper()

            boost = 0.0

            if role == "MEASURE":
                boost = 0.30  # Primary candidates for aggregation
            elif role == "IDENTIFIER":
                boost = -0.40  # Never aggregate ID columns
            elif role == "ATTRIBUTE":
                boost = -0.20  # Less useful for aggregation
            elif role == "TIME_DIMENSION":
                boost = 0.10  # Nice for GROUP BY time

            new_score = max(0.0, score + boost)  # Clamp to [0.0, 1.0]
            boosted.append((col_id, new_score))

            logger.debug(f"  AGGREGATE: {col_id} ({role}) {score:.3f} {boost:+.2f} → {new_score:.3f}")

        return boosted

    def boost_temporal(self, scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Boost scores for TEMPORAL intent.

        Strategy:
        - TIME_DIMENSION columns: +0.30 (what user wants to filter on)
        - IDENTIFIER columns: -0.30 (IDs less relevant for temporal)
        - Others: 0.0 (no special boost)

        Args:
            scores: List of (col_id, score) tuples

        Returns:
            Boosted scores
        """
        boosted = []

        for col_id, score in scores:
            col_meta = self._get_column_metadata(col_id)
            role = col_meta.get("analytics_role", "").upper()

            boost = 0.0

            if role == "TIME_DIMENSION":
                boost = 0.30  # Temporal columns most relevant
            elif role == "IDENTIFIER":
                boost = -0.30  # IDs not relevant for time filtering

            new_score = max(0.0, score + boost)
            boosted.append((col_id, new_score))

            logger.debug(f"  TEMPORAL: {col_id} ({role}) {score:.3f} {boost:+.2f} → {new_score:.3f}")

        return boosted

    def boost_multi_table(self, scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Boost scores for MULTI_TABLE intent.

        Strategy:
        - FK bridge columns: +0.15 (essential for joins)
        - Others: 0.0 (no special boost)

        Args:
            scores: List of (col_id, score) tuples

        Returns:
            Boosted scores
        """
        boosted = []

        for col_id, score in scores:
            col_meta = self._get_column_metadata(col_id)
            is_fk = col_meta.get("is_foreign_key", False)

            boost = 0.15 if is_fk else 0.0

            new_score = max(0.0, score + boost)
            boosted.append((col_id, new_score))

            logger.debug(f"  MULTI_TABLE: {col_id} (is_fk={is_fk}) {score:.3f} {boost:+.2f} → {new_score:.3f}")

        return boosted

    def apply_history_penalty(self, scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Apply penalty to history/audit tables (always).

        Strategy:
        - History tables (_history, _audit): -0.60 (deprioritize)
        - Others: 0.0 (no penalty)

        Args:
            scores: List of (col_id, score) tuples

        Returns:
            Penalized scores
        """
        penalized = []

        for col_id, score in scores:
            # Extract table name from "table.column"
            table_name = col_id.split(".")[0] if "." in col_id else ""

            penalty = 0.0
            if table_name.endswith("_history") or table_name.endswith("_audit"):
                penalty = -0.60  # Strongly deprioritize history tables

            new_score = max(0.0, score + penalty)
            penalized.append((col_id, new_score))

            if penalty != 0.0:
                logger.debug(f"  HISTORY_PENALTY: {col_id} {score:.3f} {penalty:+.2f} → {new_score:.3f}")

        return penalized

    def boost(
        self,
        scores: List[Tuple[str, float]],
        intent: str
    ) -> List[Tuple[str, float]]:
        """
        Apply intent-aware boosting.

        Args:
            scores: List of (col_id, score) tuples from RRF fusion
            intent: Query intent (AGGREGATE, TEMPORAL, MULTI_TABLE, DIRECT, SIMPLE)

        Returns:
            Boosted and sorted scores
        """
        logger.info(f"\nIntent-aware boosting: {intent}")

        boosted = list(scores)

        # Step 1: Apply intent-specific boosts
        if intent == "AGGREGATE":
            boosted = self.boost_aggregate(boosted)
        elif intent == "TEMPORAL":
            boosted = self.boost_temporal(boosted)
        elif intent == "MULTI_TABLE":
            boosted = self.boost_multi_table(boosted)
        # DIRECT and SIMPLE: no special boost

        # Step 2: Always apply history penalty
        boosted = self.apply_history_penalty(boosted)

        # Step 3: Sort by boosted score (descending)
        boosted = sorted(boosted, key=lambda x: x[1], reverse=True)

        logger.info(f"✓ Boosted {len(boosted)} results")

        return boosted


def apply_intent_boost(
    scores: List[Tuple[str, float]],
    intent: str,
    semantic_model: Dict
) -> List[Tuple[str, float]]:
    """
    Standalone function for intent boosting.

    Args:
        scores: List of (col_id, score) tuples
        intent: Query intent
        semantic_model: L2 semantic model

    Returns:
        Boosted scores
    """
    booster = IntentBooster(semantic_model)
    return booster.boost(scores, intent)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Example scores from RRF fusion
    example_scores = [
        ("payment.amount", 0.92),
        ("payment.id", 0.88),
        ("payment.description", 0.75),
        ("payment.created_at", 0.70),
        ("payment_history.amount", 0.65),
    ]

    # Example semantic model (simplified)
    example_model = {
        "tables": {
            "payment": {
                "columns": {
                    "amount": {
                        "analytics_role": "MEASURE",
                        "is_foreign_key": False,
                    },
                    "id": {
                        "analytics_role": "IDENTIFIER",
                        "is_foreign_key": False,
                    },
                    "description": {
                        "analytics_role": "ATTRIBUTE",
                        "is_foreign_key": False,
                    },
                    "created_at": {
                        "analytics_role": "TIME_DIMENSION",
                        "is_foreign_key": False,
                    },
                }
            },
            "payment_history": {
                "columns": {
                    "amount": {
                        "analytics_role": "MEASURE",
                        "is_foreign_key": False,
                    },
                }
            },
        }
    }

    # Test AGGREGATE intent
    print("\n" + "="*60)
    print("AGGREGATE Intent (boost MEASURE, penalize IDENTIFIER)")
    print("="*60)
    boosted = apply_intent_boost(example_scores, "AGGREGATE", example_model)
    for col_id, score in boosted[:5]:
        print(f"  {col_id:30} {score:.3f}")

    # Test TEMPORAL intent
    print("\n" + "="*60)
    print("TEMPORAL Intent (boost TIME_DIMENSION)")
    print("="*60)
    boosted = apply_intent_boost(example_scores, "TEMPORAL", example_model)
    for col_id, score in boosted[:5]:
        print(f"  {col_id:30} {score:.3f}")
