# =============================================================================
# query_engine/intent_detector.py
# VEDA Phase 3 - Intent Detection (Query Classification)
#
# Purpose:
#   Detect and classify query intent into 5 types:
#   - DIRECT:       "show me user IDs" (single table, straightforward)
#   - SYNONYM:      "list all accounts" (synonyms, but single semantic concept)
#   - MULTI_TABLE:  "users and their orders" (multiple tables with joins)
#   - TEMPORAL:     "users created in last 30 days" (time-based filtering)
#   - AGGREGATE:    "count users by region" (GROUP BY, aggregation)
#
# Output: Intent class + confidence score
# =============================================================================

import sys
import os
import logging
from typing import Optional
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import get_logger

logger = get_logger(__name__)


class QueryIntent(Enum):
    """Query intent types."""
    DIRECT = "DIRECT"  # Single table, simple lookup
    SYNONYM = "SYNONYM"  # Synonymous terms, single semantic concept
    MULTI_TABLE = "MULTI_TABLE"  # Multiple tables with joins/unions
    TEMPORAL = "TEMPORAL"  # Time-based filtering (date ranges, "last N days")
    AGGREGATE = "AGGREGATE"  # GROUP BY, COUNT, SUM, AVG, aggregations


@dataclass
class IntentResult:
    """Intent detection result."""
    intent: QueryIntent
    confidence: float  # 0.0-1.0
    reasoning: str
    keywords: list  # Detected keywords that informed the decision


class IntentDetector:
    """Detect query intent without LLM (rule-based for speed)."""

    def __init__(self):
        """Initialize intent detector with keyword patterns."""
        self.intent_patterns = {
            QueryIntent.AGGREGATE: [
                "count", "sum", "avg", "average", "min", "max",
                "group by", "total", "aggregate", "metric",
                "how many", "per", "breakdown",
            ],
            QueryIntent.TEMPORAL: [
                "date", "time", "today", "yesterday", "week", "month", "year",
                "last", "past", "before", "after", "since", "until",
                "2024", "2025", "2026", "january", "february", "march",
                "april", "may", "june", "july", "august", "september",
                "october", "november", "december",
            ],
            QueryIntent.MULTI_TABLE: [
                "join", "and", "with", "related", "associated",
                "user and", "customer and", "order and", "payment and",
                "between", "across",
            ],
            QueryIntent.SYNONYM: [
                "also known as", "aka", "equivalent", "same as", "or",
            ],
        }

    def detect(self, query: str) -> IntentResult:
        """
        Detect query intent.

        Args:
            query: User natural language query

        Returns:
            IntentResult with detected intent and confidence
        """
        query_lower = query.lower()

        # Score each intent based on keyword matches
        scores = {intent: 0.0 for intent in QueryIntent}
        matched_keywords = {intent: [] for intent in QueryIntent}

        import re
        for intent, keywords in self.intent_patterns.items():
            for keyword in keywords:
                # Whole-word match for single tokens so "count" doesn't match
                # "counterparties", "and" doesn't match "thousand", etc.
                # Multi-word phrases ("group by", "how many") match as substrings.
                if " " in keyword:
                    hit = keyword in query_lower
                else:
                    hit = re.search(rf"\b{re.escape(keyword)}\b", query_lower) is not None
                if hit:
                    scores[intent] += 1.0
                    matched_keywords[intent].append(keyword)

        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 0
        if max_score > 0:
            scores = {intent: score / max_score for intent, score in scores.items()}

        # Check for multi-table patterns (higher specificity)
        if self._contains_multi_table_pattern(query_lower):
            scores[QueryIntent.MULTI_TABLE] = 0.9

        # Default to DIRECT if no strong signals
        best_intent = max(scores, key=scores.get)
        best_confidence = scores[best_intent]

        if best_confidence < 0.2:
            best_intent = QueryIntent.DIRECT
            best_confidence = 0.5

        # Generate reasoning
        reasoning = self._generate_reasoning(best_intent, matched_keywords[best_intent])

        return IntentResult(
            intent=best_intent,
            confidence=min(best_confidence, 1.0),
            reasoning=reasoning,
            keywords=matched_keywords[best_intent],
        )

    def _contains_multi_table_pattern(self, query_lower: str) -> bool:
        """Check for multi-table patterns."""
        multi_table_phrases = [
            "users and their",
            "customers and their",
            "orders and their",
            "payments from users",
            "users with orders",
            "across tables",
            "join",
        ]

        return any(phrase in query_lower for phrase in multi_table_phrases)

    def _generate_reasoning(self, intent: QueryIntent, keywords: list) -> str:
        """Generate reasoning for intent classification."""
        if not keywords:
            return f"Classified as {intent.value} (default)"

        keywords_str = ", ".join(sorted(set(keywords))[:3])
        return f"Detected {intent.value} intent from keywords: {keywords_str}"

    def classify_batch(self, queries: list) -> list:
        """
        Classify multiple queries.

        Args:
            queries: List of query strings

        Returns:
            List of IntentResult objects
        """
        return [self.detect(query) for query in queries]
