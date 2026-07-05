# =============================================================================
# ingestion/semantic_postprocessor.py
# VEDA Final Architecture — Post-Processing (Synonyms + Concept Graph)
#
# Purpose:
#   Build domain synonyms and concept graph from semantic model.
#   All deterministic (zero LLM calls).
#
# Output:
#   veda_domain_synonyms.json — term → columns mapping
#   veda_concept_graph.json — concept → columns mapping
# =============================================================================

import sys
import os
import json
from typing import Dict, Any, List, Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    DOMAIN_SYNONYMS_ENABLED,
    CONCEPT_GRAPH_ENABLED,
    CONCEPT_GRAPH_CONCEPTS,
    DOMAIN_SYNONYMS_FILE,
    CONCEPT_GRAPH_FILE,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def build_domain_synonyms(
    glossary: Dict[str, str],
    semantic_model: Dict[str, Any],
) -> Dict[str, List[str]]:
    """
    Build domain synonyms: term → [columns mapping].

    Rules:
    - Each glossary term maps to columns with matching semantic_type or definition
    - Broad terms capped to 2 most relevant columns
    - History tables excluded
    - Only core tables included

    Args:
        glossary: Business glossary from Stage 2
        semantic_model: Semantic model from Stages 3-5

    Returns:
        {term: [column_names]}
    """
    synonyms = {}
    columns = semantic_model.get("columns", {})
    history_keywords = {"history", "historical", "audit_log", "log"}

    for term, definition in glossary.items():
        term_lower = term.lower()
        candidates = []

        # Score each column for relevance
        for col_key, col_info in columns.items():
            table_name = col_key.split(".")[0]

            # Skip history tables
            if any(kw in table_name.lower() for kw in history_keywords):
                continue

            col_name_lower = col_info.get("col_name", "").lower()
            business_def_lower = col_info.get("business_definition", "").lower()
            semantic_type = col_info.get("semantic_type", "").upper()

            # Check for matches
            score = 0

            # Exact column name match
            if col_name_lower == term_lower:
                score += 100

            # Substring in column name
            if term_lower in col_name_lower:
                score += 50

            # Substring in business definition
            if term_lower in business_def_lower:
                score += 30

            # Check aliases
            aliases = col_info.get("aliases", [])
            for alias in aliases:
                if alias.lower() == term_lower:
                    score += 80
                elif term_lower in alias.lower():
                    score += 40

            if score > 0:
                candidates.append((col_key, score))

        # Sort by score, cap to 2 most relevant
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_columns = [col_key.replace(".", "_") for col_key, _ in candidates[:2]]

        if top_columns:
            synonyms[term] = top_columns

    logger.info(f"Built domain synonyms: {len(synonyms)} terms")
    return synonyms


def build_concept_graph(
    semantic_model: Dict[str, Any],
    concepts: List[str] = None,
) -> Dict[str, List[str]]:
    """
    Build concept graph: concept → [columns mapping].

    Rules:
    - PERSON: columns with IDENTIFIER role or name-like patterns
    - AMOUNT: columns with MEASURE role and MONETARY semantic type
    - DATE: columns with TIME_DIMENSION role
    - METRIC: columns with MEASURE role

    Args:
        semantic_model: Semantic model from Stages 3-5
        concepts: List of concepts to extract (default: PERSON, AMOUNT, DATE, METRIC)

    Returns:
        {concept: [column_names]}
    """
    if concepts is None:
        concepts = CONCEPT_GRAPH_CONCEPTS

    concept_graph = {}
    columns = semantic_model.get("columns", {})

    for concept in concepts:
        concept_upper = concept.upper()
        matched_columns = []

        for col_key, col_info in columns.items():
            table_name, col_name = col_key.split(".")
            analytics_role = col_info.get("analytics_role", "").upper()
            semantic_type = col_info.get("semantic_type", "").upper()
            col_name_lower = col_name.lower()

            match = False

            if concept_upper == "PERSON":
                # IDENTIFIER columns that look like names/persons
                if analytics_role == "IDENTIFIER":
                    if any(
                        kw in col_name_lower
                        for kw in ["name", "user", "payer", "receiver", "owner", "agent"]
                    ):
                        match = True

            elif concept_upper == "AMOUNT":
                # MEASURE columns with MONETARY semantic type
                if analytics_role == "MEASURE" and semantic_type == "MONETARY":
                    match = True

            elif concept_upper == "DATE":
                # TIME_DIMENSION columns
                if analytics_role == "TIME_DIMENSION":
                    match = True

            elif concept_upper == "METRIC":
                # MEASURE columns (any semantic type)
                if analytics_role == "MEASURE":
                    match = True

            if match:
                matched_columns.append(col_key.replace(".", "_"))

        if matched_columns:
            concept_graph[concept_upper] = matched_columns

    logger.info(f"Built concept graph: {len(concept_graph)} concepts")
    return concept_graph


def save_domain_synonyms(synonyms: Dict[str, List[str]], output_file: str = None):
    """Save domain synonyms to JSON file."""
    if output_file is None:
        output_file = DOMAIN_SYNONYMS_FILE

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(synonyms, f, indent=2)

    logger.info(f"Domain synonyms saved to {output_file}: {len(synonyms)} terms")


def save_concept_graph(concept_graph: Dict[str, List[str]], output_file: str = None):
    """Save concept graph to JSON file."""
    if output_file is None:
        output_file = CONCEPT_GRAPH_FILE

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(concept_graph, f, indent=2)

    logger.info(f"Concept graph saved to {output_file}: {len(concept_graph)} concepts")


def load_domain_synonyms(input_file: str = None) -> Dict[str, List[str]]:
    """Load domain synonyms from JSON file."""
    if input_file is None:
        input_file = DOMAIN_SYNONYMS_FILE

    if not os.path.exists(input_file):
        logger.warning(f"Domain synonyms file not found: {input_file}")
        return {}

    with open(input_file, "r") as f:
        synonyms = json.load(f)

    logger.info(f"Domain synonyms loaded from {input_file}: {len(synonyms)} terms")
    return synonyms


def load_concept_graph(input_file: str = None) -> Dict[str, List[str]]:
    """Load concept graph from JSON file."""
    if input_file is None:
        input_file = CONCEPT_GRAPH_FILE

    if not os.path.exists(input_file):
        logger.warning(f"Concept graph file not found: {input_file}")
        return {}

    with open(input_file, "r") as f:
        concept_graph = json.load(f)

    logger.info(f"Concept graph loaded from {input_file}: {len(concept_graph)} concepts")
    return concept_graph
