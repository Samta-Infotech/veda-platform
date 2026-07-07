# =============================================================================
# retrieval/query_enrichment.py
# VEDA Phase 3a - Query Enrichment
#
# Purpose:
#   Enriches user query with domain knowledge before retrieval
#   Uses L2 outputs: domain_synonyms, concept_graph, glossary
#
# Input: User query + semantic model artifacts
# Output: Enriched token list (8-20 terms)
#
# Status: Phase 3a
# =============================================================================

import os
import json
import logging
from typing import List, Dict, Set, Any
import re

logger = logging.getLogger(__name__)


def _singularize(word: str) -> str:
    """Lightweight English singularizer for query tokens.

    categories→category, counterparties→counterparty, documents→document,
    annotations→annotation, classes→class, comments→comment. Conservative:
    leaves words it can't confidently reduce unchanged.
    """
    w = word.lower()
    if len(w) <= 3:
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


class QueryEnricher:
    """Enriches queries using domain knowledge from L2."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize enricher with L2 artifacts.

        Args:
            data_dir: Directory containing L2 output files
        """
        # Resolve a relative data_dir against the repo root, not the process CWD — else the
        # artifacts (1333 domain synonyms, concept graph, glossary) silently fail to load
        # whenever VEDA runs from a different working directory (service / cron / tests),
        # and enrichment becomes a no-op.
        if not os.path.isabs(data_dir):
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(_root, data_dir)
        self.data_dir = data_dir
        self.domain_synonyms = {}
        self.concept_graph = {}
        self.glossary = {}
        self.semantic_model = {}
        self._merged_index = {}   # Q-3: precomputed merged enrichment index (token→expansions)

        self._load_artifacts()

    def _load_artifacts(self):
        """Load L2 output files (domain_synonyms, concept_graph, glossary)."""
        try:
            # Load domain synonyms
            with open(f"{self.data_dir}/veda_domain_synonyms.json") as f:
                self.domain_synonyms = json.load(f)
            logger.info(f"✓ Loaded domain synonyms ({len(self.domain_synonyms)} terms)")
        except FileNotFoundError:
            logger.warning(f"Domain synonyms not found at {self.data_dir}/veda_domain_synonyms.json")

        try:
            # Load concept graph
            with open(f"{self.data_dir}/veda_concept_graph.json") as f:
                self.concept_graph = json.load(f)
            logger.info(f"✓ Loaded concept graph ({len(self.concept_graph)} concepts)")
        except FileNotFoundError:
            logger.warning(f"Concept graph not found at {self.data_dir}/veda_concept_graph.json")

        try:
            # Load glossary
            with open(f"{self.data_dir}/veda_glossary.json") as f:
                self.glossary = json.load(f)
            logger.info(f"✓ Loaded glossary ({len(self.glossary)} terms)")
        except FileNotFoundError:
            logger.warning(f"Glossary not found at {self.data_dir}/veda_glossary.json")

        try:
            # Load semantic model (for metadata)
            with open(f"{self.data_dir}/veda_semantic_model.json") as f:
                self.semantic_model = json.load(f)
            logger.info(f"✓ Loaded semantic model")
        except FileNotFoundError:
            logger.warning(f"Semantic model not found at {self.data_dir}/veda_semantic_model.json")

        # Q-3: prefer the precomputed merged enrichment index (one pre-inverted
        # artifact built at ingestion). Flag-gated; the four individual files above
        # remain the fallback so enrichment still works when the index is absent.
        try:
            from config import ENRICHMENT_INDEX_ENABLED
            if ENRICHMENT_INDEX_ENABLED:
                from ingestion.enrichment_index import load_enrichment_index
                merged = load_enrichment_index()
                if merged:
                    self._merged_index = merged
                    logger.info(f"✓ Loaded merged enrichment index ({len(merged)} terms)")
        except Exception as e:
            logger.warning(f"Merged enrichment index load failed ({e}); using individual files")

    def expand_with_synonyms(self, tokens: List[str]) -> Set[str]:
        """
        Expand query tokens with domain synonyms.

        Args:
            tokens: List of query tokens

        Returns:
            Set of expanded tokens
        """
        expanded = set(tokens)

        for token in tokens:
            token_lower = token.lower()

            # Direct lookup in synonyms
            if token_lower in self.domain_synonyms:
                synonym_cols = self.domain_synonyms[token_lower]
                # Add column names from synonyms
                for col_ref in synonym_cols:
                    # Extract column name from "table.column"
                    if "." in col_ref:
                        col_name = col_ref.split(".")[-1]
                        expanded.add(col_name)
                    expanded.add(col_ref)

        return expanded

    def map_concepts(self, tokens: List[str]) -> Set[str]:
        """
        Map query tokens to business concepts and find related columns.

        Args:
            tokens: List of query tokens

        Returns:
            Set of concept-related column names
        """
        concept_tokens = set()

        for concept, mappings in self.concept_graph.items():
            # Check if any token matches concept or its related columns
            for mapping in mappings:
                col_name = mapping.get("column", "").lower()
                table_name = mapping.get("table", "").lower()

                # If query mentions a concept term that exists in concept graph
                for token in tokens:
                    token_lower = token.lower()

                    # Match column name or table name
                    if token_lower in col_name or token_lower in table_name:
                        concept_tokens.add(col_name)
                        concept_tokens.add(f"{table_name}.{col_name}")

        return concept_tokens

    def add_glossary_terms(self, tokens: List[str]) -> Set[str]:
        """
        Add business glossary terms if relevant.

        Args:
            tokens: List of query tokens

        Returns:
            Set of glossary terms
        """
        glossary_terms = set()

        for token in tokens:
            token_lower = token.lower()

            # Check if token is in glossary or similar to glossary terms
            if token_lower in self.glossary:
                glossary_terms.add(token_lower)

            # Fuzzy match: if glossary term contains token
            for glossary_term in self.glossary.keys():
                if token_lower in glossary_term.lower() or glossary_term.lower() in token_lower:
                    glossary_terms.add(glossary_term)

        return glossary_terms

    def extract_literals(self, query: str) -> Set[str]:
        """
        Extract literal values (numbers, quoted strings) from query.

        Args:
            query: User query string

        Returns:
            Set of literal values
        """
        literals = set()

        # Extract numbers (IDs, amounts, years)
        numbers = re.findall(r"\b\d+\b", query)
        literals.update(numbers)

        # Extract quoted strings
        quoted = re.findall(r'"([^"]*)"', query)
        literals.update(quoted)
        quoted = re.findall(r"'([^']*)'", query)
        literals.update(quoted)

        return literals

    def enrich(self, query: str) -> List[str]:
        """
        Enrich query with all domain knowledge sources.

        Args:
            query: User natural language query

        Returns:
            List of enriched tokens (8-20 terms)
        """
        # Step 1: Tokenize query (simple split, preserving case for lowercase)
        base_tokens = query.lower().split()

        # Add singular forms so plural queries ("categories", "counterparties")
        # bridge to the singular keys in the synonym map and to singular table/
        # column names. Keep both forms — the original may also be meaningful.
        singulars = {_singularize(t) for t in base_tokens}
        base_tokens = sorted(set(base_tokens) | {s for s in singulars if len(s) > 1})

        logger.info(f"\n{'='*60}")
        logger.info(f"QUERY ENRICHMENT: {query}")
        logger.info(f"{'='*60}")
        logger.info(f"Base tokens: {base_tokens}")

        # Step 2: Expand with synonyms
        synonyms = self.expand_with_synonyms(base_tokens)
        logger.info(f"+ Synonyms: {synonyms - set(base_tokens)}")

        # Step 3: Map concepts
        concepts = self.map_concepts(base_tokens)
        logger.info(f"+ Concepts: {concepts}")

        # Step 4: Add glossary terms
        glossary_terms = self.add_glossary_terms(base_tokens)
        logger.info(f"+ Glossary: {glossary_terms}")

        # Step 5: Extract literals
        literals = self.extract_literals(query)
        if literals:
            logger.info(f"+ Literals: {literals}")

        # Combine all
        enriched = set(base_tokens)
        enriched.update(synonyms)
        enriched.update(concepts)
        enriched.update(glossary_terms)
        enriched.update(literals)

        # Q-3: union expansions from the precomputed merged index (when loaded). This is
        # a superset of the individual-file expansions above, pre-inverted at ingestion.
        if self._merged_index:
            for tok in base_tokens:
                for exp in self._merged_index.get(tok, ()):
                    enriched.add(exp)

        # Remove empty strings and very short tokens
        enriched = {t for t in enriched if t and len(t) > 1}

        # Convert to sorted list for consistency
        enriched_list = sorted(list(enriched))

        logger.info(f"\nFinal enriched tokens ({len(enriched_list)}): {enriched_list}")

        return enriched_list


def enrich_query(query: str, data_dir: str = "data") -> List[str]:
    """
    Public API for query enrichment.

    Args:
        query: User natural language query
        data_dir: Directory containing L2 output files

    Returns:
        List of enriched tokens
    """
    enricher = QueryEnricher(data_dir)
    return enricher.enrich(query)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Example 1: Simple aggregate query
    query1 = "show me total payments last 30 days"
    enriched1 = enrich_query(query1)
    print(f"\nQuery: {query1}")
    print(f"Enriched: {enriched1}\n")

    # Example 2: Multi-table query
    query2 = "payments with user names and dates"
    enriched2 = enrich_query(query2)
    print(f"Query: {query2}")
    print(f"Enriched: {enriched2}\n")

    # Example 3: Literal lookup
    query3 = "show invoice 12345"
    enriched3 = enrich_query(query3)
    print(f"Query: {query3}")
    print(f"Enriched: {enriched3}\n")
