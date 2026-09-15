# =============================================================================
# retrieval/
# VEDA Phase 2/3 - the 6-signal retrieval spine (P2-2, 2026-09-10: corrected —
# was a stale filename list from before BM25 was replaced and Signal 6 landed).
#
# Components:
#   semantic_search.py      - BGE-M3 dense embeddings, source-scoped (Signal 1)
#   sparse_ranker.py        - BGE-M3 learned-sparse ranking, replaced BM25 (Signal 2)
#   signal_builder.py       - FK subgraph + FK path signals (Signals 3/4, boost-only)
#   rrf_merger.py           - Weighted Reciprocal Rank Fusion (combines all 6 signals)
#   query_enrichment.py     - Domain synonyms / concept graph / glossary expansion
#   intent_boosting.py      - Intent-aware score adjustments (AGGREGATE/TEMPORAL/...)
#   adaptive_cutoff.py      - Semantic-cliff-based dynamic top-k cutoff
#   retrieval_cache.py      - Enriched-token-hash result cache (short TTL)
#   retrieval_engine_phase3.py - Orchestrator (combines all of the above)
#
# The cross-encoder reranker itself lives in query/reranker.py, not here.
#
# Input: Semantic model (per-source, N tables × M columns)
# Output: Top-K ranked columns for user queries
# =============================================================================
