# =============================================================================
# retrieval/
# VEDA Phase 2 - L3 Retrieval Layer
#
# Components:
#   embedding_layer.py    - BGE-M3 semantic embeddings (1024-dim)
#   sparse_ranker.py      - BGE-M3 learned-sparse ranking (Signal 2)
#   signal_builder.py     - FK adjacency + Subgraph signals
#   rrf_merger.py         - Reciprocal Rank Fusion (combine 4 signals)
#   cross_encoder.py      - Cross-encoder reranking (final refinement)
#   retrieval_engine.py   - Orchestrator (combines all 5 components)
#
# Input: Semantic model (10 tables × 123 columns)
# Output: Top-K ranked columns for user queries
# =============================================================================
