"""VEDA Unified Knowledge Graph package.

A derived, additive view over the existing artifacts (FK graph, concept graph, semantic
model, synonyms, metrics, dimensions). Does NOT replace any existing component.

  unified_graph_builder.py (in ingestion/) → builds the per-source veda_unified_graph.json artifact
  query_graph.py                            → in-memory traversal API (stdlib, <20ms)
"""
