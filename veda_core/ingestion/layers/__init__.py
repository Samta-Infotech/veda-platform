"""Layered ingestion (Track 2 / P4).

Five layers with explicit contracts, each a package of importable, individually
testable stage wrappers over the existing ingestion functions:

    l1_extract  — EXTRACT : touch the source (schema, FK, values)
    l2_analyze  — ANALYZE : pure transforms (types, metadata, REG, join paths)
    l3_enrich   — ENRICH  : the only LLM layer (semantic layer v2 / glossary)
    l4_index    — INDEX   : embeddings + search structures
    l5_publish  — PUBLISH : derived registries + unified graph (atomic activate)

``pipeline.run_layered_ingestion(ctx)`` composes L1→L5 in one process — a faithful
move of ``main.run_ingestion``'s body. ``dispatcher.dispatch(ctx)`` routes by
``source.type``.
"""
