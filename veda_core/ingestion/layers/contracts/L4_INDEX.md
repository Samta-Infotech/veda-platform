# L4 — INDEX (contract)

> **Role:** embeddings + search structures — the model-inference layer. All
> model-bound cost is isolated here.

Module: `ingestion/layers/l4_index.py` — wraps `graph_persist`, `graph_embedder`,
`biencoder`, `bm25_index`, `enrichment_index`, `rerank_docs`.

## Consumes

| Input | Source |
|---|---|
| `ctx: SourceContext` | `source_id`, `resume`. |
| `state["graph"]` | L2 REG graph. |
| `state["scan_result"]` | L1 schema scan. |
| `state["dg_result"]` | L1 data-graph (optional). |
| `state["inference_result"]` | L2 — the biencoder embeds its per-column retrieval docs. |
| config | `UNIFIED_GRAPH_ENABLED`, `GRAPH_PERSIST_ENABLED`, `GRAPH_EMBED_ENABLED`, `BIENCODER_ENABLED`. |

## Produces (all writes are search stores)

| Stage | Side effect | Gate | Fatal? |
|---|---|---|---|
| `graph_persist` | `graph_nodes` / `graph_edges` tables | `UNIFIED_GRAPH_ENABLED and GRAPH_PERSIST_ENABLED` | non-fatal |
| `graph_embed` | `graph_node_embeddings` | `UNIFIED_GRAPH_ENABLED and GRAPH_EMBED_ENABLED` | non-fatal |
| `biencoder` | `column_embeddings_v2` (BGE-M3, 1024-dim) + `table_embeddings_v2` | `BIENCODER_ENABLED` | non-fatal |
| `bm25_index` | BM25 lexical index (Q-2) | — | non-fatal |
| `enrichment_index` | query-enrichment term index (Q-3) | — | non-fatal |
| `rerank_docs` | precomputed cross-encoder text per column (Q-4) | — | non-fatal |

## Guarantees / invariants

- **Every stage is non-fatal.** Missing indexes degrade retrieval gracefully — the
  live retrieval spine is the BGE 5-signal engine; the rest are additive signals.
- **Resume-aware:** if `ctx.resume` and `column_embeddings_v2` already has rows, the
  (slow) biencoder re-embed is skipped.
- The removed MiniLM/RELGT ensemble encoder is intentionally gone — it was a
  write-only artifact never read at query time.

## Failure semantics

Non-fatal throughout. A failed index is logged; the query tier's
`retrieval_select` degrades to whatever signals exist (BGE spine at minimum).

## Downstream consumers

`column_embeddings_v2` → query L2 Signal-1 (semantic search). `table_embeddings_v2`
→ query L3 table routing (`route_tables_semantic`). BM25 / enrichment / rerank
indexes → query L2/L2b. `graph_*` → query-time `GRAPH_EXPAND`.
