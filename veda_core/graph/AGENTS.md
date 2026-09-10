# veda_core/graph/ — the unified knowledge graph traversal API

The builder is `../ingestion/unified_graph_builder.py` (→ `data/veda_unified_graph.json`).
This package is the **read/traversal** API over that artifact. Graph-expansion story:
[../../docs/RETRIEVAL.md](../../docs/RETRIEVAL.md) §Graph.

| File | Role |
|------|------|
| `query_graph.py` | `UnifiedGraph` — adjacency-indexed view of `veda_unified_graph.json` (~4k nodes / ~7.7k edges), pure stdlib, auto-reloads on file mtime/size change. Key API: `resolve_term` (synonym/alias/name → column ids, plural-aware), `get_synonyms`, `get_related_tables` (FK_TO), `shortest_path` (BFS, undirected, `max_hops=8` — only used by standalone helpers), and **`suggest_expansions(query, have_cols, have_tables, max_add=12)`** — the **Tier-1 retrieval booster** wired in `../veda/pipeline.py`: term resolution (`SYNONYM_OF` / `ALIAS_OF` edges + direct name match) + **1-hop FK join-key reach** (`REFERENCES` edges only, the specific join-key column, not the whole neighbour table). `_warn_if_stale` logs once when the graph predates its inputs. |
| `__init__.py` | Package note: derived additive view; builder is elsewhere. |

## Gotchas
- **`suggest_expansions` is NOT PPR and NOT hop-decay BFS** — it is synonym resolution +
  a single FK hop. The genuine Personalized PageRank expansion lives in
  `../query/graph_retriever.py` (Tier-2 / datalake / cross-source), which loads
  cross-source / entity / `semantic_about` edges tenant-wide and iterates
  `p = 0.15·p0 + 0.85·Pᵀp` over a cached symmetric row-normalized transition matrix.
- Additions from `suggest_expansions` get `final_score=0.0` and rely on the downstream
  cross-encoder rerank to re-score.
