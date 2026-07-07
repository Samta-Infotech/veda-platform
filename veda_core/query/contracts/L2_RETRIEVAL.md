# L2 — RETRIEVAL (contract)

> **Role:** find the columns/tables relevant to the query. The 5-signal engine is
> the recall spine; graph-expand and cross-encoder rerank refine it.

Modules: `retrieval/retrieval_engine_phase3.py`, `retrieval/semantic_search.py`,
`retrieval/*` (bm25, enrichment, rrf_merger, signal_builder), `query/reranker.py`,
`query/retrieval_select.py`, `veda/query_enhancement.py`, `graph/query_graph.py`.

## Consumes

| Input | Source |
|---|---|
| `query` / `_search` | L1 cleaned query, optionally expanded by `enhance_query` (L2+). |
| `intent` | L4 intent hint. |
| `sm` | the scope's semantic model — engine is built per (source, tenant) so signals read the right source's stores. |
| ingestion artifacts | `column_embeddings_v2` (BGE-M3), BM25 index, FK subgraph/paths, `column_values`, enrichment index, unified graph. |

## Produces

- `results`: `List[RetrievalResult]` (`col_id="t.c"`, `column_name`, `table_name`,
  `final_score`, `semantic_type`), top_k≈15, ordered by score.
- `SelectedRetrieval` (via `retrieval_select`): `columns`, `tables`, `join_path`,
  `short_circuit`, `source ∈ {schema_link, v2_rerank, graph, legacy}`,
  `semantic_layer_result`.

## Pipeline within L2

1. **L2+ Enhance** (`QUERY_ENHANCEMENT_ENABLED`): synonym/alias expansion of the
   search string. Purely additive; failure → original query.
2. **5-signal retrieve** (`get_engine(sm).retrieve`): BGE-M3 semantic + BM25 lexical
   + FK subgraph + FK path + value signal → **RRF-fused**.
3. **L2g Graph expand** (`GRAPH_EXPAND_ENABLED`): adds synonym/alias + FK-neighbour
   columns the 5-signal engine missed (`final_score=0.0`, re-scored by rerank).
4. **L2b Primary rerank** (`PRIMARY_RERANK_ENABLED`): cross-encoder re-scores all
   candidates and overwrites `final_score`, so anchor selection ranks off reranked
   scores. Uses generated `domain_synonyms` (no hardcoded business map).

## Guarantees / invariants

- **`retrieval_select` is the single source of truth** for what reaches L3 —
  interactive `main.py` and `evaluation/evaluator.py` both call it so entry points
  can't diverge.
- Every refinement (enhance, graph-expand, rerank) is **flag-guarded and fully
  try/except'd**: on any failure, retrieval is byte-identical to the RRF baseline.
- Signal-1 store is source-scoped → no cross-source leakage in multi-source mode.

## Failure semantics

Non-fatal per signal — a missing index degrades to the remaining signals (BGE spine
at minimum). Empty results → L3 emits `no_table` refusal.

## Downstream consumers

`results` → L3 (`select_primary_table` / `vet_primary`, candidate tables) and L4b
join planning.
