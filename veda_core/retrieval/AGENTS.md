# veda_core/retrieval/ — the Tier-1 hybrid retrieval spine

`get_engine(sm).retrieve(query, intent, top_k)` runs a **6-signal → weighted-RRF →
intent-boost → adaptive-cutoff** pipeline. One warm engine per `(source, tenant)` scope
(built in `../veda/runtime.py`), sharing one BGE-M3 model. Full walkthrough:
[../../docs/RETRIEVAL.md](../../docs/RETRIEVAL.md).

| File | Role |
|------|------|
| `retrieval_engine_phase3.py` | `RetrievalEnginePhase3` — the orchestrator. `retrieve()` = 6 signals → RRF → intent-boost → adaptive-cutoff → (disabled) cache. The only real entry point. **Class name / docstring still say "5-Signal"** — stale. |
| `semantic_search.py` | **Signal 1** — `SemanticSearchEngine.search()`: BGE-M3 dense, raw query, cosine ANN over `column_embeddings_v2` (in the `veda_engine` DB) via `storage_adapters.reader.ann_search` (source-scoped, `VEDA_ANN_VIA_ADAPTER=1`, the default) or the engine's own pgvector (dev/CLI, **unscoped**). `DENSE_ID_REMAP` maps UUID `col_id` → `table.column`. |
| `sparse_ranker.py` | **Signal 2** — `SparseRanker`: BGE-M3 **learned-sparse** weights, **replaced BM25** (WP3). `warm_from_store()` loads persisted `column_sparse_v1` / `table_sparse_v1`; `fit()` is a dev fallback (capped at `SPARSE_FIT_MAX_DOCS=300`, else Signal 2 is silently skipped). `table_scores()` feeds Signal 6. |
| `signal_builder.py` | Source data for **Signals 3, 4, 5**. `build_signals()` builds a FK graph (prefers `runtime.get_graph()` edges, falls back to substrate `fk_adjacency`); per column `fk_signal` (0.5 FK / 0.7 referenced) + `subgraph_signal` (`min(degree/10, 1)`). `build_value_index()` → `{value_token: [col_id]}`. Dead import: `from schema.real_schema import get_real_schema`. |
| `rrf_merger.py` | `RRFMerger(k=60)`. `merge()` — weighted RRF over 6 signals, weights from `config.FUSION_WEIGHTS` (**currently all `1.0`** — identity). Signal 6 only boosts existing candidates. |
| `intent_boosting.py` | `IntentBooster.boost(fused, intent)` — additive ±0.1–0.6 deltas on `analytics_role` (these **dwarf** the RRF score range) + a −0.60 `_history` / `_audit` table penalty; re-sorts. |
| `adaptive_cutoff.py` | `AdaptiveCutoff(gap_threshold=0.28, min_k=5, max_k=20, hard_limit=15)` — cut at the biggest consecutive-score gap ("semantic cliff"). |
| `query_enrichment.py` | `QueryEnricher.enrich(query)` → sorted token list. Loads domain synonyms / concept graph / glossary + the precomputed `enrichment_index` (WP7). Feeds Signals 2 & 5 only (dense encodes the raw query, WP1). |
| `retrieval_cache.py` | `RetrievalCache` — file or Redis, TTL 300 s. **Disabled** (`RETRIEVAL_CACHE_ENABLED=False`); `retrieve()` always called with `use_cache=False`. |
| `__init__.py` | Package header — **stale** (lists `embedding_layer.py`, `bm25_ranker`, `cross_encoder.py`, `retrieval_engine.py` — none exist). |

## Gotchas
- **6 signals, not 5.** Signal 6 (table-first prior, WP4) is undocumented in the class name.
- **`ENCODER_MODE` is dead** and the relgt/light-text/hybrid/MiniLM ensemble is gone. One
  `BAAI/bge-m3` singleton (`../ingestion/m3_encoder.py`) does dense + learned-sparse.
- Signals 1 & 2 run concurrently in a `ThreadPoolExecutor(max_workers=2)`.
- Signals 3 & 4 are **static per-column scalars**, not graph traversal at query time.
- The cross-encoder rerank (`../query/reranker.py`) runs downstream in `../veda/pipeline.py`,
  not here. If the reranker model isn't cached locally, every query silently degrades to
  pure RRF order.
- Graph expansion: `../graph/query_graph.suggest_expansions` (Tier-1 booster = synonym +
  1-hop FK, NOT PPR) vs `../query/graph_retriever.run_graph_retrieval` (Tier-2 = real PPR).
