# RETRIEVAL — the Tier-1 hybrid spine

Deep reference for `veda_core/retrieval/` — the retrieval step of the deterministic SQL
head (`veda/pipeline.py` L2). Expands [ARCHITECTURE.md](ARCHITECTURE.md) §5. Written from a
direct read of the source on `master` (2026-09-09); where a piece is dormant, a fallback,
or a stale comment, it is called out.

> **Authority.** When code and prose disagree: `veda/pipeline.py` + `veda_hybrid.py` win
> for engine behavior. The class name and docstring of `RetrievalEnginePhase3` still say
> **"5-Signal"** — that string is stale (see §2). The engine builds and fuses **six**.

Related: [ARCHITECTURE.md](ARCHITECTURE.md) ·
[DATA_SOURCES_EMBEDDINGS_AND_GRAPHS.md](DATA_SOURCES_EMBEDDINGS_AND_GRAPHS.md) ·
[SEMANTIC_ENTITY_BRIDGE.md](SEMANTIC_ENTITY_BRIDGE.md) ·
[INGESTION.md](INGESTION.md) (builds every store this doc reads) ·
`veda_core/query/contracts/L2_RETRIEVAL.md`.

---

## 1. Entry point and warm-load

Call site: `veda/pipeline.py:598` —
`get_engine(sm).retrieve(query=_search, intent=intent, top_k=15, use_cache=_RC)`, where
`_RC = RETRIEVAL_CACHE_ENABLED` (`False`), `_search` = the enhanced text if
`QUERY_ENHANCEMENT_ENABLED` else the raw query (it is `False`, so `_search == query`), and
`intent` is the router's literal string (`"SIMPLE"` on the deterministic head).

`get_engine(sm)` (`veda/runtime.py:256-282`): one `RetrievalEnginePhase3` per
`(source, tenant)` scope, LRU-capped (`ENGINE_CACHE_MAX`). Built with the caller's
per-source `sm`, `db_config = _internal_db_config()` (the VEDA internal store, **not** the
source DB), and **one shared** `SemanticSearchEngine` (`_shared_searcher()` — a single
BGE-M3 across every per-source engine).

### Warm-load — `RetrievalEnginePhase3._initialize` (`retrieval_engine_phase3.py:148-251`)

| # | Component | Notes |
|---|---|---|
| 1 | Semantic model | injected in-memory per-source `sm` (`_injected_sm`) preferred; flat-file fallback |
| 2 | `QueryEnricher()` | loads L2/L3 artifacts + the precomputed enrichment index |
| 3 | `SemanticSearchEngine` | the injected **shared** searcher is reused; else self-built |
| 4 | `SparseRanker()` | `warm_from_store(ctx.source_ids)` from `column_sparse_v1`; else `fit()` iff `len(retrieval_documents) <= SPARSE_FIT_MAX_DOCS` (300); else **Signal 2 is skipped** (a live `fit()` on a real model is ~50 min on CPU) |
| 5 | `SignalBuilder().build_signals(sm)` + `build_value_index(sm)` | static per-column scalars + the literal→column map |
| 6 | `IntentBooster(sm)` | |
| 7 | `RRFMerger(k=60)` | weights from `config.FUSION_WEIGHTS` |
| 8 | `AdaptiveCutoff(gap_threshold=0.28, min_k=5, max_k=20, hard_limit=15)` | |
| 9 | `RetrievalCache` | only if `use_cache` — never, in practice |

---

## 2. Six signals, not five — definitive

`retrieve()` builds **six** and `RRFMerger` logs `"Merging 6 signals via weighted RRF"`
(`rrf_merger.py:82`); the class name/docstring said "5-Signal"/"BM25Ranker" until this was
corrected 2026-09-10 (P2-2) alongside `docs/ARCHITECTURE.md`. `docs/ARCHITECTURE.md` §5 and
`docs/RETRIEVAL_DECISION_LAYER_AUDIT.md` §1 are the correct references.

| # | Signal | Built by | Key space | Character |
|---|--------|----------|-----------|-----------|
| 1 | **Dense semantic** | `semantic_search.py` — BGE-M3 dense, `k=50`, HNSW cosine | `table.col` (after `DENSE_ID_REMAP`) | raw query only (WP1); `ef_search` per source |
| 2 | **Learned-sparse** | `sparse_ranker.py` — BGE-M3 lexical weights, `top_k=50` | `table.col` | **replaced BM25** (WP3); carries query enrichment |
| 3 | **FK subgraph** | `signal_builder._compute_column_signals` `subgraph_signal` | `table.col` | static: `min(table_degree/10, 1.0)`; **boost-only since 2026-09-10 (P1-1)** — see below |
| 4 | **FK path / join-key** | `signal_builder._compute_column_signals` `fk_signal` | `table.col` | static: `0.5` if FK, `0.7` if referenced; **boost-only since 2026-09-10 (P1-1)** |
| 5 | **Value index** | `signal_builder.build_value_index` + `value_filter._query_value_tokens` | `col_id` | literal-in-query → the column that holds that value |
| 6 | **Table-first prior** | `retrieval_v2.table_prior_scores` (dense) ⊕ `sparse_ranker.table_scores` | `table_name` | WP4; **soft** — boosts existing candidates only, never adds |

Signals 3 and 4 are **not graph traversal at query time** — they are per-column scalars
computed once at warm from the relationship-graph edge list.

**P1-1 fix (2026-09-10):** Signals 3/4 used to be added to `rrf_merger.py`'s candidate UNION
directly (unlike Signal 6, which was always boost-only) — 25 tables have structural degree
≥ 10 (`users_user` = 275), so `min(degree/10, 1)` saturated to a virtual rank-1 hit for
**every column** of those tables, regardless of the query: a query-independent hub-table bias
that also inflated the candidate pool. Now Signals 3/4 match Signal 6's shape exactly — they
still fully contribute to the RRF SCORE of any candidate dense/sparse/value/table-prior
already surfaced, they just never introduce a candidate none of those found relevant.
Mechanical fix, not a weight retune (`config.FUSION_WEIGHTS["subgraph"]`/`["fk"]` untouched);
`scripts/retrieval_eval.py`'s golden-set harness targets `query/retrieval_select.py`, a
DIFFERENT retrieval path that doesn't call `RRFMerger` at all, so it could not measure this
change — verified instead via a live query through the real 6-signal path (`veda/pipeline.py`'s
own `[L2] Retrieval ...` log line was also corrected the same day: it said "5-signal
(BGE-M3 + BM25 + ...)", stale on both counts) completing without error and producing a sane,
unambiguous candidate set.

---

## 3. `retrieve()` step by step (`retrieval_engine_phase3.py:253-515`)

```
query
  │
  ▼  STEP 1  enrich
enriched_tokens = enricher.enrich(query)          # sorted token list; feeds S2 + S5 only
  │                                                 # (S1 dense encodes the RAW query — WP1)
  ▼  STEP 2  cache check  → SKIPPED (use_cache=False)
  │
  ▼  STEP 3  build 6 signals
  │
  │   ┌── ThreadPoolExecutor(max_workers=2) ──────────────┐   (wall-clock only; RRF is
  │   │  S1  semantic_searcher.search(query, k=50,         │    order-independent)
  │   │        allowed_cols=_iso_allow)   → BGE-M3 dense    │
  │   │  S2  sparse_ranker.rank(query, enriched_tokens,     │
  │   │        top_k=50)                  → learned-sparse   │
  │   └────────────────────────────────────────────────────┘
  │   S3+S4  single loop over sm["columns"] reading precomputed
  │          subgraph_signal / fk_signal                    (static — no traversal)
  │   S5     value_filter._query_value_tokens(query) → phrases → self.value_index lookup
  │          (skip a phrase equal to the column's own table name; score 1.0)
  │   S6     retrieval_v2.table_prior_scores(query, source_ids)  [dense, 0..1]
  │          max-combined with sparse_ranker.table_scores(...)  [normalized 0..1]
  │
  ▼  STEP 4  RRF fusion
fused = rrf_merger.merge(S1, S2, S4_fk, S3_subgraph, S5_value,
                         table_prior_signals=S6, top_k=50)
  │        score(d) = Σ_s  w_s / (k + rank_s(d)),  k = 60
  │        dense/sparse/value use real ranks; subgraph/fk/table-prior use a
  │        "virtual rank"  max(1, int((1 - score) * k))
  │        candidate union = S1/S2/S3/S4/S5 keys;  S6 only boosts existing candidates
  │
  ▼  (source-isolation backstop — only when _iso_allow is not None; no-op for normal queries)
  │
  ▼  STEP 5  intent boost
boosted = intent_booster.boost(fused, intent)     # additive ±0.1–0.6 deltas, then re-sort
  │
  ▼  STEP 6  adaptive cutoff
results = adaptive_cutoff.cutoff(boosted)         # cut at the biggest score gap in [5, 20)
  │
  ▼  STEP 7  cache set  → SKIPPED
  │
  ▼  build RetrievalResult[≤15]
        each carries its own per-signal contribution onto
        semantic_score / sparse_score / subgraph_score / fk_path_score /
        value_index_score / rrf_score / boosted_score
```

- **`_iso_allow`** (`:309-310`): sorted column keys **iff** `sm["__source_isolated__"]` is
  set (only on `veda_hybrid._datalake_isolated_sm`). `None` for every normal relational
  query, so all isolation branches are no-ops.
- **RRF arg order** (`:407-418`) maps `signal4 → fk_signals` param and
  `signal3 → subgraph_signals` param — cosmetic mismatch, harmless.
- **Trace** (`:487-513`): best-effort per-signal candidate counts + the top-8 RRF
  candidates into the ambient query trace.

### `IntentBooster` deltas (`intent_boosting.py`)

Additive on `analytics_role`, then re-sort:

| Intent | Deltas |
|---|---|
| AGGREGATE | MEASURE +0.30, IDENTIFIER −0.40, ATTRIBUTE −0.20, TIME_DIMENSION +0.10 |
| TEMPORAL | TIME_DIMENSION +0.30, IDENTIFIER −0.30 |
| MULTI_TABLE | is_fk +0.15 |
| always | `_history` / `_audit` table penalty −0.60 |

Per `docs/RETRIEVAL_DECISION_LAYER_AUDIT.md`: these ±0.1–0.6 deltas **dwarf** the RRF score
range (~0–0.098), so intent boosting effectively re-ranks the candidate set.

### `AdaptiveCutoff` (`adaptive_cutoff.py`)

`AdaptiveCutoff(gap_threshold=0.28, min_k=5, max_k=20, hard_limit=15)`. Finds the biggest
consecutive-score gap (the "semantic cliff") in `[min_k, max_k)`; cuts there if the gap
exceeds the threshold, else at `hard_limit`; clamps to `[min_k, max_k]`.

---

## 4. Downstream of `retrieve()` in `veda/pipeline.py` (`:598-720+`)

| Step | Gate | Behavior |
|---|---|---|
| **Graph-expansion booster** (`:600-633`) | `GRAPH_EXPAND_ENABLED=True` (`config.py:2363`) | `graph.query_graph.suggest_expansions(query, have_cols, have_tables, max_add=GRAPH_EXPAND_MAX=12)` — synonym + 1-hop FK (§5a). Appends `RetrievalResult(final_score=0.0)` per added `table.col`. Source-isolation filter if `__source_isolated__`. Fully try/excepted. |
| **RBAC candidate filter** (`:635-644`) | no-op unless the ambient context carries `allowed_resources` | `rbac_filter.filter_retrieval_results(results, sm, ctx)` |
| **Primary cross-encoder rerank** (`:646-720`) | `PRIMARY_RERANK_ENABLED=True` (`config.py:1250`) | skipped by `_rrf_gap_unambiguous` (top-2 same table **and** gap ≥ `RERANK_SKIP_GAP=0.02`). Otherwise `reranker._get_reranker()` scores `results[:RERANK_MAX_CANDIDATES=20]` with precomputed enriched pair text and **overwrites `final_score`**. Any failure keeps RRF order. |

Then: primary-table selection (`veda/routing.py::select_primary_table`), grain vet, anchor
confidence gate, SQL head.

---

## 5. Graph expansion — two mechanisms, two tiers

This is the source of the long-standing doc disagreement. There is **no single graph
algorithm**.

### 5a. Tier-1 spine booster — `graph/query_graph.py::suggest_expansions` (`query_graph.py:173-211`)

**Not PPR. Not hop-decay BFS.** Runs on the deterministic SQL head after `retrieve()`:

1. **Term resolution** — for each content token (`len > 2`, not a language-layer word from
   `veda.validation._gate_strip`): `resolve_term(tok)` follows `SYNONYM_OF` / `ALIAS_OF`
   edges (`query_graph.py:96-130`) plus a direct column-name match (plural-aware via
   `_singularize`). Resolved columns' `table.col` names are added.
2. **1-hop FK join-key reach** — for each resolved column, follow **only** `REFERENCES`
   edges, both directions (`query_graph.py:201-211`), and add the specific join-key column
   on the other side (e.g. `assigned_to_id` → `user.user_id`). Deliberately **not** the
   whole neighbour table.
3. Cap at `max_add` (`GRAPH_EXPAND_MAX=12`); return `(seeds, added_names, synonyms_map)`.

Additions get `final_score=0.0` and rely on the cross-encoder rerank to re-score them.
`UnifiedGraph.shortest_path` (`query_graph.py:214-235`) is a real undirected BFS
(`max_hops=8`) but is used only by standalone helpers, **not** by `suggest_expansions` or
`retrieve()`.

### 5b. Tier-2 / datalake / cross-source — `query/graph_retriever.py::run_graph_retrieval` (`graph_retriever.py:262-531`)

**Personalized PageRank (WP5)** — genuinely replaced an older hop-decay BFS (the
`# BFS expansion` comment at `graph_retriever.py:281` is a stale leftover).

```
embed_text_bge(query)
  → retrieve_graph_seeds(qvec, top_k=GRAPH_SEED_TOP_K=12, source_ids)   # cosine ANN over
  │                                                                       # graph_node_embeddings
  ▼
single-table short-circuit (:301-320) on seed dominance
  │  (gap ≥ GRAPH_SINGLE_TABLE_GAP, or all strong seeds agree on one table)
  ▼
_ppr_scores (:145-170)
  p0 = seed similarities, normalized to sum 1
  iterate  p = (1 - d)·p0 + d·(Pᵀ p)   with d = GRAPH_PPR_DAMPING = 0.85
  until Σ|Δ| < GRAPH_PPR_TOL  or  GRAPH_PPR_MAX_ITERS = 50
  ▼
transition matrix _get_transition (:112-142)
  symmetric adjacency (edges both directions), weighted by edge weight,
  row-normalized:  P = diag(1/deg) @ A          # row-normalization = hub-dilution
  cached per source scope in _PPR_CACHE
  cross-source / entity / semantic_about edges loaded tenant-wide (_load_all_edges :67-96)
  ▼
top PPR non-seed nodes up to GRAPH_PPR_MAX_NODES = 40 visited-node budget
  ▼
bounded sibling inclusion (:346-417) + chunk safety net (:419-452) + node materialization
```

Wired at `query/retrieval_select.py:106-121` and `veda_hybrid.py:1540-1544`; gate
`UNIFIED_GRAPH_ENABLED && GRAPH_RETRIEVAL_ENABLED && GRAPH_EMBED_ENABLED` (all `True`). Used
by the datalake path, the Tier-2 fallback, and cross-source composition.

### Verdict

- "Graph expansion is Personalized PageRank" (`ARCHITECTURE.md` §5) is **true for
  `graph_retriever.py`** and false as a blanket claim — the Tier-1 spine booster
  (`suggest_expansions`) was never PPR.
- Signals 3 & 4 are not graph traversal at all.

---

## 6. Encoder — single BGE-M3

**One model: `BAAI/bge-m3`.** One process-wide singleton:
`ingestion/m3_encoder.py::_get_model` → `BGEM3FlagModel` (FlagEmbedding), forced offline
(`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`). It produces:

- **dense** 1024-dim vectors (`encode_dense`, L2-normalized) — columns, tables, graph
  nodes, doc chunks.
- **learned-sparse** lexical weights (`encode_sparse`) — the Signal-2 store that replaced
  BM25.
- `encode_query` — one forward pass returns dense + sparse together, `lru_cache(64)` per
  exact text.
- `_DenseEncoder` — a `SentenceTransformer`-compatible facade so `retrieval/semantic_search.py`,
  `query/retrieval_v2.py`, and `veda.runtime._get_bge` all share the one model + pooling.
- ColBERT / multi-vector head deliberately **not** used (`return_colbert_vecs=False`).
- Optional Metal (host-MPS) offload via `METAL_EMBED_URL`; any transport error →
  in-process CPU fallback.

**`ENCODER_MODE` is dead in the engine.** `config.py:6-7` records the removal;
`EMBEDDING_MODEL_ID = "bge-m3"` (`config.py:1684`) replaces the old `ENCODER_MODE` resume
guard. The relgt / light-text / hybrid / MiniLM **ensemble is gone** —
`query/retrieval_select.py:123-132` notes the legacy MiniLM/RELGT semantic-layer signal was
removed and always returned an empty `SemanticLayerResult` anyway.
`BIENCODER_MODEL = "BAAI/bge-m3"` (`config.py:1195`), `BIENCODER_QUERY_PREFIX = ""` (M3
needs no instruction prefix). Stale `ENCODER_MODE` mentions elsewhere in
`docs/ARCHITECTURE.md` describe `apps/ingestion/tasks.py`, out of this doc's scope.

---

## 7. Reranker — cross-encoder

- **Model:** `RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"` (`config.py:1251`).
- **Runner:** `query/reranker.py`.
  - `_RemoteReranker` (`:58-76`) — POSTs `query<->candidate` pairs to the host Metal server
    (`METAL_EMBED_URL` + `/rerank`); CPU `CrossEncoder` fallback on any error.
  - `_load_local_crossencoder` (`:79-108`) — `sentence_transformers.CrossEncoder(RERANKER_MODEL,
    device=RERANKER_DEVICE)`; **fail-loud at ERROR** if `RERANKER_ENABLED` but the model
    won't load — still returns `None` → graceful pure-RRF degrade, unless
    `RERANKER_REQUIRED=True` (default `False`) which raises.
  - `_get_reranker` — remote facade if `METAL_EMBED_URL` set, else local CPU; cached
    singleton.
- **Call sites:**
  1. **Primary path** — `veda/pipeline.py:646-720` (§4). Reorders and **overwrites
     `final_score`**. `RERANK_MAX_CANDIDATES=20`, `RERANK_SKIP_GAP=0.02`,
     `RERANK_NOISE_FLOOR=0.01`.
  2. **Tier-2 / `select_retrieval`** — `rerank_columns` / `rerank_tables`
     (`reranker.py:404-421`) with a dynamic score-cliff cutoff (`_apply_cutoff`),
     `RERANKER_TOP_COLS=15` / `RERANKER_TOP_TABLES=5`.
  3. **RAG** — `rerank_chunks` (`:423-457`) scores `(query, chunk.text)` pairs.
- **Pair text is precomputed at ingestion (WP7).** `_get_rerank_docs()` (`:160-173`) loads
  the `ingestion.rerank_docs` artifact (`data/veda_rerank_docs.json`) — the **only** source
  of cross-encoder text; per-query runtime assembly was removed.
  `_precomputed_rerank_text` / `enriched_pair_text` return the precomputed enriched string
  (business definition / aliases / role / sample values), truncated to
  `RERANKER_MAX_TEXT_LEN=160` chars (cross-encoder cost is ~quadratic in sequence length,
  ~1.7 s/pair for 512-token pairs on CPU). A missing / uncovered id → bare
  `"col_name table_name"` fallback (never raises on the SQL-critical path). The artifact
  entirely missing → `_get_rerank_docs` raises `RuntimeError`, but
  `_precomputed_rerank_text` swallows it so the primary path still runs on bare names.
- **Silent degrade:** with `RERANKER_REQUIRED=False` and `METAL_EMBED_URL` unset (both
  default), if the local cross-encoder model isn't cached, **every query silently degrades
  to pure-RRF order** (logged at ERROR, `reranker.py:87-95`).

---

## 8. HNSW `ef_search` per source

`hnsw.ef_search` is `SET LOCAL` per transaction, resolved per source via
`storage_adapters.reader._resolve_ef_search` and applied inside the explicit txn that wraps
each ANN search (`retrieval/semantic_search.py:143-149,181`). The width is tuned at
ingestion (L5 `hnsw_tune`, `clamp(40 + (n_tables // 20) * 20, 40, 200)` →
`data/veda_hnsw.json` → `SubstrateVersion.hnsw_ef_search`). `HNSW_M` /
`HNSW_EF_CONSTRUCTION` / `HNSW_EF_SEARCH` live in the Django settings bridge, **not**
`veda_core/config.py`.

---

## 9. Semantic entity bridge (query-side view)

Full treatment: [SEMANTIC_ENTITY_BRIDGE.md](SEMANTIC_ENTITY_BRIDGE.md). From the retrieval
side: `ingestion/semantic_linker.py` (Tier A, SHIPPED 2026-07-15) emits `semantic_about`
(chunk→column) edges from M3 cosine similarity. `graph_retriever._load_all_edges`
(`graph_retriever.py:89-96`) loads them **tenant-wide** into the transition matrix, so the
Tier-2 PPR expansion (§5b) traverses chunk→column reachability. A `semantic_about` edge
never authorizes a join — it only makes a column reachable as a retrieval candidate. The
Tier-1 spine (`suggest_expansions`) does not see these edges. Config: `SEMANTIC_BRIDGE_*`
(`config.py:1097-1116`), `SEMANTIC_BRIDGE_ENABLED` default ON, `MIN_SIM=0.55`, `TOPK=5`.

---

## 10. Wired vs dormant / dead

**Wired & live:**

- Full Tier-1 spine: `retrieval_engine_phase3.retrieve()` — all 6 signals, RRF, intent
  boost, adaptive cutoff.
- `graph/query_graph.suggest_expansions` (Tier-1 booster);
  `graph_retriever.run_graph_retrieval` PPR (Tier-2 / datalake / cross-source).
- Cross-encoder rerank on the primary path + Tier-2 + RAG chunks.
- `semantic/registry.py` matchers (deterministic fast path), `name_tokens.py`
  (anchor / routing), `compile_semantic_layer.py` (per-ingestion).
- `FUSION_WEIGHTS` (all `1.0` → identity → bit-for-bit the old unweighted RRF),
  `DENSE_ID_REMAP` (ON), source-isolation (fires only on `__source_isolated__` datalake
  models).

**Dormant / disabled / dead:**

| Item | State |
|---|---|
| `RetrievalCache` (`retrieval_cache.py`) | code complete but `RETRIEVAL_CACHE_ENABLED=False` (`config.py:2113`); `retrieve()` always called with `use_cache=False`. Module-level `_cache_instance` + `cache_retrieval_results` / `get_cached_results` helpers unused. |
| `sparse_ranker.fit()` | dev fallback only; on a real model (>300 docs) Signal 2 is silently **skipped** rather than encoded live. Now surfaced per query (P1-4, 2026-09-10) as `sparse_active` in the explain trace's `retrieval_health` section, and `/readyz`'s `degraded` list catches an unavailable reranker/BGE-M3/SLM at startup — no longer log-only. |
| `semantic_search.py` engine-store path (`VEDA_ANN_VIA_ADAPTER=0`) | dev/CLI only — a direct, unscoped read against the engine's pgvector store, no source filter. The `storage_adapters` adapter path (`=1`, default) is the real one, and it ALSO had a real bug until 2026-09-10: `storage_adapters/reader.py::ann_search` queried `column_embeddings_v2` (lives in `veda_engine`) through the Django-targeted connection, not the engine's internal one — fixed same day as this doc; see `docs/backlog/query-engine-open-items.md`'s "ann_search database target" entry for the full incident. |
| `schema/simulate_schema.py` | POC synthetic schema; only ingestion fallbacks import it, no query path |
| `__main__` demo blocks in every `retrieval/*.py` | dead |

Fixed since the last pass (no longer dormant/stale, 2026-09-10): `retrieval/__init__.py`'s
docstring (was a filename list from before BM25's removal — see the file itself for the
corrected component list); `RetrievalEnginePhase3`'s class name/docstring said "5-Signal" (it
builds 6, including the boost-only table-first prior); `signal_builder.py:19`'s dead
`from schema.real_schema import get_real_schema` import (removed); `graph_retriever.py:281`'s
stale `# BFS expansion` comment (now correctly describes the seed-collection step it actually
labels, with a pointer to the real PPR section below it).

---

## 11. File inventory (`veda_core/retrieval/`)

See [`veda_core/retrieval/AGENTS.md`](../veda_core/retrieval/AGENTS.md) for the one-line
map. Detail:

| File | Role |
|---|---|
| `retrieval_engine_phase3.py` (627) | `RetrievalEnginePhase3` — the orchestrator. Warm-loads all components; `retrieve()` runs the 6-signal → RRF → intent-boost → adaptive-cutoff pipeline. The only real entry point (`veda/runtime.get_engine`). |
| `query_enrichment.py` (324) | `QueryEnricher.enrich(query)` → sorted token list. Loads `veda_domain_synonyms.json`, `veda_concept_graph.json`, `veda_glossary.json`, `veda_semantic_model.json`, plus the precomputed `enrichment_index`. Feeds Signals 2 & 5 only. Also exports `_singularize`, reused across `semantic/` and `graph/`. |
| `semantic_search.py` (392) | **Signal 1.** `SemanticSearchEngine.search(query, k=50, allowed_cols=)` — BGE-M3 dense, cosine ANN. Adapter path (`storage_adapters.reader.ann_search`, default) or engine pgvector (dev). `DENSE_ID_REMAP` maps UUID `col_id`→`table.column`. |
| `sparse_ranker.py` (164) | **Signal 2.** BGE-M3 learned-sparse (replaces BM25). `warm_from_store` loads `column_sparse_v1` / `table_sparse_v1`; `fit()` is the dev fallback. `table_scores()` feeds Signal 6. |
| `signal_builder.py` (227) | **Signals 3, 4, 5 source data.** `build_signals()` builds an FK graph (relationship-graph edges preferred, substrate `fk_adjacency` fallback). `build_value_index()` → `{value_token: [col_id]}`. Dead import at line 19. |
| `rrf_merger.py` (152) | `RRFMerger(k=60)`. Weighted RRF over 6 signals. Dense/sparse/value use real ranks; subgraph/fk/table-prior use a virtual rank. Weights from `config.FUSION_WEIGHTS`. |
| `intent_boosting.py` (313) | `IntentBooster(sm).boost(fused, intent)` — additive deltas on `analytics_role` (§3), then re-sort. |
| `adaptive_cutoff.py` (230) | `AdaptiveCutoff` — biggest-score-gap ("semantic cliff") cut in `[min_k, max_k)`. |
| `retrieval_cache.py` (311) | `RetrievalCache` — file or Redis, TTL 300 s. **Disabled** (`RETRIEVAL_CACHE_ENABLED=False`). |
| `__init__.py` (16) | package header — **stale** filename list. |
