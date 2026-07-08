# How VEDA Stores Data Sources as Embeddings and Graphs — and How It Queries Them

This document describes, precisely and in detail, how VEDA turns a connected data
source (relational DB, document store, NoSQL, or datalake) into (a) **vector
embeddings** and (b) a **knowledge graph**, and how those artifacts are then used
at **query time** to select the right columns/tables/chunks for SQL generation and
RAG.

It is written against the actual code. File references are given as
`path:symbol` so each claim is traceable.

---

## 0. Vocabulary and the two databases

VEDA never stores embeddings in the customer's database. There are always two
distinct Postgres endpoints:

| Name | Config | Role |
|------|--------|------|
| **Client source DB** | per-source connection (`veda_core/connectors/*`) | Read-only. Introspected for schema, sampled for values, and used *only* as the L7 SQL execution target. Never holds embeddings. |
| **VEDA internal DB** | `config.VEDA_INTERNAL_DB` | Postgres + `pgvector`. Holds every embedding table, the graph tables, FK adjacency, value store, and all derived indexes. |

Every store in the codebase has an **in-memory fallback**: if `psycopg2`/pgvector
is unavailable (`INTERNAL_DB_AVAILABLE` / `PSYCOPG2_AVAILABLE` is `False`), the
same data is kept in module-level Python lists and the retrieval interface is
byte-for-byte identical. This is why every store module has both a
`_store_pgvector_*` and a `_store_in_memory_*` path.

A "source" is identified by `source_id`. Almost every internal table carries a
`source_id` column, and every write does a **scoped delete then insert** (`DELETE
… WHERE source_id = %s`) so re-ingesting a source is fully idempotent and never
accumulates stale rows.

---

## 1. The ingestion pipeline (L1 → L5)

Entry point: an API/Celery worker calls `ingestion.dispatcher.dispatch(ctx)`
(`veda_core/ingestion/dispatcher.py`) with a resolved `SourceContext`.

- `ctx.type == "relational"` → `ingestion.layers.pipeline.run_layered_ingestion`
  runs the five layers **L1–L5** in order (`veda_core/ingestion/layers/pipeline.py`).
- `document` / `nosql` / `datalake` → `ingestion.source_dispatcher.dispatch_ingestion`.

Each layer function returns a list of `StageOutcome`. A stage may be **fatal**
(aborts the whole run) or **non-fatal** (logs a warning and continues). The layer
threads a single in-memory `state` dict from stage to stage, so later stages read
what earlier ones produced (`state["scan_result"]`, `state["inference_result"]`,
`state["graph"]`, …).

```
L1 EXTRACT   → the only layer that touches the client source
L2 ANALYZE   → pure transforms (semantic types, REG graph) — no source, no LLM
L3 ENRICH    → LLM/semantic enrichment (glossary, concepts, retrieval docs)
L4 INDEX     → EMBEDDINGS + search structures  ← the "store as vectors" layer
L5 PUBLISH   → derived registries + unified graph (atomic activate)
```

### L1 — Extract (`layers/l1_extract.py`)

1. **Schema scan** (fatal). `schema.real_schema.get_real_schema()` →
   `ingestion.schema_scanner.run_schema_scanner()`. The connector
   (`RelationalConnector.get_schema`, `veda_core/connectors/relational.py`)
   introspects the live catalog and returns a `RawSchema` of `RawTable` /
   `RawColumn`, plus `fk_edges`. Notable details:
   - Table/column IDs are **deterministic UUIDv5** of `schema.table[.col]`, so the
     same physical object always maps to the same UUID across runs
     (`uuid.uuid5(uuid.NAMESPACE_OID, …)`), which is what lets pgvector UPSERTs
     overwrite instead of duplicating.
   - Sensitive columns (`config.SENSITIVE_PATTERNS`) and `VEDA_INTERNAL_TABLES`
     are excluded.
   - PostgreSQL uses fast `pg_catalog` batched introspection (FK/PK/row-count) —
     one query per metadata type for the whole schema, not per table.
2. **FK adjacency store** (fatal). `vector_store.store_fk_adjacency(scan_result)`
   truncates and reinserts the `fk_adjacency` table (plain SQL, not vector).
3. **Data graph** (non-fatal). `ingestion.data_graph.run_data_graph` discovers
   *undeclared* FKs by value-overlap profiling. HIGH/MEDIUM discovered edges are
   appended back into `scan_result.fk_edges` and re-stored into `fk_adjacency`.
4. **Value sampling** runs a bit later (sequenced inside L2 via
   `l1_extract.run_value_sampling`, because it needs the inferred types). It samples
   distinct column values into the `column_values` store — used by the value
   signal and for enriching embedding text.

### L2 — Analyze (`layers/l2_analyze.py`)

1. **Semantic type inference** (fatal). `run_semantic_type_inference(scan_result)`
   assigns each column a `semantic_type` (IDENTIFIER, CATEGORY, TEMPORAL, METRIC,
   MONETARY, …) and picks a **display column** per table.
2. **Table metadata** (fatal). `store_table_metadata` persists the per-table
   display column into the `table_metadata` store.
3. **REG builder** (fatal). `ingestion.reg_builder.run_reg_builder` produces a
   `REGGraph` in memory — the canonical in-process graph: `table_nodes`,
   `column_nodes`, `has_column_edges` (table→col index pairs), `fk_to_edges`
   (col→col index pairs), and numpy feature matrices. This object is the shared
   input to both the graph persist and the encoders.
4. **Join paths** (non-fatal). Precomputes pairwise shortest FK paths (Q-9).

### L3 — Enrich

LLM/semantic enrichment that produces `data/veda_semantic_model.json`
(`SEMANTIC_MODEL_FILE`) with per-column `retrieval_documents`, business
definitions, aliases, concepts, glossary, domain synonyms, metrics, and
dimensions. These artifacts feed both the embedding text (L4) and the unified
graph (L5).

---

## 2. How data sources become **embeddings** (L4 INDEX)

All embedding creation is isolated in `layers/l4_index.py:run`. **One model** —
`BAAI/bge-m3` (`ingestion/m3_encoder.py`, `local_files_only`) — produces every
1024-dim **dense** vector (columns, tables, graph nodes, doc chunks) **and** the
learned-**sparse** lexical weights that replaced BM25. The relgt/light-text/hybrid/
MiniLM ensemble and `ENCODER_MODE` were removed (one embedding space → one store).

### 2.1 The live column/table store — BGE-M3 dense (`ingestion/biencoder.py`)

Gate: `BIENCODER_ENABLED`. Model: `BAAI/bge-m3` (`config.BIENCODER_MODEL`),
**1024-dim** (`BIENCODER_DIM`); dense encoding goes through the shared
`m3_encoder.encode_dense` singleton (no separate SentenceTransformer copy).

`run_biencoder_ingestion(inference_result, source_id)`:

1. Ensures two pgvector tables exist, each with an **HNSW** `(embedding
   vector_cosine_ops)` index (`m=16, ef_construction=200`):
   - `column_embeddings_v2` (`BIENCODER_COL_TABLE`) — one row per column.
   - `table_embeddings_v2` (`BIENCODER_TABLE_TABLE`) — one row per table.
   Columns: `col_id, col_name, table_id, table_name, source_id, semantic_type,
   text, embedding vector(1024)`.
2. Builds the **passage text** per column via `_passage_text(col, rdocs)`
   (`config.EMBED_TEXT_STRATEGY`, currently `"doc"` → the rich NL
   `retrieval_document`). M3 needs **no** instruction prefix, so
   `BIENCODER_PASSAGE_PREFIX` is empty.
3. `encode_dense(col_texts)` (already L2-normalized), then scoped
   `DELETE … WHERE source_id=%s` + `INSERT`. Table embeddings come from
   `"{table_name}: columns {c1, c2, …}"`.

This store answers **Signal 1** at query time.

### 2.2 Learned-sparse store (`ingestion/sparse_index.py`) — replaces BM25

Runs on the SAME passage texts as §2.1. `encode_sparse` yields per-passage
`{token_id: weight}` maps persisted (scoped delete-then-insert) to:

- `column_sparse_v1` — `col_id ("table.col"), source_id, table_id, weights jsonb`.
- `table_sparse_v1` — `table_id, table_name, source_id, weights jsonb`.

The query tier's `retrieval/sparse_ranker.py` warm-loads these rows into an in-memory
inverted index and scores columns by sparse dot product — **Signal 2** (and the
sparse half of the WP4 table prior). BM25 (`bm25_ranker`/`bm25_index`) is gone.

### 2.3 Document chunks — BGE-M3 (`ingestion/chunk_embedder.py`)

`run_chunk_embedder(chunks, source_id)` embeds each chunk with `m3_encoder.encode_dense`
(**1024-dim**, L2-normalized) into `doc_chunks`: `chunk_id, source_id, doc_id,
doc_name, chunk_index, text, page_num, doc_date, embedding vector(1024)`, with an
**HNSW** cosine index and a partial index on `doc_date`. The table is dropped +
recreated if the stored dim differs (the MiniLM 384 → M3 1024 migration). These
vectors are reused verbatim by the graph embedder for `chunk` nodes.

### 2.4 Graph node embeddings — BGE-M3 (`ingestion/graph_embedder.py`)

Gate: `UNIFIED_GRAPH_ENABLED + GRAPH_EMBED_ENABLED`. Encodes column/table node
sentences with `m3_encoder.encode_dense` (1024-dim, `GRAPH_NODE_EMB_DIM`) and
**copies** chunk-node vectors straight out of `doc_chunks`. Since chunks are now the
same 1024-dim M3 space as columns/tables, there is no mixed-dimension caveat.
Persists into `graph_node_embeddings` with source/type indexes + an **HNSW** cosine
index; auto-dropped/recreated on a dim change. This store answers the **graph seed** ANN.

### 2.5 Other precomputed indexes built in L4 (all non-fatal)

- **Learned-sparse index** (`ingestion/sparse_index.py`) — §2.2 above (replaced the
  BM25 index stage).
- **Enrichment index** (`ingestion/enrichment_index.py`).
- **Rerank docs** (`ingestion/rerank_docs.py`) — precomputed cross-encoder text per
  column/table; the query-tier reranker reads ONLY this (fail-loud if missing, WP7).

### Summary of embedding stores

| Store (internal DB table) | Built by | Dim | Model | Queried at runtime by |
|---|---|---|---|---|
| `column_embeddings_v2` | `biencoder.py` | 1024 | bge-m3 (dense) | **Signal 1** dense search |
| `table_embeddings_v2` | `biencoder.py` | 1024 | bge-m3 (dense) | table prior + table rerank |
| `column_sparse_v1` | `sparse_index.py` | sparse | bge-m3 (sparse) | **Signal 2** learned-sparse |
| `table_sparse_v1` | `sparse_index.py` | sparse | bge-m3 (sparse) | table prior (sparse half) |
| `doc_chunks` | `chunk_embedder.py` | 1024 | bge-m3 (dense) | RAG chunk retrieval |
| `graph_node_embeddings` | `graph_embedder.py` | 1024 | bge-m3 (dense) | **graph seed** ANN |

Every vector table uses an **HNSW** index; `ef_search` is pinned per source at query
time via `storage_adapters.reader._resolve_ef_search`.

---

## 3. How data sources become **graphs**

There are three graph representations, built at different stages:

### 3.1 `fk_adjacency` — the FK edge table (L1)

`vector_store.store_fk_adjacency` (`veda_core/ingestion/vector_store.py`). A plain
SQL table (no vectors) of directed FK edges:

```
fk_adjacency(from_col_id, from_col_name, from_table_id, from_table_name,
             to_col_id,   to_col_name,   to_table_id,   to_table_name)
```

Indexed on `from_table_id` and `to_table_id`. Rebuilt by **truncate + insert**
each run. Read at query time by `get_fk_adjacency(table_ids)` to find bridge
tables and to resolve JOIN paths. When a request context is set, this call is
routed through `storage_adapters.reader.get_fk_adjacency` to the Django-owned,
tenant-scoped substrate instead.

### 3.2 `graph_nodes` / `graph_edges` — the persisted property graph (L4)

`ingestion/graph_persist.py:persist_reg_graph(graph, scan_result, dg_result,
source_id)` flattens the in-memory `REGGraph` into two relational tables.

**Node id conventions** (stable across runs): `col:<col_id>`, `tbl:<table_id>`,
`chunk:<chunk_id>`.

`graph_nodes(node_id PK, node_type, source_id, ref_id, table_id, name,
table_name, semantic_type, data_type, is_pk, is_fk, attrs)` — one row per table,
column, (and later chunk).

`graph_edges(edge_id PK, src_node_id, dst_node_id, edge_type, weight, source_id,
evidence, attrs)` with a `UNIQUE(src, dst, edge_type)` triple index. Edge types
and weights (`config.GRAPH_EDGE_WEIGHTS`):

| edge_type | weight | meaning |
|---|---|---|
| `has_column` | 1.0 | table → its columns |
| `fk_to` | 3.0 | declared FK, column → column |
| `discovered_fk` | 2.0 × tier | value-overlap-discovered FK (HIGH tier 1.0, MEDIUM 0.6) |
| `mentions` | 1.0 | chunk → column it references |
| `about` | 1.5 | chunk → column it is about |
| `name_match` | 0.6 | capped so it never dominates |

`discovered_fk` edges come from `dg_result` (the Data Graph). `mentions`/`about`
edges linking document chunks to columns are added by `ingestion/chunk_linker.py`
(gated by `GRAPH_CHUNK_LINKING_ENABLED`, thresholds
`GRAPH_LINK_VALUE_OVERLAP_MIN`, `GRAPH_LINK_EMBED_SIM_MIN`, …).

Persist is idempotent: scoped `DELETE … WHERE source_id=%s AND node_type IN
(...)` / `edge_type IN (...)` before upsert. Query-time accessors: `get_nodes`,
`get_neighbors(node_ids, edge_types, direction)`, `get_node_degrees`.

### 3.3 `data/veda_unified_graph.json` — the derived unified graph (L5)

`ingestion/unified_graph_builder.py:build_unified_graph()` fuses the separate JSON
artifacts into one node/edge graph written to `UNIFIED_GRAPH_FILE`. It is a
**derived, additive view** (pure stdlib, deterministic/sorted, idempotent) — it
never replaces the other artifacts. Inputs and what they contribute:

| Input artifact | Nodes | Edges |
|---|---|---|
| `veda_semantic_model.json` | `TABLE`, `COLUMN` | `HAS_COLUMN`, `ALIAS_OF` |
| `veda_relationship_graph.json` | (tables/cols) | `FK_TO` (table↔table), `REFERENCES` (col↔col) |
| `veda_concept_graph.json` | `CONCEPT` | `IS_CONCEPT` |
| `semantic/metrics.json` | `METRIC` | `IS_METRIC`, `SYNONYM_OF` |
| `semantic/dimensions.json` | `DIMENSION` | `IS_DIMENSION`, `SYNONYM_OF` |
| `veda_domain_synonyms.json` | `SYNONYM` | `SYNONYM_OF` |

Node id scheme: `table:{t}`, `col:{t}.{c}`, `concept:{NAME}`, `metric:{id}`,
`dim:{id}`, `syn:{term}`. The accumulator dedups nodes by id and edges by
`(src, tgt, type)`, and only emits an edge when **both endpoints already exist as
nodes** (grounded-only — no phantom edges). Any missing input is skipped with a
warning rather than crashing.

### 3.4 L5 also publishes the fast-path registries

`layers/l5_publish.py` additionally builds (all non-fatal): the relationship graph
(`build_relationship_graph`), the compiled semantic registry
(`compile_semantic_layer.compile_all`), a per-source HNSW `ef_search` tuning file,
and the Redis value mirror. L5 is the **atomic-activate** point: everything is
written, then the query tier is told to rehydrate.

---

## 4. How it is **queried** at runtime

There are two cooperating retrieval subsystems. Both take a natural-language query
and return a ranked set of columns (and optionally chunks) that downstream layers
(SLM → SQL builder, or RAG) consume.

### 4.1 The 5-signal hybrid engine (`retrieval/retrieval_engine_phase3.py`)

`RetrievalEnginePhase3.retrieve(query, intent, top_k)` is the warm engine behind
the `/v1/retrieve` route (`inference/routes/retrieve.py`). Pipeline:

**Step 1 — Enrich.** `QueryEnricher.enrich(query)` expands the query with domain
synonyms, concepts, and glossary terms → `enriched_tokens`.

**Step 2 — Cache.** Keyed on a hash of the enriched tokens (5-min TTL). Hit →
return immediately.

**Step 3 — Six signals:**

1. **Signal 1 — BGE-M3 dense** (`retrieval/semantic_search.py`). Embeds the **raw**
   query (WP1: prefix + normalized, not enriched tokens) with the shared bge-m3
   model and runs pgvector cosine over **`column_embeddings_v2`**, top-50. When a
   request context is set this routes through `storage_adapters.reader.ann_search`
   → the per-(source,tenant) HNSW store with a pinned `ef_search`.
2. **Signal 2 — learned-sparse (M3)** (`retrieval/sparse_ranker.py`). Scores columns
   by sparse dot product over the warm-loaded `column_sparse_v1` inverted index; the
   raw query's sparse weights plus the enriched-expansion phrases (max-pooled) — this
   is where enrichment now lives. Replaced BM25.
3. **Signal 3 — FK subgraph proximity.** Per-column precomputed `subgraph_signal`.
4. **Signal 4 — FK path bridges.** Per-column precomputed `fk_signal`.
5. **Signal 5 — Value index.** A column whose **sampled values** match a query
   literal ("escalated") scores 1.0 — surfacing the column that *holds* the value.
6. **Signal 6 — Table-first prior** (WP4). The dense query is ANN'd against
   `table_embeddings_v2` (top-M) and combined (max) with `table_sparse_v1` scores
   into a per-table affinity; each column inherits its table's affinity as a **soft**
   prior (no hard filter — rare-table recall is preserved).

**Step 4 — weighted RRF fusion** (`retrieval/rrf_merger.py`, WP6). `score(d) =
Σ_s w_s / (k + rank_s(d))` with `k=60` and `config.FUSION_WEIGHTS` (identity 1.0 ==
the old unweighted ranking; tuned offline by `scripts/tune_fusion_weights.py`).
Rank-typed signals (dense, sparse, value) contribute by rank; the score-typed
signals (subgraph, fk, table_prior) convert to a virtual rank
`max(1, int((1 - score) · k))` first. Returns top-50 fused `(col_id, rrf_score)`.

**Step 5 — Intent boosting** (`IntentBooster`). Adjusts scores by query intent
(AGGREGATE / TEMPORAL / MULTI_TABLE / DIRECT / SIMPLE).

**Step 6 — Adaptive cutoff** (`AdaptiveCutoff`, gap_threshold 0.28, min 5 / max 20
/ hard 15). Detects the "semantic cliff" and truncates.

**Step 7 — Cache** the result and return `RetrievalResult` objects with
`final_score` plus per-signal debug scores.

### 4.2 The graph seed-and-expand retriever (`query/graph_retriever.py`)

`run_graph_retrieval(query, source_ids)` — used when `UNIFIED_GRAPH_ENABLED +
GRAPH_RETRIEVAL_ENABLED + GRAPH_EMBED_ENABLED`.

1. **Seed.** `embed_text_bge(query)` (1024-dim bge-m3) → `retrieve_graph_seeds` runs
   cosine ANN over `graph_node_embeddings` (`GRAPH_SEED_TOP_K = 12`), returning
   `(node_id, node_type, similarity)` seeds at hop 0.
2. **Single-table short-circuit.** If the top seed is very similar and either
   dominates the second by `GRAPH_SINGLE_TABLE_GAP` or all strong seeds share one
   table, it focuses on that one table (includes all its columns) and skips expansion.
3. **Personalized PageRank expansion** (WP5, replaced hop-decay BFS). The source
   edge list (loaded once per scope, cached, invalidated on rehydrate) becomes a
   **row-normalized transition matrix** `P` (scipy CSR; row-normalization dilutes
   hubs, so no degree cap is needed). Seed similarities form the restart vector
   `p0`; `p = (1-d)·p0 + d·Pᵀp` with `GRAPH_PPR_DAMPING=0.85` iterates to
   `GRAPH_PPR_TOL=1e-6` / `GRAPH_PPR_MAX_ITERS=50`. The top nodes by stationary score
   (up to `GRAPH_PPR_MAX_NODES`) join the subgraph — so 2-hop FK-reachable columns
   (`state.name` via `transition→state`) surface at their true relevance.
4. **Sibling inclusion.** Adds a bounded number of sibling columns of the seed
   tables, scored just below every real expanded node.
5. **Chunk safety net.** Directly pulls `mentions`/`about` chunk neighbors of the
   seed and hop-1 columns (chunks are leaves, so the hub cap must not block them),
   up to `GRAPH_MAX_CHUNKS`.
6. **Materialize.** `get_nodes` hydrates metadata; chunk texts are fetched from
   `doc_chunks`. Columns are sorted by score and truncated to
   `GRAPH_MAX_COLS_TO_L3`; chunks become `ChunkRetrievalResult`s. Column subgraph
   nodes are adapted back to `RetrievalResult` via `_subgraph_to_retrieval_results`
   so downstream layers see a uniform type.

### 4.3 Unified selection — `query/retrieval_select.py:select_retrieval`

This is the single source of truth that both the interactive path and the
evaluator call, so they can never diverge. It composes the pieces above with a
strict priority order:

1. **Schema linker** (`RETRIEVAL_V2_ENABLED + SCHEMA_LINK_ENABLED`). A
   high-confidence exact name match **short-circuits** and skips the bi-encoder.
2. **Bi-encoder + cross-encoder reranker** (`query/retrieval_v2.py:retrieve_v2`) —
   the primary column retriever (Signal-1 store + rerank), skipped if the schema
   linker short-circuited.
3. **Graph retrieval** (§4.2), unless a precomputed `GraphRetrievalResult` was
   passed in.
4. The **semantic layer is a stub** (`retrieval_select.py`) — the legacy ensemble
   encode path was removed; retrieval runs on the M3 dense + sparse + FK + value +
   table-prior spine, and the layer only supplies FK-bridge/JOIN-path helpers.
5. **Override decision**: V2 columns (schema-link or bi-encoder) take **final
   priority**; graph columns are second (used for `hybrid` intent or when chunks
   exist); the legacy layer is the fallback.
6. **V2 supplements** (only when V2 columns win) — additional columns injected that
   the first-stage reranker under-ranks:
   - **A. Keyword injection** — `retrieve_cols_by_name_keywords` over raw query
     words (+ singular forms) and synonym-expansion parts.
   - **B. FK-PK injection** — for every FK edge whose PK column belongs to a table
     already in results but is missing, inject the PK (so JOIN keys like
     `incident.id` that can't be keyword-matched are present).
   - **C. Display-column injection** — `get_display_columns` adds each table's
     registered display column (e.g. `documents.name`).
   - **D. Graph supplement** — merges cross-table columns the graph PPR expansion
     surfaced (e.g. `state.name` reached via a `transition → state` FK).
7. **Value-filter add-back** (`query/value_filter.py`, `VALUE_FILTER_ENABLED`) —
   re-adds columns whose **sampled values** match a query token even if their
   relevance score was low, and **prepends** them so they land within the
   `TOP_K_TO_LLM` window the SLM actually sees. Scope is expanded to FK-adjacent
   tables first.
8. **Join-path resolution.** Because V2 replaced the column set, the JOIN path is
   recomputed from the final columns via `semantic_layer._resolve_join_path` +
   `get_fk_adjacency`, so the SLM receives the correct FK JOIN edges.

The result is a `SelectedRetrieval(columns, tables, join_path, short_circuit,
source, semantic_layer_result, graph_result, stats)` handed to L3/SQL generation.

### 4.4 FK bridge injection (`query/semantic_layer.py:_inject_bridge_columns`)

Independently of which retriever won, the FK adjacency graph is used to make
retrieved tables **joinable**:

- Collect the table_ids currently in the result set and query `get_fk_adjacency`
  for every edge touching them.
- A **bridge table** is one that has FK edges to **≥ 2** retrieved tables but is
  not itself retrieved (e.g. a `role_permissions` join table between `role` and
  `permission`). Its PK is injected as an `IDENTIFIER` column with `similarity=0.0`.
- Missing PKs of tables already in the result set are also injected.
- Bounded by `FK_MAX_INJECTED_COLS`; gated by `FK_BRIDGE_INJECTION_ENABLED`.

`_resolve_join_path` then turns FK edges between the final tables into `JoinEdge`s
(preferring real FK adjacency data; falling back to `_id`-prefix name heuristics).

### 4.5 RAG chunk retrieval (`chunk_embedder.retrieve_top_k_chunks`)

For document/hybrid queries, the query is embedded with bge-m3 (1024-dim) and run
as cosine ANN over `doc_chunks`, optionally filtered by `source_id` and by a
`TemporalFilter` on `doc_date` (`BETWEEN`/`>=`/`<=`). Results are
`ChunkRetrievalResult`s fused with the SQL side by `HYBRID_SQL_WEIGHT` /
`HYBRID_RAG_WEIGHT`.

---

## 5. End-to-end trace of one query

For a query like *"total escalated incidents by assignee last month"*:

1. **Enrich** → tokens + synonyms (`incident` concept, `escalated` value hint).
2. **Signal 1 (M3 dense)** cosine-searches `column_embeddings_v2` → candidate
   columns ranked by semantic similarity.
3. **Signal 2 (learned-sparse)** scores by M3 lexical overlap; **Signal 5 (value)**
   flags the column whose sampled values contain `"escalated"` (e.g. `incident.state`);
   **Signal 6 (table prior)** boosts columns of the `incident` table.
4. **Weighted RRF (k=60, FUSION_WEIGHTS)** fuses all six signals → **intent boost**
   (AGGREGATE + TEMPORAL) → **adaptive cutoff**.
5. In `select_retrieval`, the **bi-encoder + reranker** produce the authoritative
   column set; **FK-PK / display / graph** supplements add JOIN keys and the
   `assignee` display column; the **graph retriever** PPR expansion surfaces
   `user.name` via the `incident.assignee_id → user.id` FK.
6. **FK bridge injection** guarantees the tables are joinable; **join-path
   resolution** emits the `incident → user` JOIN edge.
7. **Value-filter add-back** guarantees `incident.state = 'escalated'` reaches the
   SLM prompt; **temporal parsing** produces the `last month` date filter (applied
   to the temporal column, and to `doc_date` for any RAG side).
8. The selected columns + join path + filters go to L3 (SLM) → SQL builder → the
   **client source DB** for execution; any document chunks come from `doc_chunks`.

---

## 6. Key file map

| Concern | File |
|---|---|
| Pipeline composer (L1–L5) | `veda_core/ingestion/layers/pipeline.py` |
| Layer stages | `veda_core/ingestion/layers/l{1..5}_*.py` |
| Schema introspection | `veda_core/connectors/relational.py` |
| One encoder (dense + sparse) | `veda_core/ingestion/m3_encoder.py` |
| Column/table dense embeddings | `veda_core/ingestion/biencoder.py` |
| Learned-sparse index (Signal 2) | `veda_core/ingestion/sparse_index.py` |
| Document chunk embeddings | `veda_core/ingestion/chunk_embedder.py` |
| Graph node embeddings | `veda_core/ingestion/graph_embedder.py` |
| FK adjacency + table/display metadata | `veda_core/ingestion/vector_store.py` |
| Property graph persist | `veda_core/ingestion/graph_persist.py` |
| Unified graph builder | `veda_core/ingestion/unified_graph_builder.py` |
| 6-signal retrieval engine | `veda_core/retrieval/retrieval_engine_phase3.py` |
| Signal 1 dense / Signal 2 sparse / weighted RRF | `semantic_search.py`, `sparse_ranker.py`, `rrf_merger.py` |
| Graph seed + PPR expand | `veda_core/query/graph_retriever.py` |
| Unified column selection | `veda_core/query/retrieval_select.py` |
| FK bridge / join resolution | `veda_core/query/semantic_layer.py` |
| Retrieval eval + tuning | `scripts/retrieval_eval.py`, `build_golden_set.py`, `tune_fusion_weights.py` |
| Config (tables, dims, models, weights) | `veda_core/config.py` |
