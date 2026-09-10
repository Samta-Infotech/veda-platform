# VEDA Retrieval Quality Upgrade — Agent Execution Plan (Tier 1 + Tier 3)

> **Contract for the executing agent:** implement everything below as the ONLY code
> path — no feature flags, no legacy fallbacks kept "just in case" (the explicitly
> listed resilience fallbacks are the sole exceptions). Delete replaced code in the
> same work package that replaces it. No model training anywhere in this plan; the
> only "tuning" is a grid search of fusion weights against a golden query set
> (WP6), which is measurement, not training.
>
> **Out of scope (deliberately):** reranker/bi-encoder fine-tuning, learning-to-rank
> fusion, learned edge weights, conformal cutoffs, intent classifier — all deferred
> Tier-2 work. The cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) stays exactly
> as is.
>
> **Verification model:** the human will run ONE fresh ingestion after all WPs are
> merged and validate with WP9. Design every schema/dim change assuming a clean
> re-ingest (drop + recreate is fine; no data migration needed).
>
> **Invariants that must not change:** the firewall gates and escalation ladder in
> `veda/pipeline.py`; tenant fail-closed context; zero-egress (all model weights
> baked into images at build time, `local_files_only=True`); scoped
> delete-then-insert idempotency on every store; the `storage_adapters` seam for
> context-scoped reads.

---

## WP0 — Golden set + retrieval metrics (do FIRST, before any change)

**Goal:** a before/after number for every later WP.

1. Create `evaluation/golden_queries.jsonl`. Each line:
   `{"query": str, "gold_columns": ["schema.table.col", ...], "gold_tables": [...], "intent": str, "expected_kind": "sql|rag|hybrid"}`.
   Seed it from (a) the query cases already inside `scripts/parity_suite.py`, and
   (b) an export of `VerifiedQueryCache` — parse each verified SQL with the
   existing AST validator (`query/ast_validator.py` utilities) to extract the
   column/table sets. Target ≥ 60 queries covering: single-table filter, aggregate,
   multi-table join, value-literal ("escalated"-style), temporal, and document/RAG.
2. Create `scripts/retrieval_eval.py`:
   - Runs `query/retrieval_select.py:select_retrieval` per golden query.
   - Reports **recall@5 / recall@15 / MRR** for gold columns, **table recall@3**,
     mean candidate-set size, and wall-clock per stage.
   - Writes JSON to `evaluation/results/retrieval_<git-sha>.json`.
3. Run it against the CURRENT build + current ingestion and commit the output as
   `evaluation/results/retrieval_BASELINE.json`.

**Acceptance:** baseline JSON committed; script is deterministic across two runs.

---

## WP1 — Fix Signal-1 dense encoding (correctness bug)

**Files:** `retrieval/semantic_search.py`, `retrieval/retrieval_engine_phase3.py`.

1. `SemanticSearcher.embed_query` currently does `" ".join(tokens)` with no query
   prefix and no normalization. Replace with:
   - Signature `embed_query(self, query: str)` — takes the **raw natural-language
     query**, not enriched tokens.
   - Encode `BIENCODER_QUERY_PREFIX + query` with `normalize_embeddings=True`
     (matching how passages were stored).
2. In `RetrievalEnginePhase3.retrieve`: pass the **raw query** to Signal 1;
   enriched tokens go ONLY to Signal 2 (sparse/BM25) and Signal 5 (value index).
   Update the retrieval cache key to hash `(raw_query, enriched_tokens)` so cache
   semantics stay correct.
3. Delete the token-join embedding path entirely; no compatibility shim.

**Acceptance:** `retrieval_eval.py` recall@15 ≥ baseline (expected: improvement,
especially on multi-word natural queries); no other stage changed in this WP so
attribution is clean.

---

## WP2 — Replace ivfflat with HNSW on all engine-local vector tables

**Files:** `ingestion/biencoder.py`, `ingestion/chunk_embedder.py`,
`ingestion/graph_embedder.py`.

1. In each `CREATE INDEX` DDL, replace
   `USING ivfflat (embedding vector_cosine_ops) WITH (lists=50)` with
   `USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=200)` —
   consistent with the substrate-side HNSW already in `storage_adapters`.
2. Before each ANN query in `retrieval/semantic_search.py`,
   `chunk_embedder.retrieve_top_k_chunks`, and
   `query/graph_retriever.py` seed retrieval, issue
   `SET LOCAL hnsw.ef_search = %s` inside the transaction, resolving the value via
   `storage_adapters.reader._resolve_ef_search(source_id)` (env → SubstrateVersion
   → default 40). Reuse the existing helper; do not duplicate resolution logic.
3. Drop the old ivfflat DDL strings and any `lists=` config remnants.

**Acceptance:** fresh-ingest DDL creates HNSW (`\di` shows `hnsw`); recall@15 ≥
WP1 (exactness at this scale should only help); ANN latency unchanged or better.

---

## WP3 — BGE-M3 unification (dense + learned sparse; one embedding space)

**The core Tier-3 change.** One model — `BAAI/bge-m3` — produces the dense vectors
for columns, tables, graph nodes, AND document chunks (all 1024-dim), plus learned
sparse lexical weights that replace BM25. **Deliberately excluded:** M3's
ColBERT/multi-vector head — the existing `bge-reranker-v2-m3` cross-encoder
already provides late-interaction quality on the top candidates, and storing
per-token vectors per column is cost without measured benefit. Do not implement it.

### 3.1 Encoder singleton
- New `ingestion/m3_encoder.py`: process-wide singleton wrapping
  `FlagEmbedding.BGEM3FlagModel("BAAI/bge-m3", use_fp16=<GPU only>)`, loaded with
  local files only. Expose:
  - `encode_dense(texts: list[str]) -> np.ndarray` (1024-dim, L2-normalized)
  - `encode_sparse(texts: list[str]) -> list[dict[str, float]]` (token→weight)
  - `encode_query(text: str)` returning both in one forward pass.
  M3 requires **no instruction prefix** — set `BIENCODER_QUERY_PREFIX = ""` and
  `BIENCODER_PASSAGE_PREFIX = ""` in `config.py` with a comment stating why.
- `requirements/`: add `FlagEmbedding` (pin the version; it pulls `peft`); Docker
  build bakes `BAAI/bge-m3` into both images and **removes**
  `BAAI/bge-large-en-v1.5` and `sentence-transformers/all-MiniLM-L6-v2` from the
  bake step (reranker weights stay).

### 3.2 Column/table dense store
- `ingestion/biencoder.py`: swap the SentenceTransformer BGE-large singleton for
  `m3_encoder.encode_dense`. `BIENCODER_MODEL = "BAAI/bge-m3"`,
  `BIENCODER_DIM = 1024` (unchanged → `column_embeddings_v2` /
  `table_embeddings_v2` DDL unchanged apart from WP2's HNSW). Passage text
  strategy stays `"doc"` — the `retrieval_documents` texts are the strongest
  asset; do not change them.

### 3.3 Learned sparse replaces BM25 (Signal 2)
- New `ingestion/sparse_index.py` (L4 stage, replaces the `bm25_index` stage in
  `layers/l4_index.py`):
  - For every column passage text, `encode_sparse` → persist to internal-DB table
    `column_sparse_v1(col_id text PK per source, source_id, table_id, weights jsonb)`
    with the usual scoped delete-then-insert; same for tables into
    `table_sparse_v1`.
- New `retrieval/sparse_ranker.py` (replaces `retrieval/bm25_ranker.py`):
  - On engine warm: load the source's sparse rows once and build an in-memory
    inverted index `{token: [(col_id, weight)]}` (a few thousand columns — trivially
    fits in memory).
  - Per query: `encode_sparse(raw_query)` once (reuse the same forward pass as the
    dense query encode via `encode_query`), score columns by sparse dot product
    over the inverted index, return top-50 ranked — same interface the RRF merger
    consumed from BM25. Enriched tokens are ALSO scored (their sparse encodes are
    computed per unique expansion phrase and max-pooled per column) — this is
    where enrichment now lives.
- **Deletions in this WP:** `retrieval/bm25_ranker.py`, `ingestion/bm25_index.py`,
  the l4 `bm25_index` stage, all `BM25_*` config keys, and the BM25 references in
  `retrieval/__init__.py` / `rrf_merger.py` (signal renamed `sparse`).

### 3.4 Chunks + RAG unify on M3
- `ingestion/chunk_embedder.py`: embed chunks with `m3_encoder.encode_dense`;
  `doc_chunks.embedding` becomes `vector(1024)` (add `DROP TABLE IF EXISTS` guard
  keyed on a dim check, mirroring the pattern `graph_embedder` already uses).
- `query/rag_layer.py`: query encode via `m3_encoder` — delete the
  `_get_minilm_model` import and the MiniLM error strings.
- `ingestion/chunk_linker.py`: the `about` embedding-similarity check uses M3
  dense; re-validate `GRAPH_LINK_EMBED_SIM_MIN` (M3 cosine distributions differ —
  set initial value from a quick histogram on one ingested doc source, note the
  chosen value in the commit message).
- `ingestion/graph_embedder.py`: chunk-vector copy from `doc_chunks` now yields
  1024-dim — remove the mixed-dimension caveat comments; col/table node sentences
  encode via `m3_encoder`.

### 3.5 Retire MiniLM and the last legacy encoder remnants
With 3.4 done, MiniLM has zero callers. Delete:
- `ingestion/relgt_encoder.py` (the `_get_minilm_model` host) and the ensemble
  write functions remaining in `ingestion/vector_store.py`
  (`fk_adjacency`/`table_metadata`/`column_values` stores in that file STAY — only
  the encoder-embedding write paths go).
- Config keys: `MINILM_*`, `VECTOR_TABLE_NAME_LIGHT_TEXT`, `VECTOR_TABLE_NAME_HYBRID`,
  `RELGT_*`, `ENCODER_MODE` and its mode vocabulary — replace the `ENCODER_MODE`
  guard in `apps/ingestion/tasks.py` with a single `EMBEDDING_MODEL_ID = "bge-m3"`
  stamp (same purpose: refuse silent model change between resume runs).
- Substrate mirrors `ColumnEmbeddingLT` / `ColumnEmbeddingHybrid` in
  `apps/substrate/models.py`, their entries in `storage_adapters.reader.ann_search`'s
  mode map, and a migration that `DROP TABLE IF EXISTS` both tables.
- `scripts/rerun_bge_finetune.sh`, `scripts/fresh_homzhub.sh` (client-named,
  stale), and any now-orphaned `SYNTHETIC_*` config keys (verify zero importers
  after relgt_encoder is gone; delete only if zero).

**Acceptance:** repo-wide grep for `MiniLM|bge-large|bm25|ENCODER_MODE` returns
only CHANGELOG/docs; compile clean; unit test in `tests/` asserting
`m3_encoder.encode_query` returns (1024-dim normalized dense, non-empty sparse
dict) for a sample sentence.

---

## WP4 — Table-first prior in column scoring

**Files:** `retrieval/retrieval_engine_phase3.py`, `query/retrieval_v2.py`,
`config.py`.

`retrieval_v2` already fetches candidate tables from `table_embeddings_v2` and
reranks them; the change is to make table affinity a **prior on column scores**
instead of a parallel output.

1. New config constants: `TABLE_PRIOR_TOP_M = 10`, `TABLE_PRIOR_BETA = 0.3`
   (starting value; WP6 tunes it).
2. In the 5-signal engine: once per query, ANN the dense query vector against
   `table_embeddings_v2` (top-M) → `table_sim` map. Add a sixth ranked signal
   `table_prior` to the fusion where each column candidate inherits
   `table_sim[table_of(col)]` (0 if its table is outside top-M). Soft prior only —
   do NOT hard-filter columns to top-M tables (that silently kills rare-table
   recall).
3. In `retrieval_v2`: blend `score = col_score + TABLE_PRIOR_BETA * table_sim`
   into the first-stage candidate scores **before** the cross-encoder rerank,
   reusing the candidate-table fetch it already performs (no second ANN).
4. Sparse side symmetry: `sparse_ranker` also loads `table_sparse_v1` and
   contributes table sparse scores into the same `table_sim` map (max of dense and
   sparse per table) so keyword-only queries get the prior too.

**Acceptance:** table recall@3 improves on the golden set; column recall@15 does
not regress on single-rare-table queries (add two such queries to the golden set
explicitly).

---

## WP5 — Personalized PageRank replaces hop-decay BFS in the graph retriever

**Files:** `query/graph_retriever.py`, `config.py`.

1. On first use per (source, process), load the source's full edge list from
   `graph_edges` (one query) and build a scipy CSR weighted adjacency over node
   ids; **row-normalize** into a transition matrix (row normalization is the hub
   treatment — high-degree nodes dilute naturally). Cache it; invalidate on
   rehydrate (hook the same subscriber that clears `_SM`).
2. Seeds: the existing ANN seed retrieval stays. Build restart vector `p0` from
   seed similarities (normalized to sum 1).
3. Score: PPR `p = (1-d)·p0 + d·Pᵀp`, damping `GRAPH_PPR_DAMPING = 0.85`, iterate
   to `GRAPH_PPR_TOL = 1e-6` or `GRAPH_PPR_MAX_ITERS = 50`. Milliseconds at this
   graph size.
4. Keep unchanged: the single-table short-circuit, sibling inclusion, the chunk
   safety net (chunks are reachable through `mentions`/`about` edges in the
   transition matrix now, but keep the direct pull as belt-and-braces since chunks
   are leaves), and the `GRAPH_MAX_COLS_TO_L3` truncation.
5. Delete: the BFS expansion loop, `GRAPH_HOP_DECAY`, `GRAPH_EXPAND_HOPS`,
   `GRAPH_EXPAND_MAX_NODES`, `GRAPH_HUB_DEGREE_CAP` and their config keys.
   `scipy` is already a transitive dependency of the ML stack; pin it explicitly in
   requirements.

**Acceptance:** on the golden multi-table queries, the columns reached via 2-hop
FKs (the `state.name`-via-`transition` pattern) appear at equal or better rank;
graph stage wall-clock ≤ BFS baseline.

---

## WP6 — Weighted fusion + tuning harness (measurement, not training)

**Files:** `retrieval/rrf_merger.py`, `config.py`, new
`scripts/tune_fusion_weights.py`.

1. Generalize RRF to weighted RRF: `score(d) = Σ_s w_s / (k + rank_s(d))` with
   `FUSION_WEIGHTS = {"dense": 1.0, "sparse": 1.0, "subgraph": 1.0, "fk": 1.0,
   "value": 1.0, "table_prior": 1.0}` in config (identity start = current
   behavior). Keep the existing score→virtual-rank conversion for the two
   score-typed signals.
2. `scripts/tune_fusion_weights.py`: random search (500 samples, weights in
   [0.25, 3.0], fixed seed) + local refinement, objective = recall@10 on
   `golden_queries.jsonl` with recall@15 as tiebreak; prints the best dict and the
   per-query wins/losses. The human runs this once after fresh ingestion and
   commits the resulting constants into `config.py` — the script does NOT write
   config itself.

**Acceptance:** identity weights reproduce pre-WP6 ranking bit-for-bit (unit test
with a fixed candidate fixture); tuning script is deterministic given the seed.

---

## WP7 — De-flag the Track-4 precompute paths (precompute becomes THE path)

The seven consumption flags were a parity-rollout mechanism; this plan's fresh
ingestion IS the cutover, so remove the dual paths.

1. Delete from `config.py`: `_env_flag` and the seven flags
   (`BM25_PERSISTED_INDEX_ENABLED` — already gone via WP3 — plus
   `ENRICHMENT_INDEX_ENABLED`, `JOIN_PATHS_ENABLED`, `VALUE_MIRROR_ENABLED`,
   `SUBSTRATE_SIGNALS_ENABLED`, `RERANK_DOCS_ENABLED`,
   `FAST_PATH_EXPANSION_ENABLED`) and `NL_TEMPLATE_ENABLED`. Remove all eight from
   `apps/core/settings_bridge.py`.
2. At each read site, the precompute path becomes unconditional. Distinguish
   carefully:
   - **Delete outright (legacy duals):** `SignalBuilder`'s live
     `information_schema` introspection branch (substrate `FkEdge` read is the
     only path — this also removes the query tier's last source-DB dependency
     outside L7); the enrichment path that parses the four JSON files at warm
     (merged enrichment index only); the reranker's runtime `_table_text`/
     `_col_text` assembly (precomputed rerank docs only, with a hard error at warm
     if the artifact is missing — that means ingestion is incomplete, fail loud);
     fast-path reads the expanded registry unconditionally.
   - **Keep as resilience fallback (not a flag):** value resolution stays
     Redis-first with the existing Postgres `column_values` fallback (Redis can be
     cold/restarted); NL templates fire for canonical shapes with the SLM handling
     everything else (that IS the design, not a flag); join-planner consults the
     precompiled map first and falls back to live graph traversal for unmapped
     pairs (schema drift between ingestions).
3. Update `docs/CLEANUP_PLAN.md` / `ARCHITECTURE.md` to reflect single-path
   reality.

**Acceptance:** grep for `_env_flag|_ENABLED` in the retrieval/query modules shows
only genuinely behavioral gates (e.g. `UNIFIED_GRAPH_ENABLED`); warm start of the
inference service performs zero source-DB connections (assert in a test by
pointing the source env at a dead host and warming the engine).

---

## WP8 — Docs + mapping-document refresh

Update the data-mapping document (the one describing embeddings/graph/query flow)
to the post-plan state: M3 everywhere, sparse replaces BM25, six-signal weighted
fusion with table prior, PPR expansion, single-path precompute reads, HNSW
indexes, and drop §2.5 (legacy ensemble) entirely. Update the summary table of
embedding stores (all 1024-dim, one model, plus the two sparse tables).

---

## WP9 — Fresh-ingestion verification protocol (for the human)

Run after all WPs are merged and images rebuilt:

1. `pytest tests/` green; `python -m py_compile` over the tree clean.
2. Fresh ingestion via `POST /api/v1/admin/ingest`; all stage rows SUCCESS.
3. Store checks on the internal DB:
   - `column_embeddings_v2`, `table_embeddings_v2`, `graph_node_embeddings`,
     `doc_chunks` all `vector(1024)`; row counts match column/table/chunk counts
     from the scan log.
   - `column_sparse_v1` / `table_sparse_v1` populated; `\di` shows `hnsw` on every
     vector table; no `column_embeddings_lt/_hybrid` tables remain.
4. `scripts/retrieval_eval.py` → compare against
   `evaluation/results/retrieval_BASELINE.json`. Gate: recall@15 and MRR ≥
   baseline on every query class; investigate any per-query regression before
   accepting.
5. `scripts/tune_fusion_weights.py` → commit tuned `FUSION_WEIGHTS`; re-run eval to
   confirm the tuned numbers.
6. Smoke: one value-literal query, one 2-hop join query, one document/RAG query,
   one hybrid query; verify NL answers and that `/readyz` is green with the source
   DB temporarily unreachable (proves WP7's decoupling — L7 execution excepted).
7. Latency: compare `QueryLog.latency_ms` histograms for a day of traffic against
   the prior build.

---

## Execution order & dependencies

```
WP0 (baseline)
 ├─ WP1 (dense fix)          — independent
 ├─ WP2 (HNSW)               — independent
 ├─ WP5 (PPR)                — independent
 ├─ WP3 (M3 unification)     — largest; do on its own branch
 │    └─ WP4 (table prior)   — after WP3 (uses M3 table embeds + sparse tables)
 │         └─ WP6 (weighted fusion) — after all signals exist
 ├─ WP7 (de-flag)            — after WP3 (BM25 flag dies there)
 └─ WP8 (docs) → merge → rebuild images → WP9 (human verification)
```

Suggested agent batching: **Batch 1** = WP0+WP1+WP2+WP5 (small, independent,
low-risk). **Batch 2** = WP3 (big, isolated). **Batch 3** = WP4+WP6+WP7+WP8.
Run `scripts/retrieval_eval.py` between batches if a current ingestion is
available; otherwise rely on unit tests until the final fresh ingest.