# VEDA — Ingestion & Query Pipelines (complete reference)

One self-contained walkthrough of both halves of the system: how a raw source becomes
everything the platform trusts as fact (**ingestion**), and how a natural-language question
turns into a grounded, executed answer (**query**). Written from a direct read of the code
on `master`, verified live against a running instance on 2026-09-10 (including two real bugs
found and fixed that day — see §8).

**The one invariant that governs everything here:** the query tier **never introspects a
source**. It only reads what ingestion published. If a fact isn't in an ingestion artifact,
the query tier cannot see it, no matter how the question is worded.

For line-level depth beyond this document: [`ARCHITECTURE.md`](ARCHITECTURE.md) (system map),
[`INGESTION.md`](INGESTION.md) / [`QUERY_ENGINE.md`](QUERY_ENGINE.md) / [`RETRIEVAL.md`](RETRIEVAL.md)
/ [`MULTI_SOURCE.md`](MULTI_SOURCE.md) (the dedicated deep-dives this document distills), the
layer contracts (`veda_core/ingestion/layers/contracts/`, `veda_core/query/contracts/`), and
the per-directory `AGENTS.md` files next to the code.

---

## 0. The shape of the whole system

```
┌─────────────────────────── OFFLINE: INGESTION ───────────────────────────┐
│  a source (Postgres / CSV / Parquet / documents / Mongo…)                │
│         │                                                                │
│         ▼                                                                │
│  L1 EXTRACT → L2 ANALYZE → L3 ENRICH → L4 INDEX → L5 PUBLISH             │
│         │                                                                │
│         ▼                                                                │
│  artifacts: semantic model · embeddings (dense+sparse) · relationship    │
│  graph · unified graph · value index · rerank docs · cross-source edges  │
└──────────────────────────────────┬─────────────────────────────────────-┘
                                    │ (the query tier reads ONLY these)
┌───────────────────────────────────▼───────────────────────────────────────┐
│                         ONLINE: QUERY                                     │
│  question → scope resolution (RBAC ∩ ready ∩ pin)                        │
│      → front door (run_hybrid_query)                                     │
│          → multi-source coordinator (which source(s)?)                   │
│          → modality router (sql / rag / hybrid / nosql)                  │
│          → retrieval spine (6 signals → RRF → rerank)                    │
│          → deterministic SQL head (escalation ladder + firewall)         │
│              or RAG / hybrid / NoSQL head                                │
│      → grounded, executed answer                                        │
└────────────────────────────────────────────────────────────────────────-┘
```

Two process tiers carry this at runtime: a thin Django/DRF **api** tier (imports no engine
code, calls over HTTP) and a warm FastAPI **inference** tier (one engine per worker, per
`(source, tenant)` scope) that actually runs both pipelines. Ingestion runs as a Celery task
that launches the engine in a **subprocess** (isolates the engine's own top-level `config`
module from the Django `config` package, which otherwise collide in one interpreter).

---

# PART A — The ingestion pipeline

## A.1 Entry and real dispatch path

```
apps/ingestion/tasks.py :: task_ingest_source(source_id, tenant, force, skip_llm, resume)
  │  creates IngestionJob + ordered IngestionStage rows
  │  guards against an embedding-model change since the last successful job
  │  injects Source.as_engine_env() → VEDA_SOURCE_* into the subprocess env
  ▼
subprocess.Popen(["python","-u","-c", prog], cwd=veda_core)
  │  routes by source.source_kind():
  │    relational / tabular file → "import main; main.run_ingestion(...)"
  │    nosql / document / other datalake → "ingestion.source_dispatcher.dispatch_ingestion(...)"
  ▼
veda_core/main.py :: run_ingestion()          ← now a THIN SHIM, not a monolith
  │  ctx = SourceContext.from_env()  — the one place the engine learns which source it runs for
  ▼
veda_core/ingestion/dispatcher.py :: dispatch(ctx)
  │  relational or tabular-file engine → layers.pipeline.run_layered_ingestion(ctx)
  │  else → source_dispatcher.dispatch_ingestion(cfg)
  ▼
veda_core/ingestion/layers/pipeline.py :: run_layered_ingestion(ctx)
     composes L1 → L2 → L3 → L4 → L5, threading one in-memory `state` dict,
     emitting `[[STAGE]] <layer> <stage> <ok|fail|fatal>` on stdout (parsed by
     tasks.py into live IngestionStage progress rows)
```

**There used to be a monolithic pipeline.** It was deleted. The layer modules are thin
wrappers around the exact same stage functions the old monolith called — "a move, not a
rewrite." Any doc or comment describing `veda_ingestion.py`, `embed_only.py`, or a numbered
"Step N of 12" is describing a codepath that no longer exists.

On success, `task_ingest_source` calls `task_warm_caches` → `storage_adapters.writer.warm()`
(persists the semantic model into the Django substrate, pulls FK/value/glossary/graph data
from the engine's store, publishes the assembled model to Redis, and fans out a rehydrate
signal to every inference replica) — only then does `Source.ready` flip to `True`. A source
the query tier reads is either fully built or not visible at all; there is no half-built
state a query can land on.

## A.2 The five layers

Only **L1** ever touches the source. Everything after it works from what L1 extracted.

### L1 EXTRACT

| Stage | What it does | Fatal? |
|---|---|---|
| `schema_scan` | Introspects the live schema (tables, columns, types) via the source connector. | **FATAL** |
| `materialize_parquet` | Tabular file sources only — writes the file to `ARTIFACT_ROOT/<source_id>/tables/*.parquet` so it can be queried by DuckDB later. | non-fatal |
| `fk_adjacency` | Stores declared foreign keys — this becomes the join engine's source of truth. | **FATAL** |
| `data_graph` | Samples real column values and correlates them to find **undeclared** foreign keys (value overlap + co-null correlation), merging high/medium-confidence discoveries into the same FK-adjacency store. | non-fatal |

### L2 ANALYZE (pure transforms — no source access, no LLM)

| Stage | What it does | Fatal? |
|---|---|---|
| `semantic_types` | Classifies every column into one of 6 semantic types (MONETARY / TEMPORAL / CATEGORY / IDENTIFIER / METRIC / FREE_TEXT) via a 3-layer rule engine, plus a primary-display-column pass. | **FATAL** |
| `table_metadata` | Stores display-column choices and table notes. | **FATAL** |
| `value_profiling` | Samples distinct non-null values from CATEGORY/FREE_TEXT/IDENTIFIER columns → the value-grounding store the query firewall checks literals against. | non-fatal |
| `column_sketches` | 128-permutation MinHash over join-key-shaped columns → feeds cross-source FK discovery in L5. No-op if the `datasketch` package is absent. | non-fatal |
| `reg_graph` | Builds the in-memory Relational Entity Graph (table/column nodes + has-column/fk-to edges) that L4's graph stages persist. | **FATAL** |
| `join_paths` | Precomputes the shortest FK path (≤ 4 hops) between every table pair, for the query-time join planner. | non-fatal |

### L3 ENRICH (the only layer that calls an LLM)

| Stage | What it does |
|---|---|
| `semantic_layer` | A 5-stage hybrid build — profiling → glossary → table understanding → column understanding → retrieval-document generation — using a local Qwen model via Ollama for stages 2–4, everything else deterministic. Produces the **semantic model** (`data/veda_semantic_model.json`): the single artifact almost every downstream consumer, on both the ingestion and query sides, ultimately reads from. |

Skipped (still marked `ok`, degraded gracefully) when `resume=True` and the model file
already exists, when `skip_llm=True`, or when the semantic-layer flag is off.

### L4 INDEX (model inference — all non-fatal, degraded-tolerant)

| Stage | Produces | Feeds |
|---|---|---|
| `graph_persist` | `graph_nodes` / `graph_edges` from the REG graph | query-time `GRAPH_EXPAND`, PPR expansion |
| `graph_embed` | Embeddings for graph nodes | graph seed retrieval |
| `biencoder` | **`column_embeddings_v2` / `table_embeddings_v2`** — dense BGE-M3 vectors, HNSW-indexed, keyed by `source_id` | Retrieval Signal 1 (dense semantic search), table routing |
| `sparse_index` | `column_sparse_v1` / `table_sparse_v1` — BGE-M3 **learned-sparse** lexical weights | Retrieval Signal 2 (replaces BM25) |
| `enrichment_index` | One pre-tokenized/inverted index of synonyms + concept graph + glossary | Query-time query enrichment |
| `rerank_docs` | Precomputed cross-encoder pair text per column/table | The reranker (query time never assembles this text itself) |

**One encoder for everything:** a single `BAAI/bge-m3` process singleton produces every dense
vector and every learned-sparse weight — columns, tables, graph nodes, document chunks, and
the query itself at retrieval time. There is no fine-tuning stage any more, and no
"`ENCODER_MODE`" switch — that whole multi-encoder ensemble (relgt / light-text / hybrid /
MiniLM) was removed. `EMBEDDING_MODEL_ID = "bge-m3"` survives only as the guard that forces a
re-ingest when the embedding model itself changes.

### L5 PUBLISH (derived registries — all non-fatal)

| Stage | Produces | Feeds |
|---|---|---|
| `relationship_graph` | `data/<tenant>/<source_id>/veda_relationship_graph.json` (per-source since P0-1, 2026-09-10 — was one flat file shared by every source) — declared FK + polymorphic edges resolved by **data correlation** (not string matching), weighted so audit/history tables (weight 10) never get routed through as a join hub | Query-time join planner, fast path, firewall |
| `semantic_registry` | `semantic/{concepts,dimensions,metrics}.json` | The deterministic fast path |
| `value_referents` | Per-value direct + FK-closure referents | Deterministic planners, the qualifier-completeness gate |
| `hnsw_tune` | A per-source HNSW `ef_search` recommendation, sized by table count | Pinned into `SubstrateVersion` → the query tier's ANN search width |
| `value_mirror` | Sampled values mirrored into Redis | Query-time value resolution (Postgres fallback) |
| `unified_graph` | `data/veda_unified_graph.json` — semantic model + relationship graph + concepts + synonyms fused into one graph | Query-time `GRAPH_EXPAND` booster |
| `cross_source_fk` | **Tenant-wide**, every ingest: compares `column_sketches` across *all* sources to discover cross-source foreign keys | The multi-source coordinator's structural edge detection, federated join planning |

## A.3 Non-relational sources

Document, NoSQL, and non-tabular-file datalake sources skip the layered pipeline entirely and
go through `source_dispatcher.dispatch_ingestion` instead — a lighter, type-aware chain:

- **Document sources**: walk the directory, chunk each file (layout-aware for PDF/DOCX —
  heading hierarchy preserved), embed each chunk with the same BGE-M3 model into `doc_chunks`,
  then run **entity linking** (`entity_linker.py`) — dictionary + pattern + optional-SLM
  detection of named entities, bridging `chunk --mentions_entity--> entity --value_of-->
  column`, and the **semantic bridge** (`semantic_linker.py`) — matching each chunk's vector
  against column embeddings to add `semantic_about` edges (chunk → column, never a join,
  purely for graph expansion at query time).
- **NoSQL sources**: schema inferred by sampling documents; only a schema-only chain runs (no
  value sampling, no data-graph correlation — those assume a relational value model).
- **Tabular files (CSV/Parquet/Excel) presented as datalake**: two connectors exist —
  `tabular_files.py` runs the **full** layered pipeline with deterministic UUIDv5 ids
  (idempotent re-ingestion); `datalake.py` is a lighter DuckDB-only path with random ids, used
  for non-tabular datalake formats (Delta/Iceberg).

## A.4 Cross-source linking (built at ingestion, consumed at query time)

This is what lets the query tier ever combine two sources at all:

1. **Column sketches** (L2, per source) — 128-perm MinHash over join-key-shaped columns, plus
   an **exact-containment set** of hashed values when the column has few enough distinct
   values, so a small child table's foreign key can be matched against a large parent table's
   primary key even when MinHash's coarse resolution would miss it.
2. **`cross_source_graph.discover_and_persist`** (L5, tenant-wide, every ingest) — compares
   sketches across *every* source pair, tiers matches `HIGH`/`MEDIUM` by containment + a
   name-affinity gate (the join key must be a business/string key, or the discriminator name
   must plausibly refer to the target table — this defeats the "small surrogate IDs
   coincidentally overlap" false-positive class), and persists `cross_source_fk` edges.
3. **Entity linking** (documents) — the same graph gets `entity` nodes bridging document
   mentions to structured columns.
4. **Semantic bridge** (documents, Tier A shipped) — `semantic_about` edges from document
   chunks straight to columns by embedding similarity.

All of this lands as `graph_edges` in the engine's internal store. The query tier's
multi-source coordinator reads these edges to detect genuine cross-source joins (Part B, §3).

## A.5 Resume — stage-level skip, not resume-from-N

`VEDA_RESUME=1` is set automatically when a prior job failed, or explicitly requested. Its
actual effect: **L3 skips** if the semantic-model file already exists; **L4's biencoder stage
skips** if `column_embeddings_v2` already has rows for THIS source (fixed 2026-09-10 — it
previously checked for ANY rows at all, so ingesting source B could see source A's rows and
wrongly skip B's own embedding entirely, leaving B with no retrieval Signal 1; see
`docs/backlog/query-engine-open-items.md`). L1 and L2 always re-run — the engine holds
everything in an in-memory `state` dict between stages, so there is no persisted mid-run
checkpoint to resume from; only the two most expensive stages have their own existence check.

L3's own skip check is **still source-unaware** (the semantic-model file it looks for is flat,
shared by every source, until that artifact gets the same per-source fix the relationship graph
got in P0-1) — a real gap, deliberately left unfixed rather than patched in isolation, since a
narrow fix there would just move the bug to L3's *save* step overwriting another source's model
(see the backlog doc's "P0-3, second half" entry for the full reasoning).

## A.6 Every artifact, one table

| Artifact | Where | Written by | Read by |
|---|---|---|---|
| Schema scan result | in-memory only | L1 | L2, L4, L5 (same process) |
| FK adjacency | engine store | L1 (+ L1 data_graph) | Query join planner |
| Materialized parquet | `data/<source_id>/tables/*.parquet` | L1 (tabular only) | Query federated executor (DuckDB) |
| Sampled values | engine store | L2 | Value grounding, value arbiter, L5 value_mirror/value_referents |
| Column sketches | engine store | L2 | L5 cross-source FK discovery |
| Semantic model | `data/veda_semantic_model.json` | L3 | Query routing, grounding, qualifier gate, grain; L4 biencoder; L5 registry/graph |
| Domain synonyms / concept graph / glossary | `data/veda_*.json` | L3 | Query enrichment, unified graph |
| Column/table embeddings (dense) | `column_embeddings_v2` / `table_embeddings_v2` (pgvector, HNSW) | L4 | Retrieval Signal 1, table routing |
| Sparse index | `column_sparse_v1` / `table_sparse_v1` | L4 | Retrieval Signal 2 |
| Enrichment index | `data/veda_enrichment_index.json` | L4 | Query enrichment |
| Rerank docs | `data/veda_rerank_docs.json` | L4 | The cross-encoder reranker |
| Relationship graph | `data/<tenant>/<source_id>/veda_relationship_graph.json` (per-source, P0-1) | L5 | Join planner, fast path, firewall |
| Semantic registry | `semantic/*.json` | L5 | Deterministic fast path |
| Value referents | per-tenant/source JSON | L5 | Deterministic planners, qualifier gate |
| HNSW tuning | `SubstrateVersion.hnsw_ef_search` | L5 → `task_warm_caches` | Query-time ANN search width |
| Redis value mirror | `value:{tenant}:{source}:{norm}` | L5 | Value resolver (Postgres fallback) |
| Unified graph | `data/veda_unified_graph.json` | L5 | `GRAPH_EXPAND` booster |
| Cross-source FK edges | `graph_edges` (engine store) | L5, tenant-wide | Multi-source coordinator, federated join planner |
| Document chunks (embedded) | `doc_chunks` (pgvector) | document path | RAG head, graph retrieval |
| Entity / semantic-about edges | `graph_edges` | document path | Cross-source graph traversal (PPR) |
| Assembled `sm` + Sm* substrate | Django DB + Redis | `task_warm_caches` → `writer.warm()` | The entire query tier (canonical runtime model) |

---

# PART B — The query pipeline

## B.0 The two-tier request path

```
client → nginx → Django QueryView.post
   resolve tenant → RBAC-narrow permitted sources → intersect with ready sources
   → intersect with any explicit request pin → source_ids (primary first)
   → build X-Veda-Data-Scope (table/column allow payload) if RBAC is on
   → InferenceClient (HTTP, stdlib urllib) ──────────────────────┐
                                                                   ▼
                                          inference: _tenant_context middleware
                                          sets RequestContext from headers
                                          (fail-closed if source/tenant missing)
                                                                   ▼
                                          routes/hybrid.py → run_hybrid_query(query)
```

`InferenceClient` has **no retry and no circuit breaker** (only a 300s timeout) — any
transport failure becomes a clean `503`, never a `500`. Any unhandled exception on the
non-streaming inference route also can't leak as a raw crash to the client; the streaming
route turns a mid-run failure into an SSE `event: error` frame instead of hanging.

## B.1 The front door — `run_hybrid_query`

This is the single public entry point and it **always returns a `MultiResult`**. It mints one
trace for the whole request and runs an ordered list of layers, each of which can answer or
refuse before the next one is even reached:

```
run_hybrid_query(query)
  1. L0 runtime_context     — pure system-value questions ("what's today's date") answer
                              immediately, before any retrieval.  [ON]
  2. Multi-source coordinator (see §3) — decides which source(s), and for a genuine
                              cross-source (MULTI) decision, DRIVES the answer.
  3. Federated route        — if the ambient scope still spans ≥2 sources and retrieval
                              hits more than one, try one cross-source SQL query.
  4. Decomposition          — OFF (splits join queries incorrectly); every query goes
                              straight to the single-modality dispatch below.
  5. classify(query)        — pick ONE modality for the scope: sql / rag / hybrid / nosql
  6. dispatch to that modality's head
```

`classify()` (§B.4) runs two doc-intent overrides *before* the keyword router — a fixed
word-list check and an evidence-based check (which reuses the coordinator's own cosine
evidence) — so a document-shaped question can be routed to `rag`/`hybrid` even in a scope that
also has relational sources.

## B.2 Scope resolution — which sources can even answer

Before any of the above, `apps/query/scope.py` computes the actual source set for the
request: **RBAC-permitted sources ∩ ready sources ∩ any explicit request pin**, primary
source first. An empty permitted set refuses (`403`, no audit row); no ready source in scope
refuses (`503`). This scope — not the wording of the question — is what "which sources are in
play" means everywhere below.

## B.3 The multi-source coordinator — which source(s), and how confident is it

This is the machinery that makes the system genuinely multi-source-aware rather than just
"try one keyword-picked source." It scores **every** source in scope independently before any
answer is generated:

1. **Evidence** (`source_evidence.py`) — a plain cosine lookup, per source, over that source's
   own columns and document chunks (deliberately decoupled from the answer engine's own
   retrieval — routing needs the *raw* query↔item similarity, not a reranked answer score).
   Each source gets a `presence_tier`: **STRONG**, **WEAK**, or **NONE**, computed *relative to
   the best score across all sources* (a dominance rule) so a large database's spurious
   partial match on every query can't drown out a genuinely relevant smaller source.
2. **Policy** (`routing_policy.py`, pure, deterministic, no LLM):
   - 0 candidates → `NO_MATCH`.
   - 1 candidate, or exactly one `STRONG` source with the rest merely `WEAK` → `SINGLE`.
   - ≥2 candidates connected by a real, discovered `cross_source_fk` edge (and genuinely
     close in score — not "the top one plus a distant edge-connected runner-up") → `MULTI`
     (a computed structural fact, never a guess).
   - Same-domain sources with no join, exactly one marked canonical → `SINGLE` (canonical
     tie-break).
   - A separate deterministic override: if the **question itself names entity tokens from
     both endpoint tables** of a `HIGH`-tier cross-source edge that's in scope, force `MULTI`
     even if the raw evidence alone would have picked one side.
   - Anything genuinely unclear → escalate to **one bounded SLM call** (`routing_slm.py`) that
     picks among a compact, evidence-backed candidate set — never a free-form guess, and its
     output is structurally validated before it can route anything.
3. **Permission-aware pre-check** (flag-gated, on by default) — if the strict best-matching
   source is one the caller can't access, it distinguishes "you're not entitled to this data"
   from "a source you *do* have is just as good," using a measured score-margin rather than
   identity alone (naming the inaccessible source itself is never leaked).

### What the coordinator actually decides on your behalf (as of 2026-09-10)

| Decision | Behavior |
|---|---|
| **`MULTI`** (genuine cross-source) | **Always authoritative.** Tries a bounded doc+data grounding step first (for "which entities named in this document also exist in our data" questions — grounded, the SLM can only quote entities that literally appear in the retrieved text and pick a data column from a numbered candidate list, never invent either); then a genuine join → strict federated execution (a failed join is a **surfaced, honest refusal**, never a silent wrong answer or a silent single-source fallback); or, for an SLM-resolved multi with no structural edge, runs each source independently and merges the results under an explicit policy (append / conflict-detected / canonical-priority). |
| **`SINGLE` / `NO_MATCH` / `CLARIFICATION_REQUIRED`** | Computed and traced, but **deferred** — the legacy keyword router + federated route answer instead. |

**Why the split, not "always authoritative":** it was tried unscoped first (per the
originally-documented rollout plan) and it regressed plain single-source questions — the
coordinator's own routing-evidence pass is a much simpler signal than the real answer engine's
6-signal retrieval + fast path + rerank + Tier-2 LLM fallback, and making it gate *every*
decision meant it could refuse a question the real engine would have answered fine. A `MULTI`
decision has no such failure mode (there's no single-source legacy answer being defeated), so
it's authoritative unconditionally; everything else stays advisory until the routing-evidence
signal is proven as reliable as the engine it would otherwise override.

**Live-verified, organically** (not a test mock): `"which assets have which amenities"`
resolved `ROUTED/MULTI (RELATIONSHIP_EDGE)` on the coordinator's own evidence — it found the
real discovered edge between the amenities-catalog source and the relational assets table,
tried strict federation, couldn't build a validated join for that phrasing, and refused
honestly rather than guess. A near-identical wording, `"list amenities for each asset"`,
resolved `NO_MATCH` from the coordinator and was correctly deferred — the legacy path answered
it via a same-source join, because that particular question didn't actually need the other
source despite naming both entities.

## B.4 Modality routing — `classify()`

Once the coordinator has (or hasn't) settled the source question, exactly one modality is
picked for the query:

1. **Fixed-word doc-intent override** — a document-vocabulary word list matched against a
   probe that the scope actually has chunk-backed sources → `rag` (or `hybrid` if the
   question also has an aggregation verb).
2. **Evidence-based doc-intent override** (on by default) — reuses the coordinator's own
   cosine evidence; a chunk-backed dominant source → same `rag`/`hybrid` split.
3. **Keyword router** (`query_router.py`) — a plain keyword-signal counter (SQL / RAG / NoSQL
   / temporal keyword sets, temporal counted double). **With no document or NoSQL sources
   configured in scope it returns `sql` unconditionally** — the common single-relational-
   source case. There is no embedding fallback here despite what older comments claim.
4. Any exception anywhere in this chain → `sql` (the safe default).

## B.5 Retrieval spine — 6 signals, fused, reranked

Every SQL-shaped query calls `get_engine(sm).retrieve(query, top_k=15)`. This is where "how
much can this source tell me about this question" is actually measured for the *answer*, as
opposed to the coordinator's separate *routing* evidence:

| # | Signal | Computed by |
|---|---|---|
| 1 | Dense semantic | BGE-M3 dense cosine over `column_embeddings_v2`, raw query, HNSW, per-source `ef_search` |
| 2 | Learned-sparse | BGE-M3 lexical weights — **replaces BM25** |
| 3 | FK subgraph | Static per-column scalar (table degree), not a live traversal |
| 4 | FK path / join-key | Static per-column scalar (is-FK / is-referenced) |
| 5 | Value index | Literal-in-query matched to the column that holds that value |
| 6 | Table-first prior | Dense ⊕ sparse table-level affinity — boosts existing candidates only, never adds new ones |

Fused by **weighted RRF** (currently identity weights — all `1.0`), then an **intent boost**
(large, deliberate deltas on analytics role — e.g. a strong penalty on audit/history tables),
then an **adaptive cutoff** that cuts at the biggest score gap ("semantic cliff") rather than a
fixed top-k. A **cross-encoder rerank** (precomputed pair text from ingestion, not assembled
per query) then overwrites the final ranking before anchor selection, skipped only when the
top-2 RRF gap is already unambiguous.

**Graph expansion — two distinct mechanisms, not one:**
- The retrieval-spine booster (`suggest_expansions`) is synonym/alias resolution + a single
  FK hop — not PageRank, not BFS.
- The datalake/cross-source retrieval path (`run_graph_retrieval`) is genuine **Personalized
  PageRank** over the unified graph, which is what actually traverses the `cross_source_fk`
  and `semantic_about` edges ingestion built.

## B.6 The deterministic SQL head — the correctness path

`veda/pipeline.py::run_query` is where a SQL-shaped question actually turns into an answer.
It runs an **escalation ladder**, stopping at the first answer that survives the firewall:

```
Fast path (compiled registries, no retrieval, no LLM)
  → deterministic superlative / grouped / ratio planners (one-anchor analytical SQL)
    → fast-path evidence guard (demote a zero-evidence pick back into the full pipeline)
      → verified-query cache (BGE-M3 cosine ≥ 0.85, re-checked by the same evidence guard
        AND a fresh qualifier-completeness pass before being trusted)
        → FULL PATH:
            retrieve (§B.5) → graph-expand → RBAC filter → rerank
              → anchor selection (+ entity resolution)
                → join needed?  → deterministic join plan / existence / aggregate
                                   (LLM only ever fills SELECT/WHERE in a fixed
                                   FROM/JOIN skeleton it did not choose)
                → single table  → a deterministic sub-ladder (answer-entity, FK-value,
                                   multi-hop FK, categorical filter, temporal window,
                                   temporal-refuse) → only the last rung calls the LLM,
                                   and even then a deterministic builder is tried first
```

**Every** generated SQL — deterministic or LLM-filled — then passes a firewall before
anything executes:

1. **Value grounding** — every filter literal must exist in sampled real data.
2. **Qualifier completeness** — every word the user actually used must be represented
   somewhere in the SQL (with a retry-with-forced-anchor "salvage" step, then a grounded
   clarify using real FK domain values, before giving up).
3–8. **Six "silent-wrong" alignment guards** (all default on) — catches: answered without
   the requested grouping, answered without the requested DISTINCT, a temporal breakdown with
   no date bucketing, a stated measure column measured on the wrong table, "how many X" with
   no aggregate function at all, and a GROUP BY that isn't in the family the question asked
   about.
9. IR-equivalence (LLM-generated SQL only) — no extra filter, join, GROUP BY, ORDER BY, or
   DISTINCT the question never licensed.
10. RBAC's final allow-list gate.
11. AST validation + parameterization — single read-only `SELECT`, every table/column must
   exist, every planned join key must be a real FK edge, no cartesian product, every literal
   bound as a parameter.
12. **Execute** — a read-only session, 30-second statement timeout, capped row fetch.
13. **NL-back answer** — one further SLM call turns rows into prose, with a deterministic
   row-count fallback if the SLM is unavailable.
14. **Cache-back** — a genuinely fresh, non-trivial answer is written to the verified cache
   for next time.

A rejection at any of these is a *terminal, typed status* (`ungrounded`, `qualifier_dropped`,
`ir_mismatch`, `invalid`, `no_table`, `clarify`, `refuse`, `exec_error`) — never a silently
wrong answer.

**Tier-2**: if the deterministic head refuses on a subset of those statuses, and it didn't
take too long, one LLM-IR retry is attempted — the LLM emits a UUID-only intent envelope
(never SQL text), a deterministic builder turns that into SQL, and the *same* firewall runs
again before execution.

## B.7 The other heads

- **RAG** — BGE-M3 dense + learned-sparse chunk retrieval, RBAC-filtered, one local-SLM
  synthesis call over the retrieved passages. Deliberately does not apply SQL-style value
  expansion (it would inject irrelevant tokens into a document search).
- **Hybrid** — runs the deterministic SQL head first for correct-by-construction rows, fuses
  those rows (as ground-truth text, not as re-derived SQL) with retrieved document chunks and
  any graph-expansion chunks, one further SLM call to synthesize both into one answer. The SQL
  head's own tables/charts/explain are attached, so a hybrid answer looks and behaves like a
  plain SQL one to the caller.
- **NoSQL** — fully deterministic keyword-to-native-query translation (Mongo/Elasticsearch/
  DynamoDB) — no LLM at all.

## B.8 Where chat fits

The conversational tier (`apps/chat` + the `chatbot/` LangGraph package) is a caller of this
same query pipeline, never a fork of it — it resolves follow-ups against a small
evidence-only memory (harvested from the engine's own deterministic explain output, never from
free LLM output), then calls the *exact same* inference HTTP endpoint every other caller uses.
See [`CHAT.md`](CHAT.md) for that layer's own detail; nothing about the two pipelines above
changes when the caller is chat instead of the plain query API.

---

## C. How the two pipelines actually meet

| Ingestion produces… | …and the query tier reads it here |
|---|---|
| Semantic model | Routing, anchor selection, value grounding, qualifier gate, the deterministic fast path |
| Dense + sparse embeddings | Retrieval Signals 1 & 2 |
| FK adjacency / relationship graph | Join planner, `graph_guard`'s FK-edge check, fast path |
| Unified graph | Retrieval-spine graph-expansion booster |
| `cross_source_fk` edges | The multi-source coordinator's structural `MULTI` detection, federated join planning |
| Rerank docs | The cross-encoder reranker's pair text (never assembled live) |
| Value referents / value mirror | The qualifier-completeness gate, value resolver |
| Document chunks + entity/semantic-about edges | The RAG/hybrid heads, PPR graph expansion |

If a query behaves strangely — wrong anchor, missed cross-source join, weak retrieval — the
first question is always "what did ingestion actually publish for this source," not "what did
the query engine do wrong." The query tier is, by design, only ever as good as its inputs.

---

## D. Recent, verified fixes to this system (2026-09-10)

Kept here because they materially affect "how it works" today, not just history:

1. **`storage_adapters/reader.py::ann_search` was reading the wrong database.**
   `column_embeddings_v2` lives in the engine's internal store (`veda_engine`); the reader was
   querying it on the Django-tables connection (`veda`), which always failed and silently
   degraded Signal 1 to an **unscoped** fallback (leaking candidates across sources in a
   multi-source scope). Fixed with a dedicated internal-store connection; a second, previously
   masked bug (an `int`-vs-`TEXT` type mismatch on `source_id`) was found and fixed in the same
   pass. Live-confirmed: `✓ Signal 1 via storage_adapters (engine store, source-scoped): N
   cols` with real, source-scoped cosine matches.
2. **The multi-source coordinator's authoritative mode was scoped to `MODE_MULTI` only**
   (§B.3) after an unscoped rollout regressed plain single-source questions. This is the
   current, correct, live-verified state described above — not a stopgap.
3. **Local Postgres pinned to `pg16`** (was bumped to `pg17` in compose, but the local data
   volume was never migrated) — a deployment detail, not a pipeline behavior change.
4. A stale `METAL_EMBED_URL` in `.env` was pointing at an unreachable host, causing every
   embed/rerank call to eat a 60-second timeout before falling back to CPU — fixed to the
   correct LAN address.

Full incident write-ups: `docs/backlog/query-engine-open-items.md`.
