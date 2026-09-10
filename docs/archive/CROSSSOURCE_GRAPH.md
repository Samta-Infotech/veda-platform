# VEDA Cross-Source Knowledge Graph — Document Integration & Federated Querying Plan

> **Objective:** documents (PDF, Word, CSV/Excel) become first-class, fully graphed
> citizens alongside relational sources, and the graph carries explicit
> cross-source connections (shared entities, discovered join keys) so a single
> query can traverse from any source to any other and pull data from all of them.
>
> **Prerequisite:** the Retrieval Upgrade Plan (M3 unification, PPR, weighted
> fusion, de-flagged precompute) is merged and verified. Everything below assumes
> one 1024-dim M3 embedding space and PPR traversal.
>
> **Style contract (same as before):** final-state code, no feature flags,
> deletions land with their replacements, no model training. Phases have hard
> acceptance gates because P4/P5 change query semantics — do not start a phase
> before the previous gate passes.

---

## 0. Design in one page

**The core insight: a data source's *kind* is a property of its content, not its
file extension.** A CSV is a table. A table inside a PDF is a table. Prose in a
PDF is text. Today VEDA flattens all documents to chunks, which throws away the
joinable half of their content. The plan splits document ingestion into a
**tabular lane** (→ real tables: columns, semantic types, value samples,
embeddings — identical treatment to a Postgres table) and a **narrative lane**
(→ doc/section/chunk nodes). Once document-borne tables are real tables, the
machinery that already connects tables (value-overlap FK discovery, FK signals,
join planner) extends naturally across sources.

**Three graph additions make cross-source traversal work:**

1. **Entity nodes** — normalized values ("ACME-CORP", "INV-2024-0113",
   "priya@x.com") that appear in *both* a column's sampled values and a chunk's
   text become `entity` nodes bridging the narrative and tabular worlds.
2. **`cross_source_fk` edges** — tenant-level value-overlap discovery (MinHash
   sketches computed per column at ingest) connects join-compatible columns
   *across* sources, with confidence tiers like the existing intra-source
   `discovered_fk`.
3. **Doc structure nodes** — `doc → section → chunk` hierarchy plus doc metadata,
   so traversal can move chunk → its document → sibling chunks → linked entities,
   instead of chunks being disconnected leaves.

**Execution follows the graph:** a DuckDB-based federated layer attaches every
tabular surface (Postgres via `postgres_scanner`, CSV/Excel/doc-extracted tables
via Parquet) under per-source catalogs, so one validated SQL statement can join
across sources. Narrative evidence (chunks) is composed into the answer alongside
the SQL result with provenance.

```
                    ┌────────────── tenant-wide graph ──────────────┐
  relational src A  │ tbl/col ──fk_to── tbl/col                     │
                    │    │                 │                        │
                    │ value_of        cross_source_fk (MinHash)     │
                    │    │                 │                        │
  docs src B        │ entity ──mentions── chunk ── in_section ── doc│
                    │    │                                          │
  csv src C         │ value_of ── tbl/col (CSV = real table)        │
                    └────────────────────────────────────────────────┘
  Query → PPR over the whole tenant graph → SQL subgraph (federated DuckDB)
        + evidence chunks (RAG) → composed, cited answer
```

---

## Phase 1 — Finish tenant-scoped query serving (the P5 debt)

Cross-source traversal is impossible while the warm engine serves one source.
This phase completes the multi-source query tier.

1. **Query scope becomes a source *set*.** `RequestContext` gains
   `source_ids: list` (default: all `ready` sources of the tenant; the API accepts
   an optional subset in the request body, validated against tenant ownership in
   the Django view — never trusted from headers alone).
2. **Per-scope engine instances.** `veda/runtime.get_engine()` stops being a
   process singleton: keyed registry `{(tenant, frozenset(source_ids)): engine}`
   with an LRU cap (`ENGINE_CACHE_MAX = 4` per worker; measure RSS — the models
   are shared singletons, only the per-source indexes differ, so an engine entry
   is index state, not model weights). Warm loads (sparse index, signal maps,
   enrichment index, BM25→sparse successor) key by the same scope. Rehydrate
   invalidates matching entries.
3. **ANN and store reads accept source sets.** `storage_adapters.reader.ann_search`
   and the pgvector queries change `source_id = %s` to `source_id = ANY(%s)`;
   same for `graph_nodes`/`graph_edges`/`doc_chunks`/`column_sparse_v1` readers.
4. **Semantic model is composed per scope**: the assembler publishes per-source
   sm's (as today); the loader merges the scoped set into one namespace with
   source-qualified table keys (`src{ID}.schema.table`). Collision rule: table
   display names get a source suffix only when ambiguous.

**Gate:** two relational sources ready in one tenant; queries answer correctly
against each and against both; per-source artifacts show zero cross-talk;
worker RSS within budget at `ENGINE_CACHE_MAX`.

---

## Phase 2 — Tabular lane: CSV/Excel and document-embedded tables become real tables

1. **CSV/Excel routing.** In the dispatcher, `csv_lake`/`parquet` sources (and new
   `xlsx` dialect) stop producing chunks entirely. New
   `connectors/tabular_files.py` implements the *relational* connector interface
   over DuckDB: `read_csv_auto`/`read_parquet`/spatial `st_read` for xlsx →
   `get_schema()` returns `RawTable`/`RawColumn` with DuckDB-inferred types
   (deterministic UUIDv5 ids from `source_path + sheet/file name`, same as
   relational). Then the **standard L1–L5 pipeline runs unchanged**: semantic
   types, value sampling, M3 embeddings, sparse index, graph persist. A CSV column
   is now indistinguishable from a Postgres column to every downstream consumer.
2. **Materialization for execution.** L1 additionally writes each tabular file to
   canonical Parquet under `ARTIFACT_ROOT/<tenant>/<source>/tables/<table>.parquet`
   (typed, snappy). This is the execution surface Phase 5 attaches — the original
   CSV is never re-parsed at query time.
3. **Tables inside PDFs/Word ("derived tables").** In the document pipeline, table
   regions detected by the parser (Phase 3) are extracted as DataFrames. Any table
   with ≥ `DOC_TABLE_MIN_ROWS = 5` rows and a coherent header becomes a **derived
   table**: written to the same Parquet store, registered as
   `RawTable(name=f"{doc_stem}__t{n}")` on the *document source itself*, and run
   through the same L2–L4 stages (types, values, embeddings, graph persist). Graph
   edge `derived_from` (table → doc node) preserves lineage. Small/ragged tables
   stay as chunk text.
4. **Value store + MinHash at ingest (feeds Phase 4).** Extend
   `ingestion/value_sampler.py`: for every IDENTIFIER/CATEGORY/text column
   (any source kind), compute a **128-perm MinHash sketch** over normalized
   distinct values (casefold, strip, NFC; numeric/date canonicalization) and
   persist to `column_sketches(col_id, source_id, tenant, n_distinct,
   value_class, sketch bytea)`. `datasketch` added to requirements. Cost is one
   pass over values already being sampled.

**Gate:** ingest a CSV source → it appears as a table with typed columns, sampled
values, embeddings, graph nodes; a golden query against CSV-only data answers via
the normal SQL path (executed through the Phase 5 stub or DuckDB directly in
tests); a PDF with a clean table yields a derived table with `derived_from`
lineage.

---

## Phase 3 — Narrative lane: structure-aware document graph

1. **Parsing upgrade.** Replace the flat text extraction in
   `connectors/document.py` with layout-aware parsing:
   - PDF → `pymupdf4llm` (markdown with heading hierarchy + table detection;
     pure-local, no egress).
   - DOCX → `python-docx` walking heading styles and `Table` objects.
   - Both emit a common `ParsedDoc{metadata, sections[{path, level, text,
     tables[]}]}`. Tables route to Phase 2.3; text routes below.
2. **Structure-aware chunking** replaces fixed 512/64: chunk *within* sections,
   never across section boundaries; prepend the heading path to each chunk's
   embedded text (`"Contracts > Termination > Notice periods:\n<text>"`) — this
   measurably improves chunk retrieval precision and costs nothing.
3. **Graph structure nodes.** `graph_persist` gains node types `doc`
   (`doc:<doc_id>`, attrs: name, date, author, mime, path) and `section`
   (`sec:<doc_id>:<path-hash>`), and edge types:

   | edge | endpoints | weight |
   |---|---|---|
   | `has_section` | doc → section | 1.0 |
   | `in_section` | chunk → section | 1.0 |
   | `derived_from` | derived table → doc | 1.5 |
   | `next_chunk` | chunk → chunk (adjacent) | 0.4 |

   `doc` and `section` nodes get M3 embeddings (title + summary line) in
   `graph_node_embeddings` so they can be PPR seeds ("the Q3 vendor contract").
4. **Chunk metadata** (`doc_chunks`): add `section_path`, `doc_author`, `mime`;
   temporal filtering extends to section-level dates when present.

**Gate:** ingest a structured PDF; graph shows doc→section→chunk hierarchy;
seed-query for a document title reaches its chunks via PPR through structure
edges; chunk retrieval precision on the doc-query golden subset ≥ previous
flat-chunking baseline.

---

## Phase 4 — The bridges: entity layer + cross-source join discovery

This is the phase that makes the graph "know how to connect sources."

### 4.1 Entity extraction & linking (no training)

New `ingestion/entity_linker.py`, replacing and subsuming `chunk_linker`
(delete `chunk_linker.py` when this lands; its `mentions`/`about` edges are
regenerated by the new module with the same edge types).

Extraction per chunk, three detectors, all deterministic:
1. **Dictionary detector (primary).** Match chunk text n-grams against the
   tenant's **value store** (the Redis value mirror + `column_values` — this
   asset already exists and is exactly an entity dictionary). Normalization
   identical to the sketch pipeline. Only values from IDENTIFIER/CATEGORY/name-like
   columns participate; values shorter than 4 chars or in a stopword/common-token
   list are excluded to control noise.
2. **Pattern detector.** Typed regexes: email, phone, money, ISO dates, and
   tenant-configurable ID patterns (`Source.id_patterns` JSONField, e.g.
   `INV-\d{4}-\d{4}`). Typed entities match columns whose sampled values share the
   pattern class even when the exact value wasn't sampled.
3. **SLM detector (L3, docs only).** The existing Qwen enrichment pass extracts
   salient proper nouns per section (one call per section, bounded); SLM-only
   entities are created **only if** they also dictionary- or pattern-match — the
   SLM widens recall of detectors 1–2 (catching inflections/partial mentions), it
   never mints unlinked entities. This keeps the entity set grounded and bounded.

Graph materialization:
- Node `ent:<class>:<value_norm>` (class ∈ id, email, name, money, date, term),
  attrs: display value, class, mention/column counts. **Admission rule:** an
  entity node is created only when it links ≥ 1 chunk AND ≥ 1 column, or ≥ 2
  columns in different sources — pure single-sided values stay as plain value
  signals. This is the explosion control.
- Edges: `mentions_entity` (chunk → entity, weight 1.2; count in attrs),
  `value_of` (entity → column, weight 1.5, per column whose sample set contains
  it), and the existing `about` retained for strong whole-chunk topicality.
- **PII guard:** columns matching `SENSITIVE_PATTERNS` never emit entities; email
  entities store salted hashes as node ids with masked display values.

### 4.2 Cross-source join discovery (tenant-level L5+ stage)

New `ingestion/cross_source_graph.py`, run at the end of every ingestion (it's
cheap — sketch comparisons only) over ALL ready sources of the tenant:
1. Load all `column_sketches` for the tenant; candidate pairs = columns from
   *different* sources with compatible `value_class` and comparable cardinality
   (ratio within [0.01, 100]).
2. MinHash Jaccard estimate per pair; containment estimate for asymmetric FK-like
   relations (small set ⊂ large set).
3. Emit `cross_source_fk` edges (col → col, both directions recorded once) with
   tiers mirroring `discovered_fk`: HIGH (containment ≥ 0.8 ∧ n_distinct ≥ 25,
   weight 2.0), MEDIUM (≥ 0.5, weight 1.2). Evidence attrs: jaccard, containment,
   cardinalities — the join planner and the answer composer surface these so a
   cross-source join is always explainable.
4. Same edges feed the **join planner**: `join_paths` precompute (Q-9) runs
   tenant-wide including `cross_source_fk` edges; `fk_adjacency` reader unions
   them so `_inject_bridge_columns` and `_resolve_join_path` work across sources
   with zero code changes to their logic.
5. Idempotency: scoped delete of the tenant's `cross_source_fk` edges before
   re-emit; re-ingesting any one source re-runs the pass.

### 4.3 Tenant-wide PPR

The Phase-1 scope change plus these edges make traversal cross-source by
construction: the PPR transition matrix loads `graph_edges WHERE source_id =
ANY(scope) OR edge_type = 'cross_source_fk'`. New edge-type weights registered in
`GRAPH_EDGE_WEIGHTS`. No other traversal changes — this is why PPR replaced BFS
first.

**Gate (the demo that matters):** tenant with a Postgres CRM source, an invoices
CSV, and a contracts-PDF source. Query: *"what did we invoice ACME last quarter
and what does their contract say about late fees?"* — PPR seeds on "ACME" hit the
`ent:name:acme` node, traversal reaches `customers` (Postgres) via `value_of`,
`invoices.parquet` columns via `cross_source_fk` on customer id/email, and the
contract chunks via `mentions_entity`; the selected subgraph spans all three
sources. (Execution of the SQL half lands in Phase 5; this gate validates
*selection*.)

---

## Phase 5 — Federated execution + composed answers

### 5.1 DuckDB federation layer (L7-fed)

New `query/federated_executor.py`, replacing the single-connection execution in
L7 for any plan touching > 1 source (single-source plans keep the existing direct
path — that's a routing decision, not a flag):
1. Per (tenant, scope), build a DuckDB connection (in-memory, read-only mode):
   - Relational sources → `ATTACH 'postgres:...' AS src_<id> (READ_ONLY)` via
     `postgres_scanner`, credentials resolved server-side from `Source` secret
     refs — never present in generated SQL. Pushdown means Postgres still does the
     heavy filtering.
   - Tabular-file and derived tables → `CREATE VIEW src_<id>.<table> AS SELECT *
     FROM read_parquet('<artifact path>')`.
2. **Naming contract:** generated SQL uses `src_<id>.<schema>_<table>` everywhere;
   the sm (Phase 1.4) already exposes source-qualified names, so the SQL
   builder/SLM prompt changes are mechanical (alias map gains the catalog prefix).
3. **Firewall extension** (this is mandatory, same gates, wider scope): the AST
   validator learns the catalog naming, verifies every referenced catalog is in
   the request scope, keeps SELECT-only + parameterization + row limits, and adds
   a per-query source-count cap (`FED_MAX_SOURCES = 4`) and statement timeout.
   The graph-guard validates cross-source joins against `cross_source_fk`
   /declared edges only — no ungrounded cross-source joins, ever.
4. Timeouts/backpressure: DuckDB `SET statement_timeout`; per-attach connection
   TTL and pool; memory cap (`SET memory_limit`).

### 5.2 Hybrid answer composition

Extend the existing hybrid path (`HYBRID_SQL_WEIGHT`/`HYBRID_RAG_WEIGHT`
machinery) into a **composer** that consumes the Phase-4 selected subgraph:
1. Partition selected nodes: tabular subgraph (columns/tables + join path,
   possibly multi-source) → federated SQL; chunk nodes → evidence set.
2. Execute SQL; retrieve/rerank evidence chunks (they were already scored by PPR;
   cross-encoder reranks top evidence against the query).
3. NL answer: deterministic templates still handle canonical shapes; the SLM
   composition prompt receives SQL results + top evidence chunks *with
   provenance tags* (`[src:invoices.csv]`, `[doc:MSA_ACME.pdf §7.2]`) and is
   instructed to cite them; the response payload carries a structured
   `provenance` array (source, table/doc, section, join edges used with their
   confidence tiers) so the UI can render "how this answer was assembled."
4. Router: `query_router` gains a `federated` outcome when the selected subgraph
   spans sources; refusal path explains when a cross-source join was blocked by
   the graph-guard (low-confidence edge) rather than silently dropping a source.

**Gate:** the Phase-4 demo query end-to-end — one composed answer containing the
invoice aggregate (Postgres × CSV join over a HIGH `cross_source_fk` edge) and
the late-fee clause (PDF chunk), each cited; firewall blocks a forged
`src_<other-tenant>` reference and an ungrounded cross-source join in tests.

---

## Phase 6 — Verification & golden set extension

1. Extend `evaluation/golden_queries.jsonl` with a **cross-source class** (≥ 15
   queries): entity-bridge questions, CSV×DB joins, doc-evidence-plus-figures,
   negative cases (entities that must NOT link, join pairs below threshold).
2. `scripts/retrieval_eval.py` gains: entity-linking precision/recall on a
   hand-labeled 100-mention sample, cross-source join precision (emitted
   `cross_source_fk` HIGH edges manually audited — target ≥ 0.9 precision; recall
   is secondary since MEDIUM edges exist), subgraph source-coverage on the
   cross-source class, and end-to-end execution accuracy through the federated
   path.
3. Fresh ingestion of all three source kinds; run the full protocol from the
   Retrieval Upgrade Plan WP9 plus the above.

---

## Risks & controls (read before starting)

- **Entity explosion / noise** → admission rule (must bridge), length/stopword
  filters, per-tenant entity cap with count-ranked eviction, and the SLM detector
  gated behind dictionary/pattern corroboration.
- **False cross-source joins** → HIGH tier thresholds are conservative;
  graph-guard only permits HIGH edges for execution (MEDIUM edges guide retrieval
  but not SQL joins); every federated answer exposes the join evidence.
- **PII leakage via entities** → SENSITIVE_PATTERNS exclusion happens at
  extraction, before any node exists; hashed email ids; entity nodes are
  tenant-scoped like everything else.
- **DuckDB attach security** → read-only attaches, server-side credential
  resolution, catalog whitelist per request scope in the firewall, no DDL/COPY
  allowed by the AST gate.
- **Graph size / PPR memory** → tenant matrix is sparse CSR; entities and
  structure nodes roughly double node count — measure at Phase 3/4 gates; if a
  tenant graph exceeds ~1M edges, shard the transition matrix by connected
  component (natural for multi-source tenants).
- **Parquet staleness for live-ish CSV drops** → `Source.last_ingested_at` is the
  freshness contract; re-ingest is the refresh (document this for the tenant);
  no query-time re-parsing.

## Sequencing

```
Phase 1 (tenant-scoped serving)      ← unblocks everything
Phase 2 (tabular lane)  ┐ can run in parallel
Phase 3 (narrative lane)┘ after Phase 1
Phase 4 (entities + cross-source edges)  ← needs 2+3 artifacts
Phase 5 (federated execution + composer) ← needs 4's edges + 2's parquet
Phase 6 (verification)
```

Each phase is independently shippable and independently valuable: after Phase 2
alone, CSVs are queryable tables; after Phase 4 alone, retrieval already
surfaces cross-source context even before federated SQL exists (the composer's
evidence half). Phase 5 is the largest and touches the firewall — budget it as
its own milestone with the strictest review.