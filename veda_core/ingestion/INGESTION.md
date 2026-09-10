# veda_core/ingestion/ — the offline build

> **This file used to describe a monolithic `veda_ingestion.py` / `--ingestion-only` /
> `--embed-only` pipeline with `ivfflat` `column_embeddings`. That pipeline was deleted
> (`docs/archive/CLEANUP_PLAN.md` Phase 7).** The current build is the **layered L1–L5
> pipeline**. This file is now a pointer.

## Where the real documentation is

| For | Read |
|-----|------|
| The full end-to-end reference (real flow, per-layer stage tables, the ~30-artifact table, connectors matrix, resume logic) | [`/docs/INGESTION.md`](../../docs/INGESTION.md) |
| The per-layer contracts (consume / produce / fatal semantics / consumers) | [`layers/contracts/README.md`](layers/contracts/README.md) + `L1_EXTRACT.md` … `L5_PUBLISH.md` |
| What each file in this directory does | [`AGENTS.md`](AGENTS.md) |
| The operator runbook (add docs / datalake files, re-ingest) | [`/INGESTION_GUIDE.md`](../../INGESTION_GUIDE.md) |
| How the stores are read at query time | [`/docs/RETRIEVAL.md`](../../docs/RETRIEVAL.md), [`/docs/DATA_SOURCES_EMBEDDINGS_AND_GRAPHS.md`](../../docs/DATA_SOURCES_EMBEDDINGS_AND_GRAPHS.md) |

## The shape, in one paragraph

`apps/ingestion/tasks.py::task_ingest_source` runs the engine in a **subprocess**
(`cwd=veda_core`). `main.py::run_ingestion` is now a **thin shim** →
`dispatcher.dispatch` → for a relational or tabular-file source,
`layers/pipeline.py::run_layered_ingestion` composes five layers, threading one in-memory
`state` dict with per-stage fatal semantics:

```
L1 EXTRACT   touch the source        schema_scan* · fk_adjacency* · data_graph · (parquet materialize)
L2 ANALYZE   pure transforms         semantic_types* · table_metadata* · value_sampling · column_sketches · reg_graph* · join_paths
L3 ENRICH    the only LLM layer      semantic_layer_v2 (Qwen)  [skipped by skip_llm / resume / flag]
L4 INDEX     embeddings + search     graph_persist · graph_embed · biencoder (column_embeddings_v2) · sparse_index · enrichment_index · rerank_docs
L5 PUBLISH   derived registries      relationship_graph · semantic_registry · value_referents · hnsw_tune · value_mirror · unified_graph · cross_source_fk
                                     (* = FATAL — abort the run on failure)
```

Non-relational sources (document / NoSQL / non-tabular datalake) route to
`source_dispatcher.dispatch_ingestion` instead.

The single encoder is `BAAI/bge-m3` (`m3_encoder.py`) — dense + learned-sparse. There is
**no BGE fine-tune stage** any more, and `ENCODER_MODE` was removed; `EMBEDDING_MODEL_ID`
("bge-m3") is only the model-change guard key.

---

## Relationship graph (`ingestion/relationship_graph.py`) — still accurate

The L5 `relationship_graph` stage builds `data/veda_relationship_graph.json` — the
deterministic join foundation.

**Edge sources:**
1. **Declared FKs** — from the scanned schema.
2. **Polymorphic edges** — `object_id` + (`object_type` / `model_name`) pairs resolved by
   **data correlation**, not string matching: sample `object_id` values per discriminator
   value, correlate against candidate key columns. Accept an edge **only** if the join key
   is a string/business key OR the discriminator value has name-affinity with the target
   table — this defeats the numeric-collision trap (small surrogate IDs that coincidentally
   overlap). Example: `annotation_record.object_id → counterparty_details.counterparty_id`
   joins on the business key, not the numeric PK.

**Edge attributes:** `relationship_type`, `weight`, `cardinality` (1:1 / N:1 / N:M from
distinctness), `polymorphic`, `requires_predicate`, `discovery`, `confidence`.

**Edge weighting (load-bearing for join quality):**

| relationship_type | weight |
|---|---|
| `business_core`, `bridge` | 1 |
| `reference`, `lookup`, `polymorphic` | 2 |
| `audit`, `history` | **10** |

`audit` is assigned by table-name (`_AUDIT_TABLE_RE`: `_history` / `_log` / `_audit` /
`_archive`) **and** by column-name (`_AUDIT_COL_RE`: `*_by_id`, `owned_by*`,
`assigned_to*` → `user`). The high weight stops the join planner routing business joins
*through* the `user` hub ("edited by the same person" is not a relationship).

The rest of the graph story (the unified graph, PPR expansion, the semantic bridge) is in
[`/docs/RETRIEVAL.md`](../../docs/RETRIEVAL.md).
