# storage_adapters/ — the substrate I/O seam

Query-time reads are Django-free (raw psycopg2 + Redis); ingestion-time writes go through
the Django ORM. [../docs/ARCHITECTURE.md §2](../docs/ARCHITECTURE.md).

## `reader.py` — query-time reads (Django-free; scope from `context.current()`, fail-closed)
| Function | Does |
|----------|------|
| `_connection()` | cached psycopg2 conn to `POSTGRES_*` (default db `veda`), autocommit, **never** session READ ONLY (PgBouncer poison). |
| `source_connection()` | reads the `sources_source` row for the primary source_id, resolves the password. The query-tier counterpart to `Source.as_engine_env`. |
| `get_fk_adjacency()` | FK edges for the source SET; in-memory `_all_fk_edges` process cache (cleared on rehydrate). Reads Django `substrate_*` tables. |
| `glossary()` / `synonyms()` / `value_samples()` | `substrate_glossaryentry` / `_synonym` / `_columnvaluesample`. |
| `ann_search(mode, qvec, top_k)` | raw pgvector cosine over **`column_embeddings_v2`**; `SET LOCAL hnsw.ef_search` inside an explicit `BEGIN…COMMIT`. |
| `_resolve_ef_search(source_id)` | `VEDA_HNSW_EF_SEARCH_<id>` env → `SubstrateVersion.hnsw_ef_search` → `VEDA_HNSW_EF_SEARCH` → 40. |
| `verified_cache_lookup()` / `verified_cache_exact()` | pgvector cosine ≥ threshold / Q-8 hash short-circuit over `substrate_verifiedquerycache`. |
| `save_verified_query()` | **the one documented inference-tier WRITE** — `INSERT … ON CONFLICT (source,tenant,query_hash) DO NOTHING` + publishes `veda:rehydrate:{s}:{t}:verified_cache`. Runs synchronously from `veda/pipeline.py`. |

## `writer.py` — ingestion-time writes (Django ORM; `all_tenants().filter(...)` explicitly)
| Function | Status |
|----------|--------|
| `store_fk_adjacency` / `store_glossary` / `store_semantic_model` | wired |
| `sync_from_engine(internal_dsn)` | wired — clears + repopulates `FkEdge` / `SchemaColumn` / `SchemaTable` / `ColumnValueSample` / `Synonym` / `GraphNode` / `GraphEdge` / `GraphArtifact` from `veda_engine` (`VEDA_INTERNAL_*`). |
| `_build_lite_sm_from_graph` | wired — deterministic per-source sm for tabular/doc sources. |
| `warm()` | wired — §7 stage 10: persist sm, `sync_from_engine()`, `_persist_hnsw_tuning`, `publish_sm` + `publish_registry` / `publish_empty_registry` + `publish_rehydrate`. |
| `store_column_embeddings(mode, rows)` | **`raise NotImplementedError`** — target table `column_embeddings_bge` was dropped in migration 0008. **Dead.** |

## `assembler.py`
`SemanticModelAssembler.assemble(source, tenant)` (Sm* rows → `sm` dict), `persist()` (the
inverse), `publish_sm()` (`veda:sm:{s}:{t}` in redis-cache), `publish_registry()` /
`publish_empty_registry()` (fast-path registries scope-keyed; `publish_empty_registry` is
the anti-cross-source-leak fix for non-relational sources), `publish_rehydrate()`.

## Gotchas
- **Two connections, two databases.** `_connection()` → `veda` (Django `substrate_*` +
  `sources_source`). `_internal_connection()` → `veda_engine` (the engine's pgvector tables).
  `ann_search` is the only reader that needs the second one — it queries
  `column_embeddings_v2`, which ingestion writes to `veda_engine`. (Before 2026-09-10 it ran
  on the `veda` connection and silently errored into an unscoped fallback — see
  [../docs/backlog/query-engine-open-items.md](../docs/backlog/query-engine-open-items.md).
  Worth a live confirmation: a query should log `Signal 1 via storage_adapters (engine
  store, source-scoped): N cols`, not `Signal 1 adapter unavailable`.)
- `save_verified_query` is **synchronous** on the latency path (the archived plan wanted
  fire-and-forget) — bounded (one INSERT + one publish) but not free.
- The `sm` load: `veda_hybrid._load_semantic_model` / `veda/runtime._load_one_sm` are
  **Redis-first** (`VEDA_SM_REDIS`), on-disk `SEMANTIC_MODEL_FILE` fallback; the inference
  rehydrate subscriber clears the in-process `_SM` cache on any `veda:rehydrate:*` message.
