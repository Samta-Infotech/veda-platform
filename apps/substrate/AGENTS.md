# apps/substrate/ — ingestion outputs as Django models

`models.py` mirrors every ingestion artifact as an ORM model so the Django tier owns the
substrate. All inherit `TenantScopedModel` (UUID PK matching the engine's UUIDs).
[../../docs/ARCHITECTURE.md §6](../../docs/ARCHITECTURE.md).

| Group | Models | Notes |
|-------|--------|-------|
| Structural | `SchemaTable`, `SchemaColumn`, `FkEdge`, `TableMetadata` | `FkEdge` = the join engine's FK source of truth; undeclared FKs carry `is_declared=False`, `overlap_score`. |
| Semantic / language | `SemanticType`, `GlossaryEntry`, `Synonym`, `SyntheticPair`, `SemanticConcept` | |
| Value grounding | `ColumnValueSample`, `ColumnProfile` | value sampler (also mirrored to a Redis SET) + profiler. |
| Embeddings (`managed=False`) | `ChunkEmbedding` (`chunk_embeddings`), `GraphNodeEmbedding` (`graph_node_embeddings`) | **admin visibility only** — ANN is raw SQL in `storage_adapters.reader`. |
| Graph | `GraphNode`, `GraphEdge`, `GraphArtifact` | Kùzu removed. |
| Verified cache | `VerifiedQueryCache` | unique `(source, tenant, query_hash)`; `query_embedding vector(1024)` added by RunSQL. |
| Normalized `sm` | `SubstrateVersion` (+ `hnsw_ef_search`), `SmTable`, `SmColumn`, `SmRetrievalDoc`, `SmSynonym`, `SmConcept` | the read-model `storage_adapters.assembler` rebuilds; `SubstrateVersion` drives rehydrate. |

`admin.py` registers ~15 models via the `all_tenants()` escape hatch.

## Migrations of note
- `0002_pgvector` — creates 7 embed tables + HNSW indexes + the VQC embedding column.
- `0003` — the `Sm*` models. `0004` — `hnsw_ef_search`.
- **`0006` / `0007` / `0008`** — **drop** `column_embeddings`, `_lt`, `_hybrid`, `_bge`,
  `relgt_structural` (single BGE-M3 encoder).

## Gotchas
- The **live ANN store `column_embeddings_v2` / `table_embeddings_v2` is engine-owned in
  the `veda_engine` database** — created by ingestion, **not by any Django migration**. The
  `managed=False` models here do NOT point at it.
- Physical pgvector tables after all migrations: `chunk_embeddings`, `graph_node_embeddings`,
  `substrate_verifiedquerycache.query_embedding`. That's it on the `veda` side.
