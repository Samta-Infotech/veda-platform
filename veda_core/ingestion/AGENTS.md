# veda_core/ingestion/ — the offline build

The layered L1–L5 pipeline lives in `layers/` (see [`layers/AGENTS.md`](layers/AGENTS.md)).
The files here are the **stage functions** those layers call, plus the connectors bridge
and the non-relational dispatcher. Full reference: [../../docs/INGESTION.md](../../docs/INGESTION.md).

## Orchestration / routing
| File | Role |
|------|------|
| `dispatcher.py` | Type router called by `../main.run_ingestion`. relational / tabular → `layers.pipeline.run_layered_ingestion`; else → `source_dispatcher.dispatch_ingestion`. |
| `source_dispatcher.py` | The **non-relational path** (document / nosql / non-tabular datalake). `dispatch_ingestion()`, `_run_schema_pipeline()` (schema-only chain, no data_graph / value_sampler). For a relational source with `use_full_pipeline=True` it loops back into `main.run_ingestion`. **Not dead.** |
| `contracts.py` | `SourceContext` + `StageOutcome` dataclasses. `SourceContext.from_env()` — the one place the engine learns which source it runs for. |
| `db_abstraction.py` | DAL. `get_internal_connection` (VEDA's pgvector, `VEDA_INTERNAL_DB`) vs `get_client_connection(source_id)` (the tenant source). |

## Stage functions (called by a layer)
| File | Layer | Role |
|------|-------|------|
| `schema_scanner.py` | L1 (FATAL) | `run_schema_scanner()` — raw schema dict → flat typed `ScanResult`; sensitive-column exclusion. |
| `data_graph.py` | L1 (non-fatal) | value-overlap + co-null correlation → undeclared-FK `DiscoveredEdge`s at 3 certainty levels; HIGH/MED merged into FK-adjacency. |
| `column_sketches.py` | L2 (non-fatal) | 128-perm MinHash over join-key-shaped columns → `column_sketches`. Feeds cross-source FK discovery. No-op without `datasketch`. |
| `semantic_type_inference.py` | L2 (FATAL) | 6 semantic types via a 3-layer rule engine + primary-display-column pass. |
| `value_sampler.py` | L1/L2 (non-fatal) | samples distinct values from CATEGORY / FREE_TEXT / IDENTIFIER columns → `column_values`. `rebuild_value_index_from_db` at query warm. |
| `reg_builder.py` | L2 (FATAL) | builds the in-memory REG (Relational Entity Graph). PyG HeteroData with a pure-numpy fallback. Consumed by `graph_persist`, never persisted itself. |
| `join_paths.py` | L2 (non-fatal) | precomputes shortest FK path (≤ 4 hops) for every table pair → `veda_join_paths.json`. |
| `semantic_layer_v2.py` | L3 | **65 KB.** `run_full_semantic_layer()` / `save_semantic_model()` — the 5-stage hybrid semantic build (profiling → glossary → table → column → retrieval docs). Qwen via Ollama in stages 2–4. Writes `veda_semantic_model.json` + synonyms + concepts. |
| `deterministic_metadata.py` | L3 | rule-based semantic-layer-v2 pass (semantic_type, analytics_role, sql_usage, aliases, value_handling). Also imported by `../veda_hybrid`. |
| `data_profiler.py` | L3 (indirect) | column stats → `veda_profiling.json`. Called inside `semantic_layer_v2` stage 1. |
| `glossary_builder.py` / `domain_glossary.py` | L3 | one Qwen call → `veda_glossary.json`; the 3-layer glossary (SLM + HF BFSI + static AML/KYC) → `glossary/domain_glossary.json`. |
| `graph_persist.py` | L4 (non-fatal) | `persist_reg_graph()` — REG + discovered-FK edges → `graph_nodes` / `graph_edges` (idempotent, scoped delete). Gate `UNIFIED_GRAPH_ENABLED and GRAPH_PERSIST_ENABLED`. |
| `graph_embedder.py` | L4 (non-fatal) | `embed_graph_nodes()` — BGE → `graph_node_embeddings`. Gate `UNIFIED_GRAPH_ENABLED and GRAPH_EMBED_ENABLED`. |
| `biencoder.py` | L4 | `run_biencoder_ingestion()` — BGE-M3 dense → `column_embeddings_v2` / `table_embeddings_v2` (1024-dim, **HNSW** cosine, scoped delete-then-insert). The live retrieval store. Resume-skips if the table has rows. |
| `sparse_index.py` | L4 (non-fatal) | `build_sparse_index()` — BGE-M3 learned-sparse → `column_sparse_v1` / `table_sparse_v1`. **The WP3 replacement for the BM25 index.** |
| `enrichment_index.py` | L4 (non-fatal) | one pre-inverted index (synonyms + concept graph + glossary) → `veda_enrichment_index.json`. |
| `rerank_docs.py` | L4 (non-fatal) | `build_rerank_docs()` — cross-encoder pair text per column/table → `veda_rerank_docs.json`. Consumed by `../query/reranker.py`. |
| `relationship_graph.py` | L5 (non-fatal) | `build_relationship_graph()` — declared FK + polymorphic-by-correlation edges → `veda_relationship_graph.json`. Edge weights: business_core 1, reference 2, audit 10. See `INGESTION.md`. |
| `unified_graph_builder.py` | L5 (non-fatal, always rebuilt) | pure-stdlib fusion of sm + relationship graph + concept graph + synonyms + metrics → `veda_unified_graph.json`. |
| `value_referents.py` | L5 (non-fatal) | `write_value_referents()` — LLM-free: for every sampled value emit direct + FK-closure referents → `veda_value_referents.json`. |
| `value_mirror.py` | L5 (non-fatal) | `mirror_values_to_redis()` — `column_values` → Redis hashes `value:{tenant}:{source}:{norm}`. |
| `cross_source_graph.py` | L5 (non-fatal) | `discover_and_persist(tenant)` — compares `column_sketches` across sources → `cross_source_fk` col→col edges (Jaccard + containment). No-op until ≥ 2 sources have sketches. |

## Document / cross-source ingest
| File | Role |
|------|------|
| `chunk_embedder.py` | Embeds `DocumentChunk`s with BGE-M3 → `doc_chunks` pgvector. `retrieve_top_k_chunks()` used at RAG query time. |
| `entity_linker.py` | `link_entities()` — dictionary + pattern + optional-SLM entity detection → `entity` nodes bridging `chunk --mentions_entity--> entity --value_of--> column`. **Replaces `chunk_linker.py`.** |
| `semantic_linker.py` | Semantic bridge Tier A — chunk M3 vectors vs column `graph_node_embeddings` → `semantic_about` (chunk→column) edges. Never authorizes a join. |
| `value_embedder.py` | Semantic bridge Tier B — `embed_source_values()` → `entity_value_embeddings` (HNSW). **Only runs in `source_dispatcher`, NOT the layered relational path.** |
| `lite_semantic_model.py` | `build_lite_sm()` — deterministic structural semantic model for tabular/doc sources (no LLM). Consumed by `storage_adapters/writer.py`. |

## Support
| File | Role |
|------|------|
| `m3_encoder.py` | Process-wide `BAAI/bge-m3` singleton — dense (1024) + learned-sparse, for every embedding site + query encode. Offline-forced. Optional Metal offload via `METAL_EMBED_URL` with a CPU fallback. |
| `column_text.py` | `build_enriched_column_text()` — one column's NL embedding text. Header mentions `relgt_encoder` (gone). |
| `vector_store.py` | Metadata stores: `store_fk_adjacency` / `get_fk_adjacency` (join engine FK source of truth), `store_table_metadata` / `get_display_columns`, `retrieve_cols_by_name_keywords`. Legacy encoder-embedding + `_lt` / `_hybrid` write paths removed. |
| `schema_unifier.py` | connector dataclasses → the legacy dict `schema_scanner` wants. |

## Dead / skeleton
| File | State |
|------|-------|
| `INGESTION.md` | now a pointer doc (was badly stale). |

## Gotchas
- **No BGE fine-tune stage** any more; `synthetic_query_gen.py` / `auto_finetune.py` / the
  `client_bge/` checkpoint are gone.
- `ENCODER_MODE` removed. `EMBEDDING_MODEL_ID = "bge-m3"` is only the model-change guard key
  (stamped onto `IngestionJob.encoder_mode` by `apps/ingestion/tasks.py`).
- `artifact_scope` is plumbed but **OFF by default** (`VEDA_ARTIFACT_SCOPING != "1"`) — N
  relational sources still overwrite the same `data/veda_*.json` files.
