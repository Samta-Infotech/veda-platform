# VEDA Ingestion Cleanup & API-Driven Per-Source Ingestion — Plan

> **Status:** analysis + plan only. **No code has been changed.**
> **Author basis:** direct read of `veda_core/main.py`, `veda_core/ingestion/*`, `veda_core/query/*`, `veda_core/retrieval/*`, `veda_core/veda/*`, `veda_core/config.py`, `apps/ingestion/tasks.py`, `apps/sources/models.py`, `apps/query/views.py`, `storage_adapters/*`, plus `ARCHITECTURE.md`.
> **Definition of "unnecessary" (per the request):** *anything produced during ingestion that is never read at query time.* Everything below is classified against the actual query-time read set.

---

## 1. Executive summary

The ingestion pipeline runs **15 physical stages** (`main.run_ingestion`, UI counter says 11). We keep **both query-engine tiers** — Tier 1 (deterministic head) and Tier 2 (LLM-IR fallback) — so anything either tier reads stays. The cleanup targets what **neither tier nor any live path reads**:

| Waste | Stages | Cost | Why it's removable |
|-------|--------|------|--------------------|
| **Fine-tune chain** | Step 10 Synthetic Query Gen + Step 11 BGE Fine-Tune | **Highest** (LLM generation + model training) | The fine-tuned model is written to `ingestion/client_bge`, but **both tiers** load the **base** `BAAI/bge-large-en-v1.5` (`config.py:746`, `BIENCODER_MODEL` never repointed). `client_bge` and `training_pairs.jsonl` are never loaded by any path. |
| **GNN** | Step 7d | Low (disabled) | `GRAPH_GNN_ENABLED=False` **and** `run_gnn_graph_embedding` doesn't exist — import always fails (caught). Fully dead. |
| **Profiler / dead files** | `column_profile`/`data_profiler`; 4 zero-importer `query/*` files; legacy CLI | Low | `profiling=None` at runtime; no query-time reader; the zero-importer files have no caller anywhere; `--legacy-query` is a separate debug entrypoint, not a tier. |

> **NOTE — reversed from an earlier draft:** the **ensemble encoder** (Step 8/9, TF-IDF/SVD, `_lt`/`_hybrid` tables, `reg_graph.pkl`) and the **graph persist/embed** stages (7b/7c) are **KEPT** — Tier 2's `retrieval_select` (Step-4 ensemble signal, Step-3 graph signal) and Tier-1's `reranker` read them. They *could* be trimmed later because Tier 2 degrades to BGE-only gracefully, but per the owner directive (§1a.1) they stay for now.

Separately, the **API-driven per-source/per-tenant ingestion** the request asks for is **partially wired but hardcoded to the primary relational source**: `task_ingest_source` injects the right DB connection per source but always runs `main.run_ingestion` (primary-only) and ignores `source.type`, bypassing the type-aware `dispatch_ingestion` router. See §5.

**Headline outcome if executed:** keep both query tiers intact; drop the 2 most expensive LLM/training stages (fine-tune chain) + GNN + profiler + dead/legacy-CLI files; wire in the glossary + unified-graph builders; and make ingestion a true per-tenant, per-source API operation routed by source type.

---

## 1a. Locked-in decisions (owner directives)

> ⚠️ **Naming caution:** "**Tier 1 / Tier 2**" below are the *query-engine answering tiers* (owner's terms). The cleanup-risk buckets in §4 are labelled "**Tier A / B / C / D**" — a different axis. Don't conflate them.

These directives override the "gate vs. remove" hedging elsewhere in this doc:

1. **KEEP Tier 1 and Tier 2 — do NOT put either (or their dependencies) in the cleanup.** *(Reverses the earlier "remove all fallbacks" decision.)*
   - **Tier 1 = the deterministic SQL head** (`veda/pipeline.run_query`: fast path → verified cache → deterministic retrieve+firewall). The primary path. **KEEP.**
   - **Tier 2 = the LLM-IR fallback** (`veda_hybrid._tier2_sql`, the code's "Tier-2"): fires when Tier 1 refuses; retrieves via `query/retrieval_select.select_retrieval`, then ENVELOPE → shared-planner → LLM-IR, all through the same firewall. **KEEP.**
   - Therefore **KEEP everything Tier 2 transitively uses** and remove it from the cleanup lists: `query/retrieval_select.py`, `retrieval_v2.py`, `schema_linker.py`, `graph_retriever.py`, `envelope_slm.py`, `intent_envelope.py`, `slm_layer.run_slm_layer` + `slm_langgraph.py` + `lg_nodes.py` + `lg_prompts.py`, and `query/semantic_layer.py`. Also KEEP the ingestion artifacts these read (see §4 Tier B/C): the ensemble encoder + TF-IDF/SVD + `_lt`/`_hybrid` tables + `reg_graph.pkl`/`col_id_to_idx.pkl` (Step-4 signal), and graph persist/embed + `graph_nodes`/`graph_edges`/`graph_node_embeddings` (Step-3 signal), plus `table_metadata`/`table_embeddings_v2`.
   - **Still removable (NOT part of either tier):** the standalone **legacy CLI** debug path — `_run_single_query_legacy` + the `--legacy-query` flag + `query/execution_engine.py` — is a separate entrypoint, *not* Tier 1 or Tier 2, so it can go (its shared deps like `semantic_layer`/`temporal_parser` stay because Tier 2 uses them). And the zero-importer dead files (§4b A0) are unused by any tier.
   - 📝 **Good-to-know (no action, since we're keeping the tiers):** Tier 2's `retrieval_select` Steps 1–4 are each `try/except` and its Step-4 ensemble `semantic_layer` signal is explicitly *"never fatal — degrade to BGE+BM25+FK+value"* (`retrieval_select.py:116-121`); on the platform (BGE primary, MiniLM not eagerly loaded) that ensemble signal already fails-and-skips. So Tier 2 effectively runs BGE-only regardless — meaning the ensemble/graph ingestion artifacts *could* be dropped later without breaking Tier 2, if you ever want to. We are **not** doing that now per this directive; it's flagged as an optional future trim in §4 Tier B/C.

2. **Exception — domain glossary MUST be produced and used at query time.** Query-time enrichment already reads `data/veda_glossary.json` (`query_enrichment.py:88`), **but `run_ingestion` does not build it** — Step 9b calls `run_full_semantic_layer(..., glossary=None, force_glossary=False)` (`main.py:652`), so the glossary is only ever produced by the `--build-glossary` CLI (`main.py:1587`) or the non-primary `source_dispatcher` Step 4b. **Required change:** wire `ingestion/domain_glossary.build_glossary` into `run_ingestion` (build/refresh before or inside Step 9b) so `veda_glossary.json` is regenerated every ingest and is fresh at query time. Do **not** remove glossary as part of the fallback cleanup.

3. **Exception — every graph used at query time MUST be stored/produced by ingestion.** Keep and guarantee production of:
   - **`data/veda_relationship_graph.json`** — CORE hot (join planner, `join_planner.py:30`). Already produced by Step 12. **KEEP.**
   - **`data/veda_unified_graph.json`** — hot (graph expansion, `query_graph.py:39` ← `veda/pipeline.py:189`). **Currently orphaned** — built only by the standalone `ingestion/unified_graph_builder.py`, NOT by `run_ingestion`. **Required change:** wire `unified_graph_builder.build_unified_graph` into `run_ingestion` (after Step 9b, since it consumes the synonyms + concept-graph files at `unified_graph_builder.py:53-54`) so the file is regenerated every ingest.
   - **`graph_nodes`, `graph_edges`, `graph_node_embeddings` (Steps 7b/7c)** — read by Tier 2's `graph_retriever` (Step-3 signal) **and** by `query/reranker` on the Tier-1 hot path (`SELECT name FROM graph_nodes`, `reranker.py:72`). Since **both tiers stay, these are KEEP** — Steps 7b/7c remain in ingestion. (They were briefly marked REMOVE under the reverted "remove all fallbacks" decision.)

---

## 2. The query-time read set (ground truth — what ingestion MUST keep producing)

Config defaults that shape the hot path: `QUERY_DECOMPOSE_ENABLED=False`, `QUERY_ROUTER_ENABLED=True`, `RETRIEVAL_V2_ENABLED=True`, `FAST_PATH_ENABLED=True`, `TIER2_LLM_FALLBACK=True`.

### 2A. CORE — read on the deterministic SQL path of *every* query
| Artifact | Location | Read at (file:line) | Produced by |
|----------|----------|---------------------|-------------|
| Semantic model | `data/veda_semantic_model.json` | `veda_hybrid.py:75`, `retrieval_engine_phase3.py:134`, `fast_path.py:176` | Step 9b (Qwen) |
| BGE column embeddings | `column_embeddings_v2` (pgvector) | `retrieval/semantic_search.py:144` (Signal 1) | BGE biencoder step |
| Relationship graph | `data/veda_relationship_graph.json` | `join_planner.py:30`, used across `veda/pipeline.py` (join/FK/multi-hop/answer-entity), `fast_path.py:77` | Step 12 |
| Value samples | `column_values` (pgvector/table) | `value_resolver.py:166,189`; `value_arbiter.py:304`; `fast_path.py:334` | Step 6 Value Sampler |
| Semantic registry | `semantic/{concepts,dimensions,metrics}.json` | `semantic/registry.py:53`, `fast_path.py:23` | Step 12 |
| Unified graph (expansion) | `data/veda_unified_graph.json` | `query_graph.py:39` ← `veda/pipeline.py:189` (`GRAPH_EXPAND`) | ⚠️ **NOT in `run_ingestion`** — orphaned to standalone builder. **Must wire in (§1a.3).** |
| Query-enrichment lexicons | `data/veda_domain_synonyms.json`, `data/veda_concept_graph.json`, `data/veda_glossary.json` | `retrieval/query_enrichment.py:72,80,88` | synonyms + concept graph: Step 9b (`semantic_layer_v2.py:1544-1553`). ⚠️ **glossary NOT built by `run_ingestion`** (`glossary=None`, `main.py:652`) — **must wire in (§1a.2).** |
| Live schema (FK signals 3&4) | *no artifact* — `information_schema` on source DB | `signal_builder.py:48` → `connectors/relational.py:810` | n/a (runtime introspection) |
| Verified query cache | `substrate_verifiedquerycache` (ctx set) / `data/veda_verified_queries.json` (fallback) | `veda/cache.py`, `storage_adapters/reader.py:155` | runtime write-back |

### 2B. CONDITIONAL — read only on non-default branches
- **Tier-2 LLM fallback** (`TIER2_LLM_FALLBACK=True`, fires only on a deterministic *refusal* + Ollama up → `veda_hybrid.py:452,460` → `query/retrieval_select.py`):
  - `retrieval_select` Step 4 calls `run_semantic_layer` **with the V2 guard pinned** (`retrieval_select.py:121-130`), which runs `_encode_ensemble` (`semantic_layer.py:1105`) → loads **`tfidf_vectorizer.pkl` + `svd_transformer.pkl`** (`semantic_layer.py:317-332`) and **`reg_graph.pkl` + `col_id_to_idx.pkl`** (`semantic_layer.py:447`). *(Backstop at `semantic_layer.py:1069`: if MiniLM isn't loaded it returns `unavailable_minilm` and skips this.)*
  - `retrieval_v2` reads `column_embeddings_v2` + `table_embeddings_v2`; `graph_retriever` reads `graph_nodes`/`graph_edges`/`graph_node_embeddings`; `table_metadata` used for display.
- **RAG / hybrid intent** (only when a document source exists): `doc_chunks` (`rag_layer.py:486`).
- **Graph expansion** is actually a **hot-path** read (moved to §2A above): `data/veda_unified_graph.json` (`query_graph.py:39`), currently orphaned — see §1a.3 / §6.1.

### 2C. NEVER read by EITHER tier → cleanup targets
(The ensemble artifacts + graph tables that a prior draft listed here have **moved to KEEP** — Tier 2 / Tier-1 reranker read them. This list is now only what neither tier touches.)
| Artifact | Produced by | Note |
|----------|-------------|------|
| `ingestion/client_bge` (fine-tuned model) + `finetune_meta.json` | Step 11 | **Both tiers** load base BGE; never loaded. |
| `ingestion/client_minilm` (MiniLM fine-tune) | Step 11 (MiniLM) | Not on the BGE path. |
| `ingestion/training_pairs.jsonl` | Step 10 | Only input to Step 11. |
| `graph_node_embeddings_gnn` | Step 7d | GNN disabled + `run_gnn_graph_embedding` **does not exist** → import always fails (caught). Fully dead. |
| `column_profile` / `ColumnProfile` | (data_profiler) | `main.py:652` passes `profiling=None`; not produced by default and not read by either tier. |
| Persisted semantic-type table | Step 4 (in-mem result is used; the *table* isn't) | Display comes from semantic model + `overrides.json`. **Verify** no Tier-2 reader (§6.3). |
| `data/veda_semantic_checkpoint.json` | Step 9b checkpoint | Ingestion-resume artifact only. |

**Now KEEP (Tier-2 / Tier-1 read them — were briefly cleanup targets):** `column_embeddings_lt`/`_hybrid`, `tfidf_vectorizer.pkl`/`svd_transformer.pkl`, `reg_graph.pkl`/`col_id_to_idx.pkl` (Tier-2 Step-4 ensemble signal); `graph_nodes`/`graph_edges`/`graph_node_embeddings` (Tier-2 graph signal + Tier-1 `reranker`); `table_metadata`/`table_embeddings_v2` (Tier-2). `schema/kuzu_graph` stays with `reg_builder` unless §6.2 shows no reader.

---

## 3. Ingestion stage-by-stage classification

Legend: **KEEP** (read by Tier 1 and/or Tier 2) · **KEEP (indirect)** (in-memory result feeds a KEEP stage) · **REMOVE** (read by neither tier). *Both tiers stay, so Tier-2 reads now count as KEEP.*

| # | Stage | Module | Output | LLM? | Cost | Verdict |
|---|-------|--------|--------|------|------|---------|
| 1 | Schema Scanner | `schema_scanner.run_schema_scanner` | in-mem `ScanResult` | no | cheap | **KEEP** |
| 2 | FK Adjacency Store | `vector_store.store_fk_adjacency` → `fk_adjacency` | table | no | cheap | **KEEP** — Tier-2 v2/graph spine reads it (hot FK uses `information_schema`). |
| 3 | Data Graph (undeclared FK) | `data_graph.run_data_graph` | merged into `fk_adjacency` + feeds relationship graph | no | med (DB sampling) | **KEEP** — discovered joins reach Tier 1 via `veda_relationship_graph.json`. |
| 4 | Semantic Type Inference | `semantic_type_inference.run_...` | in-mem `InferenceResult` (+ persisted semantic-type rows) | no | cheap | **KEEP (indirect)** — in-mem result feeds Steps 5/7/9b. **Persisted table REMOVE** pending §6.3. |
| 5 | Table Metadata Store | `vector_store.store_table_metadata` → `table_metadata` | table | no | cheap | **KEEP** — Tier-2 display resolution reads it. |
| 6 | Value Sampler | `value_sampler.run_value_sampler` → `column_values` | table | no | med | **KEEP** (Tier 1 — value resolver/arbiter). |
| 7 | REG Builder | `reg_builder.run_reg_builder` → `reg_graph.pkl`, `col_id_to_idx.pkl`, kuzu | files + in-mem graph | no | cheap | **KEEP** — in-mem graph feeds encoder + graph persist; pkls read by Tier-2 `semantic_layer`. |
| 7b | Unified Graph Persist | `graph_persist.persist_reg_graph` → `graph_nodes`,`graph_edges` | tables | no | cheap | **KEEP** — Tier-2 `graph_retriever` **and** Tier-1 `reranker` (`reranker.py:72`). |
| 7c | Unified Graph Embedder | `graph_embedder.embed_graph_nodes` → `graph_node_embeddings` | table (BGE) | no (local BGE) | **expensive** | **KEEP** — Tier-2 `graph_retriever`. (Expensive; optional future trim — §4 Tier C.) |
| 7d | Phase-5 GNN | `relgt_encoder.run_gnn_graph_embedding` | `graph_node_embeddings_gnn` | no | expensive | **REMOVE** — disabled + function doesn't exist. Dead code. |
| 8 | Encoder (ensemble) | `relgt_encoder.run_relgt_encoder` | TF-IDF/SVD pkls (+ loads RELGT weights) | no (MiniLM/TF-IDF) | med | **KEEP** — Tier-2 Step-4 ensemble signal. (Degrades gracefully; optional future trim — §4 Tier C.) |
| 9 | Vector Store | `vector_store.run_vector_store` | `column_embeddings_lt`,`_hybrid` (ensemble) | no | med | **KEEP** with Step 8. |
| 9b | Semantic Layer v2 | `semantic_layer_v2.run_full_semantic_layer` → `veda_semantic_model.json` (+synonyms/concepts/glossary) | file | **YES (Qwen)** | **expensive** | **KEEP** (CORE) — main LLM cost; keep. |
| — | BGE Biencoder ingestion | `biencoder.run_biencoder_ingestion` → `column_embeddings_v2`,`table_embeddings_v2` | tables (BGE-large) | no | **expensive** | **KEEP** (CORE — the live retrieval store, both tiers). |
| 10 | Synthetic Query Gen | `synthetic_query_gen.run_...` → `training_pairs.jsonl` | file | **YES (Qwen)** | **expensive** | **REMOVE** — only feeds Step 11, which is dead. |
| 11 | BGE Fine-Tune | `auto_finetune.run_bge_finetune` → `ingestion/client_bge` | model dir | no (training) | **expensive** | **REMOVE** — output never loaded (both tiers use base BGE). |
| 12 | Derived artifacts | `relationship_graph.build_relationship_graph` + `compile_semantic_layer.compile_all` | `veda_relationship_graph.json` + `semantic/*.json` | no | cheap | **KEEP** (CORE). |

---

## 4. Cleanup targets, bucketed by risk (Tier A/B/C/D = cleanup buckets, NOT the query tiers)

### Tier A — safe removals (read by NEITHER query tier)
Pure win. No behaviour change for Tier 1 or Tier 2.
1. **Step 7d GNN** — delete the block (`main.py:539-554`) and the dead `GRAPH_GNN_ENABLED` branch. The called function doesn't even exist.
2. **Step 10 Synthetic Query Gen + Step 11 BGE Fine-Tune** — the fine-tuned model is never loaded; **both tiers** use base BGE (`BIENCODER_MODEL` hardcoded to base). Remove both stages (`main.py:685-734`) and set `AUTO_FINETUNE_ENABLED=False`; delete `synthetic_query_gen.py` + `auto_finetune.py` + `client_bge`/`client_minilm`/`training_pairs.jsonl`. **Single biggest cost saving** (two LLM/training-heavy stages).
   - ⚠️ *Decision point:* if you ever want to **use** the fine-tuned model, the fix is to repoint `BIENCODER_MODEL`/`BGE_MODEL_NAME` → `ingestion/client_bge`. Until then it's pure waste.
3. **`column_profile`/`data_profiler`** — stop writing (no reader in either tier). **Persisted semantic-type table** — remove pending §6.3 (confirm no Tier-2 reader).
4. **`data/veda_semantic_checkpoint.json`** — ingestion-resume artifact only; drop once resume no longer depends on it.
5. **Legacy CLI + dead files** — `_run_single_query_legacy` + `--legacy-query` + `query/execution_engine.py` (separate debug entrypoint, not a tier); and the 4 zero-importer files (§4b A0). None are Tier 1 or Tier 2.

### Tier B — KEEP the tier dependencies (was "remove all fallbacks" — REVERSED per §1a.1)
**No deletions here.** Both tiers stay, so everything Tier 2 reads stays produced:
- **Query modules KEEP:** `retrieval_select.py`, `retrieval_v2.py`, `schema_linker.py`, `graph_retriever.py`, `envelope_slm.py`, `intent_envelope.py`, `slm_layer.run_slm_layer` + `slm_langgraph.py`/`lg_nodes.py`/`lg_prompts.py`, `semantic_layer.py`.
- **Ingestion artifacts KEEP:** ensemble encoder (Step 8) + `_lt`/`_hybrid` tables (Step 9) + `tfidf/svd` pkls + `reg_graph.pkl`/`col_id_to_idx.pkl` + `reg_builder` (Step 7) — Tier-2 Step-4 ensemble signal reads them; `table_metadata`/`table_embeddings_v2` — Tier-2.
- 🔭 **Optional future trim (NOT now):** Tier 2's Step-4 ensemble signal degrades gracefully to BGE-only (`retrieval_select.py:116-121`; skips when MiniLM isn't loaded, which is the platform default). So the ensemble encoder + its pkls/tables *could* later be dropped and Tier 2 would still work BGE-only. Left in place per the keep-both-tiers directive; revisit only if you want to slim ingestion further and accept Tier 2 losing that one legacy signal.

### Tier C — graph stages: KEEP (both tiers read them) + wire in the two graphs
- **KEEP (do NOT remove):** Step **7b** graph persist (`graph_nodes`/`graph_edges`) and Step **7c** graph embedder (`graph_node_embeddings`). Read by Tier-2 `graph_retriever` **and** Tier-1 `reranker` (`SELECT name FROM graph_nodes`, `reranker.py:72`). Step **7 REG build** stays too (feeds the encoder + graph persist, and its pkls feed Tier-2 `semantic_layer`).
- **KEEP + guarantee production (owner directive §1a.3):**
  - `data/veda_relationship_graph.json` (Step 12) — join planner (Tier 1). Already produced.
  - `data/veda_unified_graph.json` — graph expansion (Tier 1). **Wire `unified_graph_builder.build_unified_graph` into `run_ingestion`** (after Step 9b) — today orphaned to the standalone script, so expansion may run on a stale/missing file.
- 🔭 **Optional future trim (NOT now):** 7c (`graph_node_embeddings`) is the expensive full-node BGE embed and feeds only Tier-2's graph signal — a candidate for later slimming, same tradeoff as Tier B.

### Tier D — glossary must stay wired (add, don't remove) — **decided, see §1a.2**
The domain glossary is a query-time input (`query_enrichment.py:88`) but is **not** built by `run_ingestion`. **Add** a glossary build/refresh to `run_ingestion` (call `ingestion/domain_glossary.build_glossary`, or pass a non-`None` glossary + `force_glossary` into `run_full_semantic_layer` at `main.py:651`) so `data/veda_glossary.json` is regenerated every ingest. This is an **addition**, explicitly protected from the fallback cleanup.

---

## 4b. Files & artifacts that can be removed

Derived from an import-graph trace (who imports each candidate). Classified by confidence so nothing with a surviving importer is deleted blindly. **Verify §6.7 first** — one graph module is touched by a hot-path reader.

### A0. Source modules — DEAD NOW (zero importers, independent of any cleanup)
Verified by import-graph trace: **nothing** in the repo imports these. They can be deleted immediately.
| File | Note |
|------|------|
| `veda_core/query/audit_logger.py` | No importer; audit is done by `apps/query` `QueryLog`, not this module. (Has a `__main__` demo block only.) |
| `veda_core/query/executor.py` | No importer; execution goes through `veda/execution.py`. Superseded. |
| `veda_core/query/sql_generator.py` (462 lines) | No importer; SQL is built by `veda/generation` + `query/sql_builder`. Orphaned. |
| `veda_core/query/sql_validator.py` (806 lines) | No importer; validation is `veda/validation` + `graph_guard`. Orphaned. |

### A. Source modules — safe delete (read by neither tier)
| File | Only imported by (all removed) |
|------|-------------------------------|
| `veda_core/query/execution_engine.py` | only `main.py` legacy query path — removed with `--legacy-query` |
| `veda_core/ingestion/synthetic_query_gen.py` | `main` Step 10, `auto_finetune`, `source_dispatcher` — all removed (Tier A) |
| `veda_core/ingestion/auto_finetune.py` | `main` Step 11, `source_dispatcher`, `relgt_encoder` (finetune code) — all removed (Tier A) |

### A-KEEP — ⛔ DO NOT DELETE (Tier 2 depends on these — reversed from an earlier draft)
These were on the delete list under "remove all fallbacks"; **Tier 2 uses them, so they stay:**
`query/retrieval_select.py`, `query/retrieval_v2.py`, `query/graph_retriever.py`, `query/schema_linker.py`, `query/envelope_slm.py`, `query/intent_envelope.py`, `query/slm_langgraph.py`, `query/lg_nodes.py`, `query/lg_prompts.py`, `query/semantic_layer.py`, `ingestion/graph_embedder.py` (produces `graph_node_embeddings` for Tier-2 `graph_retriever`), `ingestion/reg_builder.py` (feeds encoder + `reg_graph.pkl` for Tier-2 `semantic_layer`).

### B. Source modules — delete after cutting one import from a surviving file
| File | Cut this first |
|------|----------------|
| `veda_core/ingestion/data_profiler.py` | remove the module-level import in `semantic_layer_v2.py` (runtime already passes `profiling=None`) |
| `veda_core/ingestion/chunk_linker.py` | only `main.run_doc_ingestion` imports it — remove with the graph-linking block (only if doc ingestion isn't used) |
| `veda_core/schema/simulate_schema.py` | superseded by `real_schema.py`; referenced as a fallback in several modules — remove those fallback branches first (§6.8) |

### C. Trim, don't delete (file survives; strip only the dead Tier-A code)
- `veda_core/ingestion/relgt_encoder.py` — drop only the `run_gnn_graph_embedding` refs (Step 7d). **Keep** the ensemble/light-text/hybrid encoders (Tier-2 Step-4 signal via `semantic_layer`) and `_get_minilm_model`.
- `veda_core/main.py` — delete `_run_single_query_legacy` + `--legacy-query` flag; delete Steps 7d/10/11 blocks. **Keep** the `_tier2_sql` wiring in `veda_hybrid` (Tier 2) and Steps 7/7b/7c/8/9 (tier deps).
- `veda_core/ingestion/source_dispatcher.py` — delete only the synthetic-gen + fine-tune stages in `_run_schema_pipeline` (lines 270-313). **Keep** the ensemble encoder + graph stages.

### D. Data / model / pickle artifacts to delete & stop writing (read by neither tier)
| Path | Why |
|------|-----|
| `veda_core/ingestion/client_bge/` + `finetune_meta.json` | fine-tuned model **never loaded** (both tiers use base) |
| `veda_core/ingestion/client_minilm/` | MiniLM fine-tune output; not on BGE path |
| `veda_core/ingestion/training_pairs.jsonl` | Step-10 output; only fed Step 11 (removed) |
| `veda_core/data/veda_semantic_checkpoint.json` | ingestion-resume checkpoint only; no query-time reader |

> ⛔ **NOT deleted (Tier-2 reads them):** `schema/tfidf_vectorizer.pkl`, `schema/svd_transformer.pkl`, `schema/reg_graph.pkl`, `schema/col_id_to_idx.pkl`, `schema/kuzu_graph/`. These show as untracked in git but are **regenerated each ingest** and consumed by Tier-2 `semantic_layer` — keep producing them.

### DB tables to drop (migrations) — read by neither tier
`graph_node_embeddings_gnn` (GNN, dead), `column_profile`, persisted semantic-type table (pending §6.3).
> ⛔ **NOT dropped (Tier-2 / Tier-1 read them):** `column_embeddings_lt`, `column_embeddings_hybrid`, `table_embeddings_v2`, `graph_node_embeddings`, `graph_nodes`, `graph_edges`, `table_metadata`.

---

## 4c. Query flow — what the engine does NOT use (removable paths)

Traced `run_hybrid_query` → `_dispatch_single` → `veda/pipeline.run_query`, with an import-graph over the whole `query/` package. Config that shapes the live path: `QUERY_ROUTER_ENABLED=True`, `QUERY_DECOMPOSE_ENABLED=False`, `TIER2_LLM_FALLBACK=True` (conditional), `USE_LANGGRAPH=True`, `FAST_PATH_ENABLED=True`, `PRIMARY_RERANK_ENABLED` (on).

### The two tiers we KEEP
- **Tier 1** — `veda_hybrid.run_hybrid_query` → `classify` (`query_router.route_query`) → `_dispatch_single` → **`veda/pipeline.run_query`**. Modules used: `temporal_parser` · `fast_path` · `reranker` (reads `graph_nodes`) · `answer_entity` · `value_resolver` · `fk_path_resolver` · `value_arbiter` · `value_filter` · `nl_answer` · `sql_builder` · `target_selection` · `join_planner` · `intent` · `nl_simplifier`. Non-SQL intents: `rag_layer`, `nosql_builder`. Decompose toggle: `slm_layer.run_decomposer`, `multi_result`.
- **Tier 2** — `veda_hybrid._tier2_sql` (fires when Tier 1 refuses). Modules used: `retrieval_select` (→ `schema_linker`, `retrieval_v2`, `graph_retriever`, `semantic_layer`), `envelope_slm` + `intent_envelope`, `slm_layer.run_slm_layer` (+ `slm_langgraph`/`lg_nodes`/`lg_prompts` under `USE_LANGGRAPH`), `sql_builder`. **All KEPT.**

### Removable query-flow paths (NOT part of either tier)
| Path / module | Why removable | Reached only by |
|---------------|---------------|-----------------|
| **Legacy CLI query** `_run_single_query_legacy` (`main.py`) + `--legacy-query` flag | L1→L4 debug path superseded by the hybrid engine; a separate CLI entrypoint, not Tier 1 or Tier 2 | manual CLI flag only |
| `query/execution_engine.py` | only the legacy CLI imports it; both tiers execute via `veda/execution.execute_sql` | `main.py` legacy only |
| **Dead now** `query/audit_logger.py`, `executor.py`, `sql_generator.py`, `sql_validator.py` | **Zero importers** anywhere — superseded by `apps/query.QueryLog`, `veda/execution`, `veda/generation`, `veda/validation` | nothing |

> Note: removing `--legacy-query` does **not** let you delete `semantic_layer.py`, `slm_layer.py`, or `temporal_parser` — Tier 2 (and Tier 1) still use them. Only `execution_engine.py` is exclusive to the legacy CLI. `slm_layer.py` stays for both `run_slm_layer` (Tier 2) and `run_decomposer` (decompose toggle).

**Net:** the query engine keeps its two tiers intact. The only query-flow removals are the standalone legacy CLI debug path (`_run_single_query_legacy` + `--legacy-query` + `execution_engine.py`) and the four already-orphaned zero-importer files.

---

## 4d. Unused `config.py` keys & dead code modules (repo scan)

Method: extracted all 367 top-level `UPPER_CASE` keys in `config.py`, then a single-pass token scan across every `.py` in the repo (catches `from config import X`, `config.X`, and string-literal `getattr(cfg,"X")`). **69 keys have zero references** outside their own definition. Spot-checks confirm the reason — e.g. the Phase-3 retrieval engine hardcodes `RRFMerger(k=60)` / `top_k=50`, so the `BM25_*`/`RRF_*` knobs are orphaned leftovers from the legacy V1 retrieval.

> Caveat before deleting: none of these 69 are in the `settings_bridge.build_veda_settings()` bridge whitelist (that only bridges `ENCODER_MODE`, `TOP_K`, `QUERY_ROUTER_ENABLED`, …). A key read *only* via a `VEDA_<FLAG>` env override or dynamic `vars(config)` iteration would not be caught — verify those two patterns for any key you delete. All 69 are safe to remove otherwise.

Grouped by the dead subsystem they belong to (this **triangulates the dead files** in §4b — the config keys for `sql_generator`/`sql_validator`/`audit_logger` are all dead too):

| Dead subsystem | Unused keys | Cross-ref |
|----------------|-------------|-----------|
| **Legacy SQL generator** (`query/sql_generator.py`) | `SQL_GENERATION_V2_ENABLED`, `SQL_GENERATION_MODEL`, `SQL_GENERATION_TEMPERATURE`, `SQL_GENERATION_TIMEOUT`, `SQL_GENERATION_MAX_TOKENS`, `SQL_FALLBACK_ENABLED`, `SQL_FALLBACK_SIMPLE_SELECT`, `SQL_FALLBACK_AGGREGATE_COUNT` | §4b A0 file dead |
| **Legacy SQL validator** (`query/sql_validator.py`) | `VALIDATION_V2_ENABLED`, `VALIDATION_CHECK_BLOCKED_KEYWORDS`, `VALIDATION_BLOCKED_KEYWORDS`, `VALIDATION_CHECK_WHITELIST`, `VALIDATION_WHITELIST_EXEMPT_ALIASES`, `VALIDATION_WHITELIST_EXEMPT_FK_COLUMNS`, `VALIDATION_CHECK_AGGREGATION`, `VALIDATION_CHECK_EXPLAIN`, `VALIDATION_REPAIR_LOOP_ENABLED`, `VALIDATION_MAX_REPAIR_ATTEMPTS`, `VALIDATION_REPAIR_COLUMN_REMAP`, `VALIDATION_REPAIR_FALLBACK_AGGREGATION`, `VALIDATION_REPAIR_QWEN_REPAIR` | §4b A0 file dead |
| **Audit logger** (`query/audit_logger.py`) | `AUDIT_LOGGING_ENABLED`, `AUDIT_LOG_TABLE`, `AUDIT_LOG_COLUMNS` | §4b A0 file dead |
| **GNN** (Step 7d) | `GRAPH_GNN_OUTPUT_DIM`, `GRAPH_GNN_LAYERS`, `GRAPH_GNN_NODE_EMB_TABLE` | §4 Tier A |
| **Legacy file-based stores** (superseded by pgvector/substrate) | `RETRIEVAL_DOCUMENTS_FILE`, `BGE_EMBEDDINGS_FILE`, `TABLE_VECTORS_FILE`, `BM25_CORPUS_FILE`, `NX_GRAPH_FILE`, `VERIFIED_QUERIES_FILE`, `VERIFIED_QUERY_THRESHOLD`, `VERIFIED_QUERY_MAX_COUNT`, `SCHEMA_FINGERPRINT_ENABLED`, `SCHEMA_FINGERPRINT_FILE` | pgvector now |
| **Legacy retrieval tuning** (Phase-3 hardcodes these) | `BM25_ENABLED`, `BM25_TOP_K`, `RRF_K_VALUE`, `RRF_TOP_K_AFTER_FUSION`, `SUBGRAPH_PRIMARY_TABLE_SCORE`, `SUBGRAPH_FK_NEIGHBOR_SCORE`, `FK_PATH_SCORE`, `GROUNDING_FLOOR`, `HISTORY_TABLE_PENALTY`, `INTENT_AGGREGATE_KEYWORDS`, `INTENT_TEMPORAL_KEYWORDS`, `INTENT_BOOST_MEASURE_AGGREGATE`, `INTENT_BOOST_MEASURE_TEMPORAL`, `INTENT_BOOST_IDENTIFIER_AGGREGATE`, `INTENT_BOOST_IDENTIFIER_TEMPORAL`, `INTENT_AND_CACHE_ENABLED`, `RETRIEVAL_V2_FINAL_ENABLED` | hardcoded in engine |
| **Unused model-name / dim knobs** | `GLOSSARY_MODEL`, `TABLE_UNDERSTANDING_MODEL`, `COLUMN_UNDERSTANDING_MODEL`, `BGE_HYBRID_EMBEDDING_DIM`, `BGE_DIM`, `BGE_BATCH_SIZE`, `BGE_EMBEDDING_TIMEOUT` | — |
| **Misc unused** | `VALUE_FILTER_CASE_INSENSITIVE`, `VALUE_FILTER_MIN_VALUE_MATCH`, `TEMPORAL_DATEPARSER_SETTINGS`, `EVAL_OUTPUT_DIR`, `SCORE_PRECISION`, `BASELINE_LABEL`, `MAX_COLUMNS_PER_TABLE`, `ROUTE_LOG_PATH` | — |

### Dead code modules (zero importers anywhere — repo-wide scan of `veda_core/`)
Beyond the four already in §4b A0 (`audit_logger`, `executor`, `sql_generator`, `sql_validator`), these have **no importer**:
| Module | Note |
|--------|------|
| `veda_core/veda/consensus.py` | No import; the only match is a stray comment in `verifier.py`. Dead. |
| `veda_core/veda/ir_emit.py` | No import; "ir_emit" appears only as an SLM *purpose label* string in `_call_slm.py`. Dead. |
| `veda_core/ingestion/semantic_postprocessor.py` | Zero references anywhere. Dead. |

**Standalone `__main__` dev/CLI scripts (no importer, but runnable by hand — keep or delete at discretion, not runtime code):** `ingestion/build_intermediate_files.py`, `ingestion/enhance_semantic_model.py`, `ingestion/enrich_retrieval_documents.py`, `ingestion/gen_debug_files.py`, `graph/graph_validator.py`. These are one-off tooling; remove if you want a lean tree, but they don't affect ingestion or query at runtime.

> Add to the removal set: the 69 config keys above + `veda/consensus.py`, `veda/ir_emit.py`, `ingestion/semantic_postprocessor.py`. Delete the config keys **in the same PR** as their owning subsystem (e.g. `SQL_GENERATION_*` with `sql_generator.py`) so nothing references a just-deleted key.

---

## 5. API-driven, per-tenant / per-source ingestion (the second ask)

### Current state (verified)
- **Endpoint:** `apps/query/views.py::IngestTriggerView` (staff-only) → routed in `apps/query/urls.py` → enqueues `apps/ingestion/tasks.py::task_ingest_source(source_id, tenant, force, skip_llm, resume)`.
- **Per-source connection: ✅ works.** `task_ingest_source` reads the DB `Source` row and injects `Source.as_engine_env()` into the subprocess (`tasks.py:194-196`), so the engine targets that source's DB (`VEDA_SOURCE_*` → `veda/runtime.DB_CONFIG`).
- **Per-tenant: ✅ partially.** `set_context(RequestContext(source_id, tenant))` (`tasks.py:130`); job/stages tagged with tenant; warm/rehydrate scoped.
- **Source-type routing: ❌ MISSING.** The task always runs `python -c "import main; main.run_ingestion(...)"` (`tasks.py:202-203`), which internally calls `get_primary_relational_source()` and ingests as the **primary relational** source. It **hardcodes** `result={"source_id":"primary_db"}` (`tasks.py:247`) and never inspects `source.type`. Document/nosql/datalake sources are **not** routed to their pipelines.
- **`dispatch_ingestion` exists but is unused by the API.** `ingestion/source_dispatcher.dispatch_ingestion` is the type-aware router (relational/datalake/document/nosql), but only `main.run_all_ingestion` calls it — not the Celery task. And even its `_dispatch_relational` for the *primary* delegates back to `run_ingestion` (primary-only), ignoring `source_config`.
- **Config duplication.** `veda_core/config.py::VEDA_SOURCES` (hardcoded list) duplicates the DB `Source` rows. The engine reads sources from `VEDA_SOURCES`; the platform's source of truth is the `Source` table. These can drift.
- **Skeleton chain.** `apps/ingestion/tasks.py::STAGE_ORDER` + `task_schema_scan…task_unified_graph` all `raise NotImplementedError` (`tasks.py:44-94`) — the intended decomposition, not wired.

### Target design
1. **Make the task source-type-aware.** In `task_ingest_source`, build the source config from the DB `Source` row (id, **type**, engine, connection) and pass it into the subprocess so the engine runs the **matching** pipeline — call `source_dispatcher.dispatch_ingestion(source_config)` instead of the hardcoded `run_ingestion()`. This gives per-source routing (relational → full pipeline; document → chunk embed; nosql/datalake → schema pipeline) with the connection already injected per source.
2. **Fix `_dispatch_relational` to honour the passed source** rather than always re-deriving the primary — accept the connection/id from the caller and ingest *that* source's schema (the connector already fetches `raw_schema_dict`; feed it into the shared pipeline for non-primary too, or generalize `run_ingestion` to take a `source_config`).
3. **Single source of truth for sources.** Have the subprocess receive the one source's config as JSON (env or `--source-json`) derived from the DB `Source` row, so `VEDA_SOURCES` in `config.py` is no longer the authority (keep only as a dev fallback). Removes drift.
4. **API surface.** Keep `POST /api/v1/admin/ingest {source_id}` (tenant server-resolved). Ingestion is then a pure data operation per (tenant, source), routed by type — matching `ARCHITECTURE.md §8`.
5. **Apply the §4 cleanup inside the per-type pipelines** (`_run_schema_pipeline` in `source_dispatcher.py` currently also runs the ensemble encoder + synthetic-gen + fine-tune at lines 240-313 — same dead stages, remove there too).

---

## 6. Things to verify before deleting

Most Tier-2-dependency questions from the earlier draft are now **moot** — we keep both tiers, so their reads stay. Remaining checks are only for the Tier-A removals:

1. **`data/veda_unified_graph.json` — DECIDED: wire the builder in (§1a.3).** Confirm `build_unified_graph`'s inputs (synonyms + concept-graph files, `unified_graph_builder.py:53-54`) exist at the insertion point (after Step 9b) so it produces a complete graph.
2. **Glossary build (§1a.2)** — confirm `build_glossary` runs offline/zero-egress-safe within the ingest subprocess (it calls Ollama for synonyms) and writes to `GLOSSARY_FILE` (`data/veda_glossary.json`) so `query_enrichment` picks it up.
3. **Persisted semantic-type table** — confirm no Tier-1 **or Tier-2** reader (e.g. `semantic_layer` / display resolution) before dropping it. If any tier reads it, keep it. `column_profile` is safe (no reader; `profiling=None`).
4. **`chunk_linker.py` / `run_doc_ingestion`** — only remove if document ingestion is not in use; if doc sources are onboarded, keep it.
5. **`schema/simulate_schema.py`** — referenced as a fallback in `config.py`, `schema_scanner`, `data_graph`, `value_sampler`, `synthetic_query_gen`, `auto_finetune`. Confirm `real_schema` is the sole live path (simulated schema is dev-only) and remove the fallback branches first (some importers, e.g. `synthetic_query_gen`, are themselves being deleted).
6. **Legacy CLI** — confirm `execution_engine.py` is imported **only** by the legacy CLI path (verified: `main.py` only) before deleting; both tiers use `veda/execution.execute_sql`.

---

## 7. Recommended execution order (phased, each independently shippable)

**Phase 0 — verify (no changes).** Run `scripts/parity_suite.py` + eval to capture a baseline. Confirm the §6 questions.

**Phase 1 — Tier A removals (pure win, neither tier affected).** Remove Step 7d GNN; delete Steps 10–11 (synthetic gen + BGE fine-tune), `AUTO_FINETUNE_ENABLED=False`, delete `synthetic_query_gen.py`/`auto_finetune.py`/`client_bge`/`client_minilm`/`training_pairs.jsonl`; stop persisting `column_profile` (+ semantic-type table per §6.3); remove the legacy CLI (`_run_single_query_legacy` + `--legacy-query` + `execution_engine.py`) and the 4 zero-importer dead files. **Also (§4d):** delete the 3 dead modules (`veda/consensus.py`, `veda/ir_emit.py`, `ingestion/semantic_postprocessor.py`) and the 69 unused `config.py` keys — each key group in the same commit as its owning subsystem. Re-run eval → expect **identical** Tier-1 and Tier-2 metrics, faster ingestion.

**Phase 2 — additions the owner requires.**
- **Wire the domain glossary into `run_ingestion`** (§1a.2 / Tier D) — `veda_glossary.json` regenerated every ingest.
- **Wire `unified_graph_builder.build_unified_graph` into `run_ingestion`** (§1a.3 / Tier C) — `veda_unified_graph.json` regenerated every ingest.
- Verify both files are freshly produced and read at query time (glossary in `query_enrichment`, unified graph in `query_graph`).

**Phase 3 — API per-source routing.** Rewire `task_ingest_source` → `dispatch_ingestion` with the DB `Source` row's type + connection; fix `_dispatch_relational`; pass source config as JSON into the subprocess; make `VEDA_SOURCES` a dev-only fallback. Test with a document + a second relational source.

**Phase 4 — housekeeping.** Delete the `NotImplementedError` skeleton chain in `apps/ingestion/tasks.py` (or implement it) so there's one ingestion path. Apply the same Tier-A removals inside `source_dispatcher._run_schema_pipeline` (synthetic-gen + fine-tune, lines 270-313).

**(Deferred / optional) — slim the tier deps.** Only if you later decide to shrink ingestion further and accept Tier 2 running BGE-only: drop the ensemble encoder (Steps 8/9 + TF-IDF/SVD + `reg_graph.pkl`) and/or 7c graph embeddings. **Not part of this plan** — both stay while both tiers stay (§1a.1, §4 Tier B/C).

---

## Appendix — what ingestion looks like after cleanup (relational source)

`Schema scan → FK adjacency → data graph → semantic-type inference → table metadata → value sampler → REG build (reg_graph.pkl) → graph persist (graph_nodes/edges) → graph embed (graph_node_embeddings) → encoder (ensemble: TF-IDF/SVD + _lt/_hybrid) → vector store → semantic layer v2 (Qwen: model + synonyms + concept graph) → BGE biencoder embed (column_embeddings_v2) → domain glossary build (veda_glossary.json) → derived artifacts (relationship graph + semantic registry) → unified graph build (veda_unified_graph.json)`

**Removed vs today:** Step 7d GNN, Step 10 synthetic query gen, Step 11 BGE fine-tune (+ `client_bge`/`client_minilm`/`training_pairs.jsonl`), `column_profile`, the legacy CLI (`--legacy-query` + `execution_engine.py`), and the 4 zero-importer dead files. **Both query tiers untouched.**

**Added / guaranteed:** domain glossary build and unified-graph build now run inside `run_ingestion` (previously orphaned / CLI-only).

**Still produced — read by Tier 1 and/or Tier 2:** semantic model, `column_embeddings_v2`, relationship graph, unified graph, semantic registry, value samples, enrichment lexicons (glossary/synonyms/concept graph), **plus the Tier-2 dependencies** — ensemble `_lt`/`_hybrid` tables, TF-IDF/SVD + `reg_graph.pkl`/`col_id_to_idx.pkl`, `graph_nodes`/`graph_edges`/`graph_node_embeddings`, `table_metadata`/`table_embeddings_v2`.

> Kept-vs-trimmed at a glance: **KEEP** = anything Tier 1 or Tier 2 reads (incl. the whole ensemble + graph spine). **REMOVE** = fine-tune chain, GNN, profiler, legacy CLI, zero-importer files.
