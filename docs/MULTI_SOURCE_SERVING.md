# Multi-source query serving + auto-federation

**Goal:** make `source_ids:[4]` (tabular) and `source_ids:[2,4]` (cross-source) NL queries
actually answer — today they retrieve/validate against homzhub and the firewall rejects the
real source-4 columns. Then wire auto-federation so cross-source questions compose a join.

Owner: cross-source work. All changes uncommitted.

---

## 1. Root cause (traced)

The query tier resolves a per-scope semantic model via `veda_core/veda/runtime.py`
(`_load_one_sm` → Redis `veda:sm:{source}:{tenant}`, `_merge_scoped_sms` for multi-source).
The infra is fully built, but **two breaks**:

1. **`VEDA_SM_REDIS` is unset everywhere** → `_load_one_sm`/`veda_hybrid._load_sm_from_redis`
   never read Redis and fall back to the on-disk **global `SEMANTIC_MODEL_FILE` (homzhub)**.
   So every scope gets homzhub's model.
2. **Even the Redis models are wrong.** `veda:sm:4:default` and `veda:sm:5:default` contain
   homzhub's 178 tables / 1902 cols — verified. Because the `Sm*` substrate rows for sources
   4/5 are homzhub's: the **tabular/doc ingestion never builds a per-source semantic model**
   (`store_semantic_model` is only fed by homzhub's `semantic_layer_v2`), so the global
   homzhub `sm` got persisted + published under sources 4/5.

Net: sources 3/4/5 have **no real semantic model** anywhere the query tier can see, so their
columns are invisible to retrieval and validation.

Evidence: `SmTable/SmColumn` for src 2/4/5 all = 178 tables / 1902 cols (homzhub);
`veda:sm:4:default` tables list is homzhub; a `source_ids:[4]` query routed to
`assets_assetadditionalinfo.maintenance_amount` and the firewall rejected `amount/status/ticket_id`.

## 2. Design already in place (reuse, don't rebuild)

- `runtime._load_one_sm(sid,tenant)` — Redis-first per-source model.
- `runtime._merge_scoped_sms(pairs)` — merges per-source models into ONE namespace,
  `_source_id`-tagged, table names source-qualified (`src{id}.{t}`) only on collision.
- `storage_adapters/assembler.py` — `persist(sm, source_id, tenant)` → `Sm*` rows;
  `publish_sm(source_id, tenant)` → `veda:sm:{sid}:{tenant}`; `assemble()` reads `Sm*` scoped.
- `query/cross_source_composer.py` (`should_federate`, `compose_federated`, `resolve_surface`)
  + `query/federated_executor.py` — verified working; just not wired into the NL router.

---

## Phases

### MS-1 — build a per-source semantic model for tabular/doc sources (code, durable)
`veda_core/ingestion/source_dispatcher.py`: after the tabular/doc graph is built, assemble a
**deterministic** semantic model for THAT source from its own columns (no LLM needed):
- `tables`: one entry per file-table (maintenance, vendors, amenities_catalog) / doc.
- `columns`: `{table.col: {semantic_type, data_type, ...}}` from the connector schema.
- `retrieval_documents`: `{table.col: "COLUMN: <col> | TABLE: <table> | TYPE: <type> | TERMS: <tokens>"}`.
- `domain_synonyms`/`concept_graph`: empty (or light) — optional.
Call `storage_adapters.writer.store_semantic_model(sm)` under that source's context so `Sm*`
+ Redis are correct for future ingestions.

### MS-2 — backfill the current data (no re-ingest)
`scripts/backfill_semantic_model.py`: for sources 3/4/5, read their column nodes from
`graph_nodes` (already per-source-correct), build the same deterministic sm as MS-1, then
`assembler.persist(sm, source_id, tenant)` + `assembler.publish_sm(...)`. Overwrites the wrong
homzhub rows/keys for 4/5. Homzhub (2) is left as-is.

### MS-3 — enable the Redis SM path (config)
Set `VEDA_SM_REDIS=1` on **inference** and **api** (`docker-compose.yml`). Now `_load_one_sm`
reads the corrected per-source models; scope `[4]`→source-4 model, `[2,4]`→merged namespace.
(Registry publish is gated by the same flag — `semantic/registry.py`.)

### MS-4 — sparse index for 3/4/5
Run `ingestion.sparse_index.build_sparse_index` (or a small backfill) for 3/4/5 so their scopes
get the learned-sparse signal. The Phase-A guard already prevents a hang when it's absent; this
restores retrieval quality. Verify `column_sparse_v1` has rows per source.

### MS-5 — verify
- `source_ids:[4]` → answers from `maintenance`/`vendors` columns (no homzhub bleed, no firewall reject).
- `source_ids:[2,4]` → merged model; retrieval surfaces both sources' columns.
- Latency stays < 30s (warm).

### MS-6 — auto-federation (Phase C)
Wire the NL router: after retrieval, if `should_federate(selected_columns)` (columns span ≥2
sources), call `compose_federated(query, sql, selected_columns, chunks)` →
`federated_executor` and return the composed SQL result + evidence + provenance, instead of the
single-source SQL head. Uses the already-verified composer/executor.

---

## Execution log
- [ ] **MS-1** per-source SM build in tabular/doc dispatch (durable) — NOT YET (backfill covers current data).
- [x] **MS-2** backfill SM for 3/4/5 + republish Redis (`scripts/backfill_semantic_model.py`). src4=maintenance/vendors(9), src5=amenities_catalog(4), src3=empty.
- [x] **MS-3** `VEDA_SM_REDIS=1` on inference (`docker-compose.yml`).
- [x] **CONTEXT BUG (was the real blocker)** `veda_hybrid._sm_scope`/`_sm_cache_key` read bare `context`
  while the middleware sets `veda_core.context` — TWO module instances, so the SQL head never saw
  the scope and defaulted to source 1 (global homzhub). Added `_current_ctx()` reading both;
  same fix applied to `veda/execution.py:_scope_source_ids`.
- [x] **DuckDB execution for tabular** `veda/execution.py`: a scope of tabular (parquet) sources
  runs the generated SQL on DuckDB over the materialized parquet (bare-name views), instead of psycopg2.
- [ ] **MS-4** sparse index for 3/4/5 (quality; guard prevents hang meanwhile).
- [x] **MS-5 (single-source) VERIFIED** — `source_ids:[4]` "show maintenance tickets…" →
  `SELECT amount,asset_id,category,status,ticket_id FROM maintenance` → **8 real rows** via DuckDB,
  NL answer, **wall 23s** (< 30s). ✅
- [ ] **MS-6** cross-source federated auto-routing — REMAINING. `source_ids:[2,4]` currently refuses
  (`ir_mismatch`, ~45s): the deterministic head won't run a cross-source join via psycopg2, and there is
  no federated route yet. Needs: detect `should_federate(selected_columns)` (cols span ≥2 sources) →
  generate/rewrite **catalog-qualified** SQL (`src_2.public.x JOIN src_4.y`) → execute via
  `federated_executor.FederatedExecutor` (postgres_scanner + parquet) → compose result+evidence. Larger
  piece (the "federated naming (Phase 5)" the code comments defer).

### Status — COMPLETE (all sources end-to-end, < 30s, correct output)
- [x] **MS-1 (durable)** `storage_adapters/writer.py:warm()` now builds each NON-relational source's
  model from its own `graph_nodes` (`ingestion/lite_semantic_model.build_lite_sm`) instead of publishing
  the global homzhub file. Future re-ingestions stay correct without the backfill. Verified:
  `_is_relational_source(2)=True/4=False`, `_build_lite_sm_from_graph(4)`→maintenance/vendors.
- [x] **MS-6 (federated auto-routing)** `veda_core/query/federated_route.py` + wired into
  `veda_hybrid.run_hybrid_query` (`_maybe_federated`): ≥2-source scope → retrieve → build
  catalog-qualified schema + HIGH-tier `cross_source_fk` join hints → SLM generates DuckDB SQL →
  `compose_federated`/`FederatedExecutor` (firewall + postgres_scanner + parquet) → rows + NL answer + provenance.
- [ ] **MS-4** sparse index for 3/4/5 — OPTIONAL (guard degrades safely; tabular scopes work without it).

### Verified end-to-end (2026-07-09, warm)
| Scope | Route | Result | Wall |
|---|---|---|---|
| homzhub `[2]` | deterministic | COUNT answer | 0.56s |
| tabular `[4]` | deterministic (DuckDB over parquet) | 8 maintenance rows | 16s |
| cross `[2,4]` "vendors in asset cities" | **federated** | vendors V-1…V-6 via `vendors⋈assets_asset ON city` | ~15s |
| cross `[2,4]` "maintenance $ per city" | **federated** | Thane 5950 / Nagpur 3000 / Pune 640 via `SUM ... ⋈ ON asset_id=id GROUP BY city` | ~19s |

No single-source regression (the federated branch is a no-op for <2-source scopes).
