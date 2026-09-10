# VEDA — Latency Fix Implementation Plan
## veda_core engine only, mapped to the L1–L7 query layer contracts

Reference: `query/contracts/L2_RETRIEVAL.md`, `L4_INTENT.md`, `L5_SQLGEN.md`,
`L7_EXECUTION.md`. Findings verified against this build's actual code (not
assumed from a prior snapshot) — notably: `VEDA_ANN_VIA_ADAPTER` now defaults
**ON**, so Signal 1 (BGE-M3 dense) is real work on every query, not gated off.
Per-source `hnsw.ef_search` is already cached per-process (not a per-query DB
hit) — do not re-touch that.

---

## Already fixed in this build (do not re-touch)
1. `storage_adapters/reader.py::_resolve_ef_search` — per-source `ef_search`
   cached per process; tuned at ingestion L5_PUBLISH, persisted on
   `SubstrateVersion`, cleared by the rehydrate subscriber on re-ingest.
2. `VEDA_ANN_VIA_ADAPTER` defaults to `"1"` — Signal 1 is live HNSW work now.
3. `veda/runtime.py::warm_up()` — Ollama prewarm runs on a background thread
   concurrently with registry/engine warm-up, not serially.

---

## Findings & Fixes

### F1 — 5 retrieval signals run serially (L2_RETRIEVAL) 🔴 highest impact
**File:** `retrieval/retrieval_engine_phase3.py::retrieve()`
**Order today:** Signal 1 (BGE, real HNSW work) → Signal 2 (BM25) → Signal 3
(loop all columns) → Signal 4 (loop all columns again) → Signal 5 (nested scan).
**Fix:**
- Parallelize Signal 1 + Signal 2 (ThreadPool, 2 workers, join before RRF). RRF
  fusion is order-independent → output is unchanged.
- Merge Signal 3 & Signal 4 into one `for col_id in columns` loop (both are cheap
  `get_signal()` dict lookups — halves the passes).
- Precompute a Signal-5 inverted value index (`{value_token: [col_ids]}`) once at
  engine build time (`RetrievalEnginePhase3.__init__` / `get_engine()`), so
  Signal 5 becomes a set lookup instead of a per-query nested scan.
**Contract impact:** none — `retrieve()`'s external signature and output
(`List[RetrievalResult]`) are unchanged; only the internal signal computation is
reorganized.

### F2 — L2b Primary rerank scores every candidate, every time 🔴 high
**File:** `veda/pipeline.py`, the `PRIMARY_RERANK_ENABLED` block (L2b).
**Today:** `_pairs = [[_search, ...] for r in results]` — no cap, no skip
condition; cross-encoder forward pass per candidate.
**Fix:**
- Skip L2b entirely when the RRF score gap between candidate #1 and #2 is wide
  and both belong to the same table (new config `RERANK_SKIP_GAP`, e.g. 0.15) —
  L3's `select_primary_table`/`vet_primary` would pick the same anchor either way.
- Cap the candidates fed to the reranker (new config `RERANK_MAX_CANDIDATES`,
  e.g. 20 of the ~50 RRF output) — the tail never wins L3's anchor selection.
- GPU `RERANKER_DEVICE` in the prod inference process (config-only).
**Contract impact:** none — L2_RETRIEVAL's contract says L2b "re-scores all
candidates"; this makes that re-scoring conditional/bounded, same output shape
(`results` list with `final_score` set), same downstream L3 consumption.

### F3 — `ann_search` opens BEGIN/SET LOCAL/COMMIT per call 🟡 medium
**File:** `storage_adapters/reader.py::ann_search`.
Required for PgBouncer transaction-pool safety (`ef_search` can't be a
session-level SET). Now that Signal 1 fires by default, this cost applies on
every `retrieve()`, not conditionally.
**Fix:** if multiple ANN calls happen per query (Signal 1 + any graph-seed
lookup), batch them under one BEGIN/COMMIT instead of one per call.

### F4 — Fast-path & deterministic-branch hit-rate (L4_INTENT / L5_SQLGEN) 🟢 biggest lever
Per contract: the compiled-registry fast path and 6 of 8 SQL-generation paths
(existence, pre-aggregation, answer-entity, FK-value, value-arbiter,
temporal-only) are **LLM-free by contract** — only single-table-select and
join-skeleton-fill invoke the SLM.
**Fix (measurement-driven, not a code rewrite):**
- Instrument the fall-through ratio: % queries resolved by fast-path/verified-cache
  (zero retrieval, zero LLM) vs falling through to the full L2–L6 ladder.
- Extend the fast-path registry's covered patterns using real query logs — every
  query kept in a deterministic/fast path costs zero SLM latency, which is a
  bigger win than any L2 internal optimization.
- Tune the verified-cache similarity threshold (currently 0.85) against real
  repeat traffic.

### F5 — L7b NL-back answer blocks the response on the happy path 🟠 medium
**File:** `veda/execution.py` / `query/nl_answer.py`, called from `pipeline.py`
around `execute_sql`.
Per contract, NL-answer failure is already non-blocking (deterministic
row-count fallback). But on SUCCESS it still runs synchronously before the
result returns.
**Fix:** return `cols`/`rows`/`sql` immediately; generate the NL-back `answer`
asynchronously or as a fast follow-up call — a natural extension of the
contract's existing "best-effort" treatment of this stage.

### F6 — `save_verified_query` opens a fresh connection per save 🟡 low
**File:** `storage_adapters/reader.py` (the cache-write path — distinct from the
module-level `_connection()` used elsewhere).
**Fix:** reuse the cached `_connection()` instead of `psycopg2.connect(...)` per
successful, cacheable query. Low severity (off the answer-latency path per the
L7 contract) but a cheap fix.

---

## Task List

| ID | Fix | Layer | File | Effort | Impact |
|---|---|---|---|---|---|
| T1 | Per-signal timing instrumentation | L2 | new probe wrapping `retrieve()` | Low | unblocks measurement |
| T2 | Fast-path / deterministic fall-through ratio | L4/L5 | instrumentation | Low | unblocks F4 |
| T3 | Parallelize Signal 1 + 2 | L2 | `retrieval_engine_phase3.py` | Low | High |
| T4 | Merge Signal 3 & 4 | L2 | `retrieval_engine_phase3.py` | Low | Med |
| T5 | Signal-5 inverted index | L2 | `signal_builder.py` + engine init | Med | Med-High |
| T6 | Skip L2b rerank on unambiguous gap | L2 | `veda/pipeline.py` | Low | High |
| T7 | Cap reranker candidate width | L2 | `veda/pipeline.py` | Low | High |
| T8 | Batch ANN BEGIN/COMMIT per query | infra | `storage_adapters/reader.py` | Low | Med |
| T9 | Extend fast-path registry coverage | L4 | `query/fast_path.py` (or equivalent) | Med | High |
| T10 | Tune verified-cache threshold | L5 | `veda/cache.py` | Low | Med |
| T11 | Defer NL-back answer (non-blocking on success) | L7 | `pipeline.py` / `nl_answer.py` | Med | Med-High |
| T12 | Reuse `_connection()` in `save_verified_query` | L7 | `storage_adapters/reader.py` | Low | Low |
| T13 | Reranker GPU device (prod) | infra | config | Low (cfg) | High (prod) |
| T14 | vLLM backend for query SLM (prod) | infra | `slm/_call_slm.py` seam | Med | High (prod) |

## Suggested PRs
- **PR-1:** T1 + T2 (instrumentation — do first)
- **PR-2:** T3 + T4 + T6 + T7 (one `retrieve()`/`pipeline.py` L2 block, one owner —
  biggest win, since Signal 1 is real work in this build)
- **PR-3:** T5 + T8
- **PR-4:** T11 + T12 (L7 trims)
- **PR-5:** T9 + T10 (fast lanes, data-driven from T2)
- **PR-6:** T13 + T14 (prod config)

## Sequence
```
T1 + T2 (instrument)
  → T3 + T4 + T6 + T7  (L2, one PR — biggest win now that Signal 1 is on by default)
  → T5 + T8
  → T11 + T12  (L7 trims)
  → T9 + T10   (fast lanes, using T2's data)
  → T13 + T14  (prod)
```
