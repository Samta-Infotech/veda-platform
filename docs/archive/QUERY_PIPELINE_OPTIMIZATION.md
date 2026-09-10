# Query Pipeline Optimization — get NL query latency under 30s

**Status:** in progress · **Goal:** any `POST /api/v1/query` returns in **< 30s** on the
current on-prem (Docker-on-macOS, CPU-only containers + host Metal Ollama).

Owner: cross-source work. All changes uncommitted unless noted.

---

## 1. Problem (verified)

No query returns — every `POST /api/v1/query` times out (HTTP 000 / Django
`TimeoutError` after the 300s inference-client budget). Confirmed for both a heavy
cross-source query (`source_ids:[2,4]`) and a trivial single-source one (`source_ids:[4]`).
Scope threading itself is correct (`X-Veda-Source-Ids: 2,4` reaches inference).

## 2. Measured evidence

| Operation (measured in the inference container, CPU) | Time |
|---|---|
| SLM SQL-gen `[L5]`, Qwen-7B on **container ollama (CPU)** | **239s** / call |
| SLM prewarm on **host Metal ollama** | **0.65s** |
| BGE-M3 encode 60 **short** texts | 1.9s |
| BGE-M3 encode 60 **long (512-tok)** texts | **98.0s** |
| Cross-encoder rerank 60 **long** pairs | **101.4s** |
| BGE-M3 cold model load (first request) | 22.4s |

Live query log showed `Inference Embeddings: 12/60 [06:37<33:42, 42s/it]` — a 60-item
encode pass estimated at ~45 min. Root cause is **text length × candidate count on CPU**,
not thread contention.

## 3. Root causes (ranked)

1. **SLM on CPU** — inference used `OLLAMA_URL=http://ollama:11434` (in-container, no GPU
   passthrough on macOS) → 239s/call. **[FIXED]** repointed to
   `http://host.docker.internal:11434` (host Metal, model resident in VRAM).
2. **Query-time re-encode + rerank of 60–80 long-text candidates** (~200s):
   - `sparse_ranker` re-encodes candidate column texts live (BGE-M3 sparse) — worsened if
     the precomputed sparse index (`column_sparse_v1`) is empty for the scope.
   - Cross-encoder reranks `BIENCODER_CANDIDATE_COLS=80` pairs at `RERANKER_MAX_TEXT_LEN=512`.
3. **`WORKERS=1`** — one slow request blocks the whole tier (later requests queue + time out).
4. **No model pre-warm** — first request pays cold model load (BGE-M3 22s + reranker).
5. **No auto-federation** — query-router never routes to the federated composer (functional
   gap, not latency), so cross-source questions answer single-source.

## 4. Latency budget target

| Stage | Now | After A | After B |
|---|---|---|---|
| SLM (Metal, warm) | 239s→ | ~3–10s | ~3–10s |
| Candidate sparse encode | ~98s | ~8s (fewer+shorter) | ~0s (precomputed index) |
| Cross-encoder rerank | ~101s | ~8s (16 cands, 128 tok) | ~5s |
| Dense query encode + pgvector | ~0.5s | ~0.5s | ~0.5s |
| Graph/BM25/value/SQL exec | ~2–3s | ~2–3s | ~2–3s |
| **Total** | **>7 min** | **~20–28s** | **~12–18s** |

---

## Phase A — config/param quick wins (get to ~20–28s)

Files: `veda_core/config.py`, `docker-compose.yml`, inference startup.

- **A1. SLM → host Metal ollama.** `docker-compose.yml` inference service:
  `OLLAMA_URL: http://host.docker.internal:11434`. **[DONE]**
- **A2. Shrink reranker text.** `RERANKER_MAX_TEXT_LEN` 512 → **160**. Cross-encoder cost is
  ~quadratic in length; this is the single biggest rerank win.
- **A3. Fewer candidates.** `BIENCODER_CANDIDATE_COLS` 80 → **24**; keep
  `RERANKER_TOP_COLS=15`. Fewer texts to sparse-encode AND fewer rerank pairs.
- **A4. Trim enriched text** used by the reranker (col + sampled values) so each candidate
  text is short before the 160-char cap (fewer sampled values).
- **A5. Cap torch threads** to physical cores to avoid oversubscription across the
  BGE-M3 / reranker / sparse models (`OMP_NUM_THREADS`, `torch.set_num_threads`).
- **A6. Warm models at inference startup** — prewarm BGE-M3 dense+sparse, the reranker, and
  the SLM (`keep_alive`) so the first real query is hot.
- **A7. Safety net** — `INFERENCE_TIMEOUT_S` stays 300 (goal is to be well under it).

## Phase B — structural (durable < 18s)

- **B1. Use the precomputed sparse index** instead of live candidate re-encode. Verify
  `column_sparse_v1` is populated for sources 2/4/5; if empty, (re)build it so
  `sparse_ranker` scores against stored vectors and never calls `encode_sparse` on
  candidates at query time.
- **B2. Confirm the dense path uses stored vectors only** (query-only encode + pgvector
  cosine) — already the case in `retrieval_v2`; keep it that way.
- **B3. `WORKERS` ≥ 2** (or async) so a single slow query can't block the tier; size from
  measured RSS.
- **B4. GPU for the torch models (the reranker long-pole).** Docker-on-macOS has no GPU
  passthrough, so either (a) run the inference service **natively on the host** with an
  MPS/Metal torch, or (b) stand up a **host-native embedding+rerank HTTP service** (mirrors
  how Ollama already serves the SLM fast) and have the container call it. Documented here;
  gated on deployment decision. On real GPU/prod (CUDA) this is automatic.

## Phase C — correctness: auto-federation

- **C1. Wire the query router → `cross_source_composer.compose_federated`** so a cross-source
  question (`should_federate(selected_columns)` true) executes the federated join via
  `federated_executor` and composes SQL result + evidence, instead of answering
  single-source. Uses the already-verified executor + composer.

---

## Execution log

- [x] **A1** SLM → host Metal ollama (`docker-compose.yml` inference `OLLAMA_URL`). SQL-gen 239s → **8.7s**.
- [x] **A2** `RERANKER_MAX_TEXT_LEN` 512 → 160.
- [x] **A3** `BIENCODER_CANDIDATE_COLS` 80 → 24.
- [~] **A4** enriched-text: reranker already uses precomputed text capped by A2 — no separate change needed.
- [—] **A5** thread cap: SKIPPED — measurement showed *text length*, not thread contention, was the cost; capping would slow encodes.
- [x] **A6** startup warmup in `inference/loaders.py` (BGE-M3 + SLM; reranker warm falls back to lazy-load).
- [x] **GUARD (critical, was the hang)** `retrieval_engine_phase3.py` + `SPARSE_FIT_MAX_DOCS=300`: query-time
  `sparse_ranker.fit()` no longer live-encodes 1902 long `retrieval_documents` (~50 min). Degrades to
  dense+FK+value when the persisted sparse index is missing for the scope.
- [ ] **B1** populate `column_sparse_v1` for sources 4/5 — deferred: low value until multi-source serving
  (below) lands, since tabular columns aren't in the query-time semantic model anyway.
- [—] **B3** `WORKERS`: keep 1 on this box (one query gets all cores = fastest single-query). Raise on GPU/multi-core prod.
- [ ] **B4** GPU/native for BGE-M3 + reranker — documented; needs deployment decision (host-native MPS or host embed service).
- [ ] **C1** auto-federation wiring — blocked on multi-source serving (below).

### Measured results (2026-07-09, warm)
| Path | Wall time |
|---|---|
| Deterministic (source 2) | **0.55s** ✅ |
| LLM-IR fallback (source 2) | **25.5s** ✅ (returns http 200 < 30s) |

### Newly-surfaced deeper issue (beyond latency) — multi-source query serving
The query tier loads a SINGLE global semantic model (homzhub / source 2). Sources 3/4/5 are NOT in the
query-time model, so a `source_ids:[4]` query still retrieves + validates against homzhub columns and the
firewall rejects the real source-4 columns (`amount/status/ticket_id`). This is why tabular-scoped and
cross-source NL queries can't answer yet. Fixing it (assemble a scoped multi-source semantic model at query
time; then B1 sparse for 4/5; then C1 federation) is the remaining track — larger than this perf pass.

### Follow-up perf (optional, for margin under the LLM-IR path's 25.5s)
- Cache the per-query `[ValueSampler] Value index rebuilt: 5054 terms` (rebuilt on the LLM-IR path).
- Cap / reduce LLM-IR fallback SLM round-trips.
