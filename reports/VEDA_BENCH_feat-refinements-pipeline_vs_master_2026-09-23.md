# VEDA benchmark — `feat/refinements-pipeline` vs `master`

**Date:** 2026-09-23 · **Author:** automated audit, all numbers measured on this machine

Every latency, token and call count below comes from a run executed for this report.
Nothing is taken from docs, changelogs or commit messages. Where something could not be
measured it says **not measured** and why. Raw logs and the harness are in
[`reports/raw/`](raw/).

---

## 0. Setup

### 0.1 Code under test

| | value |
|---|---|
| branch | `feat/refinements-pipeline` |
| branch HEAD | `f768040d79a7f905b42bcc13c230c28dafd75512` |
| master HEAD | `7c5368f4d441dee1e12c6b59c4d81717199d67aa` |
| merge-base | `7c5368f4d441dee1e12c6b59c4d81717199d67aa` (identical to master — the branch is a pure fast-forward set on top of master) |
| commits on branch | 16 non-merge |
| diff | 217 files, +23,471 / −2,091 |
| working tree | clean at start; **unchanged by this audit** — master was read through `git worktree add /Users/ekesel/samta/veda-master-bench master`, never by checking out a branch in the working tree |

### 0.2 How master was run

The containers bind-mount `.:/app` (`docker-compose.yml:245`), so the running services
execute the working tree. To benchmark master without touching that tree, master was
checked out as a **separate git worktree** and mounted into one-off containers:

```
docker compose run --rm --no-deps -v /Users/ekesel/samta/veda-master-bench:/app \
    --entrypoint python inference -u /bench/bench_engine.py --label master ...
```

`veda_core/data/` (120 MB of derived ingestion artifacts — semantic model, join paths,
graphs, glossary) is gitignored, so it was **copied** into the master worktree rather than
mounted; a nested bind at `/app/veda_core/data` proved to be shadowed by `/app` in
detached containers and silently produced an empty directory. Both trees therefore read
byte-identical artifacts.

Both trees ran through the **same** `.env`, the **same** Postgres/pgvector/Redis instances,
the **same** host SLM and the **same** host Metal embed/rerank server. Each run was a fresh
Python process from the same image.

**Documented limitation.** The ingested corpus in `veda_engine` was produced by the
*branch's* ingestion code (`sources_source.last_ingested_at` = 2026-09-22). Master is
therefore measured reading branch-ingested artifacts. Re-ingesting under master was not
attempted (multi-hour). This is stated rather than hidden; it affects retrieval inputs
equally for every query.

### 0.3 Configuration actually in force

`.env` was identical for both runs — it is the compose `env_file` and was never edited.
Recorded into every raw log line (`meta.env`):

| key | value |
|---|---|
| `SLM_MODEL_NAME` | `qwen2.5:7b-instruct` |
| `SLM_BACKEND` | `ollama` |
| `OLLAMA_URL` | `http://192.168.1.35:11500` |
| `METAL_EMBED_URL` | `http://192.168.1.39:11435` |
| `NL_SUMMARY_MODEL` / `CHATBOT_CLASSIFY_MODEL` | `qwen2.5:7b-instruct` |
| `INFERENCE_WORKERS` | `1` |
| `MULTISOURCE_ROUTING_ENABLED` / `_SHADOW` | `1` / `1` |
| `VEDA_SM_REDIS` | `1` |

#### 0.3.1 Declared-but-never-read env keys (verified by grep over all Python)

The task asked this to be checked explicitly. There is **no generic `VEDA_` settings
bridge** — `grep -rn 'startswith("VEDA_")'` over the repo returns nothing — so a `VEDA_*`
key only works if some module reads that exact name.

| `.env` key | read by code? | what actually governs |
|---|---|---|
| `VEDA_TOP_K=15` | **no** | `veda_core/config.py:330` `TOP_K = 15` (literal) |
| `VEDA_TOP_K_TO_LLM=6` | **no** | `veda_core/config.py:415` `TOP_K_TO_LLM = 6` (literal) |
| `VEDA_ENCODER_MODE=ensemble` | **no** | — |
| `VEDA_FAST_PATH_ENABLED=true` | **no** | `veda_core/config.py:2546` `FAST_PATH_ENABLED = True` (literal) |
| `VEDA_IR_JOIN_FREE_ENABLED=true` | **no** | `veda_core/config.py:1657` `IR_JOIN_FREE_ENABLED = True` (literal) |
| `VEDA_QUERY_DECOMPOSE_ENABLED=false` | **no** | `veda_core/config.py:2683` `QUERY_DECOMPOSE_ENABLED = False` (literal) |
| `VEDA_HNSW_M=16` | **no** | — |
| `VEDA_HNSW_EF_CONSTRUCTION=200` | **no** | — |
| `VEDA_HNSW_EF_SEARCH=40` | yes | `storage_adapters/reader.py:278` |
| `SLM_TEMPERATURE=0` | **no (see below)** | `veda_core/config.py:381` `SLM_TEMPERATURE = 0.3` (literal) |

Today every unread key's `.env` value happens to equal the hardcoded literal, so nothing
behaves differently — but changing any of them in `.env` would have no effect at all.

**`SLM_TEMPERATURE` is the exception that does bite.** `veda_core/config.py:381` sets
`SLM_TEMPERATURE = 0.3` as a literal, and that constant is passed straight into three
query-path SLM calls:

- `veda_core/query/rag_layer.py:311` — `purpose="rag_synthesis"` (every document answer)
- `veda_core/query/slm_layer.py:596` — `purpose="ir_emit"` (the LLM SQL-IR generator)
- `veda_core/query/lg_nodes.py:126`

The only code that reads the `SLM_TEMPERATURE` *environment variable* is
`storage_adapters/env_drift.py:24`, a drift checker that compares `.env` against
`os.environ` and never feeds the SLM. So `SLM_TEMPERATURE=0` in `.env` is inert, and
document synthesis and LLM SQL generation run at **temperature 0.3 on both trees**. This
contradicts the note in `CLAUDE.md` and in `.env` itself, which claim the key was added to
make runs deterministic. It is corroborated by the runs: FS2's answer text differs between
repetitions of the same query on the same tree (`reports/raw/engine_branch.jsonl`).

### 0.4 Host topology

| component | where | detail |
|---|---|---|
| SLM (query + summary) | host `192.168.1.35:11500`, Ollama | `qwen2.5:7b-instruct`, Q4_K_M default tag quantization; `keep_alive="24h"` (`veda_core/slm/_call_slm.py:271`) |
| embeddings + reranker | host `192.168.1.39:11435`, `scripts/metal_embed_server.py`, `device=mps` | BGE-M3 dense+sparse, cross-encoder rerank |
| engine store | `pgvector/pgvector:pg16` in Docker, host port 15432, DB `veda_engine`, via PgBouncer (transaction pooling, `default_pool_size=20`) | `doc_chunks`, `column_embeddings_v2`, `source_item_embeddings`, … |
| Django substrate | same Postgres, DB `veda` | |
| source DB (homzhub) | homebrew postgresql@17 on host `:5432` | reached as `host.docker.internal` |
| inference tier | `veda-platform-inference-1`, uvicorn | `INFERENCE_WORKERS=1` |

`OLLAMA_NUM_PARALLEL` is **not set** anywhere in `.env`, `docker-compose.yml` or the repo —
the SLM host runs on its own default. **Not measured:** the remote host's parallelism
setting was not read (it is not this machine).

**Warm vs cold.** Both hosts were already warm and stayed warm throughout (models pinned by
`keep_alive="24h"`). Each benchmark process ran **2 throwaway warm-up queries** before any
timing; those are recorded in the raw logs as `kind_row: "warmup"` and are excluded from
every table.

### 0.5 Harness and what it patches

`reports/raw/bench_engine.py` drives `veda_hybrid.run_hybrid_query` directly. Everything it
patches was verified to exist byte-identically on both trees before use:

| patch | why |
|---|---|
| `veda.pipeline.verified_cache_lookup` / `save_verified_query` → no-ops | master has **no** `cache_back` kill switch (`veda_core/context.py` has no such field; `veda_core/veda/pipeline.py:690` calls the cache unconditionally, vs branch `pipeline.py:796` which honours `not _cache_back`). Without this, master's reps 2–3 would replay a cached SQL and look artificially fast. |
| `ExplainTrace.finalize` → captures `to_dict()` | per-stage timings and the SLM ledger, non-verbose (no extra recording cost) |
| `ingestion.m3_encoder._metal_post` | counts + times every embedding round-trip |
| `query.reranker._RemoteReranker.predict` | counts + times every rerank round-trip |
| `psycopg2.connect` + counting cursor | DB connects and queries |
| `urllib.request.urlopen` for POSTs to `OLLAMA_URL` | **transport-level SLM truth** — see §2.1 |

`RequestContext` is built reflectively from `dataclasses.fields()`, so the same harness runs
on master (which has neither `cache_back` nor `session_prior`).

---

## 1. Query set

Two per source type, identical text on both trees. Full reproducible set in
[Appendix A](#appendix-a--query-set).

| id | category | question | scope | expected shape |
|---|---|---|---|---|
| DB1 | Database / SQL | latest 10 payment transactions | source 2 (homzhub, relational) | `ORDER BY` a date column, `LIMIT 10`, on the payment-transaction table |
| DB2 | Database / SQL | users created last month | source 2 | `WHERE` on a created-at column over last month, on `users_user` |
| DL1 | Data Lake | total maintenance amount per category | source 4 (`invoices_csv`) | `SUM(amount) GROUP BY category` on `maintenance` |
| DL2 | Data Lake | average monthly fee per category | source 5 (`catalog_parquet`) | `AVG(monthly_fee) GROUP BY category` on `amenities_catalog` |
| FS1 | File System / RAG | what is the late fee percentage | full ready scope (2,3,4,5) | RAG answer naming 2 percent, cited to `msa_green_tower` |
| FS2 | File System / RAG | summarize the employee handbook leave policy | full ready scope | RAG answer naming leave types, cited to the Employee Handbook |
| XS1 | Cross-source | which vendors operate in cities where we have assets | full ready scope | `vendors@4` joined to `assets_asset.city_name@2` |
| XS2 | Cross-source | total maintenance amount spent per city where we own assets | full ready scope | `SUM(maintenance.amount)@4` grouped by asset city@2 |
| FU1 | Follow-up | *t1* how many maintenance records are there → *t2* **how many of those are in the Repair category** | chat session, scope 2,3,4,5, t1 pinned to 4 | filtered count on source 4, below the t1 total |
| FU2 | Follow-up | *t1* how many properties are there → *t2* **how many of those are in Mumbai** | chat session, scope 2,3,4,5, t1 pinned to 2 | filtered count on source 2, below the t1 total |

XS1/XS2 are taken verbatim from `evaluation/golden_cross_source.jsonl`, which documents the
real join edges (`maintenance.asset_id@4 → assets_asset.id@2`, `vendors.city@4 →
assets_asset.city_name@2`).

**Why FS1/FS2 run at full scope.** Pinning a document question to source 3 alone trips the
permission pre-check in `veda_core/veda_hybrid.py:860-905`, which treats every other ready
source as "denied" and refuses with *"You don't have permission to access this data"*. That
case is measured separately as probes FS1P/FS2P (§8.2, R1); the timed FS rows use the scope
the product actually serves document questions in — the same context
`veda_core/doc_bench.py:81` uses. Pinned, **master answers FS1 correctly** (*"2 percent per
month"*, cited `msa_green_tower.pdf`, 6.4 s); at full scope it does not.

**Follow-up turns** run through `chatbot.run.run_chat_turn` (the entry point
`apps/chat/services.py` uses per user turn), against a master-code inference container
(`veda-master-inference`) for the master side, so both tiers come from the same tree. Turn 1
is pinned for the reason `scripts/eval_sessions.py` documents: this deployment cannot answer
an unpinned multi-source *opening* turn from the chat path at all.

---

## 2. Method notes that change how the numbers read

### 2.1 Master's own trace under-reports its SLM calls

`veda_core/query/answer_entity.py::_llm_relation_word` on master does a **hand-rolled
`urllib` POST** to `{SLM_OLLAMA_BASE_URL}/api/chat`
(`/Users/ekesel/samta/veda-master-bench/veda_core/query/answer_entity.py:113-121`),
bypassing `call_slm()` entirely. It therefore never reaches the SLM ledger and its tokens
are never counted. The branch routes the same call through `call_slm(...)`
(`veda_core/query/answer_entity.py:112-119`, with `num_predict=8, timeout=8`).

Counting only `call_slm` would report master as making **fewer** SLM calls than it really
does. Every count in Table B is therefore **transport-measured**: the harness intercepts
every POST to `OLLAMA_URL` and reads Ollama's own `prompt_eval_count` / `eval_count`.

Measured gap between what each tree *claims* and what the wire *shows*:

| query | master ledger | master wire | branch ledger | branch wire |
|---|---|---|---|---|
| DB1 | 1 call / 1,221 tok | **2 calls / 1,325 tok** | 2 calls / 1,416 tok | 2 calls / 1,416 tok |
| DB2 | 0 calls / 0 tok | **1 call / 102 tok** | 1 call / 102 tok | 1 call / 102 tok |
| DL2 | 0 calls / 0 tok | **1 call / 103 tok** | 1 call / 850 tok | 1 call / 850 tok |

The branch's ledger matches the wire exactly on all 8 queries; master's does not on 3 of 8.

### 2.2 Run-to-run noise is large — most latency deltas do not resolve at n=3

Two independent full master runs, **identical code, identical config, minutes apart**:

#### Measurement noise — two independent master runs, same code, same config

| Q | master run A med | master run B med | spread | spread % |
|---|---:|---:|---:|---:|
| DB1 | 10,290 | 16,450 | 6,160 | 60% |
| DB2 | 5,102 | 5,278 | 176 | 3% |
| DL1 | 6,500 | 10,691 | 4,191 | 64% |
| DL2 | 1,962 | 2,652 | 690 | 35% |
| FS1 | 25,220 | 23,407 | 1,813 | 8% |
| FS2 | 11,898 | 20,087 | 8,189 | 69% |
| XS1 | 10,792 | 21,547 | 10,754 | 100% |
| XS2 | 16,179 | 24,420 | 8,242 | 51% |
| **sum** | **87,943** | **124,533** | 36,590 | 42% |



The sum of medians for the same code differs by **42%**, and XS1 alone by **100%**. The host
is shared with other work and the SLM/embed hosts are on the LAN. Consequently this report
treats a latency delta as **real only when the three branch runs and the three master runs
do not overlap at all**; everything else is reported as not resolved.

Criterion: the three branch runs and the three master runs do not overlap at all
(branch min > master max, or branch max < master min). Anything overlapping is
reported as NOT RESOLVED at n=3 — the two independent master runs of identical code
differ by up to 100% on a single query, so medians alone prove nothing.

| Q | master min..max | branch min..max | separated? | direction |
|---|---|---|---|---|
| DB1 | 15,548..16,729 | 11,140..16,169 | no (overlapping) | — |
| DB2 | 5,171..5,441 | 5,313..6,506 | no (overlapping) | — |
| DL1 | 10,380..12,186 | 8,576..11,800 | no (overlapping) | — |
| DL2 | 2,006..6,780 | 10,942..15,083 | **YES** | branch SLOWER |
| FS1 | 22,521..32,602 | 1,578..6,636 | **YES** | branch FASTER |
| FS2 | 19,159..20,364 | 11,658..19,726 | no (overlapping) | — |
| XS1 | 18,477..23,076 | 4,442..8,154 | **YES** | branch FASTER |
| XS2 | 23,180..25,539 | 32,597..43,584 | **YES** | branch SLOWER |



Only 4 of 8 queries resolve. Two of those "branch faster" results (FS1, XS1) are the branch
**declining to answer** — see Table D. A faster non-answer is not a latency win.

### 2.3 Stage timings are start-offset gaps, not true spans

`stage_durations_ms` is built by `ExplainTrace._build_totals`
(`veda_core/veda/explain.py:185-199`) from each section's first-touch offset to the next
section's — the engine's own documented method. A stage that does not re-touch its section
absorbs everything that follows it, which is why `execution` on DB1 and `output` on DL1
appear to hold the `nl_answer` SLM call. Read Table C together with the SLM ledger.

### 2.4 Time to first byte

`run_hybrid_query(..., on_event=cb)` emits the lifecycle timeline. TTFB (first event) is
**0 ms on every query, both trees** — the `received` event fires synchronously before any
work, so it is not a useful discriminator. The meaningful number is time to first
*content*-bearing event, reported in §3.

---

## 3. Comparison tables

### TABLE A — latency (ms), master vs branch

| Q | Category | master med | master p90 | branch med | branch p90 | Δ med | Δ med % | Δ p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DB1 | Database / SQL | 16,450 | 16,673 | 12,815 | 15,498 | -3,636 | -22.1% | -1,175 |
| DB2 | Database / SQL | 5,278 | 5,409 | 5,874 | 6,380 | +596 | +11.3% | +971 |
| DL1 | Data Lake | 10,691 | 11,887 | 9,016 | 11,243 | -1,675 | -15.7% | -644 |
| DL2 | Data Lake | 2,652 | 5,954 | 13,290 | 14,724 | +10,638 | +401.1% | +8,770 |
| FS1 | File System / RAG | 23,407 | 30,763 | 2,074 | 5,724 | -21,334 | -91.1% | -25,039 |
| FS2 | File System / RAG | 20,087 | 20,308 | 13,001 | 18,381 | -7,086 | -35.3% | -1,927 |
| XS1 | Cross-source | 21,547 | 22,770 | 4,472 | 7,418 | -17,074 | -79.2% | -15,352 |
| XS2 | Cross-source | 24,420 | 25,316 | 40,255 | 42,918 | +15,834 | +64.8% | +17,603 |
| **sum of medians** | | **124,533** | | **100,797** | | -23,737 | -19.1% | |


Read Table A with §2.2: only DL2, FS1, XS1 and XS2 separate from noise. The `sum of medians`
row is shown because it was asked for, but it is dominated by FS1/XS1, which the branch does
not answer. The honest latency comparison is the parity set:

#### Latency restricted to queries where BOTH trees produced an ANSWER

Parity set (both answered on all 3 reps): DB1, DB2, DL1, FS2, XS2

| Q | master min/med/p90 | branch min/med/p90 | Δ med | Δ med % |
|---|---|---|---:|---:|
| DB1 | 15,548 / 16,450 / 16,673 | 11,140 / 12,815 / 15,498 | -3,636 | -22.1% |
| DB2 | 5,171 / 5,278 / 5,409 | 5,313 / 5,874 / 6,380 | +596 | +11.3% |
| DL1 | 10,380 / 10,691 / 11,887 | 8,576 / 9,016 / 11,243 | -1,675 | -15.7% |
| FS2 | 19,159 / 20,087 / 20,308 | 11,658 / 13,001 / 18,381 | -7,086 | -35.3% |
| XS2 | 23,180 / 24,420 / 25,316 | 32,597 / 40,255 / 42,918 | +15,834 | +64.8% |
| **sum of medians** | **76,927** | **80,961** | +4,034 | +5.2% |



On the parity set the branch is **+5.2% slower on the sum of medians**, and all of that is
XS2 (+15.8 s); DB1, DB2, DL1 and FS2 all overlap with master and do not resolve.

#### Time to first content (ms, median rep)

| Q | master first-content | master completed | branch first-content | branch completed |
|---|---:|---:|---:|---:|
| DB1 | 2,868 | 16,450 | 3,087 | 12,815 |
| DB2 | 5,255 | 5,278 | 5,852 | 5,874 |
| DL1 | 594 | 10,691 | 449 | 9,016 |
| DL2 | 2,651 | 2,652 | 584 | 13,290 |
| FS1 | 23,405 | 23,407 | 2,071 | 2,074 |
| FS2 | 2,214 | 20,087 | 432 | 13,001 |
| XS1 | 21,546 | 21,547 | 4,471 | 4,472 |
| XS2 | 24,418 | 24,420 | 40,251 | 40,255 |

### TABLE B — SLM calls and tokens (median rep), master vs branch
Counts are TRANSPORT-measured (every POST to the SLM host), not ledger-derived:
master's query/answer_entity.py bypasses call_slm(), so its own trace under-reports.
`ledger` shows what each tree's own trace claims, for comparison.

| Q | master calls | master pt | master ct | master tot | master ledger | branch calls | branch pt | branch ct | branch tot | branch ledger | Δ calls | Δ tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DB1 | 2 | 1,263 | 62 | 1,325 | 1c/1,221t | 2 | 1,373 | 43 | 1,416 | 2c/1,416t | +0 | +91 |
| DB2 | 1 | 100 | 2 | 102 | 0c/0t | 1 | 100 | 2 | 102 | 1c/102t | +0 | +0 |
| DL1 | 1 | 827 | 48 | 875 | 1c/875t | 1 | 823 | 49 | 872 | 1c/872t | +0 | -3 |
| DL2 | 1 | 101 | 2 | 103 | 0c/0t | 1 | 766 | 84 | 850 | 1c/850t | +0 | +747 |
| FS1 | 3 | 1,452 | 148 | 1,600 | 3c/1,600t | 1 | 102 | 2 | 104 | 1c/104t | -2 | -1,496 |
| FS2 | 1 | 1,116 | 115 | 1,231 | 1c/1,231t | 1 | 1,116 | 135 | 1,251 | 1c/1,251t | +0 | +20 |
| XS1 | 3 | 1,266 | 94 | 1,360 | 3c/1,360t | 1 | 377 | 29 | 406 | 1c/406t | -2 | -954 |
| XS2 | 3 | 1,513 | 145 | 1,658 | 3c/1,658t | 5 | 2,319 | 234 | 2,553 | 5c/2,553t | +2 | +895 |
| **total** | **15** | | | **8,254** | **13** | | | **7,554** | **-2** | **-700** |



**Where the token delta comes from.** Ignoring FS1/XS1 (branch does not answer), the branch
spends **+895 tokens and +2 SLM calls on XS2** and **+747 on DL2** (master spends 103 there
because it clarifies without generating prose). On the true parity set:

#### Token cost restricted to the parity set (transport-measured)

| Q | master tok | branch tok | Δ | master calls | branch calls |
|---|---:|---:|---:|---:|---:|
| DB1 | 1,325 | 1,416 | +91 | 2 | 2 |
| DB2 | 102 | 102 | +0 | 1 | 1 |
| DL1 | 875 | 872 | -3 | 1 | 1 |
| FS2 | 1,231 | 1,251 | +20 | 1 | 1 |
| XS2 | 1,658 | 2,553 | +895 | 3 | 5 |
| **total** | **5,191** | **6,194** | +1,003 | | |



### TABLE C — per-stage wall time (ms, median rep)


**DB1** (Database / SQL) — total master 16,450 / branch 12,815 ms

| stage | master | branch | Δ |
|---|---:|---:|---:|
| query_understanding | 1,549 | 2 | -1,546 |
| routing | 0 | 0 | +0 |
| retrieval_health | 0 | 1,160 | +1,160 |
| rrf | 3 | 7 | +3 |
| graph_expansion | 344 | 370 | +26 |
| reranking | 132 | 10 | -121 |
| entity_resolution | 0 | 76 | +76 |
| join_planning | 413 | 788 | +375 |
| sql_planning | 3 | 3 | +0 |
| firewall | 0 | 159 | +159 |
| validation | 16 | 0 | -16 |
| execution | 13,562 | 9,707 | -3,854 |
| execution_plan | 0 | 1 | +1 |
| source_execution | 409 | 0 | -409 |
| summary | 0 | 0 | +0 |
| nl_summary | 0 | 2 | +2 |
| visualization | 0 | 2 | +2 |
| output | 11 | 12 | +1 |
| slm | 3 | 0 | -2 |
| lifecycle | 0 | 510 | +510 |
| explainability | 1 | 1 | +1 |
| projection | 5 | 0 | -5 |

**DB2** (Database / SQL) — total master 5,278 / branch 5,874 ms

| stage | master | branch | Δ |
|---|---:|---:|---:|
| query_understanding | 3,486 | 1 | -3,486 |
| routing | 0 | 0 | +0 |
| retrieval | 0 | 0 | -0 |
| retrieval_health | 0 | 3,381 | +3,381 |
| rrf | 6 | 5 | -1 |
| graph_expansion | 322 | 290 | -32 |
| reranking | 20 | 11 | -9 |
| entity_resolution | 0 | 0 | +0 |
| join_planning | 894 | 1,455 | +561 |
| sql_planning | 6 | 10 | +4 |
| firewall | 0 | 118 | +118 |
| validation | 16 | 0 | -16 |
| execution | 3 | 3 | +0 |
| execution_plan | 0 | 1 | +1 |
| source_execution | 0 | 0 | +0 |
| summary | 4 | 1 | -3 |
| output | 14 | 17 | +2 |
| slm | 0 | 6 | +6 |
| lifecycle | 505 | 573 | +68 |
| entity_selection | 0 | 0 | -0 |
| explainability | 0 | 0 | -0 |

**DL1** (Data Lake) — total master 10,691 / branch 9,016 ms

| stage | master | branch | Δ |
|---|---:|---:|---:|
| query_understanding | 115 | 11 | -104 |
| routing | 0 | 0 | +0 |
| schema_linking | 0 | 1 | +0 |
| firewall | 0 | 5 | +5 |
| validation | 4 | 0 | -4 |
| execution | 0 | 0 | +0 |
| execution_plan | 0 | 16 | +16 |
| source_execution | 475 | 417 | -58 |
| result_analysis | 0 | 0 | +0 |
| nl_summary | 0 | 5 | +5 |
| visualization | 2 | 3 | +1 |
| output | 10,094 | 8,555 | -1,539 |
| slm | 0 | 1 | +1 |
| explainability | 0 | 1 | +1 |

**DL2** (Data Lake) — total master 2,652 / branch 13,290 ms

| stage | master | branch | Δ |
|---|---:|---:|---:|
| query_understanding | 89 | 130 | +41 |
| routing | 0 | 0 | +0 |
| rrf | 16 | 0 | -16 |
| graph_expansion | 277 | 0 | -277 |
| reranking | 4 | 0 | -4 |
| schema_linking | 992 | 1 | -991 |
| join_planning | 849 | 0 | -849 |
| sql_planning | 6 | 0 | -6 |
| firewall | 0 | 41 | +41 |
| validation | 6 | 0 | -6 |
| execution | 0 | 0 | +0 |
| execution_plan | 0 | 19 | +19 |
| source_execution | 0 | 392 | +392 |
| nl_summary | 0 | 3 | +3 |
| visualization | 0 | 2 | +2 |
| output | 1 | 12,698 | +12,698 |
| slm | 0 | 1 | +1 |
| lifecycle | 411 | 0 | -411 |
| entity_selection | 0 | 0 | -0 |
| explainability | 0 | 0 | +0 |

**FS1** (File System / RAG) — total master 23,407 / branch 2,074 ms

| stage | master | branch | Δ |
|---|---:|---:|---:|
| query_understanding | 0 | 37 | +37 |
| routing | 0 | 22 | +22 |
| retrieval_health | 0 | 293 | +293 |
| rrf | 0 | 0 | +0 |
| graph_expansion | 0 | 312 | +312 |
| reranking | 0 | 15 | +15 |
| entity_resolution | 0 | 0 | +0 |
| join_planning | 0 | 804 | +804 |
| sql_planning | 0 | 67 | +67 |
| firewall | 0 | 20 | +20 |
| execution_plan | 0 | 0 | +0 |
| source_execution | 0 | 502 | +502 |
| output | 0 | 2 | +2 |
| slm | 11,414 | 0 | -11,414 |
| lifecycle | 11,982 | 0 | -11,982 |
| federation | 8 | 0 | -8 |

**FS2** (File System / RAG) — total master 20,087 / branch 13,001 ms

| stage | master | branch | Δ |
|---|---:|---:|---:|
| routing | 0 | 0 | +0 |
| execution_plan | 0 | 12,733 | +12,733 |
| source_execution | 0 | 253 | +253 |
| slm | 8 | 5 | -3 |
| lifecycle | 20,076 | 0 | -20,076 |

**XS1** (Cross-source) — total master 21,547 / branch 4,472 ms

| stage | master | branch | Δ |
|---|---:|---:|---:|
| query_understanding | 0 | 0 | +0 |
| routing | 0 | 3,094 | +3,094 |
| retrieval_health | 0 | 332 | +332 |
| rrf | 0 | 18 | +18 |
| graph_expansion | 0 | 326 | +326 |
| reranking | 0 | 15 | +15 |
| entity_resolution | 0 | 42 | +42 |
| join_planning | 0 | 296 | +296 |
| output | 0 | 0 | +0 |
| slm | 12,116 | 65 | -12,051 |
| lifecycle | 9,426 | 283 | -9,143 |
| federation | 2 | 0 | -2 |
| entity_selection | 0 | 0 | +0 |

**XS2** (Cross-source) — total master 24,420 / branch 40,255 ms

| stage | master | branch | Δ |
|---|---:|---:|---:|
| routing | 0 | 8,382 | +8,382 |
| firewall | 0 | 5,641 | +5,641 |
| slm | 19,853 | 25,918 | +6,064 |
| lifecycle | 4,558 | 287 | -4,271 |
| federation | 6 | 14 | +8 |
| federated | 0 | 9 | +9 |


### TABLE D — correctness parity

| Q | master status | branch status | master answer/SQL | branch answer/SQL | divergence |
|---|---|---|---|---|---|
| DB1 | answered | answered | `SELECT "created_at", "payment_transaction_id", "updated_at" FROM "reminders_reminderpaymen` | `SELECT "created_at", "other_payment_type", "late_fee", "payment_date", "payment_id", "paym` | SAME |
| DB2 | answered | answered | `SELECT "created_at", "last_name", "last_logout", "last_login", "first_name", "emergency_co` | `SELECT "created_at", "last_name", "last_logout", "last_login", "first_name", "emergency_co` | SAME |
| DL1 | answered | answered | `SELECT "a"."category" AS "category", SUM("a"."amount") AS "sum_amount", COUNT(*) AS "row_c` | `SELECT "a"."category" AS "category", SUM("a"."amount") AS "sum_amount", COUNT(*) AS "row_c` | SAME |
| DL2 | clarify | answered | — | `SELECT "a"."category" AS "category", AVG("a"."monthly_fee") AS "avg_monthly_fee", COUNT(*)` | **DIFFERENT** |
| FS1 | answered | refused | `SELECT t1."asset_id" AS asset, SUM(t0."price_percent") AS "late_fee_percentage" FROM src_2` | I couldn't find any data relevant to this question in the sources available to you. | **DIFFERENT** |
| FS2 | answered | answered | The leave policy requires employees to follow specific procedures for requesting leave. Fo | The leave policy requires employees to follow specific procedures for requesting leave, wi | SAME |
| XS1 | answered | clarify | `SELECT DISTINCT o."city", o."vendor_id" FROM src_4."vendors" AS o WHERE o."city" IN (SELEC` | — | **DIFFERENT** |
| XS2 | answered | answered | `SELECT t1."city_name" AS city, SUM(t0."amount") AS "total_maintenance_amount" FROM src_4."` | `SELECT t1."city_name" AS city, SUM(t0."amount") AS "total_maintenance_amount" FROM src_4."` | SAME |



**Every divergence, flagged explicitly:**

| Q | divergence | which tree is right | evidence |
|---|---|---|---|
| **DB1** | master answers from `reminders_reminderpaymenttransaction`; branch answers from `accounts_paymenttransaction` | **branch** | `accounts_paymenttransaction` holds **306** rows and the real payment columns. `reminders_reminderpaymenttransaction` holds **11** rows and only `id, reminder_id, payment_transaction_id, created_at, updated_at, end_date` — a pure junction table with no payment data. Master answered "latest 10 payment transactions" out of a reminder link table. Verified with `psql homzhub` row counts and `information_schema.columns`. |
| **DL2** | master `clarify`; branch answers with `AVG(monthly_fee) GROUP BY category` on `amenities_catalog` | **branch** | the branch SQL matches the expected shape exactly; master refuses a question this repo's own battery (`scripts/eval_per_source_battery.py:206`) lists as expected-to-answer. |
| **FS1** | master **fabricates** *"The late fee percentage is 818.000"*; branch **refuses** *"I couldn't find any data relevant to this question in the sources available to you"* | **neither — but the branch fails more safely** | The correct answer is 2%, in `msa_green_tower.pdf`. In the paired run master answers this **document** question with a federated SQL — `SELECT t1."asset_id", SUM(t0."price_percent") … FROM src_2.public."services_valuebundlepricing" t0 INNER JOIN src_4."maintenance" t1 …` — producing 818.000 in **all 3 reps**, with **no citations**, and hedging *"This value appears to be an outlier or error"*. Master is not stably wrong either: in the second independent master run it answered correctly via RAG (*"2 percent per month"*, cited `msa_green_tower.pdf`) in all 3 reps, taking `rag_synthesis` instead of `nl_answer`. Branch routing decided `mode=SINGLE, source_ids=["2"]`, built SQL against homzhub and failed to execute: `firewall={"verdict":"ok","head":"llm_sql","ir_partial":true}`, lifecycle `"a data source: This data source could not be reached"`. **Both trees route a document question to the SQL head at full scope**; only master invents a number from it. |
| **XS1** | master answers; branch `clarify` | **neither** | Master's SQL is correct — `SELECT DISTINCT o."city", o."vendor_id" FROM src_4."vendors" o WHERE o."city" IN (SELECT DISTINCT f."city_name" FROM src_2.public."assets_asset" f …)` — and returns **6 rows in all 3 reps**. Its prose does not: rep 1 *"2 vendors operate in cities where we have assets"*, rep 2 *"2 cities have vendors operating in them: Kochi and Banda"*, rep 3 *"6 vendors operate in cities where assets are located"*. **Three different answers from one identical result set.** The branch routes correctly (`mode=MULTI, source_ids=["2","4"]`) and then declines. |
| DB2, DL1, XS2 | same status | — | **byte-identical SQL** on both trees. |
| FS2 | same status | — | same document cited on both. Prose wording differs between reps **on the same tree** — temperature 0.3, §0.3.1. |

### SLM decode cost (median rep) — transport-measured

| Q | master SLM wall ms | master tok | master ms/tok | branch SLM wall ms | branch tok | branch ms/tok |
|---|---:|---:|---:|---:|---:|---:|
| DB1 | 13,919 | 1,325 | 10.51 | 10,443 | 1,416 | 7.37 |
| DB2 | 814 | 102 | 7.98 | 1,389 | 102 | 13.62 |
| DL1 | 10,083 | 875 | 11.52 | 8,539 | 872 | 9.79 |
| DL2 | 786 | 103 | 7.63 | 12,684 | 850 | 14.92 |
| FS1 | 22,224 | 1,600 | 13.89 | 781 | 104 | 7.51 |
| FS2 | 17,862 | 1,231 | 14.51 | 12,560 | 1,251 | 10.04 |
| XS1 | 19,795 | 1,360 | 14.55 | 3,093 | 406 | 7.62 |
| XS2 | 22,914 | 1,658 | 13.82 | 38,504 | 2,553 | 15.08 |



### Round-trip counts (median rep)

| Q | emb rt (m/b) | emb ms (m/b) | rerank rt (m/b) | rerank ms (m/b) | db conn (m/b) | db q (m/b) |
|---|---|---|---|---|---|---|
| DB1 | 4/4 | 1,816/1,480 | 1/1 | 343/369 | 30/32 | 38/49 |
| DB2 | 4/4 | 3,787/3,750 | 1/1 | 322/290 | 20/23 | 27/39 |
| DL1 | 2/2 | 364/332 | 0/0 | 0/0 | 31/10 | 30/18 |
| DL2 | 6/2 | 1,068/288 | 1/0 | 276/0 | 51/46 | 57/54 |
| FS1 | 2/3 | 246/908 | 3/1 | 509/311 | 17/14 | 18/31 |
| FS2 | 4/3 | 1,020/323 | 3/0 | 913/0 | 16/9 | 12/13 |
| XS1 | 2/5 | 454/767 | 3/1 | 902/325 | 16/10 | 18/27 |
| XS2 | 2/2 | 263/367 | 3/3 | 732/807 | 18/20 | 19/28 |



### SLM call ledger (median rep) — purpose, latency, tokens

- **DB1 / master**: 1 calls — nl_answer (13557ms, 1161p/60c, ok=True, timeout=no)
- **DB1 / branch**: 2 calls — answer_entity_relation (740ms, 102p/2c, ok=True, timeout=no); nl_answer (9704ms, 1271p/41c, ok=True, timeout=no)
- **DB2 / master**: 0 SLM calls
- **DB2 / branch**: 1 calls — answer_entity_relation (1389ms, 100p/2c, ok=True, timeout=no)
- **DL1 / master**: 1 calls — nl_answer (10083ms, 827p/48c, ok=True, timeout=no)
- **DL1 / branch**: 1 calls — nl_answer (8540ms, 823p/49c, ok=True, timeout=no)
- **DL2 / master**: 0 SLM calls
- **DL2 / branch**: 1 calls — nl_answer (12684ms, 766p/84c, ok=True, timeout=no)
- **FS1 / master**: 3 calls — operation_classify (10996ms, 336p/43c, ok=True, timeout=no); federated_struct_plan (7955ms, 611p/82c, ok=True, timeout=no); nl_answer (3276ms, 505p/23c, ok=True, timeout=no)
- **FS1 / branch**: 1 calls — answer_entity_relation (782ms, 102p/2c, ok=True, timeout=no)
- **FS2 / master**: 1 calls — rag_synthesis (17862ms, 1116p/115c, ok=True, timeout=no)
- **FS2 / branch**: 1 calls — rag_synthesis (12561ms, 1116p/135c, ok=True, timeout=no)
- **XS1 / master**: 3 calls — operation_classify (7847ms, 339p/27c, ok=True, timeout=no); semi_join_classify (3705ms, 433p/16c, ok=True, timeout=no); nl_answer (8245ms, 494p/51c, ok=True, timeout=no)
- **XS1 / branch**: 1 calls — decompose (3094ms, 377p/29c, ok=True, timeout=no)
- **XS2 / master**: 3 calls — operation_classify (3320ms, 340p/30c, ok=True, timeout=no); federated_struct_plan (15495ms, 628p/80c, ok=True, timeout=no); nl_answer (4110ms, 545p/35c, ok=True, timeout=no)
- **XS2 / branch**: 5 calls — decompose (8382ms, 378p/32c, ok=True, timeout=no); operation_classify (3270ms, 340p/30c, ok=True, timeout=no); federated_struct_plan (12332ms, 568p/80c, ok=True, timeout=no); query_understanding (9054ms, 488p/57c, ok=True, timeout=no); nl_answer (5476ms, 545p/35c, ok=True, timeout=no)

No SLM call in any timed run timed out or retried — every ledger entry records `ok=true`,
and the harness records failures too. The one duplicate observed is in §7.

### Follow-up turn (chat path)

#### Latency

| script | turn | master status | master med ms | branch status | branch med ms | Δ ms | Δ % |
|---|---|---|---:|---|---:|---:|---:|
| FU1 | opening | answered | 4,184 | answered | 6,518 | +2,334 | +55.8% |
| FU1 | **follow-up** | qualifier_dropped | 22,963 | clarify | 15,096 | -7,868 | -34.3% |
| FU2 | opening | clarify | 5,368 | clarify | 5,125 | -243 | -4.5% |
| FU2 | **follow-up** | clarify | 6,722 | federated_refused | 21,864 | +15,143 | +225.3% |


#### Follow-up turn — tokens and session behaviour

| script | turn | master tokens | branch tokens | master route_source | branch route_source | master sup_slm | branch sup_slm |
|---|---|---:|---:|---|---|---|---|
| FU1 | opening | 1,037 | 562 | None | pin | None | 1 |
| FU1 | **follow-up** | 1,177 | 1,024 | None | inherited | None | 1 |
| FU2 | opening | 561 | 561 | None | pin | None | 1 |
| FU2 | **follow-up** | 865 | 1,815 | None | coordinator | None | 2 |

**Neither tree answers either follow-up.** The divergences:

- **FU1** ("how many of those are in the Repair category", inheriting source 4): master
  returns `qualifier_dropped` at 23.0 s; branch returns `clarify` at 15.1 s. The branch is
  **34% faster at the same non-answer**, and its `route_source=inherited` shows the session
  kept its thread; master emits no `route_source` at all.
- **FU2** ("how many of those are in Mumbai", inheriting source 2): master `clarify` at
  6.7 s / 865 tokens; branch `federated_refused` at 21.9 s / 1,815 tokens with
  `route_source=coordinator` — the branch **re-routed a follow-up through the coordinator
  instead of inheriting**, spent 2 supervisor SLM calls and 3.3× the wall clock, and still
  did not answer.
- `supervisor_slm_calls` is **not measured on master**: `chatbot/run.py` on master does not
  return that field (it is a branch addition). Engine-side per-call SLM data is likewise
  unavailable for master's chat path, because master's `ExplainTrace.compact()`
  (`/Users/ekesel/samta/veda-master-bench/veda_core/veda/explain.py:201-221`) carries no
  `slm_calls` list and its token totals are structurally zero (§4, A3). The `engine_usage`
  column above is the combined figure `run_chat_turn` returns on both trees.

---

## 4. Bug fixes in this branch

Derived from `git log --no-merges 7c5368f..f768040` and the diff, then **verified in code
and, where possible, at runtime**. Sorted by severity.

### 4.1 Correctness

#### A1 — `plan_route()` raised on every call, so multi-source routing never ran

| | |
|---|---|
| **Broken on master** | `veda_core/query/source_coordinator.py:567` imports `query.capability_filter` **unconditionally**. That module does not exist in the tree. Every `plan_route()` call therefore raises `ModuleNotFoundError` before reaching `decide()`. |
| **Swallowed at** | `veda_hybrid.py:881-884` — `except Exception: … return None`, which falls through to the legacy merged-all-source path. Nothing is logged unless `verbose`. |
| **Fixed on branch** | `veda_core/query/source_coordinator.py:630-633` — the import is moved inside `if _cap_filter_on:` (`CAPABILITY_FILTERING_ENABLED`, `veda_core/config.py`, default `False`). |
| **Runtime proof** | master: `plan_route(...)` → `ModuleNotFoundError: No module named 'query.capability_filter'`. branch, same call, same scope → `status=ROUTED mode=MULTI sources=['2','4'] reason="Sources are joined by a discovered cross-source relationship."` Both executed in-container against the live stack. |
| **Blast radius** | Every query with more than one source in scope: source narrowing, ambiguity handling, `decide()`, the MULTI override and the permission pre-check's inputs. Master served every such query from a merged all-source scope. |
| **Test coverage** | Yes, and it catches it. `tests/test_source_coordinator.py` — **24 of 26 tests fail on master, all with this exact `ModuleNotFoundError` at `source_coordinator.py:567`**. The suite was never run: `pytest` appears in **no** file under `requirements/`, and is absent from both the `inference` and `api` images (`python -m pytest` → `No module named pytest` in each). |
| **Observable behaviour** | **Yes — this is the root cause of most of §3's divergences.** Master's numbers are "routing disabled"; the branch's are "routing enabled for the first time". That cuts both ways: it fixes DL2 and routes XS1 correctly (`MULTI [2,4]`), and it is also what sends FS1's document question to source 2. |

#### A1b — `MULTISOURCE_ROUTING_SHADOW=1` is bypassed for SINGLE and MULTI decisions

This is a **deliberate policy change, not a fix**, and together with A1 it is the actual
mechanism behind the FS1/XS1 divergences.

| | |
|---|---|
| **Master** | `veda_core/veda_hybrid.py:785-786` — `if MULTISOURCE_ROUTING_SHADOW: return None  # observe only`. Every routing decision is discarded. (Moot on master anyway: A1 meant no decision was ever produced.) |
| **Branch** | `veda_core/veda_hybrid.py:958-960` — `_effective_shadow = bool(MULTISOURCE_ROUTING_SHADOW) and not _is_multi_decision and not _is_single_decision`. A `ROUTED/MULTI` **or** `ROUTED/SINGLE` decision is now authoritative **regardless of the flag**; only `NO_MATCH` / `CLARIFICATION_REQUIRED` stay advisory. |
| **Why this matters** | `.env:126-127` sets `MULTISOURCE_ROUTING_SHADOW=1` with a 14-line comment recording a *measured* regression as the reason, and instructing that it be left off "until the gate is softened". After this change the operator's kill switch no longer disables the two modes that actually route. Turning routing off now requires `MULTISOURCE_ROUTING_ENABLED=0`, not `_SHADOW=1`. |
| **Runtime proof** | FS1 on the branch: `routing={"mode":"SINGLE","source_ids":["2"]}` and the SQL head then runs against source 2 — with `MULTISOURCE_ROUTING_SHADOW=1` set in the `.env` both trees used. Under master's rule that decision would have been discarded. |
| **Test coverage** | `tests/test_authoritative_routing.py` is modified on the branch and monkeypatches `plan_route`, but **no test found covering the `_effective_shadow` computation** or the SINGLE-authoritative condition. |
| **Observable behaviour** | Yes — this is the single highest-risk change on the branch. |

#### A2 — "latest 10 payment transactions" answered from a junction table

| | |
|---|---|
| **Broken on master** | Answered from `reminders_reminderpaymenttransaction` — 11 rows, columns `id, reminder_id, payment_transaction_id, created_at, updated_at, end_date`. No payment data at all. |
| **Fixed on branch** | Answers from `accounts_paymenttransaction` — 306 rows, real payment columns. |
| **Verified** | `psql -d homzhub` row counts and `information_schema.columns`; both SQL strings captured in `reports/raw/engine_{master,branch}.jsonl` under `qid=DB1`. |
| **Blast radius** | Relational anchor selection for entity nouns that appear in several tables. Not attributable to one line of the diff — it follows from the anchor/grounding changes in `veda_core/veda/understanding/grounding.py` (+422) and `veda_core/veda/analytical_spec.py` (+137). **Not verified to a single line.** |
| **Test coverage** | No test found asserting this table choice. |
| **Observable behaviour** | Yes — different answer. |

#### A2b — a filter value belonging to another source was written off as filler

| | |
|---|---|
| **Broken on master** | `veda_core/veda/validation.py:308,337` — both grounding oracles see only the narrowed source. A literal that is real in *another* source is invisible, the token is classed as filler, and a SQL missing that filter is approved. The failure mode is a **fabricated answer**, not a refusal: an unfiltered `SELECT COUNT(*)` presented as though the filter had been applied. |
| **Fixed on branch** | `veda_core/veda/validation.py:230` adds `_value_in_engine_store()` (exact match against `column_values`, fails **open**), wired at `:369`, `:400` and `:533`. Master contains zero occurrences of the symbol. |
| **Blast radius** | Every single-source query naming a value that lives in another source. |
| **Test coverage** | **None.** `grep -rl _value_in_engine_store tests/` is empty. |
| **Observable behaviour** | Yes — fabricated-filter answers become refusals. |
| **Severity** | Correctness. Same fabrication class as master's FS1 "818.000" (Table D), and it ships with no test. |

### 4.2 Reliability / observability

#### A3 — `_fold_usage()` had no caller: `llm_usage` was absent from every master trace

| | |
|---|---|
| **Broken on master** | `veda_core/slm/_call_slm.py:97` defines `_fold_usage`; **nothing calls it**. `get_usage()` therefore always reports `calls=0`, and `explain.py:233`'s guard `if u and u.get("calls"):` never passes — so the `llm_usage` section is missing from every trace and MLflow's token metrics sit at zero. |
| **Fixed on branch** | `veda_core/slm/_call_slm.py:443` calls `_fold_usage(purpose)`; `veda_core/veda/explain.py:273` relaxes the guard to `if u:` so "SLM never needed" and "SLM called and errored" stop producing byte-identical traces. |
| **Runtime proof** | §2.1 table — master reports 0 calls / 0 tokens on DB2 and DL2 where the wire shows 1 call / ~102 tokens. |
| **Test coverage** | None. `grep -rn 'llm_usage\|_fold_usage' tests/` → no match. |
| **Observable behaviour** | No change to answers; changes what is recorded. |

#### A4 — `answer_entity`'s SLM call bypassed the one choke point

| | |
|---|---|
| **Broken on master** | `veda_core/query/answer_entity.py:113-121` builds its own payload and POSTs with `urllib.request` to `{SLM_OLLAMA_BASE_URL}/api/chat`. Consequences: invisible to the SLM ledger and to token accounting; **ignores `SLM_BACKEND` entirely**, so on a vLLM deployment it fails every time and silently returns `None`; sends no `num_ctx`. |
| **Fixed on branch** | `veda_core/query/answer_entity.py:112-119` — `call_slm(..., purpose="answer_entity_relation", num_predict=8, timeout=8)`. |
| **Runtime proof** | §2.1. On DB1/DB2/FS1 the branch shows a `answer_entity_relation` ledger entry (~100p/2c, 340–840 ms) that master makes but never records. |
| **Blast radius** | Any query with no closed-class cue that reaches the answer-entity projection. Flag `ANSWER_ENTITY_LLM_FALLBACK_ENABLED` is `True` by default on both (`veda_core/config.py:2621` / master `:2556`). |
| **Test coverage** | No. `tests/test_answer_entity_projection.py` never touches `_llm_relation_word`, `call_slm` or `urlopen`. |
| **Observable behaviour** | Same call, same parameters on an Ollama backend → no answer change here. **On a vLLM backend it is a behaviour change** (master's call always failed). |

#### A5 — `/api/generate` responses were not token-counted

`veda_core/slm/_call_slm.py`: master calls `_note_usage(body)` only on the `/api/chat`
branch (`:271`); the branch adds it to the `generate` branch too (`:285`). Every raw-prompt
call was missing from `llm_usage.per_purpose` while still appearing in the per-call ledger —
the two views of one query disagreed. No test. No answer change.

#### A6 — `prewarm()` reported success unconditionally

master returned `None` always; branch returns `True`/`False`
(`veda_core/slm/_call_slm.py:455-460`). `inference/loaders.py` set
`_STATE["nl_summary_model_warm"] = True` regardless, which made the
`nl_summary_slm_unreachable` degrade flag unreachable code. No test. No answer change.

#### A7 — no way to disable the verified-query cache

master's `RequestContext` (`/Users/ekesel/samta/veda-master-bench/veda_core/context.py:44-47`)
has no `cache_back` field, and `veda_core/veda/pipeline.py:690` calls
`verified_cache_lookup(query)` unconditionally. The branch adds `cache_back`
(`veda_core/context.py:52`) and honours it at `veda_core/veda/pipeline.py:796`. Consequence
for anyone benchmarking master: repeated identical queries replay a cached SQL. This
harness neutralises the cache on both trees (§0.5). No test (`grep cache_back tests/` →
no match). Behaviour change: only when the flag is set.

#### A8 — two trace sections are still silently dropped, on both trees

`ExplainTrace.to_dict()` filters to the `_SECTIONS` allow-list. The branch adds nine
sections (§5, B2) but **`rbac_filter` and `semantic_validation` remain absent from it**,
while `veda_core/veda/pipeline.py:1011` writes `rbac_filter` and `:2449`/`:2451` write
`semantic_validation`. Both are discarded on the way out, on the branch as on master. Not a
regression; an unfinished fix. No test.

### 4.3 Further items — reported by a secondary static audit, NOT independently verified

A second pass over the diff surfaced the items below. They carry `file:line` and are
plausible, but **I did not verify them myself and no runtime evidence in this report depends
on them.** Listed so they are not lost; treat each as a lead, not a finding.

| claim | cited location | why it matters if true |
|---|---|---|
| Multi-source scope gave the SQL head only the primary source's schema, while retrieval used the merged model | master `veda_hybrid.py:154-155` vs branch `:152-196` | cross-source questions could only ever plan against one schema |
| Federated SQL had no qualifier-completeness gate | branch `veda/validation.py:481-536`, called from `veda/firewall.py:384` | a federated plan could drop a filter with nothing to catch it |
| Four distinct false-refusal bugs in `intent_sql_alignment.py` (substring vs word-boundary unit match; one-word column names; unscoped boolean-flag match; flag-vs-measure preference) | branch `:85-90`, `:160-170`, `:283-290`/`:343`, `:274-281` | each refused a *correct* answer; one example given is "users created **last** month" matching `users_userpreference.is_last_name_obfuscated` — the DB2 query in this benchmark |
| Routing evidence took a single global top-10, so the 1902-column source crowded out a 9-column one; and raw cross-kind score max let chunk cosine win tabular questions | branch `source_coordinator.py:451-468`, `:743-765` | the routing-quality half of A1 |
| Signal-1 dense fallback queried `column_embeddings_v2` with no `source_id` predicate | master `retrieval/semantic_search.py:166-183` vs branch `:170-200` | cross-source column leakage on any adapter exception |
| Intent boosting was a total no-op — it read `sm["tables"][t]["columns"][c]`, but column metadata lives in the flat top-level `sm["columns"]` | branch `retrieval/intent_boosting.py:66-70` | every `boost_*` method did nothing, for every intent |
| Graph and synonym caches were process-global and unscoped | branch `veda/runtime.py:129`, `veda/graph_guard.py:28-59`, `query/fast_path.py:140`, `veda/validation.py:434-467` | join validation against another source's or a stale graph |
| A source with no SQL endpoint crashed with a raw `OperationalError` instead of refusing | master `veda/execution.py:136,141` vs branch `:136-157` | crash-vs-refusal on a document source |
| `summary.model` / `summary.success` reported the model *requested* and "some summariser ran", not what actually happened | branch `veda/explain.py:696-711` | traces claimed successful SLM summaries written by the deterministic template |
| `veda/firewall.py` is claimed to be a pure move of six existing gates with zero behaviour change | branch `veda/firewall.py:1-24` | **explicitly unverified by both passes** — it sits in the path of every query |

The same audit independently reached my conclusions on A1, A1b, the `answer_entity`
choke-point bypass, `_fold_usage`, `cache_back`, and the `ROUTING_PERMISSION_DENY_GAP`
duplication being pre-existing.

### 4.4 Cosmetic / pre-existing

- `ROUTING_PERMISSION_DENY_GAP` is defined **twice** in `veda_core/config.py` (`:757` and
  `:770`) with duplicated comment blocks. It is the **only** duplicated top-level assignment
  in the file, and **master has the same duplication** — pre-existing, not branch-introduced.
  Same value both times, so no behavioural effect.

---

## 5. Improvements in this branch

| # | change | `file:line` | category | measured effect |
|---|---|---|---|---|
| B1 | Per-call SLM ledger + per-purpose token totals ride the **compact** trace record, so `scripts/slm_purpose_report.py` and the trace log can read them without verbose mode | `veda_core/veda/explain.py:244-252` | observability | Directly enabled this report's Table B for the branch. Master's compact record has neither (`master .../explain.py:201-221`), which is why the follow-up section cannot report master's engine-side SLM calls. |
| B2 | Nine new trace sections: `retrieval_health`, `nl_summary`, `understanding`, `analytical_sql_v2`, `firewall`, `dimension_alignment`, `entity_resolution`, `value_arbitration`, `federated` | `veda_core/veda/explain.py:69-84` | observability | Visible in Table C; `firewall` reports a verdict on every executed SQL (e.g. DB1 branch: `verdict=ok, head=llm_sql, ir_partial=true, checks_run=[value_grounding, qualifier, alignment, ir_equivalence, rbac, ast_parameterize]`). Master records none of it. |
| B3 | `turn_slm_calls` named explicitly in totals | `veda_core/veda/explain.py:215` | observability | Same number as `slm_call_count` today; it is the contract `scripts/eval_sessions.py` asserts against. |
| B4 | `num_ctx` defaulted centrally from `SLM_NUM_CTX` (4096) for every call that does not name its own | `veda_core/slm/_call_slm.py:401-405` (branch) — master's `_config()` has no `num_ctx` key and `call_slm` never sets one | latency | Intended to stop Ollama reloading the model between sized and unsized calls. **Not measurable in this query set** — both hosts stayed warm and no model-reload stall was observed in any of the 48 timed runs. |
| B5 | Session memory: `session_prior` / `session_anchor` on `RequestContext`, consumed by the routing coordinator | `veda_core/context.py:58-64`; `veda_core/veda_hybrid.py:909-921` | correctness | Measured on FU1: branch reports `route_source=inherited` on the follow-up (the session kept its thread); master has no such field. Did **not** produce an answer on either follow-up. |
| B6 | `chatbot/memory/delta.py` (new, +417) and `frame.py` (+398) — deterministic follow-up delta layer | `chatbot/memory/delta.py:1` | latency / token cost | FU1 follow-up: branch 15.1 s / 1,024 tok vs master 23.0 s / 1,177 tok. Does not resolve for FU2, where the branch re-routes through the coordinator instead (21.9 s / 1,815 tok vs master 6.7 s / 865). |
| B7 | `veda_core/veda/firewall.py` (new, +392) and `veda_core/veda/ir.py` (new, +263) | `veda_core/veda/firewall.py:1` | correctness | Firewall stage costs 5–159 ms on simple queries but **5,641 ms (median) on XS2** — the largest new stage cost measured. Note §2.3: this is a gap-derived duration and may overlap the SLM calls beside it. |
| B8 | 29 new `AGENTS.md` files + docs | — | maintainability | Not measurable. |

**Implemented but disabled by a flag (current effective value from `.env` + `config.py`):**

| flag | `file:line` | default | effective now |
|---|---|---|---|
| `CAPABILITY_FILTERING_ENABLED` | `veda_core/config.py:964` | `"0"` | **OFF** — and the module it would import (`query/capability_filter.py`) still does not exist, so turning it on reintroduces A1 |
| `CAPABILITY_PLANNING_SHADOW_ENABLED` | `veda_core/config.py` | `False` | OFF |
| `MULTISOURCE_ROUTING_SHADOW` | `.env` | `1` | **shadow mode ON** |
| `REQUIRED_SOURCE_ESCALATION_ENABLED` | `.env` | `0` | OFF |
| `SUPERLATIVE_JOIN_ROUTING` | `veda_core/config.py` | off | superlative planner remains trace/route-only |
| `VEDA_QUERY_DECOMPOSE_ENABLED=false` in `.env` | — | — | **inert** — `QUERY_DECOMPOSE_ENABLED = False` is a literal at `veda_core/config.py:2683` (§0.3.1). The `decompose` SLM call still fired on XS1/XS2 on the branch, so decomposition is reached by a different path than this flag. |

**The inverse is the real risk: every new flag ships ON.** None of the branch's new settings
appear in `.env` (verified key by key), so all of them run at their code default, and none
exist on master at all:

| new flag | `veda_core/config.py` | default | effective now |
|---|---|---|---|
| `ROUTING_CARDS_ENABLED` | `:394` | `"1"` | **ON** |
| `INTENT_SQL_ENTITY_COVERAGE_ENABLED` | `:1057` | `"1"` | **ON** |
| `ENTITY_COVERAGE_CONFIDENCE` | `:1062` | `0.6` | 0.6 |
| `ROUTING_EDGE_MIN_JACCARD` | `:731` | `0.05` | 0.05 |
| `ROUTING_CHUNK_KIND_OFFSET` | `:735` | `0.07` | 0.07 |
| `QUERY_UNDERSTANDING_MIN_CONFIDENCE` | `:2156` | `0.5` | 0.5 |
| `RETRIEVAL_INTENT_BOOST_SCALE` | `:2015` | `0.12` | 0.12 — **hardcoded literal, no env override at all** |
| `SLM_NUM_CTX` | `:401` | `4096` | 4096 |

The two routing scalars (`0.07`, `0.05`) are empirically fitted to this one deployment's
data, per the comments at their definitions — tuned, not derived.

---

## 6. Latency factors, ranked by measured cost

### 6.1 Duplicate work — every embedding round-trip is issued twice, byte-identically

The largest single measured waste, and it is present on **both** trees. The harness hashes
each `_metal_post` payload; every query issues each encode **exactly twice with an identical
payload**:

| query | tree | duplicated round-trips | redundant wall time |
|---|---|---|---|
| DB2 | master | `/encode_dense`(1 text) ×2, `/encode_query` ×2, `/encode_sparse`(409 texts) ×2 | **2,167 ms** |
| DB2 | branch | same shape, `/encode_sparse`(407 texts) ×2 | **1,959 ms** |
| DB1 | master | same shape, `/encode_sparse`(101 texts) ×2 | 828 ms |
| DB1 | branch | same shape, `/encode_sparse`(95 texts) ×2 | 731 ms |

On DB2 that is **~35% of the query's entire wall clock** (5.3 s master / 5.9 s branch) spent
re-computing a result the process already had. Raw evidence:
`reports/raw/embed_identity_{master,branch}.jsonl`.

Note also *what* is being encoded: `/encode_sparse` with **409 texts** is the column catalog,
learned-sparse-encoded **at query time**, twice, for 1.9 s each. Neither tree fixes this.

### 6.2 SLM decode dominates everything else

#### Decode-seconds cost model (self-hosted; no $/token applies)

- Observed across all 84 timed SLM round-trips in both runs:
  total SLM wall time 605.1 s, prompt tokens 43,843, completion tokens 3,563
  aggregate 12.76 ms per total token, 169.8 ms per completion token

| tree | SLM wall s / query (median-rep sum over 8 queries) | tokens | per-1000-queries SLM wall (hours) |
|---|---:|---:|---:|
| master | 108.4 | 8,254 | 3.76 |
| branch | 88.0 | 7,554 | 3.06 |

Measured across all 84 timed SLM round-trips in both runs: **12.76 ms per total token**,
**169.8 ms per completion token**. A single `nl_answer` or `rag_synthesis` call costs
4–10 s. On DB1 master, `nl_answer` alone is 10.1 s of a 16.5 s query.

### 6.3 Branch-specific: XS2 adds two SLM calls worth 15.9 s, for an identical answer

XS2 produces **byte-identical SQL and byte-identical prose** on both trees, yet:

| | master | branch |
|---|---|---|
| SLM calls | 3 — `operation_classify`, `federated_struct_plan`, `nl_answer` | 5 — `decompose`, `operation_classify`, `federated_struct_plan`, **`query_understanding`**, `nl_answer` |
| tokens | 1,658 | 2,553 (+895) |
| `firewall` stage | absent | **5,641 ms** (median of 5,176 / 5,641 / 8,088) |
| median wall | 24,420 ms | 40,255 ms (**+64.8%, non-overlapping**) |

The two added calls cost, at their medians across the three reps:

| added call | median latency | tokens |
|---|---:|---|
| `decompose` | 6,834 ms | 378p / 32c |
| `query_understanding` | **9,054 ms** | 488p / 61c |

`query_understanding` runs **after** `federated_struct_plan` has already produced the plan,
and the answer is byte-identical to master's without it. The two added calls are
**15.9 s of directly-measured SLM wall time** — on their own they account for the entire
+15.8 s median regression. (The `firewall` figure is *not* added to that: it is a
gap-derived stage duration per §2.3 and may overlap the same SLM calls.) This is the
clearest case of "work that runs but does not change the output" in the branch.

### 6.4 Per-request connection overhead

Measured connects per single query (via a patched `psycopg2.connect`):

| query | master connects (engine / source) | branch connects (engine / source) |
|---|---|---|
| DL2 | 51 (44 / 0) | 46 (46 / 0) |
| DB1 | 30 (26 / 4) | 32 (28 / 4) |
| DL1 | 31 (31 / 0) | 10 (10 / 0) |
| FS2 | 16 (16 / 0) | 9 (9 / 0) |

10–51 connects for **one** query. PgBouncer is in transaction-pooling mode with
`default_pool_size = 20` (`docker/pgbouncer.ini`), so these are cheap, but the count is a
per-request structural cost, not a pool problem. The branch roughly halves it on DL1/FS2 and
slightly increases it on DB1/DB2.

### 6.5 Serial work that could be parallel

`veda_core/veda/pipeline.py`'s L-stages and the federated plan/execute steps run
sequentially; `execution_plan` on the branch records `planner_mode: "PARALLEL"` but
`executed_mode: "sequential"` even for the single-source case (FS1 branch trace). **Not
measured:** no parallel variant exists to compare against.

### 6.6 SLM calls with no output cap or explicit timeout — tail risk

Of 33 `call_slm` sites in `veda_core/`, these query-path sites pass **neither** `num_predict`
nor an explicit `timeout`, so they inherit `SLM_TIMEOUT_SECS = 240`
(`veda_core/config.py:382`) with an unbounded decode budget:

| site | purpose |
|---|---|
| `veda_core/query/operation_classifier.py:185` | `operation_classify` |
| `veda_core/query/federated_route.py:381` | `federated_plan` |
| `veda_core/query/federated_route.py:513` | `federated_struct_plan` |
| `veda_core/query/semi_join_planner.py:133` | `semi_join_classify` |
| `veda_core/query/doc_data_planner.py:88` | `doc_data_ground` |
| `veda_core/veda_hybrid.py:505` | `multi_summary` (has `num_predict`, no timeout) |

All six are on the cross-source path — exactly the path where XS2 already reaches 40 s.
No call timed out in any run, but the ceiling is 240 s each.

### 6.7 Infra ceilings are inverted

| layer | limit | `file:line` |
|---|---|---|
| **nginx `proxy_read_timeout`** | **120 s** | `docker/nginx.conf:66` |
| api → inference | 300 s | `.env:107` `INFERENCE_TIMEOUT_S` |
| gunicorn worker | 600 s | `.env:110` `GUNICORN_TIMEOUT` |
| one SLM call | 240 s | `veda_core/config.py:382` |

The **outermost** hop is the **tightest**. A single SLM call is permitted to run twice as
long as nginx will hold the client connection, and the app layers are allowed 2.5–5× the
proxy budget. The comment at `docker/nginx.conf:60-65` justifies 120 s against an "observed
~35-40 s typical case" — the branch's XS2 median is **40.3 s** and p90 **42.9 s**, i.e. it
has moved a real query onto the edge of that stated headroom.

`INFERENCE_WORKERS=1` (`.env:87`) with a synchronous engine means the query tier serves
essentially one query at a time; every number in this report is a single-query,
zero-contention measurement and says nothing about behaviour under concurrency.

### 6.8 Cold start

Not cleanly measurable here: both hosts were warm (`keep_alive="24h"`), the in-container
encoder is bypassed by `METAL_EMBED_URL`, and the two warm-up queries take different paths on
the two trees (master clarifies warm-up 1, the branch answers it), so their durations are not
comparable. **Not measured.**

---

## 7. Token cost

Per-query, per-tree, split prompt/completion and broken down by SLM site: **Table B** and the
**SLM call ledger** in §3.

**Top 3 token sinks** (transport-measured, median rep):

1. **`nl_answer` / `rag_synthesis` prompts** — 460–1,373 prompt tokens per call, on nearly
   every answered query. DB1 branch: 1,373p for a 39-token answer. These prompts carry the
   result rows and schema context; they are the single biggest prompt-side cost.
2. **XS2's five-call chain on the branch** — 2,319 prompt tokens across `decompose`,
   `operation_classify`, `federated_struct_plan`, `query_understanding`, `nl_answer`, for an
   answer master produces with three calls and 1,513 prompt tokens.
3. **`federated_struct_plan`** — 551–628 prompt tokens, and it is one of the sites with no
   `num_predict` cap (§6.6). In master run A this call is issued **twice per query with
   identical prompts** on FS1, in **all three reps** (9,546 + 7,134 ms; 9,529 + 7,098 ms;
   9,521 + 7,161 ms — roughly **16.7 s of duplicated planning per query**). It does not
   appear at all in the paired master run, where FS1 takes a different head. So it is
   reproducible within a run and varies between runs, not random per call. Raw:
   `reports/raw/engine_master_runA.jsonl`, `qid=FS1`.

**Cost.** The SLM is self-hosted (Ollama on a LAN Metal box), so there is **no $/1M-token
price to apply**. Expressed as decode-seconds instead, as required:

| | master | branch |
|---|---|---|
| SLM wall time, sum over the 8-query median reps | **108.4 s** | **88.0 s** |
| tokens over the same 8 queries | 8,254 | 7,554 |
| projected SLM wall time per 1,000 queries | **3.76 h** | **3.06 h** |

Caveat: the branch's lower figure includes FS1 and XS1, which it does not answer. On the
**parity set only** (DB1, DB2, DL1, FS2, XS2) the branch spends **6,194 tokens vs master's
5,191 — +19.3%**.

**Unbounded output length:** the six sites in §6.6 have no `num_predict`. `rag_synthesis`
and `ir_emit` are capped by `SLM_MAX_TOKENS`; `nl_answer` and the ingestion sites pass
explicit caps.

---

## 8. Verdict

### 8.1 Net effect

| dimension | master | branch | delta | resolves above noise? |
|---|---|---|---|---|
| median latency, parity set (5 queries) | 76,927 ms | 80,961 ms | **+5.2%** | only XS2 does |
| p90 latency, XS2 | 25,316 ms | 42,918 ms | **+69.5%** | **yes** |
| tokens, parity set | 5,191 | 6,194 | **+19.3%** | n/a (deterministic counts) |
| SLM calls, parity set | 8 | 10 | +2 | n/a |
| answered and **correct** | 3 — DL1, FS2, XS2 | **5** — DB1, DL1, DL2, FS2, XS2 | **+2** | — |
| answered and **wrong** (confidently wrong output) | **3** — DB1 (junction table), FS1 (fabricated 818%), XS1 (prose contradicts its own 6 rows, differently in each rep) | **0** | **−3** | — |
| refused / clarified | 1 — DL2 | 2 — FS1, XS1 | +1 | — |
| answered, shape not verifiable | 1 — DB2 † | 1 — DB2 † | 0 | — |
| follow-up turns answered | 0 of 2 | 0 of 2 | 0 | — |
| observability | `llm_usage` absent from every trace; 3 of 8 queries under-report SLM calls | ledger matches the wire on 8 of 8 | **clear win** | — |

† DB2 ("users created last month") returns **byte-identical SQL and 0 rows on both trees**,
but that SQL filters `users_user` through `WHERE "id" IN (SELECT "created_by_id" FROM
"list_of_values_listofvalue" WHERE LOWER(code)=… )` — a subquery with no evident relation
to the question. It is scored as parity, not as correct, on either tree.

### 8.2 Regressions found

**R1 — a pinned document question is refused with a permission message.**
Reproduce:
```
docker compose run --rm --no-deps -v <bench>:/bench --entrypoint python inference \
  -u /bench/bench_engine.py --label branch --out /bench/out/p.jsonl \
  --reps 1 --warmup 0 --only FS1P
```
branch → `route="no_access"`, 409 ms, 0 SLM calls, *"You don't have permission to access
this data. Contact your Admin to request access."*  master, same command against the master
worktree → **answered in 6.4 s**. Setting `ROUTING_PERMISSION_PRECHECK_ENABLED=0` makes the
branch answer correctly in 3.6 s with the right citation (`msa_green_tower.pdf`), which
isolates the cause to the pre-check at `veda_core/veda_hybrid.py:860-905`. The flag defaults
to `"1"` on both trees (`veda_core/config.py:548`); what changed is the routing evidence
feeding it, now that A1 is fixed.

**R2 — FS1 at full scope: a document question is routed to the relational source.**
Branch trace: `routing={"mode":"SINGLE","source_ids":["2"]}` → SQL built against homzhub →
execution fails → user sees *"I couldn't find any data relevant to this question"*.
Reproduce with `--only FS1`.

**Mechanism:** A1 + A1b stacked. A1 makes `plan_route()` return a decision at all; A1b
(`veda_hybrid.py:958-960`) makes a `ROUTED/SINGLE` decision authoritative **even though
`MULTISOURCE_ROUTING_SHADOW=1`** in the `.env` both trees ran under. The trace shows exactly
that: `mode=SINGLE, source_ids=["2"]` then the SQL head. On master the same decision would
have been discarded at `veda_hybrid.py:785-786`.

This is a regression **in routing**, not in the answer: in the paired run master routes the
same question to the SQL head too and returns a **fabricated** *"late fee percentage is
818.000"* with no citations (3/3 reps). Master is therefore not a clean baseline here — it
is unstable, answering correctly via RAG in one full run and fabricating in another. Both
behaviours are wrong; refusing is the less harmful of the two.

Secondary defect: the lifecycle message for a failed SQL execution is *"This data source
could not be reached"*, which is misleading — the source was reachable (DB1/DB2 executed
against it in the same run).

**R3 — XS2 costs +64.8% wall clock and +895 tokens for a byte-identical answer** (§6.3).
Reproduce with `--only XS2`.

**R4 — FU2's follow-up re-routes through the coordinator instead of inheriting**, costing
21.9 s / 1,815 tokens vs master's 6.7 s / 865, and still not answering (§3, follow-up).

**R5 — the branch broke `tests/test_source_coordinator.py` in a new way.** The branch
deleted `import config as _cfg_mod` (present on master at
`/Users/ekesel/samta/veda-master-bench/tests/test_source_coordinator.py:88`) while leaving
14 `_cfg_mod` references. Result: master **24 failed / 2 passed**, all
`ModuleNotFoundError` (the real product bug A1); branch **14 failed / 11 passed**, all
`NameError: name '_cfg_mod' is not defined` — a broken test file. Reproduce:
```
PYTHONPATH=$PWD:$PWD/veda_core .venv/bin/python -m pytest tests/test_source_coordinator.py -q
```

### 8.3 Is this branch safe to merge?

**Conditional — yes.** On correctness the branch is clearly ahead: on this query set it
answers **5 of 8 correctly and 0 incorrectly**, against master's **3 correct and 3
confidently wrong** (a junction-table answer, a fabricated 818% late fee, and a prose
summary that contradicted its own result set differently in each of three reps). It fixes
A1, a bug that had silently disabled multi-source routing across the entire product, and it
makes SLM cost measurable for the first time.

What it costs: **+19.3% tokens and +5.2% median latency on the parity set** — effectively
all of it XS2, which is +64.8% wall clock for a byte-identical answer — and it converts two
previously-answered queries into refusals. FS1's refusal is a routing regression, but not an
answer regression: master routes that document question to the SQL head as well and invents
a number from it. The branch also ships a test file that cannot run.

Merge is recommended **only with conditions 1–4 below**; 5–6 can follow.

Conditions:

1. **Fix R1/R2** — a document question must not be routed to a relational source, and a
   pinned single-source scope must not be read as an RBAC denial. Minimum viable: exclude
   sources whose `source_type` is `document`/`filesystem` from the SQL head, and skip the
   permission pre-check when the caller's scope was explicitly pinned rather than
   RBAC-narrowed (`veda_core/veda_hybrid.py:860-905`).
2. **Decide A1b deliberately, and write the test it was gated on.** `veda_hybrid.py:958-960`
   makes SINGLE/MULTI routing authoritative in spite of `MULTISOURCE_ROUTING_SHADOW=1`, which
   `.env:126-127` documents as deliberately off pending "a SINGLE-vs-legacy-engine regression
   test". The branch comment claims that test now exists as
   `scripts/eval_cross_source_battery.py` — **that claim was not verified for this report**
   (the battery was not run). Either verify it, or restore `_SHADOW` as a working kill switch;
   as it stands the flag no longer does what `.env` says it does.
3. **Fix R5** — restore `import config as _cfg_mod` in
   `tests/test_source_coordinator.py`, then get the suite to green.
4. **Make the suite runnable** — add `pytest` to `requirements/` and to the `inference`
   image. A1 shipped because a test that catches it has never been executed.
5. **Justify or remove XS2's extra `query_understanding` call** (§6.3) — 9.1 s and 549
   tokens at its median, and the answer is byte-identical without it.
6. **Re-benchmark on a quiet host with n≥10.** §2.2 shows n=3 here cannot resolve anything
   below roughly ±60%.

**Test coverage does not track risk on this branch.** The branch adds ~793 test lines,
concentrated on its two safest changes — session memory (`tests/test_memory_ir_stack.py`,
new, 25 tests) and entity coverage (7 new tests for a feature that never refuses by design).
The changes that alter refusals and answers have none: I verified by grep that there is **no
test** referencing `_value_in_engine_store` (A2b), `cache_back` (A7), `llm_usage` /
`_fold_usage` (A3), `num_ctx` / `SLM_NUM_CTX` (B4), or `prewarm` (A6), and no test covering
the `_effective_shadow` condition (A1b). The one suite that *does* cover the biggest fix,
`tests/test_source_coordinator.py`, is the one the branch broke (R5).

Not blocking, but should be filed: `SLM_TEMPERATURE` is inert while `ir_emit` and
`rag_synthesis` run at 0.3 (§0.3.1) — this makes the engine non-deterministic and every
regression suite noisy, on both trees.

### 8.4 Top 5 next actions, ranked by (win ÷ effort)

| # | action | `file:line` | measured win | effort |
|---|---|---|---|---|
| 1 | Memoise the per-query encode. Every `_metal_post` payload is sent **twice, byte-identically**; a dict keyed on the payload hash for the life of one request removes it. | `veda_core/ingestion/m3_encoder.py:69` (`_metal_post`) | **0.7–2.2 s per query**, ~35% of DB2's wall clock | ~20 lines |
| 2 | Make `SLM_TEMPERATURE` actually read the env var, then set it to 0. | `veda_core/config.py:381` | removes non-determinism from `ir_emit` (SQL generation) and `rag_synthesis` | 1 line + a re-benchmark |
| 3 | Skip the permission pre-check when the scope was explicitly pinned; exclude document sources from the SQL head. | `veda_core/veda_hybrid.py:860-905`; `veda_core/query/source_coordinator.py:630` | fixes R1 + R2 — 2 of the 8 benchmark queries | moderate |
| 4 | Drop the post-plan `query_understanding` call on the federated path. | `veda_core/veda_hybrid.py` federated branch; call recorded as `purpose="query_understanding"` in `reports/raw/engine_branch.jsonl` `qid=XS2` | **−9.1 s, −549 tokens** on XS2 (median of 3 reps) | small, needs a parity check |
| 5 | Give the six uncapped cross-source SLM sites a `num_predict` and a timeout below nginx's 120 s. | `query/operation_classifier.py:185`, `query/federated_route.py:381,513`, `query/semi_join_planner.py:133`, `query/doc_data_planner.py:88`, `veda_hybrid.py:505` | bounds a 240 s tail on the path that already runs at 40 s | small |

---

## Appendix A — query set

Reproduce the engine set (8 queries × 3 reps + 2 warm-ups) with:

```bash
# branch (working tree)
docker compose run --rm --no-deps -v <scratch>/bench:/bench \
  --entrypoint python inference -u /bench/bench_engine.py \
  --label branch --out /bench/out/branch.jsonl --reps 3 --warmup 2

# master (separate worktree; veda_core/data copied in, see §0.2)
git worktree add /Users/ekesel/samta/veda-master-bench master
docker compose run --rm --no-deps -v <scratch>/bench:/bench \
  -v /Users/ekesel/samta/veda-master-bench:/app \
  --entrypoint python inference -u /bench/bench_engine.py \
  --label master --out /bench/out/master.jsonl --reps 3 --warmup 2
```

The query set is the `QUERIES` / `PROBES` list in `reports/raw/bench_engine.py`. Warm-ups
(never timed): `"how many vendors are there"` (source 4), `"what does the site notes
document say"` (source 3).

| id | question | `source_id` | `source_ids` |
|---|---|---|---|
| DB1 | latest 10 payment transactions | 2 | (2) |
| DB2 | users created last month | 2 | (2) |
| DL1 | total maintenance amount per category | 4 | (4) |
| DL2 | average monthly fee per category | 5 | (5) |
| FS1 | what is the late fee percentage | 3 | (2,3,4,5) |
| FS2 | summarize the employee handbook leave policy | 3 | (2,3,4,5) |
| XS1 | which vendors operate in cities where we have assets | 2 | (2,3,4,5) |
| XS2 | total maintenance amount spent per city where we own assets | 2 | (2,3,4,5) |
| FS1P | what is the late fee percentage | 3 | (3) — probe |
| FS2P | summarize the employee handbook leave policy | 3 | (3) — probe |

`set_source_profiles({'2':'relational','3':'filesystem','4':'csv_lake','5':'parquet'})` on
every run, matching `veda_core/doc_bench.py:81`.

Follow-up set (`reports/raw/bench_chat.py`), 2 turns × 2 scripts × 3 reps + 1 warm-up,
through `chatbot.run.run_chat_turn` with one `session_id` per script per rep:

| id | turn 1 (pinned) | turn 2 (measured follow-up) |
|---|---|---|
| FU1 | how many maintenance records are there → source 4 | how many of those are in the Repair category |
| FU2 | how many properties are there → source 2 | how many of those are in Mumbai |

The verified-query cache table `substrate_verifiedquerycache` is levelled to its 3-row
baseline before every rep on both trees (master has no `cache_back` switch — §4, A7).

## Appendix B — raw logs

| file | contents |
|---|---|
| `raw/engine_branch.jsonl` / `raw/engine_master.jsonl` | the paired runs all tables are computed from |
| `raw/engine_master_runA.jsonl` | the second, independent master run used for the noise estimate in §2.2 |
| `raw/chat_branch.jsonl` / `raw/chat_master.jsonl` | follow-up turns |
| `raw/probe_pinned_fs_{branch,master}.jsonl` | FS1P/FS2P pinned-scope probes (R1) |
| `raw/embed_identity_{branch,master}.jsonl` | payload-hashed embedding round-trips (§6.1) |
| `raw/*.console.log` | full engine stdout for each run |
| `raw/bench_engine.py`, `raw/bench_chat.py` | the harnesses |
| `raw/analyze.py`, `raw/analyze2.py`, `raw/analyze3.py` | the scripts that produce every table above |
