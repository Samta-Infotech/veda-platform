# VEDA — fixes for the 2026-09-23 benchmark findings

**Date:** 2026-09-23 · **Branch:** `feat/refinements-pipeline` · **Baseline:** `f768040` (branch HEAD before these changes)
**Findings:** [`VEDA_BENCH_feat-refinements-pipeline_vs_master_2026-09-23.md`](VEDA_BENCH_feat-refinements-pipeline_vs_master_2026-09-23.md)

Everything below is verified at the wire, in the trace, or by running the query. Where a fix
could not be made safely, or could not be verified, it says so. **Nothing was committed** —
all changes are uncommitted in the working tree (see §Commits).

> **Latency is not reported.** The run budget was cut short at the user's request. The rules
> for this work require n≥10 on a quiet host for any latency claim, and the benchmark report
> itself measured 42% run-to-run noise on identical code. Every latency number that would
> have gone here is therefore **not resolved**, and none is stated. Correctness, token and
> round-trip counts below are exact counts, not timings, and are unaffected.

---

## Summary

| group | finding | outcome |
|---|---|---|
| 0 | 10 `.env` keys read by nothing; `SLM_TEMPERATURE=0` inert while the engine sampled at 0.3 | **Fixed.** 11 keys wired (an 11th found), 4 dead keys deleted, 1 renamed, 10 always-on flags surfaced. FS2 byte-identical 5/5 (was 5 distinct answers in 5 runs). |
| 1 | every embedding round-trip issued twice, byte-identically | **Fixed.** Per-request single-flight; every payload hash now appears exactly once. |
| 2.1 | R1 — pinned document scope refused as a permission denial | **Fixed and stable.** FS1P answers "2 percent per month" cited to `msa_green_tower.pdf`, every run. |
| 2.2 | R2 — document question routed to the relational source | **NOT fixed.** Two contributing bugs fixed, but FS1 still flips between the correct RAG answer and a fabricated `818.000%`. See §2.2 — this is the most important caveat in this report. |
| 3.1 | `veda/firewall.py` claimed a pure move, never verified | **Claim REFUTED.** 1 of 6 gates is a pure move. Two defects found, one verified by execution: qualifier salvage is dead on every query that reaches it. See §3.1. |
| 3.2 | `MULTISOURCE_ROUTING_SHADOW` was not a kill switch | **Fixed** via option (b). Option (a) is impossible — the cited battery cannot detect this class. |
| 4 | `pytest` installed nowhere; the suite catching the dead-router bug had never run | **Fixed.** In both images; 55 tests passing where 11 passed before. |
| 5.1 | XS2's "redundant" `query_understanding` call | **Not removed, deliberately.** It is a correctness gate. See §5.1. |
| 5.2 | six uncapped SLM sites | **Partially done.** All six given timeouts; output caps removed from the two plan sites after they changed an answer. See §5.2. |
| 5.3 | inverted timeout ladder | **Fixed.** Innermost-tightest, every layer under nginx. |

---

## 0. Config plumbing

### 0.1 The typed env layer

`veda_core/config.py:40-130` — `ConfigError`, `_env_present`, `_env_str`, `_env_int`,
`_env_float`, `_env_bool`. Contract: the literal already in the file stays the default;
absent **or empty** means default (compose expresses "unset" as empty); a **malformed value
raises `ConfigError` at import time naming the key** rather than silently running the old
default. `_env_bool` accepts `1/0/true/false/yes/no/on/off`, case-insensitive.
`_ENV_KEYS_READ` records every key consulted.

### 0.2 Constants converted

| constant | `file:line` | env key | default kept |
|---|---|---|---|
| `TOP_K` | `veda_core/config.py:422` | `VEDA_TOP_K` | 15 |
| `SLM_TEMPERATURE` | `veda_core/config.py:477` | `SLM_TEMPERATURE` | 0.3 |
| `SLM_TIMEOUT_SECS` | `veda_core/config.py:496` | `SLM_TIMEOUT_SECS` | 60 (was 240 — §5.3) |
| `TOP_K_TO_LLM` | `veda_core/config.py:511` | `VEDA_TOP_K_TO_LLM` | 6 |
| `QUERY_ROUTER_ENABLED` | `veda_core/config.py:700` | `VEDA_QUERY_ROUTER_ENABLED` | True |
| `IR_JOIN_FREE_ENABLED` | `veda_core/config.py:1754` | `VEDA_IR_JOIN_FREE_ENABLED` | True |
| `RETRIEVAL_INTENT_BOOST_SCALE` | `veda_core/config.py:2112` | `RETRIEVAL_INTENT_BOOST_SCALE` | 0.12 |
| `FAST_PATH_ENABLED` | `veda_core/config.py:2643` | `VEDA_FAST_PATH_ENABLED` | True |
| `QUERY_DECOMPOSE_ENABLED` | `veda_core/config.py:2781` | `VEDA_QUERY_DECOMPOSE_ENABLED` | False |
| `HNSW_M` / `HNSW_EF_CONSTRUCTION` | `veda_core/config.py:433-434` | `VEDA_HNSW_M` / `VEDA_HNSW_EF_CONSTRUCTION` | 16 / 200 |

**`VEDA_QUERY_ROUTER_ENABLED` was an eleventh dead key the benchmark report missed.**
`QUERY_ROUTER_ENABLED` was a hardcoded `True` at `config.py:700` driving a live branch at
`veda_hybrid.py:357`. Found by the new guard, not by inspection.

Verified: `config` imports clean in-container and every value is unchanged from its literal
except `SLM_TEMPERATURE`, which became `0.0` the moment it was wired — because `.env` already
said `0`.

### The three keys that needed a decision

- **`VEDA_HNSW_M` / `VEDA_HNSW_EF_CONSTRUCTION` → WIRED** (not deleted). They are consumed at
  `CREATE INDEX` time by `ingestion/biencoder.py:73`, `ingestion/chunk_embedder.py:186`,
  `ingestion/graph_embedder.py:115`, all previously hardcoded `m = 16, ef_construction = 200`.
  `scripts/hnsw_parity_sweep.py:11` calls the pinned ef_search "and build m/ef_construction"
  the shipping params, so they are settings, not magic numbers. `.env` now states plainly
  that they are **build-time**: every site uses `CREATE INDEX IF NOT EXISTS`, so changing them
  does nothing to an existing index and applies to the next fresh build.
- **`VEDA_ENCODER_MODE` → DELETED.** The ENCODER_MODE switch and its
  `relgt/light_text/hybrid/ensemble` vocabulary were removed in WP3 (`veda_core/config.py:6`).
  Nothing has read it since; `ensemble` was a value from a vocabulary that no longer exists.

### 0.3 `SLM_TEMPERATURE` — wired, verified at the wire, determinism proven

Wired at `veda_core/config.py:477`; `.env` keeps `SLM_TEMPERATURE=0`.

**Wire assertion** (`reports/raw/fixes_wire_temp.py`) — intercepts the POST to `OLLAMA_URL`
and reads `options.temperature` off the actual request body, for the two named call sites:

```
config.SLM_TEMPERATURE = 0.0  (env SLM_TEMPERATURE='0')
PASS rag_synthesis    wire options.temperature = 0.0  model=qwen2.5:7b-instruct
PASS ir_emit          wire options.temperature = 0.0  model=qwen2.5:7b-instruct
WIRE ASSERTION: PASS
```

**Determinism, controlled before/after.** FS2 (`summarize the employee handbook leave
policy`), 5 reps each. "Before" reproduces the old behaviour exactly by setting the
now-live key to its old literal:

| | distinct answers / 5 | completion tokens |
|---|---|---|
| `SLM_TEMPERATURE=0.3` (old effective value) | **5 of 5** | 137 / 126 / 102 / 91 / 139 |
| `SLM_TEMPERATURE=0` (`.env`) | **1 of 5 — byte-identical** | 106 / 106 / 106 / 106 / 106 |

Raw: `reports/raw/fixes_fs2_temp03.jsonl`, `reports/raw/fixes_fs2_temp0.jsonl`.

Comments corrected in `.env` and in `CLAUDE.md:45-56` (which claimed the key already made
runs deterministic).

**Still non-deterministic, out of the scope this task named:** `nl_answer` and
`insight_engine` (`query/result_explainer.py:809,1200`), `nl_simplifier.py:103` and
`multi_summary` (`veda_hybrid.py:555`) pass a hardcoded `temperature=0.1`. This is not
cosmetic — it is why XS1's prose still contradicts its own row count between reps (§2.2).

### 0.4 The drift guard

`tests/test_env_wiring.py` (new). For every key in `.env` it searches all code files
(`.py .yml .yaml .sh .ini .conf .sql`, excluding `env_drift.py` itself) for a
word-boundary match and fails with the list of orphans. Deliberately **excludes `.md`** — a
key documented in an AGENTS.md but read by no code is the failure being guarded against. A
third test fails if a generic `VEDA_*` bridge ever appears, because the orphan check assumes
there is none.

It immediately found **four more dead keys** beyond the report's ten:

| key | resolution |
|---|---|
| `POSTGRES_HOST`, `POSTGRES_PORT` | **deleted** — read by nothing: not by any module, not by compose, and not by the postgres image (which reads only `POSTGRES_DB/USER/PASSWORD`). The app reaches Postgres only through PgBouncer. |
| `MODEL_CACHE_DIR` | **deleted** — `/models` is fixed by the compose mount and the explicit `HF_HOME`/`HF_HUB_CACHE`. |
| `SOURCE_REGISTRY_DB` | **renamed → `SOURCE_REGISTRY_DB_NAME`** — `config/settings/base.py:116` reads `SOURCE_REGISTRY_DB_NAME`. The old spelling matched only as a *substring*, which is why my first hand-rolled scan reported it as live and the word-boundary guard did not. Setting the old name did nothing. |

### 0.5 Flags that were shipping ON with no `.env` entry

Written into `.env` at their current defaults — zero behaviour change, the point is
visibility: `ROUTING_CARDS_ENABLED=1`, `INTENT_SQL_ENTITY_COVERAGE_ENABLED=1`,
`ENTITY_COVERAGE_CONFIDENCE=0.6`, `QUERY_UNDERSTANDING_MIN_CONFIDENCE=0.5`,
`SLM_NUM_CTX=4096`, `ROUTING_EDGE_MIN_JACCARD=0.05`, `ROUTING_CHUNK_KIND_OFFSET=0.07`, plus
`SLM_TIMEOUT_SECS` and the new `ROUTING_AUTHORITATIVE_MODES`.

`ROUTING_EDGE_MIN_JACCARD` and `ROUTING_CHUNK_KIND_OFFSET` are flagged in `.env` as
**empirically fitted to this one deployment** per the comments at their definitions.

---

## 1. Duplicate embedding round-trips

**The finding was right about the waste and wrong about the mechanism**, which changed the fix.
A thread-annotated probe (`reports/raw/fixes_thread_probe.py`) shows two different shapes:

```
/encode_dense    sha=86b3a1a94614 thread=MainThread              355ms
/encode_dense    sha=86b3a1a94614 thread=MainThread              223ms   <- sequential
/encode_sparse   sha=2a8a922946cc thread=ThreadPoolExecutor-0_1 1823ms
/encode_sparse   sha=2a8a922946cc thread=MainThread             1827ms   <- CONCURRENT
```

A plain dict memo fixes the first and **not** the second: both threads miss it and both still
compute. So the fix is **single-flight** — the first caller for a key computes, the rest block
on an `Event` and reuse the result.

`veda_core/ingestion/m3_encoder.py:69-190` — `request_embed_cache()` (a ContextVar scope),
`_InFlight`, and `_metal_post` as a single-flight wrapper over `_metal_post_uncached`. Opened
once per request at `veda_core/veda_hybrid.py:1511`.

Two properties verified rather than assumed:
- **Worker threads see the ContextVar** — the probe shows the marker visible inside
  `ThreadPoolExecutor-0_1`, because `veda_core/context.with_context` carries the parent
  context into the pool. Without this the cache would have been a no-op.
- **Outside a request the cache is `None`** — a plain pass-through, so ingestion (which
  encodes thousands of *distinct* texts) never accumulates entries.

Failures are never cached: the owner drops the key and every waiter re-raises, so each caller
still takes its own documented CPU fallback.

**Verified** (`reports/raw/fixes_embed_{before,after}.jsonl`), probing *below* the cache so
these are real network round-trips:

| query | wire round-trips before | after | repeated payloads after |
|---|---:|---:|---|
| DB1 | 6 | **3** | none |
| DB2 | 6 | **3** | none |
| DL1 | 2 | **1** | none |
| FS2 | 3 | **2** | none |

Every payload hash appears exactly once. SQL and answers byte-identical before vs after on
all four.

**Reported, not fixed (as instructed):** `/encode_sparse` is called with **409 texts** — the
column catalog, learned-sparse-encoded at query time, on every query. It is computed from
`column_embeddings_v2` content that ingestion already has, so it is **precomputable at
ingestion**; nothing about it depends on the question. Not implemented in this pass.

---

## 2. Routing correctness

### 2.1 R1 — a pinned document scope was read as a permission denial · FIXED

`veda_core/veda_hybrid.py:815-835`. The pre-check computed
`_denied = all_ready_source_ids() - set(sids)` — "every ready source outside the scope I was
handed is one the caller is denied". That conflates an **RBAC-narrowed** scope with a
**caller-pinned** one.

The fix uses the signal that already exists and already means exactly this:
`RequestContext.allowed_resources`, documented in `veda_core/context.py` as `None` == "no
restriction". Denial is now derived only from RBAC; with no RBAC scope nothing is denied and
the pre-check correctly does not fire.

**Verified:** FS1P (`what is the late fee percentage`, pinned to source 3) goes from
`route="no_access"` / *"You don't have permission to access this data"* to
**`2 percent per month.` cited to `msa_green_tower.pdf`** — in every run since, including
3/3 in the stability check and both reps of the final pass.

### 2.2 R2 — a document question routed to the relational source · NOT FIXED

Two real bugs were found and fixed, and FS1 still is not reliable. This is the most important
caveat in this report.

**What the finding said** — exclude document/filesystem sources from the SQL head — does not
describe the failure. The measured mechanism is different, and there are three layers:

1. **The SQL head resolved the wrong source.** Routing decided `SINGLE/source 2`, but
   execution failed with *"this source has no SQL endpoint (no host configured)"* — source
   **3**. `dispatch` (`veda_core/query/source_coordinator.py:995-1012`) passes the routed
   source to the agent as a kwarg but never narrows the ambient `RequestContext`, so
   `get_db_config()` still resolved the primary. **Not fixed, deliberately:** I measured what
   narrowing would produce — source 2 answers this question
   *"All 100 rows shown have a late_fee of 0.0"*, from `accounts_userinvoice`. Narrowing would
   convert a refusal into a fabrication. The mismatch is a real latent bug and is reported
   here rather than "fixed" into a worse answer.
2. **The intent router had no guard.** Implemented as instructed:
   `veda_hybrid._guard_sql_head` + `_primary_is_document_source` (`veda_core/veda_hybrid.py:343-420`)
   demote a bare `sql` intent to `rag` when the primary source cannot run SQL, reading the
   source profiles the api tier already sets. Verified load-bearing: with the guard
   `classify` returns `('rag', [])`, without it `('sql', [])`. `hybrid` is left alone — that
   lane already consults `_scope_has_structured_source()`.
3. **The coordinator pre-empted the guard**, which §3.2 fixed.

**And it is still not reliable.** With all three in place, FS1 at full scope:

| run | route | answer |
|---|---|---|
| stability check, 3 reps, FS1 alone | `rag` 3/3 | `2 percent per month.` cited `msa_green_tower.pdf` |
| final pass rep 1 | `rag` | `2 percent per month.` cited `msa_green_tower.pdf` |
| final pass rep 2 | `federated` | **`The late fee percentage is 818.000%`** — fabricated, no citation |

The correct outcome depends on the federated planner's JSON being **rejected** so the query
falls through to RAG. When the planner happens to emit an accepted plan, the federated SQL
head answers with a number summed out of `services_valuebundlepricing` — master's exact
failure. The outcome also depends on what ran before it in the same process.

So: **FS1 is not fixed.** The target ("FS1 and FS1P both answer 2 percent") is met for FS1P
every time and for FS1 only sometimes. Making it deterministic needs routing not to send a
document question to the federated SQL head in the first place — which is a routing-evidence
change the findings do not specify, and I did not invent one.

**Secondary — the false lifecycle message · FIXED.** `veda_core/veda/exec_records.py:46-80`.
`FAILED` claimed *"This data source could not be reached"* unconditionally; the source was
reached and the SQL failed, and DB1/DB2 executed against it seconds earlier in the same run.
Base wording is now the neutral, always-true *"This data source did not return a result"*,
and `safe_status_message(status, error_class)` selects the sharper sentence from the
`error_class` the record already carries (`transient` → could not be reached, `permanent` →
could not run this question). Used by both projections (`:129` and `:223`). Verified in the
trace: the lifecycle event now reads *"a data source: This data source did not return…"*.

**Also observed, not in the findings:** `federated_struct_plan` is frequently issued **twice
per query** with an identical prompt (visible in FS1 and in `reports/raw/fixes_after.jsonl`).
The benchmark saw this on master too and called it intermittent; it is reproducible here.

---

## 3. Unverified changes in every query's path

### 3.1 `veda/firewall.py` — claim REFUTED, two defects

**The "pure move, zero behaviour change" claim is false.** One of six gates is a pure move.
A gate-by-gate comparison against master's original call sites:

| gate | master impl | branch impl | verdict |
|---|---|---|---|
| `value_grounding` | `veda/validation.py:575` | `firewall.py:222` (body changed) | CHANGED |
| `qualifier` | `veda/validation.py:389` | `firewall.py:234` + new `_ir_vs_sql` `firewall.py:229-233` | CHANGED |
| `alignment` | `veda/intent_sql_alignment.py:186/211/267/364` | `firewall.py:251-273` | CHANGED |
| `ir_equivalence` | `veda/ir_equivalence.py:168` (body byte-identical) | `firewall.py:282` | CHANGED (placement) |
| `rbac` | `veda/rbac_filter.py:227` (byte-identical) | `firewall.py:297` | **IDENTICAL — the one true pure move** |
| `ast_parameterize` | `veda/validation.py:10` | `firewall.py:304` | IDENTICAL for the pipeline |

**Defect 1 — qualifier salvage is dead on every query that reaches it. VERIFIED BY EXECUTION.**

`veda_core/veda/pipeline.py:2192` wraps the gate's detail in a list:

```python
missing = (_fv.detail if isinstance(_fv.detail, list) else [_fv.detail]) if not ok_q else []
```

but `qualifier_completeness` returns a **bare string** (`veda/validation.py:658`:
`return (not missing), (missing[0] if missing else None)`), and `missing` is then passed
straight to `referent_tables` at `pipeline.py:2215`. That function opens
`token = (token or "").lower().strip()` outside any try. Master passed the string
(`master pipeline.py:1797`).

Executed repro (`reports/raw/fixes_repro_qualifier_salvage.py`), in-container:

```
STRING (what master passed)        -> returned list len=1
LIST   (what the branch passes)    -> RAISED AttributeError: 'list' object has no attribute 'lower'
```

The exception is swallowed by the caller's `except Exception: _refs = []` (`pipeline.py:2226`),
so `_refs` is **always empty** and both salvage paths are unreachable: the re-anchor retry
(`pipeline.py:2239`) and the referent clarify (`pipeline.py:2286`). Master re-anchored and
retried; the branch refuses. `value_referents` survives only by accident —
`query/resolution.py:184` does `str(v or "").lower()`, which normalises `["payment"]` back to
`"payment"`. The trace and feedback payload also changed shape: `pipeline.py:2195` now records
`"['payment']"` rather than `"payment"`.

**One-line fix, NOT applied** (outside the six groups, and the run was stopped): pass the
string, or `missing[0]`, to `referent_tables`.

**Defect 2 — `_tier2_validate` is dead code. VERIFIED BY GREP.**

`veda_core/veda_hybrid.py:2946` defines it; a repo-wide search finds **no caller** — only its
own comment and a docstring mention in `scripts/eval_per_source_battery.py:29`. Master called
it twice, at `veda_hybrid.py:2916` and `:3123`. Two checks are therefore unreachable: the
`SEMANTIC_VALIDATION_ENFORCE` block (latent — the flag is `False` on both trees) and the
branch's **own** `_constraint_kind`/`_sql_keeps_constraint` threshold-and-negation check
(`veda_hybrid.py:2991-2993`), added for "properties with more than 3 floors", which has
therefore never run.

**Four further behaviour changes — reported, NOT independently verified by me.** These come
from a static read and are reasoned from code, not measured: the Tier-2 shared-planner
(`veda_hybrid.py:3651`) and IR (`:3698`) paths each gained four gates they did not have; the
federated head gained a firewall it never had (branch `query/federated_route.py:840,865` vs
master's ungated `compose_federated`), including `validation.py:481`
`federated_qualifier_completeness`, which does not exist on master; a `DIM_CLARIFY` carve-out
(`firewall.py:264-270`) lets grouped complete-IR SQL through where master refused
unconditionally (`master pipeline.py:1983`); and `_ir_vs_sql` (`firewall.py:109-151`) is
entirely new, emitting a refusal class master could not produce.

**Verdict:** the firewall should not be merged on the strength of its own comment. Defect 1
is a live correctness regression with a one-line fix; defect 2 means a guard the branch
added for a named query has never executed.

### 3.2 `MULTISOURCE_ROUTING_SHADOW` restored as a kill switch · FIXED (option b)

**Option (a) was rejected on evidence, not preference.** The branch justified making
SINGLE/MULTI authoritative by citing `scripts/eval_cross_source_battery.py` as the
SINGLE-vs-legacy regression test the `.env` comment demanded. It is not that test:
`scripts/eval_cross_source_battery.py:92` compares **pinned vs unpinned** and explicitly
**passes when the unpinned run returns a typed refusal** (`verdict: "typed_refusal"`) — which
is exactly the FS1/XS1 failure mode the flip introduced. It never runs the legacy engine, so
it cannot compare SINGLE against it. Option (a)'s precondition is false.

**Option (b) implemented.** `veda_core/config.py:1159-1176` adds
`ROUTING_AUTHORITATIVE_MODES`, default **empty** = no mode overrides SHADOW.
`veda_core/veda_hybrid.py:1031-1043` replaces the hardcoded
`SHADOW and not is_multi and not is_single` with a check against that setting. `.env` carries
`ROUTING_AUTHORITATIVE_MODES=` with the measurement that chose it.

**Measured on the 8-query set, same `.env` otherwise:**

| setting | outcome |
|---|---|
| `SINGLE,MULTI` (the branch's hardcoded behaviour) | 2 refusals (FS1, XS1) |
| *(empty — restored kill switch)* | 0 refusals; FS1 answered "2 percent" cited correctly in that run |

Unit-tested: `tests/test_fixes_2026_09_23.py` pins that SHADOW is honoured for SINGLE and
MULTI when no mode is opted in, that only opted-in modes override it, and that non-ROUTED
decisions are never authoritative.

---

## 4. Test suite

### 4.1 `pytest` — now installed

`requirements/test.txt` (new), installed in both images
(`docker/Dockerfile.api:26-29`, `docker/Dockerfile.inference:31-34`). Both rebuilt and
verified: `python -m pytest --version` → `pytest 9.1.1` in each, and the suite runs **inside
the inference image**: `26 passed`.

This is the process finding: `tests/test_source_coordinator.py` catches the dead-router bug
24 times over and had never been executed once, because pytest was in no requirements file
and in neither image.

### 4.2 / 4.3 Suite state, per file

| file | before (`f768040`) | after |
|---|---|---|
| `tests/test_source_coordinator.py` | **14 failed / 11 passed** | **0 failed / 23 passed** |
| `tests/test_env_wiring.py` | *did not exist* | **3 passed** |
| `tests/test_fixes_2026_09_23.py` | *did not exist* | **29 passed** |
| **total** | 14 failed / 11 passed | **0 failed / 55 passed** |

R5 fixed by restoring `import config as _cfg_mod` (`tests/test_source_coordinator.py:12-18`),
which the branch deleted while leaving 14 references.

Three remaining failures were **not** masked: they asserted the `SourceAdapter` /
`ExecutionRequest` dispatch branch, which this branch deliberately deleted
(`veda_core/query/source_coordinator.py:938-940`, "the adapter branch imported
query/source_adapters, a module deleted in the P2-3 cleanup — removed with its flags"). Two
were replaced by `test_dispatch_always_uses_the_bare_agent_not_the_adapter`, which pins the
*current* contract; the third was removed with a comment saying why. The stale docstring on
`_resolve_executable` that still promised the removed behaviour was corrected.

New coverage for the changes that alter answers: the typed env layer (10 keys parametrised,
empty/absent, malformed-raises), `SLM_TEMPERATURE` reaching both named call sites,
`_guard_sql_head`, `_primary_is_document_source`, `ROUTING_AUTHORITATIVE_MODES` /
`_effective_shadow`, and `_value_in_engine_store` — hit, miss, **fail-open**, and the
short-token skip.

---

## 5. Bounded wins

### 5.1 XS2's `query_understanding` call · NOT REMOVED, deliberately

It is not decoration. `veda_core/query/federated_route.py:678-689` builds the IR the
**firewall checks the federated plan against, derived from the QUESTION rather than from the
plan**, and `_question_gate` (`:808-834`) refuses a plan whose shape disagrees with it.
Removing the call makes `qir` `None`, which makes `_question_gate` return `None` — the gate is
disabled entirely.

Its docstring records the measured failure it exists for: *"how many maintenance records and
how many amenities are there"* was answered *"4 … and an average of 10.88 amenities"* (truth 5
and 7) because the planner chose AVG for a "how many" and `_fed_ir_from_structured` validated
the planner against itself.

Verifying byte-identical answers on XS1/XS2 would only show those two queries do not need the
gate — it cannot license removing a guard that exists for others. **The safe alternative,
not implemented:** the call is independent of the plan (it needs only the question and the
per-source schema), so it could run **concurrently** with `federated_struct_plan` instead of
after it, hiding its latency without weakening the gate.

### 5.2 Uncapped SLM sites · PARTIALLY DONE

Timeouts added to all six, all well under nginx's 120s:

| site | purpose | `num_predict` | `timeout` |
|---|---|---|---|
| `query/operation_classifier.py:186` | `operation_classify` | 128 (measured 27-43c) | 45s |
| `query/semi_join_planner.py:133` | `semi_join_classify` | 128 (measured 16c) | 45s |
| `query/doc_data_planner.py:88` | `doc_data_ground` | 256 | 45s |
| `veda_hybrid.py:555` | `multi_summary` | already bounded | 45s |
| `query/federated_route.py:381` | `federated_plan` | **none — see below** | 60s |
| `query/federated_route.py:513` | `federated_struct_plan` | **none — see below** | 60s |

**The output cap on the two plan sites was removed after it changed an answer.** With
`num_predict=384`, `federated_struct_plan` returned a plan the question-gate accepted, so FS1
took the federated SQL head and answered `818.000%` — fabricated — in **2/2 reps**. Without
the cap the plan is rejected and the query falls through to RAG. The finding (§6.6) is about
unbounded **tail latency**, which the timeout closes without touching what the model emits, so
the timeouts stayed and the caps went. Recorded at `federated_route.py:374-381`.

### 5.3 The timeout ladder · FIXED

Was inverted — the outermost hop was the tightest, so a single SLM call could outlive the
client connection waiting for it. Now innermost-tightest, every layer under nginx:

| rung | before | after | `file:line` |
|---|---:|---:|---|
| one SLM call | 240s | **60s** | `veda_core/config.py:496` (+ per-site 45-60s) |
| inference request | 300s | **95s** | `.env` `INFERENCE_TIMEOUT_S` |
| gunicorn worker | 600s | **110s** | `.env` `GUNICORN_TIMEOUT` |
| nginx `proxy_read_timeout` | 120s | **120s** (unchanged) | `docker/nginx.conf:66` |

nginx was deliberately **not raised** — the app layers were brought under it, so the
client-facing bound does not get looser. Comment at `docker/nginx.conf:60-72` updated.

**One outlier, flagged not changed:** `veda_core/veda/generation.py:418` passes an explicit
`timeout=120`, now above the 95s inference budget. It was a deliberate tightening from the
old 240s default; lowering it further blind would be a guess, so it is reported instead.

---

## Before / after — the 10-query set

Status and content. Latency omitted (see the note at the top). Raw:
`reports/raw/fixes_before.jsonl`, `reports/raw/fixes_after.jsonl`.

| query | before | after | after answer | cited |
|---|---|---|---|---|
| DB1 | answered | answered | `accounts_paymenttransaction`, unchanged | — |
| DB2 | answered | answered | unchanged (0 rows) | — |
| DL1 | answered | answered | unchanged | — |
| DL2 | answered | answered | unchanged | — |
| FS1 | **refused** | **answered, UNSTABLE** | `2 percent` (rag) *or* `818.000%` (federated) — see §2.2 | sometimes |
| FS2 | answered | answered | unchanged; now byte-identical across reps | Employee Handbook |
| XS1 | **clarify** | **answered** | `6 vendors` *or* `2 vendors` from the same 6-row result — prose still non-deterministic (`nl_answer` at 0.1) | — |
| XS2 | answered | answered | unchanged | — |
| FS1P | **refused (`no_access`)** | **answered** | `2 percent per month.` — stable | `msa_green_tower.pdf` |
| FS2P | answered | answered | unchanged | Employee Handbook |

SLM calls and tokens (wire-measured, exact counts):

| query | before calls/tokens | after calls/tokens |
|---|---|---|
| DB1 | 2 / 1414 | 2 / 1414 |
| DB2 | 1 / 102 | 1 / 102 |
| DL1 | 1 / 872 | 1 / 872 |
| DL2 | 1 / 850 | 1 / 850 |
| FS1 | 1 / 104 *(refused early)* | 4 / ~2139 |
| FS2 | 1 / 1233 | 1 / 1222 |
| XS1 | 1 / 406 *(clarified early)* | 4 / ~1890 |
| XS2 | 5 / 2553 | **4 / 2219** |
| FS1P | 0 / 0 *(refused before any call)* | 1 / 476 |
| FS2P | 1 / 1212 | 1 / 1222 |

FS1/XS1/FS1P cost more because they now *answer* instead of refusing. XS2 dropped one call
(the `decompose` the authoritative-routing path used to make).

---

## Appendix — full `.env` audit

Every key currently in `.env`, and what reads it (generated by re-running the guard's own scan):

| key | status | read by (first reader) |
|---|---|---|
| `DJANGO_SETTINGS_MODULE` | already live | `config/asgi.py` |
| `SECRET_KEY` | already live | `config/settings/prod.py` |
| `DEBUG` | already live | `veda_core/utils/logger.py` |
| `ALLOWED_HOSTS` | already live | `config/settings/dev.py` |
| `POSTGRES_DB` | already live | `storage_adapters/reader.py` |
| `POSTGRES_USER` | already live | `storage_adapters/reader.py` |
| `POSTGRES_PASSWORD` | already live | `apps/core/views.py` |
| `PGBOUNCER_HOST` | already live | `apps/core/views.py` |
| `PGBOUNCER_PORT` | already live | `apps/core/views.py` |
| `SOURCE_REGISTRY_DB_NAME` | **newly added** (was implicit) | `config/settings/base.py` |
| `REDIS_BROKER_URL` | already live | `apps/core/views.py` |
| `REDIS_CACHE_URL` | already live | `veda_core/veda_hybrid.py` |
| `SLM_BACKEND` | already live | `veda_core/config.py` |
| `SLM_MODEL_NAME` | already live | `veda_core/config.py` |
| `METAL_EMBED_URL` | already live | `veda_core/ingestion/m3_encoder.py` |
| `OLLAMA_URL` | already live | `veda_core/config.py` |
| `VEDA_SLM_CHAT_URL` | already live | `apps/sources/item_profiler.py` |
| `VLLM_URL` | already live | `veda_core/config.py` |
| `NL_SUMMARY_MODEL` | already live | `veda_core/config.py` |
| `CHATBOT_CLASSIFY_MODEL` | already live | `chatbot/llm.py` |
| `SLM_TEMPERATURE` | **newly wired** (was dead) | `veda_core/config.py` |
| `VEDA_TOP_K` | **newly wired** (was dead) | `veda_core/config.py` |
| `VEDA_TOP_K_TO_LLM` | **newly wired** (was dead) | `veda_core/config.py` |
| `VEDA_QUERY_ROUTER_ENABLED` | **newly wired** (was dead) | `veda_core/config.py` |
| `VEDA_IR_JOIN_FREE_ENABLED` | **newly wired** (was dead) | `veda_core/config.py` |
| `VEDA_FAST_PATH_ENABLED` | **newly wired** (was dead) | `veda_core/config.py` |
| `VEDA_QUERY_DECOMPOSE_ENABLED` | **newly wired** (was dead) | `veda_core/config.py` |
| `VEDA_HNSW_M` | **newly wired** (was dead) | `veda_core/config.py` |
| `VEDA_HNSW_EF_CONSTRUCTION` | **newly wired** (was dead) | `veda_core/config.py` |
| `VEDA_HNSW_EF_SEARCH` | already live | `veda_core/config.py` |
| `ROUTING_CARDS_ENABLED` | **newly added** (was implicit) | `veda_core/config.py` |
| `INTENT_SQL_ENTITY_COVERAGE_ENABLED` | **newly added** (was implicit) | `veda_core/config.py` |
| `ENTITY_COVERAGE_CONFIDENCE` | **newly added** (was implicit) | `veda_core/config.py` |
| `QUERY_UNDERSTANDING_MIN_CONFIDENCE` | **newly added** (was implicit) | `veda_core/config.py` |
| `SLM_NUM_CTX` | **newly added** (was implicit) | `veda_core/config.py` |
| `SLM_TIMEOUT_SECS` | **newly added** (was implicit) | `veda_core/config.py` |
| `ROUTING_EDGE_MIN_JACCARD` | **newly added** (was implicit) | `veda_core/config.py` |
| `ROUTING_CHUNK_KIND_OFFSET` | **newly added** (was implicit) | `veda_core/config.py` |
| `INFERENCE_WORKERS` | already live | `docker-compose.yml` |
| `HF_HUB_OFFLINE` | already live | `veda_core/run_homzhub_query.sh` |
| `TRANSFORMERS_OFFLINE` | already live | `veda_core/run_homzhub_query.sh` |
| `VEDA_INTERNAL_HOST` | already live | `veda_core/config.py` |
| `VEDA_INTERNAL_PORT` | already live | `veda_core/config.py` |
| `VEDA_INTERNAL_DBNAME` | already live | `veda_core/config.py` |
| `VEDA_INTERNAL_USER` | already live | `veda_core/config.py` |
| `VEDA_INTERNAL_PASSWORD` | already live | `veda_core/config.py` |
| `INFERENCE_URL` | already live | `apps/core/views.py` |
| `INFERENCE_TIMEOUT_S` | already live | `apps/query/inference_client.py` |
| `GUNICORN_TIMEOUT` | already live | `docker/entrypoint.api.sh` |
| `MULTISOURCE_ROUTING_ENABLED` | already live | `veda_core/config.py` |
| `MULTISOURCE_ROUTING_SHADOW` | already live | `veda_core/config.py` |
| `ROUTING_AUTHORITATIVE_MODES` | **newly added** (was implicit) | `veda_core/config.py` |
| `REQUIRED_SOURCE_ESCALATION_ENABLED` | already live | `veda_core/config.py` |
| `ROUTING_PERMISSION_DENY_GAP` | already live | `veda_core/config.py` |
| `SEMANTIC_PARALLEL_QWEN_ENABLED` | already live | `veda_core/config.py` |
| `SEMANTIC_MAX_PARALLEL_REQUESTS` | already live | `veda_core/config.py` |

| key | status | why |
|---|---|---|
| `VEDA_ENCODER_MODE` | **deleted** | ENCODER_MODE switch removed in WP3 (`veda_core/config.py:6`); `ensemble` was a value from a vocabulary that no longer exists |
| `POSTGRES_HOST` | **deleted** | read by nothing — not by any module, not by compose, and not by the postgres image (which reads only POSTGRES_DB/USER/PASSWORD). The app reaches Postgres only via PgBouncer |
| `POSTGRES_PORT` | **deleted** | as above |
| `MODEL_CACHE_DIR` | **deleted** | read by nothing; `/models` is fixed by the compose mount and the explicit HF_HOME/HF_HUB_CACHE env |
| `SOURCE_REGISTRY_DB` | **renamed** -> `SOURCE_REGISTRY_DB_NAME` | `config/settings/base.py:116` reads `SOURCE_REGISTRY_DB_NAME`; the old spelling matched only as a substring and did nothing |

---

## Deliberately not fixed

| item | why |
|---|---|
| FS1 at full scope (§2.2) | Needs routing not to send a document question to the federated SQL head. That is a routing-evidence change the findings do not specify. |
| `dispatch` routed-source vs executed-source mismatch (§2.2) | Real bug. Fixing it in isolation converts FS1's refusal into a fabrication — measured. |
| `veda/firewall.py` defects 1 and 2 (§3.1) | Found and verified, but fixing them is outside the six groups and the run was stopped. Defect 1 is a one-line fix. |
| XS2's `query_understanding` (§5.1) | It is the federated correctness gate. |
| `/encode_sparse` over 409 texts per query (§1) | Instructed to report, not fix. It is precomputable at ingestion. |
| `nl_answer`/`insight_engine`/`multi_summary` at `temperature=0.1` | Outside the three call sites §0.3 named. It is why XS1's prose still varies. |
| `generation.py:418` `timeout=120` (§5.3) | Above the new inference budget, but lowering it blind would be a guess. |

## Operational note

`.env` changed, and env is read at container **create** time. The running stack still has the
old values — `docker compose up -d` (not `restart`) is needed to pick them up. The `api` and
`inference` images were rebuilt for pytest and also need recreating.

## Commits

**None.** `CLAUDE.md` in this repo says never to run `git commit` or `git push`, and that the
permission does not carry forward within a session. All changes are uncommitted in the
working tree, grouped as described above. Two scratch worktrees were created for baselines
and can be removed with `git worktree remove`:
`/Users/ekesel/samta/veda-master-bench`, `/Users/ekesel/samta/veda-prefix-bench`.
