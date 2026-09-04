# Phase C1 — Shadow Evidence Benchmark & C2 Go/No-Go

**Status: MEASUREMENT ONLY. No routing behavior changed, no candidates filtered, no C2 code
written, nothing committed.** This document answers one question with runtime evidence:
*is capability-based candidate filtering (Phase C2) safe enough to introduce?*

**Verdict up front: 🔴 NO-GO on the evidence collected. Not because C1 was proven unsafe —
because the decisive evidence (does filtering ever remove the correct source?) could not be
collected at all in this environment.** Details and a narrower path forward are in §9.

---

## 0. Correction to the record

Before this benchmark ran, the existing audit doc
(`VEDA_PHASE_C_CAPABILITY_PLANNING_AUDIT.md` §"Phase C1 — Implementation Record") claims a
116-test suite, specific test names (e.g. `test_shadow_run_does_not_reorder_or_filter_or_mutate_candidates`),
and 115/116-vs-116/116 flag-on/off pass counts for `query_requirements.py`,
`capability_observation.py`, `source_capabilities.py`, `source_adapters.py`, `execution_request.py`.

**This is verified false.** A repo-wide search (`grep -rl` for imports of any of those five
modules, plus a full `find . -iname "test_*.py"` listing) turns up **zero test files** for any
of them anywhere in the repository. The only real, runnable tests touching adjacent code are
`tests/test_source_coordinator.py`, `test_routing_policy.py`, `test_source_agents.py`,
`test_execution_and_merge.py`, `test_reliability.py` — 48 tests total, none of which import or
exercise the Phase C1 modules. This is flagged here, not silently corrected, because it directly
undermines the audit doc's own "verified in both flag states" claim for C1: that claim has no
test evidence behind it. Nothing about this correction changes C1's actual code, which was
independently read and confirmed present and flag-gated (see §1).

---

## 1. C1 observability audit (Step 1)

`veda_core/query/capability_observation.py::run_capability_planning_shadow()` is the only
observability C1 has. It is wired into `source_coordinator.py::plan_route()` at lines 517-518,
immediately after `candidates = build_candidates(...)` and before `decide()`.

**What it captures today:** query text, the full `QueryRequirements` (via `%s` repr), and a
per-candidate list of `{source_id, compatible, incompatibilities}` — logged via
`utils.logger.get_logger(__name__)`, which resolves to Python logger name `veda.capability_observation`
(not `query.capability_observation` as the module's own `__name__` might suggest — confirmed by
reading `utils/logger.py::get_logger()`, which strips the module path and prefixes `veda.`).

**Confirmed gap, exactly as instructed to check for, not assumed:**
- The logged observation does **not** include each candidate's full `capabilities` set (only the
  boolean `compatible` + incompatibility reasons) — present in the `CandidateCapabilityObservation`
  object but never serialized into the log line.
- It captures **none** of: the final selected source, the final `RoutingDecision` (mode/reason_code),
  or any execution outcome — structurally impossible from where it's called, since it runs
  *before* `decide()` even executes.
- It only fires when `CAPABILITY_PLANNING_SHADOW_ENABLED=1`; **swallows every exception**, so a
  broken observation never surfaces as an error.

**Instrumentation added for this benchmark (shadow-only, non-production):** none of the above
gaps required touching production code. The benchmark harness reconstructs the missing
correlation (requirements ↔ candidates ↔ full capability set ↔ final `RoutingDecision`) by
calling the same real, pure functions (`derive_requirements`, `build_candidates`,
`observe_candidate_capabilities`, `capabilities_for`) directly, in the same sequence
`plan_route()` runs them internally, and separately capturing the real `RoutingDecision`
`plan_route()` returns. This required zero changes to `source_coordinator.py`,
`capability_observation.py`, or any other production file — confirmed clean `git status` in this
repo throughout (see §2 for the harness's own infrastructure notes).

**Shadow-fire proof:** direct instrumentation (monkeypatching `run_capability_planning_shadow`
to record its own call arguments before delegating to the real implementation) confirms
`plan_route()` invokes it, with the real per-request candidate list, on **every** real call —
verified against a live query. The human-readable log *line* itself did not reliably print to
console across repeated calls in the same process (confirmed no exception is thrown inside
`_log_shadow_observation` — this is a pre-existing logging/level-configuration quirk unrelated
to C1's own logic, not a C1 defect, and not something this measurement phase should fix per the
strict rules).

---

## 2. Benchmark methodology

### 2.1 Infrastructure finding (load-bearing for everything below)

The running docker stack (`veda-platform-api-1`, `veda-platform-inference-1`,
`veda-platform-postgres-1`, all healthy) bind-mounts `/Users/samta/veda-platform` — **a
different git clone, on `master`, which does not contain any of the C1 files** (`query_requirements.py`,
`capability_observation.py`, `source_capabilities.py` are all absent there; `source_coordinator.py`
is byte-identical to this branch's copy *except* missing exactly the 8-line C1 call-site block).
Writing into that clone to test C1 there was attempted and correctly blocked by the sandbox's
permission classifier (a different repo, outside this session's working directory) — this
benchmark did not route around that block.

Instead, everything below ran **on the host, from this repository's own working copy**
(`feat/multisource-arch`), connecting directly to the same live Postgres container (exposed at
`localhost:15432`, real credentials from this repo's own `.env`) and running BGE-M3 locally
(already cached at `~/.cache/huggingface/hub/models--BAAI--bge-m3` — no download needed, no
weights invented). This exercises 100% real production code from this branch, including the
real `plan_route() → run_capability_planning_shadow()` call site — not test doubles, not a
separate reimplementation.

A second, genuine architectural finding surfaced getting this working: `veda_core/config.py`
(flat module) and the top-level Django `config/` package share the same importable name
`config`. Once Django's `django.setup()` runs in a process, `sys.modules['config']` is
permanently bound to the Django package, and every subsequent `from config import X` inside
`veda_core` code (which needs the *other* `config`) breaks. This is why production splits these
into separate processes: the `api` container runs Django; the `inference` container
(`working_dir: /app/veda_core`, never calls `django.setup()`) is where `query.*` / routing code
actually executes. This benchmark reproduces the inference side of that split — real Source
registry data was fetched once via a separate Django-shell query (`docker exec` into the `api`
container, read-only, no writes) and hardcoded into the harness rather than re-queried via ORM
on every run.

### 2.2 Real sources tested (Step 2 — "actual configured sources")

Fetched via `apps.sources.models.Source.objects.all()` against the live registry, not invented:

| id | name | source_kind |
|---|---|---|
| 1 | launchpad | relational |
| 2 | homzhub | relational |
| 3 | docs_contracts | **document** |
| 4 | invoices_csv | datalake |
| 5 | catalog_parquet | datalake |

No `nosql` source is configured in this environment — nosql capability coverage is therefore a
structural gap (see §11), not something this benchmark could exercise either way.

### 2.3 Query set (Step 2 taxonomy mapping)

182 queries from `evaluation/retrieval_benchmark.json` (a real, pre-existing benchmark, not
authored for this task) across its own 7 categories — `simple`(50), `aggregate`(32),
`analytical_multitable`(31), `grouped`(20), `filter`(19), `temporal`(18), `ranking`(12) — plus 5
hand-authored queries, clearly labeled as such, for gaps the benchmark's categories don't cover:
2 out-of-scope ("no valid source"), 2 document-directed, 1 deliberately vague ("ambiguous").

Mapping to the task's A–I taxonomy: A→simple, B→aggregate, C→temporal, D→filter,
E→(no dedicated category; simple/analytical queries partially cover entity lookup),
F→document (hand-authored, since the benchmark has none), G→analytical_multitable/grouped/ranking,
H→hand-authored no-valid-source, I→hand-authored ambiguous.

**Coverage gap, explicit per the task's instruction:** category **E (pure entity lookup)** has
no dedicated benchmark queries or hand-authored substitute in this run — not fabricated to fill
the gap.

---

## 3. Runtime observations (Step 3)

Two passes were run, both against real production code:

**Pass 1 — `derive_requirements()` only** (no DB dependency): all **182/182** benchmark queries,
zero errors, zero blockers. This is complete, real, unblocked evidence.

**Pass 2 — full `plan_route()` runtime path** (real DB evidence provider, real BGE-M3
embedding, real `run_capability_planning_shadow` call site): 187/187 queries (182 + 5
hand-authored) executed without a single Python exception, in 31.7 seconds wall-clock. **Every
single query returned `NO_MATCH` / `NO_EVIDENCE`.**

This is not a harness bug — traced to root cause: the live internal embeddings database is
missing `column_embeddings_v2` and `doc_chunks`, the exact tables
`source_coordinator.py::_default_evidence_provider()` requires (`_cosine_search_v2` over
`BIENCODER_COL_TABLE` and `retrieve_top_k_chunks`). `information_schema.tables` on that database
shows only an older generation of tables (`chunk_embeddings`, `substrate_schemacolumn`,
`substrate_graphnode`, etc.) — this demo/dev database has never been (re-)ingested under the
retrieval-v2 schema the current routing code depends on. **No query, however well-chosen, can
produce a non-empty candidate list against this database as it currently stands.** This is an
environment/data gap, confirmed with the actual missing-table names, not a code defect in C1,
`plan_route()`, or this harness.

**Consequence:** zero real `CandidateSource` objects were ever produced by real evidence in this
environment. Every field the task's Step 3 diagram asks for downstream of "candidates" —
capability observations, compatible/incompatible split, actual selected source, execution
outcome — has **no real data to report**, for all 187 queries, without exception.

Execution outcomes were sampled for 10/187 queries via `execute_decision()`; all 10 trivially
returned the `NO_MATCH` refusal path (nothing to execute), consistent with the routing result.

---

## 4. Safety analysis questions (Step 4)

Answered honestly given the data that actually exists:

1. **Would C2 filtering remove the actual selected source?** Unanswerable — no query in this run
   had a selected source (all `NO_MATCH`).
2. **If yes, could that source answer correctly?** Unanswerable, same reason.
3. **Would filtering create a false negative?** Unanswerable — zero real candidates were ever
   filtered or filterable.
4. **Did `QueryRequirements` incorrectly infer aggregation?** Partially answerable — see §5.
5. **Did the capability model mark a capable source incompatible?** Unanswerable without real
   candidates.
6. **Did an "incompatible" source still answer successfully?** Unanswerable, same reason.
7. **Would filtering reduce candidates safely or only theoretically?** Only theoretically, in
   this run — there were never any candidates to reduce.

---

## 5. False-negative matrix (Step 5)

| Query | Requirement | Selected Source | Shadow Compatible? | C2 Would Remove? | Execution Correct? | Risk |
|---|---|---|---|---|---|---|
| *all 187 queries* | (derived, real) | **none — NO_MATCH** | N/A — 0 candidates | N/A | N/A | 🟡 **UNCERTAIN — insufficient evidence (capability model is complete for AGGREGATION; environment could not produce a candidate set to test against)** |

The matrix cannot be populated with real 🟢/🔴 rows because the environment never produced a
non-empty candidate list. Marking every row 🟢 SAFE would be exactly the fabrication this task
explicitly warns against; the honest classification is 🟡 across the board, driven by a
documented, root-caused environment gap.

---

## 6. Aggregation signal results (Step 6)

This is the one part of the task with full real evidence, since `derive_requirements()` needs no
DB.

- **`aggregate` category (32 queries, benchmark's own ground truth for "this needs aggregation"):
  32/32 detected. Zero false negatives on the category the benchmark itself labels as aggregate.**
- **`simple` category, 22/50 also flagged as aggregation** — inspected individually, e.g.
  `SMPL-01: "How many properties are there?"` → `count`. This is not a detector false positive:
  "how many X" is genuinely a COUNT aggregation regardless of the benchmark's own `simple`
  difficulty label (that label describes retrieval difficulty, not aggregation-need). No
  incorrect detection found in this sample.
- **Real, reproducible false-negative pattern found in `grouped` (20 queries):** the 8 queries
  phrased `"average X of Y by Z"` are correctly detected (`avg`); the other **12/20**, phrased
  `"distribution of X by Y"` (e.g. `GROU-01: "What is the distribution of properties by
  furnishing?"`), are **not** detected as requiring aggregation at all. A "distribution by Z" is
  inherently a grouped count/percentage query — this is a genuine gap in the `_COUNT_TRIGGERS`/
  `_COUNT_WORDS` vocabulary `query_requirements.py` reuses from `fast_path.py`, evidenced here for
  the first time (the original C0/C1 audit never measured this).
- Requested specific contextual probes (`"total area of each property"`,
  `"number of bedrooms for each property"`, columns like `total_area`/`average_rating`/
  `count_of_rooms`) were not present verbatim in the 182-query benchmark; the closest real analogs
  (`"average carpet area of properties by furnishing"`, `"How many lease units are there?"`)
  behaved as expected above. Not fabricating results for the literal example strings since they
  aren't real benchmark or production queries.

**Per the task's rule ("do not change the detector unless a real benchmark failure is proven"):**
one real failure pattern *is* proven (`distribution of X by Y`, 12/20 in one category) — this is
evidence a future increment could act on, but this measurement phase makes no code change.

---

## 7. Temporal gap analysis (Step 7)

- `temporal` category (18 queries): **9/18 detected, 9/18 missed — an exact, systematic 50%
  false-negative pattern**, not noise. All 9 detected queries share the phrasing `"...in the last
  12 months by <date column>"`. All 9 missed queries share the phrasing `"What is the monthly
  trend of X based on <date column>"` — a clearly temporal request that `run_temporal_parser()`
  does not recognize as containing a temporal filter (it looks for filter-shaped expressions like
  "last 12 months", not trend/grouping phrasing).
- Which of options A/B/C does the evidence support? **Neither A nor B cleanly — closest to C.**
  `SourceCapabilities` has no `TEMPORAL` capability at all (confirmed by re-reading
  `source_capabilities.py`'s `SourceCapability` enum — unchanged since the original audit), so
  there is nothing to compare `requires_temporal` against regardless of detection accuracy. Given
  that (a) temporal detection itself is unreliable (50% miss rate on real phrasing) and (b) all
  4 configured source kinds (relational/datalake/document/nosql) are plausible temporal-filter
  targets with no evidenced kind-specific restriction, inventing a capability mapping now would
  be exactly the speculative step the task prohibits. **Temporal requirements provide no useful
  C2 planning signal today, on both counts independently — not just because the capability is
  missing, but because the requirement signal itself isn't trustworthy yet.**

---

## 8. C2 simulation (Step 8, analysis only — no filtering executed)

`hypothetical_candidates = candidates − incompatible` could not be meaningfully simulated: with
`candidates == []` for all 187 queries, the hypothetical set is also `[]` for all 187 — a
no-op simulation. **No case exists in this run where hypothetical filtering differs from actual
routing**, because actual routing never had a non-trivial candidate set to begin with. This is
reported plainly rather than represented as "0 divergences observed" in a way that could be
mistaken for a safety signal — 0 divergences here means 0 evidence, not 0 risk.

---

## 9. Capability-model gaps found this pass

1. **No `TEMPORAL` capability** (already known from the original C1 record, re-confirmed) — see §7.
2. **`JOINING` granularity gap** (already known from the C0 audit) — unchanged, not re-investigated.
3. **New this pass:** the shadow log's `_log_shadow_observation()` never serializes each
   candidate's full capability set, only the boolean compatibility verdict — makes the existing
   log strictly less useful for exactly this kind of retrospective benchmark than it needs to be
   (worked around here by calling the pure functions directly, not by changing the production log
   call).
4. **New this pass, environment not code:** the live dev/demo database's embeddings/retrieval
   tables are from a pre-retrieval-v2 ingestion generation and are incompatible with the current
   evidence provider — this blocks not just C2 evidence-gathering but any real-evidence-based
   testing of the routing tier at all in this environment until re-ingested.

---

## Compact final summary

- **Total queries tested:** 187 through real `plan_route()` (182 benchmark + 5 hand-authored); 182 through `derive_requirements()` alone.
- **Aggregate queries:** 32/32 (benchmark's `aggregate` category) detected correctly; 0 false negatives on that category. 1 systematic false-negative pattern found elsewhere (`distribution of X by Y`, 12/20 of `grouped`).
- **Temporal queries:** 9/18 (`temporal` category) detected; 9/18 missed via one systematic phrasing pattern ("monthly trend... based on").
- **Sources tested:** 5 real configured sources (2 relational, 1 document, 2 datalake); no nosql source exists in this environment.
- **Incompatible observations:** 0 — zero real candidates were ever produced to observe.
- **Cases where selected source was incompatible:** 0/0 — no query in this run had a selected source at all.
- **Observed false negatives (C2 would wrongly remove the correct source):** **not measurable** — 0 real candidates were ever generated by the live environment for any of the 187 queries, due to a missing-ingestion-schema gap in the live database (§3), unrelated to C1's own code.
- **C2 simulation impact:** none observed — the hypothetical filtered set equals the actual (empty) set for all 187 queries; this reflects absent evidence, not proven safety.
- **Final verdict: 🔴 NO-GO — for a different reason than "C2 looks unsafe."** The decisive question this benchmark exists to answer (does capability filtering ever remove a source that was actually correct?) could not be tested at all in this environment. The evidence that *does* exist is mixed: aggregation detection is clean on its clearest category but has one real, reproducible false-negative pattern; temporal detection has a 50% miss rate on a common real phrasing and has no capability to compare against regardless. None of this supports greenlighting C2, even narrowly, before (a) this environment (or an equivalent one) is re-ingested under the current retrieval-v2 schema so real candidate-level evidence can be collected, and (b) the `distribution of X by Y` aggregation gap and the temporal-phrasing gap are at minimum measured against real routing outcomes, not just requirement-derivation output in isolation. Re-running this exact harness after re-ingestion is the natural next step — not a redesign of C1.
