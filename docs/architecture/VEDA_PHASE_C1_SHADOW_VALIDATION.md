# Phase C1 — Shadow-Data Validation Audit (pre-C2 evidence)

**Status: READ-ONLY ANALYSIS. No routing behavior changed. No candidate filtering implemented.
No flags/defaults touched. No changes to Phases A/B/C1 code.** This is a second, independent
evidence pass — it supersedes nothing in
`docs/architecture/VEDA_PHASE_C1_SHADOW_BENCHMARK.md` (the prior benchmark, blocked on missing
retrieval tables), it adds a data source that report didn't have: a **real production
verified-query cache** with real (query → winning source → executed SQL) triples, which does not
depend on the broken embedding tables.

**Verdict up front: 🟡 HOLD.** Details in §7. Short version: everything measurable here is
reassuring (0 real false-negative incidents, capability metadata confirmed correct by code), but
the one scenario that actually matters — a document-kind source genuinely being the correct
answer for an aggregation-flagged query — has **zero real examples either way** in any data
available in this environment, so "safe" here means "untested," not "proven."

---

## 1. Sample size and data sources

Two independent real corpora, no fabricated queries:

| Source | N | What it gives | DB dependency |
|---|---|---|---|
| `evaluation/retrieval_benchmark.json` | 182 | Real benchmark queries with category labels | None — `derive_requirements()` is pure text |
| `substrate_verifiedquerycache` (live Postgres, `veda-platform-postgres-1`) | 123 | **Real production data**: query text, the **real source_id that actually answered it**, and the **real SQL that was actually executed** (confirmed by reading `veda/cache.py::save_verified_query()` — "Record a successfully-executed query") | Read-only `SELECT` against the live DB, no writes |

Combined: **305 real queries** analyzed. `plan_route()`'s full runtime path (real embeddings →
real candidate list → real `RoutingDecision`) is **still blocked**, same root cause as the prior
report, re-verified this session: `column_embeddings_v2` / `doc_chunks` still absent from the
live DB (`\dt` re-run, unchanged). The verified-query cache is the workaround that gets real
winning-source data despite that block — see §4.

Real configured sources (re-verified via `docker exec` into `veda-platform-api-1`, Django shell):

| id | name | kind | occurs in verified-cache (123 rows) |
|---|---|---|---|
| 1 | launchpad | relational | 0 |
| 2 | homzhub | relational | 121 |
| 3 | docs_contracts | **document** | 0 |
| 4 | invoices_csv | datalake | 2 |
| 5 | catalog_parquet | datalake | 0 |

**Coverage gap, stated plainly:** the document source (`docs_contracts`) has never once won a
real cached query in this environment's history. This limits §4/§6 to what can be said about
relational/datalake — see there for why this matters.

---

## 2. QueryRequirements accuracy (aggregation)

### 2.1 Benchmark (182, category-labelled)

`requires_aggregation` by category:

| Category | Detected / Total |
|---|---|
| aggregate | 32/32 |
| grouped | 8/20 |
| simple | 22/50 |
| analytical_multitable | 13/31 |
| temporal | 9/18 |
| filter | 0/19 |
| ranking | 0/12 |

No false positives found in `filter`/`ranking` (0/0 as expected). The 22/50 `simple` hits are
correct by inspection, not noise — e.g. `SMPL-01: "How many properties are there?"` is genuinely
a COUNT regardless of the benchmark's own "simple" difficulty label (that label describes
retrieval difficulty, not aggregation-need).

**Confirmed false-negative pattern #1 — "distribution of X by Y" (12/20 of `grouped`):**
queries like `"What is the distribution of properties by furnishing?"` are not detected; the
sibling phrasing `"average carpet area of properties by furnishing"` (8/20) is. A distribution
query is inherently a grouped count — this is a real gap in the `_COUNT_TRIGGERS`/`_COUNT_WORDS`
vocabulary reused from `fast_path.py`.

**Confirmed false-negative pattern #2 — "monthly trend of X based on Y" (9/18 of `temporal`,
exact overlap with the `requires_temporal` misses in §3):** e.g. `TEMP-02: "What is the monthly
trend of properties based on created at?"`. Neither `requires_aggregation` nor `requires_temporal`
fires for this phrasing — it's invisible to C1 on both axes at once. The sibling phrasing `"How
many X were there in the last 12 months by Y"` (9/18) is detected on both axes. A "monthly trend"
question is both an aggregation (grouped count) and a temporal filter — this is the largest
combined blind spot found in this pass.

### 2.2 Real production cache (123, semantic judgment against real executed SQL)

Using "does the executed SQL contain an aggregate function" as an automated ground-truth proxy
gave a misleading 14 TP / 22 "FP" / 5 FN / 82 TN split. **The 22 "FP" figure is mostly an
artifact of a different, real, and separately-noteworthy system behavior, not detector error —
see the callout below.** Manually re-judging each of the 22 against the actual NL intent:

- **21 of 22 are correct detections** the automated SQL-proxy simply couldn't see. Example:
  `"How many properties are there?"` → detector: `count` (correct — this unambiguously asks for
  a count) → but the *executed* SQL was `SELECT ... FROM assets_asset LIMIT 100` — no `COUNT(*)`
  at all.
- **Side finding, out of this audit's scope but too material to bury:** for a "how many" question,
  the real production system's cached answer SQL does not compute an aggregate in the database —
  it fetches raw rows (capped at `LIMIT 100`) and appears to count/summarize downstream. If the
  true row count exceeds 100, this would silently under-report. This is a SQL-generation/execution
  concern (`fast_path.py`/`generation.py`), **not** a C1 requirement-detection or
  capability-model defect, and this audit makes no change related to it — flagged for visibility
  only, per the same "surface it, don't fix it" discipline as everything else here.
- **1 of 22 is a genuine, confirmed false positive:** `"Show 5 top total debit and credit amounts
  (for Single Financial Transactions...)"` → detector: `sum` (from the bare `"total "` trigger in
  `fast_path.py::_SUM_VERBS = ("sum of", "total ")`). The real executed SQL is a plain
  `... LIMIT 5` ranking query — no SUM. Root cause: `"total "` is a substring match that fires on
  **"total" as a noun/column-name modifier** ("total debit and credit amounts" = a metric name
  being ranked) exactly the same as it fires on "total" as a genuine SUM verb ("total revenue").
  This is a real, reproducible false-positive mechanism in the reused `fast_path.py` trigger
  tuple, evidenced here for the first time.

**5 confirmed false negatives (SQL genuinely contains `COUNT`/`SUM`/`MIN`/`MAX`/`GROUP BY`,
detector said `requires_aggregation=False`):**

| Query | Real SQL evidence | Gap |
|---|---|---|
| "Group our properties for sale based on their currency configuration in descending order." | `GROUP BY ... COUNT(*)` | "Group ... based on" phrasing not recognized |
| "Identify the very first expected sale price recorded historically for each of our properties." | `MIN(...) GROUP BY` | superlative "very first ... for each" — MIN/MAX has **no representation at all** in `aggregate_type` (model only knows count/sum/avg) |
| "What is the difference between the highest and lowest transaction amounts recorded?" | `MAX(...)-MIN(...)` | "highest and lowest" — same MIN/MAX gap |
| "What is the distribution of amenities across projects?" | `COUNT(DISTINCT ...)` in an agg CTE | **same "distribution of X" pattern as §2.1** — independent corroboration in real production data |
| "Show top five general ledger entries..." | `COUNT(DISTINCT ...)` in an agg CTE | borderline — phrased as ranking, aggregation is an implementation detail of the join, not part of the user's literal ask; noted as ambiguous rather than a clean miss |

**The "distribution of X by/across Y" gap appears independently in both the 182-query benchmark
and the 123-row real production cache — this is the single most load-bearing, best-corroborated
finding in this audit.** The MIN/MAX representation gap (`aggregate_type` has no slot for it) is
the second-most load-bearing, found only in the real cache (the benchmark has no MIN/MAX-style
queries to have caught it).

**Is `aggregate_type` stable?** One real ambiguity found: `"If unpaid and open records are
combined into a single pending bucket, how many pending items exist and what is their total
value?"` — genuinely asks for BOTH a count and a sum. `derive_requirements()`'s `if/elif` chain
(count checked before sum before avg) can only report one `aggregate_type`, silently dropping the
second. Not wrong, but a real information-loss case worth naming under "ambiguous aggregate
phrases," per the task's explicit ask.

---

## 3. Temporal detection (benchmark only, no capability to compare against — unchanged finding)

9/18 of the `temporal` category detected, 9/18 missed — exactly the same 9 queries that also miss
aggregation (§2.1). `SourceCapabilities` still has no `TEMPORAL` capability (re-confirmed,
unchanged from the original C1 record), so this remains ungated regardless of detection accuracy.
No new conclusion beyond the prior report; repeated here only because the aggregation/temporal
overlap on the "monthly trend" pattern is new evidence this pass.

---

## 4. Capability metadata validation

**For every candidate a shadow comparison would mark incompatible due to missing `AGGREGATION`:**
only `document` and (unconfigured) `nosql` kinds lack it in `source_capabilities.py::_PROFILES`.
`relational` and `datalake` both have it — meaning **100% of this environment's real historical
winners (121 relational + 2 datalake) are structurally always capability-compatible for
aggregation**, regardless of detector accuracy.

**Is "document lacks AGGREGATION" correct metadata, or a metadata gap (classification B)?**
Checked directly, not assumed: read `veda_core/query/doc_data_planner.py` and
`veda_core/query/rag_layer.py` for any document-side aggregation/count execution path. Found
none — `doc_data_planner.py` is a bounded, SLM-mediated entity-*grounding* tool (extract named
entities from document chunks, intersect against a data column), gated behind a completely
separate flag (`DOC_DATA_GROUNDING_ENABLED`), not an aggregation mechanism. **Classification: A
(detector) is not implicated here, B (metadata wrong) is not supported by the code — the
`document` capability profile is correct.** No JOINING-style granularity issue like the one the
original C0 audit found for `JOINING` was found for `AGGREGATION`.

**Compatibility logic (classification C):** `observe_candidate_capabilities()` is a single `if`
checking `requires_aggregation and not caps.has(AGGREGATION)` — reviewed, matches the model
exactly, no logic defect found.

**Net: for the one capability dimension C1 actually enforces (AGGREGATION), detector logic (A)
has the two real gaps in §2, metadata (B) is correct, comparison logic (C) is correct.**

---

## 5. Winning-source safety check (the decisive question)

Classified per the task's four buckets, against all 123 real historical (query, winning source,
SQL) triples:

| Class | Count | Basis |
|---|---|---|
| TRUE_CONFLICT (winner genuinely incompatible AND wrongly so) | **0** | — |
| FALSE_NEGATIVE_RISK (shadow would flag winner incompatible, but winner is actually correct) | **0 observed** | 0/123 real winners were ever shadow-marked incompatible (§4 — winners are always relational/datalake, both AGGREGATION-capable) |
| METADATA_GAP | 0 | document's lack of AGGREGATION is correct (§4), not a gap |
| DETECTOR_ERROR | 1 (harmless in this instance) | the "total debit and credit amounts" false positive (§2.2) — but its real winning source was `homzhub` (relational, AGGREGATION-capable), so the false-positive detection would **not** have caused C2 to remove the correct source in this actual case |

**Why this is 🟡, not 🟢:** C2 can only ever produce a false-negative when (a) the detector wrongly
fires `requires_aggregation=True` for a query, AND (b) the *actually correct* source for that
query happens to be `document`/`nosql`-kind (the only kinds lacking AGGREGATION). Part (a) has
exactly one real, confirmed example (§2.2). Part (b) has **zero real examples in either
direction** — no query in this environment's real history was ever correctly answered by
`docs_contracts`. The combination that would actually hurt (a document-correct query that also
false-positives on aggregation) has never been observed **because the document source has never
won anything real to check against**, not because it's been checked and found safe. This is
exactly the "insufficient evidence" condition the task's HOLD bucket describes.

`decide()` itself remains fully capability-blind by construction (re-confirmed by code read,
matches the original C0 audit finding, unchanged) — nothing today prevents it from selecting an
AGGREGATION-incapable source if one ever legitimately had the strongest evidence. This is a
standing structural possibility, not something this pass measured away.

---

## 6. Expected value of C2 (is filtering worth it, not just clean?)

- Aggregation-flagged rate: 84/182 benchmark (46%), 36/123 real cache (29%) — **120/305 combined
  (39%)** of real queries would trigger a capability check under C2 as currently scoped.
- Of the 5 real configured sources, filtering can **only ever remove exactly one** (`docs_contracts`)
  — `nosql` isn't configured, `AGGREGATION` is the only enforced dimension. Max theoretical
  candidate reduction per aggregation-flagged query: **1 candidate**, and only if `docs_contracts`
  was already a retrieved candidate for that query to begin with.
- No real data exists (blocked evidence layer) on how often `docs_contracts` is actually a
  *competing* candidate for aggregation-style queries about property/lease/financial tables — the
  kind of content it holds (contracts) is topically distant from those queries, so it plausibly
  already scores low/NONE-tier and gets excluded by the existing embedding-tier stage before a
  capability filter would ever matter.
- **Honest conclusion: no real evidence that C2, as currently scoped (AGGREGATION-only, one
  affected source kind), would change real routing outcomes measurably often.** The value case
  rests on architectural correctness (a capability-aware layer existing for future source kinds)
  more than on a demonstrated routing-quality improvement today. This matches the task's own
  warning not to justify C2 on cleanliness alone.

---

## 7. Decision matrix and verdict

| Criterion | Result |
|---|---|
| No false-negative risk found | **0 observed**, but the one scenario where it could occur (document source is correct + detector false-positives) has zero real coverage either way |
| Detector accuracy acceptable | Mixed — high real precision (~97% by semantic judgment, 1 confirmed FP in 305 real queries) but 2 systematic, cross-corroborated false-negative patterns (`distribution of X by Y`; MIN/MAX phrasing entirely unmodeled) |
| Capability metadata trustworthy | **Yes**, confirmed by code for the one dimension in use (§4) |
| Measurable candidate reduction | **Not empirically measurable** here (blocked evidence layer); theoretical ceiling is narrow (§6) |

**🟡 C2 HOLD.** Not NO-GO — nothing found actively demonstrates unsafety, and the real evidence
that does exist (0/123 real winners ever incompatible, correct capability metadata) is
reassuring. Not GO — the one scenario that would actually matter (document source is the correct
answer for a query the detector also flags as needing aggregation) has never occurred in any real
data available in this environment, so its risk is unmeasured, not low.

---

## 8. Recommended next action (exactly one, per the task's rule)

**Collect more shadow data** — specifically, real candidate-level evidence covering the one
untested scenario: a query where `docs_contracts` (or a future non-aggregating source) is
genuinely the correct answer, cross-checked against what the shadow comparison would have said.
Two concrete paths, either sufficient:
1. Re-ingest this dev/demo environment onto the retrieval-v2 schema (`column_embeddings_v2` /
   `doc_chunks`) so `plan_route()`'s real evidence path works, then re-run this same query set end
   to end (the blocker identified in the prior benchmark report, unchanged).
2. If a different environment or longer production history exists with real `docs_contracts`
   traffic in its verified-query cache, mine that instead — no re-ingestion needed, same method
   as §5 here.

The two detector gaps found in §2 (`distribution of X by Y`; MIN/MAX phrasing) are real and worth
fixing on their own merits, but per the task's "recommend exactly ONE action" rule they are named
here as supplementary findings, not the primary recommendation — fixing them improves C2's future
*value* (§6) more than its *safety* (§7), since both are false-negative-direction gaps (they make
C2 filter less, not incorrectly more).

---

## Compact summary

- **Sample size:** 305 real queries (182 benchmark + 123 real production verified-query-cache
  records), plus code-level verification of the capability model.
- **Aggregation findings:** 120/305 (39%) flagged as requiring aggregation. 1 confirmed real false
  positive (harmless in its one real instance). 2 confirmed, cross-corroborated false-negative
  patterns ("distribution of X by Y"; MIN/MAX phrasing unmodeled).
- **Winning-source safety check:** 0/123 real historical winners were ever shadow-marked
  incompatible. Zero coverage (not zero risk) for the one scenario that would matter — a document
  source winning a query the detector flags as needing aggregation.
- **Capability metadata:** confirmed correct by code for the one enforced dimension (AGGREGATION).
- **Expected value of C2:** narrow by construction (1 of 5 sources affected, ~39% of queries
  touch the check at all) and unmeasured in practice — no evidence it would change real outcomes
  often.
- **Verdict: 🟡 HOLD.**
- **Recommended next action: collect more shadow data**, specifically real candidate-level
  evidence for document-source-wins-an-aggregation-flagged-query, via environment re-ingestion or
  a longer/different real production history.
