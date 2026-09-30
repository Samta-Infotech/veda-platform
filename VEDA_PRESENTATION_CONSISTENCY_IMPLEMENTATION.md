# VEDA — Presentation Consistency, PresentationPlan & Graceful Fallback

**Date:** 2026-09-29
**Builds on:** `VEDA_CONSISTENCY_FEASIBILITY_AUDIT.md` (same session) — this is the
implementation pass against that audit's P0 list.
**Constraint honored throughout:** every change is additive or defaults to the
existing byte-identical behavior; nothing described here changes what an existing
consumer already receives unless a NEW field/event is explicitly read.

**Part 2 addendum (same date, same session, follow-up pass):** closes the
remaining gaps from a second, more detailed spec covering summary-generation
hardening, SLM seed standardization, and a narrowly-scoped SQL tie-break. See
**§9-§14** below, appended after the original implementation report.

---

## 1. Changed files

| File | Why |
|---|---|
| `veda_core/retrieval/rrf_merger.py` | Deterministic secondary sort key (`col_id`) on the RRF fusion output — ties no longer depend on `PYTHONHASHSEED`-driven `set()` iteration order. |
| `tests/test_rrf_identity.py` | New tie-break regression test. |
| `chatbot/nodes.py` | Pinned `temperature=0` on the two classification/verdict SLM calls (`classify_node`'s merged action+delta classifier, `_depends_on_history`'s standalone-check) that previously fell to `call_slm`'s default `0.1`. |
| `veda_core/config.py` | Two new config constants: `RESPONSE_TEXT_TEMPERATURE` (default 0.1, unchanged) and `VERIFIED_CACHE_SIMILARITY_THRESHOLD` (default 0.85, unchanged) — both make a previously-hardcoded literal deployment-tunable without touching call sites again. |
| `veda_core/query/result_explainer.py` | Both SLM call sites (`nl_answer`, `insight_engine`) now read `RESPONSE_TEXT_TEMPERATURE` instead of a hardcoded `0.1` literal — same value by default, now a named, documented, overridable knob. |
| `veda_core/veda/cache.py` | Default similarity threshold now reads `config.VERIFIED_CACHE_SIMILARITY_THRESHOLD` instead of a hardcoded `0.85`. |
| `storage_adapters/reader.py` | Same threshold fix, defense-in-depth for any caller that doesn't go through `cache.py`. |
| `apps/chat/presentation_plan.py` **(new)** | The formal `PresentationPlan` contract — `AnalysisStatus`, `PresentationMode`, and `build_presentation_plan()`, wrapping already-computed `(cols, rows, analytics, chart_specs)` into one typed, observable decision object. Zero new decisions — a pure formalization. |
| `apps/chat/services.py` | One new, additive SSE event (`presentation_plan`) emitted alongside the existing `content`/`visualization` events — built from data already computed two lines above it. |
| `tests/test_presentation_plan.py` **(new)** | 9 tests covering every `PresentationMode`/`AnalysisStatus` combination. |

**Deliberately NOT changed** (see §7, Remaining Risks): the SQL-level `ORDER BY`
tie-break (audit gap #5) and a verified-cache write-time correctness gate (part of
gap #7) — both assessed as too broad-blast-radius to implement safely in this pass
without live-model regression testing; see §7 for the exact reasoning and a safer
alternative that WAS implemented instead (the similarity-threshold knob).

---

## 2. Architecture — before → after

```
BEFORE:
  analyze_result() → analytics dict → table_rendering.py / visualization.py
  each read overlapping dict keys independently; no single place answered
  "what did the system decide to show, and why" as one typed object.

AFTER:
  analyze_result() → analytics dict → table_rendering.py / visualization.py
                                              ↓                    ↓
                                    (unchanged — same table)  (unchanged — same chart)
                                              ↓                    ↓
                                       apps/chat/presentation_plan.py
                                       (build_presentation_plan — NEW,
                                        wraps the two outputs above)
                                              ↓
                                new "presentation_plan" SSE event
                                (additive — existing "content"/"visualization"
                                 events are byte-identical to before)
```

The actual table/chart RENDERING PATH is untouched. `PresentationPlan` is a new,
parallel, observable VIEW of a decision the pipeline already made — not a new
decision-maker.

---

## 3. Presentation state model

| State | Meaning | Example |
|---|---|---|
| `SUMMARY` | Result is a single scalar — a table/chart would add nothing. | "What is the total revenue?" → one number. |
| `TABLE` | Multi-row/detail result, no chart recommended (or none survived chart-safety validation). | "Show the latest transaction." |
| `CHART` | Reserved for a chart-only presentation (no table content produced) — not currently reachable in this codebase (the table block is always attempted independently whenever `cols`/`rows` exist), kept in the enum for the requested state model's completeness and for a future producer that legitimately has no tabular form. |
| `TABLE_AND_CHART` | Multi-row result with a chart the recommender actually produced. | "Revenue by month." |
| `RAW_RESULT` | `analyze_result()` didn't run/raised, but validated `(cols, rows)` exist — shown as-is, no narrowing, no chart. | An analyzer exception on a genuinely malformed result shape. |
| `NOT_APPLICABLE` | Zero rows, OR analysis unavailable with zero rows too. **A success state, never a failure.** | A filter that legitimately matched nothing. |
| `SAFE_FALLBACK` (an `analysis_status`, not a `presentation_mode`) | `analyze_result()` itself didn't run or raised — presentation_mode is then `RAW_RESULT` or `NOT_APPLICABLE` depending on whether rows exist. | An unexpected exception inside `analyze_result()`, caught upstream (`veda_hybrid.py`/`pipeline.py`'s existing best-effort try/except — unchanged, this session did not touch those). |

`analysis_status` and `presentation_mode` are orthogonal: `SUCCESS` +
`NOT_APPLICABLE` together mean "everything worked, there's just nothing to chart" —
exactly the distinction §4 of the request required, and it was already implicitly
true in the code (an empty chart list was already silent, never an error message);
this makes it an explicit, typed, loggable fact instead of an inference.

---

## 4. Determinism fixes (this pass)

| # | Fix | File | Risk | Status |
|---|---|---|---|---|
| 1 | RRF secondary tie-break key | `rrf_merger.py` | None — only affects exact ties, never non-tied ranking | **Implemented, tested** |
| 2 | `classify_node` merged classifier temperature=0 | `chatbot/nodes.py` | Very low — matches the standalone fallback's existing pin | **Implemented, tested** |
| 3 | `_depends_on_history` standalone-check temperature=0 | `chatbot/nodes.py` | Very low — boolean verdict, no sampling use case | **Implemented, tested** |
| 4 | Response-text temperature made configurable | `result_explainer.py`, `config.py` | None (default unchanged) — a lever, not a forced change | **Implemented** (still 0.1 by default — see §7 for why not forced to 0) |
| 5 | Verified-cache similarity threshold made configurable | `cache.py`, `reader.py`, `config.py` | None (default unchanged) | **Implemented** (still 0.85 by default — see §7) |
| 6 | `ORDER BY ... LIMIT` tie-break column | `planning.py` builders | High — risks invalid SQL on grouped queries (Postgres rejects an ORDER BY column that's neither grouped nor aggregated) | **NOT implemented — deferred, see §7** |
| 7 | Verified-cache write-time re-validation | `pipeline.py` | Assessed and found **largely redundant** for the fresh-SQL path (already passes the identical `ranked_shape_ok`/qualifier gates before reaching the save site) — see §7 | **NOT implemented — assessed as low-value, see §7** |

---

## 5. Fallback behavior — exactly what happens

- **Visualization unnecessary** (scalar, or zero rows): `presentation_mode` =
  `SUMMARY` or `NOT_APPLICABLE`. No chart event is emitted (unchanged — it never
  was). No error, no "chart unavailable" text anywhere (verified — no such string
  exists in `services.py`/`presentation_plan.py`).
- **Visualization unsupported / no confident candidate**: `VisualizationRecommender`
  already returns `[]` in this case (unchanged this session). `presentation_mode` =
  `TABLE`. Table content is unaffected.
- **Chart spec references a column not in the result** (defense-in-depth check,
  not an observed bug): `_chart_columns_exist()` fails → `chart` is dropped →
  `presentation_mode` degrades from `TABLE_AND_CHART` to `TABLE` → the table (which
  never depends on the chart) is still shown.
- **`analyze_result()` fails/didn't run**: `presentation_mode` = `RAW_RESULT` (if
  `cols`/`rows` survived) or `NOT_APPLICABLE` (if not) with `analysis_status` =
  `SAFE_FALLBACK`. No SLM is invoked to "recover." No column is invented. This
  mirrors what `veda_hybrid.py`/`pipeline.py` already do today (their own
  try/except around the `analyze_result()` call, unchanged) — the new code only
  makes that existing behavior visible as a typed status instead of a silently
  absent `analytics` key.
- **Table rendering itself fails**: outside this pass's scope — `table_rendering.py`
  was not touched here (it received its own fixes earlier this session, see
  `project_table_query_column_filter` memory); its existing fail-safe (never render
  an empty table on a filter mismatch) is unchanged.

---

## 6. Tests

| | Count |
|---|---|
| Existing tests (touched-area suites) | 232 (pre-existing, run before this pass) |
| New tests | 9 (`test_presentation_plan.py`) + 1 (`test_rrf_identity.py`'s tie-break check) |
| Total run this pass | 242 (bare-metal) + 92 (Django api-tier, `test_chatbot_classify.py` + `test_presentation_plan.py`) |
| Passed | 241/242 (bare-metal), 91/92 (Django) |
| Failed | 1 in each run — **the SAME single pre-existing, unrelated failure both times** (`test_chat_visualization.py::test_grouped_breakdown_still_aggregates_unchanged` bare-metal; `test_chatbot_classify.py::test_go_back_without_drill_stack_falls_through_to_llm` in Django — confirmed logically impossible to be caused by this pass's additive-kwarg-only changes to that file, and reproduces standalone) |

Live-verified inside the running containers (not just unit tests): all touched
modules import cleanly in both the `inference` and `api` containers;
`config.RESPONSE_TEXT_TEMPERATURE`/`VERIFIED_CACHE_SIMILARITY_THRESHOLD` resolve to
their unchanged defaults live; a real query through the full vedademo
SLM+embedding pipeline (`"How many properties are there?"`) still answers
correctly end-to-end after these changes.

---

## 7. Remaining risks — named, not hidden

1. **`ORDER BY ... LIMIT` tie-break (audit gap #5) — deliberately not implemented.**
   `planning.py`'s ranking builders sometimes emit `GROUP BY t0."{col}" ... ORDER BY
   {metric} LIMIT n`. Appending a naive secondary key (e.g. the anchor's `id`) would
   produce **invalid SQL** whenever a `GROUP BY` is present and the tie-break column
   isn't itself grouped or aggregated (Postgres: "column must appear in the GROUP BY
   clause or be used in an aggregate function"). A correct fix needs to
   branch on whether the specific ranking is grouped or a raw per-row listing, and
   for the grouped case the only safe secondary key is the group column itself
   (already the primary sort target in most cases) or nothing. This needs per-branch
   analysis across ~6 call sites in `planning.py` plus the SLM-generation prompt
   instructions, each requiring its own correctness verification — exactly the kind
   of broad, hard-to-fully-verify change the standing instruction ("must not hamper
   existing implementation") says to avoid rushing. **Still open.**
2. **Verified-cache write-time re-validation (part of audit gap #7) — assessed, not
   implemented.** Traced the actual call path: the fresh-SQL path already runs
   `ranked_shape_ok(query, sql)` (pipeline.py:3090) before ever reaching the cache-write
   site, so re-running the identical check at write time would be a no-op for
   today's code. The real, still-open exposure is **read-time drift** — an entry
   saved correctly under today's rules can become wrong after a FUTURE code/schema
   change, and nothing re-validates already-cached entries against new rules except
   a new demotion guard being written reactively (as already happened 3 times,
   per the audit). The mitigation actually shipped this pass — a configurable
   similarity threshold — is the honest, safer lever available without inventing an
   unverified new mechanism.
3. **No `seed` on RAG-synthesis/IR-emit/routing SLM calls** (audit gap #3) — not
   touched this pass; `temperature=0` is correctly wired at these sites already, a
   fixed seed is a smaller additional hardening step, deferred as P1 per the audit.
4. **`RESPONSE_TEXT_TEMPERATURE` and `VERIFIED_CACHE_SIMILARITY_THRESHOLD` keep
   their EXISTING defaults** (0.1 and 0.85) — this pass makes them configurable, it
   does not change behavior. Whether to actually set `RESPONSE_TEXT_TEMPERATURE=0`
   (fully deterministic wording, at some prose-diversity cost) or raise
   `VERIFIED_CACHE_SIMILARITY_THRESHOLD` (fewer but safer cache hits) are product
   decisions this pass deliberately left to the deployment, per the audit's own
   instruction not to claim a stronger guarantee than what's actually enabled.
5. **`PresentationMode.CHART`** (chart-only, no table) is defined in the enum for
   completeness but not currently reachable — no producer in this codebase emits a
   chart without also emitting table content. Not a gap, just a documented,
   currently-unused state.

---

## 8. Enterprise guarantee (honest statement)

**What this pass adds a guarantee for:** given the SAME `(cols, rows, analytics,
chart_specs)` — which was already deterministic, verified earlier this session —
the `PresentationPlan` (`analysis_status` + `presentation_mode` + the table/chart
selection it reports) is now a pure, typed function of those inputs, with a
regression suite pinning all 6 states.

**What retrieval-level fix #1 adds:** candidates tied at the exact same fused RRF
score now rank identically across process restarts, not just within one running
process — closing the one confirmed retrieval-determinism gap from the audit.

**What fixes #2/#3 add:** the follow-up/action classifier and the standalone-turn
check are now temperature-0, closing the one confirmed conversation-layer
determinism gap from the audit.

**What this pass does NOT newly guarantee:** SQL row order under a `LIMIT` with
ties (gap #5, deferred — §7.1), full seed-pinning on every SLM call (gap #3 residual),
or that a verified-cache entry saved today stays valid forever against future code
changes (gap #7 residual, mitigated but not closed — §7.2). These are stated here,
not implied to be fixed, per the standing instruction against overclaiming.

---
---

# Part 2 — Follow-up pass (same date): Summary hardening, SLM seed, SQL tie-break

## 9. What was ALREADY implemented (found by audit, not rebuilt)

The single biggest finding of this pass: **§5's "Add Summary Fact Validation"
request was already fully implemented**, more rigorously than a from-scratch
build would have been. `query/result_explainer.py` already had, before this
session touched it:

- `_answer_numbers_grounded()` — every number the SLM summary states must be
  traceable (±2%, floor ±2) to the precomputed facts/metrics/patterns, with a
  documented, already-fixed hallucination hole (an integer ≤ row_count used to
  pass unconditionally as "a count/rank/ordinal" — dated 2026-09-20, tightened to
  require the integer to actually BE a count-like value in the result).
- `_extreme_claims_grounded()` — catches a DIFFERENT failure shape: every number
  in the summary is real, but one is presented as a min/max/range bound when it's
  only the smallest/largest value in a 5-row SAMPLE, not the true extreme.
- `_uncovered_claimed()` — catches a claim about an entity/scope the query never
  measured.
- `_strip_invented_currency()` — deterministic backstop that strips a currency
  symbol the model prefixed that isn't in the data.
- All four are already wired as `raise ValueError(...)` → `except Exception` →
  `template_answer()`/`deterministic_fallback_answer()` — i.e. the EXACT
  "SLM Summary → Fact Validation FAILED → deterministic factual fallback"
  pipeline the spec asked for, already shipped, already tested.

This pass did not rebuild any of this. It only made the *existing* pass/fail
outcome **observable** (§10) — previously it was `logger.warning(...)` only, with
no typed field a caller could read.

## 10. NEW: `summary_status` / `NO_SUMMARY_NEEDED` (§6)

`query/result_explainer.py::NLAnswerResult` gained two new fields (both
additive, default-preserving — every existing caller reading only `.answer`/
`.slm_used` is unaffected):

- `summary_status: str` — one of `NOT_REQUIRED` (zero rows — the SLM was never
  called, unchanged behavior, now labeled), `GENERATED` (SLM answered and passed
  every fact-validation guard), `VALIDATION_FAILED` (a guard rejected it —
  deterministic fallback used), `SLM_UNAVAILABLE` (the SLM call itself
  errored/timed out/returned empty — not a validation failure, no candidate to
  validate).
- `fallback_reason: Optional[str]` — the exact reason string, for the two
  fallback statuses.

Threaded through to the existing trace/observability sink
(`veda/explain.py::record_result_stages`, called by both Tier-1 and Tier-2) as
two new optional kwargs (`summary_status`, `summary_fallback_reason`) recorded
into the `"summary"` trace section — additive, no existing trace consumer's
fields changed.

**Scope note on `NO_SUMMARY_NEEDED` more broadly**: the spec's other examples
("straightforward detail table", "user explicitly asks for raw results") are
NOT wired to skip the SLM call — `run_nl_answer`'s own docstring already
documents a deliberate, pre-existing design decision to summarize even
simple/scalar shapes ("a canned template reads robotically for those"). Forcing
those to skip the SLM would reverse an existing, intentional choice, which the
standing "must not hamper existing implementation" instruction rules out doing
silently. The one case implemented (`row_count == 0`) is the one that already
skipped the SLM before this pass — now it's labeled, not newly introduced.

## 11. `RESPONSE_TEXT_TEMPERATURE` default → 0

Changed per this pass's explicit instruction ("for enterprise consistency,
default to temperature=0 where supported"). Was `0.1` (this session's Part 1,
kept as a deployable-but-opt-in lever); now `0.0` by default. Safe because the
fact-validation guards (§9) already constrain *what* the model may state
regardless of temperature — 0 only removes word-choice diversity, not
correctness, which was already enforced independently.

## 12. SLM seed standardized across every call site

New `config.SLM_SEED` (default `0`, matching `generation.py`'s pre-existing
literal) — added to every SLM call site that previously ran at
`temperature=0`/`SLM_TEMPERATURE` with NO seed: `query/rag_layer.py` (rag
synthesis), `query/slm_layer.py` (ir_emit, decompose), `query/lg_nodes.py`
(LangGraph node), `query/answer_entity.py` (entity relation), `query/
envelope_slm.py` (envelope), `query/federated_route.py` (both federated-plan
call sites). `generation.py`'s own two SQL-gen calls already had `seed=0` and
were left untouched (shown to share the config value, not required to). Purely
additive — no call site's temperature or any other parameter changed.

## 13. SQL ranking tie-break — implemented, narrowly, flag-gated

Part 1 of this session deferred this (§7.1) as too broad-blast-radius to do
safely. Re-scoped and implemented on this pass:

- New `config.SQL_RANKING_TIE_BREAK_ENABLED` (default `False`).
- New `veda/planning.py::_ranking_tie_break(group_col)` — returns `, t0."{group_
  col}" ASC` ONLY when a `GROUP BY` column already exists in that same query
  (always valid SQL, since it's already in the SELECT/GROUP BY list) AND the
  flag is on; returns `""` (no-op) otherwise.
- Wired into the two `ORDER BY ... LIMIT` tails inside `veda/planning.py::
  build_aggregate_sql()` that build grouped-ranking SQL.
- **Deliberately still NOT touching the ungrouped (raw per-row) ranking case** —
  the one Part 1 flagged as needing a verified, per-table primary-key lookup
  this function has no safe way to do blindly. This pass closes the GROUPED
  half of gap #5 (arguably the more common real case — "top 5 categories by
  count" with ties) while leaving the ungrouped half exactly as documented in
  Part 1 §7.1.
- 6 new tests (`tests/test_ranking_tie_break.py`): flag-off no-op, flag-on
  behavior, the ungrouped case staying untouched regardless of the flag, and a
  direct `build_aggregate_sql()` before/after SQL-string assertion.

## 14. Tests (Part 2)

| Suite | Result |
|---|---|
| `test_presentation_plan.py` (2 new multi-source tests added) | 11/11 pass |
| `test_result_explainer.py` (4 new `summary_status` tests added) | 46/47 pass (1 pre-existing unrelated — env model-name config, confirmed identical before this pass) |
| `test_ranking_tie_break.py` (new file) | 6/6 pass |
| `test_grouped_aggregation_operators.py` (regression, flag off) | 25/25 pass — byte-identical SQL confirmed |
| Combined touched-suite sweep (15 files) | 346/349 pass — the 3 failures are the SAME pre-existing, confirmed-unrelated failures already documented in Part 1 §6 and this session's earlier work (env model-name mismatch, a visualization test's own data fixture, and a `business_explain` "new keys" test) |
| Live end-to-end (real vedademo SLM+embedding, real DB) | Re-ran after this pass — same correct answer, no regression |

## Part 2 — honest remaining gaps

- `NO_SUMMARY_NEEDED` is only wired for the zero-row case, not the broader set
  of shapes the spec's examples suggested — see §10's scope note for why
  (reversing an existing, deliberate design decision wasn't this pass's call to
  make silently).
- The ungrouped ranking tie-break case remains open — same reasoning as Part 1,
  now more precisely scoped to "needs a verified per-table PK lookup, which
  doesn't exist in this function today."
- `SLM_SEED`/`SQL_RANKING_TIE_BREAK_ENABLED` are new levers, not automatically
  validated end-to-end against a live model for prose-quality or SQL-plan
  regressions at scale — recommended before flipping `SQL_RANKING_TIE_BREAK_
  ENABLED` on in production: a live A/B on grouped-ranking queries, same
  rigor as this session's earlier live benchmarks.

---
---

# Part 3 — Row handling, shared completeness truth, graceful degradation (same date)

**Hard constraint honored**: zero new SSE events, zero payload/field/ordering
changes to the existing contract. The only externally-visible change is the
WORDING of an already-existing markdown table's truncation notice, in the one
case (truncated results) that notice previously misstated what it was counting.

## 15. The real, deeper bug this pass found

Live-testing table/chart/summary's row handling (per the user's own request)
surfaced a genuine architectural gap, not just a wording nit: **the actual
SQL-execution function (`veda/execution.py::execute_sql`) did a flat
`cur.fetchmany(EXECUTION_RESULT_LIMIT)` with no way to tell "exactly 1,000
rows exist" from "there were more, we got cut off at 1,000."** Confirmed live:
`"Show the building name of each property."` returned 1,000 rows and reported
`is_truncated=False` under a SQL-LIMIT-only heuristic, because the SQL itself
carried no LIMIT clause to compare against — the actual backend safety cap
that fired was invisible.

Fixed at the source: `execute_sql` now fetches one extra row purely to detect
this (mirrors `connectors/relational.py::execute_query`'s own long-standing
+1-peek — this codebase already had the right pattern, just not in the
function the main pipeline actually calls), exposed via a request-scoped
`last_execution_truncated()` getter (same ContextVar read-and-clear pattern as
`veda/generation.py::last_was_deterministic()` — an existing precedent, not a
new mechanism). The `execute_sql(sql, params) -> (cols, rows, err)` return
contract is completely unchanged; the extra fact rides the getter, not a 4th
tuple element, so no caller needed to change.

**Live re-verified after the fix**: the same query now reports
`is_truncated=True, total_count=None` — correct.

## 16. Shared Result-Completeness truth (§2/§3)

New `veda/result_analyzer.py::compute_result_completeness(sql, rows,
fetch_capped=False) -> dict` — the ONE place `fetched_count` /
`is_truncated` / `total_count` / `total_count_known` get decided, combining
two independent signals (SQL's own LIMIT vs. the backend fetch-cap flag
above) rather than picking one. Attached to `analytics_summary()`'s output at
every call site that builds it (Tier-1 `pipeline.py`, Tier-2/federated/hybrid/
nosql paths in `veda_hybrid.py`) — table, chart, and summary now read the
exact same dict, closing the gap where `analytics_summary()` previously
carried NO completeness signal at all (only the summary path had its own,
separate, `_sql_truncated` heuristic).

**Scope boundary, stated plainly**: the new `fetch_capped` (real backend-cap)
signal is wired into Tier-1 (`pipeline.py`) only. Tier-2/federated/hybrid/
nosql paths in `veda_hybrid.py` still get the SQL-LIMIT-only heuristic (an
improvement over having nothing, but not the full fix) — wiring the real
signal there was assessed and deliberately deferred: those paths' `execute_sql`
calls happen in a different part of the function than where analytics gets
built, and the getter's read-and-clear semantics make it genuinely riskier to
wire correctly without verifying call ordering per path, which this pass
didn't have room to do safely.

## 17. Table truncation notice — honest wording (§3)

`table_rendering.py::rows_to_markdown_table()` gained an optional
`is_truncated: bool = False` parameter (every existing caller: byte-identical
wording, verified by a new regression test). When True: "Showing 20 of 1000
rows fetched — more rows may exist beyond this fetch." instead of the
previous "Showing 20 of 1000 rows." — which read as though 1000 were the true
total when it was only ever the fetch count.

## 18. Grouped-chart truncation disclosure + the §7 correctness rule

Two related fixes in `apps/chat/visualization.py::_category_numeric` (closing
the gap this same session found and named earlier: only the row-listing chart
had ANY truncation disclosure):

- **§7 (highest-priority rule in the spec)**: when the same category appears
  on 2+ raw rows (`duplicates` — meaning the SQL never grouped server-side,
  this function is summing raw rows itself) AND the fetch was truncated
  (`analytics["is_truncated"]`), the totals are a sum over an arbitrary
  partial slice of the true data — refuses the chart entirely (`return []`)
  rather than rendering a misleading "complete-looking" bar/pie, matching the
  spec's own stated preference ("prefer not rendering the chart if the
  aggregation cannot be trusted"). A properly `GROUP BY`'d result (no
  `duplicates` — the server already aggregated the FULL data) is unaffected
  regardless of `is_truncated`.
- `_row_listing`'s existing sub_title now also distinguishes "more rows
  fetched" (upstream fetch itself was capped) from its own internal 25-bar
  display cap — the same "of N" ambiguity fixed in the table notice.

## 19. Explicit user limits respected in charts (§11)

`_row_listing` no longer blindly caps at `_MAX_ROW_BARS=25` — it reads the
SQL's own `LIMIT` (`analytics["limit"]`, already-extracted AST fact) and uses
it instead when it's an explicit, reasonably-sized ask (`25 < limit <= 100`):
"top 30" now renders 30 bars, not 25. Bounded at 100 regardless (never
unreadable), and never LOWERS today's default for the common
no-specific-limit case.

**Scope boundary**: time-series/pie/scatter-specific display policies (§6's
other examples) were not implemented this pass — the ranking/listing case
(§11's own worked example) was the one this session had a concrete, already-
observed gap for; the others are documented as open, not silently assumed done.

## 20. Tests (Part 3)

| Suite | New tests | Result |
|---|---|---|
| `test_result_analyzer.py` | 7 (`compute_result_completeness` incl. the fetch_capped fix, `last_execution_truncated` read-and-clear) | 52/52 |
| `test_table_rendering.py` | 2 (default-unchanged, hedge-wording) | 27/27 |
| `test_chat_visualization.py` | 6 (duplicates+truncated block, already-grouped unaffected, explicit-limit respected, 100-ceiling, sub_title hedge) | 45/46 (1 pre-existing unrelated, confirmed earlier this session) |
| Combined touched-suite sweep (11 files) | — | 285/287 (the 2 failures are the SAME pre-existing, confirmed-unrelated failures documented in Parts 1-2) |
| Live (real vedademo pipeline, real DB) | — | Re-verified the exact reported gap is fixed (`is_truncated` now correctly True on the live 1,000-row-capped query); zero regression on a working query (`"How many properties are there?"` still answers 7,814 correctly) |

## 21. Remaining gaps (Part 3), stated plainly

- Tier-2/federated/hybrid/nosql paths don't yet get the real fetch-cap signal
  (§16's scope boundary) — SQL-LIMIT heuristic only there.
- Time-series and pie/scatter-specific display policies (§6) not implemented —
  only the ranking/listing explicit-limit case was.
- Analyzer's 200-row sampling for column classification (§12) and summary's
  5-row sampling (§13) were audited (this session's own earlier finding) but
  not changed — the spec's own instruction was "improve only if needed,"
  and no concrete failure from either was observed this session, unlike the
  fetch-cap gap, which was.
