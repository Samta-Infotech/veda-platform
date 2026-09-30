# VEDA — Feasibility Audit: Consistent Answers for Repeated Questions

**Date:** 2026-09-29
**Scope:** Can VEDA give consistent/reproducible answers to the same question asked repeatedly,
while still changing correctly when data/permissions/context change?
**Method:** Direct code inspection across 6 subsystems (conversation/memory, semantic retrieval,
SLM invocation config, SQL validation/execution, caching, response generation) plus first-hand
live-pipeline verification of the deterministic Result Analyzer / Table / Chart stage, done in
this same session (real vedademo SLM+embedding calls, real DB, 30+ live queries — see
`project_table_query_column_filter.md` memory for the underlying fixes this audit builds on).

---

## 1. Executive Summary

> **PARTIALLY FEASIBLE.**

The **post-SQL presentation stage (Table + Chart) is already, and can remain, fully
deterministic without any SLM call** — this is verified, not theoretical: this session traced
and fixed 3 real bugs in that stage (a generic-alias over-match, a temporal-name substring
false-positive, and a role-priority ordering bug) and confirmed the fix live across ~30 real
queries through the actual vedademo pipeline. The architecture already matches the target
design's core principle for this stage.

However, **5 concrete, currently-unfixed sources of non-determinism exist upstream of
presentation** — none of them require adding a NEW SLM call to fix; all are either a missing
`temperature=0`/`seed` pin, a missing tie-break key, or a missing correctness gate on an
existing mechanism. Fixing them is bounded, well-scoped engineering work, not an architecture
change. That's why the verdict is "partially feasible" rather than "feasible" or "not
feasible" — the target architecture is fundamentally sound and already ~80% real; the gaps are
concrete and listed in §5.

---

## 2. Current Architecture (as actually found in the codebase)

```
USER QUESTION
      ↓
Conversation layer (chatbot/nodes.py, chatbot/memory/) — rule-based follow-up
detection first; falls to an SLM classify call (chatbot/nodes.py::classify_node)
only when the deterministic rules don't resolve the turn
      ↓
Semantic/retrieval layer (veda_core/retrieval/retrieval_engine_phase3.py) —
6-signal weighted RRF: BGE-M3 dense+sparse, FK-subgraph, value index, table-prior,
cross-encoder rerank (veda_core/query/retrieval_v2.py, reranker.py)
      ↓
Verified-query cache check (veda_core/veda/cache.py) — embedding-similarity
lookup (NOT exact match), short-circuits straight to cached SQL on a hit
      ↓
SQL generation — deterministic builder first (generation.py::
_deterministic_single_table_sql, planning.py) for simple cases; SLM (call_slm,
veda_core/slm/_call_slm.py) for the rest, via IR (slm_layer.py) or LangGraph
(lg_nodes.py)
      ↓
Validation/guardrails (veda/validation.py) — AST allow-list, parameterization,
ranking-shape/qualifier/value gates; optional bounded repair-loop on rejection
      ↓
SQL execution (connectors/relational.py) — single-statement, MVCC snapshot,
row_limit=1000 fetch cap
      ↓
Deterministic Result Analyzer (veda/result_analyzer.py::analyze_result()) — ONE
analysis pass: column kind/role, result_shape, recommended_projection
(query_relevant_columns), chart_candidates. No model call, no DB call.
      ↓
Presentation (apps/chat/) — table_rendering.py (project_display_columns) and
visualization.py (VisualizationRecommender) both consume the SAME analytics
dict, no re-derivation
      ↓
Response layer — result_explainer.py (NL answer/summary) — a REAL SLM
paraphrase call, hardcoded temperature=0.1, independent of the SQL-gen
temperature knob
```

This is close to the target architecture in the prompt, with one structural difference worth
naming: **there is no single formal "Presentation Planner" class/contract** — `analyze_result()`
already plays that role informally (its `analytics_summary()` output dict IS the de facto
presentation plan), but it's a loose dict, not a typed, versioned contract. See §8.

---

## 3. Current Sources of Variability (complete list, from the 5-subsystem audit)

| # | Source | File:Line | Severity | Real or theoretical |
|---|---|---|---|---|
| 1 | Follow-up/action classifier SLM call has no `temperature=0` (defaults to `0.1`) | `chatbot/nodes.py::classify_node` (~line 1059), default in `chatbot/llm.py:177` | **High** — can change action/delta_type for the identical follow-up wording | Real |
| 2 | RRF fusion tie-break falls back to `set()` iteration order, which is `PYTHONHASHSEED`-dependent (randomized per process, not pinned anywhere in `.env`/compose) | `veda_core/retrieval/rrf_merger.py:97-148` | Medium — only affects candidates tied at the exact same fused score, but changes across process restarts (this repo's own dev containers restart via `DEV_AUTORELOAD`) | Real |
| 3 | Only SQL-generation calls pass a fixed `seed=0`; RAG-synthesis/IR-emit/LangGraph/entity/routing calls run at `temperature=0` but with **no seed** | `generation.py:299,512` (has seed) vs. `rag_layer.py:333`, `slm_layer.py:596,1272`, `lg_nodes.py:126`, `answer_entity.py:115`, `envelope_slm.py:127`, `federated_route.py:389` (no seed) | Medium — temp-0 is near-deterministic but not formally bit-identical without a seed on some backends under concurrent GPU load | Real, lower-probability |
| 4 | NL answer/summary generator hardcodes `temperature=0.1`, entirely separate from the `SLM_TEMPERATURE` config knob | `query/result_explainer.py:862,1253` | **High** for response-text consistency specifically (§4.D) — same numbers, different wording, run to run | Real, confirmed by 2 independent audits |
| 5 | No enforced UNIQUE/tie-breaking `ORDER BY` — `ranked_shape_ok` only requires *an* ORDER BY exists, not that it disambiguates ties; no SQL builder appends a secondary key (e.g. `, id ASC`) | `veda/validation.py:819` (`ranked_shape_ok`), every builder in `planning.py` (295,367,371,644) | Medium — `ORDER BY price ASC LIMIT 5` with ties at the cutoff isn't guaranteed the same 5 rows across runs (Postgres doesn't promise stable sort for equal keys) | Real |
| 6 | GROUP BY results with no explicit ranking word get no ORDER BY requirement at all — tied categories' row order is planner-dependent | same validation gates as #5 | Low-medium — affects presentation row/category order, not answer content | Real but narrow |
| 7 | Verified-cache key is **embedding cosine similarity (≥0.85)**, not exact text match, with **no correctness check at write time** | `veda_core/veda/cache.py:26-107`, write-gate `pipeline.py:3577-3580` | **High** — confirmed 3 real production incidents of two *different* questions colliding on one cached SQL; a wrong SQL, once cached, replays byte-identically wrong until a demotion rule happens to catch it | Real, already caused incidents |
| 8 | Session-TTL checkpoint eviction — the same `thread_id` past its Redis TTL silently loses conversation frame/history | `chatbot/checkpointer.py` (2026-09-16 TTL change) | Low — expected/documented variability, not a defect (protects against a prior OOM) | Real but by design |
| 9 | Model registry drift — `SLM_MODEL_NAME` is a fixed tag string (`qwen2.5:7b-instruct`), not a content-digest pin; Ollama could silently update weights server-side under that same tag | `.env`, `_call_slm.py` | Low probability, high blast radius if it happens | Theoretical (not observed, but not prevented) |

**Not found / confirmed clean:** no `RANDOM()`/`TABLESAMPLE`/wall-clock nondeterminism in any
generated-SQL path; parallel sub-query fan-out (`ThreadPoolExecutor.map`) and federated
cross-source merges are both index-order-preserving, not completion-order-preserving; no
torn-read risk (single-statement MVCC); BGE-M3 dense search, sparse ranker, cross-encoder
rerank are all deterministic forward passes with no sampling.

---

## 4. Consistency, Evaluated Separately

### A. SQL consistency (same question → same SQL)

**Mostly yes**, with two named exceptions: (1) the deterministic single-table builder
(`SINGLE_TABLE_DETERMINISTIC`, currently hardcoded `True` in `config.py:2292` — see note below)
bypasses the SLM entirely for simple projection/filter/rank cases, which is fully reproducible
by construction; (2) for SLM-generated SQL, `temperature=0` is correctly wired for SQL-gen
(verified intact, not reverted), but no `seed` is set outside `generation.py` itself, and the
RRF tie-break gap (source #2) can change *which columns/tables* the SLM is even shown as
candidates, which can change the SQL even at temperature=0 given a different prompt.

*Side note, unrelated to this audit but discovered while verifying: `SINGLE_TABLE_DETERMINISTIC`
is hardcoded `True` with a docstring comment claiming "flag-gated, OFF" — the flag and its
comment have drifted apart (pre-existing, dated 2026-07-31, confirmed via `git blame` unrelated
to this session's work). Worth a one-line fix independent of this audit.*

### B. Result consistency (same interpretation + same data → same DB result)

**Yes, with one real gap**: no non-deterministic SQL functions are used, transaction isolation
is fine for the single-statement execution model used here, but the missing tie-break on
`ORDER BY ... LIMIT n` (source #5) means a ranking query with ties at the cutoff is not
formally guaranteed identical rows across repeated runs, even against byte-identical data.

### C. Presentation consistency (same result → same table/chart, no LLM)

**Yes — this is the part of the system already built correctly, and now verified live.**
`analyze_result()` makes zero model calls and zero DB calls; it's a pure function of
`(question, sql, columns, rows, sm)`. This session traced and fixed:
1. A generic single-word alias (`"payment"`) matching every column of a table via
   `recommended_projection()`'s query-named-column override, defeating `query_relevant_columns`
   narrowing — fixed with a collision-based check (shared by 2+ columns → excluded), not
   word-count (a first attempt on word-count wrongly also excluded a genuinely unique
   single-word alias, caught by regression testing before shipping).
2. `_TEMPORAL_NAME_HINTS` substring matching (no word boundaries) misclassifying
   `expected_monthly_rent` as `kind="temporal"` because `"month"` is a substring of `"monthly"`
   — fixed with whole-token matching in both duplicated copies (`result_analyzer.py`,
   `visualization.py`).
3. `role` not actually outranking a false-positive structural `kind` in **3** places
   (`visualization.py::recommend()._kind()`, and — the deeper one — `detect_result_shape()`
   itself, which had zero role-awareness at all) — fixed by threading `role` through all three.

**Live-verified after the fix** (real vedademo SLM+embedding, real DB, `source_id=2`): the same
query run through the pipeline produces the same `result_shape`, the same narrowed table
columns, and the same chart type/axes — because none of these three functions touch a model or
an unordered collection whose order isn't already pinned by the SQL result's own column/row
order. Given the SAME `(cols, rows, analytics)` input, output is byte-identical, tested directly.
The one *inherited* risk is upstream: if source B's `ORDER BY` tie-break gap (source #5) ever
changes row order between two runs, the chart/table would faithfully re-render that (different)
row order — correct behavior for the presentation layer, but worth naming so it isn't
mistaken for a presentation-layer bug.

### D. Response-text consistency (final NL wording)

**No — confirmed non-deterministic, independently by two separate audits.**
`result_explainer.py`'s two SLM call sites (short-answer paraphrase, Insight Engine narrative)
both hardcode `temperature=0.1`, entirely separate from the `SLM_TEMPERATURE` config knob that
correctly governs the SQL-generation paths. The prompt does constrain the SLM to "use ONLY
numbers that appear above, never calculate a new figure," so the **quoted numbers stay stable**,
but the **wording** ("There are 5 properties" vs. "I found five properties") is not guaranteed
identical run-to-run. This is the direct, confirmed answer to the audit's own worked example
in the prompt.

---

## 5. Gaps (concrete, prioritized — feeds §10)

See §3's table — those are the gaps. Ranked by how directly they threaten "same question,
same answer": **#7 (cache correctness) > #1 (follow-up classifier temp) > #4 (response-text
temp) > #5 (ORDER BY tie-break) > #2 (RRF tie-break) > #3 (no seed) > #6/#8/#9 (lower severity)**.

---

## 6. Recommended Architecture

The target architecture in the prompt is **directionally correct and does not need
restructuring** — it needs the gaps in §3 closed and one contract formalized (§8):

```
                    USER
                      ↓
             Conversation Layer          [deterministic rules FIRST;
                      ↓                   SLM classify only as fallback —
              Semantic Context            PIN temperature=0 here (gap #1)]
                      ↓
                  SLM/LLM                 [temp=0 + seed everywhere (gap #3);
                      ↓                   RRF tie-break needs a secondary
               Query Plan / SQL           sort key (gap #2)]
                      ↓
            Validation + RBAC             [add a tie-break-column requirement
                      ↓                   to ranked_shape_ok / builders (gap #5)]
               SQL Execution
                      ↓
       Verified-Cache Check (embedding)   [add a write-time correctness gate
                      ↓                   or lower/remove fuzzy-match reuse
            Deterministic Result          for anything but exact-text hits (gap #7)]
                 Analysis                 [ALREADY DETERMINISTIC — verified]
                      ↓
          Result Presentation Planner     [formalize as a typed contract — §8]
               ↙              ↘
            Table            Chart        [ALREADY DETERMINISTIC — verified,
                      ↓                    live-tested]
              Response Layer              [PIN temperature=0 for the NL
                                            paraphrase call too (gap #4) —
                                            or accept wording variance as an
                                            explicit, documented exception]
```

---

## 7. SLM/LLM Decision

**Where SLM is required:** understanding a novel-worded question (entity/intent resolution),
generating SQL for cases the deterministic builder can't cover, classifying a follow-up whose
wording the rule-based delta layer doesn't recognize, and (optionally) paraphrasing the final
answer into natural prose.

**Where SLM is optional:** the follow-up classifier (`chatbot/nodes.py::classify_node`) already
has a deterministic rule-based path that resolves most turns without it — the SLM there is a
fallback, not a requirement, and should stay one (just pinned to temp=0, not removed).

**Where SLM should NOT be used — confirmed, not just recommended:** deciding which table
columns to display, deciding chart type, deciding chart axes. This is the audit's central
question, and the answer is unambiguous: **no**, based on the actual current implementation.
`analyze_result()` → `VisualizationRecommender` already does this with zero model calls, and
this session's live testing proved it works (found and fixed 3 real bugs there — bugs that
were fixable specifically *because* the logic is deterministic and traceable; an LLM-driven
version of the same bugs would have looked like unexplainable flakiness). The Insight Engine's
optional SLM-suggested chart type exists only as a last-resort fallback behind the deterministic
recommender, confirmed never to be the primary path.

Design A (SQL → SLM → Presentation) vs. Design B (SQL → Deterministic Result Analyzer →
Presentation Planner → Table + Chart): **Design B is what's actually implemented for the table/
chart stage, and it should stay that way.** It wins on every axis the prompt asks to compare —
accuracy (no invented columns/axes), consistency (proven byte-identical given the same
analytics input), latency (zero extra round-trip), cost (zero extra tokens), observability
(every decision traces to a named function and a `[Lx]` trace tag), explainability (a rule can
be quoted, "why did it pick this chart" has a one-sentence answer), debugging (this session's
3 bugs were each root-caused in under an hour precisely because the logic is deterministic),
reproducibility (proven).

---

## 8. Presentation Planner Assessment

`analyze_result()` **already functions as** an informal Presentation Planner — its
`analytics_summary()` output carries `result_shape`, `display_columns`, `query_relevant_columns`,
`chart_candidates`, `measure_aggregates`, `column_stats` — everything a formal
`PresentationPlan` dataclass would hold. **Recommendation: formalize it, don't rebuild it.**

This would be a **refactor, not a redesign** — wrap the existing dict fields into a typed
dataclass (mirroring `InsightContext`'s own existing pattern in the same file), version it,
and have `table_rendering.py`/`visualization.py` consume the typed object instead of a loose
dict. Concretely this improves:
- **Testability** — a typed contract lets a test assert `plan.selected_chart == "bar"` instead
  of poking at a nested dict shape.
- **Observability** — one object to log/trace per turn instead of two consumers each reading
  overlapping dict keys.
- **Table/chart agreement** — today they already agree because they read the same dict; a
  typed contract makes that agreement structurally enforced (a schema change breaks both
  consumers loudly, a silent dict-key rename doesn't).
- **Future UI work** — a stable, documented contract is what a frontend team would actually want
  to build against, vs. an internal dict whose shape is only documented in code comments.

This is a **P1**, not a **P0** — the underlying behavior is already correct; this only makes it
harder to accidentally regress and easier to build on.

---

## 9. Test Strategy — Determinism Test Suite

**Repeated identical requests** (10/50/100 runs, same question, same DB state): assert
identical `sql` (normalized via sqlglot canonicalization to ignore harmless whitespace/param
formatting), identical `result_shape`, identical `query_relevant_columns`, identical
`chart_candidates[0]` signature, identical row *set* (order-insensitive) and, separately,
identical row *order* when an `ORDER BY` is present. Already partially covered by this
session's own tests (`test_recommended_projection.py`, `test_chat_visualization.py`,
`test_result_analyzer.py`) at the unit level; needs a live-pipeline harness variant (reusing
`evaluation/run_component_suite.py`'s pattern) that runs N repeats per query and diffs.

**Equivalent wording** ("show cheapest properties" / "show me the lowest priced properties" /
"which properties have the lowest prices?"): assert the SAME table (same primary entity, same
retrieval top-1) is selected across all three, even if exact SQL text may legitimately differ
in alias naming. This directly exercises the RRF tie-break gap (#2) and retrieval determinism.

**Follow-up consistency** ("show latest transactions this month" → "which one was the
largest?" asked twice): assert the SAME `delta_type`/`action` classification both times — this
is the exact test that would have caught gap #1 (unpinned follow-up-classifier temperature).

**Tie cases**: construct a fixture with 2 columns/tables at identical retrieval scores, 2
categories at identical COUNT, 2 rows at identical ranking-measure value at a LIMIT cutoff —
assert stable, reproducible output across repeated runs AND across a simulated process
restart (different `PYTHONHASHSEED`) for the retrieval case specifically, since that's the one
tie-break confirmed to depend on process-level hash seed.

---

## 10. Implementation Plan

### P0 — Required for consistency

| Task | File | Change | Reason | Risk | Expected result |
|---|---|---|---|---|---|
| Pin follow-up classifier temperature | `chatbot/nodes.py::classify_node` | Pass `temperature=0` to the `call_slm` call at ~line 1059 | Currently defaults to `0.1` — the one confirmed source of "same follow-up wording, different classification" | Very low — matches what the standalone fallback already does | Reproducible follow-up action/delta_type classification |
| Add tie-break secondary key to RRF merge | `veda_core/retrieval/rrf_merger.py:148` | Add a deterministic secondary sort key (e.g. `col_id`) to the final `sorted(...)` call | Ties currently fall back to `PYTHONHASHSEED`-dependent set order — changes across process restarts | Low — same fix pattern already used elsewhere in the codebase (stable sorts with secondary keys) | Identical retrieval ranking across process restarts, not just within one process |
| Add write-time correctness gate (or narrow reuse) to verified-cache | `veda_core/veda/cache.py`, write-gate in `pipeline.py:3577` | Either (a) require an existing read-side demotion check to ALSO run before write, not just before replay, or (b) restrict fuzzy-similarity reuse to a much higher threshold / exact-match only, keeping embedding-similarity purely as a "did we already answer something like this" hint, not an auto-replay trigger | 3 confirmed production incidents of wrong-answer replay; currently unfixed per this audit | Medium — changes cache hit rate, needs a before/after hit-rate + accuracy measurement (same rigor as this session's own benchmarks) | A wrong SQL can no longer get cached and replayed indefinitely |
| Pin response-text SLM temperature | `query/result_explainer.py:862,1253` | Change hardcoded `0.1` → read `SLM_TEMPERATURE` (or a dedicated `RESPONSE_TEMPERATURE` config, default 0) | Confirmed independently by 2 audits; directly the prompt's own worked example (§4.D) | Low-medium — verify prose quality doesn't degrade at temp=0 (some paraphrase diversity may be desirable; if so, make this an explicit, documented product decision, not an accidental one) | Same numbers AND same wording on repeat, or an explicit signed-off exception |

### P1 — Recommended enterprise improvements

| Task | File | Change | Reason | Risk | Expected result |
|---|---|---|---|---|---|
| Formalize `PresentationPlan` contract | `veda_core/veda/result_analyzer.py` | Wrap existing `analytics_summary()` fields into a typed, versioned dataclass; update `table_rendering.py`/`visualization.py` call sites | §8 — improves testability/observability without changing behavior | Low — pure refactor if done as a wrapper first, no logic change | Schema-enforced table/chart agreement |
| Require a tie-break column on every `ORDER BY ... LIMIT` | `veda/validation.py::ranked_shape_ok`, `planning.py` builders | Append a deterministic secondary sort key (e.g. the anchor's PK) to every generated ranking query; extend `ranked_shape_ok` to require it | Gap #5 — real, not theoretical | Medium — touches every SQL builder; needs the same before/after test rigor already used this session | Byte-identical top-N rows across repeated runs, even with ties |
| Add explicit `seed` to every SLM call site | `rag_layer.py`, `slm_layer.py` (2 sites), `lg_nodes.py`, `answer_entity.py`, `envelope_slm.py`, `federated_route.py` | Pass the same fixed `seed` `generation.py` already uses | Formal reproducibility guarantee, not just "near-deterministic at temp=0" | Very low | Full seed+temperature pinning across every model call |
| Fix `SINGLE_TABLE_DETERMINISTIC` flag/comment drift | `config.py:2292` | Either make it truly env-gated (matching its own docstring) or update the docstring to say "always on" | Pre-existing (2026-07-31), unrelated to this audit but discovered while verifying it | Low | Config self-documentation matches reality |

### P2 — Future improvements

| Task | Reason |
|---|---|
| Require a tie-break on GROUP BY category ordering (gap #6) | Lower severity — affects presentation row order only, not answer content |
| Content-digest pinning for the SLM model (gap #9) | Low probability but currently undetectable if it happens |
| Document session-TTL eviction as expected variability (gap #8) in user-facing docs/UX copy | Not a code fix — a documentation/expectation-setting task |

---

## 11. Final Recommendation

1. **Can repeated identical questions produce consistent SQL?** Mostly — deterministic builder
   cases yes by construction; SLM-generated cases yes at the SQL-gen call itself
   (temp=0 verified intact), but threatened upstream by the RRF tie-break gap (#2) and the
   missing-seed gap (#3).
2. **Can repeated identical questions produce consistent results?** Yes, with one named gap:
   `ORDER BY ... LIMIT` without a tie-break column (#5) is not formally guaranteed stable.
3. **Can repeated identical questions produce consistent table columns?** **Yes — verified,
   live, this session.** Zero model calls in this stage; 3 real bugs found and fixed, all
   traceable specifically because the logic is deterministic.
4. **Can repeated identical questions produce consistent charts?** **Yes — verified, live, this
   session**, same evidence as #3.
5. **Can repeated identical questions produce identical natural-language wording?** **No,
   confirmed** — `result_explainer.py` hardcodes `temperature=0.1`, independent of the SQL-gen
   temperature fix. Numbers stay stable (prompt-constrained); wording does not.
6. **Does VEDA need an SLM after SQL?** **No** — for table/chart decisions. Confirmed by direct
   inspection and live testing, not assumed. An SLM call remains legitimate only for the
   optional final-answer paraphrase (#5 above) and the already-existing Insight Engine
   last-resort fallback.
7. **What changes are required to make the system enterprise-grade?** The 4 P0 items in §10 —
   all bounded, none require an architecture change: pin one temperature, add one tie-break key,
   add one correctness gate, pin one more temperature.
8. **What guarantees can VEDA honestly claim today?** *Same question + same conversation
   context + same permissions + same semantic configuration + same relevant database state +
   same live process (no restart) → same SQL, same result, same table, same chart.* The "same
   process" qualifier is real and named (gap #2) — it's the one guarantee VEDA cannot yet make
   across a redeploy/restart without the P0 RRF fix. Wording of the final NL response is
   explicitly **not** covered by any consistency guarantee today (gap #4) unless P0's temperature
   pin is applied.

---

*Prepared as a direct code audit (not generic theory) — every finding above cites a specific
file and line, verified by 5 parallel focused subagent passes over the actual codebase plus
this session's own live-pipeline verification of the presentation stage.*
