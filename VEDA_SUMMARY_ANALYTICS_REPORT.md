# VEDA Summary Pipeline — Analytical Depth Improvements

Goal: make analytical summaries genuinely useful (grounded business analytics) instead
of shallow 1–3-sentence answers — **without** just raising the word limit, without new
LLM calls, and without letting the SLM invent/calculate analytics. Builds on the
existing semantic/validation/population-consistency work; none of it is undone.

Architecture realized:
`EXECUTED RESULT → DETERMINISTIC VERIFIED ANALYSIS → STRUCTURED ANALYTICAL CONTEXT →
SUMMARY SLM (narrator only) → GROUNDED BUSINESS NARRATIVE`.

---

## 1. Audit — how the summary is built (Phase 1)

Trigger: `veda/pipeline.py` L7b block after execution (`NL_ANSWER_ENABLED`). It:
1. materializes `row_dicts`; 2. computes a deterministic fallback; 3. runs
`result_analyzer.analyze_result` → `InsightContext` (column roles, `result_shape`,
`patterns`, chart candidates); 4. picks `patterns[:2]`; 5. branches
`INSIGHT_ENGINE_ENABLED` (default off) → `run_insight_engine`, else `run_nl_answer`.
`run_nl_answer` (`query/result_explainer.py`) builds facts (`_extract_facts` +
`_numeric_aggregates`), glossary, rank/findings/shape lines, a style exemplar, and a
prompt; a numeric guard rejects any ungrounded number. Paths: Tier-1 (pipeline) and
Tier-2/heads (`veda_hybrid`) both call the same `run_nl_answer`; Insight Engine is an
alternate single-call enrichment (off by default). No duplicate/conflicting generator.

## 2. Root causes of shallow summaries (Phase 2 report)

- **RC-A** — `run_nl_answer` prompt hardcodes "SHORT answer 1-3 sentences, max ~55 words"
  + "add ONE short sentence", uniformly for every shape (`result_explainer.py`).
- **RC-B** — only `patterns[:2]` reach the summarizer (`pipeline.py`, `result_explainer.py`).
- **RC-C** — `NL_SUMMARY_MAX_TOKENS=160` caps length regardless of evidence.
- **RC-D** — deterministic findings lacked the highest-value analytics: no **leader/laggard
  entity**, no **spread**, no **above/below-average** distribution for GROUPED/RANKING
  (only concentration + top_gap ratios).
- **RC-E** — no summary **mode**: scalar and grouped/trend used identical policy.

Upstream analytics (measure/dimension/operator/display/validation) were already correct
— the deficiency was entirely in the summary layer.

## 3. Design (Phase 2–4, minimal & generalized)

1. **Enrich verified findings** (deterministic, in `result_analyzer.detect_patterns`):
   add `leader`, `laggard`, `spread`, `distribution` for RANKING/GROUPED/DISTRIBUTION.
   Labels come from the **dimension column's own value** — already the display name when
   the SQL grouped by the display column (the canonical upstream decision); the id→name
   choice is NOT re-derived here, and an identifier-role dimension is never used as a label.
2. **Evidence-adaptive modes** (`result_explainer._summary_mode`): `brief` (scalar /
   single-row / non-analytical) vs `analytical` (multi-row RANKING/GROUPED/DISTRIBUTION/
   TREND/PIVOT). Mode drives prompt shape, findings count, and token budget.
3. **Structured analytical context** (`_analytical_context_block`) — operation, ranking,
   temporal, explicit-id — REUSED from the run's own understanding (canonical
   `aggregate_operator`, `_rank_column_for_nl`, `tf`, `user_requested_identifier`), never
   recomputed in the summary layer.
4. **Prompt rewrite** — grounded analytical **narrator**: answer first (most
   decision-relevant result), then synthesize the supplied verified findings
   (leaders/laggards, comparisons/gaps, spread/concentration, trends, outliers) that are
   relevant; prioritize insight over dumping metrics; use display names but **preserve
   explicit id requests**; only supplied numbers; **never calculate** totals/%/avg/diff/
   growth; no invented causes/currency/units; avoid generic filler; evidence-adaptive
   length with bounded budgets.

## 4. Population consistency + scalability (Phases 6–7)

`ANALYSIS_MAX_ROWS` (default 50000) bounds BOTH the pattern sweep (`analyze_result`) and
the summary's numeric aggregates (`_numeric_aggregates`) with the **same deterministic
cap** — so (a) the two never mix populations (RC-4 holds even when bounded) and (b)
neither does an unbounded O(rows) pass on a very large enterprise result. `row_count`
reported to the user is always the TRUE total; only the analysis scan is bounded. Under
the cap (typical), everything is full/exact. Leader/laggard/spread are single-pass
max/min/count (O(1) extra memory).

## 5. Files changed

| File | Change |
|---|---|
| `veda_core/veda/result_analyzer.py` | `_fmt_num`; `detect_patterns` gains `dimensions` + leader/laggard/spread/distribution findings; `analyze_result` bounds the sweep to `ANALYSIS_MAX_ROWS` (true `row_count` preserved) |
| `veda_core/query/result_explainer.py` | `_summary_mode`, `_analytical_context_block`; `run_nl_answer` mode-aware prompt/budget/findings + `analytical_context` param; `_numeric_aggregates` bounded to `ANALYSIS_MAX_ROWS` |
| `veda_core/veda/pipeline.py` | pass ALL findings + resolved `analytical_context` to `run_nl_answer` (top-2 still used for the deterministic-fallback blend) |
| `veda_core/config.py` | `NL_SUMMARY_ANALYTICAL_MAX_TOKENS` (320), `NL_SUMMARY_MAX_FINDINGS` (5), `ANALYSIS_MAX_ROWS` (50000) |
| `tests/test_summary_analytics.py` | **new** — 13 model-free tests |
| `tests/test_summary_prompt.py`, `tests/test_summary_numeric_guard.py` | updated 3 assertions to the new (intended) prompt wording |

Diff: 4 source files, +307/−39; +212-line test file.

## 6. Before / after (deterministic, model-free evidence)

Query: "average carpet area per project" (grouped, 5 projects) — verified findings now:
```
leader       :: Green Heights has the highest avg_carpet_area_sqft at 2800
laggard      :: Sunrise has the lowest avg_carpet_area_sqft at 550
distribution :: 2 of 5 project_name are above the average avg_carpet_area_sqft of 1365.6
spread       :: avg_carpet_area_sqft ranges from 550 to 2800 across 5 groups
```
- **Before**: prompt = "1-3 sentences, max ~55 words", ≤2 findings, ~160-token budget →
  a one-line answer with at most one pattern.
- **After**: analytical mode → "3-5 concise sentences", up to 5 verified findings woven,
  320-token budget, resolved-context block, explicit-id + no-invented-calc guards intact.
- Scalar "total revenue" → brief mode (1-2 sentences, 210-token budget) — unchanged
  brevity, no padding.

## 7. Test results

New `test_summary_analytics.py`: **13 passed** (modes, findings, schema-independence,
grounding, population-consistency+bound, explicit-id). Adjacent summary/analyzer suites
green in isolation: `test_result_analyzer` 34, `test_analytics_context` 15,
`test_result_explainer` 36, `test_summary_blend` 6, `test_summary_numeric_guard` 12,
`test_summary_prompt` 8, `test_grouped_aggregation_operators` 25. Full per-file sweep of
`tests/`: only **2 files fail, both pre-existing and unrelated** (verified failing
identically on untouched HEAD) — `test_qsr_resolution` (needs a populated value-store
DB) and `test_tier2_thinking` (`ImportError: SLM_OLLAMA_BASE_URL`, a sys.path bug in
that test, untouched here). **Zero regressions attributable to this work.**

## 8. Performance implications

- Leader/laggard/spread/distribution are O(n) single-pass, O(1) extra memory.
- The pattern sweep and summary aggregates are now **bounded** by `ANALYSIS_MAX_ROWS`
  (previously the Stage-1 full-population change was unbounded) — eliminating the
  large-result OOM/latency risk while keeping population consistency.
- No new SLM/DB calls; prompt token cost is bounded by mode (≤320 completion tokens).

## 9. Invariants preserved / non-goals

- No new LLM calls; SLM narrates only precomputed findings (numeric guard still active).
- No `_id`→`_name` re-derivation in the summary layer; dimension labels come from the
  role-correct dimension column decided upstream. Identifier-role dimensions are not
  used as labels; explicit id requests are preserved end-to-end.
- No schema/table/column/business-entity hardcoding (schema-independence test proves
  role-driven behavior).
- Population-consistency (RC-4) preserved and now bounded.
- No visualization/chart work touched (out of scope, per instruction).

## 10. Remaining limitations / follow-ups

- Temporal findings still rely on the existing `growth`/`decline` pattern (start→end
  change); peak/trough and period-over-period deltas are candidates for a future
  deterministic finding family.
- The measure/dimension names in the resolved-context block are currently the operation/
  ranking/temporal/id facts; adding the resolved measure/dimension *column identities*
  (from the shared semantic plan) is a small future enrichment.
- Analytical depth benefits are realized when a live summary model runs; on this
  hardware only the deterministic inputs and the prompt/mode selection are verifiable
  (done, model-free).
