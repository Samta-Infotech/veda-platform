# VEDA Visualization — Validation & Targeted Fixes

Validated the CURRENT visualization pipeline model-free (no SLM/LLM/embeddings/Ollama),
built a full shape matrix, traced first-failure points across all deciders, and fixed
only proven generalized gaps. No semantic/summary architecture changed. No SLM added.

New suite: `tests/test_visualization_matrix.py` — **21 passed**. Baseline preserved:
`tests/test_analytics_integration.py` **47 passed**, `tests/test_chat_visualization.py`
**24 passed**. All affected analyzer/summary/semantic suites green.

---

## 1. Current behavior found (already correct)

The pipeline is largely sound. Precedence (in `apps/chat/services.py::_build_visualizations`):
**VisualizationRecommender (primary) → validated SLM suggestion → `chart_candidates[0]`
(analyzer fallback) → none.** Verified correct final output for:

| Shape | Final chart |
|---|---|
| A category + measure (GROUPED) | pie + bar (x=category, y=measure) |
| C temporal + measure (TREND) | line + bar (x=temporal, y=measure) |
| E category + 2 measures (PIVOT) | line_histogram (combo) |
| F 2 measures, no dimension | no chart |
| G identifier + measure | no chart |
| I detail table | no chart |
| J scalar | no chart |
| K high-cardinality category | single bar (top-N + "Other", ≤10 labels) |
| L single grouped row | no chart |
| M null-heavy | no chart, no crash |
| N empty | no chart |
| O distribution (COUNT/dim) | pie (+bar) |

Semantic reuse: the recommender consumes the engine's `analytics.column_stats`
(kind/role from the semantic model) as authoritative; identifiers and free-text are
correctly excluded from axes; the analyzer's `chart_candidates` is a real fallback.

## 2. Genuine bugs found & fixed

### FIX-3 — `_spec_from_suggestion` returned a list → `_build_visualizations` crash ⚠️ HIGHEST
- **Fixture**: recommender returns `[]` and a **bar/pie** candidate or SLM suggestion
  reaches the fallback (e.g. the pre-fix numeric-category case; any bar suggestion with
  Insight Engine on).
- **Expected**: a single chart spec or none.
- **Actual**: `_spec_from_suggestion` returned `_category_numeric(...)` — a **list** — and
  `_build_visualizations` called `spec.to_dict()` on it (`AttributeError`), un-wrapped
  (services.py:244) → the turn's response generator crashes.
- **First failing component**: FINAL_VISUALIZATION_BUILD (`services._spec_from_suggestion`).
- **Root cause**: inconsistent return type (list for bar/pie, single spec for line).
- **Fix**: return the spec matching the requested type (`specs[0]` fallback), or None.
  `apps/chat/services.py::_spec_from_suggestion`.

### FIX-1 — a RANKING led with a misleading PIE
- **Fixture**: `SELECT dim, AGG(m) ... GROUP BY dim ORDER BY AGG(m) DESC LIMIT N`.
- **Expected**: BAR (a top-N is not part-of-whole; analyzer's canonical for RANKING is bar).
- **Actual**: recommender's `_category_numeric` ignored `result_shape` → returned pie+bar,
  leading with **pie** (misrepresents proportions — the N don't sum to the whole).
- **First failing component**: CHART_CANDIDATE_GENERATION / recommender chart selection.
- **Root cause**: category+numeric branch was shape-agnostic.
- **Fix**: when `analytics.result_shape == "RANKING"`, drop the pie and lead with bar.
  `apps/chat/visualization.py::recommend`. Shape-driven, not name-driven.

### FIX-2 — recommender re-guessed role from dtype (numeric-valued CATEGORY dimension)
- **Fixture**: a CATEGORY dimension whose values are numeric (year, rating, postal code) +
  a measure.
- **Expected**: chart with the category as the X axis.
- **Actual**: `_kind` binned by structural `kind` ("numeric") → the dimension landed in the
  measure pool → recommender returned `[]` (only the analyzer fallback recovered a chart,
  and before FIX-3 that recovery path crashed).
- **First failing component**: COLUMN_ROLE_CLASSIFICATION (recommender `_kind`).
- **Root cause**: the authoritative semantic `role` was ignored except for identifier/text.
- **Fix**: `_kind` now honors `role` — `dimension`→categorical, `measure`→numeric,
  `date`→temporal — falling back to structural `kind` only when no role exists (federated).
  `apps/chat/visualization.py::recommend._kind`. Naming/dtype never override metadata (Phase-5).

## 3. Before / after

```
B  RANKING (top-3 by amount)
   BEFORE: [pie, bar]  (pie leads — misleading part-of-whole)
   AFTER:  [bar]       x=dimension, y=measure

NUMCAT  numeric-valued CATEGORY dimension + measure
   BEFORE: recommender []  → analyzer-fallback bar → (pre-FIX-3) .to_dict() CRASH on the list
   AFTER:  recommender [pie, bar]  x=category, y=measure  (no fallback needed, no crash)

Fallback path (any bar/pie candidate/suggestion, recommender empty)
   BEFORE: _spec_from_suggestion → list → _build_visualizations spec.to_dict() → AttributeError
   AFTER:  single spec → renders correctly
```

## 4. No-chart cases preserved (did NOT inflate chart count)

Scalar (J), detail table (I), empty (N), two-measures-no-dimension (F),
identifier+measure (G), single row (L), null-heavy (M) all remain **no chart** — verified
in the matrix. High cardinality (K) stays a single bucketed bar, not hundreds of bars.

## 5. Documented limitations (NOT bugs — intentionally unsupported)

- **Multi-series** (D temporal+category+measure, Q multi-dimension): the second dimension
  is not rendered as a series — the chart uses the primary axis + measure. Encoded as
  test evidence (`test_D_..._series_dropped_limitation`, `test_Q_..._second_dropped_limitation`).
  Adding series support is a feature, out of scope for a fix pass.
- **Numeric-identifier RANKING** (H): a ranking whose only label is a numeric identifier
  yields no chart (the string-id RANKING rescue can't find a non-numeric label). String-id
  rankings do rescue. Consistent, defensible; recorded as current behavior.
- **Explicit-vs-accidental identifier**: no explicit-id signal currently reaches the
  visualization layer, so both are treated the same (suppressed / RANKING-rescued by type).
  Charting an explicitly-requested id would require threading that intent into viz.

## 6. Test results

- `tests/test_visualization_matrix.py`: **21 passed** (shapes A–Q, axis correctness,
  cross-tier consistency, schema independence, all three fixes).
- `tests/test_analytics_integration.py`: **47 passed** (baseline preserved).
- `tests/test_chat_visualization.py`: **24 passed** (no regression).
- Also green: result_analyzer 34, analytics_context 15, result_explainer 36,
  summary_analytics 13, summary_prompt 8, summary_numeric_guard 12, summary_blend 6,
  semantic_validation 16, grouped_aggregation_operators 25, business_explain 23,
  execution_state_reuse 14.
- Pre-existing, unrelated (not regressions): `test_qsr_resolution` (value-store DB),
  `test_tier2_thinking` (`SLM_OLLAMA_BASE_URL` import path).

## 7. Files changed (production)

| File | Function | Change |
|---|---|---|
| `apps/chat/visualization.py` | `recommend._kind` | honor semantic role over structural kind (FIX-2) |
| `apps/chat/visualization.py` | `recommend` | RANKING leads with bar, drops pie (FIX-1) |
| `apps/chat/services.py` | `_spec_from_suggestion` | return one spec, not a list (FIX-3) |

## 8. Remaining limitations

- Multi-series charts (temporal/category × category) — unimplemented feature.
- Numeric-identifier rankings and explicit-id charting — policy decisions, not bugs.
- `apps/chat/services.py::_build_visualizations` is Django-coupled and cannot be imported
  model-free; its precedence is faithfully replicated in the matrix suite (mirroring the
  fixed logic). Full integration verification needs the app/Django test layer.
- Live-model validation of the Insight Engine SLM suggestion path still pending (that path
  is off by default; FIX-3 makes it crash-safe if enabled).
