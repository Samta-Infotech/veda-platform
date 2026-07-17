# Visualization: How Chart Recommendation Works

This document explains the chart-recommendation system that turns a query's
tabular result into an optional chart. It covers where the code lives, the
full decision logic, the data contract with the frontend, its known upstream
dependencies, and the bug fixes applied on 2026-07-10.

## 1. Where it lives

| Concern | File |
|---|---|
| Chart-type decision logic | `apps/chat/visualization.py` |
| Wiring into the chat response stream | `apps/chat/services.py` |
| API surface (JSON + SSE) | `apps/chat/views.py` |

Automated test suite: `tests/test_chat_visualization.py` (added 2026-07-15,
extended 2026-07-16 for multi-viz — see §5/§7).

## 2. What it is not

- Not an LLM call — `VisualizationRecommender` is deterministic Python, no
  model involved.
- Not a chart-rendering library — no matplotlib/plotly/D3, no image or HTML
  output.
- Not a LangGraph node — it's not part of the `veda_core` retrieval/SQL
  pipeline. It runs entirely in the Django `apps/chat` layer, downstream of
  wherever the query engine result comes from.
- Not a replacement for the table — the plain markdown table
  (`services.py::_rows_to_markdown_table`) is always rendered too. Charting is
  additive, never a substitute.

## 3. Full data flow

```
User query
  → chatbot / LangGraph engine (veda_core)
  → executes SQL, returns res0 = {"cols": [...], "rows": [...], ...}
  → apps/chat/services.py: res0 = response.get("engine_result")
  → _build_visualizations(res0)                     [services.py:294-309]
      → _visualization_recommender.recommend(cols, rows)
      → returns 0+ VisualizationSpec objects (2026-07-16: often 2 now — see
        §5's multi-viz update)
      → ALL specs' .to_dict()s are yielded as ONE SSE event (2026-07-16,
        changed from one event PER spec):
          {"event": "visualization", "data": {"visualizations": [spec_dict, ...]}}
  → apps/chat/views.py: both the JSON response builder and the SSE
    generator append this single "visualization" event into the response[]
    array identically to "content" blocks — so a chart-eligible answer now
    contributes exactly one response[] entry containing an array, not one
    entry per chart
  → frontend renders each entry in data.visualizations[] — order preserved,
    visualizations[0] is always the SAME chart type this module would have
    returned before the 2026-07-16 multi-viz change (see §5)
```

`res0`'s `cols`/`rows` trace back through:
`chatbot/run.py` → `chatbot/nodes.py::call_engine_node` →
`veda_core/veda/pipeline.py::run_query` → `execute_sql(param_sql, params)`
(`pipeline.py:752`).

**Important upstream caveat:** SQL feeding `cols`/`rows` comes from one of
four divergent generators in `veda_core`, and **none of them guarantee**
`ORDER BY` or `GROUP BY`:

- Hand-built f-string templates in `pipeline.py` (plain `SELECT ... LIMIT
  100`, no GROUP BY/ORDER BY at all).
- `veda_core/veda/generation.py::generate_sql()` — an LLM free-writes SQL;
  its own prompt only adds `GROUP BY` when the question says "per"/"by"/
  "each" — otherwise it's told not to.
- `veda_core/veda/planning.py::try_multitable` — template-based; `ORDER BY`
  is only added for ranked/top-N queries, never for general time-series.
- `veda_core/query/sql_builder.py` (IR-driven / SLM path) — only emits
  `GROUP BY`/`ORDER BY` if the upstream IR JSON already populated those
  arrays, which defaults to `[]` and depends on the SLM deciding to fill it.

Because of this, `VisualizationRecommender` cannot assume `rows` arrive
sorted or pre-aggregated — it has to defend against both (see §6).

## 4. Column classification

`_infer_kind(col_name, values)` (`visualization.py:130`) classifies every
returned column, in this order:

1. **temporal** — column name contains one of `date`, `month`, `year`,
   `week`, `day`, `time`, `period`, `quarter`, OR sampled values (first 20
   non-null) all look like dates (`datetime`/`date` objects, or strings
   matching `YYYY-MM` / `YYYY-MM-DD`).
2. **numeric** — all sampled values are `int`/`float`/`Decimal` (excluding
   `bool`).
3. **categorical** — anything else (default fallback).

Known blind spot: a column literally named `day` containing integers 1-31
(day-of-month, not a date) would be misclassified as temporal by the
name-hint check, since name hints are checked before value sampling.

## 5. Chart-type decision tree

`recommend(cols, rows)` (`visualization.py:99`):

**Step 1 — pick one dimension:**
```python
dimension_idx = temporal_idx[:1] or categorical_idx[:1]
```
The **first** temporal column wins if any exists; otherwise the **first**
categorical column. Only one dimension is ever used — additional
dimension-like columns are not charted (they remain visible in the table).

**Step 2 — match rules, in priority order:**

| Priority | Condition | Result |
|---|---|---|
| 1 | dimension + **≥2 numeric** columns | `line_histogram`: combo chart using the dimension + the **first two** numeric columns (bars = first numeric, line = second numeric). Single chart — see §9, this shape is intentionally NOT extended to multi-viz. |
| 2 | **temporal** dimension + exactly 1 numeric used, >1 row | **`[line, bar]`** (2026-07-16, was `line` only) — bar is a second, equally-confident rendering of the exact same ordered `(labels, values)` data, not a separate guess. `line` is always `visualizations[0]`. |
| 3 | **categorical** dimension + 1 numeric, ≤6 distinct categories after aggregation | **`[pie, bar]`** (2026-07-16, was `pie` only) — same totals, same confidence for both. `pie` is always `visualizations[0]`. |
| 3b | **categorical** dimension + 1 numeric, >6 distinct categories | top-9-by-value + "Other" bucket → `bar` only (unchanged — a pie with this many slices is unreadable, so this deliberately stays single-chart) |
| — | none match (e.g. all-text columns, single scalar, no dimension) | no chart — table remains the only output |

Priority 1 (combo chart) wins even over a temporal dimension, if a second
numeric column is present — it's considered more informative than either
metric charted alone.

**Multi-viz (2026-07-16):** `recommend()` can now return more than one
`VisualizationSpec` for the SAME result (priorities 2 and 3 above). This is
never "synthesizing" unrelated charts — every additional spec reuses the
exact `(labels, values)`/`(slices)` data and confidence the primary spec
already computed. The response wire format changed accordingly (§3): all
specs are now sent in one `{"visualizations": [...]}` event instead of one
event per spec.

Note: the "first two numeric columns" in priority 1 are taken positionally,
not semantically — if the first numeric column happens to be an ID or row
count rather than a real metric, it will still be used as the bar series in
the combo chart.

## 6. Building each chart (with the 2026-07-10 fixes)

### `line_histogram` — `_combo()` (`visualization.py:190`)
Zips the dimension + 2 numeric columns row-by-row; drops rows where either
numeric value isn't actually numeric; requires ≥2 usable rows or returns
`None` (falls through to the next rule). **Not sorted** — currently assumes
whatever order the SQL returned.

### `line` — `_line()` (`visualization.py:180`)
```python
ordered = sorted(rows, key=lambda row: (row[x_idx] is None, row[x_idx]))
```
**Fixed 2026-07-10:** previously plotted rows in raw DB order, which — given
§3's lack of guaranteed `ORDER BY` — could zig-zag a time series instead of
showing a clean trend. Now explicitly sorts by the x-axis (temporal) value
before building `labels`/`values`, with `None` values sorted last.

### `pie` / `bar` — `_category_numeric()` (`visualization.py:151`)
```python
totals: dict[str, float] = {}
for row in rows:
    if not _is_numeric(row[val_idx]):
        continue
    name = str(row[cat_idx])
    totals[name] = totals.get(name, 0) + _to_number(row[val_idx])
pairs = list(totals.items())
```
**Fixed 2026-07-10:** previously treated every row as its own slice/bar. If
the SQL wasn't `GROUP BY`'d (common — see §3), the same category (e.g.
"Mumbai") appearing in multiple rows produced multiple duplicate
slices/bars instead of one summed value. Now sums values per category name
first.

Then:
- `len(pairs) <= 6` → **`[pie, bar]`** (2026-07-16, was `pie` only) — one
  slice per category for the pie; the SAME `pairs` reused as `labels`/
  `values` for the paired bar, same confidence (0.9) as the pie.
- `len(pairs) > 6` → rank descending by value, keep top 9, collapse the rest
  into a single `"Other"` slice (sum of their values), then:
  - if the bucketed result is still ≤6 slices → `pie` only (in practice this
    branch is unreachable, since bucketing only ever runs when the original
    count was >6, and top-9 + "Other" is always ≥7 when rest is non-empty,
    or equal to the original >6 count when rest is empty)
  - otherwise → `bar` only, using `labels`/`values` arrays — deliberately
    NOT paired with a pie here; see §5 priority 3b.

## 7. Output contract (`VisualizationSpec.to_dict()`, `visualization.py:50`)

Each individual chart spec's own shape is unchanged and matches a fixed
frontend contract, never an image:

```jsonc
// bar / line
{"type": "bar", "title": "...", "x_axis_title": "...", "y_axis_title": "...",
 "chart_data": {"labels": ["..."], "values": [1, 2, 3]}}

// pie
{"type": "pie", "title": "...",
 "chart_data": {"slices": [{"name": "...", "value": 1}]}}

// line_histogram
{"type": "line_histogram", "title": "...", "x_axis_title": "...",
 "histogram_title": "...", "line_title": "...",
 "chart_data": {"labels": ["..."], "histogram_values": [1, 2],
                "line_values": [3, 4]}}
```

**Wire-level wrapper (2026-07-16, wire-contract change — see §3):** what
actually goes out on the SSE/JSON response is now ALWAYS this array wrapper,
even when there's only one chart:

```jsonc
{"visualizations": [
  {"type": "pie", "title": "...", "chart_data": {"slices": [...]}, "confidence": 0.9},
  {"type": "bar", "title": "...", "chart_data": {"labels": [...], "values": [...]}, "confidence": 0.9}
]}
```

`visualizations[0]` is always the same single chart this module would have
returned before 2026-07-16 — existing frontend code needs to change from
reading `data.type`/`data.chart_data` directly off a `"visualization"` event
to reading `data.visualizations[0].type` etc. (or iterating the whole array
to render/offer all of them).

`Decimal` values from psycopg2 (`NUMERIC`/`SUM`/`AVG` results) are normalized
to `float` via `_to_number()` (`visualization.py:59`) so they serialize
cleanly to JSON.

## 8. Config / thresholds

No environment variables or settings control this module. Thresholds are
hardcoded constants at the top of `visualization.py`:

```python
_MAX_PIE_SLICES = 6
_TOP_N_CATEGORIES = 9
_TEMPORAL_NAME_HINTS = ("date", "month", "year", "week", "day", "time", "period", "quarter")
_DATE_RE = re.compile(r"^\d{4}-\d{2}(-\d{2})?")
```

## 9. Known remaining gaps (not yet fixed)

- **Positional, not semantic, numeric-column pairing** in the combo chart
  (`_combo`) — the first two numeric columns are used regardless of whether
  one is actually an ID/count rather than a meaningful metric.
- **Single dimension only** — if a result has two categorical columns (e.g.
  `region` and `product_category`), only the first is used; no attempt to
  pick the more meaningful one or chart both.
- **Name-hint temporal misclassification** — a column named `day` holding
  day-of-month integers (not dates) is misclassified as temporal, since name
  hints are checked before value inspection.
- **Upstream ordering/grouping is still not guaranteed** — the fixes in §6
  make the recommender defensive, but the deeper fix (making
  `veda_core`'s SQL generators reliably emit `ORDER BY`/`GROUP BY`) has not
  been made.
