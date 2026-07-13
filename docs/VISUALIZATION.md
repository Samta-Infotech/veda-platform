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

There is **no dedicated automated test suite** for this module today.

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
  → _build_visualizations(res0)                     [services.py:227-231]
      → _visualization_recommender.recommend(cols, rows)
      → returns 0+ VisualizationSpec objects
      → each .to_dict() is yielded as an SSE event:
          {"event": "visualization", "data": spec_dict}
  → apps/chat/views.py: both the JSON response builder and the SSE
    generator append "visualization" events into the response[] array
    identically to "content" blocks
  → frontend renders bar / line / pie / line_histogram from the JSON spec
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
| 1 | dimension + **≥2 numeric** columns | `line_histogram`: combo chart using the dimension + the **first two** numeric columns (bars = first numeric, line = second numeric) |
| 2 | **temporal** dimension + exactly 1 numeric used, >1 row | `line` |
| 3 | **categorical** dimension + 1 numeric | `pie` if ≤6 distinct categories after aggregation, else top-9-by-value + "Other" bucket → `bar` |
| — | none match (e.g. all-text columns, single scalar, no dimension) | no chart — table remains the only output |

Priority 1 (combo chart) wins even over a temporal dimension, if a second
numeric column is present — it's considered more informative than either
metric charted alone.

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
- `len(pairs) <= 6` → `pie`, one slice per category.
- `len(pairs) > 6` → rank descending by value, keep top 9, collapse the rest
  into a single `"Other"` slice (sum of their values), then:
  - if the bucketed result is still ≤6 slices → `pie` (in practice this
    branch is unreachable, since bucketing only ever runs when the original
    count was >6, and top-9 + "Other" is always ≥7 when rest is non-empty,
    or equal to the original >6 count when rest is empty)
  - otherwise → `bar`, using `labels`/`values` arrays.

## 7. Output contract (`VisualizationSpec.to_dict()`, `visualization.py:50`)

Every chart is serialized as JSON matching a fixed frontend contract, never
an image:

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
