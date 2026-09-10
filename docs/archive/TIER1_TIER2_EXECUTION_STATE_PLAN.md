# Tier1 → Tier2 ExecutionState Propagation — Implementation Plan

Status: **awaiting approval** — no code changes have been made yet.

## Goal

Eliminate unnecessary recomputation when Tier1 (`veda/pipeline.py::run_query`) refuses
and falls back to Tier2 (`veda_hybrid.py::_tier2_sql`), by passing forward the work
Tier1 already did (temporal parsing, query understanding, candidate tables/columns,
primary table, refusal reason) instead of Tier2 starting cold. Incremental,
backward-compatible — not a merge of the two tiers, no architecture redesign.

## Key design finding (not in the original spec — needs confirmation)

Tier1 and Tier2 use **two structurally different retrieval result types** with
different field names:

- **Tier1** (`veda/pipeline.py:247`, `get_engine(sm).retrieve(...)`) →
  `retrieval.retrieval_engine_phase3.RetrievalResult` — fields `col_id` (already a
  `"table.column"` string, e.g. `"users.email"`), `column_name`, `table_name`,
  `final_score`.
- **Tier2** (`query/retrieval_select.py::select_retrieval`) →
  `ingestion.vector_store.RetrievalResult` — fields `col_id` (a **UUID** from the
  `BIENCODER_COL_TABLE` pgvector store), `col_name`, `table_id` (UUID), `table_name`,
  `semantic_type`, `similarity`.

Tier1's `col_id` and Tier2's `col_id` are different namespaces — one system's objects
cannot be passed directly into the other (the original spec already anticipated this:
*"Do NOT wrap Tier1 results into fake SelectedRetrieval objects... Use explicit seed
candidates instead"*). The one thing both sides agree on is the plain `"table.column"`
string format already used everywhere as dict keys (`all_cols`).

**Proposal**: `ExecutionState.candidate_columns` / `candidate_tables` hold plain
strings in that format. Tier2 does one exact-match DB lookup to resolve them to its
own `RetrievalResult` shape (new helper, below) — tier-specific objects never cross
the boundary.

## Files to modify/create

### 1. `veda_core/veda/execution_state.py` (new)

```python
@dataclass
class ExecutionState:
    original_query:      str
    normalized_query:    str = None
    temporal_result:      object = None   # TemporalParseResult from run_temporal_parser
    query_understanding:  dict = field(default_factory=dict)  # intent, existence, aggregation
    candidate_tables:     list = field(default_factory=list)  # ["users", "orders"]
    candidate_columns:    list = field(default_factory=list)  # ["users.email", "orders.total"]
    primary_table:        str = None
    routing_metadata:     dict = field(default_factory=dict)  # source, stats
    sql_planning:         dict = field(default_factory=dict)
    validation_metadata:  dict = field(default_factory=dict)
    trace:                dict = None     # tr.to_dict()
    refusal_reason:       str = None
    warnings:             list = field(default_factory=list)
```

Internal only — never serialized to the API response.

### 2. `veda_core/veda/pipeline.py`

Populate `ExecutionState` progressively as `run_query` executes:

- After `_tp_result = run_temporal_parser(query)` (line 165) →
  `es.temporal_result = _tp_result`
- At the existing `tr.set("query_understanding", ...)` call (line 190) → mirror the
  same values into `es.query_understanding`
- After `results = get_engine(sm).retrieve(...)` (line 247) →
  `es.candidate_columns = [r.col_id for r in results]` (already `"table.column"`
  strings), `es.candidate_tables = list({r.table_name for r in results})`
- After `primary = vet_primary(...)` (line 335) → `es.primary_table = primary`
- In `_done()` (line 128), on any non-`"answered"` status → `es.refusal_reason =
  kw.get("msg") or kw.get("error") or kw.get("missing")`
- In `_done()`'s return dict (line 144-145) → add `"context": es` alongside the
  existing keys. Additive by construction — every existing caller reads specific
  keys (`status`, `ok`, `cols`, `rows`, ...) and ignores the rest; verified by reading
  `_dispatch_single`'s and `_tier2_finish`'s consumption of this dict.

### 3. `veda_core/veda_hybrid.py`

- `_tier2_sql(query, sm, all_cols, verbose=False, execution_state=None)` (line 785) —
  new optional param, default `None` preserves current behavior exactly for any other
  caller.
- `_dispatch_single` (line 532) — change `t2 = _tier2_sql(query, sm, cols,
  verbose=verbose)` to pass `execution_state=res.get("context")` (Tier1's result dict
  is already in scope here as `res`).
- Inside `_tier2_sql`:
  - `tf = run_temporal_parser(query).temporal_filter` (line 800) → only recompute if
    `execution_state is None or execution_state.temporal_result is None`; else reuse
    `execution_state.temporal_result.temporal_filter`.
  - `sel = select_retrieval(query=query, intent="sql", verbose=verbose)` (line 801) →
    pass `seed_tables=execution_state.candidate_tables,
    seed_columns=execution_state.candidate_columns` when available.
  - Repair loop (line 849, `_repair_hint = None`) → seed with
    `execution_state.refusal_reason` via the existing `_repair_hint_for()` (line 677)
    when present, so attempt 0 already carries Tier1's failure reason instead of
    starting cold. Reuses the existing mechanism — no new retry framework.
  - Add a reuse-logging block (`"Tier1 completed. Reusing: ✓ Temporal ✓ ..."`) right
    before the retrieval call.

### 4. `veda_core/query/retrieval_select.py`

`select_retrieval(query, source_ids=None, intent="sql", graph_result=None,
seed_tables=None, seed_columns=None, verbose=False)`:

- When `seed_columns` provided: skip Step 1 (schema linker, line 62) and Step 2
  (bi-encoder+reranker, line 78) entirely — set `_v2_cols` directly from a new
  exact-lookup helper (below), `_source = "tier1_seed"`.
- Steps 3 (graph), the value-filter add-back (line 298), and join-path recompute
  (line 360) all run unchanged on top of the seeded columns — satisfying "still allow
  semantic refinement, routing, join-path pruning, validation."
- No change to behavior when `seed_columns` is `None` (default) — existing callers
  (`main.py`, evaluator) unaffected.

### 5. `veda_core/ingestion/vector_store.py`

New helper `get_columns_by_names(table_col_pairs: List[Tuple[str,str]]) ->
List[RetrievalResult]`, modeled on the existing `retrieve_cols_by_name_keywords`
(line 69) and `get_display_columns` (line 590) query patterns — exact
`WHERE (table_name, col_name) IN (...)` lookup against `BIENCODER_COL_TABLE`, both
pgvector and in-memory-fallback branches, returning real `col_id`/`table_id`. This is
the resolver for seed strings → Tier2's native shape.

### 6. Tests

`tests/test_execution_state_reuse.py` (new, following `test_tier2_answer.py`'s
pure-python/monkeypatch convention, no DB/network):

- `ExecutionState` round-trips through `run_query`'s `"context"` key without changing
  `status`/`ok`/`cols`/`rows` for an existing passing case (backward compat /
  identical API response).
- `_tier2_sql(..., execution_state=None)` behaves identically to today (regression
  guard).
- `_tier2_sql(..., execution_state=<populated>)` skips `run_temporal_parser` (mock
  call-count assertion) and calls `select_retrieval` with `seed_columns` set.
- `select_retrieval(..., seed_columns=[...])` skips schema-linker/bi-encoder (mock
  call-count assertion) and still returns FK-adjacent/value-filter-augmented columns.
- Repair-hint seeding: `_repair_hint_for` is invoked with Tier1's `refusal_reason`
  before attempt 0, not only after Tier2's own failures.

## What this does NOT touch

No changes to `run_slm_layer`, `run_sql_builder`, `emit_envelope`, SQL-gen prompts,
`plan_join_tree`/`build_from_entities`, the graph-guard firewall,
`validate_and_parameterize`, API/SSE response shapes, or any frontend code.
`graph_result` reuse is explicitly out of scope (it's RAG/hybrid document chunks,
unrelated to this SQL path). No contextvars, no new LLM/planner.

## Note on pre-existing uncommitted changes

`pipeline.py`, `veda_hybrid.py`, `generation.py`, `business_explain.py`, and
`planning.py` already have uncommitted edits from an earlier session (the "Result
Explanation Layer" work). This plan is grounded in the current on-disk versions of
those files — if they're reverted first, the logic/structure described here won't
change, but the cited line numbers will shift.

## Open question for approval

Is the `candidate_columns`-as-plain-strings design (vs. some richer shared type)
acceptable? This is the one real judgment call made beyond the original spec.
