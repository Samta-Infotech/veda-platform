# veda_core/veda/ — the deterministic L1–L7 head + firewall

The correctness head. `run_query()` in `pipeline.py` is the orchestrator; every
other file is a stage, a gate, or a support module it calls. Full walkthrough:
[../../docs/QUERY_ENGINE.md](../../docs/QUERY_ENGINE.md).

## Orchestration

| File | Role |
|------|------|
| `pipeline.py` | `run_query()` — L1–L7 orchestrator + escalation ladder (fast path → deterministic planners → verified cache → full retrieval/anchor/plan/gen) then the firewall gate stack then execute then NL-back. `_done()` is the single exit funnel. ~2100 lines. |
| `__init__.py` | 6-line package doc. |

## Planning / routing

| File | Role |
|------|------|
| `planning.py` | L4b/L4c deterministic planners: `existence_mode` / `aggregate_mode` / `superlative_mode` / `grouped_mode` / `ratio_mode` (grammar classifiers), `try_multitable` (deterministic join plan), `build_from_entities`, `_plan_and_build`, `build_aggregate_sql`, `build_existence_sql`, `_junction_tables`, `_try_measure_by_entity`. ~1380 lines. |
| `routing.py` | L2/L3 table routing: `route_tables_semantic` (table-embedding cosine), `select_primary_table` (semantic ⊕ column ⊕ lexical), `vet_primary` (grain-hint / word-order / junction / IDF / value / typed-anchor re-ranks + single-table ambiguity gate → can return `{"clarify": …}`), `recommended_projection` (deterministic business SELECT list). |
| `generation.py` | L5 LLM SQL generation: `generate_sql` (single table), `generate_join_sql` (fills SELECT/WHERE in a fixed FROM/JOIN skeleton), `_deterministic_single_table_sql` (SLM-free builder for safe cases), `_resolve_display_column` (governed label-column resolver — one source of truth). |

## Firewall

| File | Role | Gate |
|------|------|------|
| `validation.py` | L6 core: `validate_and_parameterize` (AST read-only + table/column existence + ON-integrity + graph-guard hook + fan-out guard + parameterize), `value_grounding`, `qualifier_completeness`, `grouped_shape_ok`, `distinct_shape_ok`, `_gate_strip` (the query-language stoplist). | 1, 2, 3, 4, 13 |
| `intent_sql_alignment.py` | Generalized intent↔SQL referent guards (flag-gated, default-ON): `alignment_ok` (temporal + entity-anchor), `aggregate_presence_ok`, `filter_presence_ok`, `dimension_alignment`. Each catches a "silent-wrong" class. | 5, 6, 7, 8 |
| `ir_equivalence.py` | L6b+ — reject LLM SQL that introduced filters/joins/grouping/ordering/DISTINCT the query never licensed. **LLM-generated SQL only.** | 10 |
| `graph_guard.py` | Author-agnostic FK-graph guards used inside `validate_and_parameterize`: `verify_joins_against_graph`, `check_connectivity`, `fanout_parent_aliases`. | 13 |
| `semantic_validation.py` | `validate_analytical_semantics` — operator preserved / GROUP BY present / dimension not an identifier / joins FK-connected. **Advisory in Tier-1** (trace only); enforceable in Tier-2 behind `SEMANTIC_VALIDATION_ENFORCE` (off). | 11 |
| `canonical_intent_shadow.py` | `record_shadow` — OBSERVE-ONLY comparison of the fast path's preserved `QueryIntent` to the final SQL. `CANONICAL_INTENT_SHADOW_ENABLED` **default OFF**. Never rejects. | 9 |

## Execution / cache / RBAC

| File | Role |
|------|------|
| `execution.py` | L7: `execute_sql` (psycopg2 `readonly=True, autocommit=True`, `statement_timeout=30000`, `SET search_path`, `fetchmany(EXECUTION_RESULT_LIMIT)`), `_execute_duckdb` (parquet/datalake), `PARAM_MISMATCH_ERROR` classification. |
| `cache.py` | Verified-query cache: `verified_cache_lookup` (exact-hash → BGE-M3 cosine ≥ 0.85, hard-coded), `save_verified_query`. Routes through `storage_adapters.reader` when a request context is set, else the legacy JSON file store. |
| `rbac_filter.py` | **Platform seam — Gate 1.** `filter_retrieval_results` (candidate list), `narrow_allowed` (the ONE centralised allow-list choke point, right before `validate_and_parameterize`), `restricted_names` (feedback path), `filter_nosql_collections`, `filter_doc_chunks`. Pure identity when `ctx.allowed_resources is None`. |
| `runtime.py` | Shared per-process handles: `get_db_config` (source DB from the `Source` row / `VEDA_SOURCE_*`), `_pg`, `_internal_db_config` (the `veda_engine` store), `get_engine` (per-`(tenant, source-set)` engine, LRU-capped at `ENGINE_CACHE_MAX`), `_shared_searcher`, `_load_scoped_sm` / `_merge_scoped_sms`, `get_graph`, `warm_up`. |

## State / explainability

| File | Role |
|------|------|
| `execution_state.py` | `ExecutionState` dataclass — the Tier-1→Tier-2 handoff (temporal result, candidate fields w/ provenance, primary table, rerank query, refusal reason, resolved entities). Stripped at the HTTP boundary. |
| `explain.py` | `ExplainTrace` — JSON "why" record. `new_trace` / `use_trace` / `current_trace` / `bind_trace`; `finish` (inner-stage checkpoint) vs `finalize` (owner end-of-query persist). Contextvar-scoped; `_NullTrace` when `EXPLAIN_TRACE_ENABLED=False`. |
| `business_explain.py` | LLM-free end-user explainability: `extract_sql_facts` (parse final SQL → entities/filters/aggregations/…), `build_explain`, `build_refusal_explain`, `_business_table_name` / `_business_field_name`. `f(final SQL, sm, checks)` — never retrieval internals. |
| `result_analyzer.py` | Deterministic Result Analyzer: `analyze_result` → `InsightContext` (column kinds, result shape, business patterns, chart candidates, grounding). `analytics_summary` = the JSON payload attached to every answered result. No LLM. |
| `feedback.py` | `explain_failure(status, sm, …)` → `{why, what_needed, suggestions, text}` using real schema values. Distinct `access_denied` wording for RBAC-restricted names. Optional gated LLM rephrase (`FEEDBACK_LLM_POLISH`, off). |

## Dormant / dead (flag-OFF — do not assume these run)

| File | State |
|------|-------|
| `understanding/` (`__init__`, `orchestrator`, `extractor`, `grounding`, `schema`) | `QUERY_UNDERSTANDING_ENABLED = False`. Complete extract → ground → firewall layer, dormant. |
| `analytical_spec.py` | `ANALYTICAL_SQL_V2 = False`. `derive_spec` + `emit_sql` unused. |
| `query_enhancement.py` | `QUERY_ENHANCEMENT_ENABLED = False`. `enhance_query` typo/plural/synonym/alias machinery never consulted on the hot path. |
| `routing_slm.py` | **Empty file (0 bytes).** |
| `canonical_intent_shadow.py` | observe-only (see firewall table). |

## Gotchas

- **L4 intent is the literal string `"SIMPLE"`** (`pipeline.py:345`). The
  `query_engine/intent_detector.py` classifier is deliberately not called. Query
  shape comes only from the grammar classifiers in `planning.py`.
- **`fetch ≤ 20` is wrong.** `EXECUTION_RESULT_LIMIT = 1000`. The `[L7]` log line,
  `_print_rows`, and `truncated = len(rows) >= 20` all still say 20.
- **The firewall runs on the ORIGINAL query** — `pipeline.py:1563` is a literal
  `assert`. Enhancement only reaches retrieval + rerank text.
- **Half the firewall gates run on `_llm_sql` SQL only** (gates 3, 4, 10);
  deterministic-planner SQL (existence / aggregate CTEs) skips them.
- **Qualifier salvage retries `run_query` once** with a forced `anchor_hint` —
  the head is not single-pass.
- `filter_presence_ok` reads a config symbol (`INTENT_SQL_FILTER_PRESENCE_ENABLED`)
  that **does not exist** in `config.py`; `getattr(…, True)` makes it default-ON.
