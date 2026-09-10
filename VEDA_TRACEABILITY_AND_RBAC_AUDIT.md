# VEDA — Traceability, Explainability & RBAC Visibility Audit

**Date:** 2026-09-08
**Branch:** `feat/multisource-arch`
**Scope:** Inventory-only. No architecture proposed, nothing implemented, nothing refactored.
**Method:** Direct code inspection + one live trace record read from `veda_core/logs/explain_trace.jsonl`.

> Every claim below carries a `file:line`. Anything not verified in code is marked ❌ / "not tracked".

---

# PART A — QUERY LIFECYCLE TRACEABILITY

## A0. Capability Inventory

### A0.1 Query Understanding

| Capability | Code Location | Current Data | Available? | Persisted? | Safe for User UI? |
|---|---|---|---|---|---|
| Intent (grammar-derived) | `veda_core/veda/pipeline.py:344-375` | `"SIMPLE"` or `"AGGREGATE"` only | ✅ | ✅ trace | ⚠️ too coarse |
| Temporal range | `query/temporal_parser.py`; traced `pipeline.py:370` | `{start, end}` | ✅ | ✅ | ✅ |
| Existence mode | `pipeline.py:349,373` | semi/anti-join operator | ✅ | ✅ | ✅ |
| Aggregation spec | `pipeline.py:355,373` | `{threshold, op, top_n, ranked, direction}` | ✅ | ✅ | ✅ |
| Superlative / grouped / ratio | `pipeline.py:355,373-374` | dicts or None | ✅ | ✅ | ✅ |
| Typed intent contract | `veda/understanding/schema.py:33-93` | RawIntent → GroundedIntent → Refusal; intent, grain, measure, dimensions[], filters[], entities[], confidence, `schema_version=1` | ⚠️ flag-gated, degrades to `None` | ✅ trace `understanding` | ✅ |
| Grounding firewall decision | `understanding/orchestrator.py:36-85` | status ∈ extracted/grounded/degrade/refuse + `refuse_reason` + `unresolved[]` | ⚠️ flag-gated | ✅ | ✅ |
| Query rewrite (L0) | `veda_hybrid.py:1008-1015` | `original_query`, `effective_query`, `rewrite_reason` | ⚠️ `NL_SIMPLIFIER_ENABLED` off | ✅ | ✅ |
| Extracted filters (real) | `veda/business_explain.py:369-375` | `{field, operator, value}` from **final SQL AST** | ✅ | ✅ ChatMessage.metadata | ✅ |
| Ambiguity detection | `pipeline.py:838-875` | `entity_resolution.status` / `grounded_clarify` / `clarify_options` | ✅ | ✅ | ✅ |
| Metrics / dimensions | `veda/explain.py:526-535` | from `InsightContext` — **result-side, not question-side** | ✅ | ✅ | ✅ |

### A0.2 Routing

| Capability | Code Location | Current Data | Available? | Persisted? | Safe for User UI? |
|---|---|---|---|---|---|
| `RoutingDecision` | `query/routing_contracts.py:53-67` | 12 fields (status, mode, source_ids, candidate_sources, evidence_summary, decision_method, reason_code, reason, relationship_basis, canonical_basis, validation_status, query_id, trace_id) | ✅ | ⚠️ **only 5 traced** | partly |
| Traced routing fields | `veda_hybrid.py:637-641` | `status, mode, source_ids, reason_code, decision_method, shadow` | ✅ live-verified | ✅ | ✅ |
| Single vs multi | `routing_contracts.py:19-21` | SINGLE / MULTI / NONE | ✅ | ✅ | ✅ |
| Reason codes (stable enum) | `routing_contracts.py:29-35` | NO_EVIDENCE, SINGLE_CANDIDATE, RELATIONSHIP_EDGE, CANONICAL_SELECTED, AMBIGUOUS_SOURCE_SELECTION, SLM_RESOLVED, INVALID_SLM_DECISION | ✅ | ✅ | ✅ **best UI candidate** |
| Per-source evidence | `query/source_evidence.py:60-71` | presence_tier STRONG/WEAK/NONE, columns[], tables[], documents[], counts | ✅ computed | ❌ **not traced** | ⚠️ leaks schema |
| Routing confidence | `routing_contracts.py:43-47` | `top_score`, `top_item_score` — raw cosine | ✅ computed | ❌ never persisted | ❌ **uncalibrated** |
| Selected executor | `query/agents.py:179` → `AgentResult.engine` | `deterministic_sql` / `rag` / `nosql` | ✅ | ⚠️ in-memory | ⚠️ admin |
| Legacy route label | `query/fast_path.py:161-176` | `logs/route_log.jsonl`: `{t, route, latency_ms, query, table, rows, error}` | ✅ | ✅ separate file | ⚠️ table names |
| Permission pre-check refusal | `veda_hybrid.py:614-633` | "no permission" refusal | ⚠️ flag off | ❌ | ✅ |

### A0.3 Execution Plan

| Capability | Code Location | Current Data | Available? | Persisted? | Safe for User UI? |
|---|---|---|---|---|---|
| `ExecutionPlan` | `query/execution_planner.py:40-45` | `mode`, `strategy`, `steps[]`, `reason` | ✅ built | ❌ **never traced** | ✅ (`reason` is human copy) |
| `ExecutionStep` | `execution_planner.py:32-37` | `source_id, source_type, depends_on[], required` | ✅ | ❌ | ✅ |
| Dependency graph | `execution_planner.py:52-54` | `depends_on` **always empty** | ❌ | ❌ | — |
| DEPENDENT mode | `execution_planner.py:25` | declared; comment: *"deferred — never emitted yet"* | ❌ | ❌ | — |
| Parallel groups | `source_coordinator.py:848` | **sequential `for` loop** despite `mode=PARALLEL` | ❌ | ❌ | — |
| SQL plan shape | `pipeline.py:1061-1462` | `sql_planning.action` — 10 named shapes | ✅ | ✅ | ⚠️ jargon |
| Join plan | `pipeline.py:316-324` | `confidence, max_fanout, join_path[], unreachable[], ambiguous[], why[]` | ✅ | ✅ | ⚠️ admin |

### A0.4 Source Execution

| Capability | Code Location | Current Data | Available? | Persisted? | Safe for User UI? |
|---|---|---|---|---|---|
| `AgentResult` | `query/agents.py:31-41` | `source_id, source_type, status, engine, data{...}, provenance[], error, reason` | ✅ | ❌ **never traced** | partly |
| Source **name** | — | only numeric `source_id` (`"2"`); `Source.name` at `apps/sources/models.py:46` never joined back | ❌ | ❌ | ✅ once mapped |
| Operation performed | `AgentResult.engine` | deterministic_sql / rag / nosql | ✅ | ❌ | ⚠️ |
| Per-source status | `AgentResult.status` | ok / refused / failed | ✅ | ❌ | ✅ |
| Per-source timestamps | — | **none** | ❌ | ❌ | — |
| Per-source duration | `rag_layer.py:73,105` | RAG/Hybrid only; SQL agents none | ⚠️ partial | ❌ | ✅ |
| Records processed | `veda/explain.py:516-519` | `row_count`, `column_count`, `truncated` — whole query only | ✅ | ✅ | ✅ |
| Generated SQL | `pipeline.py:1846` | parameterized SQL + params | ✅ | ✅ trace + QueryLog + **frontend** | ⚠️ **exposed unconditionally** |
| Errors | `veda/execution.py:87-133` | raw DB error string | ✅ | ⚠️ into refusal | ❌ raw |
| Retries | `query/reliability.py:44-56` | transient/permanent class, bounded retry | ⚠️ flag off | ❌ **attempts not counted** | ⚠️ admin |
| Fallback chain | `veda_hybrid.py:900-928` | which tier answered → `SubResult.route` | ✅ | ✅ QueryLog.route | ⚠️ |
| DB-side exec time | `veda/execution.py:87` | **not measured** | ❌ | ❌ | — |

### A0.5 Cross-Source / Federation

| Capability | Code Location | Current Data | Available? | Persisted? | Safe for User UI? |
|---|---|---|---|---|---|
| Federated SQL validation | `query/federated_executor.py:73-103` | `{catalogs[], tables[]}`, SELECT-only AST gate | ✅ | ❌ | ⚠️ |
| Catalogs attached | `federated_executor.py:130,157` | `ATTACH … AS src_<id> (READ_ONLY)` | ✅ | ❌ | ⚠️ |
| Provenance array | `cross_source_composer.py:297-312` | `[{kind:"sql",catalog}, {kind:"evidence",doc,section}]` | ✅ | ❌ **discarded** | ✅ **best untapped asset** |
| Join / matching keys | — | only inside SQL text | ❌ | ❌ | — |
| Entity resolution | `query/entity_resolver.py`; `pipeline.py:838-845` | `status, anchor, confidence, candidates[]` | ✅ | ✅ | ⚠️ |
| Merge operations | `query/result_orchestrator.py:25-33` | `policy` (APPEND / CANONICAL_PRIORITY / CONFLICT_DETECTED), `parts[]`, `provenance[]`, `conflict`, `winner_source_id`, `needs_clarification` | ✅ | ❌ **never traced** | ✅ |
| Aggregation pushdown | `cross_source_composer.py:407` | per-metric plan dict | ✅ | ❌ | ⚠️ |
| Intermediate results | `federated_executor.py:164-211` | joined in DuckDB, not retained | ❌ | ❌ | — |
| Cross-source dependencies | `ExecutionStep.depends_on` | always `[]` | ❌ | ❌ | — |
| Federated refusal reasons | `cross_source_composer.py:379-400` | `not_federated`, `refused_rbac`, `refused_federated`, `exec_error_federated` | ✅ | ❌ flattened | ✅ sanitized |

### A0.6 Validation / Quality

| Capability | Code Location | Current Data | Available? | Persisted? | Safe for User UI? |
|---|---|---|---|---|---|
| Named check ledger | `veda/explain.py:110-114` | `[{name, status, detail}]` | ✅ live-verified | ✅ | ✅ |
| `value_grounding` | `veda/validation.py:513`; `pipeline.py:1552` | filter values exist in data | ✅ | ✅ | ✅ |
| `qualifier_completeness` | `validation.py:327`; `pipeline.py:1569` | nothing dropped | ✅ | ✅ | ✅ |
| `ir_equivalence` | `veda/ir_equivalence.py`; `pipeline.py:1777` | no unrequested filters/joins/grouping | ✅ | ✅ | ✅ |
| `ast_readonly_parameterized_fanout` | `validation.py:10`; `pipeline.py:1843` | read-only, parameterized, no fan-out | ✅ | ✅ | ✅ |
| Plain-language labels | `business_explain.py:35-41` | already mapped to user sentences | ✅ | ✅ shipped | ✅ |
| Repairs | `explain.py:116-118` | `{what, from, to}` | ✅ API | ⚠️ rarely called | ⚠️ admin |
| Semantic validation | `pipeline.py:1802-1804` | findings list | ✅ | ✅ | ⚠️ |
| Confidence | `pipeline.py:243-251` | weakest-link of anchor_conf + join_conf; **`1.0` when neither exists** | ✅ | ✅ shipped | ⚠️ **misleading** |
| Truncation | `explain.py:516-519` | `truncated: bool` | ✅ | ✅ | ✅ |
| Partial results | `source_coordinator.py:863-867` | `{failures[], any_required_failed, ok_count, complete}` | ✅ | ❌ **discarded** | ✅ |
| Source conflict | `result_orchestrator.py:31,69-71` | `conflict{values:[{source_id,value}]}`, `needs_clarification` | ✅ | ❌ | ✅ |
| Unmatched records | — | **not tracked** | ❌ | ❌ | — |
| Warnings channel | — | **no warning primitive exists** | ❌ | ❌ | — |

### A0.7 Final Response

| Capability | Code Location | Current Data | Available? | Persisted? | Safe for User UI? |
|---|---|---|---|---|---|
| End-user explain payload | `veda/business_explain.py:280-399` | `version, understanding{summary,breakdown}, data_used{datasets,fields}, operations[], filters{applied,summary}, validation{passed,checks}, sql{enabled,query}, confidence, timeline[], visualization{type,reason,fields}` | ✅ **shipping** | ✅ ChatMessage.metadata | ✅ LLM-free |
| Refusal payload | `business_explain.py:402-422` | `version, status, understanding.summary, why, what_would_help, suggestions[]` | ✅ | ✅ | ✅ |
| Timeline | `pipeline.py:175-190` | `[{phase, message}]` | ✅ | ✅ | ✅ |
| Business names | `business_explain.py:80-113` | table → `primary_entity` phrase; column → `business_role` | ✅ | ✅ | ✅ |
| Doc citations | `rag_layer.py:71` | `"doc_name (p.N)"` | ✅ RAG only | ⚠️ | ✅ |
| Source attribution | — | **none** — `datasets` are tables, not sources | ❌ | ❌ | — |
| Supporting records | `AgentResult.data.rows` / `RAGResult.chunks` | not surfaced as evidence | ⚠️ | ❌ | ✅ |
| Limitations / warnings | — | none | ❌ | ❌ | — |

### A0.8 Governance / Audit

| Capability | Code Location | Current Data | Available? | Persisted? | Safe for User UI? |
|---|---|---|---|---|---|
| Query / correlation ID | `apps/core/middleware.py:12-23` → `inference/routes/hybrid.py:40-49` → `explain.py:464-468` | one id: `X-Request-Id` = `trace_id` | ✅ **clean chain** | ✅ QueryLog + trace | ✅ as support ref |
| Separate execution ID | — | none | ❌ | — | — |
| Timestamps | `QueryLog.created_at`; `route_log.jsonl.t` | ✅ | ✅ | ✅ | ✅ |
| Tenant | `apps/query/views.py:187-192` | ✅ | ✅ | ❌ internal |
| User identity | — | **`QueryLog` has NO user FK**; tenant = `user.username` (`views.py:186-190`) — a proxy | ⚠️ | ⚠️ | ❌ |
| Source access | `QueryLog.source` FK | single source only; multi-source scope not recorded | ⚠️ | ✅ | ⚠️ |
| Authorization checks | `apps/access_management/gate.py:104,119` | `logger.warning` only — **no DB model** | ⚠️ | ❌ | ❌ |
| Data-scope filtering | `veda/rbac_filter.py:149-292`; `pipeline.py:644` | `rbac_filter.before/after` counts | ✅ | ✅ trace only | ❌ |
| Terminal status taxonomy | `apps/query/models.py:13-23` | 9 frozen statuses | ✅ | ✅ | ✅ grouped |
| Token usage | `slm/_call_slm.py:69-150`; `explain.py:211-223` | prompt/completion/total + `per_purpose` | ✅ | ✅ QueryLog + SSE | ⚠️ admin |
| Per-SLM-call ledger | `explain.py:120-136`; `_call_slm.py:392-398` | `[{purpose, model, duration_ms, ok, error}]` | ✅ choke-point | ✅ | ❌ internal |
| Model version | `pipeline.py:233-240` | model *tag* e.g. `qwen2.5:7b-instruct` | ✅ | ✅ | ❌ |
| Prompt/template version | — | **does not exist** | ❌ | ❌ | — |
| Executor/pipeline version | `mlflow_observability/mapper.py:289-291` | `VEDA_GIT_SHA` env, MLflow sidecar only | ❌ in trace | ❌ | — |
| Prometheus metrics | `apps/core/views.py:84-138` | queries_total by status, refusal_rate, per-route latency, cache hits | ✅ | ✅ | ❌ ops |

---

## A1. Current Query Lifecycle (actual implementation)

```
HTTP POST /api/v1/chat/…  |  /api/v1/query
  └─ apps.core.middleware.RequestIdMiddleware:18        → request.request_id (X-Request-Id)
  └─ apps.access_management.gate.RequiresPermission:67  → permission check (see Part B)
  └─ apps.query.scope.resolve_query_scope:85            → permitted source_ids (RBAC)
     apps.query.scope.permitted_source_ids:198
     apps.access_management.services.data_scope         → serialize_data_scope

  ── chat path ─────────────────────── ── direct query path ──────────
  chatbot.run.run_chat_turn:16                 apps.query.views.QueryView.post
    └─ LangGraph: classify_node → context_resolve_node → call_engine_node
       (chatbot/nodes.py:273,487,595 _emit)
    └─ apps.query.inference_client:68
       headers X-Source-Ids / X-Tenant / X-Request-Id / X-Data-Scope / X-Source-Profiles
                       ↓
inference/routes/hybrid.py:113  run_hybrid_query_route
  └─ _incoming_trace_id:40   (X-Request-Id → trace_id)
                       ↓
veda_core/veda_hybrid.run_hybrid_query:934
  └─ explain.new_trace / use_trace:423,489   ← THE ONE AMBIENT TRACE (ContextVar)
  └─ _run_hybrid_query_inner:968
     ├─ L0  query.nl_simplifier                       [NL_SIMPLIFIER_ENABLED, off]
     ├─ L0  query.runtime_context.answer_runtime_context
     ├─ ROUTING  _run_coordinator:574
     │     └─ query.source_coordinator.plan_route
     │          ├─ query.rag_layer._encode_rag_query  (embed once, ContextVar)
     │          ├─ query.source_evidence.group_evidence_by_source
     │          ├─ _attach_item_summaries  (pgvector source_item_embeddings)
     │          ├─ query.routing_policy.decide        → deterministic
     │          └─ query.routing_slm.resolve_boundary → bounded SLM tie-break
     │          ⇒ RoutingDecision → tr.set("routing", …)      [veda_hybrid.py:637]
     │     └─ execute_decision:815
     │          └─ execution_planner.plan_execution:48 → ExecutionPlan (NOT traced)
     │               ├─ single      → agents.resolve_agent → Agent.execute
     │               ├─ federated   → _federated_delegate
     │               └─ independent → per-step agent + reliability.execute_reliably
     │                                → result_orchestrator.merge_results:57
     ├─ FEDERATED  _maybe_federated → cross_source_composer.compose_federated:367
     │     ├─ resolve_pg_schema:202 / resolve_surface:134
     │     ├─ _federated_rbac_block:323                (fail-closed)
     │     ├─ federated_executor.FederatedExecutor:107
     │     │     validate_federated_sql:73 → ATTACH …READ_ONLY:130
     │     └─ build_provenance:297
     └─ SINGLE  _dispatch_single → veda.pipeline.run_query
          ├─ temporal_parser                          → tr "query_understanding"
          ├─ grammar: existence/agg/superlative/grouped/ratio  [pipeline.py:349-375]
          ├─ query.fast_path.try_fast_path → log_route:161 → logs/route_log.jsonl
          ├─ retrieval → rrf → graph_expansion        → tr
          ├─ veda.rbac_filter.filter_retrieval_results:185     → tr "rbac_filter"
          ├─ query.reranker (cross-encoder)           → tr "reranking"
          ├─ understanding.orchestrator:47 EXTRACT→GROUND      → tr "understanding"
          ├─ query.entity_resolver                    → tr "entity_resolution"
          ├─ schema_linking / entity_selection        → tr
          ├─ join_planner (Steiner) _rec_plan:313     → tr "join_planning"
          ├─ sql_planning (10 actions)                → tr "sql_planning"
          ├─ value_grounding:513 / qualifier_completeness:327 / ir_equivalence   → tr.check
          ├─ validate_and_parameterize:10 (AST)       → tr.check
          ├─ veda.execution.execute_sql:87  (psycopg2 READ ONLY, statement_timeout 30s)
          ├─ result_analyzer → InsightContext
          ├─ result_explainer (NL answer, SLM)        → tr "summary"
          └─ _done:220 → business_explain.build_explain:280

  every SLM call → slm._call_slm.call_slm:361 → tr.slm_call(purpose, model, ms, ok)
                       ↓
tr.finalize:271 → _persist:232
   → logger.info "query trace_id=… status=… total_ms=…"   (veda_core/logs/veda_pipeline.log)
   → veda_core/logs/explain_trace.jsonl                   [EXPLAIN_TRACE_PERSIST]
                       ↓
inference/routes/hybrid.py:76 _serialize  ── STRIPS {"context","trace","_debug"} ──
                       ↓
apps/query/views.py:201  QueryLog.objects.create(...)     [direct path ONLY]
apps/chat/services.py:349 _build_reply_events
   → SSE: thinking* → content* → visualization? → explainability → usage
   → apps/chat/turn_events.TurnEventAccumulator.metadata:68
   → ChatMessage.metadata = {thinking, explainability, usage}
```

---

## A2. Live Evidence

Last record of `veda_core/logs/explain_trace.jsonl` (23 MB), `trace_id=dc96dcb1951f`,
query *"How many sale negotiations are there?"*:

| Section | Recorded |
|---|---|
| `query_understanding` | intent SIMPLE, temporal None, existence None, aggregation dict |
| `routing` | `{status:ROUTED, mode:SINGLE, source_ids:['2'], reason_code:SINGLE_CANDIDATE, decision_method:deterministic, shadow:True}` |
| `schema_linking` | `selected_table: assets_salenegotiation` |
| `validation` | 4 checks, all `pass` |
| `execution` | `row_count:1, column_count:1, truncated:False, column_names:[…]` |
| `result_analysis` | `result_shape:SCALAR, measures:['id'], entities:[…]` |
| `summary` | `engine:run_nl_answer, model:qwen2.5:7b-instruct, success:True` |
| `slm` | `[{purpose:nl_answer, model:qwen2.5:7b-instruct, duration_ms:3130.7, ok:True}]` |
| `output` | full SQL, params, confidence 1.0, status answered |
| `totals` | `stage_durations_ms{...}, slm_call_count:1, slm_total_tokens:41718` |
| `explainability` | **`datasets:None, validation_passed:None`** ← defect, see A5 |

Persistence targets verified on disk:
`veda_core/logs/explain_trace.jsonl` (23 MB) · `veda_core/logs/route_log.jsonl` (625 KB) ·
`veda_core/logs/veda_pipeline.log` (568 KB, rotating 10 MB × 3 via `utils/logger.py:35-40`) ·
`logs/explain_trace.jsonl` (377 KB — second copy; `_TRACE_LOG` at `explain.py:25` is a **relative** path).

---

## A3. Existing Explainability Potential

**User already sees today** (on the wire, no backend change):
✓ Plain-English restatement of the question · ✓ datasets + fields (business names) · ✓ operations
(count/group/sort/limit) · ✓ filters (field, operator, value) · ✓ 4 validation badges · ✓ the exact
SQL · ✓ a confidence number · ✓ step timeline · ✓ chart type + reason · ✓ token usage + latency ·
✓ on refusal: why / what would help / suggestions · ✓ live progress ticks (30 mapped phases).

**In the backend today, needs only plumbing:**
✓ `trace_id` (returned by `/v1/run_hybrid_query`, **dropped by chat tier** — `grep -rn "trace_id" apps/` → 0 hits)
✓ routing source(s) + mode + `reason_code` · ✓ per-stage durations (`totals.stage_durations_ms`) ·
✓ row/column counts + `truncated` · ✓ federation `provenance[]` · ✓ cross-source conflict + merge policy ·
✓ per-source partial-failure list · ✓ RAG citations.

**✗ Not available at all:**
✗ human-readable source names · ✗ per-source status/duration/rowcount · ✗ join keys · ✗ any
warning/limitation channel · ✗ unmatched/dropped records · ✗ retry attempts + fallback chain ·
✗ prompt or pipeline version · ✗ user identity on the audit row.

---

## A4. Internal vs User-Safe Data

### A. SAFE TO SHOW
`understanding.summary` + `breakdown` · `data_used.datasets/fields` · `operations[]` ·
`filters.applied` + `summary` · `validation.checks` (already plain-English via `_CHECK_LABELS`) ·
`timeline[]` · `visualization.type/reason/fields` · refusal `why` / `what_would_help` / `suggestions` ·
RAG `citations[]` · `row_count` / `truncated` · turn `latency_ms` · `trace_id` (opaque support ref) ·
routing `mode` + `reason_code` (mapped to friendly copy) · `MergeResult.policy` + conflict presence ·
`partial.failures` as *"1 of 3 sources didn't respond"*.

### B. TECHNICAL / ADMIN VIEW ONLY
Final parameterized SQL + params — **note: currently shown to every user unconditionally,
`business_explain.py:377` hardcodes `"enabled": True`, there is no flag** · `sql_planning.action` ·
`join_planning.{join_path, confidence, max_fanout, unreachable, ambiguous}` · `entity_resolution.*` ·
`retrieval`/`rrf`/`reranking` sections · `stage_durations_ms` · token usage + `per_purpose` ·
`AgentResult.engine` · federated `catalogs[]` · `route_log.jsonl` route labels ·
`rbac_filter.before/after` counts.

### C. INTERNAL ONLY — DO NOT EXPOSE
`ExecutionState` (`veda/execution_state.py:5-10` says so explicitly; enforced at
`inference/routes/hybrid.py:63` `_INTERNAL_ONLY_KEYS`) · full `ExplainTrace` / `res0["trace"]` ·
`_debug` · **raw routing confidence / cosine `top_score`** (uncalibrated → false trust) ·
`CandidateSource.evidence_summary` (leaks table/column names of sources the user may not access) ·
raw DB/DuckDB error strings (`execution.py:130-131`) · SLM model tags + per-call ledger ·
verbose candidate lists / rejected paths (**`EXPLAIN_TRACE_VERBOSE=True` today** — captures column
names + row samples) · `pg_dsn` / `SourceSurface` credentials · `Source.password_env` /
`password_inline` / `connection_secret_ref` · `X-Data-Scope` header contents ·
any LLM prompt body or reasoning text.

> The architecture already respects the CoT boundary: `business_explain.py:5-15` states explain is
> `f(final validated SQL, semantic model, validation checks)` and **never** `f(retrieval/routing internals)`;
> chart reasons are template strings (`_CHART_REASON_TEMPLATES:52-58`), not SLM prose.
> **Keep that invariant.**

---

## A5. Missing Traceability

| # | Missing | Why useful | Where to capture | Needs |
|---|---|---|---|---|
| 1 | Source display name | Every explanation says `"2"` | `plan_route` (profiles already fetched at `apps/query/scope.source_profiles_for:173`) | metadata field |
| 2 | Per-source execution record | Biggest gap — multi-source has no per-leg accounting | `agents.BaseSourceAgent.execute` | new fields + trace section |
| 3 | `ExecutionPlan` in the trace | Fully built at `execution_planner.py:48`, thrown away | `execute_decision:830` | one `tr.set` |
| 4 | Federation join keys + match counts | First question of any cross-source answer | `federated_executor.execute` | result metadata |
| 5 | Merge/conflict persisted | `MergeResult` discarded at `execute_decision:864` | same | trace + explain field |
| 6 | Partial-failure surface | Computed `source_coordinator.py:865`, never leaves | same | trace + `warnings[]` |
| 7 | Warning / limitation channel | No primitive between "pass" and "refuse" | `build_explain` contract | new field + producers |
| 8 | DB-side execution duration | Stage `_ms` is a gap approximation | `veda/execution.py:87` | timer |
| 9 | Retry attempts / fallback chain | `execute_reliably` retries silently | `query/reliability.py` | counter |
| 10 | Prompt version + git SHA in trace | Can't attribute a metric change to a code/prompt change | `explain._stamp` | new fields |
| 11 | User FK on `QueryLog` | Audit is tenant-only, tenant = username proxy | `apps/query/models.py` | migration |
| 12 | Authorization-decision audit table | `gate.py:104` denials are log lines only | `gate.py` | new model |
| 13 | `trace_id` on chat response | Support can't correlate a complaint to a trace | `apps/chat/services` | pass-through |
| 14 | `QueryLog` for the chat path | Chat turns audited only as `ChatMessage` | `apps/chat/services` | persistence call |
| 15 | Question-side metrics/dimensions | Both read off the *result* → a wrong answer looks self-consistent | understanding layer | enable flag + trace |

### Two confirmed defects (facts, not designs)

1. **Trace `explainability` section reads the wrong keys.**
   `explain.py:566-576` and `pipeline.py:274-283` read `explain_payload.get("datasets")` and
   `.get("check_items")`, but `build_explain` returns them nested as `data_used.datasets` and
   `validation.checks`. Result: `datasets=None, validation_passed=None` in every trace —
   **reproduced in the live record above**.

2. **`MODE_PARALLEL` is a label with no runtime behaviour.**
   `execution_planner.py:24` declares it; `source_coordinator.py:848` executes steps in a plain
   sequential `for` loop.

---

## A6. Explainability Readiness Score

| Stage | Score | Basis |
|---|---|---|
| Query Understanding | **5/10** | Rich grammar signals traced + live-verified, but `intent` has only 2 values (`pipeline.py:344-359`); the good artifact (`GroundedIntent`) is flag-gated to `None`; metrics/dimensions are result-side. |
| Routing | **6/10** | Typed contract + stable `reason_code`, 5 fields live-verified. But 6 of 12 fields never traced, only numeric ids, and `plan_route` itself contains zero trace calls (all tracing at one call site, `veda_hybrid.py:637`). |
| Execution Plan | **3/10** | Well-traced for single-source SQL; the multi-source `ExecutionPlan` is built and discarded. `depends_on` always empty, DEPENDENT never emitted, PARALLEL runs sequentially. |
| Source Execution | **4/10** | Excellent SLM ledger at a real choke-point + whole-query counts. But **no per-source record at all**, no timestamps, no per-source duration, no retry count, `execute_sql:87` measures nothing. |
| Federation | **3/10** | `build_provenance`, `MergeResult`, `partial{}` are all real, well-designed — and **all three discarded in memory**. Zero federation data reaches the log or the UI. |
| Validation | **8/10** | Strongest stage: 4 named checks with detail, live-verified, already labelled for users, already shipping. Loses points for no warning tier, no unmatched-record tracking, and a `confidence` that returns `1.0` on the fast path (verified: `anchor_conf=None, join_conf=None, confidence=1.0`). |
| Result Evidence | **6/10** | `build_explain` is genuinely well-architected, deterministic, LLM-free, persisted and streamed. But no source attribution, no supporting records on the SQL path, no limitations. |
| Governance/Audit | **5/10** | Clean single correlation id end-to-end; `QueryLog` with a frozen 9-value taxonomy + tokens + latency; Prometheus. But no user FK, no authorization audit table, no `QueryLog` for chat, `trace_id` dropped before the client, no prompt/pipeline version. |

---

# PART B — RBAC: "KISKO PERMISSION HAI" — KYA DIKHA SAKTE HAIN

## B0. Short answer

**Haan — poora dikha sakte ho, aur backend already ready hai.** RBAC ek complete, well-modelled
subsystem hai jiske **read endpoints already exist and are already wired** into
`config/urls.py:25`. Koi naya resolver ya naya model banane ki zaroorat nahi — sirf UI banani hai.

**Current mode:** `.env:104` → `VEDA_RBAC_MODE=enforce` (default in code is `off`,
`config/settings/base.py:185`). Matlab gate **live** hai.

**Ek real gap hai:** ek normal user apni khud ki permissions nahi dekh sakta — har read endpoint
`IsAdminUser` + `user.manage` ke peeche hai. "Main kya access kar sakta hoon" wala self-service
endpoint **exist hi nahi karta**. Detail §B5 item 1.

---

## B1. The RBAC data model

| Model | File | Key fields |
|---|---|---|
| `Permission` | `models/permissions.py:51` | `code` (dotted `domain.action`, CI-unique), `name`, `is_active`. **Code-seeded via migration `0004_seed_permissions.py`, exposed read-only** — a runtime-invented permission would be one no gate ever checks (`permissions.py:12-19`). |
| `Role` | `models/roles.py:51` | `name` (CI-unique), `description`, `is_active`, `deleted_at`. The layer admins compose freely. |
| `UserRole` | `models/grants.py:59` | `user`, `role`, `granted_by`. Unique `(user, role)`. Indexed both ways — `userrole_user_idx` ("which roles does this user hold?"), `userrole_role_idx` ("who holds this role?"). |
| `RolePermission` | `models/grants.py:97` | `role`, `permission`, `resource_path`, `effect` (allow/deny), `granted_by`. Unique `(role, permission, resource_path)` — **`effect` deliberately NOT in the key**, so re-granting the opposite effect UPDATEs rather than creating a contradiction. Indexed on `resource_path` for the reverse "who can read this?" question. |
| `CatalogResource` | `models/catalog.py:63` | `path`, `parent_path`, `kind`, `source_id`, `is_active`, `substrate_id`. The noun side. |
| `UserProfile` | `models/profile.py:35` | profile + `deleted_at`. |

### The permission vocabulary (7 codes, `codes.py`)

```
query.execute · data.read · source.manage · ingestion.run
evaluation.run · user.manage · role.manage
```

### Resource path grammar (`resource_path.py`, ADR-0001)

```
<kind>:<source>[:<segment>]*

db:crm_postgres                    the whole source
db:crm_postgres:employee           one table
db:crm_postgres:employee:salary    one column
```

Kinds: `db` · `nosql` · `files` · `lake`, derived from `Source.dialect`
(`resource_path.py:57-79`). Segment-boundary matching — `db:crm` must **not** match
`db:crm_postgres` (`resource_path.py:24-27`, the classic prefix-authorization bug, explicitly avoided).

---

## B2. The resolution rules (this is what you'd actually render)

`services/resolver.py:11-27` — **four rules, both fail-closed:**

1. Collect every grant whose path is a prefix-or-equal of the requested resource.
2. If **ANY** is `deny` → **DENY** (unpierceable at any depth).
3. Else if the **SOURCE-level ancestor** (2-segment `db:<source>` prefix) is itself allowed → **ALLOW**.
4. Else → **DENY**.

**Strict hierarchy (product call, 2026-08):** the source is the gate. An ALLOW on a table/column
with no source-level ALLOW above it grants **nothing**. Model is *"allow the source, refine DOWN
with denies"*, not *"allow-list individual tables from nothing"* (`resolver.py:106-118`).

**A blank `resource_path` does NOT mean "everything".** It means "this permission is not
resource-scoped" (`user.manage`). A `data.read` check on `db:crm:employee` is **not** satisfied by a
blank-path grant (`resolver.py:41-47`).

**A grant only counts when the whole chain is live** — inactive user, role, or permission
contributes nothing. Filtered in SQL, not Python (`resolver.py:245-249`).

⚠️ **Three implementations of this same rule must stay in sync** (`resolver.py:24-27` says so explicitly):

| Implementation | File | Purpose |
|---|---|---|
| `EffectivePermissions.allows()` | `services/resolver.py:120-134` | the enforcement answer |
| `permitted_source_ids()` | `apps/query/scope.py:198` | coarse source gate |
| `CatalogService._resolve_effect()` | `services/catalog.py:479-500` | the admin tree overlay |

A live bug was already found here (2026-08): parent DENY + child ALLOW painted **green** in the tree
while the query was correctly denied (`catalog.py:481-484`). **If you build a permission-visibility
UI, this drift risk is the single biggest correctness hazard** — the tree must never lie about access.

---

## B3. What the API ALREADY exposes (all live, `config/urls.py:25`)

| Endpoint | View | Returns | Guard |
|---|---|---|---|
| `GET /api/v1/users/permissions/effective?user_id=&permission_code=&resource_path=` | `views/resolver.py:19` | **The headline endpoint.** `{user_id, username, is_active, permissions:[{permission_code, resource_path, effect}], permission_codes:[…]}` — plus, when `permission_code` given, a `decision` block: `{allowed, explicitly_denied, granted_on:[…]}` | `AdminView` + `USER_MANAGE` |
| `GET /api/v1/catalog/tree?role_id=&category=&parent_path=&search=` | `services/catalog.py:397` | Hierarchical resource tree grouped into `database` / `datalake` / `file_system` tabs. With `role_id`, **every node carries `effect` (ALLOW/DENY/null) and `is_allowed`** | `AdminView` |
| `GET /api/v1/catalog/list`, `/catalog/detail` | `catalog.py:336,364` | Flat paginated resource list / one resource | `AdminView` |
| `GET /api/v1/roles/permissions/list?role_id=&resource_path=` | `services/grants.py:285` | Grants, filterable **by role and by resource_path** → the reverse "which roles reach this table" question | `AdminView` + `ROLE_MANAGE` |
| `POST /api/v1/roles/permissions/grant` / `/revoke` | `grants.py:237,268` | Write side | `ROLE_MANAGE` |
| `GET /api/v1/users/roles/list` | `grants.py:200` | User↔role assignments, paginated + searchable | `USER_MANAGE` |
| `POST /api/v1/users/roles/assign` / `/revoke` | `grants.py:136,167` | Write side | `USER_MANAGE` |
| `GET /api/v1/roles/list` | `views/roles.py` + `grants.role_stats:341` | Roles **with `users_count` and `connected_sources`** (`["Database","Datalake",…]`), 2 queries for a whole page | `ROLE_MANAGE` |
| `GET /api/v1/roles/dropdown`, `/detail`, `+create/update/delete` | `views/roles.py` | Role CRUD | `ROLE_MANAGE` |
| `GET /api/v1/permissions/list`, `/dropdown`, `/detail` | `views/permissions.py` | The 7-code vocabulary, read-only | `AdminView` |
| `GET /api/v1/users/list`, `/detail` | `services/users.py:152,244` | `id, username, email, first_name, is_active, is_staff, date_joined, last_login` (`.only()` — password hash never fetched, `users.py:48-56`) | `USER_MANAGE` |

**Guard chain:** `views/base.py:AdminView` = DRF `IsAdminUser` **+** `gate.RequiresPermission`
(`gate.py:67`). Gate modes: `off` / `shadow` / `enforce` (`gate.py:46-48`). An unrecognised value
falls back to `off` with an error logged — deliberately, because guessing `enforce` would take a
deployment offline (`gate.py:56-59`). Resolver is called **at most once per request**, cached on the
request object, never a module global (`gate.py:135-143`).

---

## B4. What can safely be shown, and to whom

### ✅ SAFE — ADMIN / ACCESS-MANAGEMENT SCREEN
- Users list (username, email, active, last_login) and the roles each holds
- Roles list with `users_count` + `connected_sources` labels
- The 7-code permission vocabulary with human names
- Per-user effective permission set (`permissions[]` + `permission_codes[]`)
- The `decision` block: `allowed` / `explicitly_denied` / `granted_on[]` — **`explicitly_denied` is
  the valuable one**: "denied on purpose" vs "never granted" are very different things to fix
  (`resolver.py:136-142`)
- Catalog tree with per-node ALLOW / DENY / not-granted colouring for a chosen role
- `granted_by` + `created_at` on each grant (who gave this, when)

### ✅ SAFE — END-USER SELF-SERVICE (once an endpoint exists — see B5.1)
- *"You can access: Sales DB, Contracts (documents)"* — **source display names only**
- *"This question needs data you don't have access to"* + who to ask
- The user's own role names

### ⚠️ ADMIN-ONLY, NEVER END-USER
- Raw `resource_path` strings (`db:crm_postgres:employee:salary`) — these leak the **existence,
  naming and shape of schemas the user cannot read**. Show business labels instead.
- The full catalog tree of sources a user is denied
- `effect: DENY` rows for other users
- `is_staff`, `granted_by`, internal user ids
- `data_scope` payload contents (`X-Data-Scope` header) — enumerates exact reachable tables/columns

### ❌ NEVER EXPOSE
- `Source.password_env` / `password_inline` / `connection_secret_ref` (`apps/sources/models.py:58-62`)
- `pg_dsn` on `SourceSurface`
- Anything that confirms a restricted resource exists. `veda/feedback.py` already handles this
  correctly — a restricted table produces *"you don't have permission"*, never *"that doesn't
  exist"*, and **never names other resources as suggestions**
  (`apps/chat/services.py:322-333` documents this reasoning explicitly).

---

## B5. RBAC visibility gaps

| # | Gap | Evidence | Impact |
|---|---|---|---|
| 1 | **No self-service "what can I access?" endpoint** | `views/resolver.py:30` requires `USER_MANAGE`; `apps/authentication/urls.py` has only login/refresh/logout/password-change | A normal user cannot see their own access. Every refusal is a dead end — they don't know what they *do* have, or that the problem is permissions at all. **Biggest product gap in this area.** |
| 2 | **No "who can access this resource?" endpoint returning USERS** | `rolepermission_path_idx` (`grants.py:135`) exists *for exactly this question*; `list_grants(resource_path=…)` (`grants.py:288`) answers it at **role** level only — no role→user hop | An admin asking "who can read the salary column?" must query grants, then assignments, then join by hand |
| 3 | **`resources_for()` returns granted paths, not the closure** | `resolver.py:145-155` — a grant on `db:crm` returns `db:crm`, not every table beneath | UI can't show "these 47 tables" without separately walking the catalog |
| 4 | **Authorization decisions are not auditable** | `gate.py:104,119` — `logger.warning("gate: DENIED …")` and `"gate[shadow]: WOULD DENY …"`. No model, no table | Cannot answer "who was denied what, when" without grepping rotating log files |
| 5 | **`QueryLog` has no user FK** | `apps/query/models.py:26-53`; tenant is set from `user.username` at `views.py:186-190` | The query audit trail identifies a *tenant string*, not a *user* |
| 6 | **Chat path writes no `QueryLog`** | Only `apps/query/views.py:201` creates rows | Chat traffic — the actual product surface — is audited only as `ChatMessage` rows |
| 7 | **Computed data-scope is never surfaced or stored** | `services/data_scope.py:compute_data_scope` runs per request, forwarded as `X-Data-Scope`, then discarded | Cannot show "your answer was narrowed to N of M tables"; cannot debug an over-narrowed answer after the fact |
| 8 | **Engine-side RBAC narrowing is invisible to the user** | `pipeline.py:644` traces `rbac_filter.before/after` counts into the trace only; `veda/rbac_filter.py:185` silently drops candidates | A user can get a *quietly narrower* answer with no indication anything was filtered |
| 9 | **Federated RBAC is coarser than single-source** | `cross_source_composer.py:340-347` — matches **restricted table names only**; per-column narrowing within an allowed table is explicitly a follow-up | Cross-source answers enforce a weaker policy than single-source ones |
| 10 | **Three copies of the hierarchy rule** | `resolver.py:120-134`, `apps/query/scope.py:198`, `catalog.py:479-500` — the docstring at `resolver.py:24-27` mandates changing all three together | Already caused one live bug (tree showed ALLOW where query denied) |
| 11 | **No permission cache / version counter** | `resolver.py:50-55` — "Phase 7 caches the result… the cache key will need a `PermissionVersion`, which does not exist yet — noted rather than half-built" | One resolver query per request |

---

## B6. RBAC Readiness Score

| Dimension | Score | Basis |
|---|---|---|
| Model completeness | **9/10** | Users, roles, permissions, resource-scoped grants with allow/deny, catalog, `granted_by` provenance, CI-unique constraints enforced **in the database** not by check-then-insert. Only miss: no permission-version counter. |
| Rule clarity | **9/10** | Four rules, both fail-closed, documented at `resolver.py:11-27`, strict hierarchy is an explicit product decision. −1 for three copies that must move together. |
| Admin visibility (API) | **8/10** | Effective-permission endpoint with a `decision` block, catalog tree with per-node effect, grants filterable by resource, role stats. −2 for no role→user reverse hop and no path closure. |
| End-user visibility | **1/10** | Effectively zero. No self-service endpoint; a user cannot see their own access, and cannot tell a permission refusal from a data refusal. |
| Enforcement coverage | **7/10** | Source-level (`scope.py`), table/column-level (`data_scope` + `rbac_filter`), federated (`cross_source_composer`, fail-closed), gate on admin endpoints. −3: federated is table-granularity only, and `ROUTING_PERMISSION_PRECHECK_ENABLED` is off. |
| Auditability | **3/10** | Denials are log lines only, no audit model, no user FK on `QueryLog`, chat path unaudited. |
| Safety of refusals | **9/10** | No existence disclosure, no resource-name suggestions, fail-closed on evaluation error (`cross_source_composer.py:361`), inactive chain grants nothing. |

---

## B7. Bottom line on the question asked

> *"RBAC kisko permission hai woh sab dikha sakte hai?"*

**Admin ko — haan, aaj hi, bina backend change ke.** Teen endpoints kaafi hain:

1. `GET /api/v1/users/list` → user chuno
2. `GET /api/v1/users/permissions/effective?user_id=…` → uska poora effective set + `permission_codes`
3. `GET /api/v1/catalog/tree?role_id=…` → colour-coded resource tree (ALLOW / DENY / not granted)

Plus `roles/list` (`users_count` + `connected_sources` already computed) aur
`roles/permissions/list?resource_path=…` reverse lookup ke liye.

**End user ko — abhi nahi.** Har read endpoint `user.manage` ke peeche hai. Self-service
*"main kya dekh sakta hoon"* ke liye ek naya read-only endpoint chahiye jo apne hi `request.user` par
resolve kare aur **source display names** return kare, raw `resource_path` nahi. Resolver already
ready hai (`PermissionResolver.resolve(user)` → `as_dict()`), sirf ek view aur ek safe projection
missing hai.

**Sabse bada risk agar UI banate ho:** wahi drift jo pehle ek baar bug bana chuka hai — tree/UI ne
ALLOW dikhaya jabki query correctly deny hui. Jo bhi screen banao, wo `resolver.allows()` ke
**same** answer se render honi chahiye, apni copy se nahi.

---

# FINAL SUMMARY

## CURRENTLY AVAILABLE FOR EXPLAINABILITY
- Deterministic, LLM-free end-user explain payload (`build_explain`) — understanding summary,
  datasets, fields, operations, filters, validation checks, SQL, confidence, timeline, chart reason —
  already persisted in `ChatMessage.metadata` and streamed as the `explainability` SSE event
- Four named validation checks with pass/fail/detail, already mapped to plain-English guarantees
- Structured refusal payload (`why` / `what_would_help` / `suggestions`)
- One correlation id (`X-Request-Id` = `trace_id`) flowing browser → api → inference → engine → log
- Full 24-section per-query `ExplainTrace` in `explain_trace.jsonl` with per-stage durations + `totals`
- Complete per-SLM-call ledger (purpose, model, duration, ok/error) at a single choke-point
- `QueryLog` audit row on the direct-query path
- Routing status/mode/source_ids/reason_code/decision_method — verified live
- Live business-friendly progress stream (30 mapped phases, no internal names leaked)
- Prometheus `/metrics`; RAG citations on the document path
- **RBAC: a complete admin-facing permission API — effective permissions per user, a colour-coded
  catalog tree per role, grants filterable by resource, role stats**

## PARTIALLY AVAILABLE
- Routing: 5 of 12 `RoutingDecision` fields traced; candidates/evidence/relationship basis dropped
- Federation: `provenance[]`, `MergeResult`, `partial{}` fully built then discarded in memory
- Execution plan: rich for single-source SQL, absent for multi-source
- Typed `GroundedIntent` exists and is well-designed but flag-gated to `None`
- Confidence ships to the UI but is `1.0` whenever no anchor/join gating ran
- Duration: whole-query and per-stage yes; per-source and DB-side no
- Audit identity: tenant yes, user no; chat turns write no `QueryLog`
- `trace_id`: returned by the inference route, dropped by the chat tier
- RBAC enforcement: source + table/column granular single-source, table-only federated

## NOT CURRENTLY TRACEABLE
- Human-readable source names — the engine only ever sees numeric source ids
- Any per-source execution record (status, timestamps, duration, rows, error)
- Join keys, matched/unmatched counts, intermediate results in federated queries
- A warnings/limitations channel — only pass/fail and refuse exist
- Retry attempts and the fallback chain actually walked
- Prompt/template version, executor version, pipeline git SHA (MLflow sidecar env only)
- **Authorization decisions as queryable records** — denials are log lines only
- **A user's own permissions, to that user** — no self-service endpoint exists
- Real dependency graph / parallel execution groups
- Question-side metrics and dimensions

## TOP 5 EXISTING SIGNALS VEDA CAN SHOW TODAY
1. The four validation guarantees as pass/fail badges — already labelled, already on the wire
2. Plain-English restatement + operation breakdown + applied filters, all from the SQL that ran
3. Routing outcome: single vs multi, which source(s), stable `reason_code`
4. Step timeline + per-stage durations — both already computed
5. **Per-user effective permissions and the ALLOW/DENY catalog tree** — three existing endpoints

## TOP 5 TRACEABILITY GAPS
1. **No per-source execution record** — fatal for a multi-source product
2. **Federation evidence built and thrown away** — provenance, merge policy, detected conflicts, partial failures
3. **No source display names anywhere in the engine** — attribution is impossible
4. **No warning tier** — truncation, partial failures, low confidence, RBAC narrowing all invisible
5. **Governance thin at both ends** — no user FK, no chat `QueryLog`, no authorization audit table,
   no prompt/pipeline version, and **no way for a user to see their own access**
