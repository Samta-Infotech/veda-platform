# Multi-Source Query Routing — Production Architecture Review & Plan
**Grounded in the actual VEDA codebase (verified file:line).** No code written yet — this is the pre-implementation review the brief asked for.

---

## 1. Current Architecture Analysis

Today a query flows: `veda_hybrid.run_hybrid_query` → mint `ExplainTrace` → runtime/federated shortcuts → **keyword `classify()`** (`query_router.py`) → intent branch (`sql`/`rag`/`hybrid`/`nosql`) → per-intent retrieval + execution. Source **scope** is resolved upstream in Django (`apps/query/scope.py`) from RBAC ∩ ready ∩ client-pin and arrives in the engine as `ctx.source_ids`. Cross-source (`len(source_ids) >= 2`) is already handled by `_maybe_federated` → `federated_route.run_federated`.

**The single biggest correctness weakness** is that source/intent selection is keyword-frequency counting (`query_router.py` `_SQL_KEYWORDS`/`_RAG_KEYWORDS`), not evidence-grounded. Everything else the brief wants (registry, RBAC, cross-source SQL, provenance, retry, tracing) already exists in some form.

---

## 2. Existing Component Reuse Map

| Target need | Exists as | File:line | Verdict |
|---|---|---|---|
| Source registry + readiness | `Source` model + `_ready_source_ids` | `apps/sources/models.py`, `apps/query/scope.py:254-260` | **REUSE** |
| Tenant/RBAC/scope (auth before retrieval) | `resolve_query_scope`, `permitted_source_ids` (fail-closed, typed exceptions) | `apps/query/scope.py:84-230` | **REUSE (authoritative)** |
| Source states | `SourceStatus` enum (REGISTERED/INGESTING/READY/FAILED) + `ready` | `apps/sources/models.py:36-40` | **REUSE, extend states** |
| Stores source_id-keyed | `column_embeddings_v2`, `doc_chunks` | `ingestion/biencoder.py:53,192-204`; `ingestion/chunk_embedder.py:158,175` | **REUSE** |
| Multi-source retrieval | `select_retrieval(source_ids: List)` | `query/retrieval_select.py:32,90,112` | **REUSE, fix source-tag gaps** |
| Bi-encoder path carries source_id | `retrieval_v2` populates `RetrievalResult.source_id`, filters `source_id IN` | `query/retrieval_v2.py:84-86,93,111` | **REUSE** |
| Doc-chunk evidence source-tagged | `retrieve_top_k_chunks` | `ingestion/chunk_embedder.py:127,529,605-616` | **REUSE** |
| Evidence grouped by source | `partition_subgraph`, `selected_source_ids` | `query/cross_source_composer.py:31,48-59` | **REUSE (fix upstream tag)** |
| Cross-source federation (SQL + plan) | `run_federated` (3-tier: structured plan → free-form → flat SQL) | `query/federated_route.py:556-696` | **REUSE** |
| Federated executor (DuckDB ATTACH) | `execute` (combined) + `execute_plan` (per-metric FULL JOIN) | `query/federated_executor.py:139-218` | **REUSE** |
| Relationship metadata | `cross_source_fk` edges + query-time `_join_hints`; intra-source `fk_to`/`discovered_fk` | `ingestion/cross_source_graph.py:197-353`; `federated_route.py:66-113`; `ingestion/graph_persist.py:388-419` | **REUSE** |
| Provenance | `build_provenance` (per-source SQL + per-doc evidence) | `query/cross_source_composer.py:119-134` | **REUSE** |
| Bounded retry/repair | Tier-2 hint-driven loop, config-gated, deadline guard | `veda_hybrid.py:1443-1600`; `config.py:1782-1783` | **REUSE pattern** |
| Timeouts | SLM per-call + DB `statement_timeout` + federated timeout | `veda/execution.py:112`; `federated_executor.py:148` | **REUSE** |
| SLM circuit breaker | `_slm_circuit_breaker()` | `slm/_call_slm.py:390` | **REUSE** |
| Observability trace + trace_id | `ExplainTrace`, `mint_trace_id`, ContextVar propagation, `.set/.slm_call/.finalize` | `veda/explain.py:74-490` | **REUSE, add sections** |
| Derived spans (MLflow) | `build_spans` | `mlflow_observability/mapper.py:227-256` | **REUSE, add span types** |
| SLM per-call ledger | `call_slm` → `.slm_call(purpose,model,ms,ok)` | `slm/_call_slm.py:396-399` | **REUSE** |

---

## 3. Gap Analysis (per target component)

| Component | Status |
|---|---|
| Source Profiling | **MISSING** (Source has connection fields only) |
| Source Catalog (business metadata) | **MISSING** (no description/domain fields) |
| Source Metadata Search | **EXISTS BUT NEEDS EXTENSION** (embeddings exist; no source-description embedding) |
| Shared Evidence Retrieval | **EXISTS BUT NEEDS EXTENSION** (`select_retrieval` multi-source; fix source-tag breaks) |
| SourceEvidence Contract | **MISSING** (no unified evidence object; `partition_subgraph` is closest) |
| Candidate Selection | **MISSING** (no presence-tier; keyword scoring today) |
| Query Coordinator | **MISSING** (only `query_router` keyword + intent branch) |
| Deterministic Routing (evidence-based) | **MISSING** (keyword-based only) |
| NO_MATCH | **PARTIAL** (Tier-1 refuse exists; not a routing-level first-class outcome) |
| Clarification | **PARTIAL** (Tier-1 `clarify` status exists; not at routing layer) |
| Bounded SLM (routing) | **MISSING** (SLM infra exists; not used for source routing) |
| Routing Validation | **MISSING** |
| RoutingDecision contract | **MISSING** |
| Execution Planner | **PARTIAL** (federation = single combined / independent-then-join; no planner object) |
| Parallel Execution | **EXISTS** (`execute_plan` independent metrics; federated combined) |
| Dependent Execution | **MISSING** (no staged A→B) |
| Source Agents | **MISSING** as objects (per-type pipelines exist to wrap: Tier-1/2, DuckDB, RAG) |
| Reliability Layer | **PARTIAL** (Tier-2 loop + federated 1-retry/tier; no unified layer) |
| Retry/Repair | **EXISTS** (Tier-2 hint loop) |
| Partial Failure Policy | **MISSING** (federated is all-or-nothing, single DuckDB conn) |
| Result Orchestration | **PARTIAL** (federation joins + provenance; no merge object) |
| Merge Policy (APPEND/JOIN/CANONICAL) | **MISSING** |
| Conflict Resolution | **MISSING** |
| Provenance | **EXISTS** (`build_provenance`) |
| Observability | **EXISTS BUT NEEDS EXTENSION** (add routing/execution/merge sections + span types) |
| Canonical/authoritative source | **MISSING** (zero in codebase) |

---

## 4. Validation of Key Architectural Decisions

**A. Shared evidence retrieval (one logical interface) — CORRECT, and the codebase proves it.**
`select_retrieval` already takes `List[source_ids]` and searches all at once (`retrieval_select.py:32`); stores are physically `source_id`-keyed; `cross_source_composer.partition_subgraph` already groups tabular columns by `source_id` and separates doc chunks. Building three independent `assess()` retrievals would **duplicate** this and produce **incomparable** scores. **Decision: one shared retrieval, group by source_id.** The only fixes needed are two source-tag breaks (§below).

**B. Source agents own execution, not routing retrieval — CORRECT.** Per-type pipelines already exist (Tier-1/2, DuckDB `execution.py`, RAG `rag_layer.py`). Agents should be thin `execute(context, evidence)` wrappers. Routing evidence must be **passed into** execute to avoid a second retrieval.

**C. SLM as bounded ambiguity resolver, not primary router — CORRECT.** With `cross_source_fk` edges + a new canonical flag + presence tiers, most multi-source decisions resolve deterministically. SLM only for no-edge/no-canonical/≥2-unrelated. SLM infra (`_call_slm`, structured output, circuit breaker) already exists.

**D. PARALLEL vs DEPENDENT — codebase has NO dependent execution.** `federated_executor` does single-combined or independent-metrics-then-FULL-JOIN — both effectively **parallel/combined**. **Decision: ship PARALLEL now (reuse `run_federated`/`execute_plan`); DEFER DEPENDENT** — it overlaps the already-known "compound dependent sub-query" gap and adds real complexity for a rare case.

**E. Reliability reuse.** Reuse the Tier-2 hint-driven repair loop pattern (`veda_hybrid.py:1443-1600`, config `config.py:1782-1783`) and federated's 1-retry-per-tier. **Net-new needed:** per-source partial-failure (federated is all-or-nothing today) and a transient-vs-permanent classifier (only `_is_param_mismatch` + circuit breaker exist).

**F. Merge/federation reuse + what's missing.** Reuse `run_federated` + `cross_source_composer` + `build_provenance` for JOIN-federation with provenance. **Net-new needed (all MISSING):** canonical-priority selection, same-metric conflict detection, unresolved-conflict handling. The composer is provenance-tagging only — no reconciliation stage exists.

**G. Observability integration.** Reuse `ExplainTrace` + `mint_trace_id` (ContextVar propagation already threads api→pipeline→SLM→execution). **Net-new needed:** add `routing`, `execution_plan`, `merge` section names to `_SECTIONS` (`explain.py:43`) and to `_STAGE_SPAN_TYPE` (`mapper.py:216`) so they appear as spans; add a per-source-execution sub-record.

---

## 5. Recommended Final Architecture (for THIS codebase)

```
INGESTION (existing) ──► + Source Profiler (new, post-ingest hook)
                              → writes domain_tags, generated_description, is_canonical to Source
                              → embeds generated_description (reuse biencoder)
                              (cross_source_fk edges already built here)
════════════════════════════════════════════════════════════════════
USER QUERY
  │ mint query_id/trace_id (REUSE mint_trace_id + ExplainTrace)
  ▼
resolve_query_scope()  ← REUSE (RBAC ∩ ready ∩ pin, fail-closed)   [auth BEFORE retrieval]
  ▼
Shared Evidence Retrieval  ← REUSE select_retrieval + retrieve_top_k_chunks (fix 2 source-tag breaks)
  ▼
Group evidence by source_id  ← REUSE cross_source_composer.partition_subgraph
  │  → per source: presence_tier STRONG/WEAK/NONE   [NEW, calibrated per source-type]
  ▼
Query Coordinator (NEW)  →  Deterministic Routing Policy (NEW):
     0 cand           → NO_MATCH → clarify/controlled
     1 cand           → SINGLE
     ≥2 + cross_source_fk edge   → MULTI   ← REUSE _join_hints / graph_edges
     ≥2, no edge, is_canonical set → SINGLE (canonical)   [NEW flag]
     ≥2, else         → AMBIGUOUS → Bounded SLM (candidates+evidence only) → Routing Validation (NEW)
                                    → valid ? execute : CLARIFICATION_REQUIRED
  ▼
RoutingDecision (NEW contract)  → Execution Planner (NEW, thin): PARALLEL only for now
  ▼
Source Agents (NEW thin wrappers) . execute(context, REUSED evidence):
     DatabaseAgent  → Tier-1/Tier-2 (REUSE)
     DataLakeAgent  → DuckDB exec (REUSE execution.py)
     FileSystemAgent→ RAG (REUSE rag_layer)
     [MULTI → run_federated / execute_plan  (REUSE)]
  ▼
Reliability Layer (extend): Tier-2 repair loop (REUSE) + per-source partial-failure (NEW)
  ▼
Result Orchestrator: JOIN+provenance (REUSE composer) + Merge/Conflict policy (NEW)
  ▼
FINAL RESPONSE  (trace_id/query_id propagated end-to-end — REUSE ExplainTrace)
```

**One-line thesis:** ~70% of this already exists; the genuinely new work is (1) source profile fields + profiler, (2) evidence→presence-tier + deterministic routing policy + coordinator, (3) source-agent wrappers with evidence reuse, (4) merge/conflict policy, (5) per-source partial-failure, (6) three new trace sections. Two small bug-fixes unlock the shared-retrieval spine.

---

## 6. Component / Module Design

- **Source Profiler** (`ingestion/source_profiler.py`, NEW) — post-ingest hook (same point `cross_source_graph.discover_and_persist` runs). Reads observed schema/chunks, produces `domain_tags` + `generated_description`, embeds the description via existing biencoder. Writes to `Source`.
- **Coordinator** (`query/source_coordinator.py`, NEW) — orchestrates: call shared retrieval → group by source → presence-tier → routing policy → optional SLM → validation → RoutingDecision → dispatch. No SQL/RAG logic inside.
- **Evidence adapter** (extend `retrieval_select.py` + `cross_source_composer.partition_subgraph`) — fix source_id tagging so ALL candidates carry source_id.
- **Source Agents** (`query/agents/`, NEW thin) — `{source_type: AgentClass}` map (mirror `_DIALECT_TO_ENGINE`), each `execute(ctx, evidence)` delegates to the existing pipeline.
- **Execution Planner** (`query/execution_planner.py`, NEW, minimal) — SINGLE → one agent; MULTI → `run_federated`/`execute_plan`. DEPENDENT stubbed/deferred.
- **Reliability** (extend) — wrap agent.execute with the Tier-2 retry pattern; add per-source try/except for MULTI.
- **Result Orchestrator** (`query/result_orchestrator.py`, NEW) — reuse `build_provenance`; add merge-policy resolution.

---

## 7. Data Contracts

```
SourceProfile          producer: Source Profiler / Source model      consumer: Coordinator, SLM
  source_id            identity                                        (registry)
  source_type          relational|datalake|document|nosql             agent resolution
  observed_metadata    tables/fields/topics actually seen             grounding
  domain_tags[]        business domain(s)                              ambiguity/canonical check
  generated_description grounded NL summary (+ embedding)             SLM context, source prior
  is_canonical         authoritative-for-domain flag  [NEW]           deterministic tie-break
  status/ready/version state + staleness                              scope filter

SourceEvidence         producer: shared retrieval (grouped)          consumer: Coordinator
  source_id, source_type
  evidence_items[]     {kind: column|chunk|exact, ref, name}          candidate build
  retrieval_method     bi-encoder|sparse|graph|chunk                  explainability
  internal_score       raw (NOT cross-source-comparable)              debug only
  presence_tier        STRONG|WEAK|NONE  [NEW, calibrated]            routing decision
  provenance           source refs                                    trace

CandidateSource        producer: Coordinator   consumer: routing policy / SLM
  source_id, presence_tier, evidence_summary, is_canonical, edges_to[]

RoutingDecision        producer: Coordinator/Validator   consumer: Execution Planner, trace
  status: ROUTED|NO_MATCH|CLARIFICATION_REQUIRED|FAILED
  mode: SINGLE|MULTI|NONE
  source_ids[], candidate_sources[], evidence_summary
  decision_method: deterministic|slm|clarification
  reason_code, relationship_basis, canonical_basis, validation_status
  query_id, trace_id

ClarificationRequired  producer: Coordinator   consumer: API/UI
  reason_code (AMBIGUOUS|NO_MATCH|INVALID_SLM|INSUFFICIENT_EVIDENCE|CONFLICT)
  candidate_sources[], question

ExecutionPlan / ExecutionStep   producer: Planner   consumer: agents/reliability
  mode: PARALLEL|DEPENDENT(deferred), steps[{source_id, depends_on[], required}]
  timeout_policy, retry_policy

AgentResult            producer: Source Agent   consumer: Orchestrator, trace
  source_id, source_type, status: ok|failed|partial
  data/rows/chunks, provenance, execution_metadata{latency,retries}, warnings[], error

FailureResult          producer: Reliability   consumer: Orchestrator
  source_id, failure_class: transient|permanent|timeout|auth, retryable, attempts, last_error

MergeResult            producer: Orchestrator   consumer: final response
  policy: APPEND|JOIN|CANONICAL_PRIORITY|CONFLICT_DETECTED|UNRESOLVED_CONFLICT
  merged_data, per_source_provenance[], conflict_report?
```

---

## 8. Routing Policy (deterministic-first)
```
candidates = sources whose presence_tier ∈ {STRONG, WEAK} (STRONG preferred; WEAK only if no STRONG)
0 candidates                          → NO_MATCH (clarify)
1 candidate                           → SINGLE
≥2 candidates:
   cross_source_fk edge among them?   → MULTI          (REUSE _join_hints)
   else same domain_tag & is_canonical set → SINGLE (canonical)
   else                               → AMBIGUOUS → bounded SLM (or clarify)
```
No keywords anywhere. Every branch is a computed fact (tier, edge, flag).

## 9. Execution Planning Policy
- SINGLE → that agent.execute(evidence).
- MULTI → PARALLEL via existing `run_federated` (structured-plan-preferred) / `execute_plan`. Independent by construction (no cross-metric join fan-out).
- DEPENDENT → **deferred** (documented, stub interface only). Do not build the autonomous graph now.

## 10. Reliability & Partial-Failure Policy
- Wrap `agent.execute` in the Tier-2 hint-driven retry (REUSE config `VALIDATION_*`). Retry only transient/timeout; never auth/validation/param-mismatch (REUSE `_is_param_mismatch`, circuit breaker).
- MULTI: NEW per-source try/except. If a **required** source fails → controlled failure; if **optional** → return partial result flagged in `AgentResult.status=partial` + provenance. Never silently present incomplete as complete.

## 11. Merge / Conflict Policy (net-new)
- APPEND: independent answers presented together (default multi-doc/DB).
- JOIN: related via `cross_source_fk` key (REUSE federated join).
- CANONICAL_PRIORITY: same metric, two sources, one `is_canonical` → canonical wins, other shown as provenance.
- CONFLICT_DETECTED / UNRESOLVED_CONFLICT: same metric+period+definition disagree, no canonical → surface conflict, request clarification. Never blend silently. Extend the `NL_SUMMARY_NUMERIC_GUARD` grounding check to cross-source claims.

## 12. Observability Design
- REUSE `mint_trace_id` + `ExplainTrace` (ContextVar already propagates end-to-end).
- ADD sections to `_SECTIONS` (`explain.py:43`): `routing`, `execution_plan`, `merge`; and span types in `_STAGE_SPAN_TYPE` (`mapper.py:216`).
- Record per stage: candidates, evidence_summary, decision_method, slm_used, mode, per-source latency/retries/failure_class, merge_policy, conflict_status, final outcome. SLM routing call flows through existing `.slm_call` ledger automatically.

---

## 13. Phased Implementation Plan

**Phase 1 — Source Profiling & Catalog**
Reuse: `Source` model, biencoder embedder, `cross_source_graph` post-ingest hook. New: `domain_tags`/`description`/`is_canonical` fields + migration; `source_profiler.py`. Integration: fire profiler at end of ingestion. Risk: stale profiles → tie to same re-scan point. Accept: every ready source has a grounded description + tags; manual overrides win. Tests: profiler grounding, manual-override precedence, re-ingest refresh.

**Phase 2 — Shared Evidence + Source Agents**
Reuse: `select_retrieval`, `retrieve_top_k_chunks`, `partition_subgraph`, per-type pipelines. New: fix source_id tagging (phase3 `RetrievalResult` + `_subgraph_to_retrieval_results:543`); `SourceEvidence`; presence-tier calibration; thin `agents/`. Integration: retrieval → group → evidence; agents wrap pipelines and accept evidence. Risk: double retrieval → thread evidence into execute. Accept: every candidate carries source_id; no second retrieval in execute. Tests: source-tag completeness, evidence-reuse, per-type agent execute.

**Phase 3 — Coordinator + Deterministic Routing + Bounded SLM**
Reuse: `_join_hints`/graph_edges, `_call_slm` structured + circuit breaker, `ExplainTrace`. New: `source_coordinator.py`, routing policy, `RoutingDecision`, routing validation, `ClarificationRequired`. Integration: replace `classify()` call site in `veda_hybrid`. Risk: SLM hallucination → validate against candidate IDs. Accept: single/multi/no-match/ambiguous all resolve correctly; SLM output always validated. Tests: all routing outcomes, invalid-SLM rejection, clarify path.

**Phase 4 — Cross-Source Orchestration + Merge/Conflict**
Reuse: `run_federated`, `execute_plan`, `build_provenance`. New: `result_orchestrator.py`, merge policies, conflict detection, canonical priority, cross-source grounding guard. Integration: MULTI → planner → federated → orchestrator. Risk: silent blend → conflict guard. Accept: APPEND/JOIN/CANONICAL work; conflicts surfaced not blended. Tests: each merge policy, conflict, unresolved-conflict.

**Phase 5 — Reliability & Partial Failure**
Reuse: Tier-2 repair loop + config, timeouts, circuit breaker. New: per-source partial-failure in MULTI; transient-vs-permanent classifier; required-vs-optional. Integration: wrap agent.execute. Risk: masking real failures → classify explicitly. Accept: transient retried, permanent not; partial flagged. Tests: transient/permanent/timeout, required-fail, optional-fail-partial.

**Phase 6 — Testing, Benchmarking, Observability**
Reuse: `evaluation/` suites, `ExplainTrace`, mlflow mapper. New: routing/execution/merge sections+spans; curated routing eval-set. Accept: before/after accuracy+latency; full trace per decision. Tests: §14.

All phases **flag-gated, default OFF** (project convention; reuse the `config.py` flag pattern like `VALIDATION_REPAIR_LOOP_ENABLED`).

---

## 14. Testing & Benchmark Plan
- **Routing:** single, multi(edge), ambiguous, no-match, insufficient-evidence, canonical-pick, relationship-pick.
- **Security:** unauthorized source (expect `SourceAccessDenied`), tenant isolation, inactive/not-ready source.
- **SLM:** invalid source id, non-candidate id, malformed output, timeout, still-ambiguous → all must reject/clarify, never execute.
- **Execution:** parallel multi, (dependent = deferred/xfail), duplicate-retrieval detection (assert single retrieval).
- **Reliability:** transient→retry, permanent→no-retry, required-fail→controlled, optional-fail→partial.
- **Merge:** append, join, canonical priority, conflicting values, unresolved conflict.
- **Observability:** trace_id/query_id propagation, per-source spans, retry spans, merge spans present.
- **Performance:** old-vs-new routing latency, retrieval overhead, no-double-retrieval, p50/p95, routing accuracy on the curated set.

---

## 15. Risks & Recommendations
1. **Source-tag breaks (phase3 `RetrievalResult` no source_id; graph adapter `source_id=""` at `retrieval_select`/`graph_retriever.py:543`)** — these silently drop candidates from source grouping. **Fix first** (Phase 2) or the whole spine is unreliable.
2. **Presence-tier calibration** is the hardest tuning task — column-cosine vs chunk-cosine vs exact need per-type floors. Budget iteration; don't compare raw cross-type scores.
3. **Merge/conflict + canonical are fully net-new** — highest design risk and highest hallucination risk (cross-source synthesis). Guard aggressively; prefer clarify over blend.
4. **Do NOT rebuild** `scope.py` (security), Tier-1/2/RAG, `federated_route`/executor — reuse. Rebuilding trades correctness for new bugs.
5. **Defer DEPENDENT execution** and the 12-field profile — over-engineering for 3 source-types.
6. **Recommendation:** ship Phase 1–3 behind flags as the MVP (single-source correctness + evidence routing), validate with the curated set, then Phase 4–5 for multi-source reliability. Enterprise-grade but incremental.
