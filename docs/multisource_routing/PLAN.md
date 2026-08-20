# Multi-Source Query Routing — Implementation Plan

> Source of truth for WHAT we are building. Design rationale lives in
> `../../MULTISOURCE_ARCH_REVIEW.md`. Track progress in `PROGRESS.md`, status in
> `PM.md`, durable decisions in `MEMORY.md`.

## Goal
Replace keyword-based intent/source routing (`query_router.py`) with an evidence-grounded,
deterministic-first multi-source router that reuses existing pipelines. Coordinator + one thin
agent per source-type; SLM only for bounded ambiguity. Everything flag-gated, default OFF.

## Golden rules (do not violate)
- Authorization BEFORE retrieval — reuse `apps/query/scope.py`, never rebuild RBAC.
- ONE shared retrieval, group by `source_id` — never N independent per-source retrievals.
- Evidence flows into `execute()` — never retrieve twice.
- Deterministic first; SLM only when edge + canonical + tiers cannot decide; validate SLM output.
- Reuse Tier-1/2, RAG, `run_federated`, `federated_executor` — wrap, don't reimplement.
- Every change behind a `config.py` flag, default OFF; prod byte-identical until validated.

---

## Phase 1 — Source Profiling & Catalog
- [ ] 1.1 Add `domain_tags` (JSON), `description` (text), `is_canonical` (bool) to `Source` model + migration
- [ ] 1.2 Admin/API expose the fields (manual entry allowed)
- [ ] 1.3 `ingestion/source_profiler.py` — post-ingest hook (same point as `cross_source_graph.discover_and_persist`)
- [ ] 1.4 Auto-generate `description` from observed schema/chunks when blank; manual entry wins
- [ ] 1.5 Embed `generated_description` via existing biencoder (source-level prior)
- **Reuse:** `Source` model, biencoder, cross_source_graph hook point
- **Acceptance:** every ready source has grounded description + tags; manual override precedence; re-ingest refreshes

## Phase 2 — Shared Evidence + Source Agents (spine)
- [ ] 2.1 **BUG-FIX:** add `source_id` to phase3 `RetrievalResult` (`retrieval_engine_phase3.py:48-64`)
- [ ] 2.2 **BUG-FIX:** stop hardcoding `source_id=""` in `graph_retriever.py:543` + `retrieval_select.py:238,267`
- [ ] 2.3 `SourceEvidence` contract + group-by-source (reuse `cross_source_composer.partition_subgraph`)
- [ ] 2.4 Presence-tier calibration (STRONG/WEAK/NONE) per source-type
- [ ] 2.5 `query/agents/` — `{source_type: AgentClass}` map; each `execute(ctx, evidence)` wraps existing pipeline
- [ ] 2.6 Thread routing evidence into execute (no second retrieval)
- **Reuse:** `select_retrieval`, `retrieve_top_k_chunks`, `partition_subgraph`, Tier-1/2/RAG/DuckDB
- **Acceptance:** every candidate carries source_id; zero double-retrieval; per-type agent executes

## Phase 3 — Coordinator + Deterministic Routing + Bounded SLM
- [ ] 3.1 `query/source_coordinator.py` — orchestrate retrieval→group→tier→policy→dispatch
- [ ] 3.2 Deterministic routing policy (0/1/≥2 + edge + canonical → SINGLE/MULTI/NO_MATCH/AMBIGUOUS)
- [ ] 3.3 `RoutingDecision` + `ClarificationRequired` contracts
- [ ] 3.4 Bounded SLM (candidates + evidence only) — reuse `_call_slm` structured + circuit breaker
- [ ] 3.5 Routing validation (source exists / is candidate / authorized / active / mode valid)
- [ ] 3.6 Wire coordinator into `veda_hybrid._run_hybrid_query_inner` (replace `classify()` seam)
- **Reuse:** `_join_hints`/graph_edges, `_call_slm`, `ExplainTrace`, `scope.py`
- **Acceptance:** single/multi/no-match/ambiguous resolve correctly; SLM output always validated

## Phase 4 — Cross-Source Orchestration + Merge/Conflict
- [ ] 4.1 `query/execution_planner.py` (PARALLEL now; DEPENDENT stub/deferred)
- [ ] 4.2 MULTI → `run_federated`/`execute_plan` (reuse)
- [ ] 4.3 `query/result_orchestrator.py` — merge policies (APPEND/JOIN/CANONICAL_PRIORITY)
- [ ] 4.4 Conflict detection + unresolved-conflict → surface, don't blend
- [ ] 4.5 Cross-source grounding guard (extend `NL_SUMMARY_NUMERIC_GUARD`)
- **Reuse:** `run_federated`, `execute_plan`, `build_provenance`
- **Acceptance:** append/join/canonical work; conflicts surfaced not blended

## Phase 5 — Reliability & Partial Failure
- [ ] 5.1 Wrap `agent.execute` in Tier-2 repair-retry pattern (reuse `VALIDATION_*` config)
- [ ] 5.2 Transient-vs-permanent classifier (reuse `_is_param_mismatch`, circuit breaker)
- [ ] 5.3 Per-source partial-failure in MULTI (required vs optional)
- **Reuse:** Tier-2 loop, timeouts, circuit breaker
- **Acceptance:** transient retried, permanent not; partial flagged, never silent

## Phase 6 — Testing, Benchmarking, Observability
- [ ] 6.1 Add `routing`/`execution_plan`/`merge` sections to `_SECTIONS` + span types
- [ ] 6.2 Curated routing eval-set (single/multi/ambiguous/no-match/canonical)
- [ ] 6.3 Before/after accuracy + latency (p50/p95); no-double-retrieval assertion
- **Reuse:** `evaluation/` suites, `ExplainTrace`, mlflow mapper
- **Acceptance:** trace per decision; measurable accuracy gain; latency within budget

---

## MVP cut line
Phase 1–3 = MVP (single-source correctness + evidence routing). Ship behind flags, validate, then
Phase 4–5 for multi-source reliability, Phase 6 continuous.
