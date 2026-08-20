# Multi-Source Routing — PM Task Tracker

> Status dashboard. One row per task. Status: TODO / IN-PROGRESS / DONE / BLOCKED.
> Append a row when a task completes (index, not a retelling). Mirror to repo-root `PM_LOG.md`.

| # | Phase | Task | Status | Notes |
|---|-------|------|--------|-------|
| 1.1 | 1 | Source model fields (domain_tags/description/description_generated/is_canonical) + migration 0005 | DONE | applied to dev DB; check clean |
| 1.2 | 1 | Admin exposes profile fields (manual entry) | DONE | grouped fieldset; description_generated read-only; no source-edit API so read-contract untouched |
| 1.3 | 1 | `source_profiler.py` post-ingest hook | TODO | reuse cross_source_graph hook point |
| 1.4 | 1 | Auto-generate description (manual wins) | TODO | |
| 1.5 | 1 | Embed generated_description | TODO | reuse biencoder |
| 2.1 | 2 | BUG: source_id on phase3 RetrievalResult | TODO | spine-blocker |
| 2.2 | 2 | BUG: stop source_id="" in graph adapter | TODO | spine-blocker |
| 2.3 | 2 | SourceEvidence + group-by-source | TODO | reuse partition_subgraph |
| 2.4 | 2 | Presence-tier calibration | TODO | hardest tuning |
| 2.5 | 2 | Source agent wrappers | TODO | {type: AgentClass} |
| 2.6 | 2 | Thread evidence into execute | TODO | no double-retrieval |
| 3.1 | 3 | source_coordinator.py | TODO | |
| 3.2 | 3 | Deterministic routing policy | TODO | |
| 3.3 | 3 | RoutingDecision + ClarificationRequired | TODO | |
| 3.4 | 3 | Bounded SLM | TODO | reuse _call_slm |
| 3.5 | 3 | Routing validation | TODO | |
| 3.6 | 3 | Wire into veda_hybrid seam | TODO | replaces classify() |
| 4.x | 4 | Cross-source orchestration + merge/conflict | TODO | net-new merge policy |
| 5.x | 5 | Reliability + partial-failure | TODO | per-source failure |
| 6.x | 6 | Testing + benchmarking + observability | TODO | before/after numbers |

## Milestones
- [ ] **M1 (MVP):** Phase 1–3 done, single-source evidence routing working behind flag
- [ ] **M2:** Phase 4–5, multi-source combine + reliability
- [ ] **M3:** Phase 6, benchmarked + observable, flags on in staging
