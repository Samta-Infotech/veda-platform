# Archived documentation

Everything in this folder is a **historical record** — a plan whose work has shipped, a
point-in-time analysis, or a spec that has been superseded. Nothing here describes the
system as it currently behaves. It is kept for provenance (some engine code comments
still cite `migration_plan.md §N`) and to preserve the reasoning behind decisions.

For the current picture, start at [`../README.md`](../README.md).

| Archived doc | What it was | Superseded / continued by |
|---|---|---|
| `migration_plan.md` | The 94 KB Django-migration target spec + phased runbook (Phases 0–7). Migration is complete. | [`../ARCHITECTURE.md`](../ARCHITECTURE.md) |
| `CLEANUP_PLAN.md` | Dead-code inventory + layered-ingestion + query-latency precompute plan. Executed through P7. | [`../INGESTION.md`](../INGESTION.md), [`../QUERY_ENGINE.md`](../QUERY_ENGINE.md) |
| `RETRIEVAL_UPGRADE_PLAN.md` | WP0–WP9: BGE-M3 unification, learned-sparse, HNSW, PPR, weighted fusion, de-flag precompute. Shipped. | [`../RETRIEVAL.md`](../RETRIEVAL.md) |
| `CROSSSOURCE_GRAPH.md` | 6-phase documents-as-first-class + federated-query plan. Phases 1–2, 4–5 shipped. | [`../MULTI_SOURCE.md`](../MULTI_SOURCE.md), [`../SEMANTIC_ENTITY_BRIDGE.md`](../SEMANTIC_ENTITY_BRIDGE.md) |
| `MEMORY_ARCHITECTURE.md` | 53 KB design for the 3-tier chat memory subsystem. Core thesis shipped; several named components did not. | [`../CHAT.md`](../CHAT.md) §Memory |
| `TIER1_TIER2_EXECUTION_STATE_PLAN.md` | `ExecutionState` Tier-1→Tier-2 handoff plan. Shipped (`veda/execution_state.py`). | [`../QUERY_ENGINE.md`](../QUERY_ENGINE.md) §Tier-2 |
| `VEDA_Latency_Implementation_Plan.md` | F1–F6 / T1–T14 latency tasks. Largely done; residuals overlap later work. | [`../QUERY_ENGINE.md`](../QUERY_ENGINE.md) §Performance |
| `QUERY_PIPELINE_OPTIMIZATION.md` | Latency execution log (SLM→host Metal, reranker text cap, candidate cap). | [`../QUERY_ENGINE.md`](../QUERY_ENGINE.md) §Performance |
| `RETRIEVAL_DECISION_LAYER_AUDIT.md` | Read-only audit of the retrieval decision layer (signal scoring, rerank call sites). Accurate as an audit. | [`../RETRIEVAL.md`](../RETRIEVAL.md) |
| `MULTISOURCE_ARCH_REVIEW.md` | "Multi-Source Query Routing — Production Architecture Review & Plan". Built (`query/source_coordinator.py` et al.). | [`../MULTI_SOURCE.md`](../MULTI_SOURCE.md) |
| `ROUTING_FIX_PLAN.md` | PM task breakdown for the multi-agent-per-source coordinator. Same work as the review above. | [`../MULTI_SOURCE.md`](../MULTI_SOURCE.md) |
| `ARCHITECTURE_ROOT_CAUSE_PLAN.md` | RC-A…RC-E root-cause plan for the anchor/routing/latency failure class. Phases 0/A/B/C/D delivered. | [`../QUERY_ENGINE.md`](../QUERY_ENGINE.md) |
| `ANCHOR_ROUTING_FIX_PLAN.md` | Tactical "completed payments" mis-anchor fix. Superseded by the root-cause plan above. | `ARCHITECTURE_ROOT_CAUSE_PLAN.md` → [`../QUERY_ENGINE.md`](../QUERY_ENGINE.md) |
| `mlflow_impl.md` | Requirements spec for the MLflow observability platform. Implemented. | [`../OBSERVABILITY.md`](../OBSERVABILITY.md), `mlflow_observability/README.md` |
| `SPEC_poc_phase1.md` | The original `veda-poc` functional spec (POC Phase 1, L1–L4). Every specific is superseded. | [`../ARCHITECTURE.md`](../ARCHITECTURE.md), `veda_core/query/contracts/` |
| `MULTI_SOURCE_SERVING.md` | July design/execution log of how multi-source serving was built. | [`../operations` docs](../MULTI_SOURCE_DEPLOYMENT.md) + [`../MULTI_SOURCE.md`](../MULTI_SOURCE.md) |
| `api_contract_v1_omnibus.md` | The original omnibus "Admin + Chatbot RBAC" API contract. Split into the two living contracts. | `../../ACCESS_MANAGEMENT_API_CONTRACT.md`, `../../AUTH_API_CONTRACT.md` |
| `VEDA_SUMMARY_ANALYTICS_REPORT.md` | Point-in-time work report for the analytical-summary upgrade. Shipped. | [`../QUERY_ENGINE.md`](../QUERY_ENGINE.md) §Answer generation |
| `VEDA_VISUALIZATION_VALIDATION_REPORT.md` | Validation pass + FIX-1/2/3 on the viz pipeline. Shipped. | [`../VISUALIZATION.md`](../VISUALIZATION.md) |
| `HYBRID_PIPELINE_EXISTING.md` | An as-is snapshot of the hybrid (SQL-first → RRF fusion → single synthesis) path, Aug 2026. Accurate when written; the multi-source coordinator now fronts source selection. | [`../QUERY_ENGINE.md`](../QUERY_ENGINE.md), [`../ARCHITECTURE.md`](../ARCHITECTURE.md) §3.4 |

## Open items lifted out of these plans

Before archiving, residual engineering items were carried forward to
[`../backlog/`](../backlog/) so nothing is lost:

- `../backlog/query-engine-open-items.md` — from `ARCHITECTURE_ROOT_CAUSE_PLAN.md`,
  `ANCHOR_ROUTING_FIX_PLAN.md` §5, `VEDA_Latency_Implementation_Plan.md`.
- `../PRODUCTION_READINESS_PLAN.md` (still at `docs/`) — B1–B13, none applied yet; that
  is the live prod-hardening backlog, not an archive candidate.
