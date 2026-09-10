# VEDA Platform documentation

VEDA turns a natural-language question about connected data sources into a grounded,
executed answer. On-premise, zero-egress. Django/DRF api + Celery ingestion + a warm
FastAPI inference tier wrapping the preserved `veda_core` engine.

## Start here

| Doc | What it covers |
|-----|----------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The two process tiers, the request flow, every answering head, the substrate, tenancy & security — the map of the whole system |
| [QUERY_ENGINE.md](QUERY_ENGINE.md) | The deterministic SQL head in depth: the escalation ladder, the ~10-gate firewall, the planners, Tier-1 → Tier-2 |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Stand the whole stack up on one VM |

## Subsystems

| Doc | What it covers |
|-----|----------------|
| [RETRIEVAL.md](RETRIEVAL.md) | The 6-signal retrieval spine, the BGE-M3 encoder, the reranker, the two graph-expansion mechanisms |
| [INGESTION.md](INGESTION.md) | The offline L1–L5 build and every artifact it produces |
| [MULTI_SOURCE.md](MULTI_SOURCE.md) | Routing a question to the right source(s); the shadow coordinator; cross-source federation |
| [RBAC.md](RBAC.md) | Users, roles, permissions, the resource catalog, `VEDA_RBAC_MODE`, Gate 1 / Gate 2, query-scope resolution |
| [CHAT.md](CHAT.md) | The conversational tier (`apps/chat` + `chatbot/` LangGraph) and its evidence-only memory |
| [VISUALIZATION.md](VISUALIZATION.md) | Turning a result table into a recommended chart |
| [OBSERVABILITY.md](OBSERVABILITY.md) | The per-query explain trace and the MLflow exporter |
| [EVALUATION.md](EVALUATION.md) | Golden sets, eval harnesses, CI gates, the benchmark archive |
| [DATA_SOURCES_EMBEDDINGS_AND_GRAPHS.md](DATA_SOURCES_EMBEDDINGS_AND_GRAPHS.md) | How a source becomes embeddings + a knowledge graph, and how query time reads them |
| [SEMANTIC_ENTITY_BRIDGE.md](SEMANTIC_ENTITY_BRIDGE.md) | The semantic bridge between unstructured documents and the structured semantic layer |

## Layer contracts (next to the code)

- **Query:** [`veda_core/query/contracts/`](../veda_core/query/contracts/README.md) — L1 temporal … L7 execution + the non-SQL heads
- **Ingestion:** [`veda_core/ingestion/layers/contracts/`](../veda_core/ingestion/layers/contracts/README.md) — L1 extract … L5 publish

Every significant package directory also carries an `AGENTS.md` that maps its files and
responsibilities.

## API contracts (frontend integration)

At the repo root:
[`AUTH_API_CONTRACT.md`](../AUTH_API_CONTRACT.md) ·
[`ACCESS_MANAGEMENT_API_CONTRACT.md`](../ACCESS_MANAGEMENT_API_CONTRACT.md) ·
[`CHAT_API_CONTRACT.md`](../CHAT_API_CONTRACT.md) ·
[`TOKEN_USAGE_API_CONTRACT.md`](../TOKEN_USAGE_API_CONTRACT.md)

## Operations

| Doc | What it covers |
|-----|----------------|
| [DEPLOYMENT.md](DEPLOYMENT.md) | Single-VM + Docker Compose deploy |
| [MULTI_SOURCE_DEPLOYMENT.md](MULTI_SOURCE_DEPLOYMENT.md) | Onboarding source #2 and beyond; the routing flags |
| [DEMO_QUERY_ONLY.md](DEMO_QUERY_ONLY.md) | The query-only demo box — ship a dump, no ingestion |
| [MAC3_APP_HOST_SETUP.md](MAC3_APP_HOST_SETUP.md) | The 3-Mac-mini LAN topology (SLM host / embed host / app host) |
| [OPERATIONS.md](OPERATIONS.md) | Backups, blue/green + engine-only rollback, scaling |
| [PRODUCTION_READINESS_PLAN.md](PRODUCTION_READINESS_PLAN.md) | B1–B13 prod-hardening backlog — **none applied yet** |
| [`INGESTION_GUIDE.md`](../INGESTION_GUIDE.md) | Operator runbook for adding documents / datalake files and re-ingesting |

## Decisions & backlog

- [adr/](adr/) — architecture decision records
- [backlog/](backlog/) — tracked open engineering items lifted from archived plans
- [archive/](archive/) — completed plans and point-in-time reports, kept for provenance
  (engine code still cites `migration_plan.md §N`)

## Living-doc conventions

- A doc under `docs/` (not `archive/`) is meant to track the code. If you change behavior
  a doc describes, update the doc in the same change.
- `RBAC_PROGRESS_LOG.md` (repo root) is the working log for auth/RBAC — kept current for
  status + decisions; historical findings move to `archive/`.
- Point-in-time analyses and completed plans go to `archive/` with a row in
  `archive/README.md` naming their living successor.
