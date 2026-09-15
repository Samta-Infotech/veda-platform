# VEDA Platform

VEDA turns a natural-language question about connected data sources into a **grounded,
executed answer**. It is SQL-first and correctness-first: a deterministic engine builds
the query, an LLM only ever fills in names it is handed, and a firewall of validation
gates refuses rather than guesses.

On-premise, **zero-egress**: every model runs locally; no source-DB schema, values, or
query text leaves the server.

```
        ┌───────────────────────────────────────────────────────────────┐
  user  │  Django / DRF api  ──HTTP──▶  FastAPI inference (warm engine)  │
  ────▶ │  · REST + chat/SSE          · veda_core.veda_hybrid            │
        │  · RBAC, tenancy, audit     · retrieval → SQL head → firewall  │
        │  Celery workers             · reads Postgres+pgvector, Redis   │
        │  · ingestion pipeline       · calls local SLM (Ollama / vLLM)  │
        └───────────────────────────────────────────────────────────────┘
```

## What's in the tree

| Path | What it is |
|------|------------|
| `veda_core/` | The preserved NL→SQL engine. `veda_hybrid.py` is the single front door; `veda/` is the deterministic L1–L7 SQL head + firewall; `query/` is routing, IR/SLM, and the RAG / hybrid / NoSQL heads; `retrieval/` is the 6-signal retrieval spine; `ingestion/` is the offline L1–L5 build; `graph/`, `semantic/`, `slm/`, `connectors/`, `schema/` support them. `context.py` carries the ambient `(source, tenant)`. |
| `apps/` | Ten Django apps: `core` (tenancy, health, settings bridge), `sources` (source registry + routing catalog), `substrate` (ingestion outputs as models), `ingestion` (Celery pipeline driver), `query` (REST `/api/v1/query` + audit), `chat` (conversational API + SSE), `authentication` (login / JWT), `access_management` (RBAC: users, roles, permissions, catalog), `evaluation` (tracked eval runs). |
| `chatbot/` | The LangGraph conversational supervisor that fronts the engine for `apps/chat` — classify, memory, follow-up resolution, one turn at a time. |
| `inference/` | The warm FastAPI service that hosts `run_hybrid_query`. One engine per worker, per `(source, tenant)` scope. |
| `storage_adapters/` | The substrate I/O seam: `reader` (Django-free query-time reads), `writer` (ORM ingestion-time writes), `assembler` (the normalized semantic model ⇄ Redis). |
| `config/` | Django project — split settings, Celery, URL roots. |
| `docker/`, `docker-compose*.yml` | Dev / prod / demo / observability topologies. |
| `mlflow_observability/` | A standalone process that tails the engine's explain-trace into MLflow. Never imported by the engine. |
| `evaluation/`, `scripts/` | Eval harnesses, golden sets, and operational scripts. |
| `docs/` | **Start at [`docs/README.md`](docs/README.md).** |

## Run it

```bash
# Whole stack, dev topology (Postgres+pgvector, PgBouncer, split Redis, redis-stack,
# Ollama, api, worker, beat, ingest-worker, inference, nginx):
docker compose up -d
# nginx is the only published port → http://localhost:8080

# Django checks without Postgres (sqlite fallback):
python -m venv .venv && . .venv/bin/activate
pip install -r requirements/api.txt
export DJANGO_SETTINGS_MODULE=config.settings.dev
python manage.py check
```

Onboarding a data source is a **data operation, not a code change**: register a `Source`
row with its connection, `POST /api/v1/admin/ingest {source_id}` (staff), and the query
path picks it up once ingestion succeeds. See
[`docs/operations`](docs/MULTI_SOURCE_DEPLOYMENT.md) and
[`INGESTION_GUIDE.md`](INGESTION_GUIDE.md).

## Local environment notes

Two settings live only in `.env` (gitignored) and matter — without them the engine asks
Ollama for a model it won't serve and intent detection drifts run-to-run:

```
SLM_MODEL_NAME=qwen2.5:7b-instruct
SLM_TEMPERATURE=0
```

Two Postgres databases are in play: `veda` (Django tables) and `veda_engine` (the
engine's pgvector store — `column_embeddings_v2`, `graph_node_embeddings`, `doc_chunks`,
…). Engine tables are reached through `VEDA_INTERNAL_*`, never Django's `connection`.
See [`CLAUDE.md`](CLAUDE.md) for the full list of environment gotchas.

## Documentation

| Doc | Covers |
|-----|--------|
| [`docs/INGESTION_AND_QUERY_PIPELINES.md`](docs/INGESTION_AND_QUERY_PIPELINES.md) | **One-read walkthrough of both pipelines end-to-end** — start here for the whole system |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The two process tiers, the request flow, every answering head, tenancy & security |
| [`docs/QUERY_ENGINE.md`](docs/QUERY_ENGINE.md) | The deterministic SQL head in depth: the escalation ladder, the firewall, Tier-1 → Tier-2 |
| [`docs/RETRIEVAL.md`](docs/RETRIEVAL.md) | The 6-signal retrieval spine, graph expansion, the reranker, the encoder |
| [`docs/INGESTION.md`](docs/INGESTION.md) | The offline L1–L5 build and every artifact it produces |
| [`docs/MULTI_SOURCE.md`](docs/MULTI_SOURCE.md) | Routing a question to the right source(s); cross-source federation |
| [`docs/RBAC.md`](docs/RBAC.md) | Users, roles, permissions, the resource catalog, enforcement modes, query-scope resolution |
| [`docs/CHAT.md`](docs/CHAT.md) | The conversational tier (`apps/chat` + `chatbot/`) and its structured memory |
| [`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md) | Per-query explain traces and the MLflow exporter |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Golden sets, eval harnesses, CI gates, the benchmark archive |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Stand the whole stack up on one VM |

Per-directory `AGENTS.md` files document each package's files and responsibilities.
Layer contracts live next to the code they govern
(`veda_core/query/contracts/`, `veda_core/ingestion/layers/contracts/`).
