# VEDA Platform

Production Django platform for the VEDA **NL → SQL** engine, migrated from the
`veda-poc` research repo per [`migration_plan.md`](migration_plan.md).

> **The one rule that governs everything (plan §0):** the runtime query flow does
> not change. `run_hybrid_query → router → {sql | rag | hybrid | nosql} → firewall
> → MultiResult` stays behaviourally identical. The migration only re-homes where
> the ingestion substrate is *stored* and how it is *loaded/served* at query time.

## Layout

```
veda-platform/
├── veda_core/          # PRESERVED VEDA library — moved VERBATIM from veda-poc
│   ├── veda/ query/ retrieval/ connectors/ graph/ semantic/ ingestion/
│   ├── schema/ utils/  config.py  veda_hybrid.py   (unchanged engine)
│   ├── __init__.py      # path shim: keeps the moved code's top-level imports working
│   ├── context.py       # NEW §4.1 — ambient (source, tenant) contextvar
│   └── slm/_call_slm.py # NEW §8b — SLM Strategy seam (Ollama · vLLM)
├── apps/               # Deep Django — one app per bounded context (§5)
│   ├── core/           #   tenancy base models, settings bridge, health
│   ├── sources/        #   Source registry + connection profiles
│   ├── substrate/      #   ALL ingestion outputs as models (§6)
│   ├── ingestion/      #   Celery L0 pipeline + job/stage tracking (§7)
│   ├── query/          #   DRF QueryView, InferenceClient, QueryLog audit
│   └── evaluation/     #   eval runs + results + report artifact
├── inference/          # ASGI query/retrieval service — the warm get_engine() (§8)
├── storage_adapters/   # Repository/Adapter seam: substrate I/O ↔ ORM + raw pgvector (§3)
├── config/             # Django project (split settings, celery, urls, asgi/wsgi)
├── docker/             # Dockerfiles, compose, nginx, pgbouncer, entrypoints
├── docker-compose.yml  # dev topology
└── docker-compose.prod.yml
```

## Migration status (by plan phase)

| Phase | Scope | Status |
|---|---|---|
| **0 — Scaffold & preserve** | repo skeleton; move VEDA lib verbatim; import shim | ✅ **done & verified** — `from veda_core.veda_hybrid import run_hybrid_query` imports with zero edits |
| **1 — Infra & containers** | Dockerfiles, compose (split Redis, PgBouncer, SLM backends), nginx | 🟨 **scaffolded** — files written & YAML-valid; `docker compose up` needs a Docker host |
| **2 — Substrate models** | §6 models + pgvector tables | 🟨 **models done & verified** — `manage.py check` 0 issues, all 6 apps generate valid migrations. pgvector RunSQL migrations + HNSW indexes are TODO |
| **3 — storage_adapters seam** | reader/writer/assembler, context, SLM seam | 🟨 **interfaces scaffolded** — frozen signatures in place; bodies are `NotImplementedError` awaiting Phase 3 wiring |
| **4 — Ingestion (Celery)** | 10-stage task chain | 🟨 **task skeleton** — `apps/ingestion/tasks.py` stages defined; logic hoist from `main.py` is Phase 4.0 |
| **5 — Inference service** | ASGI warm-load + endpoints | 🟨 **service skeleton** — `inference/` app, lifespan, 4 routes, context middleware, offload primitive |
| **6 — DRF API/auth/audit** | QueryView, tenancy, QueryLog | 🟨 **skeleton** — view/client/urls/models present; auth+wiring is Phase 6 |
| **7 — Parity & hardening** | golden-baseline diff, HNSW tuning, load/soak | ⬜ **not started** — requires running infra + the Phase 0.5 baseline |

Legend: ✅ complete & verified · 🟨 scaffolded (skeleton/interface, not runtime-complete) · ⬜ not started.

**What "scaffolded" means here:** every file exists, parses (`py_compile`), and the
Django project passes `manage.py check` with 0 issues. Phases needing Docker /
Postgres+pgvector / Redis / GPU cannot be *executed and gated* in a plain
workstation environment — their exit criteria (plan Part B) run against live infra.

## Preserved artifacts requiring a mounted volume

Two heavy binaries were copied verbatim but belong on a mounted volume per plan §9 / §6.5
(they are git-ignored):

- `veda_core/ingestion/client_bge/model.safetensors` (1.2 GB fine-tuned BGE checkpoint)
- `veda_core/schema/kuzu_graph` (26 MB persisted knowledge graph)

## Verify locally (no Postgres needed — sqlite fallback)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements/api.txt          # Django + DRF + Celery
export DJANGO_SETTINGS_MODULE=config.settings.dev
python manage.py check                        # → 0 issues
python manage.py makemigrations --dry-run     # → valid migrations for all apps

# preserved engine still imports the old way (needs the ML deps):
pip install -r requirements/inference.txt
python -c "from veda_core.veda_hybrid import run_hybrid_query; print('ok')"
```

See [`migration_plan.md`](migration_plan.md) for the full spec and phased runbook,
and [`ARCHITECTURE.md`](ARCHITECTURE.md) for the frozen query flow.
