# VEDA Platform — Query-Only Demo (ship a dump, no ingestion)

Run **only the query layer** on a demo box, loaded from a dump of data you ingested on your
own machine. No ingestion runs on the demo server — `worker`, `beat`, `ingest-worker`, and `vllm`
are never started.

The catch this guide solves: **a single `pg_dump` is not enough.** This stack keeps ingested
state in three places, and only two of them are in Postgres.

---

## What the query layer reads at runtime

| Store | Contents | Captured by `pg_dump`? |
|---|---|---|
| **`veda_engine` DB** (`VEDA_INTERNAL_DB`) | The embeddings/graphs the retrieval path reads: `column_embeddings_v2`, `table_embeddings_v2`, `column_sparse_v1`, `table_sparse_v1`, `fk_adjacency`, `table_metadata`, `column_values`, `doc_chunks`, `graph_nodes/edges/embeddings`, `source_registry` | ✅ **yes — required** |
| **`veda` DB** (Django substrate) | `sources_source` registry (**the engine reads this at query time** to resolve scope, source kind, per-source FK edges, glossary, verified-cache — see `storage_adapters/reader.py`), plus auth tokens + `QueryLog` | ✅ **yes — required, not optional** |
| **`veda_core/data/`** (~16 MB, git-ignored) | `veda_semantic_model.json`, glossary, synonyms, concept/relationship/unified graphs, bm25/hnsw/rerank indexes, **per-source parquet** (`data/4/tables/*.parquet`, `data/5/…`) | ❌ **no — plain files, must be tarred** |

Plus two things that **cannot be dumped** and are provisioned on the demo box:

- **Model weights** — BGE-M3 + `bge-reranker-v2-m3` (the `model_cache` Docker volume, offline HF cache).
- **An SLM backend** — the query path does routing / decompose / SQL-gen / NL-answer via `call_slm`, so
  the demo needs **Ollama** with the SLM pulled (`qwen2.5-coder:7b` by default). Bundled as the
  `ollama_models` volume, or pulled fresh on first boot.

> Without `veda_core/data/veda_semantic_model.json` present, the inference tier's `/readyz` reports
> *"semantic model not found"* and queries won't run.

---

## What must be running (services)

Traced end-to-end through `veda_hybrid.run_hybrid_query`. The query-only stack is:

| Service | Why the query pipeline needs it |
|---|---|
| **postgres** | Holds both `veda_engine` (embeddings/graphs) and `veda` (registry/FK/glossary/cache). Required. |
| **pgbouncer** | Both `VEDA_INTERNAL_HOST` and `PGBOUNCER_HOST` dial it. Required (or point both env vars straight at `postgres:5432`). |
| **redis-cache** | Verified-query cache, `ef_search` cache, rehydrate pub/sub, Django cache/throttle. Required. |
| **redis-broker** | Only present because the `api` container `depends_on` it; the query path never uses Celery. Started, unused. |
| **ollama** | The SLM for routing / decompose / SQL generation / NL answer. Required. |
| **inference** | The warm engine — BGE-M3 dense+sparse, reranker, 5-signal retrieval, DuckDB/psycopg2 execution. Required. |
| **api** | HTTP `/api/v1/query`: resolves scope+tenant server-side, forwards to inference with context headers. Required for the HTTP surface. |

**Never started** (query-only): `worker`, `beat`, `ingest-worker`, `vllm`, `nginx` (nginx optional — see gotchas).

### ⚠️ The one thing that decides self-containment: source **dialect**

SQL is executed differently per source kind (`veda_core/veda/execution.py`):

- **Parquet / CSV / XLSX sources** (`dialect ∈ {parquet, csv, csv_lake, xlsx, excel}`) → executed by
  **DuckDB over the materialized `data/<id>/tables/*.parquet`**. **Fully self-contained** — no live
  database beyond the dump.
- **Relational sources** (`dialect ∈ {postgres, mysql, sqlite, …}`) → executed by **psycopg2 against
  the LIVE client source DB** (resolved from the `sources_source` row), and value-validation +
  `postgres_scanner` federation also dial it. **A dump does not include this** — that live DB must be
  reachable from the demo box, or you must re-materialize the source as parquet before exporting.

**Your current sources 4 and 5 are parquet** (`maintenance.parquet`, `vendors.parquet`,
`amenities_catalog.parquet`), so the cross-source example query runs entirely on DuckDB over parquet —
**no live source DB required.** `restore.sh` prints a per-source dialect preflight so you catch any
relational source before demoing.

---

## VM specifications (query-only demo)

Much lighter than the full production box — no ingestion, single SLM, one inference worker.

| Resource | Minimum (works) | Recommended (snappy demo) | Notes |
|---|---|---|---|
| **GPU** | none (CPU-only) | 1× 12–16 GB (**T4 16 GB**, **RTX 3060 12 GB**, **L4 24 GB**) | Only Ollama 7B q4 (~6 GB) + BGE-M3/reranker (~3 GB) ≈ **9 GB**. CPU-only works but the SLM runs ~5–8× slower — fine for a low-traffic demo, sluggish per query. A modest GPU makes queries feel interactive. |
| **vCPU** | 4 | **8** | Postgres, pgbouncer, 2× Redis, 1× uvicorn inference, Ollama, api. No ingestion parsing load. |
| **RAM** | 16 GB | **32 GB** | Postgres (demo corpus is small) + inference ~4 GB (1 worker: torch + BGE) + Ollama host buffers + thin api/redis. 16 GB is enough for a small dump; 32 GB is comfortable. |
| **Disk** | 60 GB SSD | **100 GB SSD** | Inference image (torch+CUDA) ~10 GB + models ~10–15 GB (BGE-M3 ~2.5 GB, reranker ~1 GB, SLM ~5 GB) + `pg_data` (small demo corpus, a few GB) + the bundle. |
| **OS** | Ubuntu 22.04 | Ubuntu 22.04 | Docker Engine + compose plugin. GPU path also needs `nvidia-container-toolkit`. |

`INFERENCE_WORKERS=1` for the demo (already set in `docker-compose.demo.yml`).

---

## Step 1 — Export from your (ingesting) machine

With your local stack up (so the `model_cache` / `ollama_models` volumes exist and Postgres is
reachable on host port `15432`):

```bash
scripts/demo/export.sh                    # writes ./demo_bundle/
# override defaults if needed:
#   PGPORT=15432 PGPASSWORD=change-me COMPOSE_PROJECT=veda-platform scripts/demo/export.sh /path/out
```

Produces `demo_bundle/`:
```
veda_engine.dump     # engine store (embeddings/graphs)          — REQUIRED
veda_django.dump     # Django substrate (sources/auth/log)       — for the HTTP api
veda_data.tgz        # veda_core/data filesystem artifacts       — REQUIRED
model_cache.tgz      # BGE-M3 + reranker weights                 — REQUIRED
ollama_models.tgz    # the pulled SLM (optional; else pull on box)
client_bge.tgz       # fine-tuned BGE checkpoint (only if present)
```

Copy the whole directory (plus your **`.env`** and **`docker/userlist.txt`**) to the demo box.

---

## Step 2 — Restore + launch on the demo box

From the repo root on the demo server, with `.env` and `docker/userlist.txt` in place:

```bash
scripts/demo/restore.sh                   # reads ./demo_bundle/ by default
#   scripts/demo/restore.sh /path/to/demo_bundle
```

It imports the volumes, untars `veda_core/data`, loads both DB dumps, starts the query-only stack,
pulls the SLM if it wasn't bundled, and prints a `/readyz` check. Under the hood it runs:

```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml up -d \
    postgres pgbouncer redis-broker redis-cache ollama inference api
```

---

## Step 3 — Verify

```bash
curl -s http://localhost:8000/readyz | jq          # expect all checks "ok"
curl -s -X POST http://localhost:8000/api/v1/query \
     -H 'Content-Type: application/json' \
     -d '{"query": "your natural-language question"}' | jq
```

`readyz` should report `postgres`, `redis_cache`, `inference`, and `slm_backend` all `ok`. The demo
uses dev settings + `VEDA_ALLOW_ANONYMOUS=1`, so no auth token is required.

---

## Gotchas

- **Dump direct, not through PgBouncer.** `pg_dump` needs a real session — use host port `15432`
  (→ Postgres), never `6432` (PgBouncer transaction pooling). The export script already does this.
- **`source_id` must line up.** The `data/<id>/` dirs (e.g. `4`, `5`), the `veda_engine.source_registry`
  rows, and the Django `Source` rows all key on the same ids. Shipping all three together keeps them
  consistent — don't hand-edit one.
- **Models must be offline.** The inference/ollama images start with `HF_HUB_OFFLINE=1`; if a weight
  is missing from the imported `model_cache`, the tier fails at boot rather than fetching. Re-run the
  export's volume step if `model_cache.tgz` is incomplete.
- **Same password everywhere.** `.env` (`change-me` by default) must match `docker/userlist.txt`
  (PgBouncer md5). Copy both from the source machine, or regenerate `userlist.txt` (see
  `PRODUCTION_READINESS_PLAN.md` B1a).
- **CPU-only latency.** With no GPU, the 7B SLM dominates per-query time. Uncomment the GPU block in
  `docker-compose.demo.yml` if the demo box has an NVIDIA card.
- **Want TLS/nginx?** Add `nginx` to the `up` list — but for a headless demo, hitting the api on
  `:8000` directly (as configured here) is simpler.
