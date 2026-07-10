# Mac mini 3 — Application Host: Setup Requirements

**Purpose of this doc:** a grounded requirements spec for the planning agent to turn into a concrete
plan + step list. It describes what Mac mini 3 must run, how it wires to the other two minis, what
data it needs, and the open decisions. All file/config references are verified against the repo.

---

## 1. Topology

Three Mac minis on the same LAN, query-only (no ingestion runs here — data arrives as a dump):

| Node | Role | Serves | Port |
|---|---|---|---|
| **mini 1** | SLM | Ollama, `qwen2.5-coder:7b` (Metal) | `11434` |
| **mini 2** | Embeddings | `metal_embed_server.py` — BGE-M3 + reranker (Metal) | `11435` |
| **mini 3** | **Application host (THIS doc)** | Postgres, PgBouncer, Redis ×2, inference, api, nginx | `8000` / `443` |

mini 3 runs the stack in **Docker Desktop for Mac**. That's a Linux VM with **no GPU passthrough** —
which is fine here because all model compute is **offloaded** to mini 1 (SLM) and mini 2 (BGE). mini 3
only does orchestration, retrieval math (RRF/signals), DuckDB execution, and DB/Redis I/O.

---

## 2. Services to run on mini 3

Start **only** this subset of `docker-compose.yml`:

| Service | Role | Notes |
|---|---|---|
| `postgres` | holds `veda_engine` (embeddings/graphs) + `veda` (registry/FK/glossary/cache) | loaded from dump |
| `pgbouncer` | connection pool — both `PGBOUNCER_HOST` and `VEDA_INTERNAL_HOST` dial it | userlist must match DB password |
| `redis-cache` | verified-query cache, ef_search cache, rehydrate pub/sub, Django cache | required |
| `redis-broker` | only because `api` depends on it; query path never uses Celery | started, unused |
| `inference` | the warm engine (orchestration + retrieval), **offloads models to mini 1/2** | see §4 |
| `api` | DRF `/api/v1/query`; resolves scope/tenant, forwards to inference | HTTP surface |
| `nginx` | optional TLS ingress | or expose `api:8000` directly |

**Do NOT run on mini 3:** `ollama` (on mini 1), `vllm`, `worker`, `beat`, `ingest-worker`, and no
in-process BGE (offloaded to mini 2).

---

## 3. The critical wiring — point mini 3 at mini 1 and mini 2

In `.env` on mini 3 (replace with the actual LAN IPs / mDNS hostnames):

```bash
# SLM → mini 1 (LAN IP, NOT host.docker.internal — it's a different machine)
SLM_BACKEND=ollama
OLLAMA_URL=http://<mini1-ip>:11434

# BGE embeddings + reranker → mini 2
METAL_EMBED_URL=http://<mini2-ip>:11435
METAL_EMBED_TIMEOUT=60

# internal (unchanged — these are inside mini 3's compose network)
INFERENCE_URL=http://inference:8001
PGBOUNCER_HOST=pgbouncer
PGBOUNCER_PORT=6432
REDIS_CACHE_URL=redis://redis-cache:6379/0
REDIS_BROKER_URL=redis://redis-broker:6379/1
VEDA_INTERNAL_HOST=pgbouncer
VEDA_INTERNAL_PORT=6432
VEDA_INTERNAL_DBNAME=veda_engine
# demo convenience
DJANGO_SETTINGS_MODULE=config.settings.dev
VEDA_ALLOW_ANONYMOUS=1
INFERENCE_WORKERS=1
```

> Both `OLLAMA_URL` and `METAL_EMBED_URL` are env-driven in the code (`config.py:342`,
> `m3_encoder.py:51`, `reranker.py:48`), so no code change is needed — only these env values.

---

## 4. Compose — use `docker-compose.demo.yml` (already offload-ready)

mini 3 *is* the demo box, so the offload settings live in **`docker-compose.demo.yml`** — no separate
override file. It already: sets `OLLAMA_URL`/`METAL_EMBED_URL` from `.env` (overriding the base
compose's hardcoded `host.docker.internal`), runs dev settings + anon + `WORKERS=1`, and publishes
`api` on `:8000`.

One caveat baked into the file: the base compose's `inference.depends_on` still lists `ollama`
(Compose **merges** `depends_on`, so an override can't remove it). The fix is `--no-deps` at startup,
so `up` never tries to launch a local ollama:

```bash
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.demo.yml"
$COMPOSE up -d postgres pgbouncer redis-broker redis-cache     # infra
$COMPOSE up -d --no-deps inference api                         # app (no local ollama)
```

`scripts/demo/restore.sh` wraps this whole flow (data restore + reachability pre-check + startup).

---

## 5. Data provisioning

mini 3 needs the **dump bundle** produced by `scripts/demo/export.sh` (on the ingesting machine):

- `veda_engine.dump` → restore into `veda_engine` DB (embeddings/graphs)
- `veda_django.dump` → restore into `veda` DB (registry, auth, QueryLog, **verified-query cache**)
- `veda_data.tgz` → untar into `veda_core/data/` (semantic model, graphs, per-source parquet)

Use `scripts/demo/restore.sh`, but **skip the Ollama volume import and the SLM pull** — the SLM lives
on mini 1. The agent should adapt/parameterize `restore.sh` (or document a manual restore) so it does
not try to start `ollama` or pull the model on mini 3.

**Model weights on mini 3 — see the decision in §7.**

---

## 6. Hardware / OS

| Resource | Minimum | Recommended | Notes |
|---|---|---|---|
| Mac mini | M-series, 16 GB | **M-series, 32 GB** | no models load here, but Postgres + the inference container (torch imports ~2 GB RSS even idle) + Docker Desktop's VM add up |
| Free disk | 60 GB | **100 GB** | inference image (torch/CUDA userland) ~10 GB + `pg_data` + bundle |
| OS | macOS + Docker Desktop | — | allocate the Docker VM ≥ 8 GB RAM / 4 CPU in Docker Desktop settings |

---

## 7. Open decisions (operator must choose — flag these in the plan)

1. **Model-weight fallback on mini 3.** With `METAL_EMBED_URL`/`OLLAMA_URL` set and reachable, mini 3
   loads **no** models. But `reranker.py` and `m3_encoder.py` **fall back to in-process CPU** on any
   transport error (`reranker.py:66`). That fallback needs the weights in the `model_cache` volume.
   - **Option A (recommended):** also provision `model_cache` on mini 3. If mini 2 hiccups, queries
     degrade to slow-but-working CPU BGE instead of failing.
   - **Option B:** no local weights → hard dependency on mini 1 + mini 2; a query fails if either is
     down. Lighter disk, less resilient.
2. **Ingress:** nginx+TLS, or expose `api:8000` directly on the LAN (simpler for an internal demo).
3. **Auth:** `VEDA_ALLOW_ANONYMOUS=1` (no token, demo) vs. token auth.

---

## 8. Verification criteria (the plan must include these)

```bash
# health of mini 3 itself + its view of mini 1
curl -s http://localhost:8000/readyz | jq
#   expect: postgres ok, redis_cache ok, redis_broker ok, inference ok,
#           slm_backend ok   ← this confirms mini 3 → mini 1 (Ollama) reachability

# /readyz does NOT probe the BGE server — check mini 2 explicitly:
curl -s http://<mini2-ip>:11435/healthz          # {"status":"ok","device":"mps"}

# end-to-end query (uses BOTH external servers)
curl -s -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"which properties are priced above 10000?"}' | jq

# source-kind preflight (restore.sh prints this): parquet sources 4/5 are self-contained;
# a relational ready-source would need its live DB reachable from mini 3.
```

Gap to note for the agent: `/readyz` (`apps/core/views.py:134`) probes `OLLAMA_URL` but has **no BGE
check**. Optional improvement — add a `METAL_EMBED_URL/healthz` probe to `readyz` so the dashboard
covers all three nodes.

---

## 9. Latency expectation

This 3-node split is the **good** topology: the two dominant costs — SLM generation and BGE
encode/rerank — are both Metal-accelerated on dedicated machines, not on a CPU box. mini 3's residual
work (LAN round-trips in ms, Postgres, DuckDB, RRF/signals) is light. Expect **far** better than the
single-CPU-box eval (which showed ~81 s median). Still:

- Pre-warm the **verified-query cache** with the exact demo queries before export — cached queries
  skip the SLM/BGE round-trips entirely and return in ~1–5 s. The cache ships inside `veda_django.dump`.
- Consider a smaller SLM (`qwen2.5-coder:3b`) on mini 1 if any live (uncached) query feels slow.

---

## 10. Summary of gotchas (verified against the repo)

- `inference.depends_on` includes `ollama` → **must be overridden** to drop it on mini 3.
- `OLLAMA_URL` / `METAL_EMBED_URL` must be **LAN IPs**, not `host.docker.internal` (separate machines).
- mini 1 (`11434`) and mini 2 (`11435`) already bind `0.0.0.0`; ensure macOS firewall allows inbound
  from mini 3, and that mini 3 can reach both.
- PgBouncer `docker/userlist.txt` md5 must match the DB password in the restored dump / `.env`.
- The verified-query cache lives in `veda_django.dump` (`substrate_verifiedquerycache`) — pre-warm
  before export to benefit.
- Query execution is self-contained for **parquet** sources (4, 5) via DuckDB; a **relational**
  source would require its live DB reachable from mini 3.
```
