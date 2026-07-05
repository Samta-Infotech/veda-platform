# VEDA Platform — Operations & Runbook (migration plan §7.3, §7.4)

## Backups (§7.3)

Substrate is rebuildable (re-ingest) but expensive, so back it up anyway.

```bash
# Internal store (Django substrate + pgvector) and the engine operational store.
docker compose exec postgres pg_dump -U veda -d veda        -Fc -f /backup/veda_$(date +%F).dump
docker compose exec postgres pg_dump -U veda -d veda_engine -Fc -f /backup/veda_engine_$(date +%F).dump

# Restore (into a fresh DB):
docker compose exec postgres pg_restore -U veda -d veda --clean --if-exists /backup/veda_YYYY-MM-DD.dump
```

Verified: `pg_dump -Fc` of `veda` → 46 tables / 92 restorable objects; `pg_restore -l` lists them.

## Rollback (§7.3)

The preserved engine still runs standalone from `veda_core/`, so a bad platform deploy
falls back to the pre-migration path:

1. **Blue/green on api + inference** — keep the previous image tags; flip the compose
   `image:`/tag back and `docker compose up -d --no-deps api inference`. Health-gated
   (`/healthz`, `/readyz`) so a failed rollout never takes traffic.
2. **Engine-only fallback** — the legacy front door works directly:
   `docker compose run --rm --entrypoint python inference -c \
      "import main; print(main.run_hybrid_query('...'))"` (reads the engine store),
   independent of the Django tiers. This is the ultimate safety net if the platform
   layer is wedged.
3. **Substrate rollback** — restore the `veda` dump (above); bump `SubstrateVersion`
   and publish a rehydrate fan-out so every inference replica reloads (`/v1/rehydrate`).

## Production security posture (§7.4 — verified)

- **Single ingress**: only `nginx` publishes host ports (80/443). Verified via
  `docker compose -f docker-compose.yml -f docker-compose.prod.yml config`.
- **Split Redis**: `redis-broker` (unbounded, no evict) and `redis-cache`
  (`maxmemory` + `allkeys-lru`) are separate instances — a cache-eviction storm can't
  touch the broker.
- **PgBouncer**: every DB pool dials PgBouncer (`pool_mode=transaction`), so
  `N workers × M replicas × pool_size` can't exceed Postgres `max_connections`.
  Two hazards fixed in code: never `SET SESSION READ ONLY` (poisons a pooled conn) and
  `SET LOCAL` only inside an explicit transaction (scoped + released at COMMIT).
- **Secrets**: `postgres_password`, `django_secret_key` are Docker secrets in prod, not
  env plaintext.
- **Zero-egress**: `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`; models pre-pulled to the
  `model_cache` volume; SLM (Ollama/vLLM) on `veda_net`. Startup makes no network calls.
- **vLLM** is the prod query-time SLM backend (`SLM_BACKEND=vllm`); Ollama stays for
  dev/ingestion. GPU reserved on `inference` + `vllm` + `ollama` only.
- **TLS**: terminate at nginx (`docker/certs`), `listen 443 ssl` block in `nginx.conf`.

## Load & scaling (§7.2, §8.1)

- Size `inference` **workers from measured per-worker RSS**, not CPU: a fully warmed
  worker holds BGE-M3 + bge-reranker-v2-m3 + MiniLM. `workers_per_replica =
  floor((replica_RAM_GB × 0.8 − OS) / PER_WORKER_RSS_GB)`; prefer more replicas over
  more workers (each worker re-pays the model RAM).
- **SLM is the throughput bottleneck**: on CPU, a single qwen SQL-gen call is ~60–170s.
  Production moves the query-time SLM to **vLLM + GPU** (continuous batching) — this is
  the primary latency/throughput lever. Keep a bounded concurrency gate in front.
- PgBouncer holds the Postgres connection ceiling under `N×M` pools at peak.

## Known production behaviors (documented, not bugs)

- **Unordered `LIMIT` non-determinism** (§7.1): the deterministic head emits
  `LIMIT 100`/`LIMIT 20` without `ORDER BY` (pipeline.py), so the *same* query can return
  different rows run-to-run. Inherited from the legacy engine ("flow preserved"); flagged
  here and in release notes rather than silently changed. Parity asserts row COUNT + SQL
  (not row identity) for these.
- **HNSW `ef_search`** is pinned to the §7.1a-tuned value (`VEDA_HNSW_EF_SEARCH=40`,
  recall@k=1.0 vs exact cosine on the home schema). Re-run `scripts/hnsw_parity_sweep.py`
  for a new/larger source and re-pin.
