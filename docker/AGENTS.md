# docker/ — images, entrypoints, infra config

Topology + service table: [../docs/ARCHITECTURE.md §11](../docs/ARCHITECTURE.md),
[../docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md). Prod hardening gaps:
[../docs/PRODUCTION_READINESS_PLAN.md](../docs/PRODUCTION_READINESS_PLAN.md).

| File | Role |
|------|------|
| `Dockerfile.api` | `python:3.11-slim` **thin** image → `api` / `worker` / `beat` (Django + DRF + Celery + gunicorn, **no torch**). `COPY . /app`, non-root `veda` uid 1000. |
| `Dockerfile.inference` | `python:3.11-slim` **heavy** image → `inference` / `ingest-worker` (torch / transformers / FlagEmbedding / sentence-transformers + FastAPI/uvicorn; re-installs django/celery so the one image also runs the ingest worker). `HF_HUB_OFFLINE=1`. Model `COPY` commented out — weights come from the `model_cache` volume. |
| `entrypoint.api.sh` | Role dispatch by `$ROLE`: `api` runs `migrate` + `collectstatic` every container then gunicorn (`--reload` only if `DEV_AUTORELOAD=1`); `worker` → `celery worker -Q high,default -c 4`; `beat` → file `PersistentScheduler`. |
| `entrypoint.inference.sh` | `uvicorn inference.main:app :8001`. `WORKERS` is **required, no default** (size from measured RSS). |
| `nginx.conf` | Single ingress. Rate-limit `10r/s` burst 20, `client_max_body_size 20m`, static `upstream veda_api { server api:8000; }`, `proxy_read_timeout 120s`. **TLS block commented out.** `/healthz` forces `Host $host` (upstream name has an underscore Django rejects). |
| `pgbouncer.ini` | Transaction pooling `:6432`. `default_pool_size=20`, `reserve_pool_size=5`, `max_prepared_statements=200`, `auth_type=plain`. Comment says "pg16" — matches `docker-compose.yml` again as of 2026-09-10 (see gotchas). |
| `userlist.txt` | Dev plaintext creds (`"veda" "change-me"`). Prod should regenerate md5/scram. |
| `initdb/01-init.sql` | Runs once on first cluster boot: `CREATE EXTENSION vector` in `POSTGRES_DB`, creates the `veda_engine` DB + `vector` there. |
| `download_models.py` | Run-once (online) pre-pull of `BAAI/bge-m3` + `BAAI/bge-reranker-v2-m3` into `HF_HOME=/models`. |
| `reingest_chain.sh` | **Dev-only, machine-specific cruft** — hardcoded `/Users/ekesel/...` + container names. |

Compose files (repo root): `docker-compose.yml` (dev base, 12 services / 13 with the
`proxy` profile), `.prod.yml` (adds `vllm`, GPU, secrets, replicas), `.demo.yml` (query-only,
`VEDA_ALLOW_ANONYMOUS`), `.mlflow.yml` (additive observability sidecar).

## Gotchas
- **`docker-compose.yml`'s `postgres` is pinned to `pg16` as of 2026-09-10** (was `pg17` —
  the local `pg_data` volume was PG16-formatted and refused to start under pg17; pinning
  back was the user's explicit call over re-migrating the volume). `docker-compose.demo.yml`
  still says `pg17` and has the same latent failure mode if pointed at that volume. Re-bump
  the dev image via a proper `pg_dumpall`-and-restore (or `pg_upgrade`) when convenient.
- **Prod compose is not deployable as-is** — `PRODUCTION_READINESS_PLAN.md` B1–B13
  (secrets not read, TLS commented out, no `release` migrate service, no prod
  `ingest-worker`, torch unpinned, vLLM not offline) are all open.
- The demo override can't remove `inference.depends_on: ollama` (compose merges
  `depends_on`) — it uses `--no-deps` instead.
