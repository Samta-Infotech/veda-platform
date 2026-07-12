# VEDA Platform — Deployment Guide (single VM + Docker Compose)

Deploys the full stack on **one VM** with a TLS-terminating nginx as the only published service:

```
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

**Audience:** ≤ 5 concurrent users, but both **query** (interactive) and **ingestion** (admin-triggered)
must be fast. The single hard constraint is the **GPU** — vLLM (query SLM), the BGE-M3 + reranker
inference tier, and Ollama (ingestion SLM) all want it.

> **Prerequisite:** apply the fixes in [`PRODUCTION_READINESS_PLAN.md`](PRODUCTION_READINESS_PLAN.md)
> first (B1–B13). Steps below assume secrets are wired (B1), TLS is active in nginx (B4), the api
> replicas balance (B5), migrations run one-shot (B6), and the ingest worker exists in prod (B7).

---

## 1. VM specifications

| Resource | Minimum | Recommended (fast query **and** ingestion) | Why |
|---|---|---|---|
| **GPU** | 1× 24 GB (NVIDIA **L4** / **A10** / **RTX 4090**) | 1× 48 GB (**L40S** / **A6000**) | vLLM 7B (AWQ ~6 GB) + BGE-M3 & reranker (~3 GB) + Ollama 7B q4 (~6 GB) ≈ **15 GB** on one card. 24 GB fits with headroom; 48 GB removes all three-way contention and allows fp16 7B. |
| **vCPU** | 8 | **16** | 3× gunicorn api (12 thin procs), celery worker + beat, ingest-worker (PDF/DOCX parse + MinHash), 2× uvicorn inference, Postgres, 2× Redis, PgBouncer, nginx. |
| **RAM** | 32 GB | **64 GB** | Postgres 4 GB + redis-cache 2 GB + inference ~8 GB (torch + BGE per worker × 2) + Ollama/vLLM host buffers ~4 GB + api/worker ~4 GB + ingestion parse spikes + OS/Docker. |
| **Disk** | 100 GB NVMe SSD | **200 GB NVMe SSD** | Inference image (torch+CUDA) ~10 GB, model weights ~15–30 GB, `pg_data` (pgvector embeddings grow with corpus) 50 GB+, image/build layers. NVMe matters for pgvector scans + model load latency. |
| **OS** | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS | NVIDIA driver 535+, `nvidia-container-toolkit`. |
| **Network** | — | Static IP + DNS A record + open 80/443 | nginx is the only ingress; everything else stays on the internal `veda_net`. |

**Sizing notes for ≤ 5 users**
- Serve a **quantized 7B** (AWQ/GPTQ) in vLLM — for this scale it's near-indistinguishable in quality
  and keeps first-token latency + VRAM low.
- `INFERENCE_WORKERS=2` is plenty. Size from **measured** per-worker RSS (§8.1), not CPU — `WORKERS` is
  intentionally required with no default.
- `GUNICORN_WORKERS=4` per api replica is more than enough; the api is thin (no models).
- Keep BGE-M3 + reranker on the **GPU** — putting them on CPU is the most common source of slow query
  latency in this stack.

---

## 2. Host setup (one-time)

```bash
# Docker Engine + compose plugin
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"   # re-login after

# NVIDIA driver + container toolkit (GPU passthrough)
sudo apt-get install -y nvidia-driver-535
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

# verify GPU is visible to containers
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## 3. Get the code

```bash
git clone <repo-url> veda-platform && cd veda-platform
```

---

## 4. Provision model weights (offline / zero-egress)

Both `inference` and `vllm`/`ollama` start with `HF_HUB_OFFLINE=1`, so weights must be present in the
`model_cache` volume **before** first boot. Two categories:

**a) Preserved fine-tuned artifacts** (git-ignored, ship out-of-band):
- `veda_core/ingestion/client_bge/model.safetensors` — 1.2 GB fine-tuned BGE checkpoint
- `veda_core/schema/kuzu_graph` — persisted knowledge graph

Copy these into place on the VM (scp/rsync from your artifact store).

**b) Base HF models + SLM** — download into the `model_cache` volume once:
```bash
# create the named volume and populate it via the helper
docker volume create veda-platform_model_cache
# BGE-M3, bge-reranker-v2-m3 into /models/hf_cache (see docker/download_models.py)
docker run --rm -v veda-platform_model_cache:/models \
  -e HF_HOME=/models/hf_cache \
  <inference-image> python /app/docker/download_models.py

# Ollama ingestion SLM (into the ollama_models volume)
docker compose up -d ollama
docker compose exec ollama ollama pull "$SLM_MODEL_NAME"     # e.g. qwen2.5-coder:7b

# Ollama NL-summary SLM — a SEPARATE, small instruct model (NOT the coder model
# above) used by query/result_explainer.py to phrase result rows into prose.
# Missing this doesn't crash anything (it falls back to deterministic template
# answers) but silently loses the natural-language summary feature — pull it too:
docker compose exec ollama ollama pull "${NL_SUMMARY_MODEL:-qwen2.5:1.5b-instruct}"

# vLLM query SLM (quantized) — pre-download so boot is offline (B9)
#   pull an AWQ/GPTQ 7B into /models/hf_cache and set SLM_MODEL_DIR accordingly
```

> **Zero-egress check:** after this step, `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` must let every
> tier start with **no** outbound network. If a tier tries to fetch at boot, a weight is missing.

---

## 5. Configure environment

```bash
cp .env.example .env
```
Edit `.env` for prod:
- `DJANGO_SETTINGS_MODULE=config.settings.prod`
- `DJANGO_ALLOWED_HOSTS=your.domain.com`   *(prod.py reads this; must be set or every request 400s)*
- `DEBUG=0`, `VEDA_ALLOW_ANONYMOUS=0`   *(enforce auth — B8)*
- `PGBOUNCER_HOST=pgbouncer`, `PGBOUNCER_PORT=6432`
- `POSTGRES_DB`, `POSTGRES_USER` (password comes from the Docker secret, not here — B1)
- `REDIS_BROKER_URL=redis://redis-broker:6379/0`, `REDIS_CACHE_URL=redis://redis-cache:6379/0`
- `SLM_BACKEND` is set per-service in compose (worker→ollama, inference→vllm); `SLM_MODEL_NAME=...`
- `INFERENCE_WORKERS=2` (measured RSS), `INFERENCE_URL=http://inference:8001`
- `VEDA_HNSW_*` = your tuned parity values; engine `VEDA_*` flags as needed

`.env` is git-ignored. **Never** put the DB password or Django secret key here in prod — use step 6.

---

## 6. Create secrets

Pick **one** mechanism (matching your choice in B1):

**Option A — Docker Swarm secrets** (compose keeps `external: true`):
```bash
docker swarm init
printf '%s' "$(openssl rand -base64 48)" | docker secret create django_secret_key -
printf '%s' 'YOUR_STRONG_DB_PASSWORD'    | docker secret create postgres_password -
```

**Option B — file secrets** (no Swarm; switch compose `secrets:` to `file:`):
```bash
mkdir -p secrets && chmod 700 secrets
openssl rand -base64 48 > secrets/django_secret_key
printf '%s' 'YOUR_STRONG_DB_PASSWORD' > secrets/postgres_password
chmod 600 secrets/*        # add `secrets/` to .gitignore
```

Then regenerate the PgBouncer md5 line from the **same** DB password (B1a):
```bash
PW='YOUR_STRONG_DB_PASSWORD'; USER='veda'
echo "\"$USER\" \"md5$(printf '%s%s' "$PW" "$USER" | md5sum | cut -d' ' -f1)\"" > docker/userlist.txt
```

---

## 7. TLS certificates

Place PEM files where prod compose mounts them (`prod.yml:84` → `/etc/nginx/certs`):
```bash
mkdir -p docker/certs
# Let's Encrypt (recommended):
sudo certbot certonly --standalone -d your.domain.com
sudo cp /etc/letsencrypt/live/your.domain.com/fullchain.pem docker/certs/fullchain.pem
sudo cp /etc/letsencrypt/live/your.domain.com/privkey.pem   docker/certs/privkey.pem
# (or self-signed for a staging box:)
# openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
#   -keyout docker/certs/privkey.pem -out docker/certs/fullchain.pem -subj "/CN=your.domain.com"
```
Set up a certbot renewal cron that re-copies into `docker/certs/` and runs
`docker compose exec nginx nginx -s reload`.

---

## 8. Build and launch

```bash
export COMPOSE="docker compose -f docker-compose.yml -f docker-compose.prod.yml"

$COMPOSE build                                   # api + inference images
$COMPOSE up -d postgres pgbouncer redis-broker redis-cache   # data tier first
$COMPOSE run --rm release                        # one-shot migrate + collectstatic (B6)
$COMPOSE up -d                                   # everything else
$COMPOSE ps                                       # all healthy?
```

Create an admin user + an API token for a client:
```bash
$COMPOSE exec api python manage.py createsuperuser
$COMPOSE exec api python manage.py drf_create_token <username>   # → token for API auth
```

Register a source and trigger the first ingestion (admin token required):
```bash
# via Django admin (/admin) create an apps.sources.Source row, OR the API:
curl -sk -X POST https://your.domain.com/api/v1/admin/ingest \
  -H "Authorization: Token <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"source_id": 1, "force": true}'
# track the IngestionJob in /admin until it reports ready
```

---

## 9. Smoke tests

```bash
BASE=https://your.domain.com

curl -sk $BASE/healthz                       # {"status":"ok"}
curl -sk $BASE/readyz | jq                    # postgres/redis/inference/slm all "ok"
curl -sk $BASE/metrics | head                 # Prometheus text

# a real query (auth enforced in prod — B8)
curl -sk -X POST $BASE/api/v1/query \
  -H "Authorization: Token <user-token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "how many orders last month?"}' | jq
```
Expect `readyz` → `"status":"ready"` with every dependency `ok`. A query returns
`{"status": ..., "result": ..., "latency_ms": ..., "request_id": ...}`.

---

## 10. Operations

- **Logs:** `$COMPOSE logs -f api inference worker` (structured/JSON per B12; grep the request id).
- **Rolling update:** `git pull && $COMPOSE build && $COMPOSE run --rm release && $COMPOSE up -d`.
- **Postgres backup (nightly cron):**
  ```bash
  $COMPOSE exec -T postgres pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" | gzip > backup-$(date +%F).sql.gz
  ```
  The `pg_data` volume is the only copy of the ingested substrate — back it up off-host.
- **GPU watch:** `nvidia-smi -l 5` while a query + ingestion run concurrently; if VRAM is tight,
  quantize the vLLM model further or move to the 48 GB card.
- **Scale api:** `deploy.replicas` in `prod.yml` (nginx re-resolves per request after B5).
- **Re-ingestion fan-out:** the rehydrate pub/sub (`inference/main.py`) drops warm caches on every
  inference replica automatically after an ingest — no restart needed.

---

## 11. Post-deploy checklist

- [ ] `readyz` reports all dependencies `ok` (not `degraded`)
- [ ] HTTP (`:80`) 301-redirects to HTTPS; HTTPS serves with a valid cert
- [ ] `manage.py check --deploy` reports no warnings (secret key, SSL redirect, HSTS)
- [ ] Anonymous `/api/v1/query` is **rejected** (auth enforced)
- [ ] An ingestion job completes and a subsequent query returns real results
- [ ] `nvidia-smi` shows vLLM + inference (+ ollama during ingest) resident on the GPU
- [ ] Nightly `pg_dump` cron + certbot renewal cron installed
- [ ] Secrets are **not** in `.env` or git; `docker/userlist.txt` matches the DB password
