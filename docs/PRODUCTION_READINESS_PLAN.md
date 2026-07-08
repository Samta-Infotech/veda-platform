# VEDA Platform — Production Readiness Implementation Plan

**Target:** single VM + `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d`, TLS-terminating nginx **inside** the compose (the one published service).
**Scale:** ≤ 5 concurrent users, but both the **query** and **ingestion** flows must stay low-latency.

This plan fixes the gaps that stop a *correct, secure* boot behind nginx. Each item lists the
evidence, the fix, and a concrete patch. Work top-to-bottom: 🔴 blockers gate boot, 🟠 hardening
gates real traffic, 🟡 operational is should-have.

Status legend: `[ ]` todo · `[x]` done.

---

## 🔴 Blockers — box will not come up correctly without these

### [ ] B1 — Wire Docker secrets into Django (currently declared but never read)

**Evidence:** `docker-compose.prod.yml:12-15,53` mounts `postgres_password` / `django_secret_key`
at `/run/secrets/*`, but `config/settings/base.py:22,81` only reads `os.environ`. Nothing reads
`/run/secrets`. **Result: prod boots with the insecure dev `SECRET_KEY` and an empty DB password.**

**Fix:** add a file-or-env secret reader and use it for the two secrets.

`config/settings/base.py`:
```python
def _read_secret(name: str, default: str = "") -> str:
    """Prefer ${name}_FILE (Docker secret at /run/secrets/...), then env, then default."""
    path = os.environ.get(f"{name}_FILE")
    if path and os.path.exists(path):
        with open(path) as fh:
            return fh.read().strip()
    return os.environ.get(name, default)

SECRET_KEY = _read_secret("DJANGO_SECRET_KEY", "insecure-dev-key-do-not-use-in-prod")
# ...and for the DB password (keep the VEDA_DB_PASSWORD override):
_DB_PASSWORD = os.environ.get("VEDA_DB_PASSWORD") or _read_secret("POSTGRES_PASSWORD", "")
```

`docker-compose.prod.yml` — point the api/worker/beat/inference services at the secret files:
```yaml
  api:
    environment:
      DJANGO_SECRET_KEY_FILE: /run/secrets/django_secret_key
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
```
(repeat the two `*_FILE` env vars on `worker`, `beat`, `inference`; they already list `secrets:`).

**Also:** the `postgres` service must use the same secret. The official image supports it natively:
```yaml
  postgres:
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    secrets: [postgres_password]
```
The PgBouncer `docker/userlist.txt` md5 hash **must be regenerated from this same password** (see B1a).

Create the external secrets once on the host:
```bash
printf '%s' "$(openssl rand -base64 48)"  | docker secret create django_secret_key -
printf '%s' 'YOUR_STRONG_DB_PASSWORD'     | docker secret create postgres_password -
```
> Note: `docker secret` requires Swarm mode (`docker swarm init`). On a plain single-VM compose
> deploy, either (a) run `docker swarm init` and keep `external: true`, or (b) switch the compose
> `secrets:` blocks to `file:` form pointing at root-owned files under `./secrets/` that are **not**
> committed (add `secrets/` to `.gitignore`). Pick one in the deployment guide.

---

### [ ] B1a — Regenerate PgBouncer `userlist.txt` from the real password

**Evidence:** `docker/userlist.txt` holds an md5 auth line that must match the Postgres password.
If B1 changes the password, PgBouncer auth breaks.

**Fix:**
```bash
# md5 format: "md5" + md5(password + username)
PW='YOUR_STRONG_DB_PASSWORD'; USER='veda'
echo "\"$USER\" \"md5$(printf '%s%s' "$PW" "$USER" | md5sum | cut -d' ' -f1)\"" > docker/userlist.txt
```
Keep `docker/userlist.txt` git-ignored (it already is per `.gitignore`).

---

### [ ] B2 — Fix env-var name mismatch: `SECRET_KEY` vs `DJANGO_SECRET_KEY`

**Evidence:** `.env.example` exports `SECRET_KEY=`, but the code reads `DJANGO_SECRET_KEY`
(`base.py:22`). Any key set as `SECRET_KEY` is silently ignored.

**Fix:** rename the `.env.example` key to `DJANGO_SECRET_KEY` (dev only; prod uses the secret file).

---

### [ ] B3 — Add `SECURE_PROXY_SSL_HEADER` (TLS redirect loop)

**Evidence:** `config/settings/prod.py:13` sets `SECURE_SSL_REDIRECT = True`. nginx terminates TLS and
forwards plain HTTP with `X-Forwarded-Proto: https` (`nginx.conf:54`), but Django doesn't trust that
header by default → it sees `http` → 301 → nginx → 301 → **infinite redirect loop**.

**Fix:** `config/settings/prod.py`:
```python
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
```

---

### [ ] B4 — Activate real TLS in nginx

**Evidence:** `docker/nginx.conf:31-34` has the `listen 443 ssl` block **commented out** and no
HTTP→HTTPS redirect, yet `docker-compose.prod.yml:85` publishes `443:443`. HTTPS will not serve.

**Fix:** replace the single `server { listen 80; }` with an HTTP→HTTPS redirect + a TLS server:
```nginx
    server {                       # HTTP → HTTPS
        listen 80;
        server_name _;
        return 301 https://$host$request_uri;
    }

    server {
        listen 443 ssl;
        server_name _;
        ssl_certificate     /etc/nginx/certs/fullchain.pem;
        ssl_certificate_key /etc/nginx/certs/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;

        client_max_body_size 20m;

        location /healthz { proxy_pass http://veda_api/healthz; proxy_set_header Host $host; }
        location /static/ { alias /app/staticfiles/; }
        location /metrics {                      # internal-only (B12/ops)
            allow 127.0.0.1; allow 10.0.0.0/8; deny all;
            proxy_pass http://veda_api/metrics; proxy_set_header Host $host;
        }
        location / {
            limit_req zone=veda_api burst=20 nodelay;
            proxy_pass http://veda_api;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 600s;
        }
    }
```
Certs live in `./docker/certs/{fullchain,privkey}.pem` (mounted read-only per `prod.yml:84`).

---

### [ ] B5 — Make nginx balance all api replicas (runtime DNS re-resolution)

**Evidence:** `nginx.conf:23-25` `upstream veda_api { server api:8000; }` resolves the `api` name
**once at startup**. With `prod.yml:54 replicas: 3`, Docker's DNS round-robin is never used — all
traffic pins to whichever replica IP nginx cached at boot.

**Fix:** use Docker's embedded DNS (`127.0.0.11`) + a variable `proxy_pass` so nginx re-resolves per
request. Drop the static `upstream` block:
```nginx
http {
    resolver 127.0.0.11 valid=10s ipv6=off;
    # ...
    location / {
        set $api_upstream http://api:8000;   # variable => runtime resolution
        proxy_pass $api_upstream;
        # ...same proxy_set_header lines...
    }
}
```
(Apply the same `set $api_upstream ...; proxy_pass` pattern to `/healthz` and `/metrics`.)

---

### [ ] B6 — Run migrations/collectstatic once, not in every replica

**Evidence:** `docker/entrypoint.api.sh:14-15` runs `migrate` + `collectstatic` for **every** api
container. With `replicas: 3` all three race on DDL through transaction-mode PgBouncer.

**Fix:** add a one-shot init service and gate the api on it. `docker-compose.prod.yml`:
```yaml
  release:
    extends: { service: api }         # or reuse build/image of api
    command: >
      sh -c "python manage.py migrate --noinput &&
             python manage.py collectstatic --noinput"
    restart: "no"
    deploy: { replicas: 1 }

  api:
    depends_on:
      release: { condition: service_completed_successfully }
```
And in `entrypoint.api.sh`, gate the boot-time migrate behind an env flag so the api role only
serves:
```bash
    api)
        if [ "${RUN_RELEASE:-0}" = "1" ]; then
            python manage.py migrate --noinput
            python manage.py collectstatic --noinput
        fi
        exec gunicorn config.wsgi:application --bind 0.0.0.0:8000 \
            --workers "${GUNICORN_WORKERS:-4}" --timeout "${GUNICORN_TIMEOUT:-60}"
        ;;
```
> Migrations run session-style DDL; PgBouncer is transaction-pooled. Running the one-shot `release`
> against **postgres:5432 directly** (not pgbouncer:6432) avoids pooled-DDL edge cases. Set
> `PGBOUNCER_HOST=postgres PGBOUNCER_PORT=5432` for the `release` service only.

---

### [ ] B7 — Add the ingestion worker to prod compose

**Evidence:** dev has `ingest-worker` (`docker-compose.yml:105`, GPU/ML image, consumes
`ingestion,high`). `docker-compose.prod.yml` omits it, so `IngestTriggerView`
(`apps/query/views.py:135`) enqueues to a queue **nobody consumes** → ingestion never runs.

**Fix:** port the `ingest-worker` service into `prod.yml` (it uses `Dockerfile.inference` because L0
ingestion needs torch + Django + celery together), with `SLM_BACKEND=ollama`, the `model_cache`
mount, GPU reservation, `restart: always`, and the secret `*_FILE` env from B1.

---

## 🟠 Hardening — before real traffic

### [ ] B8 — Enforce auth in prod (query path is currently open)

**Evidence:** `base.py:157` `VEDA_ALLOW_ANONYMOUS` defaults `"1"`; `QueryView.permission_classes`
= `[AllowAny]` (`apps/query/views.py:35`); tenant is taken from the request body when unauthenticated
(`_resolve_tenant`, line 115-119). Phase 6.2 is unfinished.

**Fix:** in prod, require a token and derive tenant from the principal:
- Set `VEDA_ALLOW_ANONYMOUS=0` in the prod env.
- Make `QueryView.permission_classes` conditional: `[AllowAny] if settings.VEDA_ALLOW_ANONYMOUS else [IsAuthenticated]`.
- In `_resolve_tenant`, when `VEDA_ALLOW_ANONYMOUS` is off, **ignore** `data.get("tenant")` and use
  only `request.user`.
- Provision tokens via `manage.py drf_create_token <user>` (rest_framework.authtoken is already installed).

### [ ] B9 — Keep vLLM zero-egress (pre-provision weights)

**Evidence:** `prod.yml:30-39` runs `vllm --model ${SLM_MODEL_NAME}` with only a `model_cache:/models`
volume and no offline flags → vLLM fetches weights from HuggingFace at boot, contradicting the stated
zero-egress constraint (`prod.yml` header, §9/§2).

**Fix:** pre-download the SLM into `model_cache` (use/extend `docker/download_models.py`), then:
```yaml
  vllm:
    environment: { HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1", HF_HOME: /models/hf_cache }
    command: ["--model", "/models/hf_cache/${SLM_MODEL_DIR}", "--download-dir", "/models/hf_cache"]
```
For ≤ 5 users, serve a **quantized** 7B (AWQ/GPTQ, ~5–6 GB VRAM) so the KV cache has headroom and
first-token latency stays low. See deployment guide for the VRAM budget.

### [ ] B10 — Install `django-celery-beat` for the prod DatabaseScheduler

**Evidence:** `entrypoint.api.sh:29` documents that prod should use
`django_celery_beat.schedulers:DatabaseScheduler`, but `requirements/api.txt` doesn't install it.

**Fix:** add `django-celery-beat>=2.6` to `requirements/api.txt`, add `django_celery_beat` to
`INSTALLED_APPS`, set `CELERY_BEAT_SCHEDULER=django_celery_beat.schedulers:DatabaseScheduler` in the
`beat` service env. (Its tables come in via the `release` migrate.)

### [ ] B11 — Health-gate the model tiers

**Evidence:** `vllm`, `ollama`, `inference` use bare `depends_on` (no `condition: service_healthy`).
Rollout can route traffic before the engine is warm.

**Fix:** add healthchecks and gate dependents:
- `vllm`: `CMD curl -f http://localhost:8000/health`
- `ollama`: `CMD curl -f http://localhost:11434/api/tags`
- `inference` already has one (`docker-compose.yml:150`); make `api`/nginx depend on it with
  `condition: service_healthy`.

### [ ] B12 — Structured logging to stdout

**Evidence:** no `LOGGING` dict in any settings file; `RequestIdMiddleware` sets a request id that
never reaches a log line.

**Fix:** add a JSON/stdout `LOGGING` config in `base.py` that includes the request id, so
api/worker/inference logs are aggregatable (`docker logs` / journald / Loki).

### [ ] B13 — Pin `torch`

**Evidence:** `requirements/inference.txt` pins `transformers==4.49.0`, `FlagEmbedding==1.2.10`,
`scipy==1.11.4` but leaves `torch` unpinned — a rebuild can drift the CUDA/torch ABI.

**Fix:** pin to the tested CUDA build, e.g. `torch==2.3.1` (match the host CUDA / driver; see guide).

---

## 🟡 Operational — should-have

- [ ] **`.dockerignore`** — both Dockerfiles `COPY . /app`; with no `.dockerignore` they pull
  `.venv/`, `.git/`, `logs/`, `data/` into the image (bloat + secret risk). Add one.
- [ ] **Postgres backup/restore** — no runbook. Add a nightly `pg_dump` (or WAL archiving) + a
  documented restore. The `pg_data` volume is the only copy of ingested substrate.
- [ ] **Resource limits on GPU tiers** — `prod.yml` limits only `postgres`/`api`. Add memory limits to
  `inference`/`vllm`/`ollama` so a model load can't OOM the host.
- [ ] **CI gate** — `manage.py check --deploy`, `pytest tests/`, migration-drift check
  (`makemigrations --check --dry-run`).
- [ ] **Delete vestigial stub** — `inference/engine.py:get_engine()` raises `NotImplementedError` but
  is unused (the real warm-load is `inference/loaders.py:44`). Remove to avoid confusion.
- [ ] **nginx `/readyz`** — optionally expose for external uptime checks (internal-only like `/metrics`).

---

## Execution order

1. **B1, B1a, B2, B3, B4, B5, B6** → correct, secure HTTPS boot behind nginx.
2. **B7, B8** → complete + secure the request path (ingestion consumer + auth).
3. **B9, B10, B11, B13** → zero-egress, scheduler, rollout safety, reproducible builds.
4. **B12 + 🟡 operational** → observability, backups, supply chain.

Verify after each stage with the deployment guide's smoke tests (`/healthz`, `/readyz`, a signed
`/api/v1/query`, an admin `/api/v1/admin/ingest`).
