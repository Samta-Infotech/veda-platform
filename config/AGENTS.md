# config/ — the Django project

| File | Role |
|------|------|
| `settings/__init__.py` | **empty** — `DJANGO_SETTINGS_MODULE` must be named explicitly (`celery.py` / `asgi.py` / `wsgi.py` `setdefault` to `config.settings.dev`). |
| `settings/base.py` | Ten `INSTALLED_APPS`. Two DB aliases (`default`, `source_registry`) both dialing PgBouncer; `DISABLE_SERVER_SIDE_CURSORS=True`. Split Redis (`REDIS_CACHE_URL` → `CACHES`, `REDIS_BROKER_URL` → Celery). DRF: `TokenAuthentication` + `SessionAuthentication` always; `JWTAuthentication` prepended **only when `VEDA_JWT_AUTH=1`**. Global throttles (`AnonRateThrottle` 60/min, `UserRateThrottle` 240/min) + scoped (`login` 10/min, `token_refresh` 60/min, `password_change` 5/min). `SIMPLE_JWT` (15-min access / 7-day refresh, rotation + blacklist, `CHECK_REVOKE_TOKEN`, HS256). Feature flags: `VEDA_JWT_AUTH`, `VEDA_RBAC_MODE`, `VEDA_ALLOW_ANONYMOUS`, `VEDA_AUTO_SYNC_CATALOG`, `SOURCE_PROFILER_ENABLED`, `SOURCE_ITEM_PROFILER_ENABLED`. `AUTH_PASSWORD_VALIDATORS` (4 Django + `PasswordComplexityValidator`). `VEDA = build_veda_settings()`. |
| `settings/dev.py` | `DEBUG=True`, `ALLOWED_HOSTS=["*"]`, **sqlite fallback** when no `VEDA_DB_HOST` / `PGBOUNCER_HOST`, `SLM_BACKEND=ollama`, CORS/CSRF for `:4001` / `:8080`. |
| `settings/prod.py` | `DEBUG=False`, **refuses boot** if `VEDA_JWT_AUTH` and `SECRET_KEY == INSECURE_DEV_SECRET_KEY`, HSTS / SSL redirect / secure cookies, `SLM_BACKEND=vllm`. |
| `celery.py` | `Celery("veda")`, queues `ingestion` / `high` / `default`, `autodiscover_tasks()`. |
| `urls.py` | `/admin/`; `/api/v1/` includes **five** urlconfs (`apps.query`, `apps.chat`, `apps.authentication`, `apps.access_management`, `apps.sources`); `/healthz` (inline), `/readyz`, `/metrics` (`apps.core.views`). |
| `asgi.py` / `wsgi.py` | Django entrypoints — **not** the FastAPI `inference` app. |

## Gotchas
- **`config` (this package) ≠ the engine's top-level `config` module** (`veda_core/config.py`).
  Ingestion runs in a subprocess with `cwd=veda_core` to keep them apart.
- The dev Redis defaults collide on `localhost:6379` (`/0` vs `/1`); prod uses separate hosts.
- Engine flags are **not** duplicated in Django settings — they reach Django only through
  `apps/core/settings_bridge.build_veda_settings()`.
- **Secrets: `prod.py` declares Docker secrets but `settings` never reads `/run/secrets`**
  (`PRODUCTION_READINESS_PLAN.md` B1).
