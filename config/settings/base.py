"""Shared Django settings (migration_plan.md §4, §9, §9a).

Infra values (DB, Redis, secrets) come from the environment — 12-factor,
per §9a. Engine flags (ENCODER_MODE, TOP_K, SLM_*, HNSW_*, ...) come from
`veda_core.config` through the settings bridge (§0.3/§9) so config.py stays
the single source of truth; they are never duplicated here.

Both Django DB aliases and every raw pool are expected to dial PgBouncer's
port, not Postgres directly (§1.1), so N workers × M replicas × pool_size
cannot exceed Postgres max_connections. Redis is split into two instances
(§1.2): `redis-broker` (Celery broker/result backend, unbounded, no
eviction) and `redis-cache` (Django cache + hot substrate indices +
rehydrate pub/sub, allkeys-lru).
"""
import os
from datetime import timedelta
from pathlib import Path

from apps.core.settings_bridge import build_veda_settings

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Named (and not underscore-prefixed) so prod.py can import it and refuse to boot
# on this value — it is also the JWT signing key, and a well-known signing key
# means anyone can forge a token.
INSECURE_DEV_SECRET_KEY = "insecure-dev-key-do-not-use-in-prod"

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", INSECURE_DEV_SECRET_KEY)

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "rest_framework.authtoken",  # token auth (§6.2)
    # JWT refresh-token state: OutstandingToken + BlacklistedToken. Installed
    # UNCONDITIONALLY (not behind VEDA_JWT_AUTH) so migrations are identical in
    # every environment regardless of the flag — installing it changes no
    # behaviour on its own; only the flag does. Its unique constraint on a
    # blacklisted jti is what makes refresh-token rotation race-safe
    # (apps/authentication/services.py).
    "rest_framework_simplejwt.token_blacklist",
    "corsheaders",
    "apps.core",
    "apps.sources",
    "apps.substrate",
    "apps.ingestion",
    "apps.query",
    "apps.evaluation",
    "apps.chat",
    # Identity: verification (authentication) vs administration + authorization
    # (access_management) are separate bounded contexts on purpose.
    "apps.authentication",
    "apps.access_management",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "apps.core.middleware.RequestIdMiddleware",  # §6.3 request-id propagation
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# PgBouncer host/port in front of Postgres — never dial Postgres directly (§1.1).
_PGBOUNCER_HOST = os.environ.get("PGBOUNCER_HOST", "localhost")
_PGBOUNCER_PORT = os.environ.get("PGBOUNCER_PORT", "6432")

# Credentials come from the same POSTGRES_* env the postgres/pgbouncer services use
# (.env); VEDA_DB_* still override if set, so a separate app role can be swapped in.
_DB_NAME = os.environ.get("VEDA_DB_NAME", os.environ.get("POSTGRES_DB", "veda"))
_DB_USER = os.environ.get("VEDA_DB_USER", os.environ.get("POSTGRES_USER", "veda"))
_DB_PASSWORD = os.environ.get("VEDA_DB_PASSWORD", os.environ.get("POSTGRES_PASSWORD", ""))

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": _DB_NAME,
        "USER": _DB_USER,
        "PASSWORD": _DB_PASSWORD,
        "HOST": _PGBOUNCER_HOST,
        "PORT": _PGBOUNCER_PORT,
    },
    # Optional separate source-registry DB (§5); defaults to the same DB unless
    # SOURCE_REGISTRY_DB_NAME is set, so dev needs no second database.
    "source_registry": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("SOURCE_REGISTRY_DB_NAME", _DB_NAME),
        "USER": os.environ.get("SOURCE_REGISTRY_DB_USER", _DB_USER),
        "PASSWORD": os.environ.get("SOURCE_REGISTRY_DB_PASSWORD", _DB_PASSWORD),
        "HOST": _PGBOUNCER_HOST,
        "PORT": _PGBOUNCER_PORT,
    },
}

# redis-cache: Django cache + hot substrate indices + rehydrate pub/sub (§1.2).
_REDIS_CACHE_URL = os.environ.get("REDIS_CACHE_URL", "redis://localhost:6379/0")
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": _REDIS_CACHE_URL,
    },
}

# redis-broker: Celery broker + result backend ONLY, unbounded, no eviction (§1.2).
_REDIS_BROKER_URL = os.environ.get("REDIS_BROKER_URL", "redis://localhost:6379/1")
CELERY_BROKER_URL = _REDIS_BROKER_URL
CELERY_RESULT_BACKEND = _REDIS_BROKER_URL
CELERY_TASK_DEFAULT_QUEUE = "default"

# Behind transaction-pooling PgBouncer, server-side cursors don't survive across
# transactions (§1.1). Disable them so .iterator()/large querysets stay correct.
DATABASES["default"]["DISABLE_SERVER_SIDE_CURSORS"] = True
DATABASES["source_registry"]["DISABLE_SERVER_SIDE_CURSORS"] = True

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
    # Composition (upper/lower/digit/special) the four stock validators above don't
    # check. Thresholds are OPTIONS, not code — change the policy here, not in
    # apps/authentication/password_validators.py.
    {"NAME": "apps.authentication.password_validators.PasswordComplexityValidator",
     "OPTIONS": {"min_uppercase": 1, "min_lowercase": 1, "min_digits": 1, "min_special": 1}},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"  # leading slash → absolute URLs so {% static %} works on nested /admin/ pages
STATIC_ROOT = BASE_DIR / "staticfiles"  # collectstatic target (api entrypoint); served by nginx
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# JWT authentication rollout flag (apps.authentication). Default OFF: with it off,
# POST /api/v1/auth/login returns the same placeholder-token payload the pre-JWT
# view returned and no JWT authentication class is installed, so every existing
# endpoint behaves byte-identically. Switching it to "1" turns on real signed
# tokens, rotation and revocation. The refresh/logout endpoints are routed either
# way — with the flag off no real token is ever issued, so refresh has nothing
# valid to rotate (401) and logout has nothing to revoke (idempotent 200).
VEDA_JWT_AUTH = os.environ.get("VEDA_JWT_AUTH", "0") == "1"

# RBAC enforcement mode (apps.access_management.gate). Default OFF: with no value
# set, apps.access_management.gate.rbac_mode() falls back to "off" regardless of
# what this env var says, because that fallback triggers on an ABSENT Django
# setting attribute, not on an absent env var — this line is what actually wires
# the two together. Found missing entirely (2026-08-08) during live/manual
# Postman testing: VEDA_RBAC_MODE was set to "enforce" in the environment, but
# rbac_mode() reported "off" regardless, because there was no code anywhere that
# copied the env var onto a Django setting — every unit/integration test in this
# whole RBAC programme set the mode via Django's own override_settings(), which
# bypasses this exact wiring gap, so none of them (or three rounds of code
# review) caught that a real deployment's env var was silently a no-op.
VEDA_RBAC_MODE = os.environ.get("VEDA_RBAC_MODE", "off")

# Catalog auto-sync on ingestion success (apps.ingestion.tasks). Default OFF: with
# it off, task_ingest_source behaves byte-identically to before this flag existed —
# CatalogDiscoveryService is never called, and the catalog stays stale until an
# operator runs `manage.py sync_catalog` by hand, exactly as today. Switching it to
# "1" reconciles the just-ingested source's resources automatically, right after
# `Source.ready` flips — closing the window where a freshly (re-)ingested table is
# absent from the catalog, and therefore denied, until someone remembers to sync it.
# A sync failure is logged and swallowed either way: the ingestion job that just
# succeeded must never be turned into a failure by a catalog-projection problem.
VEDA_AUTO_SYNC_CATALOG = os.environ.get("VEDA_AUTO_SYNC_CATALOG", "0") == "1"

# Multi-source routing (docs/multisource_routing/, Phase 1.3): after a source is ingested,
# auto-generate a grounded `Source.description` from its observed substrate schema, for the
# query-time routing coordinator. Default OFF — additive and consumed only by the (not-yet-live)
# coordinator, but gated per the "prod stays byte-identical until opted in" convention. The hook
# is never allowed to fail an ingestion job (mirrors VEDA_AUTO_SYNC_CATALOG).
SOURCE_PROFILER_ENABLED = os.environ.get("SOURCE_PROFILER_ENABLED", "0") == "1"

# Multi-source routing (docs/multisource_routing/): after ingestion, build the uniform SourceItem
# layer (tables/documents/datasets) and profile each item — SLM one-line summary + BGE-M3 embedding
# into `source_item_embeddings`, the query-time routing PRIOR. Default OFF; never fails an ingestion
# job (mirrors SOURCE_PROFILER_ENABLED). Needs the SLM chat + Metal embed endpoints reachable.
SOURCE_ITEM_PROFILER_ENABLED = os.environ.get("SOURCE_ITEM_PROFILER_ENABLED", "0") == "1"

# Login lockout (apps.authentication.services). TWO counters, deliberately:
#
#   * VEDA_AUTH_LOGIN_MAX_FAILURES — per (account, client-IP). Trips a HARD refusal
#     before any password hash is computed. Bounds one source hammering one account
#     and cannot be used to lock a legitimate user out, because an attacker cannot
#     make failures appear against the victim's IP.
#   * VEDA_AUTH_LOGIN_ACCOUNT_MAX_FAILURES — account-wide across all IPs, so
#     credential stuffing spread over many addresses is still visible and rate-
#     limited. Deliberately SOFT: it turns a wrong guess into a 429 but NEVER
#     refuses a correct password, so it cannot be weaponised into a denial of
#     service against a real user. Set higher than the per-source threshold.
VEDA_AUTH_LOGIN_MAX_FAILURES = int(os.environ.get("VEDA_AUTH_LOGIN_MAX_FAILURES", "10"))
VEDA_AUTH_LOGIN_ACCOUNT_MAX_FAILURES = int(
    os.environ.get("VEDA_AUTH_LOGIN_ACCOUNT_MAX_FAILURES", "50"))
VEDA_AUTH_LOGIN_LOCKOUT_SECONDS = int(os.environ.get("VEDA_AUTH_LOGIN_LOCKOUT_SECONDS", "300"))

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    # Token auth available; tenant is resolved from the authenticated principal (§6.2).
    # JWTAuthentication is prepended only when VEDA_JWT_AUTH is on, so the flag-off
    # path leaves this list exactly as it was. Token/Session auth stay in place
    # either way — the admin site and any existing token client keep working.
    "DEFAULT_AUTHENTICATION_CLASSES": (
        ["rest_framework_simplejwt.authentication.JWTAuthentication"] if VEDA_JWT_AUTH else []
    ) + [
        "rest_framework.authentication.TokenAuthentication",
        "rest_framework.authentication.SessionAuthentication",
    ],
    # Throttling (§6.2). nginx also rate-limits at the edge.
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.UserRateThrottle",
    ],
    # "login"/"token_refresh" are ScopedRateThrottle scopes used by the auth views,
    # deliberately tighter than "anon": these are the only endpoints where an
    # unauthenticated caller can guess a secret or replay a captured token.
    "DEFAULT_THROTTLE_RATES": {
        "anon": "60/min", "user": "240/min",
        "login": "10/min", "token_refresh": "60/min",
        # password/change is the one AUTHENTICATED endpoint where a caller can
        # guess a secret (``current_password``) repeatedly — a stolen access token
        # is worth more if it can also brute-force its way to a full account
        # takeover before it expires. Tighter than the general "user" rate for
        # exactly that reason.
        "password_change": "5/min",
    },
    # How many reverse proxies sit in front of the api, so DRF (and the login
    # lockout, which reuses its BaseThrottle.get_ident) reads the real client IP
    # from X-Forwarded-For instead of trusting the whole header.
    #
    # nginx.conf sends `X-Forwarded-For $proxy_add_x_forwarded_for`, appending the
    # true peer as the LAST entry; with NUM_PROXIES=1 DRF takes exactly that entry,
    # so a client that pre-seeds the header with fake addresses cannot mint itself
    # fresh throttle/lockout quota. Left unset (the DRF default) the ENTIRE header
    # is used as the identity, which is fully attacker-controlled.
    #
    # This MUST match the real proxy depth: too low reads a spoofed entry, too high
    # reads the shared proxy address and groups unrelated clients together. It also
    # means the api tier must never be exposed to clients directly, bypassing nginx.
    "NUM_PROXIES": int(os.environ.get("VEDA_NUM_PROXIES", "1")),
}

# JWT parameters (rest_framework_simplejwt). Short access lifetime + rotating,
# blacklist-on-rotation refresh tokens: a leaked access token is useful for
# minutes, and a leaked refresh token can be spent exactly once before rotation
# detection revokes the account's tokens (apps/authentication/services.py).
#
# SIGNING_KEY is Django's SECRET_KEY. base.py's dev default is a well-known
# string, so a deployment that serves JWTs without setting DJANGO_SECRET_KEY would
# be signing forgeable tokens — prod.py refuses to boot in that state.
SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(
        minutes=int(os.environ.get("VEDA_JWT_ACCESS_MINUTES", "15"))),
    "REFRESH_TOKEN_LIFETIME": timedelta(
        days=int(os.environ.get("VEDA_JWT_REFRESH_DAYS", "7"))),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "ALGORITHM": "HS256",
    "SIGNING_KEY": SECRET_KEY,
    "AUTH_HEADER_TYPES": ("Bearer",),
    "USER_ID_FIELD": "id",
    "USER_ID_CLAIM": "user_id",
    # Bind every token to the password it was issued under (an md5 of the stored
    # hash travels as a claim), so changing a password invalidates the tokens
    # minted before it. Without this, a user who changes their password because
    # they suspect compromise leaves the attacker a working refresh token for its
    # full lifetime. simplejwt enforces this claim for ACCESS tokens only, inside
    # JWTAuthentication.get_user — the refresh path checks it explicitly in
    # apps/authentication/services.py::AuthService._password_unchanged.
    "CHECK_REVOKE_TOKEN": True,
    # No clock skew allowance: api and clients are not distributed peers here, and
    # leeway only widens the window in which an expired token still works.
    "LEEWAY": 0,
}

# Dev convenience: allow anonymous queries (tenant defaults). Set to "0" in prod to
# require a token and derive tenant from the principal (§6.2).
VEDA_ALLOW_ANONYMOUS = os.environ.get("VEDA_ALLOW_ANONYMOUS", "1") == "1"

# Base CORS/CSRF trusted origins; not CORS_ALLOW_ALL_ORIGINS, so env-specific
# settings (dev.py/prod.py) should extend these with += rather than replace them.
CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

CSRF_TRUSTED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# Engine flags bridged from veda_core.config — the single source of truth (§9).
VEDA = build_veda_settings()
