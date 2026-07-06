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
from pathlib import Path

from apps.core.settings_bridge import build_veda_settings

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "insecure-dev-key-do-not-use-in-prod")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "rest_framework.authtoken",  # token auth (§6.2)
    "apps.core",
    "apps.sources",
    "apps.substrate",
    "apps.ingestion",
    "apps.query",
    "apps.evaluation",
    "apps.chat",
]

MIDDLEWARE = [
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
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"  # collectstatic target (api entrypoint)
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    # Token auth available; tenant is resolved from the authenticated principal (§6.2).
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.TokenAuthentication",
        "rest_framework.authentication.SessionAuthentication",
    ],
    # Throttling (§6.2). nginx also rate-limits at the edge.
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.UserRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {"anon": "60/min", "user": "240/min"},
}

# Dev convenience: allow anonymous queries (tenant defaults). Set to "0" in prod to
# require a token and derive tenant from the principal (§6.2).
VEDA_ALLOW_ANONYMOUS = os.environ.get("VEDA_ALLOW_ANONYMOUS", "1") == "1"

# Engine flags bridged from veda_core.config — the single source of truth (§9).
VEDA = build_veda_settings()
