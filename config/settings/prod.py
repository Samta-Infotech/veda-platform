"""Production settings (migration_plan.md §1, §3, §9a).

SLM_BACKEND defaults to vllm — the production query-time hot path (§8b) —
since a single Ollama instance would serialize SLM calls across the fleet.
"""
import os

from .base import *  # noqa: F401,F403

DEBUG = False
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "").split(",")

SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_CONTENT_TYPE_NOSNIFF = True

VEDA.setdefault("SLM_BACKEND", "vllm")  # noqa: F405
