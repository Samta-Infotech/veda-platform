"""Production settings (migration_plan.md §1, §3, §9a).

SLM_BACKEND defaults to vllm — the production query-time hot path (§8b) —
since a single Ollama instance would serialize SLM calls across the fleet.
"""
import os

from django.core.exceptions import ImproperlyConfigured

from .base import *  # noqa: F401,F403
from .base import INSECURE_DEV_SECRET_KEY

DEBUG = False

# SECRET_KEY doubles as the JWT signing key (SIMPLE_JWT in base.py). Serving JWTs
# signed with the well-known dev default would let anyone mint a token for any
# user, so fail at boot rather than start an api that trusts forged credentials.
if VEDA_JWT_AUTH and SECRET_KEY == INSECURE_DEV_SECRET_KEY:  # noqa: F405
    raise ImproperlyConfigured(
        "DJANGO_SECRET_KEY must be set to a real secret when VEDA_JWT_AUTH=1 — "
        "it is the JWT signing key."
    )
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "").split(",")

SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_CONTENT_TYPE_NOSNIFF = True

VEDA.setdefault("SLM_BACKEND", "vllm")  # noqa: F405
