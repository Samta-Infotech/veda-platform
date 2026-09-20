"""storage_adapters.env_drift — startup check: log the EFFECTIVE values of the env keys
that have bitten this deployment, and warn loudly when they differ from `.env`.

Why (2026-09-15, M1 close-out item 5): `docker compose restart` does not re-read `.env`
(env is fixed at container CREATE time), so a container created before a `.env` edit keeps
running with the old value silently. This bit twice in one week: the `ingest-worker` was
created with a stale `METAL_EMBED_URL` (every embed call waited out a 60 s timeout → a
197 s ingest that takes ≈30 s), and earlier the inference container had the same stale IP.
Nothing surfaced it — the fallback "worked". Now every worker and the inference replica log
what they are actually running with, and flag drift at startup.

Importable from both tiers (no Django, no veda_core). Never raises.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

# OLLAMA_URL is deliberately NOT here: docker-compose.yml sets it explicitly for
# `inference` and `ingest-worker` (host Metal-GPU ollama), so `.env` is not its authority
# for those services and comparing against it produced a standing false "drift"
# (2026-09-16). Compose-level overrides can't be read from inside the container.
KEYS = ("METAL_EMBED_URL", "SLM_MODEL_NAME", "SLM_TEMPERATURE")


def _read_dotenv(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def check_env_drift(role: str, dotenv_path: Optional[str] = None,
                    log: Optional[logging.Logger] = None) -> Dict[str, dict]:
    """Log effective values for KEYS and warn on drift vs `.env`. Returns
    {key: {"effective": ..., "dotenv": ..., "drift": bool}} for tests/health."""
    log = log or logging.getLogger("veda.env_drift")
    dotenv_path = dotenv_path or os.path.join(os.environ.get("VEDA_APP_DIR", "/app"), ".env")
    dotenv = _read_dotenv(dotenv_path)
    report: Dict[str, dict] = {}
    for k in KEYS:
        eff, want = os.environ.get(k), dotenv.get(k)
        drift = (want is not None and eff != want)
        report[k] = {"effective": eff, "dotenv": want, "drift": drift}
    log.info("[%s] effective env: %s", role,
             ", ".join(f"{k}={report[k]['effective']!r}" for k in KEYS))
    drifted = [k for k in KEYS if report[k]["drift"]]
    if drifted:
        for k in drifted:
            log.warning("[%s] ENV DRIFT: %s is %r in this process but %r in %s — this container "
                        "was created before .env changed; `docker compose up -d <service>` to "
                        "recreate it (a plain `restart` will NOT pick up .env)",
                        role, k, report[k]["effective"], report[k]["dotenv"], dotenv_path)
        try:
            print(f"  [env-drift] {role}: {', '.join(drifted)} differ from .env — recreate the "
                  f"container with `docker compose up -d`", flush=True)
        except Exception:
            pass
    return report
