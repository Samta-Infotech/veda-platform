"""Settings bridge — config.py stays the single source of truth (migration_plan.md §0.3, §9).

Django settings never hardcode engine flags; they read `veda_core.config`
and allow an env override per flag. `config.py` values win over the
fallback defaults below, and `os.environ` wins over `config.py` — this
lets ops flip a flag per-deployment without editing the library.

Import of `veda_core.config` is guarded: a missing/renamed attribute (or
the module itself, e.g. during isolated unit tests) degrades to the
fallback default instead of hard-crashing settings load.
"""
import os
from typing import Any

try:
    import veda_core.config as _cfg
except ImportError:
    _cfg = None

# (attribute name, fallback default, caster) — §9 env-overridable engine flags.
# SLM_BACKEND and the HNSW_* params are NEW (§8b, §7.1a) and not yet in
# config.py; their fallbacks here are the pre-tuning defaults.
_FLAGS: list[tuple[str, Any, type]] = [
    ("ENCODER_MODE", "ensemble", str),
    ("TOP_K", 15, int),
    ("TOP_K_TO_LLM", 6, int),
    ("QUERY_ROUTER_ENABLED", True, bool),
    ("SLM_MODEL_NAME", "qwen2.5-coder:7b", str),
    ("SLM_BACKEND", "ollama", str),
    ("IR_JOIN_FREE_ENABLED", True, bool),
    ("FAST_PATH_ENABLED", True, bool),
    ("QUERY_DECOMPOSE_ENABLED", False, bool),
    ("HNSW_M", 16, int),
    ("HNSW_EF_CONSTRUCTION", 200, int),
    ("HNSW_EF_SEARCH", 40, int),
]


def _cast(raw: str, caster: type) -> Any:
    if caster is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return caster(raw)


def build_veda_settings() -> dict:
    settings: dict = {}
    for name, fallback, caster in _FLAGS:
        value = getattr(_cfg, name, fallback) if _cfg is not None else fallback
        env_value = os.environ.get(f"VEDA_{name}")
        if env_value is not None:
            value = _cast(env_value, caster)
        settings[name] = value
    return settings
