"""veda_core/slm/_call_slm.py

Migration plan §8b — the SLM backend seam (Strategy pattern).

All SLM calls (ir_emit, decompose, rag_synthesis, nl_answer) route through the
single frozen signature `call_slm(prompt, *, purpose, timeout=240, **opts) -> str`.
Two backends implement it: `OllamaBackend` (dev + ingestion-time, on `worker`)
and `vLLMBackend` (production query-time hot path, on `inference`). The
backend is a config flag (`SLM_BACKEND`), never a code change at the call site
— call sites are rewired to `call_slm(...)` exactly as storage call sites are
rewired to `storage_adapters` (Phase 3 pattern).

Preserved semantics (do not drop these when filling in bodies later):
  - 240s default timeout, matching `config.SLM_TIMEOUT_SECS` today.
  - A circuit breaker wraps every call so a saturated/unreachable backend
    degrades to the caller's deterministic fallback (row-count NL answer,
    refuse-over-guess) — never a hung request.
  - Both backends must be shape-identical: same exceptions, same return type,
    so switching `SLM_BACKEND` changes latency/throughput characteristics
    only, never output shape (§17 SLM-backend parity row).
"""

from contextlib import contextmanager
from typing import Any, Optional, Protocol

import json
import os
import urllib.error
import urllib.request

from veda_core import config

_DEFAULT_TIMEOUT = 240   # mirrors config.SLM_TIMEOUT_SECS (query/slm_layer.py)


class SLMBackend(Protocol):
    """Strategy interface every SLM backend implements."""

    def generate(self, prompt: str, *, purpose: str, timeout: int, **opts: Any) -> str: ...


class OllamaBackend:
    """HTTP client to `ollama/ollama` on `veda_net`.

    Default for local dev and for ingestion-time generation (glossary,
    synthetic pairs) called by `worker`, where throughput is not
    latency-critical. Adapted from `veda_core/query/slm_layer.py::_call_ollama`
    — the URL/model/keep_alive conventions are the same; this is kept a thin
    backend (no IR-specific prompt construction, no retry/parse logic — those
    stay in the call sites until Phase 3.7 rewires them).
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        # Env override (§9) so the container network name (ollama:11434) wins over
        # config.py's localhost default without editing the preserved library.
        self._base_url = base_url or os.environ.get("OLLAMA_URL") or config.SLM_OLLAMA_BASE_URL
        self._model = model or os.environ.get("SLM_MODEL_NAME") or config.SLM_MODEL_NAME

    def generate(self, prompt: str, *, purpose: str, timeout: int = _DEFAULT_TIMEOUT, **opts: Any) -> str:
        payload = {
            "model": self._model,
            "stream": False,
            "keep_alive": "24h",   # pin the model in memory — no per-query reload
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": opts.get("temperature", config.SLM_TEMPERATURE),
                "num_predict": opts.get("num_predict", config.SLM_MAX_TOKENS),
                "num_ctx": opts.get("num_ctx", config.SLM_NUM_CTX),
            },
        }
        url = f"{self._base_url.rstrip('/')}/api/chat"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"[{purpose}] Ollama unreachable at {url}: {exc}") from exc

        body = json.loads(raw)
        content = body.get("message", {}).get("content", "")
        if not content:
            raise RuntimeError(f"[{purpose}] Ollama returned empty content: {raw[:300]}")
        return content


class vLLMBackend:
    """OpenAI-compatible client to a vLLM server (§3 SLM backend placement).

    Production query-time backend on `inference`. Continuous batching serves
    concurrent SLM calls from N replicas without the single-instance
    serialization Ollama imposes. Skeleton only — real body wired in Phase 5.4
    once the vLLM service is stood up in compose.
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self._base_url = (
            base_url
            or os.environ.get("VLLM_URL")
            or getattr(config, "VLLM_BASE_URL", "http://vllm:8000/v1")
        )
        self._model = model or os.environ.get("SLM_MODEL_NAME") or config.SLM_MODEL_NAME

    def generate(self, prompt: str, *, purpose: str, timeout: int = _DEFAULT_TIMEOUT, **opts: Any) -> str:
        # OpenAI-compatible chat completions — vLLM's continuous batching serves
        # concurrent SLM calls from N inference replicas (§3, §8b). Shape-identical
        # return to OllamaBackend (a plain string), so SLM_BACKEND is a pure flag.
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": opts.get("temperature", config.SLM_TEMPERATURE),
            "max_tokens": opts.get("num_predict", config.SLM_MAX_TOKENS),
            "stream": False,
        }
        url = f"{self._base_url.rstrip('/')}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"[{purpose}] vLLM unreachable at {url}: {exc}") from exc

        body = json.loads(raw)
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"[{purpose}] vLLM unexpected response: {raw[:300]}") from exc
        if not content:
            raise RuntimeError(f"[{purpose}] vLLM returned empty content: {raw[:300]}")
        return content


_backend_cache: Optional[SLMBackend] = None


def _get_backend() -> SLMBackend:
    """Select the backend by `SLM_BACKEND` from `veda_core.config`, cached per process.

    `worker` defaults to `ollama`; `inference` defaults to `vllm` in prod,
    `ollama` in dev (§8b). `SLM_BACKEND` is not yet declared in config.py —
    default to "ollama" until Phase 3.7/9 adds the setting.
    """
    global _backend_cache
    if _backend_cache is not None:
        return _backend_cache

    backend_name = getattr(config, "SLM_BACKEND", "ollama")
    if backend_name == "vllm":
        _backend_cache = vLLMBackend()
    else:
        _backend_cache = OllamaBackend()
    return _backend_cache


@contextmanager
def _slm_circuit_breaker(purpose: str):
    """Skeleton circuit breaker preserving the existing intent: trip on
    repeated failures for `purpose` and fail fast rather than piling up
    hung requests against a saturated/unreachable backend. Real trip-state
    tracking (failure counts, open/half-open/closed, per-purpose keys) is
    wired in Phase 3.7 alongside the call-site rewire; today this is a
    pass-through so the interface shape is frozen ahead of that work.
    """
    yield


def call_slm(prompt: str, *, purpose: str, timeout: int = _DEFAULT_TIMEOUT, **opts: Any) -> str:
    """The single choke point every SLM call site routes through (§8b).

    Frozen signature — do not add positional args. `purpose` identifies the
    call site (e.g. "ir_emit", "decompose", "rag_synthesis", "nl_answer") for
    breaker bookkeeping and logging; it is not sent to the model.
    """
    backend = _get_backend()
    with _slm_circuit_breaker(purpose):
        return backend.generate(prompt, purpose=purpose, timeout=timeout, **opts)
