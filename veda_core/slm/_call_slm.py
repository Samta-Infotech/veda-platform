# =============================================================================
# slm/_call_slm.py — SLM backend seam (Strategy) — RESTORED (review Finding 1)
#
# The single place the engine talks to a local SLM. Every query-time and
# ingestion-time call site routes through call_slm(); the backend is selected by
# SLM_BACKEND ("ollama" | "vllm") and cached per process.
#
#   OllamaBackend — POST {OLLAMA_URL}/api/chat  (or /api/generate for raw-prompt
#                   sites), keep_alive="24h" so the model stays resident.
#   VLLMBackend   — POST {VLLM_URL}/v1/chat/completions (OpenAI-compatible).
#                   Raw-prompt calls are mapped to a single-user-message chat.
#
# Contract (identical for both backends):
#   call_slm(user_message, *, system=None, purpose="general", timeout=None,
#            temperature=0.0, num_predict=None, num_ctx=None, seed=None,
#            json_format=False, endpoint="chat", model=None) -> str
#   • returns the assistant content as a plain string
#   • raises RuntimeError with a uniform "SLM unreachable/invalid" message on
#     network or API errors — call sites keep their existing degrade behaviour
#     (some catch and fall back, some propagate; that contract is unchanged).
#
# Zero-egress: both backends live on the internal network (OLLAMA_URL / VLLM_URL);
# nothing here dials outside the deployment.
# =============================================================================
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from typing import Optional


# ---------------------------------------------------------------------------
# Circuit breaker — intentionally a pass-through hook (parity with the reviewed
# engine: introducing trip/cooldown semantics would change SLM-path behaviour;
# wire real state in here later behind its own flag + parity run).
# ---------------------------------------------------------------------------
class _slm_circuit_breaker:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False  # never swallow — call sites own their degrade behaviour


def _config():
    """Lazy config import (engine's top-level config.py) with env fallbacks, so
    this module also imports cleanly outside the engine cwd (e.g. unit tests)."""
    try:
        import config as _cfg
    except Exception:  # pragma: no cover - config always present in the engine
        _cfg = None

    def g(name, default, *env_aliases):
        for env_name in (name,) + env_aliases:
            env = os.environ.get(env_name)
            if env not in (None, ""):
                return env
        return getattr(_cfg, name, default) if _cfg is not None else default

    return {
        "backend": str(g("SLM_BACKEND", "ollama")).strip().lower(),
        "model": g("SLM_MODEL_NAME", "qwen2.5-coder:7b"),
        "ollama_url": g("SLM_OLLAMA_BASE_URL", "http://localhost:11434", "OLLAMA_URL"),
        "vllm_url": g("VLLM_BASE_URL", "http://vllm:8000", "VLLM_URL"),
        # vLLM serves under the model's HF path, not the Ollama tag — overridable.
        "vllm_model": g("VLLM_MODEL_NAME", None),
        "timeout": int(g("SLM_TIMEOUT_SECS", 240)),
    }


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"SLM unreachable at {url}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"SLM returned non-JSON from {url}: {raw[:300]}") from exc


class OllamaBackend:
    """Ollama /api/chat (+ /api/generate for raw-prompt sites). Byte-parity with
    the previous direct call sites: same payload shape, keep_alive 24h."""

    name = "ollama"

    def __init__(self, base_url: str, model: str, default_timeout: int):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.default_timeout = default_timeout

    def call(self, user_message, *, system=None, timeout=None, temperature=0.0,
             num_predict=None, num_ctx=None, seed=None, json_format=False,
             endpoint="chat", model=None) -> str:
        timeout = timeout or self.default_timeout
        options = {"temperature": temperature}
        if num_predict is not None:
            options["num_predict"] = num_predict
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if seed is not None:
            options["seed"] = seed

        if endpoint == "generate":
            payload = {"model": model or self.model, "prompt": user_message,
                       "stream": False, "keep_alive": "24h", "options": options}
            body = _post_json(f"{self.base_url}/api/generate", payload, timeout)
            return (body.get("response") or "").strip()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})
        payload = {"model": model or self.model, "stream": False,
                   "keep_alive": "24h", "messages": messages, "options": options}
        if json_format:
            payload["format"] = "json"
        body = _post_json(f"{self.base_url}/api/chat", payload, timeout)
        content = (body.get("message", {}) or {}).get("content", "")
        if not content:
            raise RuntimeError(
                f"SLM returned empty content from {self.base_url}/api/chat: "
                f"{json.dumps(body)[:300]}")
        return content


class VLLMBackend:
    """vLLM OpenAI-compatible /v1/chat/completions. Raw-prompt ("generate") calls
    are mapped to a single-user-message chat — equivalent for these prompts."""

    name = "vllm"

    def __init__(self, base_url: str, model: str, default_timeout: int):
        # Accept both conventions: "http://vllm:8000" and "http://vllm:8000/v1"
        # (the prod compose passes the latter) — the endpoint path is appended here.
        u = base_url.rstrip("/")
        if u.endswith("/v1"):
            u = u[: -len("/v1")].rstrip("/")
        self.base_url = u
        self.model = model
        self.default_timeout = default_timeout

    def call(self, user_message, *, system=None, timeout=None, temperature=0.0,
             num_predict=None, num_ctx=None, seed=None, json_format=False,
             endpoint="chat", model=None) -> str:
        timeout = timeout or self.default_timeout
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})
        payload = {"model": model or self.model, "messages": messages,
                   "temperature": temperature, "stream": False}
        if num_predict is not None:
            payload["max_tokens"] = num_predict
        if seed is not None:
            payload["seed"] = seed
        if json_format:
            payload["response_format"] = {"type": "json_object"}
        # num_ctx is an Ollama runtime knob; vLLM's context is fixed at serve time.
        body = _post_json(f"{self.base_url}/v1/chat/completions", payload, timeout)
        try:
            content = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"SLM returned unexpected shape from {self.base_url}: "
                f"{json.dumps(body)[:300]}") from exc
        content = content.strip()
        if not content:
            raise RuntimeError(f"SLM returned empty content from {self.base_url}")
        return content


_BACKEND = {"v": None}
_LOCK = threading.Lock()


def get_backend():
    """Backend chosen by SLM_BACKEND, cached per process (thread-safe)."""
    if _BACKEND["v"] is None:
        with _LOCK:
            if _BACKEND["v"] is None:
                cfg = _config()
                if cfg["backend"] == "vllm":
                    _BACKEND["v"] = VLLMBackend(
                        cfg["vllm_url"], cfg["vllm_model"] or cfg["model"],
                        cfg["timeout"])
                else:
                    _BACKEND["v"] = OllamaBackend(
                        cfg["ollama_url"], cfg["model"], cfg["timeout"])
    return _BACKEND["v"]


def reset_backend():
    """Test/ops hook: drop the cached backend (e.g. after flipping SLM_BACKEND)."""
    with _LOCK:
        _BACKEND["v"] = None


def call_slm(user_message: str, *, system: Optional[str] = None,
             purpose: str = "general", timeout: Optional[int] = None,
             temperature: float = 0.0, num_predict: Optional[int] = None,
             num_ctx: Optional[int] = None, seed: Optional[int] = None,
             json_format: bool = False, endpoint: str = "chat",
             model: Optional[str] = None) -> str:
    """The one SLM entry point. `purpose` is a label for tracing/metrics only
    (e.g. "ir_emit", "nl_answer", "decompose") — it never changes routing."""
    with _slm_circuit_breaker():
        return get_backend().call(
            user_message, system=system, timeout=timeout, temperature=temperature,
            num_predict=num_predict, num_ctx=num_ctx, seed=seed,
            json_format=json_format, endpoint=endpoint, model=model)


def prewarm(model: Optional[str] = None, timeout: int = 120) -> None:
    """Best-effort model prewarm (Ollama: loads + pins via keep_alive; vLLM: a
    1-token completion touches the weights). Never raises."""
    try:
        call_slm("ok", purpose="prewarm", timeout=timeout, num_predict=1, model=model)
    except Exception:
        pass
