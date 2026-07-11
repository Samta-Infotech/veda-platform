"""chatbot.llm — backend-aware SLM caller (Ollama | vLLM) for the supervisor's
classify/smalltalk/follow-up steps.

Deliberately a SEPARATE, minimal implementation from veda_core/slm/_call_slm.py
— NOT an import of it. veda_core/slm/_call_slm.py lives under veda_core/, which
the api tier must never import directly (same boundary that makes
apps/query/inference_client.py a standalone HTTP client rather than a veda_core
import — see that module's own docstring). chatbot/ now runs inside the api
container's process (imported by apps/chat/services.py), so it is subject to
that same rule. This is deliberate, mild duplication of _call_slm.py's two
backend request shapes to preserve the boundary — not an oversight.

SLM_BACKEND ("ollama" default | "vllm") and OLLAMA_URL / VLLM_URL match the env
var names already used by veda_core/slm/_call_slm.py, so ops only ever sets
these once per service.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

SLM_BACKEND = os.environ.get("SLM_BACKEND", "ollama").strip().lower()
SLM_MODEL_NAME = os.environ.get("SLM_MODEL_NAME", "qwen2.5-coder:7b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1").rstrip("/")

# Classify/smalltalk/follow-up-resolution are conversational judgment calls
# (is this smalltalk? does it depend on history? rewrite it as a standalone
# question?) — NOT code generation, so they default to a small INSTRUCT model
# instead of SLM_MODEL_NAME's 7B CODER model, same reasoning as veda_core's
# NL_SUMMARY_MODEL (query/result_explainer.py). Falls back to SLM_MODEL_NAME
# if unset, so an untouched deployment behaves exactly as before.
CHATBOT_CLASSIFY_MODEL = os.environ.get("CHATBOT_CLASSIFY_MODEL") or SLM_MODEL_NAME


def _post_json(url: str, payload: dict, timeout: int) -> dict | None:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        # OSError covers both urllib.error.URLError (connect failures) and the bare
        # TimeoutError a socket read timeout raises (not wrapped in URLError) — callers
        # rely on None-on-any-failure to fall back gracefully (refuse-over-guess).
        logger.warning("chatbot.llm: request to %s failed: %s", url, exc)
        return None


def _call_ollama(system: str, user: str, *, temperature: float, max_tokens: int, timeout: int,
                 model: str = SLM_MODEL_NAME) -> str | None:
    """Matches veda_core/slm/_call_slm.py::OllamaBackend.call()'s /api/chat shape."""
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    body = _post_json(f"{OLLAMA_URL}/api/chat", payload, timeout)
    if body is None:
        return None
    return (body.get("message") or {}).get("content", "").strip() or None


def _call_vllm(system: str, user: str, *, temperature: float, max_tokens: int, timeout: int,
               model: str = SLM_MODEL_NAME) -> str | None:
    """Matches veda_core/slm/_call_slm.py::VLLMBackend.call()'s OpenAI-compatible
    shape. VLLM_URL may or may not carry a trailing /v1 (docker-compose.prod.yml
    sets VLLM_URL=http://vllm:8000/v1) — normalize before appending the path."""
    base = VLLM_URL[: -len("/v1")] if VLLM_URL.endswith("/v1") else VLLM_URL
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    body = _post_json(f"{base}/v1/chat/completions", payload, timeout)
    if body is None:
        return None
    try:
        return (body["choices"][0]["message"]["content"] or "").strip() or None
    except (KeyError, IndexError, TypeError):
        logger.warning("chatbot.llm: unexpected vLLM response shape: %r", body)
        return None


def call_slm(system: str, user: str, *, temperature: float = 0.1,
             max_tokens: int = 200, timeout: int = 20, model: str = SLM_MODEL_NAME) -> str | None:
    """One-shot chat completion routed to Ollama or vLLM per SLM_BACKEND.
    Returns None on any failure — callers must have a deterministic fallback
    (refuse-over-guess, same as the rest of this codebase). `model` defaults to
    SLM_MODEL_NAME (unchanged behavior); classify/smalltalk/follow-up call
    sites in nodes.py pass CHATBOT_CLASSIFY_MODEL instead."""
    fn = _call_vllm if SLM_BACKEND == "vllm" else _call_ollama
    return fn(system, user, temperature=temperature, max_tokens=max_tokens, timeout=timeout, model=model)
