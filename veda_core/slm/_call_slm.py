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
#
# Token usage: both backends' responses carry token counts (Ollama:
# prompt_eval_count/eval_count; vLLM: usage.{prompt,completion}_tokens).
# call_slm() records them into whatever collect_usage() scope is open (see
# below) without changing its own str return contract — wrap a query's whole
# processing in `with collect_usage() as u:` and read `u.calls()`/
# `usage_totals(u.calls())` afterwards.
# =============================================================================
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from contextvars import ContextVar
from typing import Optional


# ---------------------------------------------------------------------------
# Per-query token accounting. Both backends already RETURN token counts on
# every response (Ollama: prompt_eval_count/eval_count; vLLM: OpenAI-style
# `usage`) — this captures them into a ContextVar accumulator keyed by the
# existing `purpose` label. Purely additive: call_slm()'s contract is
# unchanged, and every step is best-effort (accounting can never fail a call).
#
# Lifecycle: veda/explain.new_trace() calls reset_usage() at query start;
# ExplainTrace.finish() reads get_usage() into the trace's llm_usage section.
# When reset_usage() was never called (standalone/ingestion use) folding is a
# no-op. ContextVar scoping: worker threads start with an empty context, so
# SLM calls made INSIDE a thread pool are not attributed (none exist on the
# query path today; context.with_context is the carry-over pattern if needed).
# ---------------------------------------------------------------------------
_USAGE_ACC: ContextVar[Optional[dict]] = ContextVar("veda_slm_usage", default=None)
_PENDING: ContextVar[Optional[tuple]] = ContextVar("veda_slm_pending", default=None)


def reset_usage() -> None:
    """Start a fresh per-query accumulator (called at query start)."""
    _USAGE_ACC.set({"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "per_purpose": {}})


def get_usage() -> Optional[dict]:
    """Deep-copied snapshot of the accumulator; None if reset_usage() never ran."""
    acc = _USAGE_ACC.get()
    return json.loads(json.dumps(acc)) if acc else None


def _note_usage(body: dict) -> None:
    """Stash this response's token counts for call_slm() to fold. Called by the
    backends right after a successful POST; shape-tolerant, never raises."""
    try:
        u = body.get("usage")
        if isinstance(u, dict):          # vLLM / OpenAI-compatible
            pt = int(u.get("prompt_tokens") or 0)
            ct = int(u.get("completion_tokens") or 0)
        else:                            # Ollama
            pt = int(body.get("prompt_eval_count") or 0)
            ct = int(body.get("eval_count") or 0)
        _PENDING.set((pt, ct))
    except Exception:
        _PENDING.set(None)


def _fold_usage(purpose: str) -> None:
    """Fold the just-noted call into the per-query accumulator (no-op when
    there is no accumulator or the backend noted nothing). NEVER raises —
    accounting must not be able to fail a query."""
    try:
        pending, acc = _PENDING.get(), _USAGE_ACC.get()
        _PENDING.set(None)
        if pending is None or acc is None:
            return
        pt, ct = pending
        acc["calls"] += 1
        acc["prompt_tokens"] += pt
        acc["completion_tokens"] += ct
        pp = acc["per_purpose"].setdefault(
            purpose, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        pp["calls"] += 1
        pp["prompt_tokens"] += pt
        pp["completion_tokens"] += ct
    except Exception:
        pass


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


# ---------------------------------------------------------------------------
# Token usage capture — one thread-local buffer per active collection scope.
# Every call_slm() records its usage here (no-op if no scope is open), and
# callers that want the numbers open a `with collect_usage() as u:` around a
# whole query. Nestable: an inner collect_usage() (e.g. a Tier-1 run_query()
# invoked from inside Tier-2's dispatch) is a harmless no-op re-entry — the
# outermost scope is the one that actually collects and clears the buffer.
# ---------------------------------------------------------------------------
_usage_tls = threading.local()


def _record_usage(purpose: str, model: Optional[str],
                   prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> None:
    buf = getattr(_usage_tls, "calls", None)
    if buf is None:
        return  # no active collection scope — zero overhead
    buf.append({
        "purpose": purpose,
        "model": model,
        "prompt_tokens": prompt_tokens or 0,
        "completion_tokens": completion_tokens or 0,
    })


class collect_usage:
    """Context manager: opens a per-thread usage buffer for the duration of
    one user query. Nestable — an inner scope is a no-op passthrough; only
    the outermost scope collects and clears on exit."""

    def __enter__(self):
        self._owns = getattr(_usage_tls, "calls", None) is None
        if self._owns:
            _usage_tls.calls = []
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._owns:
            _usage_tls.calls = None
        return False

    def calls(self) -> list:
        return list(getattr(_usage_tls, "calls", None) or [])


def usage_totals(calls: list) -> dict:
    """Sum a list of per-call usage dicts (as produced by collect_usage().calls())
    into the aggregate shape shared by explain_trace, MLflow, ChatMessage.metadata
    and QueryLog. No cost figure: self-hosted SLMs have no real per-token
    billing, so there is nothing meaningful to estimate a $ cost from."""
    pt = sum(c.get("prompt_tokens", 0) for c in calls)
    ct = sum(c.get("completion_tokens", 0) for c in calls)
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
    }


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
             endpoint="chat", model=None) -> tuple:
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
            usage = {"prompt_tokens": body.get("prompt_eval_count"),
                      "completion_tokens": body.get("eval_count")}
            return (body.get("response") or "").strip(), usage

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})
        payload = {"model": model or self.model, "stream": False,
                   "keep_alive": "24h", "messages": messages, "options": options}
        if json_format:
            payload["format"] = "json"
        body = _post_json(f"{self.base_url}/api/chat", payload, timeout)
        _note_usage(body)
        content = (body.get("message", {}) or {}).get("content", "")
        if not content:
            raise RuntimeError(
                f"SLM returned empty content from {self.base_url}/api/chat: "
                f"{json.dumps(body)[:300]}")
        usage = {"prompt_tokens": body.get("prompt_eval_count"),
                  "completion_tokens": body.get("eval_count")}
        return content, usage


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
             endpoint="chat", model=None) -> tuple:
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
        _u = body.get("usage") or {}
        usage = {"prompt_tokens": _u.get("prompt_tokens"),
                  "completion_tokens": _u.get("completion_tokens")}
        return content, usage


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
    (e.g. "ir_emit", "nl_answer", "decompose") — it never changes routing.
    Return contract is unchanged (plain str); token usage is recorded on the
    side into whatever collect_usage() scope is currently open, if any."""
    backend = get_backend()
    with _slm_circuit_breaker():
        content, usage = backend.call(
            user_message, system=system, timeout=timeout, temperature=temperature,
            num_predict=num_predict, num_ctx=num_ctx, seed=seed,
            json_format=json_format, endpoint=endpoint, model=model)
    _record_usage(purpose, model or backend.model,
                  usage.get("prompt_tokens"), usage.get("completion_tokens"))
    return content


def prewarm(model: Optional[str] = None, timeout: int = 120) -> None:
    """Best-effort model prewarm (Ollama: loads + pins via keep_alive; vLLM: a
    1-token completion touches the weights). Never raises."""
    try:
        call_slm("ok", purpose="prewarm", timeout=timeout, num_predict=1, model=model)
    except Exception:
        pass
