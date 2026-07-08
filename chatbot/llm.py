"""chatbot.llm — thin Ollama caller for the supervisor's classify step.

Deliberately plain urllib (no veda_core import needed here) — matches the
pattern used elsewhere in this codebase (veda_core/ingestion/domain_glossary.py,
glossary_builder.py) so this module has no hard dependency on veda_core beyond
what run_engine() needs.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
SLM_MODEL_NAME = os.environ.get("SLM_MODEL_NAME", "qwen2.5-coder:7b")


def call_ollama(system: str, user: str, *, temperature: float = 0.1,
                 max_tokens: int = 200, timeout: int = 20) -> str | None:
    """One-shot chat completion. Returns None on any failure — callers must
    have a deterministic fallback (refuse-over-guess, same as the rest of
    this codebase)."""
    payload = {
        "model": SLM_MODEL_NAME,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return (body.get("message", {}).get("content") or "").strip()
    except Exception as exc:
        logger.warning("call_ollama: request to %s failed: %s", OLLAMA_URL, exc)
        return None
