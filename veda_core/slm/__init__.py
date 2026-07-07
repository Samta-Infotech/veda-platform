"""SLM backend seam (§10) — restored per the post-cleanup review (Finding 1).

`call_slm` is the single entry every engine call site uses to reach the local
SLM; `SLM_BACKEND` selects Ollama (dev/ingestion) or vLLM (prod query tier).
"""
from slm._call_slm import call_slm, get_backend, prewarm, reset_backend  # noqa: F401
