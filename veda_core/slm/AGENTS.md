# veda_core/slm/ — the SLM backend seam (platform seam)

New for the platform. One entry point, two backends, chosen by `SLM_BACKEND`.

| File | Role |
|------|------|
| `_call_slm.py` | `call_slm(user_message, *, system, purpose, timeout, temperature=0.0, num_predict, num_ctx) -> str` over a Strategy: `OllamaBackend` (`/api/chat` + `/api/generate` raw-prompt, `keep_alive:"24h"`, `format:"json"` on request) and `VLLMBackend` (`/v1/chat/completions`, OpenAI-compat, maps `num_predict → max_tokens`, ignores `num_ctx`). `get_backend()` cached per process by `SLM_BACKEND`; `reset_backend()` test hook. Token accounting (`collect_usage` / `usage_totals` thread-local + a `reset_usage` / `get_usage` ContextVar). `prewarm()`. |
| `__init__.py` | re-exports. |

## Gotchas
- **`_slm_circuit_breaker` is a pass-through no-op** — `__enter__` returns self, `__exit__`
  returns False. Real trip/cooldown is deferred behind its own flag.
- ~41 engine call sites use `call_slm`; **three modules still hold direct Ollama HTTP**:
  `../query/slm_layer.py`, `../query/answer_entity.py`, `../query/rag_layer.py`.
- `SLM_MODEL_NAME` / `SLM_TEMPERATURE` are read from **`.env`**. `config.py` defaults
  (`qwen2.5-coder:7b`, `0.3`) differ — a machine without the `.env` override runs a
  different model and non-deterministic intent. `call_slm`'s own default temperature is
  `0.0`; call sites pass `temperature=config.SLM_TEMPERATURE`.
- `chatbot/llm.py` is a **separate** backend-agnostic caller — deliberately NOT an import
  of this module (the `apps/` ↔ `veda_core/` boundary).
