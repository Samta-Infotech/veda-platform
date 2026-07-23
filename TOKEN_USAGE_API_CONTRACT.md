# Token Usage + Latency — API Contract

Adds a `usage` field to the chat and query APIs so clients can show per-turn
LLM token counts and total query latency. This document is the frontend
integration reference.

## Shape

```ts
type Usage = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  latency_ms: number | null;   // total end-to-end response time — chat endpoints only, see §1/§2
};
```

No cost figure. Self-hosted SLMs (Ollama/vLLM, `qwen2.5-coder:7b` /
`qwen2.5:1.5b-instruct`) have no real per-token billing — there's nothing
meaningful to compute a `$` cost from, so it's deliberately not part of this
contract (an earlier draft had a nominal `estimated_cost` field; it was
removed as misleading).

**Scope (chat endpoints only):** `usage` on `/api/v1/conversations/query`
covers the FULL turn, not just the engine. It's the sum of two independently
captured totals — the engine's own token spend (SQL generation, NL-answer/
Insight Engine summarization, envelope/IR emission — `veda_core/slm/_call_slm.py`)
**plus** the LangGraph supervisor's own SLM calls that run before the engine
is even reached (`chatbot/llm.py`: intent classification, smalltalk replies,
follow-up-context resolution, the standalone-question second-opinion check —
`chatbot/run.py::run_chat_turn()` sums both). `/api/v1/query` (the raw,
non-chat endpoint) has no supervisor layer, so its `usage` is engine-only.

Token counts (`prompt_tokens`/`completion_tokens`/`total_tokens`) are
**always present**, zero only when NEITHER the supervisor NOR the engine
called an LLM this turn — e.g. a canned-pattern smalltalk reply
(`_canned_smalltalk_reply`, no LLM at all) answered by a deterministic
fast-path/cached SQL query with the Insight Engine disabled. A smalltalk
reply that falls through to the LLM, or a cache-hit query that still
generates a fresh NL summary, will show nonzero tokens even though SQL
generation itself was skipped:

```json
{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
```

Clients should treat zero token counts as "no LLM was used for this turn,"
not as an error or missing-data state.

`latency_ms` (chat endpoints only) is the **total end-to-end response time**
for the turn — the full server-side turn wall clock (engine + supervisor graph
+ serialization/streaming overhead). It is **always present** on both a
successful and a failed/refused turn, so it is never `null`. It is the number a
UI shows as "response time." (Server-side turn time, not the browser HTTP
round-trip.)

---

## 1. `POST /api/v1/conversations/query` — non-streaming (`stream: false`)

`usage` is nested at `data.metadata.usage`.

```json
{
  "status_code": 200,
  "message": "Query processed successfully.",
  "data": {
    "chat_id": 42,
    "message_id": 501,
    "summary": "The 5 most recent asset accounting entries are dated 2026-05-24, 2026-05-22, 2026-05-16, 2026-05-12, and 2026-05-06.",
    "response": [
      { "type": "text", "content": "The 5 most recent asset accounting entries are dated ..." }
    ],
    "metadata": {
      "thinking": "Done — here's your answer",
      "explainability": {
        "version": "1.0",
        "understanding": { "summary": "...", "breakdown": ["..."] },
        "data_used": { "datasets": ["asset_entries"], "fields": ["entry_date", "asset_count"] },
        "operations": [ { "summary": "..." } ],
        "filters": { "applied": [], "summary": "No filters applied." },
        "validation": { "passed": true, "checks": [ { "label": "...", "passed": true } ] },
        "sql": { "enabled": true, "query": "SELECT entry_date AS date, COUNT(*) AS asset_count FROM asset_entries GROUP BY entry_date ORDER BY entry_date DESC LIMIT 5" },
        "confidence": 0.87,
        "timeline": [ { "phase": "output", "message": "Done — here's your answer" } ]
      },
      "usage": {
        "prompt_tokens": 1240,
        "completion_tokens": 312,
        "total_tokens": 1552,
        "latency_ms": 2680
      }
    }
  }
}
```

`latency_ms` here is the total end-to-end response time for the turn (engine +
the chatbot supervisor graph + serialization/streaming overhead) — server-side,
**not** the HTTP round-trip. Compare with §3's top-level
`latency_ms`, which is a different (Django-side, HTTP-inclusive) timer on the
raw query endpoint.

`explainability.confidence` — see `CHAT_API_CONTRACT.md` for the full
confidence contract; the short version is: it's a real number for every
answered Tier-1 (deterministic SQL) turn, `null` for Tier-2 unless
`INSIGHT_ENGINE_ENABLED=true`, and absent-context (not applicable) for a
refusal.

---

## 2. `POST /api/v1/conversations/query` — streaming (`stream: true`, SSE)

New SSE event: **`event: usage`**. Emitted once per turn, after
`explainability` and before `insights` (when present) / `completed`. Same
3-key payload as above.

```
event: thinking
data: {"phase": "visualization_prep", "message": "Preparing your chart..."}

event: content
data: {"type": "text", "content": "The 5 most recent asset accounting entries are dated ..."}

event: visualization
data: {"type": "line", "x_axis": "entry_date", "y_axis": "asset_count", "data": [...]}

event: explainability
data: {"version": "1.0", "understanding": {...}, "sql": {...}, "confidence": 0.87, "timeline": [...]}

event: usage
data: {"prompt_tokens": 1240, "completion_tokens": 312, "total_tokens": 1552, "latency_ms": 2680}

event: completed
data: {"chat_id": 42, "message_id": 501, "summary": "The 5 most recent asset accounting entries are dated ...", "is_complete": true}
```

The same `usage` object is persisted into `metadata.usage` on the saved
`ChatMessage`, so it's also readable later from chat-history endpoints
(same shape as §1's `data.metadata.usage`).

---

## 3. `POST /api/v1/query` (raw query endpoint, non-chat)

`usage` is a **top-level key**, sibling to `status` / `result` / `latency_ms`.
This `usage` does **not** carry `latency_ms` inside it — latency is already
its own top-level field on this endpoint (a separate, pre-existing,
Django-side timer around the whole HTTP call, not engine-internal).

```json
{
  "status": "answered",
  "result": {
    "items": [
      {
        "sub_query": "Show me the top 5 most recently dated accounting entries for our assets.",
        "status": "answered",
        "route": "deterministic",
        "result": {
          "cols": ["date", "asset_count"],
          "rows": [["2026-05-24", 1]],
          "sql": "...",
          "usage": { "prompt_tokens": 1240, "completion_tokens": 312, "total_tokens": 1552 }
        }
      }
    ]
  },
  "latency_ms": 2400,
  "request_id": "req-abc123",
  "cache_hit": false,
  "usage": {
    "prompt_tokens": 1240,
    "completion_tokens": 312,
    "total_tokens": 1552
  }
}
```

`result.items[0].result.usage` also carries the same numbers (it's the raw
engine payload passed through) — clients should read the **top-level**
`usage` key; it's the one guaranteed to always resolve to a clean dict even
if the engine payload is malformed or missing. Top-level `latency_ms` is
always present (int, milliseconds) — unlike the chat endpoints' `usage.latency_ms`,
it's never `null`, since it's measured unconditionally around the whole
request in the Django view.

Note: this endpoint does **not** currently surface `confidence` anywhere in
its response (unlike the chat endpoint's `explainability.confidence`) — it's
only wired into the chat path today.

---

## Notes

- **Stage-by-stage latency breakdown** (a "plan 0.6s · SQL 0.3s · run 1.1s ·
  render 0.4s" style breakdown) is **not part of this contract** — only the
  aggregate latency (now surfaced via `usage.latency_ms` on chat endpoints,
  and the pre-existing top-level `latency_ms` on the raw query endpoint) is
  live. Treat per-stage timing as a separate, future addition if/when needed.
- Backward compatible: existing clients that don't read `usage` are
  unaffected — it's an additive key/event, nothing existing changed shape.
- Where the numbers are stored server-side: `ExplainTrace` → `explain_trace.jsonl`
  → MLflow (via `mlflow_observability/mapper.py`, top-level metrics
  `total_prompt_tokens`/`total_completion_tokens`/`total_tokens`), `QueryLog`
  (3 columns, `apps/query/models.py`), and `ChatMessage.metadata["usage"]`
  (`apps/chat/models.py`). One capture point (`veda_core/slm/_call_slm.py`),
  fanned out to all three.
