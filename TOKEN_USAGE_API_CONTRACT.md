# Token Usage — API Contract

Adds a `usage` field to the chat and query APIs so clients can show per-turn
LLM token counts. This document is the frontend integration reference.

## Shape

```ts
type Usage = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
};
```

No cost figure. Self-hosted SLMs (Ollama/vLLM, `qwen2.5-coder:7b` /
`qwen2.5:1.5b-instruct`) have no real per-token billing — there's nothing
meaningful to compute a `$` cost from, so it's deliberately not part of this
contract (an earlier draft had a nominal `estimated_cost` field; it was
removed as misleading).

`usage` is **always present** with this exact 3-key shape — never a missing
key, never partially populated. For turns that never call an LLM
(deterministic fast paths, smalltalk, cache hits), it is the zero value:

```json
{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
```

Clients should treat this as "no LLM was used for this turn," not as an
error or missing-data state.

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
        "total_tokens": 1552
      }
    }
  }
}
```

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
data: {"prompt_tokens": 1240, "completion_tokens": 312, "total_tokens": 1552}

event: completed
data: {"chat_id": 42, "message_id": 501, "summary": "The 5 most recent asset accounting entries are dated ...", "is_complete": true}
```

The same `usage` object is persisted into `metadata.usage` on the saved
`ChatMessage`, so it's also readable later from chat-history endpoints
(same shape as §1's `data.metadata.usage`).

---

## 3. `POST /api/v1/query` (raw query endpoint, non-chat)

`usage` is a **top-level key**, sibling to `status` / `result` / `latency_ms`.

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
`usage` key; it's the one guaranteed to always resolve to a clean 3-key dict
even if the engine payload is malformed or missing.

Note: this endpoint does **not** currently surface `confidence` anywhere in
its response (unlike the chat endpoint's `explainability.confidence`) — it's
only wired into the chat path today.

---

## Notes

- **Stage-by-stage latency breakdown** (a "plan 0.6s · SQL 0.3s · run 1.1s ·
  render 0.4s" style breakdown) is **not part of this contract** — only the
  aggregate `total_ms` (pre-existing, inside `explainability.timeline`) and
  the token numbers above are live. Treat as a separate, future addition
  if/when needed.
- Backward compatible: existing clients that don't read `usage` are
  unaffected — it's an additive key/event, nothing existing changed shape.
- Where the numbers are stored server-side: `ExplainTrace` → `explain_trace.jsonl`
  → MLflow (via `mlflow_observability/mapper.py`, top-level metrics
  `total_prompt_tokens`/`total_completion_tokens`/`total_tokens`), `QueryLog`
  (3 columns, `apps/query/models.py`), and `ChatMessage.metadata["usage"]`
  (`apps/chat/models.py`). One capture point (`veda_core/slm/_call_slm.py`),
  fanned out to all three.
