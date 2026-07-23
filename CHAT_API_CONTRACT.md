# Chat API Contract — Query + History

Covers `POST /api/v1/conversations/query` (send a message) and
`POST /api/v1/conversations/history` (read a conversation back). Both
require an authenticated user (session/token — see `_resolve_user()`); an
unauthenticated request gets a `401`.

For the `usage` (token count) field details, see `TOKEN_USAGE_API_CONTRACT.md`
— this doc focuses on the full request/response shape of the two endpoints.

---

## 1. `POST /api/v1/conversations/query`

Send a user message, get the assistant's reply. Two modes controlled by
`stream`.

### Request

```json
{
  "message": "Show me the top 5 most recently dated accounting entries for our assets.",
  "chat_id": 42,
  "stream": true
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `message` | string | yes | non-blank |
| `chat_id` | int \| null | no | omit/null → a new chat is created; the created/resolved `chat_id` comes back in the response |
| `stream` | bool | no | default `true`. `false` → single JSON response; `true` → SSE stream |

`404` if `chat_id` is given but doesn't belong to the user (`{"message": "Chat not found."}`).
`400` on validation error (`{"errors": {...}}`, DRF serializer format).

---

### 1a. Non-streaming (`stream: false`) — response

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
    },
    "insights": [],
    "follow_up_questions": []
  }
}
```

`metadata.usage.latency_ms` is the **total end-to-end response time** for the
turn in milliseconds — the full server-side turn wall clock (engine + the
chatbot supervisor graph + serialization/streaming overhead). It is **always
present**, on a successful turn and on a failed/refused one alike, so it is
never `null`. It is server-side turn time, **not** the browser HTTP round-trip
— a client wanting true wall-clock still measures its own.

See `TOKEN_USAGE_API_CONTRACT.md` for the distinction from `/api/v1/query`'s
top-level `latency_ms`, which is a separate Django-side HTTP timer.

`response[]` is an ordered array of content blocks — `type` is one of
`"text"` / `"table"` / a chart type (`"line"`, `"bar"`, `"pie"`, ...) for
visualizations. `summary` is the block flagged `is_summary: true`, pulled out
as a convenience string.

`insights` / `follow_up_questions` are top-level `data` keys, only present
when the Insight Engine actually ran (`INSIGHT_ENGINE_ENABLED=true`) and
produced them — absent otherwise, not just empty.

**Confidence lives in exactly one place: `metadata.explainability.confidence`.**
There is no separate top-level `confidence` key — it was folded into
`explainability` so clients only ever check one field. See below for its
full contract.

### `explainability.confidence`

| Situation | Value |
|---|---|
| Tier-1 (deterministic SQL), answered | a real number, e.g. `0.87` — always present |
| Tier-2 (LLM-IR fallback), answered, `INSIGHT_ENGINE_ENABLED=true` | a real number |
| Tier-2, answered, Insight Engine off (default) | `null` |
| Refusal (any tier) | `null`, or `explainability` itself may be `null`/a refusal-shaped object with no `confidence` key at all — see `build_refusal_explain()` |

Deterministic, weakest-link value derived from the anchor-selection and
join-planning gating confidences already computed during retrieval/routing —
**never an LLM self-report**. `1.0` when neither gating signal applies (e.g.
a single-table query with a high-confidence anchor). Computed by
`query/result_explainer.py::synthesize_confidence()`, called from
`veda/pipeline.py`'s `_done()` (Tier-1) and `veda_hybrid.py`'s
`_tier2_finish()` (Tier-2, Insight-Engine-only).

On an engine error mid-turn: `502` — `message` is a safe user-displayable
string (never raw exception text) and `data.code` is one of the error codes in
the table below (`LLM_UNAVAILABLE` when the inference tier is down,
`MODEL_ERROR` otherwise):
```json
{"status_code": 502, "message": "The AI assistant is temporarily unavailable. Please try again in a moment.", "data": {"chat_id": 42, "code": "LLM_UNAVAILABLE"}}
```

---

### 1b. Streaming (`stream: true`) — SSE event sequence

`Content-Type: text/event-stream`. Each frame:
```
event: <name>
data: <json>

```

Typical sequence for one turn:

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

event: insights
data: {"insights": [], "follow_up_questions": []}

event: completed
data: {"chat_id": 42, "message_id": 501, "summary": "The 5 most recent asset accounting entries are dated ...", "is_complete": true}
```

| Event | When | Notes |
|---|---|---|
| `thinking` | 0+ times | progress/status messages; only the **last** one is persisted to `metadata.thinking` |
| `content` | 1+ times | one per content block, same shape as `response[]` items above |
| `visualization` | 0+ times | only when a chart is actually produced |
| `explainability` | always, once | `res0.explain` or a neutral fallback object. **`confidence` lives here** — see §1a |
| `usage` | always, once | see `TOKEN_USAGE_API_CONTRACT.md`. Token counts default to zero if no LLM call happened this turn; `latency_ms` is the total end-to-end response time (always present, never `null`) |
| `insights` | conditionally, once | only emitted if `insights`/`follow_up_questions` are non-empty server-side (Insight Engine ran) |
| `error` | on failure, terminates stream | `{"code": "...", "message": "..."}` — no further events after this. See the error-code table below |
| `completed` | always, last (success path) | signals the turn is fully persisted; carries `message_id` for later reference |

**Error codes** — the `message` is always a safe, user-displayable string;
raw exception text / tracebacks are never sent (logged server-side only):

| `code` | Meaning | Suggested UI |
| --- | --- | --- |
| `LLM_UNAVAILABLE` | The inference/LLM tier is down or unreachable — a transient outage, **not** a problem with the user's question. Message: *"The AI assistant is temporarily unavailable. Please try again in a moment."* | Show the message and offer a **Retry**; do not ask the user to rephrase |
| `MODEL_ERROR` | An unexpected fault while generating the answer. Message: *"Something went wrong while generating a response. Please try again."* | Show the message; retry is reasonable |
| `STREAM_ERROR` | A mid-stream failure in the SSE generator itself. Same safe copy as `MODEL_ERROR`. The connection closes with no `completed` event | Show the message; retry |

Note: previously an LLM outage surfaced the *clarify* fallback ("Could you
clarify what you're asking about?"), which misleadingly implied the question
was at fault. Outages now always return `LLM_UNAVAILABLE` with the copy above.

The assistant's `ChatMessage` is saved with
`metadata = {"thinking": ..., "explainability": ..., "usage": ...}` — same
values as the SSE events, in both streaming and non-streaming mode. Since
`confidence` lives inside `explainability`, it is persisted and therefore
**does** survive a history read (unlike `insights`/`follow_up_questions`,
which are not persisted — see §2).

---

## 2. `POST /api/v1/conversations/history`

Read a full conversation back (used to hydrate a chat window on load/reload).

### Request

```json
{ "chat_id": 42 }
```

`404` if the chat doesn't exist / doesn't belong to the user
(`{"message": "Conversation not found."}`).

### Response

```json
{
  "status_code": 200,
  "message": "Conversation retrieved successfully.",
  "data": {
    "chat_id": 42,
    "conversation_title": "Asset accounting entries",
    "created_at": "2026-07-16T10:02:11Z",
    "messages": [
      {
        "message_id": 500,
        "role": "USER",
        "content": "Show me the top 5 most recently dated accounting entries for our assets.",
        "created_at": "2026-07-16T10:02:11Z"
      },
      {
        "message_id": 501,
        "role": "ASSISTANT",
        "content": {
          "response": [
            { "type": "text", "content": "The 5 most recent asset accounting entries are dated ..." }
          ],
          "metadata": {
            "thinking": "Done — here's your answer",
            "explainability": { "version": "1.0", "confidence": 0.87, "...": "..." },
            "usage": {
              "prompt_tokens": 1240,
              "completion_tokens": 312,
              "total_tokens": 1552,
              "latency_ms": 2680
            }
          }
        },
        "created_at": "2026-07-16T10:02:14Z"
      }
    ]
  }
}
```

| Field | Notes |
|---|---|
| `role` | `"USER"` / `"ASSISTANT"` / `"SYSTEM"` / `"TOOL"` (`MessageType` uppercased) |
| USER message `content` | plain string — the raw message text |
| ASSISTANT message `content` | object: `{response: [...], metadata: {...}}` — same shape as query endpoint's `data.response` / `data.metadata` |
| ASSISTANT `metadata.usage` | same shape as the query endpoint, including `latency_ms`. Falls back to the zero-value object if the stored message predates this change (old rows have no `usage` in their saved `metadata`) |
| ASSISTANT `metadata.explainability.confidence` | **persisted and replayed** — same value the turn originally produced, survives page reload |
| `insights` / `follow_up_questions` | **not** included in history — they're only ever streamed live via the SSE `insights` event at the time the turn originally ran, never written into `metadata` |

Messages are returned in the session's stored order (oldest first, matching
`ChatMessage.objects.filter(session=chat).order_by("created_at")` semantics —
confirm against `ConversationQueryService.get_conversation_history()` if a
different order is ever needed).

---

## Notes

- `usage` and `explainability.confidence` are additive — pre-existing
  clients that don't read them are unaffected.
- History rows created **before** this change won't have `usage` in their
  stored `metadata`, and their `explainability.confidence` will be whatever
  `build_explain()` produced at the time (`null`, since confidence wasn't
  wired yet) — both degrade gracefully, no crash, just missing/null data for
  old rows.
- There is intentionally **one** confidence field across the whole contract:
  `metadata.explainability.confidence`. Do not add a second one elsewhere —
  earlier drafts of this contract had a duplicate top-level `confidence` key
  (in the `insights` event / `data.confidence`); that duplication was removed.
