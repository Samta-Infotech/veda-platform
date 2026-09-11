# Chat API Contract — Query + History

Covers `POST /api/v1/conversations/query` (send a message) and
`GET /api/v1/conversations/history` (read a conversation back). Both
require an authenticated user (session/token — see `_resolve_user()`); an
unauthenticated request gets a `401`.

For the `usage` (token count) field details, see `TOKEN_USAGE_API_CONTRACT.md`
— this doc focuses on the full request/response shape of the two endpoints.

For how to obtain and refresh the token you authenticate with, see
`AUTH_API_CONTRACT.md`.

**Explainability (updated 2026-09-10).** Both endpoints can carry a richer
progress model and a "how this answer was produced" payload. Everything in
§1c and §1d is **additive and feature-flagged OFF by default** — with the flags
off the wire is byte-identical to what is documented in §1a/§1b, so an existing
client needs no change. §1c is the normalized `thinking` model; §1d is the v2
`explainability` payload.

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
| Federated (cross-source), answered | `null` — that path computes no confidence at all |
| Refusal (any tier) | `null`, or `explainability` itself may be `null`/a refusal-shaped object with no `confidence` key at all — see `build_refusal_explain()` |

Deterministic, weakest-link value derived from the anchor-selection and
join-planning gating confidences already computed during retrieval/routing —
**never an LLM self-report**. `1.0` when neither gating signal applies (e.g.
a single-table query with a high-confidence anchor). Computed by
`query/result_explainer.py::synthesize_confidence()`, called from
`veda/pipeline.py`'s `_done()` (Tier-1) and `veda_hybrid.py`'s
`_tier2_finish()` (Tier-2, Insight-Engine-only).

**A low score now produces a visible caveat.** When `confidence` is a real
number BELOW `LOW_CONFIDENCE_WARNING_BELOW` (default `0.5`), the turn also
carries a `low_evidence` entry in `explainability.warnings` and `limitations`
(§1d). A **missing** score does not — deliberately: the federated path computes
none at all, so warning on `null` put a caveat on every federated answer,
including correct ones. Absent evidence is a gap in the SIGNAL, not a signal.

**`sql.query` is populated by default.** `sql.enabled` reflects
`EXPLAIN_EXPOSE_SQL`, which defaults **on**: the generated SQL is the one part
of the explanation a reader can verify rather than take on trust, so a client
should expect the text to be there. This default was briefly flipped off (D2,
2026-09-09) on the grounds that raw SQL names tables and columns — the
vocabulary the rest of this payload keeps out of a user-facing explanation — and
was restored on 2026-09-11 at the user's explicit request; that exposure is the
accepted trade-off, not an oversight. An operator can set `EXPLAIN_EXPOSE_SQL=0`
to withhold it again for a non-technical audience. The block keeps its shape
either way — a client reading `sql.query` gets `null` rather than a missing key
— so nothing needs to change on the client when the flag moves.

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

---

## 1c. The `thinking` model (flag `VEDA_THINKING_STEPS`, default off)

With the flag off, a `thinking` frame is exactly what §1b documents:
`{"phase": ..., "message": ...}`. With it on, **the same frames** additionally
carry a `steps` object — the legacy keys never change, so old and new clients
read the same stream.

`steps` is a **normalized model of the whole turn**, not a rendering of the
event that carried it. The api tier folds the backend's ~26 internal phases into
four fixed steps, so no client has to decide which of several phases means the
same thing:

```json
{
  "type": "thinking",
  "status": "active",
  "current_step": "analyzing",
  "steps": [
    { "id": "understanding", "index": 1, "title": "Understanding your request",
      "state": "completed", "duration_ms": 6835,
      "summary": "You're asking for a ranking, for August 2026.",
      "expandable": true,
      "details": [ { "type": "operation", "label": "Looking for a ranking",
                     "state": "completed" } ] },
    { "id": "finding",    "index": 2, "title": "Finding the right information", "...": "..." },
    { "id": "analyzing",  "index": 3, "title": "Analyzing the information",     "...": "..." },
    { "id": "preparing",  "index": 4, "title": "Preparing your answer",         "...": "..." }
  ],
  "evidence":  { "sources": 1, "rows": 5 },
  "execution": { "type": "sql" },
  "timing":    { "total_duration_ms": 24512 }
}
```

**The four steps are fixed.** Same four, same order, same ids, for SQL,
documents, data lake, multi-source, refusals and small talk. Only their `state`,
`summary` and `details` vary.

**Do not render `phase` / `message`.** Those two are the pre-step-model thinking
line and are kept only for clients that predate `steps`. A steps-aware client that
renders both shows the same progress twice — once as a top-level status line and
once as the step list. `message` now mirrors the running step's `summary`, so the
two can never disagree, but the step list is the one to render.

**`steps` reveals progressively.** A step enters the array when the turn actually
reaches it — the first frame carries one step, not four. Render the array as it
arrives, in order; the array only ever **grows**, and a step already in it never
disappears and never moves. Use `total_steps` (always `4`) as the denominator for
"step 2 of 4" — never `steps.length`, which is the progress so far.

Listing all four up front described a *plan*, not progress, and on a turn that
never reaches step 3 the plan was simply wrong. A step is withheld only while it
is still `pending` and nothing after it has begun; once the turn moves past a step
it appears with a resolved state (`completed` or `skipped`), never as a `pending`
circle between two ticks.

| Field | Contract |
|---|---|
| `status` | `active` while the turn runs, then exactly one of `completed` / `failed` |
| `current_step` | the `id` of the running step; **`null` once `status` is terminal** |
| `total_steps` | always `4` — the size of the model. `steps.length` is the progress so far and grows during the turn |
| `steps[].state` | `pending` · `active` · `completed` · `warning` · `failed` · `skipped` |
| `steps[].duration_ms` | real measured time; `null` for a step that never started, and for one whose work happened but was never reported as a phase. **Never fabricated** — `null` means "not measured", not zero |
| `steps[].index` | 1-4, matching the fixed order — for a client that renders by position rather than by `id` |
| `steps[].summary` | one line, in the user's terms. Falls back to a generic sentence until the backend supplies facts |
| `steps[].details` | ordered evidence rows (below). Empty until the step starts |
| `steps[].expandable` | whether `details` is non-empty — the UI's cue that opening the step shows more |
| `evidence` | **counted facts only**: `sources`, `passages`, `rows`. A key is ABSENT when the backend did not report it — absent and `0` are different claims |
| `execution.type` | `sql` · `documents` · `multi_source` · `unknown` |
| `timing` → `total_duration_ms` | the whole turn, from the first event to the terminal frame. Live while running, frozen when terminal. This is the QUERY's time, not an overhead the progress display adds |
| `error` | present only on a failed turn: `{"code": ..., "retryable": true|false}` |

### `details[]` — a closed vocabulary

Each row is `{"type", "label", "state"}`, plus `duration_ms`/`message` on a
timed check. `type` is one of **six** values and no others, so a new backend
event cannot introduce a new row shape in the UI:

| `type` | Meaning | Example `label` |
|---|---|---|
| `access` | authorization, timed | `Checking access permissions` |
| `source` | a source that took part, by display name | `homzhub` |
| `evidence` | what was retrieved | `5 relevant passages retrieved` |
| `operation` | a semantic step of the analysis | `Ordered by Transaction Amount` |
| `validation` | a safety/consistency check | `Checking the result is complete and safe` |
| `output` | what is being produced | `Preparing table · 5 rows` |

### `steps[].state` rules a client can rely on

* A `pending` step never precedes a resolved one — a step the turn moved past
  resolves to `completed` (its work happened but went unreported) or `skipped`
  (it genuinely did not run).
* A step not yet reached is **absent from `steps`**, not present as `pending`.
  `pending` therefore appears only transiently, on a step the turn has entered.
* Steps only move **forward**. A late backend phase cannot reopen a finished step.
* A step that has not started shows **nothing** inside it: `details` is empty even
  when a measurement for it already exists.
* Once `status` is terminal, **no step is `active`** and `current_step` is `null`.

### Terminal frame

Exactly one terminal `thinking` frame is emitted before `content`, on **every**
exit including an error. On the error path it arrives before the `error` event,
so a client can freeze the progress display rather than leave a step spinning
next to an outage message.

---

## 1d. `explainability` v2 (flag `EXPLAIN_V2_ENABLED`, default off)

With the flag off the payload is exactly §1a's v1 object. With it on,
`version` becomes `"2.0"` and the payload gains the blocks below. **Additive by
contract:** no v1 key is removed, renamed or reshaped.

| Block | What it is |
|---|---|
| `routing` | why this source set — `{mode, summary, source_count, reason_code}` |
| `execution` | per-source outcome — `{summary, status, sources: [{name, type, status, duration_ms, rows_returned, required}]}` |
| `sources` | **where the answer came from** — `[{name, type, known, rows?}]`, by display name, never an identifier. `rows` is that source's own contribution and is **omitted when it was not recorded** (omitted and `0` are different claims). Present for a single-source answer too; an EMPTY list means the backend could not establish which source answered, not that none did |
| `result` | `{row_count, truncated, partial, reused_verified_query}` |
| `warnings` | stable machine codes + user copy (below). Always a list |
| `limitations` | the same warnings as plain sentences. Always a list |
| `provenance` | present only when the answer **replayed a previously verified query** — `{reused_verified_query, summary}`. Absent on an ordinary turn |
| `flow` | the "how this answer was produced" walkthrough (below) |
| `support` | `{trace_id}` — quote this in a bug report |
| `audit` | **level 3.** The technical record. See the disclosure levels below |

### Warning codes

Stable identifiers; the `message` is user-facing copy that may be reworded.

| `code` | Meaning |
|---|---|
| `result_truncated` | the result was cut to the first page |
| `partial_source_failure` | at least one source did not respond |
| `restricted_data` | permissions removed data that would otherwise have been included |
| `unmatched_records` | records on one side of a cross-source join had no match |
| `source_conflict` | sources disagreed |
| `fallback_used` | the primary method could not answer; an alternate was used |
| `low_evidence` | the answer's confidence is below the configured floor |

### `flow` — how this answer was produced

One shape for every execution type. Stages appear **only when the turn holds
evidence for them**, so a document answer legitimately has no query-validation
stage and a database answer has no passage count. Render them in order:

```json
{ "stages": [
  { "stage": "request",    "label": "Your request" },
  { "stage": "access",     "label": "Access verified" },
  { "stage": "sources",    "label": "Data source", "items": ["homzhub"] },
  { "stage": "evidence",   "label": "5 rows returned" },
  { "stage": "validation", "label": "4 checks passed" },
  { "stage": "operations", "label": "Operations applied",
    "items": ["Sort by Transaction Amount (highest first)", "Return top 5"] },
  { "stage": "answer",     "label": "Answer" } ] }
```

`stage` is one of `request` · `access` · `sources` · `evidence` · `validation` ·
`operations` · `result` · `answer`. `flow` is **absent** when there is nothing
beyond the request to show.

### Three disclosure levels

| Level | What to render | Where it lives |
|---|---|---|
| 1 — normal | the four steps and their state | `thinking.steps` |
| 2 — detail | each step's evidence rows, the warnings, the flow | `steps[].details`, `warnings`, `flow` |
| 3 — audit | backend phase names, per-phase timings, **source identifiers** | `explainability.audit` |

**Levels 1 and 2 contain no backend phase names and no source identifiers**, with
ONE legacy exception: the **v1 `timeline`** key (§1a) is a list of raw stage ticks
`[{phase, message}]` that predates this layer, and its `phase` values ARE internal
names. Treat that key as level-3 data too; `audit.timeline` is the maintained
replacement. (It no longer leaks anything worse — a tick that interpolated the
primary table name into its `message` was found and fixed.)

Otherwise `audit` is the only block that carries them — `{timeline, timeline_summary, sources}` —
and it is not part of the primary experience. `audit.sources` is where a raw
source id can still be recovered (`[{id, name}]`); every other block names a
source by display name only.

---

---

## 1e. Stability — what the frontend can safely build against

The explainability work continues behind these flags. This section is the promise
about **what will not move**, so a client built today keeps working while the
internals improve.

### Frozen — depend on these

| | |
|---|---|
| The four step `id`s | `understanding` · `finding` · `analyzing` · `preparing`, always all four, always this order |
| `steps[].state` values | `pending` · `active` · `completed` · `warning` · `failed` · `skipped` |
| `details[].type` values | `access` · `source` · `evidence` · `operation` · `validation` · `output` |
| Warning `code`s | the seven in §1d — stable machine identifiers |
| `flow` `stage` values | `request` · `access` · `sources` · `evidence` · `validation` · `operations` · `result` · `answer` |
| `status` values | `active` · `completed` · `failed` |
| Model keys | `type` `status` `current_step` `steps` `total_steps` `evidence` `execution` `timing` |
| Event names and order | `thinking`\* → `content`\* → `visualization`? → `explainability` → `usage` → `completed` |
| Error `code`s | `LLM_UNAVAILABLE` · `MODEL_ERROR` · `STREAM_ERROR` |
| The three-level split | backend phase names and source ids stay inside `audit`, never above it |

New values may be **added** to a closed set (a new warning code, a new
`execution.type`); nothing listed above will be renamed or removed. Treat an
unrecognised value as "something new" and fall through to a default rather than
throwing.

### Not frozen — do not depend on these

1. **Every human-readable string.** `summary`, `label`, `message`, `title`,
   `flow[].label`, warning `message` — all of it is copy and all of it will be
   reworded. **Never branch on a string, never match it, never translate by
   comparing it.** Branch on `id` / `type` / `code` / `state` and render whatever
   text arrives.
2. **Which optional blocks are present.** A document answer currently returns the
   **v1** payload — no `flow`, no `audit`, no `routing`, no `execution`. That is a
   known gap being closed. Render what is there; never require a block.
3. **`evidence` keys.** `sources` / `passages` / `rows` today, more later. A key is
   ABSENT when the backend did not report it — do not substitute `0`.
4. **`confidence`.** May be a number or `null`, and `null` is not "low". Show the
   `warnings` instead; that is what the caveat lives in.
5. **`duration_ms`.** May be `null` on a step whose work genuinely happened. Render
   "—", not "0s".
6. **The `steps` block may be absent entirely** on a turn that bypassed the engine
   (a canned greeting, answered in ~50 ms). No progress happened, so none is
   reported — show the answer with no progress UI, not four empty steps.

### The one rule that covers most of it

> Render structure from the **identifiers**; render text from the **strings**.
> Never let a string decide behaviour.

A client that follows this keeps working through copy rewrites, new warning codes,
new execution types, and the document-payload gap being closed.

---

## 2. `GET /api/v1/conversations/history`

Read a full conversation back (used to hydrate a chat window on load/reload).

### Request

Read-only, so **GET** — the parameter travels as a query string, not a JSON body.
(This doc previously showed `POST` with a body; that returns
`{"detail": "Method \"POST\" not allowed."}`.)

```
GET /api/v1/conversations/history?chat_id=42
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
            "trace_id": "0b28a3c4234e4d169afca077ac23bae8",
            "timeline": [
              { "phase": "received", "title": "Received your question",
                "status": "completed", "message": "Got your question" }
            ],
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

### `metadata.trace_id` and `metadata.timeline` (history)

Present only when the lifecycle flag was on for that turn — **absent, not null**,
for turns recorded before it, so a client checks for the key rather than for a
value. Both are written from the events the user actually watched, so the stored
record and the live stream can never disagree.

* `trace_id` — the same support reference as `explainability.support.trace_id`.
* `timeline` — `[{phase, title, status, message}]`, the per-phase record.

**Treat `metadata.timeline` as level-3 audit data** (§1d): it carries raw backend
phase names in `phase`, and is intended for a technical/audit view, not the normal
UX — render `title`/`status` if you show it at all. Unlike the SSE payload it is
**not** nested under an `audit` key, so that turns stored before the level split
stay readable without a migration.

A refusal or clarify stores a refusal-shaped `explainability`
(`{version, status, why, what_would_help, suggestions, warnings, limitations,
support, timeline_summary}`) with no `sql`, `operations` or `data_used` — a client
must not assume the answered-turn shape.

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

---

## Revision history

| Date | Change |
|---|---|
| 2026-09-10 (explainability v2) | Added §1c (the normalized `thinking` model) and §1d (`explainability` v2: `routing` / `execution` / `sources` / `result` / `warnings` / `limitations` / `provenance` / `flow` / `support` / `audit`). Both **flag-gated OFF by default** — with the flags off the wire is byte-identical to §1a/§1b. Three disclosure levels defined; backend phase names and source identifiers confined to `audit`. `sql.enabled` default flipped to **off**. A confidence below `LOW_CONFIDENCE_WARNING_BELOW` (default 0.5) now raises a `low_evidence` warning; a MISSING confidence deliberately does not. History `metadata` gained `trace_id` + `timeline`. |
| 2026-09-10 (stability) | Added §1e — the frozen-vs-not-frozen split, so a frontend can be built now while the internals keep improving. Identifiers, state values, closed vocabularies and event order are frozen; all human-readable copy and the presence of optional blocks are explicitly NOT. |
| 2026-09-10 (correction) | §2 documented `POST /conversations/history`; the endpoint is **`GET`** with a query parameter and rejects POST. Corrected. |
| 2026-09-10 (progressive steps) | `thinking.steps` now reveals **one step at a time** — a step enters the array when the turn reaches it, instead of all four being listed from the first frame. Added `total_steps` (always `4`) so a client can still render "step 2 of 4"; `steps.length` is now the progress so far. The array only grows, and a step already present never disappears or moves. A step the turn has moved past appears with a resolved state (`completed`/`skipped`), never as `pending` between two ticks — the same rule the terminal frame already applied, now applied live so consecutive frames cannot contradict each other. |
| 2026-09-10 (legacy message) | The legacy top-level `message` shipped **empty** on most `thinking` frames (9 of the 12 a normal relational turn emits) — most internal phases have no user-facing copy of their own — so a client rendering it showed a status line that appeared, blanked and reappeared several times per turn. It now falls back to the running step's `summary`: never empty, and never able to disagree with the step model. **`phase` and `message` are legacy**; a client that renders `steps` must not render them too, or the same progress appears twice. |
| 2026-09-10 (sources) | `explainability.sources` — "where did this answer come from" — was built only from proof of participation: a per-source execution record, or a routing decision that actually drove execution. A plain single-source query produces **neither** (only the cross-source coordinator writes execution records, and the routing decision is observe-only under shadow mode), so the commonest query in the system shipped **no `sources` block at all** (measured: a Tier-1 relational answer, `sources: null`). It now falls back to the one source the request was scoped to, when there is exactly one — with two or more and no record of which answered, the list stays empty rather than naming a guess. Each entry may now carry `rows`, that source's own recorded contribution, omitted when never recorded. Source identifiers remain confined to `audit.sources`. |
| 2026-09-11 (sql restored) | `explainability.sql` is **populated by default again**: `EXPLAIN_EXPOSE_SQL` returns to defaulting **on**, reversing the 2026-09-10 row's flip to off (decision D2). Restored at the user's explicit request — the generated SQL is what lets a reader verify an answer instead of trusting it, and that is worth the table/column names it discloses. No shape change: `sql` is still always present, and `EXPLAIN_EXPOSE_SQL=0` still yields `{enabled: false, query: null}`. |
| 2026-09-11 (no-engine turns) | A turn the engine never saw — small talk, a canned greeting, a thank-you — now emits **neither** a `thinking` nor an `explainability` event. It previously shipped a progress frame reading "Finalizing the results…" when there were no results, and the fixed-shape empty `explainability` skeleton (`sql.enabled: false`, `validation.passed: null`, "No filters applied."), which a client renders as a panel explaining nothing. The four-step block was already suppressed on these turns for the same reason; the other two surfaces were left behind. A client must already tolerate a stream with zero `thinking` events. A FAILED turn with no progress (the outage path) still emits its frame — that is how the error code arrives. |
