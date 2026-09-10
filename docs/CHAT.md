# CHAT — the conversational tier

**What.** `apps/chat/` (Django/DRF API surface + response assembly) and `chatbot/`
(a LangGraph supervisor package that runs *inside* the api container process). Together
they turn a multi-turn conversation into a sequence of single-shot engine queries and
stream the answers back as SSE.

**Basis.** Direct read of `apps/chat/`, `chatbot/`, `apps/query/inference_client.py`,
`apps/query/scope.py`, and `config/urls.py` on 2026-09-09. Skeleton / dormant / aspirational
pieces are called out as such.

> **Authority.** When code and prose disagree: `apps/chat/services.py` +
> `chatbot/graph.py` + `chatbot/nodes.py` are authoritative for turn behavior;
> `chatbot/memory/frame.py` + `classify.py` + `store.py` for the memory subsystem;
> the migrations for what physically exists in Postgres. `docs/archive/MEMORY_ARCHITECTURE.md`
> is a **design spec written before the code** — intent, not ground truth (see §9).

This doc expands [`ARCHITECTURE.md`](ARCHITECTURE.md) §3.5. Related:
[`VISUALIZATION.md`](VISUALIZATION.md), [`../CHAT_API_CONTRACT.md`](../CHAT_API_CONTRACT.md),
[`../TOKEN_USAGE_API_CONTRACT.md`](../TOKEN_USAGE_API_CONTRACT.md),
[`archive/MEMORY_ARCHITECTURE.md`](archive/MEMORY_ARCHITECTURE.md).

---

## 1. The boundary that shapes everything

`apps/chat/` and `chatbot/` run in the **api container** (thin tier — imports **no**
`veda_core`). The chatbot graph reaches the engine **only over HTTP**, through
`apps/query/inference_client.py::InferenceClient.stream_hybrid_query` — the *same* client,
the *same* `/v1/run_hybrid_query/stream` endpoint that `POST /api/v1/query` uses
(`inference_client.py:134`). There is no direct `run_hybrid_query` import anywhere in
`apps/` or `chatbot/`.

Consequences that recur below:

- `chatbot/llm.py` is a **standalone** Ollama/vLLM caller, deliberately *not* an import of
  `veda_core/slm/_call_slm.py` (`llm.py:1-16`).
- The memory subsystem's validation is deliberately light — schema/value validation happens
  server-side in the engine's L6a–L6c firewall on every call regardless
  (`chatbot/memory/frame.py:1-23`).
- The "sm-validation gate" from `MEMORY_ARCHITECTURE.md` **cannot ship** in this package —
  it would require importing `veda_core`'s semantic model (§9).
- `ask_clarification_node` cannot import `veda_core.veda.feedback` for a richer fallback —
  that import always fails in the api container (`nodes.py:784-799`).

---

## 2. The turn flow (traced, file:line)

### 2.1 HTTP in — `ConversationQueryView.post` (`apps/chat/views.py:89`)

| # | Step | file:line |
|---|------|-----------|
| 1 | `ConversationQuerySerializer` validates `{message, chat_id?, stream=true}` | `views.py:93` |
| 2 | `_resolve_user(request)` — the authenticated principal or **401**. The old fallback to the seeded dummy `admin` was **removed** (User Story 3 audit) — an unauthenticated caller is now rejected, not silently promoted | `views.py:100` → `views.py:47` |
| 3 | **RBAC gate 1**: `resolve_effective_permissions(user)` → `permitted_source_ids(user, effective)`. Permitted **zero** sources → `_denied_turn_response` (a *synthetic turn*, not a raw 403 — see §2.5) | `views.py:110-128` → `views.py:179` |
| 4 | `resolve_query_scope(request.data, tenant="default", user, effective)` → `source_ids` (ready sources, primary first). `SourceAccessDenied` → 403, `NoReadySource` → 503 | `views.py:140-150` |
| 5 | `serialize_data_scope(compute_data_scope(user, source_ids, effective))` → `data_scope` (the `X-Veda-Data-Scope` payload; `None` = no narrowing) | `views.py:154` |
| 6 | `source_profiles_for(source_ids)` → `source_profiles` (per-source `source_type` / `is_canonical` / `domain_tags` / `description`) | `views.py:162` |
| 7 | `ConversationQueryService(user, source_id=source_ids[0], source_ids, tenant="default", data_scope, source_profiles)` | `views.py:163` |
| 8 | `service.resolve_chat(chat_id, name_hint=message)` — ownership-checked existing `ChatSession`, or a new one titled from the first message; `ChatNotFound` → 404 | `views.py:167` → `services.py:155` |
| 9 | `service.save_user_message(chat, message)` — a `ChatMessage(type=USER)` row | `views.py:173` → `services.py:179` |
| 10 | branch on `stream`: `_stream_response` (SSE, `views.py:238`) or `_json_response` (buffered, `views.py:198`) | `views.py:175` |

`data_scope`, `source_profiles`, `source_ids` are the same values `POST /api/v1/query`
computes — chat was previously the only front door that omitted `source_profiles`, which
made the engine plan SQL against the primary source's schema for datalake/document
questions (`services.py:131-139`, `state.py:55-67`).

### 2.2 Service — `ConversationQueryService.run_turn` (`services.py:194`)

- `access_denied` short-circuit → `_access_denied_events` (content + explainability + usage,
  **no engine call**) (`services.py:212`, `services.py:335`).
- Builds `kwargs` (tenant, source_id, source_ids, request_id, data_scope, source_profiles)
  and a wall-clock `_turn_t0` — the source of `usage.latency_ms`, which is therefore
  **always present** even on a refusal that never reached the engine's own timer
  (`services.py:219-227`, `services.py:410-413`).
- **Streaming** (`services.py:229` → `_run_streamed` `services.py:264`): a daemon `Thread`
  runs `run_chat_turn(message, session_id, on_event=on_event, **kwargs)`; the node-level
  `on_event(phase, msg, extra)` pushes `("thinking", …)` items onto a `queue.Queue`; the
  generator drains the queue, yielding `{"event":"thinking",…}` frames until a
  `("result"|"error"|"done")` sentinel. This is the sync-Django equivalent of
  `inference/routes/hybrid.py`'s asyncio SSE route — there is no asyncio in the Django view.
- **Non-streaming** (`services.py:233`): a direct blocking `run_chat_turn(...)`.
- `response.get("engine_unavailable")` → an `error` event with code `LLM_UNAVAILABLE`
  (`services.py:243-258`). Otherwise `_build_reply_events(response)` (`services.py:361`).

### 2.3 `chatbot.run.run_chat_turn` (`chatbot/run.py:16`)

- `get_graph()` process-wide singleton (`graph.py:127`).
- `collect_usage()` scope wraps the **whole** turn — the supervisor's own classify /
  smalltalk / followup SLM calls *and*, nested inside, the engine's `collect_usage()`.
  Supervisor token spend is folded into `engine_result["usage"]` (`run.py:66-97`) — it was
  previously dropped. `_chat_usage.calls()` **must** be read inside the `with` block
  (`run.py:81-86`).
- `graph.invoke({message, history=[], session_id, tenant, source_id, source_ids,
  request_id, data_scope, source_profiles}, config={"configurable": {"thread_id":
  session_id, "on_event": …}})` (`run.py:67-80`).
- History is **not** threaded in by the caller — the checkpointer accumulates it per
  `thread_id` (`run.py:73`; the `history` param exists but every production caller passes
  `None` — only `test_manual.py` uses it).
- Returns a **flat dict**: `{session_id, answer_text, reply_text, needs_clarification,
  clarification_question, sql, rows, status, engine_unavailable, engine_result}`
  (`run.py:99-113`). `status` falls back to `"smalltalk"` / `"answered"` when the graph
  reset it to `None` (`run.py:110`).

### 2.4 Reply assembly — `_build_reply_events` (`services.py:361`)

Yields, in order:

1. `thinking:visualization_prep` — **only** if a chart will actually render (`services.py:367`).
2. `content` block(s): the summary markdown (with `res0["insights"]` folded in as bullet
   lines, `services.py:437-439`) marked `is_summary:true`, then a markdown table
   (`_rows_to_markdown_table` over `project_display_columns`-projected cols).
3. `visualization` — **ONE** event, `{"visualizations": [spec.to_dict(), …]}` (a
   wire-contract change from one-event-per-spec; `services.py:377-386`).
4. `explainability` — `res0["explain"]` or the fixed-shape `_NO_EXPLAIN`
   (`services.py:58-66`, `services.py:396`). Confidence lives **only** inside this object.
5. `usage` — `res0["usage"]` merged with `latency_ms = _turn_latency_ms` (turn wall clock).
6. `insights` — only if `res0` carries `insights` / `follow_up_questions`
   (`services.py:416-420`); **not persisted**, live SSE only.

### 2.5 View finalizes

- **SSE** (`views.py:248`): forwards each frame through `TurnEventAccumulator.consume`;
  on `error`, returns with **no** `completed` frame and persists nothing; on success,
  persists the assistant `ChatMessage` and emits `completed`
  `{chat_id, message_id, summary, is_complete:true}` (`views.py:273-280`).
- **JSON** (`views.py:198`): buffers via `TurnEventAccumulator`; a `turn_error` → **502**
  `{message, data:{chat_id, code}}` (`views.py:212-218`); else persists and returns
  `{chat_id, message_id, summary, response, metadata, insights?, follow_up_questions?}`.
- Assistant persistence (`services.py:184-192`): `content = json.dumps(content_blocks)`,
  `metadata = {thinking, explainability, usage}`.
- **Access-denied synthetic turn** (`_denied_turn_response`, `views.py:179`): a
  source-level RBAC denial still creates the chat + user message and streams/persists an
  access-denied assistant turn (`ConversationQueryService(access_denied=True)`), so it
  looks and saves like any other refusal rather than a shape the frontend must special-case.
  Wording mirrors `veda.feedback.ACCESS_DENIED_*`, duplicated not imported (`services.py:335-359`).

---

## 3. The LangGraph graph (`chatbot/graph.py`)

### 3.1 Shape

```
memory_read ──► classify
classify ──(_route_after_classify)──► smalltalk | context_resolve | call_engine
context_resolve ──► call_engine
call_engine ──(_route_after_engine)──► memory_write | ask_clarification
memory_write ──► format_reply
smalltalk ──► END
format_reply ──► END
ask_clarification ──► END
```

**8 nodes** (`graph.py:94-101`): `memory_read`, `classify`, `smalltalk`,
`context_resolve`, `call_engine`, `memory_write`, `format_reply`, `ask_clarification`.

**Entry point: `memory_read`** (`graph.py:103`) — *not* `classify`. This differs from
`MEMORY_ARCHITECTURE.md` §26, which shows `classify` first with `memory_read` after it.

### 3.2 The two conditional routers

| Router | file:line | Predicate |
|--------|-----------|-----------|
| `_route_after_classify` | `graph.py:59` | `action=="smalltalk"` → `smalltalk`; else `action!="runtime_context" and state.get("history")` → `context_resolve`; else `call_engine` |
| `_route_after_engine` | `graph.py:77` | `status in {"answered"}` → `memory_write`; else → `ask_clarification`. `_ANSWERED_STATUSES` is a **1-element set** (`graph.py:56`) |

`_route_after_classify` is **history-presence based, not LLM-label based**. A real question
the classifier mislabels `"answer"` still gets history-aware resolution — `context_resolve`
is a no-op when the message is already self-contained. `runtime_context` (the deterministic
current-date/time fast path) is the one exception: always self-contained, so it skips
`context_resolve` regardless of history (`graph.py:59-74`).

### 3.3 `ChatState` (`chatbot/state.py:30`)

`ChatState(TypedDict, total=False)`:

| Group | Fields |
|-------|--------|
| input | `message`, `history: Annotated[List[Turn], _capped_append]`, `session_id`, `tenant`, `source_id`, `source_ids`, `request_id`, `data_scope`, `source_profiles` |
| supervisor decision | `action` (`smalltalk`\|`answer`\|`clarify`\|`followup`\|`runtime_context`\|`clarify_reply`), `resolved_query` |
| engine result | `engine_result: Dict`, `status` (`answered`\|`refuse`\|`clarify`\|`no_table`\|`ungrounded`\|`qualifier_dropped`\|`ir_mismatch`\|`error`\|`unavailable`) |
| output | `reply_text`, `needs_clarification`, `clarification_question`, `sql`, `rows`, `engine_unavailable` |
| structured memory | `frame: Dict`, `drill_stack: List[Dict]`, `delta_type: str`, `episodic: List[Dict]` |

**`history` reducer — `_capped_append`** (`state.py:22-27`): appends each terminal node's
`[user, assistant]` pair to whatever the checkpointer holds, then trims to the last
`_HISTORY_MAX_TURNS * 2 = 20` entries (**10 turns**). This is audit fix **H4** — `history`
previously used `operator.add` and the checkpoint payload grew unbounded for the life of a
session (only the *prompt* read side, `history[-6:]` in `supervisor.py:117`, was bounded).
`classify_node` / `context_resolve_node` only **read** `history`, never return it, so they
never trigger an append.

`classify_node` **resets** the per-turn output fields
(`resolved_query`/`sql`/`rows`/`status`/`engine_result`/`needs_clarification`/
`clarification_question`/`engine_unavailable`/`delta_type`) to `None`/`{}` every turn
(`nodes.py:392-403`) — the checkpointer persists the full state across turns, so without
this reset a prior answer's `sql`/`rows` leak into a later smalltalk turn.

---

## 4. Node-by-node walk (`chatbot/nodes.py`)

### `memory_read_node` (`nodes.py:432`)
`_RESET_RE` whole-message match ("start over", "reset", "forget everything", "new topic") →
`MemoryStore.reset` + return empty frame/stack/episodic (audit fix **H2** —
`MemoryStore.reset()` existed but nothing called it). Otherwise 3 plain Redis reads
(`read_frame` / `read_stack` / `read_episodic`), each soft-failing to empty.

### `classify_node` (`nodes.py:225`)
Deterministic fast paths first (each skips the ~20s classify SLM call on this deployment's
hardware):

| Fast path | Regex | Result |
|-----------|-------|--------|
| bare greeting / thanks / bye | `_GREETING_RE` / `_THANKS_RE` / `_BYE_RE` (anchored whole-message) | `action="smalltalk"`, and `smalltalk_node` also skips its own SLM call (`_canned_smalltalk_reply`) |
| current date/time | `_RUNTIME_CONTEXT_RE` (a minimal duplicate of `query/runtime_context.py`, not an import) | `action="runtime_context"` |
| "go back" navigation | `_DRILL_UP_RE` **and** `frame["entity"]` **and** non-empty `drill_stack` | `action="followup"`, `delta_type="drill_up"` |

Otherwise: one `call_slm(build_supervisor_system_prompt(frame), build_supervisor_user_prompt(message, history))`
(`nodes.py:274`). `build_supervisor_system_prompt` appends the `_DELTA_BLOCK` addendum
**only when `frame["entity"]` is set** — the same call then also returns `delta_type` +
grounded `slot_candidates`, parsed by `parse_delta_response` (the merged call — a latency
fix that folds what used to be two SLM round-trips into one; `supervisor.py:1-18`,
`nodes.py:293-300`). Action defaults to `"answer"` on any parse/SLM failure
(refuse-over-guess).

**Override ladder** (`nodes.py:302-374`) — deterministic guards on top of the LLM verdict,
all "refuse-over-guess":

| # | Condition | Action |
|---|-----------|--------|
| 1 | `smalltalk` + history + `frame["entity"]` + `_REFERENTIAL_HINTS` matched + `_depends_on_history()` says "dependent" | → `followup` |
| 2 | `smalltalk` + `_DATA_QUESTION_HINTS` (count / how many / list / status / …) | → `answer` |
| 3 | `smalltalk` + `frame["entity"]` + not a genuine greeting/thanks/bye | → `followup` |
| 4 | (`followup`\|`answer`) + **no** `frame["entity"]` + `_BARE_REFERENTIAL_RE` (a dangling pronoun: "…the other one", "what about this?") + no data hints | → **downgrade to `smalltalk`** |

Guard 4's target — `smalltalk_node` has **zero DB access**, so downgrading is a *hard*
guarantee, not a lower probability, against forwarding ungrounded referential text to the
engine.

**Two production incidents motivate these guards** (`nodes.py:57-88`,
`tests/test_chatbot_classify.py` docstring):
- A bare `"hi"` and `"my name is raj"` were each rewritten into bogus, unfiltered database
  queries → the `_REFERENTIAL_HINTS` pre-filter on `_depends_on_history` (a message with no
  referential language *cannot* depend on history, so don't ask a model).
- `"what about the other one"` sent right after a bare `"hi"` (no `QueryFrame` ever
  established) was forwarded as raw text; the engine's retrieval "successfully" matched it
  against an unrelated table and returned a real row **including a real name and email** — a
  confident fabrication → guard 4 (`_BARE_REFERENTIAL_RE` + no frame → `smalltalk`).

### `smalltalk_node` (`nodes.py:406`)
Canned reply (`_canned_smalltalk_reply`) or one `call_slm(build_smalltalk_system_prompt(), …)`
capped at 60 tokens; falls back to `FALLBACK_REPLY`. Appends `[user, assistant]` to history.

### `context_resolve_node` (`nodes.py:460`)
Resolves a context-dependent message into a self-contained `resolved_query`:

- **No frame entity** (`nodes.py:493`) — Path A: `call_slm(FOLLOWUP_SYSTEM_PROMPT, …)`, the
  free-text "output verbatim if self-contained, else minimally rewrite from the last 6
  turns" rewrite. Returns `resolved_query`, `delta_type="new_topic"`. This is the one
  remaining "LLM invents a query from text" surface, kept as the staged-rollout safety net.
- **Frame exists** — Path B: reuse `delta_type` from `classify_node`'s merged call (**0**
  extra SLM calls); fall back to a standalone `classify_delta()` only if that failed.
  `drill_up` → `pop_drill(drill_stack)` + `rebuild_frame_from_stack`. Then
  `render_frame_as_query(frame, message, delta_type)` — a **deterministic** merge (message
  verbatim + ` (for <entity, filters>)` suffix). `"ambiguous"` passes the message through
  raw and lets the engine's own refuse/clarify path handle it (`nodes.py:529-537`).

### `call_engine_node` (`nodes.py:600`)
`InferenceClient().stream_hybrid_query(resolved_query or message, source_id, source_ids,
tenant, request_id, data_scope, source_profiles)`:

- `progress` frames → `_emit` (forwarded to the SSE `on_event`).
- `error` frame **or** `InferenceUnavailable` → `status="unavailable"`,
  `engine_unavailable=True` — kept **distinct** from a reachable engine's own refusal.
- `result` frame → `_extract_engine_result` (`nodes.py:545`): walks the MultiResult wire
  shape `payload["result"]["items"][0]["result"]`; aliases the NoSQL head's `columns` →
  `cols`; lifts `refuse_reason` / `route` from item0 up onto `res0` (a router refusal has a
  null item0 result, so the reason lives one level up); derives the pipeline-level status
  (`res0["status"]`, falling back to `item0["status"]=="ok"` → `"answered"` for
  RAG/hybrid/NoSQL heads that carry no pipeline status).

### `memory_write_node` (`nodes.py:667`)
**Only** on `status=="answered"` (`nodes.py:674`). `harvest_frame(engine_result)` →
`merge_frame_post_execution` → drill-stack update (reset `[]` / `push_drill` for
`drill_down` / evidence-based `newly_added_filter` + `push_drill_level` for anything else) →
`MemoryStore.write_frame` (optimistic-locked) + `write_stack` + `push_episodic_turn`. Every
field traces to `engine_result`'s already-validated output — nothing invented here.

### `ask_clarification_node` (`nodes.py:719`)
| Case | Reply | `needs_clarification` |
|------|-------|-----------------------|
| `status=="unavailable"` | honest "I couldn't reach the query service just now… try again in a moment" — **not** recorded to history (a transient outage isn't conversation) | `False` |
| router refusal, `route ∈ {no_access, no_match}` | `refuse_reason` **verbatim** — terminal | `False` (`definite`) |
| router refusal, `route == clarify` | `refuse_reason` verbatim | `True` |
| engine built a `feedback` dict | `feedback["text"]` | `True` |
| else | `engine_result["answer"]` or generic "Could you clarify…" | `True` |

### `format_reply_node` (`nodes.py:816`)
`reply_text = engine_result["answer"]`, sets `sql` / `rows`, appends `[user, assistant]` to
history.

### Thinking-phase translation
`_emit` forwards the internal `phase` string **verbatim** (logging/tracing upstream
unaffected); only the *displayed* `message` is swapped for a business-friendly one via
`business_friendly_message` (`services.py:282-292`, `thinking_messages.py`).

---

## 5. Checkpointer (`chatbot/checkpointer.py`)

| Property | Value |
|----------|-------|
| Backend | `RedisSaver` from `langgraph-checkpoint-redis` (`checkpointer.py:24`) |
| Instance | `CHATBOT_CHECKPOINTER_REDIS_URL`, default `redis://localhost:6380/0` — the **redis-stack** container, **not** the plain redis-cache/redis-broker on `:6379` |
| Requires | the **RediSearch** module (`FT.*`). The project's plain Redis does **not** have it — this is why a second Redis instance exists (`checkpointer.py:1-10`, `ARCHITECTURE.md` §11) |
| `thread_id` | `str(chat.pk)` — the `ChatSession` primary key |
| Setup | `saver.setup()` creates the search indices (idempotent) |
| Failure mode | **fail-loud at startup**: `get_checkpointer()` raises if Redis is unreachable — a broken checkpointer means no conversation can be tracked at all (`checkpointer.py:36-58`). Contrast `MemoryStore` and `call_slm`, which soft-fail per request |
| Postgres | **No LangGraph checkpoint tables in Postgres.** Checkpoint state lives only in redis-stack keys, subject to that instance's eviction/restart semantics |
| Future | the docstring flags a possible swap to `PostgresSaver` against the Django `veda` DB — **not implemented** (`checkpointer.py:12-17`) |

---

## 6. Memory subsystem (`chatbot/memory/`)

**Thesis: memory is evidence, not intelligence.** Every stored fact comes from either
(a) `harvest_frame()` extracting from an *executed, `status=="answered"`* engine result, or
(b) the user's own message copied verbatim. Nothing an LLM freely produced is ever stored
(`frame.py:1-23`).

### 6.1 Redis layout (`store.py:11-17`)

Keys `veda:mem:{tenant}:{session_id}:`:

| Suffix | Type | Holds | Cap |
|--------|------|-------|-----|
| `:frame` | STRING (JSON) | the current `QueryFrame` | — |
| `:stack` | LIST | the `DrillStack` (`{dimension, value}` breadcrumbs) | 10 (`_STACK_MAX`) |
| `:episodic` | LIST | `[user, assistant-gist]` buffer | 3 turns (`_EPISODIC_MAX`) |

Sliding TTL `VEDA_MEMORY_TTL_SECS` (default **4h**), refreshed on **every read and write**.
URL `CHATBOT_MEMORY_REDIS_URL` → falls back to `CHATBOT_CHECKPOINTER_REDIS_URL` → `:6380/0`.
A **separate** `redis` client from the checkpointer (may hit the same instance; different
key prefixes; never touches LangGraph checkpoint keys). Any Redis error → soft-fail to
empty (`store.py:56-63`).

### 6.2 `QueryFrame` (shipped shape — `frame.py:42`)

`version`, `tenant`, `session_id`, `entity` (raw table), `entity_display`, `understanding`
(the engine's own deterministic summary sentence), `filters: List[FilterFact]`, `group_by`,
`drill_path`, `last_sql`, `last_row_count`, `last_status`, `confidence`, `updated_at`,
`turn_index`. `FilterFact.source` is only ever `"executed_sql"`. **No `metric`, no
`time_range`** (contrast `MEMORY_ARCHITECTURE.md` §3 — §9).

### 6.3 Harvest (`harvest_frame`, `frame.py:88`)

Pure, zero-LLM. Reads `engine_result["explain"]` — the deterministic
`business_explain.build_explain()` output (sqlglot-parsed server-side): `datasets[0]` →
`entity_display`, `understanding.summary`, `filters.applied` (field/operator/value),
`operations` of type `group` → `group_by`, `sql`, row count. Returns `None` (skip the
write, answer unaffected) when status ≠ `"answered"` or `explain` is absent.

### 6.4 Merge / drill (all pure, `frame.py`)

- `merge_frame_post_execution` (`frame.py:142`): `harvested` **always wins** field-by-field;
  `prev` only fills gaps. Resets to `empty_frame` on `delta_type=="new_topic"`, on
  `is_topic_switch` (the engine independently routed to a different `entity`), or when there
  is no `prev`. Bumps `version` / `turn_index`; sets `confidence=1.0`, `last_status="answered"`.
- Drill stack (`nodes.py:695-708`): `reset` → `[]`; `drill_down` → `push_drill`; **anything
  else** → evidence-based `newly_added_filter(prev, harvested)` + `push_drill_level`. This
  last branch is the fix for the classifier labelling ordinary narrowing ("only the open
  ones") as `refine` not `drill_down`, which left the stack permanently empty so `drill_up`
  had nothing to pop.
- `write_frame` optimistic lock (`store.py:78`): `WATCH` the key; if the stored `version` ≠
  `expected_version`, **abort** the write (return `False`) — audit fix **C3**. The old code
  detected the conflict and clobbered anyway with a merge computed against stale data. The
  user's reply this turn is unaffected either way; only the *next* turn's read changes.
- `read_stack` / `read_episodic` carry audit fixes **C1** / **H1** for double-`reversed()`
  ordering bugs (`store.py:123-192`).

### 6.5 The ONE SLM call (`classify_delta` / `parse_delta_response`, `classify.py`)

Constrained output: `delta_type ∈ {new_topic, refine, drill_down, drill_up, compare,
ambiguous}` + `slot_candidates`. Two deterministic gates (`classify.py:32-91`):

1. **Vocabulary gate** (barrier 2): a proposed slot must be a case-insensitive **substring**
   of the user message, enforced independently of the prompt's own instruction.
2. **Confidence gate** (audit fix H3): if `delta_type ∈ {refine, drill_down, compare}` and
   slots were proposed but **none** are grounded → downgrade to `"ambiguous"`.
   `slots == []` (model proposed nothing) is *not* downgraded — a softer signal.

Fails closed to `"ambiguous"` on any parse/SLM failure. No prior frame entity →
deterministically `"new_topic"`, **no SLM call at all** (`classify.py:109-113`).

### 6.6 What feeds what

`memory_read_node` does 3 plain Redis reads (no embeddings, no fuzzy search) **before**
`classify` runs. The frame then feeds:

1. `classify_node`'s merged prompt (`build_supervisor_system_prompt(frame)` → the
   `_DELTA_BLOCK` addendum → `delta_type` + grounded `slot_candidates` in the same call).
2. `context_resolve_node` — `render_frame_as_query(frame, message, delta_type)` prepends a
   compact `<display (raw_table)>, <field op value>, …` noun phrase (`_describe_frame`,
   `frame.py:222`) to the verbatim message. Both the display name **and** raw table name
   appear — the display name alone was ambiguous enough that a bare "go back" re-resolved
   to a differently-named table in live testing (`frame.py:246-257`). `drill_up` returns
   just the popped frame's restatement (the words "go back" carry no data content).
3. `classify_delta` fallback also gets `episodic` — for `"it"` / `"that"` / "tell me more"
   reference resolution **only**, never a source of new frame facts (audit fix H1).

---

## 7. Follow-up / context resolution — the two paths

There is **no** `standalone_check.py` / `followup.py` / `delta_classify.py` *node* file —
those are **prompt** files under `chatbot/prompts/`. All resolution logic is in
`chatbot/nodes.py` + `chatbot/memory/`.

| | Path A — no frame yet | Path B — frame exists |
|---|---|---|
| Trigger | first real question, or no successful query this session | a prior turn produced `status=="answered"` |
| Mechanism | `call_slm(FOLLOWUP_SYSTEM_PROMPT, …)` — free-text "rewrite as standalone" | `delta_type` from `classify_node`'s merged call (fallback: standalone `classify_delta()`), then deterministic `render_frame_as_query` merge |
| Extra SLM calls in `context_resolve` | 1 (`nodes.py:498`) | **0** when the merged call produced a `delta_type`; 1 fallback otherwise |
| `delta_type` set | `"new_topic"` | the classified value; `"ambiguous"` passes the raw message through |
| Risk posture | the one remaining "LLM invents a query" surface — staged-rollout safety net | no free rewrite; message verbatim + proven-fact suffix only |

**`_depends_on_history` / `standalone_check`** (`nodes.py:90`): a *second-opinion* SLM call
(`STANDALONE_CHECK_SYSTEM`, one-word `dependent`/`standalone` output) run **only** when the
classifier said `"smalltalk"` **and** history exists **and** `frame["entity"]` is set
**and** `_REFERENTIAL_HINTS` matched. Fails closed to `False` (trust the original verdict).

The **override ladder's** "refuse-over-guess" guards and the two incidents that motivated
them are in §4.

---

## 8. Visualization (`apps/chat/visualization.py`)

`VisualizationRecommender.recommend(cols, rows, analytics=None)` — **deterministic, no LLM,
no DB** (`visualization.py:143`). Called from `services.py::_build_visualizations`
(`services.py:457`). Full treatment in [`VISUALIZATION.md`](VISUALIZATION.md); the essentials:

**Column classification — `_kind(i)` (`visualization.py:161`).** When the engine supplied
`analytics["column_stats"]`, its **semantic role outranks structural dtype**:
`role=="dimension"` → categorical (so numeric-valued dimensions like year / rating /
postal-code chart on the X axis), `measure` → numeric, `role=="date"` or `kind=="temporal"`
→ temporal, `role=="text"` → text, `role=="identifier"` → excluded. Falls back to the
structural `_infer_kind` (`visualization.py:268`) only for columns the engine didn't cover
(e.g. federated results). Identifiers (`_id` / `_uuid` / `_code` / camelCase `AccountID`)
and free-text (name-hint or avg length > 40) are banned from every candidate pool.

**Decision tree — `recommend` (`visualization.py:143`):**

| # | Condition | Result |
|---|-----------|--------|
| 1 | a dimension + **≥ 2 numeric** cols | `line_histogram` combo (first two numeric cols, positional) |
| 2 | temporal + 1 numeric + > 1 row | `[line, bar]` — bar is the same data at the same confidence |
| 3 | categorical + numeric | `_category_numeric` → `[pie, bar]` for ≤ 6 categories, `[bar]` for > 6 (top-9 + "Other"). `analytics["result_shape"]=="RANKING"` → drop the pie (a top-N isn't part-of-whole) |
| 4 | RANKING rescue: identifier-only labels + a real measure + > 1 row | a single bar leaderboard keyed by the id, confidence 0.65 |
| — | else | `[]` (table only) |

`_CONFIDENCE_THRESHOLD = 0.6` — below it, no chart. Multi-viz: `recommend` returns a list;
`services.py` emits **one** `visualization` event `{"visualizations": [spec.to_dict(), …]}`.
`spec[0]` is always the pre-multi-viz single choice.

**Fallback chain in `_build_visualizations`** (`services.py:466-487`): recommender →
`_spec_from_suggestion(res0["visualization"])` (the engine's Insight-Engine suggestion,
already server-validated) → `_spec_from_suggestion(analytics["chart_candidates"][0])` →
`[]`. All routed through the recommender's own public builders
(`build_category_specs` / `build_line_spec`) so the payload shape can't drift
(`services.py:84-110`).

> **`docs/VISUALIZATION.md` caveats.** Its `visualization.py:NN` line references were stale
> (the file is now 464 lines); the ones for the functions named above have been corrected
> in this pass. Its §4 "Column classification" still describes only the structural
> `_infer_kind` and does **not** mention the engine-role-aware `_kind` or the `analytics`
> parameter — treat §8 here as the current behavior. Its "2026-07-10 / 2026-07-15" dates
> predate later work and were left as-is (no reliable corrected date).

---

## 9. Memory: shipped vs. the `MEMORY_ARCHITECTURE.md` design

`docs/archive/MEMORY_ARCHITECTURE.md` (52 KB) is a design spec written **before** the code
and never reconciled. The core thesis shipped; several concrete claims are **wrong** for
the shipped code. Treat `frame.py` + `classify.py` + `store.py` + `nodes.py` as ground
truth. (Cross-ref: analysis report 07 §4.)

| `MEMORY_ARCHITECTURE.md` claim | Shipped reality |
|---|---|
| §0/§5/§26: `history` **removed** from `ChatState`; graph entry `classify` → `memory_read` | `history` is **still in `ChatState`** (`state.py:42`), now with the `_capped_append` 10-turn reducer. Entry point is **`memory_read`** → `classify` (`graph.py:103-104`) |
| §2/§8/§13/§14: a `memory/validate.py::validate_against_schema()` rejecting any column/value not in the semantic model; `merge_frame` resolving `column_hint` against `sm["domain_synonyms"]` | **Does not exist.** `chatbot/` must not import `veda_core`/`sm`. No `validate.py`, `merge.py`, `harvest.py`, or `stack.py` — it is all `frame.py` + `classify.py`. Validation is deliberately light; the engine re-validates L6a–L6c server-side |
| §3: `QueryFrame` has `metric`, `time_range: TimeRange`, `FilterClause` with an `op` enum + `source ∈ {…, "planner_validated"}` | Shipped `QueryFrame` (`frame.py:42`) has no `metric`, no `time_range`. `FilterFact.source` is only ever `"executed_sql"` |
| §8/§11/§27: `context_resolve_node` returns `needs_clarification=True` with a `build_clarifying_question(...)` when `delta_type=="ambiguous"` | `context_resolve_node` **never** asks for clarification — `"ambiguous"` passes the raw message through and lets the engine's own refuse/clarify path handle it (`nodes.py:535-537`). No `build_clarifying_question` exists |
| §18: version conflict → "retries merge once; second conflict → proceeds anyway" | Conflict **aborts** the write, no retry (`store.py:101-109`, audit C3) |
| §4: a `:lock` Redis key with a 2s TTL | Not used — the lock is Redis `WATCH`/`MULTI`, no separate key |
| §0/§2: `resolve_followup_node` fully eliminated | The free-text `FOLLOWUP_SYSTEM_PROMPT` rewrite still runs in `context_resolve_node` **when no frame exists** (Path A, §7) |
| §16/§29: add `maxmemory` / `volatile-ttl` to `redis-stack` in `docker-compose.yml` | A recommendation, not verified here — likely still open |

**Accurate parts:** the "evidence not intelligence" thesis, harvest-from-`build_explain`,
the provenance gate (write only on `"answered"`), the vocabulary gate, the 6 delta types,
the `:frame`/`:stack`/`:episodic` key layout, the 4h sliding TTL, drill push/pop semantics,
topic-switch detection via the engine's independent table routing, the `_RESET_RE` fast
path, and the first-turn `new_topic` short-circuit. The doc's audit tags **H1–H4 / C1–C3**
*do* correspond to real code comments — a later audit pass reconciled some of it.

**Verdict: ~60% descriptive, ~40% aspirational/superseded.**

---

## 10. `table_rendering` / `thinking_messages` / `turn_events` (pure helpers)

All three were split out of `views.py` / `services.py` **specifically to be unit-testable**
without pulling in the `chatbot.run → langgraph → redis` import chain. None has a Django,
DRF, or chatbot dependency.

### `table_rendering.py`
| Function | Behavior |
|----------|----------|
| `fmt_cell` | `None`/`""` → em-dash `—`; `bool` → yes/no; `Decimal`/`int`/`float` → thousands separator; `\|` escaped; cells > 80 chars truncated with `…`. **No currency symbol** — currency is a data column in this multi-source platform |
| `fmt_header` | `customer_name` → `Customer Name` (same humanization the summary prose uses) |
| `rows_to_markdown_table` | 20-row cap → appends `_Showing 20 of N rows._`; right-aligns a column only when **every** rendered non-null cell is a real number |
| `project_display_columns` | drops identifier-role columns using the engine's `analytics["display_columns"]`; **fails safe** to the original cols when that signal is absent or would drop everything |

### `thinking_messages.py`
Pure `dict` (`THINKING_PHASE_MESSAGES`) mapping ~35 internal phase ids → business-friendly
strings. Phases come from `chatbot/nodes.py::_emit` + forwarded verbatim from the inference
SSE stream (`veda_hybrid.py` / `pipeline.py` / `lg_nodes.py` / `rag_layer.py`). Never
exposes "SQL", table names, "supervisor", "routing", "RAG", "tier2". Unknown phase →
`business_friendly_message` returns the raw message (graceful degradation).
`visualization_prep` is api-tier-local — emitted in `_build_reply_events` only when a chart
will render.

### `turn_events.py`
`TurnEventAccumulator` — the shared fold used by **both** the JSON and SSE view paths
(previously two byte-identical if/elif ladders — `insights` was handled in only one).
Last-write-wins for scalars (`thinking` keeps the *last* message); `content` +
`visualization` append to `content_blocks`; `error` is deliberately **not** accumulated
(errored turns aren't persisted). `metadata()` → `{thinking, explainability, usage}`.

**Streaming mechanics:** real SSE (`text/event-stream`, `Cache-Control: no-cache`,
`X-Accel-Buffering: no` — `views.py:238-246`). The inference tier streams SSE →
`InferenceClient.stream_hybrid_query` parses frames incrementally (`inference_client.py:178`)
→ `call_engine_node` `_emit`s them → `_run_streamed` bridges the sync callback through a
`queue.Queue` on a daemon thread → `_sse_generator` reformats as `event:` / `data:` frames.
Non-streaming mode buffers the same stream and returns one JSON body (502 on engine error).

---

## 11. Persistence — three decoupled planes, one `chat.pk`

| Plane | Store | Keyed on | Lifetime | Purpose |
|-------|-------|----------|----------|---------|
| Django rows | Postgres `veda` — `ChatSession` / `ChatMessage` | `ChatSession.pk` | durable | the history API (`GET /api/v1/conversations/history`) |
| LangGraph checkpoint | redis-stack `:6380` (RediSearch) | `thread_id = str(chat.pk)` | evictable | the conversation context the graph replays; bounded to 10 turns by `_capped_append` |
| Analytical memory | redis `veda:mem:*` | `{tenant}:{chat.pk}` | 4h sliding TTL | `QueryFrame` + `DrillStack` + episodic buffer |

### Models (`apps/chat/models.py`)

- `ChatSession` (`models.py:15`): `name`, `user` FK (`related_name="chat_sessions"`,
  CASCADE), `created_at`/`updated_at`, `is_deleted`/`deleted_datetime` + `soft_delete()`.
  Indexes `(user, is_deleted)`, `(created_at)`. **`ChatSession.pk` doubles as the LangGraph
  `thread_id`.**
- `ChatMessage` (`models.py:40`): `session` FK (`related_name="messages"`, CASCADE), `type`
  (indexed), `content: TextField`, `metadata: JSONField`, `feedback` / `comment`,
  `soft_delete()`. Indexes `(session, is_deleted)`, `(created_at)`.
- User message: `content` = raw text. Assistant message: `content` =
  `json.dumps(content_blocks)`, `metadata` = `{thinking, explainability, usage}`
  (`services.py:184-192`).

### History retrieval

`ConversationHistoryView` is **GET** (`views.py:355`, `urls.py:16`) — its `chat_id`
travels as a query param (`ConversationHistorySerializer` reads `request.query_params`).
`CHAT_API_CONTRACT.md` currently documents it as POST — **the code is GET**.

`service.get_conversation_history` → `chat.messages.filter(is_deleted=False).order_by("created_at")`.
`_serialize_history_message` (`views.py:384`) rebuilds the assistant shape
`{response: [...], metadata: {thinking, explainability, usage}}` — parses the `content`
JSON, falls back to a single markdown block on parse failure, defaults `usage` to a
zero-triple for pre-`usage` rows. `insights` / `follow_up_questions` are **not persisted** —
live SSE only. `explainability.confidence` **is** persisted (it lives inside `explainability`).

`ListConversationsView` (GET, `views.py:314`): owned non-deleted sessions,
`order_by("-updated_at")`. `CreateConversationView` (POST, `views.py:283`).

---

## 12. Diagrams

### 12.1 LangGraph state machine

```
                    ┌──────────────┐
   (entry) ───────► │ memory_read  │   _RESET_RE → MemoryStore.reset, return empty
                    └──────┬───────┘   else: read frame/stack/episodic from Redis
                           ▼
                    ┌──────────────┐   fast paths: greeting/thanks/bye → smalltalk
                    │  classify    │                date/time         → runtime_context
                    └──────┬───────┘                "go back" (+stack) → followup/drill_up
                           │                 else: 1 merged SLM call → action + delta_type
             _route_after_classify(action, history)
        ┌──────────────────┼───────────────────────────┐
        │ action==smalltalk│ action!=runtime_context    │ else (first turn,
        ▼                  │   and history              │  or runtime_context)
 ┌────────────┐            ▼                             ▼
 │ smalltalk  │     ┌──────────────────┐         ┌──────────────┐
 └─────┬──────┘     │ context_resolve  │────────►│ call_engine  │
       │            └──────────────────┘         └──────┬───────┘
       │              Path A: FOLLOWUP rewrite          │  InferenceClient.stream_hybrid_query
       │              Path B: render_frame_as_query     │  (HTTP SSE to the inference tier)
       │                                        _route_after_engine(status)
       │                            ┌───────────────────┼────────────────────┐
       │                            │ status=="answered"│  else              │
       │                            ▼                   ▼                    │
       │                     ┌──────────────┐   ┌──────────────────┐         │
       │                     │ memory_write │   │ ask_clarification│         │
       │                     └──────┬───────┘   └────────┬─────────┘         │
       │                            ▼                    │                   │
       │                     ┌──────────────┐            │                   │
       │                     │ format_reply │            │                   │
       │                     └──────┬───────┘            │                   │
       ▼                            ▼                    ▼                   ▼
      END                          END                  END      (all terminal → END)
```

### 12.2 Streaming turn sequence (thread/queue bridge + nested SSE)

```
browser ── POST /api/v1/conversations/query {stream:true} ──► ConversationQueryView.post
   │                                                              │
   │                          RBAC gate 1 · resolve_query_scope · data_scope · source_profiles
   │                                                              │
   │                          save_user_message (ChatMessage USER row)
   │                                                              ▼
   │                                       ConversationQueryService.run_turn(stream=True)
   │                                                              │
   │                                              _run_streamed:  spawn daemon Thread
   │                                                              │        │
   │   ┌───────────────── queue.Queue ◄──── on_event(phase,msg,extra) ◄────┤
   │   │                                                          │        │
   │   │                                       run_chat_turn ──► graph.invoke  (BLOCKS)
   │   │                                                          │        │
   │   │                                    call_engine_node ──► InferenceClient
   │   │                                                          .stream_hybrid_query
   │   │                                                          │        │
   │   │                              ══ nested SSE stream ══► inference tier
   │   │                              progress frames ──► _emit ──► on_event ──► queue
   │   │                                                          │
   │   │                              result frame ──► _extract_engine_result
   │   │                                                          │
   │   │                              worker puts ("result", …) then ("done", …)
   │   ▼                                                          │
   │  _sse_generator drains queue:  "thinking" frames ... then    │
   │  _build_reply_events → content / visualization / explainability / usage / insights
   │                                                              │
   ◄── event: thinking / content / visualization / … / completed ─┘
        (on error: event: error, stream closes, nothing persisted)
```

### 12.3 The three persistence planes

```
                         one ChatSession.pk
        ┌────────────────────────┼───────────────────────────┐
        ▼                        ▼                           ▼
 Postgres  veda            redis-stack :6380           redis  veda:mem:*
 ┌──────────────────┐   ┌──────────────────────┐   ┌───────────────────────────┐
 │ ChatSession      │   │ LangGraph RedisSaver │   │ :frame   QueryFrame (JSON)│
 │ ChatMessage      │   │ checkpoint           │   │ :stack   DrillStack       │
 │  content JSON    │   │ thread_id = chat.pk  │   │ :episodic 3-turn buffer   │
 │  metadata JSON   │   │ ChatState, incl.     │   │ harvested from            │
 │                  │   │ history (10-turn cap)│   │ engine_result["explain"]  │
 │ durable          │   │ evictable            │   │ 4h sliding TTL            │
 │ history API (GET)│   │ needs RediSearch     │   │ soft-fails to empty       │
 └──────────────────┘   └──────────────────────┘   └───────────────────────────┘
   the durable record      the graph's context        the analytical working set
```

---

## 13. Wired vs. skeleton / dead code

**Fully wired:** the whole turn path (views → services → run → graph → all 8 nodes →
inference HTTP), SSE streaming, JSON mode, `conversations/history` (GET) / `list` / `create`,
RBAC gate 1 + scope + `data_scope` + `source_profiles` threading, the memory subsystem
(read / write / classify / merge / drill / reset), all deterministic fast paths and the
override ladder, the visualization recommender + 3-level fallback chain, thinking-message
translation, token-usage folding, the access-denied synthetic turn.

**Skeleton / not reachable:**

| Item | Status |
|------|--------|
| `ChatMessage.feedback` / `ChatMessage.comment` | model fields + migration, **no writer, no endpoint** — a future thumbs-up/comment feature |
| `ChatSession.soft_delete()` / `ChatMessage.soft_delete()` | defined on both models, **never called, no delete route** |
| `MessageType.SYSTEM` / `MessageType.TOOL` | enum members; only `USER` / `ASSISTANT` are ever written |
| `chatbot/checkpointer.py` PostgresSaver swap | documented future option, not implemented |
| migration `0002_seed_dummy_admin_user` | seeds a dev `admin`/`admin123` user; explicitly marked "remove once real auth replaces LoginView" — auth has since moved to `apps.authentication` |
| `action` value `"clarify_reply"` | the classifier can emit it, but `_route_after_classify` treats it identically to any other non-smalltalk action (history-based routing) — no distinct behavior |
| `run_chat_turn`'s `history` parameter | accepted and passed to `graph.invoke`, but every production caller passes `None` — only `test_manual.py` uses it |
| `MEMORY_ARCHITECTURE.md` components (`validate.py` / `merge.py` / `harvest.py`, `QueryFrame.metric` / `time_range`, in-node clarification, `history` removal, the sm-validation gate) | **did not ship** — blocked by the api↔`veda_core` import boundary (§9) |

**Near-dead but now live** (per code comments): `state.py::episodic` (written every turn,
was never read — now feeds `classify_delta`); `analytics["chart_candidates"]` (computed
every turn, now the 2nd viz fallback); `MemoryStore.reset()` (existed, unwired — now the
`_RESET_RE` fast path); `slot_candidates` grounding (computed + discarded — now downgrades
an ungrounded classification to `"ambiguous"`).

**Tests present:** `tests/test_chat_visualization.py`, `tests/test_visualization_matrix.py`,
`tests/test_chatbot_classify.py`. **No** tests for `services.py::run_turn`, the graph, the
memory subsystem, `turn_events.py`, or `table_rendering.py` (the last two were split out
specifically to be unit-testable — no test file exists yet).

---

## 14. Key file:line index

| Thing | Location |
|-------|----------|
| Turn entrypoint | `apps/chat/views.py:89` (`ConversationQueryView.post`) |
| Access-denied synthetic turn | `apps/chat/views.py:179`, `apps/chat/services.py:335` |
| Orchestration | `apps/chat/services.py:194` (`run_turn`), `:264` (`_run_streamed` thread/queue bridge) |
| Reply assembly | `apps/chat/services.py:361` (`_build_reply_events`) |
| Chatbot entrypoint | `chatbot/run.py:16` (`run_chat_turn`) |
| Graph | `chatbot/graph.py:87` (`build_graph`), `:59` / `:77` (edge fns), `:103` (entry point) |
| Nodes | `chatbot/nodes.py:225` classify, `:406` smalltalk, `:432` memory_read, `:460` context_resolve, `:600` call_engine, `:667` memory_write, `:719` ask_clarification, `:816` format_reply |
| Engine-result unwrap | `chatbot/nodes.py:545` (`_extract_engine_result`) |
| Checkpointer | `chatbot/checkpointer.py:36` (redis-stack `:6380`, RediSearch) |
| Memory store | `chatbot/memory/store.py:56` (`MemoryStore`), `:78` (`write_frame` optimistic lock) |
| Memory pure fns | `chatbot/memory/frame.py:88` harvest, `:142` merge, `:222` `_describe_frame`, `:265` render |
| Delta classifier | `chatbot/memory/classify.py:94` (`classify_delta`), `:32` (`parse_delta_response` + gates) |
| Inference HTTP | `apps/query/inference_client.py:134` (`stream_hybrid_query`) |
| Viz recommender | `apps/chat/visualization.py:143` (`recommend`), `:161` (`_kind`) |
| URL registration | `config/urls.py:23` (`include("apps.chat.urls")`), `apps/chat/urls.py:12` |
