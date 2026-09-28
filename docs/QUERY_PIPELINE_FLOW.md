# QUERY_PIPELINE_FLOW — one chat question, end to end

This doc follows one chat question from `POST /api/v1/conversations/query` to the
persisted assistant message. It covers every hop, every SLM call and every early
exit, and says which ones are **live today**.

> **Authority and date.** This doc was written 2026‑09‑26 from a direct read of the
> **working tree** on `feat/refinements-pipeline`, including uncommitted and untracked
> files. The source is authoritative. For flags, `veda_core/config.py` is the source of
> truth, and `.env` gives the live values. Line numbers drift, so treat them as anchors,
> not exact addresses.
>
> This doc supersedes the flow sections of [QUERY_ENGINE.md](QUERY_ENGINE.md) (2026‑09‑09)
> and [INGESTION_AND_QUERY_PIPELINES.md](INGESTION_AND_QUERY_PIPELINES.md) where the two
> disagree. Those docs are still useful for depth on individual layers.
>
> **Updated 2026‑09‑27 (integration pass)** — §2.3, §3.1, §3.2, §8, §9 and §10 now describe
> the tree AFTER that pass; §10 records what was fixed, what was measured and kept off, and
> what is still open. Full write-up: `reports/VEDA_INTEGRATION_2026-09-27.md`.
>
> **Uncommitted or untracked code on this path:**
> - the compound front door (`veda_hybrid.py:606‑936`)
> - the frame path (`veda/understanding/frame*.py`, `producers.py`, `vocabulary.py`)
> - the planner agent (`veda/agent/`), plus its hook in `federated_route.py:1055‑1081`
> - `slm_deadline` (`slm/_call_slm.py`)
> - chat compound and agent‑plan memory (`chatbot/nodes.py`, `chatbot/memory/{context,delta,frame}.py`)
> - `apps/chat/services.py`
> - `docker/nginx.conf`

---

## 0. The big picture

A chat turn crosses **three processes**. The api tier never imports `veda_core` for
chat; it reaches the engine over HTTP.

```
 browser
   │  POST /api/v1/conversations/query   {message, chat_id?, stream=true}
   ▼
 ┌─────────────────────────── api (Django, gunicorn) ─────────────────────────────┐
 │ ConversationQueryView ─ auth ─ RBAC ─ scope ─ persist user msg                  │
 │   └─ ConversationQueryService.run_turn ─ chatbot.run.run_chat_turn (Redis lock) │
 │        └─ LangGraph: memory_read → classify → [context_resolve] → call_engine   │
 │                                                    │                            │
 │        memory_write → format_reply  ◄── answered ──┤── else → ask_clarification │
 │   └─ _build_reply_events → SSE frames → persist assistant msg + QueryLog        │
 └───────────────────────────────────────────────│─────────────────────────────────┘
                     SSE POST /v1/run_hybrid_query/stream  (scope in X‑Veda‑* headers)
 ┌───────────────────────── inference (FastAPI) ─▼─────────────────────────────────┐
 │ middleware → RequestContext (source_ids, allowed_resources, session_prior, …)   │
 │ run_hybrid_query(query, on_event, trace_id=X‑Request‑Id, conversation_context)  │
 │   L0 runtime ─ L1 compound ─ L2 coordinator(shadow) ─ L3 federated ─ dispatch   │
 │   dispatch → classify → head: sql (veda/pipeline.run_query + Tier‑2) | rag |    │
 │              hybrid | nosql                                                     │
 │   → MultiResult → _serialize → SSE `result`                                     │
 └──────────────────────────────────────────────────────────────────────────────────┘
        all SLM calls → host Ollama  http://host.docker.internal:11434  (qwen2.5:7b-instruct, T=0)
        SQL execution → the Source row's DB (homzhub on host :5432) — engine tables on :15432
```

**What is live today.** `.env` switches these on: `FRAME_PATH_ENABLED=1`,
`AGENT_PLANNER_ENABLED=1`, `AGENT_JUDGE_MODE=enforce`, `ROUTING_CARDS_ENABLED=1`,
`MULTISOURCE_ROUTING_ENABLED=1` with `MULTISOURCE_ROUTING_SHADOW=1`, and an empty
`ROUTING_AUTHORITATIVE_MODES`. The live path for a first‑turn data question is:

```
runtime‑context regex (miss)
 → compound front door (1 SLM intent extraction; usually "single")
 → routing coordinator (permission pre‑check only; plan_route skipped — no authoritative mode)
 → federated (only if ≥2 sources in scope AND the question is cross‑source shaped;
              computes the request's ONE routing‑evidence pass, reused by classify)
 → _dispatch_single → classify (trace: classify.lane) → "sql"
 → pipeline.run_query:
      frame path (1 SLM frame extraction, probes, compile;
                  router primary from a lazy memoised retrieval, advisory anchors only)
        ├─ answers                → firewall → execute → NL summary (1 SLM)
        ├─ clarify/decline/snap   → planner agent (≤10 SLM steps) → firewall → execute
        │                            (split plan → extra parts; rag plan → RAG head)
        └─ degrade                → fast path → … → deterministic branches / LLM SQL
 → Tier‑2 on eligible refusals
```

> ⚠️ **`.env` only reaches code through docker.** The `load_dotenv` call in `config.py`
> is commented out (`config.py:15‑16`). `.env` becomes live only through `env_file:`
> in `docker-compose.yml`. A bare CLI or script run on the host gets the **code
> defaults**: FRAME_PATH and AGENT_PLANNER OFF, `SLM_MODEL_NAME=qwen2.5-coder:7b`, and
> `REQUIRED_SOURCE_ESCALATION` ON.

---

## 1. api tier: HTTP → graph

| # | Step | Where |
|---|------|-------|
| 1 | URL `conversations/query` → `ConversationQueryView` | `apps/chat/urls.py:13` |
| 2 | Request id: reuse `X-Request-Id` or mint one. This same id becomes the engine **trace_id**. | `apps/core/middleware.py:18‑22` |
| 3 | Validate: `message` is required, `chat_id` is optional, `stream` defaults to **True** | `apps/chat/serializers.py:52‑54`, `views.py:95‑99` |
| 4 | Auth: DRF Token + Session (JWT only if `VEDA_JWT_AUTH`, which is off). Unauthenticated → 401. | `views.py:49‑65` |
| 5 | RBAC: `resolve_effective_permissions` returns `None` while `VEDA_RBAC_MODE` is off (**live: off**). Zero permitted sources → a persisted synthetic "access denied" turn with **no engine call**. | `views.py:118‑137`, `services.py:466‑490` |
| 6 | Scope: all READY sources ∩ RBAC ∩ the optional request pin. `SourceAccessDenied` → 403, `NoReadySource` → 503. | `apps/query/scope.py:85‑173` |
| 7 | Data scope and `source_profiles` are computed. The service is built with `source_id = source_ids[0]`. | `views.py:163‑179` |
| 8 | Resolve or create the chat. **The user message is persisted before the engine runs.** | `services.py:205‑232`, `views.py:187` |
| 9 | Stream mode → SSE `_stream_response`; otherwise `_json_response`. Both drive the same graph, and the graph **always** uses the inference SSE route. | `views.py:212‑275` |
| 10 | `run_turn` → `run_chat_turn` on a thread. `on_event` → a queue → SSE `thinking` frames, using the four‑step model (`VEDA_THINKING_STEPS`, default **ON**). | `services.py:272‑464` |
| 11 | `run_chat_turn` takes the per‑session Redis turn lock, opens `collect_usage()`, and calls `graph.invoke(..., thread_id=chat.pk)`. | `chatbot/run.py:17‑162`, `memory/store.py:137‑184` |

---

## 2. Chat graph (LangGraph, `chatbot/`)

```
memory_read ─► classify ─┬─ smalltalk | represent | recall | no_match | reset ─► END
                         ├─ clarify_reply ───────────────┐
                         ├─ context_resolve (history≠∅) ─┤
                         └───────────────────────────────┴─► call_engine ─┬─ answered ─► memory_write ─► format_reply ─► END
                                                                          └─ else ─────► ask_clarification ─────────► END
```
`graph.py:118‑159`. The docstring diagram at `graph.py:3‑13` is stale.

### 2.1 `memory_read` (`nodes.py:1632`)
- Loads the frame and IR stack from Redis under the key
  `veda:mem:{tenant}:{session}:src:{source_ids[0]}:…`.
- Also loads the session‑wide episodic and comparison memory.
- `_frame_still_authorised` wipes memory for a source the user can no longer see.
- A reset phrase clears everything.

### 2.2 `classify` (`nodes.py:819`, wrapped by `classify_with_entry_gate`)

The checks below run in order. **Every rule before the last one costs zero SLM calls.**
1. reset
2. recall
3. represent (presentation change on the previous rows)
4. pending‑clarification reply
5. canned smalltalk
6. runtime‑context regex
7. document‑frame shortcuts
8. drill_up
9. **deterministic follow‑up (M4):** `memory/delta.detect` compares the message against the
   *previous result's own* dimensions and `top_values`:
   - a value seen in `top_values` → `add_filter`
   - a known dimension → `change_group`
   - `top N` → `change_order`
   - a measure word → `change_measure`
   - an earlier question → `switch_frame`
   - a compound part reference → `target_frame_index`

   The detector never invents a value; an unmatched content word returns `ambiguous`.
   A confident result (≥ 0.75, and not `new_topic`) skips the SLM (`nodes.py:796‑816`).
10. Otherwise **one** `call_slm(purpose="classify")`. It decides the action and, when a
    frame exists, the delta type (`prompts/supervisor.py`, parsed by `memory/classify.py`).

   These supervisor calls go through `chatbot/llm.py`, which is **not** the engine's
   client, so they never show up in the engine explain trace. They are reported
   separately as `supervisor_slm_calls`.

### 2.3 `context_resolve` (`nodes.py:1756`)
Runs when history exists. The steps:
1. Delta priority: rule delta → classify's `delta_type` → a fallback `classify_delta` SLM call.
2. drill_up pops the drill stack. Remove and replace are applied in Python.
3. Choose the **text sent to the engine**:
   - drill_up or remove → the frame's `base_query`
   - a document frame → `render_frame_as_query`
   - compare → the comparand
   - **otherwise the user's message verbatim**.

   The IR is **not** rewritten into a question: `frame.apply_delta` and `ir_to_question`
   have no callers.
4. Build `ConversationContext.from_frame(...).to_payload()` and return it as
   `conversation_context`.

> ✅ **Fixed 2026‑09‑27: the structured context now reaches the engine.**
> `conversation_context` is a declared `ChatState` channel (as are `comparison`,
> `entry_path`, `classification_latency_ms`, `requires_*`, `parts`), reset per turn in
> `classify_with_entry_gate`. A first turn sends `flags=None`; a `new_topic` turn sends only
> `{user_message}`; every other follow‑up sends the frame's state. The inference validator
> (`inference/routes/hybrid.py`) now also accepts `agent_plan`, `agent_log`, `agent_question`,
> `comparison`, `entry_path`, `classification_latency_ms`, `target_frame_index`,
> `part_index` — each shape‑checked; invalid values are dropped with a warning, never a 500.
> Inference prints one line per request: `[inference] conversation_context trace_id=… keys=…`.
> `_serialize` lifts a bounded `agent_memory` out of the trace before stripping it, so a
> planner‑agent plan can reach chat memory at all.
>
> Verified live: turn 2 of "how many properties are there in each city" → "only Mumbai" arrives
> with `entity_table=assets_asset`, group‑by and aggregation, and the engine records
> `classify.lane = continuity` (new trace field; also in the compact record as
> `classify_lane`). Graph‑level test: `tests/test_chat_graph_contract.py`.
>
> `frame.apply_delta` / `ir_to_question` / `compact_stack` were **deleted** (the engine applies
> the delta structurally now). The answering source now comes from the result's own
> `source_id` when `explain.sources` names no id (it never does), so an unpinned session no
> longer files its frame under source `None`.

### 2.4 `call_engine` (`nodes.py:2185`)
```
InferenceClient().stream_hybrid_query(
    query = resolved_query or message,
    flags = {"conversation_context": …} | None,   # None only on a first turn
    source_id, source_ids, session_prior, tenant, request_id,
    data_scope, source_profiles, no_cache)
```
- Transport: POST `{INFERENCE_URL}/v1/run_hybrid_query/stream`. The scope travels in
  `X-Veda-Source-Id(s)`, `X-Veda-Data-Scope`, `X-Veda-Source-Profiles`,
  `X-Veda-Session-Prior`, `X-Veda-Tenant`, `X-Request-Id` and `X-Veda-No-Cache`.
- Timeout: `INFERENCE_TIMEOUT_S` = **260** live (was 150 until 2026‑09‑27).
- A transport error → `InferenceUnavailable` → `status="unavailable"`.
- Engine progress events are relayed back up as SSE `thinking` frames.

---

## 3. inference tier → engine front door

`inference/main.py:154‑245` middleware builds the `RequestContext` from the headers. The
context holds:
- `source_id`, `source_ids`, `tenant`
- `allowed_resources` from `X-Veda-Data-Scope` (malformed → fail closed)
- `cache_back = not no_cache`
- `session_prior` / `session_anchor`
- source profiles

The stream route re‑binds the context on its worker thread and calls:

```
run_hybrid_query(query, verbose, on_event, trace_id=X-Request-Id, conversation_context)   veda_hybrid.py:1972
```

`run_hybrid_query` then:
1. calls `_set_conv(conversation_context)`
2. mints the **one** `ExplainTrace` for the request
3. starts the lifecycle timeline, the exec recorder and the per‑request embedding cache
4. runs `_run_hybrid_query_inner` (`:2594`), whose layers are listed below
5. post‑processes: `_clean_refuse_on_empty_error`, `_mark_empty_results`,
   `_backfill_missing_explain`, `_sync_reported_row_count`, `_reconcile_access_check`,
   `_emit_terminal_lifecycle`
6. calls `tr.finalize()`, which persists to `logs/explain_trace.jsonl`

### 3.1 Pre‑dispatch layers (in order; each can exit)

| # | Layer | Entry | Live | What happens |
|---|-------|-------|------|--------------|
| L0b | Runtime context | `:2650`, `query/runtime_context.py` | **ON** | Regex date/time question → **EXIT** route `runtime_context`. No SLM. |
| L1 | **Compound front door** `_maybe_compound` | `:2670` → `:668` | **ON** (`FRAME_PATH_ENABLED`) | See §3.2. **EXIT** if `compound`. |
| L2 | **Routing coordinator** `_run_coordinator` | `:2679` → `:1173` | ON, **shadow** | Since 2026‑09‑27: with `ROUTING_AUTHORITATIVE_MODES` empty (live) `plan_route` does **not run** — only the **permission pre‑check** does (trace: `routing.skipped=no_authoritative_modes`). With `SINGLE` in the set, an authoritative SINGLE **narrows the scope** to the routed source and falls through to the normal path (classify → head → Tier‑2) — it no longer dispatches to `query/agents.py`. SINGLE was re‑measured and kept **off** (see §10). See §3.3. |
| L3 | Federated `_maybe_federated` | `:2688` → `:1583` | ON when ≥2 `source_ids` | `run_federated`, see §3.4. **EXIT** on ok or a surfaced refusal; otherwise fall through. |
| L4 | `_dispatch_single(query)` | `:2702` | **the normal path** | `QUERY_DECOMPOSE_ENABLED` is off, so this runs and returns a 1‑item `MultiResult`. |
| L5 | Legacy decomposer | `:2706‑2801` | off (only after an authoritative MULTI handoff) | classify → probe → `run_decomposer` → `_fan_out` or a dependent refusal |

### 3.2 Compound front door (`veda_hybrid.py:668‑936`, `understanding/compound.py`)

**Skipped (returns `None`) when:**
- the flag is off
- the turn carries conversation `entity_table` or `filters`
- there are no `ctx.source_ids` (the CLI never takes this path)
- the vocabulary is empty
- any exception is raised

**Otherwise:**
1. `front_door_vocab(sids, …)`: entity cards per source, plus one document card per doc.
2. `extract_intents`: **one constrained SLM call** (timeout `FRAME_INTENTS_TIMEOUT` = 45 s)
   that returns 1–5 frames plus a relation.
3. `ground_intents`: each frame is grounded to a **source** (not a table).
   - rag frame → the matching doc source
   - data frame → the source whose cards *name* the entity
   - several unlinked sources → a per‑part CLARIFY
   - dependent frame → inherits its parent's source
4. `plan_compound` returns `single` when there is one clause segment and all parts land
   on the same (source, lane) → fall through. **This is the common case.** Since
   2026‑09‑27 the single frame *can* be handed to the SQL head instead of re‑extracting
   (`FRONT_DOOR_FRAME_REUSE`), but that switch ships **OFF**: the front‑door frame dropped
   value filters the head's own extraction keeps (see §10.3).
5. `compound` → `_run_compound` runs the parts **sequentially**. Each part runs on a
   worker thread under `slm_deadline` (`COMPOUND_PART_BUDGET_S` = 30 s, total 135 s):
   - rag lane → `run_rag_layer(part, source_ids=[sid])`
   - sql / tabular lane → `inject_frame(fr)` → `pipeline.run_query(part, sm, cols)`
     (no Tier‑2). A dependent part inherits the parent's ids and `top_values` as filters.
   - DEGRADE → `_run_sub` → coordinator → `_dispatch_single`
6. The reply is composed by `_compose_compound` (`compose_reply`, `summary_is_safe`,
   `fallback_summary`). A numeric guard stops the SLM summary from adding numbers.

### 3.3 Multi‑source routing (`query/source_coordinator.py::plan_route`, `:578`)
1. **Evidence**, per source:
   - clean cosine over column embeddings (top max(RAG_TOP_K, 5))
   - table embeddings (top 3)
   - doc chunks
2. **Item prior**: max cosine over `source_item_embeddings` per source. It can *add* a source.
3. **Dominance retier**: STRONG / WEAK / NONE.
   - Chunk scores are reduced by `ROUTING_CHUNK_KIND_OFFSET` (0.07).
   - The signal is the max of item / col / chunk.
4. **Capability filter** C2 (`query/capability_filter.py`; the file now exists) runs, ON.
   The comment at `source_coordinator.py:620‑627` saying the file is missing is stale.
5. **Edges**: `cross_source_fk` pairs with Jaccard ≥ 0.05.
6. `routing_policy.decide` → NO_MATCH / SINGLE / MULTI (relationship edge) / canonical
   SINGLE / AMBIGUOUS. An **edge‑MULTI override** applies when the query names both
   endpoints of a HIGH‑tier edge.
7. **Boundary** (ambiguous, no STRONG, ≥2 STRONG, or a runner‑up within 0.08):
   - a doc source as the strict top → deterministic SINGLE
   - otherwise `routing_slm.resolve_boundary`: **1 SLM call** that includes **per‑source
     routing cards** (`ROUTING_CARDS_ENABLED`) and the top‑3 item summaries. Timeout 20 s.
     Invalid output → CLARIFICATION_REQUIRED.
8. **Authority**: the decision drives the answer only if `status == ROUTED` and the mode
   is in `ROUTING_AUTHORITATIVE_MODES`. That set is **empty live**, so every decision is
   shadow.
   - Authoritative SINGLE would go to `query/agents.py` source agents.
   - Authoritative MULTI would go to decomposer handoff → doc+data grounding →
     federated → independent merge.

   **All of those branches are unreachable under the live `.env`.**

### 3.4 Federated (`query/federated_route.py::run_federated`, `:913`)

**Returns `None` (normal path)** when there are fewer than 2 sources or column‑bearing
sources, or when `should_federate` is false. `should_federate` is question‑level: it
requires cross‑source phrasing or an edge‑multi pair.

**Otherwise**, after `select_retrieval` and `qualify_columns`, the planners run in order:
1. operation classifier
2. structured semi‑join
3. **planner agent** (`veda/agent/federated.py::plan_federated`). It accepts only a plan
   spanning ≥2 sources with a `cross_source_fk` join. Head `federated.agent`.
4. deterministic join‑path plan
5. per‑metric plan with bounded repair
6. otherwise a typed refusal

Execution runs on DuckDB (`federated_executor.py`), with transient retry
(`query/reliability.py`) and an RBAC block.

---

## 4. Dispatch: `classify` → head (`veda_hybrid.py:346`, `:2925`)

`classify(query)` takes the first match:
1. Conversation `entity_table` set → `sql`. Unreachable from chat, see §2.3.
2. A document‑reference regex plus a doc source in scope → `hybrid` (if the query is
   aggregate‑shaped and a structured source exists) or `rag`.
3. `DOC_INTENT_EVIDENCE_ENABLED`: re‑runs the routing evidence. A dominant
   chunk‑backed source → `rag` / `hybrid`. This is a **second evidence retrieval** in the
   same turn.
4. `QUERY_ROUTER_ENABLED`: a keyword scorer, `query_router.route_query`, over the
   *config registry* of sources rather than the request scope.
5. `_guard_sql_head`: an `sql` result whose primary source is a doc or filesystem source
   → `rag`.

**Heads:**

| Route | Call | Notes |
|---|---|---|
| `sql` | `veda.pipeline.run_query(query, sm, cols)` → Tier‑2 on an eligible refusal (§6) | Empty `sm.tables` → RAG if a doc source matches ≥ 0.35, else `not_materialized` / `access_denied` |
| `rag` | `query/rag_layer.run_rag_layer(query, source_ids, temporal_filter)` | chunk retrieval → `rag_synthesis` SLM |
| `hybrid` | `run_query(summarise=False)` for the rows, optional graph chunks, then `run_hybrid_layer` | Merges document evidence with the SQL rows |
| `nosql` | `_run_nosql` → `nosql_builder` → execute → NL answer | RBAC via `filter_nosql_collections` |

The result is wrapped by `_to_subresult` → `SubResult{sub_query, status(ok|refused|error),
route, result, refuse_reason}`. Status mapping: `ok` → ok, `tier2_exec_error` → error,
anything else → refused.

---

## 5. SQL head: `veda/pipeline.py::run_query` (`:210`)

The "L1…L7" print tags are historical. This is the actual order:

| # | Stage | Module | SLM | Exits | Live gate |
|---|-------|--------|-----|-------|-----------|
| 0 | Setup: trace, `ExecutionState`, usage scope, conversation context (dropped if its source is out of scope); `entity_table` → `anchor_hint` | `pipeline.py:232‑375` | – | – | – |
| 1 | Temporal parse | `query/temporal_parser` | – | – | – |
| 2 | Grammar intent: existence / aggregate / superlative / grouped / ratio modes. `intent` is effectively always `SIMPLE`. | `veda/planning.py` | – | – | – |
| **F** | **Meaning‑first frame path** `run_frame_path(query, sm)` | `understanding/frame_path.py:155` | **`frame_extract`**: JSON‑schema constrained, 200 tokens, 45 s timeout. Plus one revision re‑extract after probes. | sql → **FrameLane** takes the fast‑path slot with a *complete* IR. clarify → `_done("clarify")`. degrade → continue. | `FRAME_PATH_ENABLED` **ON** |
| F‑a | **Planner agent** `run_planner`: ReAct over read‑only tools (`find_entities, describe, columns, join_path, values, similar_questions, probe, doc_sections, retrieve`). `Plan.validate` rejects any identifier not seen in a tool result. Compiled via the frame compiler. Checked by the cross‑encoder **judge** (enforce). | `veda/agent/` | **`agent_plan`**: ≤10 steps, ≤8 tool calls, 90 s, seed 7 | Used when the frame path would clarify / decline / fail entity coverage, or when its answer is "snapped" narrower than the question. Also for chat follow‑ups carrying `agent_plan` (unreachable, §2.3). Only `sql` or `clarify(probe)` plans are consumed; `rag` / `split` plans are dropped. | `AGENT_PLANNER_ENABLED` **ON** |
| FP | Deterministic fast path | `query/fast_path.try_fast_path` | – | miss → continue | ON (skipped when the frame answered) |
| FP2 | Superlative / grouped / ratio planners | `superlative_plan`, `ratio_plan` | – | clarify (terminal) | ON |
| FPG | Fast‑path evidence guard: demote the fast path when no table has evidence ≥ 0.3 (the frame lane is exempt) | `pipeline.py:984` | – | demote → full pipeline | ON |
| C | Verified cache: exact hash → pgvector cosine ≥ 0.85, then evidence / qualifier / shape demotions | `veda/cache.py` | – (BGE encode) | hit → reuse SQL | skipped whenever any fast path or frame lane exists, or on `no_cache` |
| 3 | Retrieval, 6 signals fused by weighted RRF: BGE‑M3 dense (`storage_adapters/reader.ann_search`, source‑scoped, engine DB), learned‑sparse, FK subgraph / path, value index, table prior | `retrieval/retrieval_engine_phase3` | – | – | top_k 15 |
| 3g | Graph expansion (+≤12 cols; outside‑sm additions dropped when source‑isolated) | `graph/query_graph` | – | – | ON |
| R1 | RBAC candidate filter plus scope filter | `veda/rbac_filter` | – | – | no‑op while RBAC is off |
| 3r | Cross‑encoder rerank of the top 20 (skipped on a clear RRF gap) | `query/reranker` | – | – | ON |
| 4 | **Anchor selection**: `select_primary_table` → ER pin or `vet_primary` → `anchor_hint` | `veda/routing.py` | – | clarify, or **`no_table`** (Tier‑2 eligible) | – |
| 4u | Query‑understanding layer (advisory) | `understanding/orchestrator` | `query_understanding` | – | **OFF** |
| 4e | Entity resolution V1: can pin the primary; ≥2 distinct tables → join path | `query/entity_resolver` | – | (AMBIGUOUS clarify is gated OFF) | ON |
| 5 | **Multi‑table planning**: `join_planner` → skeleton. Existence and aggregate are deterministic; `sql` has the LLM fill a fixed join skeleton. | `veda/planning.try_multitable`, `generation.generate_join_sql` | **`sql_join`** (timeout hard‑coded 120 s) | clarify, or **refuse** (Tier‑2 eligible) | `TYPED_MULTITABLE_ROUTE` ON |
| 6 | **Single‑table deterministic branches**, in priority order (list below) | `pipeline.py:1881‑2630` | – | clarifies (lifecycle, numeric, bare‑count); refuse on a temporal filter over a dateless anchor | ON |
| 7 | LLM single‑table SQL (only the final `else` branch): try `_deterministic_single_table_sql` first, then the SLM, then strip any unrequested LIMIT | `veda/generation.generate_sql` | **`sql_single_table`** | – | – |
| IR | Build `QueryIR`: complete from the frame / deterministic branches, partial from the LLM / cache | `veda/ir.py` | – | – | – |
| V1 | **Firewall** stage 1: value grounding + qualifier completeness. A complete `frame.*` IR skips the lexical qualifier check. | `veda/firewall.check`, `validation.py` | – | `ungrounded`, `qualifier_dropped` → re‑anchor retry / grounded clarify / access_denied / refuse (Tier‑2 eligible) | salvage OFF, re‑anchor ON |
| V2 | Entity coverage: caps confidence at 0.6 and tells the summariser what was not covered (never refuses) | `intent_sql_alignment.entity_coverage` | – | – | ON |
| V3 | Shape guards (grouped / distinct / ranked), drill‑narrowing check | `validation.py` | – | clarify | ON |
| V4 | Alignment (aggregate / filter / dimension presence). Skipped for a complete frame IR. | `firewall.check(run_alignment)` | – | clarify | ON |
| V5 | IR equivalence (LLM SQL only) | `veda/ir_equivalence` | – | `ir_mismatch` (**not** Tier‑2 eligible) | ON |
| V6 | Semantic validation (advisory; trace only) | `veda/semantic_validation` | – | – | ON, not enforcing |
| V7 | RBAC narrow + AST `validate_and_parameterize`: read‑only, allow‑list, join ON‑integrity, fan‑out guard, graph guard, literal parameterisation, **no silent LIMIT** | `firewall.check(run_rbac)`, `validation.py:11` | – | `invalid` (not Tier‑2 eligible) | ON |
| X | **Execute**. Parquet‑only scope → DuckDB (⚠️ `fetchmany(20)`). Otherwise psycopg2 on the Source row's DB: read‑only, autocommit, `statement_timeout` 30 s, `fetchmany(1000)`. **No Tier‑1 retry.** | `veda/execution.execute_sql`, `runtime.get_db_config` | – | `exec_error` (Tier‑2 eligible) | – |
| S | Result analysis + **NL summary**. Guards: numeric grounding, uncovered‑entity claim, extreme‑value claim, currency strip. Any failure falls back to a template / deterministic answer. | `result_analyzer`, `query/result_explainer.run_nl_answer` | **`nl_answer`** (T=0.1, 45 s) | – | ON (insight engine OFF) |
| W | Verified‑cache write (non‑fast‑path, rows > 0, non‑temporal, not context‑dependent) | `veda/cache.save_verified_query` | – | – | – |
| ✓ | `_done`: re‑anchor `table` to the SQL's FROM, confidence synthesis (< 0.5 → LOW_EVIDENCE warning; **0 rows at very low confidence → clarify**), `business_explain.build_explain` / `build_refusal_explain` | `pipeline.py:474‑691` | – | – | – |

**Single‑table branch order (stage 6):**
1. analytical_v2 re‑entry
2. answer‑entity
3. FK value subquery
4. multi‑hop FK
5. value / numeric / lifecycle filters (+ conversation reshape)
6. bare count
7. temporal‑only
8. ranked temporal
9. ranked metric
10. `else` → LLM SQL

**Frame answers take a lighter path.** They skip the fast‑path evidence guard, the lexical
qualifier gate, the text alignment guards and IR equivalence. They are never read from or
written to the verified cache. The AST firewall (V7) still applies.

**Result dict** (`_done`):
- Always: `status, ok, source_id, ir, _from_cache, trace, explain, usage, latency_ms, context`.
- Answered: `cols, rows, answer, sql, table, analytics, business_intent`, plus optional
  `confidence / visualization / follow_up_questions`.
- Refused: `msg / error / missing / detail, feedback`.

---

## 6. Tier‑2 (`veda_hybrid.py:3086`, `_tier2_sql` `:3890`)

**Trigger.** Tier‑1 status is one of `refuse | qualifier_dropped | ungrounded | no_table | exec_error`.
- **Never** for `clarify, invalid, ir_mismatch, access_denied, not_materialized`.
- Skipped if the head already took > 120 s. Budget 120 s.
- **Not run** on compound parts or source‑agent paths.

**Steps:**
1. Reuse the Tier‑1 `ExecutionState`, then `select_retrieval`.
2. **Envelope path**: `emit_envelope` (SLM `envelope`) → intent → `build_sql` → full firewall
   (`strict_qualifier`) → execute. Skipped for ranking, threshold and negation questions.
3. **IR path**: `run_slm_layer`. Live it goes through LangGraph (`USE_LANGGRAPH`), which makes
   4 SLM node calls: `classify_intent`, `select_entity`, `select_columns`, `build_filters`.
   - ≥2 entities → `planning.build_from_entities`.
   - Otherwise `sql_builder`.
   - Then firewall → execute.
4. The repair loop is OFF (`VALIDATION_REPAIR_LOOP_ENABLED`).
5. **Outcome:**
   - A semantic rejection → `tier2_rejected`: Tier‑1's refusal stands and gets a `tier2_note`.
   - Success → `_tier2_finish`: partial IR via `from_sql_facts`, analytics, NL answer, explain,
     plus the `FALLBACK_USED` warning.

---

## 7. Back up the stack: result → chat reply

1. **Inference `_serialize`** (`inference/routes/hybrid.py:51‑164`):
   - strips `context / trace / _debug`
   - sanitises engine error text
   - converts `Decimal` → float
   - sends the terminal SSE `result` `{status, trace_id, result: MultiResult}`
   - an engine exception becomes SSE `error`, which chat treats as `unavailable`
2. **`_extract_engine_result`** (`nodes.py:2079`) normalises to `res0`:
   - compound → a synthesized `res0` with `compound_parts`
   - multi‑source → a summary as `answer`
   - otherwise `items[0].result`
3. **Routing:** `answered` → `memory_write` → `format_reply`. Anything else →
   `ask_clarification`. Its reply priority is:
   1. unavailable copy
   2. router refusal reason
   3. `feedback.text`
   4. `res0.answer`
   5. the generic question

   It also arms `pending_clarification`.
4. **`memory_write`** (answered turns only):
   - `harvest_frame` and `harvest_entry(engine_result.ir, analytics)` push a `FrameEntry` onto
     the IR stack (≤10 entries, older ones compacted). Document answers get `ir=None`; an
     agent plan is attached when the trace has one.
   - Drill stack, `base_query` and episodic gist are updated.
   - Redis writes use an optimistic `WATCH` on the frame version.
   - Compound turns write one entry per part.
   - ⚠️ Memory is *read* under `source_ids[0]` but *written* under the engine‑attributed
     source. In a multi‑source scope, the next turn can miss the frame.
5. **`format_reply`**: `reply_text = res0.answer`, plus sql/rows, a `context_strip`, and
   `follow_up_questions` from the stack top.
6. **`run_chat_turn`** adds supervisor SLM usage to the engine usage and returns.
7. **`_build_reply_events`** (`services.py:540‑801`) yields, in order:
   - the terminal step frame
   - `context`? (rare now, because `resolved == message` on ordinary follow‑ups)
   - `content` blocks: the summary markdown, plus one table (per part for compound turns)
   - `visualization`? (from `VisualizationRecommender`, then `viz_override`, then
     `res0.visualization`, then `analytics.chart_candidates[0]`)
   - `explainability` (`res0.explain`)
   - `usage`
   - `insights`? (off)
8. **Persistence:**
   - `TurnEventAccumulator` → `save_assistant_message(content_blocks, metadata{thinking,
     explainability, usage, steps, timeline, context?, trace_id, action?})`.
   - `_audit_chat_turn` → `QueryLog`, whose `request_id` is the same value as `trace_id`.
   - SSE ends with `completed`; JSON returns the `api.success` envelope.
   - On an error, the user message is already stored but no assistant row is written.
9. **Engine‑side record:** `ExplainTrace.finalize()` appends the full trace to
   `logs/explain_trace.jsonl`. The record includes routing, frame, agent, `llm_usage`,
   `nl_summary`, the timeline and exec records. The api tier writes neither `llm_usage`
   nor `nl_summary` itself.

---

## 8. SLM call budget per turn (live)

All calls go to the host Ollama (`OLLAMA_URL=http://host.docker.internal:11434`) with
`SLM_MODEL_NAME=qwen2.5:7b-instruct` and `SLM_TEMPERATURE=0` (0.1 for `nl_answer`),
`SLM_NUM_CTX=4096`, and `SLM_TIMEOUT_SECS=60` unless stated. **SQL generation runs on the
instruct model, not the coder model.** `AGENT_SLM_URL` is empty, so the agent uses the
same Ollama.

| Where | Purpose | When |
|---|---|---|
| chat classify | `classify` | Only when no deterministic rule fired (≈20% of follow‑ups, and every non‑trivial first turn) |
| chat context_resolve | `chatbot` (classify_delta) / `standalone_check` | Rare fallbacks |
| compound front door | intent extraction (45 s) | Every first data turn with a source scope. On `single` it is still discarded (reuse ships OFF). |
| routing boundary | routing SLM (20 s) | Not live: `plan_route` does not run while `ROUTING_AUTHORITATIVE_MODES` is empty. |
| frame path | `frame_extract` (×1, +1 revision) | Every first data turn in the SQL head |
| planner agent | `agent_plan` (≤10 steps) | When the frame path clarifies, declines or snaps |
| multi‑table | `sql_join` | Join skeleton fill |
| single‑table | `sql_single_table` | Final `else` branch |
| summary | `nl_answer` | Every answered SQL turn |
| rag | `rag_synthesis` | rag / hybrid heads |
| Tier‑2 | `envelope` + 4 LangGraph node calls | Eligible Tier‑1 refusals |

**Timeout ladder:**
- per‑SLM‑call 60 s
- agent min(90 s, the part's remaining deadline − 10 s); skipped under 8 s
- compound part 100 s, compound total 240 s (= `INFERENCE_TIMEOUT_S` − 20); a part past its
  deadline returns a typed `timeout` outcome at once
- Tier‑2 120 s
- `INFERENCE_TIMEOUT_S` 260 s
- `GUNICORN_TIMEOUT` 280 s (`VEDA_TURN_LOCK_WAIT_SECS` 280 s)
- nginx `proxy_read_timeout` 300 s

---

## 9. Live flag snapshot (the ones that shape the flow)

| Flag | Live | Source |
|---|---|---|
| `FRAME_PATH_ENABLED` | **ON** | `.env` |
| `AGENT_PLANNER_ENABLED`, `AGENT_JUDGE_MODE=enforce` | **ON** | `.env` |
| `FRAME_INTENTS_TIMEOUT` | 45 | `.env` |
| `FRAME_PROBES_ENABLED` / `FRAME_SELF_CONSISTENCY` | ON / OFF | default |
| `MULTISOURCE_ROUTING_ENABLED` / `_SHADOW` | ON / **ON (shadow)** | `.env` |
| `ROUTING_AUTHORITATIVE_MODES` | *(empty)*: nothing is authoritative and `plan_route` does not run (SINGLE re‑measured 2026‑09‑27: 2/8 worse) | `.env` |
| `FRONT_DOOR_FRAME_REUSE` | **OFF** (built 2026‑09‑27, drops filters) | default |
| `ROUTING_PERMISSION_PRECHECK_ENABLED` | ON (the only live coordinator exit) | default |
| `ROUTING_CARDS_ENABLED` | ON | `.env` |
| `REQUIRED_SOURCE_ESCALATION_ENABLED` | **OFF** | `.env` = 0 |
| `CAPABILITY_FILTERING_ENABLED` | ON | default |
| `CROSS_SOURCE_EDGE_MULTI_ENABLED` | ON | default |
| `QUERY_DECOMPOSE_ENABLED` (`VEDA_…`) | OFF | `.env` |
| `QUERY_ROUTER_ENABLED` (`VEDA_…`) / `DOC_INTENT_EVIDENCE_ENABLED` | ON / ON | `.env` / default |
| `FAST_PATH_ENABLED` (`VEDA_…`) | ON | `.env` |
| `QUERY_UNDERSTANDING_ENABLED` / `ANALYTICAL_SQL_V2` | OFF / OFF | default |
| `QUALIFIER_SALVAGE_ENABLED` / `QUALIFIER_REANCHOR_RETRY` | OFF / ON | default |
| `TIER2_LLM_FALLBACK` / `USE_LANGGRAPH` / `VALIDATION_REPAIR_LOOP_ENABLED` | ON / ON / OFF | config |
| `NL_ANSWER_ENABLED` / `INSIGHT_ENGINE_ENABLED` | ON / OFF | config |
| `VEDA_RBAC_MODE` | **off**, so there is no RBAC narrowing in chat | unset |
| `VEDA_THINKING_STEPS` | ON | default |
| `INFERENCE_TIMEOUT_S` / `GUNICORN_TIMEOUT` / nginx | 260 / 280 / 300 | `.env`, `docker/nginx.conf` |

---

## 10. Known gaps — status after the 2026‑09‑27 integration pass

1. **Chat conversation context dropped before the engine** — ✅ FIXED (§2.3), verified live.
2. **The coordinator costs time but decides nothing** — ✅ FIXED: with no authoritative mode,
   `plan_route` no longer runs. Routing evidence is computed once per request and reused by
   `classify` and the federation gate (`source_coordinator.routing_evidence`, trace
   `routing.evidence_passes`). Making SINGLE authoritative was re‑measured with the new
   narrow‑and‑fall‑through semantics and **kept off**: on the 8‑query benchmark it answered
   FS1 ("late fee percentage") from homzhub's `late_fee` column instead of the MSA PDF, and
   narrowed the cross‑source XS2 to one source (clarify).
3. **Frame extraction paid twice** — ⚠ BUILT, SHIPPED OFF (`FRONT_DOOR_FRAME_REUSE`). Reusing the
   front door's frame saves one SLM call per first turn and scored the same on question.txt
   (12/20 both ways), but the front‑door frame dropped value filters on the per‑source battery
   ("vendors in Kochi" → no WHERE). Needs a front‑door prompt that carries filters faithfully.
4. **Frame path gets no router hint** — ✅ FIXED: a lazy, memoised `_retrieve_rank` supplies the
   router primary only when the frame has an advisory anchor (no second retrieval; NAME turns
   pay nothing). Agreement also needs evidence from the question itself.
5. **Agent `rag` / `split` plans discarded; agent limited to `ctx.source_id`; budget > part** —
   ✅ FIXED: split → extra budgeted parts, rag → `run_rag_layer` on the scope's doc sources, a
   plan on any in‑scope source narrows the request to it (RBAC kept); agent wall = min(90 s,
   part deadline − 10 s). Split is covered by a mocked test only (the live example did not
   reproduce).
6. **Memory key mismatch** — ✅ FIXED: `memory_read` tries every source in scope and takes the
   newest frame (`written_at`); a cross‑key write no longer aborts on the version check.
7. **Row caps** — ✅ FIXED: DuckDB fetches `EXECUTION_RESULT_LIMIT`; the pipeline's LIMIT‑100
   tails are gone; `SLM_JOIN_TIMEOUT_SECS` is config. **Still open:** the frame compiler emits
   `LIMIT 100` on list answers (seen on `frame.*` SQL).
8. **Dead code** — ✅ deleted: NL simplifier, ER ambiguous clarify, `_tier2_validate` (+
   `_sql_keeps_constraint`), `cross_source_guard.py`, the duplicate
   `_emit_terminal_lifecycle`, the frame IR helpers. **Kept on purpose** (env‑toggleable,
   documented, tested — not dead): the QU re‑entry, qualifier salvage, the Tier‑2 repair loop.
   Follow‑up: `query/execution_request.py` looks dead.
9. **Stale comments / drifts** — ✅ FIXED (all listed items).
10. **`route_query` ignores the request scope** — ✅ FIXED (scores `ctx.source_ids`, typed by the
    source profiles). Found and fixed on the way: when the doc‑primary guard demoted a `sql`
    route to `rag` it kept the router's SQL source ids, so RAG searched a source with no
    documents (FS1 "No relevant document passages found").

**Environment facts found during the pass:** `.env` still reaches code only through docker
(`load_dotenv` stays off, reason in `config.py`); every harness now prints and can assert its
flags (`scripts/_flags.py`, `--expect KEY=1`). The inference container runs uvicorn with a
file‑watching reloader, so saving an engine file reloads live inference. Nothing configures
app loggers in inference, so `logger.info` there is dropped — use `print`, as the engine does.
