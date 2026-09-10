# chatbot/ — the LangGraph conversational supervisor

Runs **inside the api container process**. One turn at a time: classify → resolve
follow-ups against memory → call the engine over HTTP → harvest evidence back into memory.
Full reference: [../docs/CHAT.md](../docs/CHAT.md).

## Root package
| File | Role |
|------|------|
| `run.py` | `run_chat_turn(message, session_id, **kwargs)` — the ONE entrypoint. `get_graph()` singleton, `collect_usage()` scope, `graph.invoke(..., config={"configurable": {"thread_id": session_id, "on_event": …}})`, folds supervisor SLM tokens into `engine_result["usage"]`, returns a flat dict. |
| `graph.py` | `build_graph()` — compiles the `StateGraph(ChatState)` with the RedisSaver checkpointer. 8 nodes; `_route_after_classify` (history-presence based, **not** LLM-label based) / `_route_after_engine` (`status == "answered"`). `get_graph()` process singleton. |
| `state.py` | `ChatState` TypedDict + `Turn`. `history` uses a `_capped_append` reducer (last 10 turns = 20 entries). Fields: input, supervisor decision, engine result, output, structured memory (frame / drill_stack / delta_type / episodic). |
| `nodes.py` | All 8 node functions + deterministic regex fast-paths + the "refuse-over-guess" override ladder. `classify_node`, `smalltalk_node`, `memory_read_node`, `context_resolve_node`, `call_engine_node`, `memory_write_node`, `ask_clarification_node`, `format_reply_node`. `_extract_engine_result` unwraps the `MultiResult` wire shape. |
| `llm.py` | `call_slm(system, user, ...)` — backend-agnostic (Ollama `/api/chat` \| vLLM OpenAI-compat) one-shot caller, stdlib `urllib`, returns `None` on any failure. `collect_usage()` / `usage_totals()`. `CHATBOT_CLASSIFY_MODEL` env. **Deliberately NOT an import of `veda_core/slm/_call_slm.py`** (the api ↔ veda_core boundary). |
| `checkpointer.py` | `get_checkpointer()` → process-wide `RedisSaver` (`langgraph-checkpoint-redis`) at `CHATBOT_CHECKPOINTER_REDIS_URL` (default `redis://localhost:6380/0` — **redis-stack, needs the RediSearch module**; the plain `:6379` Redis does not have it). Fails loud at startup. `thread_id = str(chat.pk)`. **No Postgres checkpoint tables.** |
| `test_manual.py` | Non-pytest 4-turn smoke script. Runs inside the api container. |

## `memory/` — evidence-only structured memory
| File | Role |
|------|------|
| `store.py` | `MemoryStore` — Redis I/O for `veda:mem:{tenant}:{session}:{frame,stack,episodic}` (STRING + 2 LISTs), sliding 4 h TTL, optimistic-lock `write_frame` (WATCH; **conflict aborts the write**, no retry). Soft-fails to empty on any Redis error. Own client, separate from the checkpointer. |
| `frame.py` | Pure functions (no I/O, no LLM, no `veda_core`): `QueryFrame` / `FilterFact` / `DrillLevel` TypedDicts, `harvest_frame` (from `engine_result["explain"]` — deterministic `build_explain` output, zero LLM), `is_topic_switch`, `merge_frame_post_execution` (harvested wins, reset on `new_topic` / topic-switch), `push_drill` / `pop_drill`, `render_frame_as_query`. |
| `classify.py` | `classify_delta(frame, message, episodic)` — the **ONE** constrained-output SLM call of the memory package (`delta_type` ∈ 6-value enum + grounded `slot_candidates`). Vocabulary gate (slot must be a substring of the message) + confidence gate (all-ungrounded → downgrade to "ambiguous"). Fails closed to "ambiguous". |

## `prompts/`
`common.py` (`today_str()`), `supervisor.py` (classify prompt + the `_DELTA_BLOCK`
addendum that makes the SAME call also return `delta_type` — the latency fix),
`delta_classify.py` (standalone fallback), `followup.py` (free-text rewrite, fallback only),
`standalone_check.py` (the "does this depend on history?" second opinion),
`smalltalk.py` (must never invent data facts).

## Gotchas
- **Entry point is `memory_read`, not `classify`** (the archived `MEMORY_ARCHITECTURE.md`
  design shows the reverse). See `docs/archive/MEMORY_ARCHITECTURE.md` and `docs/CHAT.md`
  §"shipped vs design".
- `chatbot/` is api-container code and **must not import `veda_core` / the `sm`** — that's
  why the schema-validation gate in the design never shipped; the engine re-validates.
- `context_resolve_node` **never asks for clarification** — "ambiguous" passes the raw
  message through and lets the engine's own refuse/clarify path handle it.
- `run.py`'s `history` param is accepted but every production caller passes `None` (the
  checkpointer owns history).
