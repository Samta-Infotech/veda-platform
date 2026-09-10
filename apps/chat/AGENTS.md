# apps/chat/ — the conversational API surface

The DRF/SSE layer around the `chatbot/` LangGraph supervisor. Turn flow, memory, and
visualization: [../../docs/CHAT.md](../../docs/CHAT.md). Wire contract:
[../../CHAT_API_CONTRACT.md](../../CHAT_API_CONTRACT.md).

| File | Role |
|------|------|
| `views.py` | 4 APIViews (`AllowAny` + manual `_resolve_user` → **401** for anonymous). `ConversationQueryView` is the turn entrypoint: RBAC gate 1 → scope resolution → persist user msg → SSE stream or JSON. Also `_serialize_history_message`, the denied-turn synthetic-turn path, the `_sse_generator`. |
| `services.py` | `ConversationQueryService` — resolve/create chat, persist messages, `run_turn()` (calls `chatbot.run.run_chat_turn`; for streaming spawns a daemon thread and bridges the graph's sync `on_event` callback into a generator via a `queue.Queue`), `_build_reply_events` (content blocks / visualizations / explainability / usage / insights). |
| `models.py` | `ChatSession` (name, owner FK, soft-delete) + `ChatMessage` (type / content / `metadata` JSON / `feedback` / `comment` — the last two have **no writer**, soft-delete). `ChatSession.pk` doubles as the LangGraph `thread_id`. |
| `serializers.py` | Input-only: `ConversationQuerySerializer`, `CreateConversationSerializer`, `ConversationHistorySerializer` (reads `request.query_params` — history is **GET**). |
| `urls.py` | `conversations/query`, `conversations/create`, `conversations/list`, `conversations/history` under `/api/v1/`. |
| `visualization.py` | Pure deterministic `VisualizationRecommender` — `(cols, rows[, analytics])` → 0+ `VisualizationSpec`. `_kind` honours the engine's semantic role over structural dtype. `_CONFIDENCE_THRESHOLD = 0.6`. Multi-viz → one `{"visualizations":[…]}` event. |
| `table_rendering.py` | Pure markdown-table rendering (numeric right-align, 20-row cap, `\|` escaping, `project_display_columns`). |
| `thinking_messages.py` | Pure dict — internal pipeline phase ids → business-friendly progress strings. Never exposes "SQL" / table names / "supervisor" / "RAG". |
| `turn_events.py` | Pure `TurnEventAccumulator` — folds the `{event,data}` stream into `content_blocks` / `summary` / `thinking` / `explainability` / `usage` / `insights`. Shared by the JSON and SSE paths. |
| `migrations/0002_seed_dummy_admin_user.py` | Seeds dev `admin`/`admin123`. Marked "remove once real auth replaces LoginView" (auth has since moved to `apps/authentication`). |

## Gotchas
- **Chat calls the engine only over HTTP** (`apps/query/inference_client.stream_hybrid_query`)
  — never a direct `veda_core` import. This boundary is why the chat memory layer can't call
  the semantic model to validate.
- **A denied turn is a synthetic turn**, not a raw 403 — it still creates the chat + user
  message and streams/persists an access-denied assistant turn.
- `feedback` / `comment` fields, `soft_delete()`, `MessageType.SYSTEM` / `TOOL` — all
  skeleton (no writer / no route / never emitted).
- Two persistence planes: Django `ChatMessage` rows (durable, the history API) vs the
  LangGraph RedisSaver checkpoint on redis-stack `:6380` (evictable). See `chatbot/AGENTS.md`.
