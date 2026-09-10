# Production Memory Subsystem for VEDA (Enterprise NL2SQL)

Author: architecture design (grounded in this repo's actual code, not generic chatbot-memory
patterns). Every component below names the real file/function it replaces or hooks into.

---

## 0. What's actually broken today (verified in code, not assumed)

Current "memory" is `chatbot/state.py::ChatState.history`:

```python
history: Annotated[List[Turn], operator.add]   # Turn = {"role": str, "content": str}
```

Persisted forever, per session, via `chatbot/checkpointer.py::RedisSaver` on `redis-stack:6380`.
Two follow-up mechanisms consume it, and **both violate "memory is evidence, not intelligence"**:

1. `chatbot/nodes.py::classify_node` → `chatbot/prompts/supervisor.py::build_supervisor_user_prompt` —
   dumps the last 6 raw turns into the LLM and asks it to classify intent.
2. `chatbot/nodes.py::resolve_followup_node` → `chatbot/prompts/followup.py::FOLLOWUP_SYSTEM_PROMPT` —
   asks the LLM to **freely rewrite** the message into "a fully self-contained question using
   the conversation history for the missing context." This is an LLM inventing a query from
   raw text with zero schema grounding and zero validation. It is the single highest-risk
   hallucination surface in the whole platform, and it silently overwrites `resolved_query`
   which then goes straight into retrieval/SQL-gen.

Meanwhile, the engine (`veda_core/veda/pipeline.py`) already computes a fully structured,
schema-validated record of every answered query — `tr.set("query_understanding", ...)`,
`tr.set("sql_planning", ..., table=primary, filters=...)`, `_rec_plan()` for joins — and
`veda_core/veda/business_explain.py::build_explain()` deterministically re-derives
`understanding / data_used / operations / filters / validation / sql` **by parsing the
generated SQL with `sqlglot`, zero LLM involved**. This structured, validated output is
currently computed, shown to the user once, and thrown away. It is the natural raw material
for memory — nothing about it needs to be invented, because it already IS the evidence.

**Core design thesis:** stop asking an LLM to reconstruct context from text. Harvest the
structured, already-validated output the engine produces for every answered turn, store it as
a small typed object, and use a *narrow, constrained-output* SLM call only for the one thing
regex/keyword logic can't do — classifying *what kind of continuation* an utterance is
(refine / drill / drill-up / compare / new-topic / ambiguous). Never let the SLM supply a
column name, table name, or filter value that didn't come from validated schema/execution.

---

## 1. Memory architecture (three tiers, not one blob)

```
┌─────────────────────────────────────────────────────────────────────┐
│ Tier 1 — WORKING MEMORY (per session, current turn)                  │
│   QueryFrame  — the ONE structured analytical state. O(1) size.      │
│   DrillStack  — bounded stack of {dimension, value} breadcrumbs.      │
├─────────────────────────────────────────────────────────────────────┤
│ Tier 2 — SHORT EPISODIC BUFFER (last N=3 turns, capped, TTL'd)        │
│   Used ONLY for pronoun/ellipsis disambiguation hints to the SLM.     │
│   Never used to reconstruct SQL directly. Not the source of truth.   │
├─────────────────────────────────────────────────────────────────────┤
│ Tier 3 — LANGGRAPH EXECUTION CHECKPOINT (RedisSaver, unchanged infra) │
│   Only what LangGraph needs to resume a turn. history[] REMOVED      │
│   from ChatState (moved into Tier 2, capped, no operator.add growth).│
└─────────────────────────────────────────────────────────────────────┘
```

Nothing in Tier 1/2 is free text turned into SQL directly. Tier 1 is the only thing the
planner/SQL-generator ever reads for context. Tier 2 exists solely to give the classifier
SLM enough surface to recognize "this/it/that/ones" without letting it invent field values.

---

## 2. Components

| Component | Responsibility | Replaces / hooks into |
|---|---|---|
| `memory.frame.QueryFrame` | Typed structured analytical state (dataclass/TypedDict) | New |
| `memory.stack.DrillStack` | Bounded LIFO of drill-down breadcrumbs | New |
| `memory.store.MemoryStore` | Redis read/write for Tier 1 + Tier 2, TTL mgmt | New; sits next to `chatbot/checkpointer.py` |
| `memory.harvest.harvest_frame()` | Pure function: engine result → `QueryFrame` (no LLM) | Consumes `veda_core/veda/business_explain.py::build_explain()` output + `tr.to_dict()` |
| `memory.classify.classify_delta()` | ONE constrained-output SLM call: delta type + candidate slots | Replaces `resolve_followup_node`'s free rewrite; merges into `classify_node`'s existing call |
| `memory.merge.merge_frame()` | Deterministic, schema-validated merge of delta into current frame | New — the actual "intelligence" is here, not in the LLM |
| `memory.validate.validate_against_schema()` | Rejects any table/column/value not present in `sm` (semantic model) | Reuses the already-loaded semantic model (same one `veda_core/veda/pipeline.py` uses) |
| `memory_read_node` / `memory_write_node` | LangGraph nodes | New, added to `chatbot/graph.py` |

---

## 3. Data model

```python
# memory/frame.py
from typing import TypedDict, Optional, Literal

class FilterClause(TypedDict):
    column: str            # MUST exist in sm["columns"][f"{table}.{column}"]
    op: Literal["=", "!=", ">", ">=", "<", "<=", "in", "between"]
    value: str | int | float | list
    source: Literal["user_stated", "planner_validated", "executed_sql"]  # provenance — never "llm_guessed"

class TimeRange(TypedDict, total=False):
    column: str             # canonical temporal column, from sm, e.g. "created_at"
    start: Optional[str]    # ISO date, resolved by query/temporal_parser.py (deterministic)
    end: Optional[str]

class QueryFrame(TypedDict, total=False):
    version: int                       # bumped on every write — optimistic concurrency
    tenant: str
    session_id: str
    entity: Optional[str]              # primary table, e.g. "users_user" — from L3 Routing
    metric: Optional[str]              # aggregate/column being asked about, e.g. "count", "revenue"
    filters: list[FilterClause]
    group_by: list[str]                # validated columns only
    time_range: Optional[TimeRange]
    drill_path: list[dict]             # denormalized mirror of DrillStack for prompt rendering
    last_sql: Optional[str]            # verbatim, from the executed query (audit trail)
    last_row_count: Optional[int]      # NOT the rows themselves — no PII/data in memory
    last_status: str                   # "answered" | "refuse" | "clarify" | "unavailable"
    confidence: float                  # 1.0 if fully schema-validated, else discarded (see §14)
    updated_at: str                    # ISO timestamp
    turn_index: int
```

Note what's deliberately **absent**: no raw NL text, no full result rows, no free-form
"summary" string as the analytical source of truth (a `reply_text` NL summary can still be
shown to the user, but it is never re-parsed back into structured memory — one-way only).

---

## 4. Redis structure

```
veda:mem:{tenant}:{session_id}:frame        STRING (JSON, RedisJSON if available, else plain
                                              string blob) — the CURRENT QueryFrame. ~300-800
                                              bytes. TTL 4h sliding (EXPIRE on every touch).

veda:mem:{tenant}:{session_id}:stack        LIST — DrillStack, each element a small JSON
                                              {"dimension": "region", "value": "APAC"}.
                                              Capped at 10 (LTRIM after LPUSH). TTL 4h.

veda:mem:{tenant}:{session_id}:episodic     LIST — Tier 2 buffer, each element
                                              {"role","content_gist"} where content_gist is
                                              the user message VERBATIM (short) but the
                                              assistant side is a one-line templated summary
                                              (e.g. "answered: active_users=5,857"), NEVER the
                                              full markdown/table blob. Capped at 3
                                              (LTRIM 0 2). TTL 4h.

veda:mem:{tenant}:{session_id}:lock         STRING, short TTL (2s) — optimistic-lock guard so
                                              a slow SLM classify + a fast concurrent retry
                                              can't both write stale frame versions.
```

All four keys share one TTL policy (sliding 4h idle timeout — configurable via
`VEDA_MEMORY_TTL_SECS`, following this repo's existing `VEDA_*` env-override convention in
`apps/core/settings_bridge.py`). Expiry = clean slate, not corruption: a stale/expired frame
is never partially trusted; `memory_read_node` treats a Redis miss identically to turn 1.

**Size math:** 300-800 bytes (frame) + ~400 bytes (stack, 10×40B) + ~600 bytes (episodic,
3×200B) ≈ **under 2KB per active session**, flat, regardless of conversation length — versus
today's `history[]` which grows by ~1-3KB *per turn* forever (bounded only by Redis
eviction/OOM, and `redis-stack` in this compose has no `maxmemory-policy` set, unlike
`redis-cache`'s `allkeys-lru` — see `docker-compose.yml`'s `redis-stack` service definition,
which currently has **no memory cap at all**). At 10,000 concurrent active sessions: ~20MB.
At 1M sessions/day with 4h TTL and realistic overlap, steady-state footprint stays in the
tens-of-MB range, not GB — this alone is a production-readiness fix independent of the rest
of this design.

---

## 5. LangGraph integration

Modified graph (additions marked `+`, everything else is `chatbot/graph.py` unchanged):

```
classify
   |
   +-- memory_read_node (NEW, runs first, injects frame+stack+episodic into state)
   |
   +-- smalltalk ------------------------------------------------> smalltalk_node -> END
   +-- runtime_context ------------------------------------------------------.
   +-- (else) -> context_resolve_node (REPLACES resolve_followup_node) -.    |
                                                                          +---+--> call_engine_node
                                                                                     |
                                                    +-- answered -> memory_write_node (NEW)
                                                    |                  -> format_reply_node -> END
                                                    `-- refuse/error/clarify (memory_write SKIPPED)
                                                                       -> ask_clarification_node -> END
```

`ChatState` changes (`chatbot/state.py`):

```python
class ChatState(TypedDict, total=False):
    # history: Annotated[List[Turn], operator.add]   # REMOVED — moved to Redis Tier 2, capped
    frame: "QueryFrame"          # loaded by memory_read_node, mutated by context_resolve_node
    drill_stack: list[dict]      # loaded by memory_read_node
    delta_type: str              # "new_topic"|"refine"|"drill_down"|"drill_up"|"compare"|"ambiguous"
    delta_slots: dict            # SLM's proposed slot changes — RAW, pre-validation
    ... # everything else unchanged (message, session_id, tenant, action, resolved_query,
        # engine_result, status, reply_text, needs_clarification, sql, rows, ...)
```

`chatbot/checkpointer.py`'s `RedisSaver` is **kept as-is** for LangGraph's own
resume/replay semantics (that's an execution-engine concern, orthogonal to analytical
memory) — but since `history` no longer accumulates inside `ChatState`, the per-checkpoint
payload LangGraph persists stops growing turn-over-turn, which shrinks that Redis footprint
too as a side effect.

---

## 6. Memory update algorithm (`memory_write_node`)

Runs only when `call_engine_node` returns `status == "answered"` (never on refuse/error/
clarify/unavailable — hard-enforced, see §12).

```
def memory_write_node(state):
    engine_result = state["engine_result"]          # veda_core/veda/pipeline.py's _done() payload
    if state["status"] != "answered":
        return {}                                    # NO-OP. Evidence must be executed+validated.

    harvested = harvest_frame(engine_result)          # pure function, §3 harvest.harvest_frame
    # harvested = {entity, metric, filters[], group_by[], time_range, last_sql, last_row_count}
    # every field comes from engine_result["explain"] (business_explain.build_explain output)
    # or engine_result["table"]/["sql"] — i.e. ALREADY validated by L6a-L6c in pipeline.py.

    prev_frame = state.get("frame") or empty_frame(state["tenant"], state["session_id"])

    if state["delta_type"] == "new_topic":
        new_frame = harvested_as_fresh_frame(harvested)   # explicit reset, see §9
        new_stack = []
    elif state["delta_type"] == "drill_down":
        new_frame = merge_frame(prev_frame, harvested)
        new_stack = push_drill(state["drill_stack"], harvested)   # §10
    elif state["delta_type"] == "drill_up":
        new_stack = pop_drill(state["drill_stack"])
        new_frame = rebuild_frame_from_stack(new_stack, harvested)
    else:  # "refine" | "compare"
        new_frame = merge_frame(prev_frame, harvested)
        new_stack = state["drill_stack"]

    new_frame["version"] = prev_frame.get("version", 0) + 1
    new_frame["confidence"] = 1.0     # reaching here means schema-validated + executed
    MemoryStore.write_frame(state["tenant"], state["session_id"], new_frame, expected_version=prev_frame.get("version", 0))
    MemoryStore.write_stack(state["tenant"], state["session_id"], new_stack)
    MemoryStore.push_episodic(state["tenant"], state["session_id"],
                               role_gist=templated_gist(engine_result))   # NOT raw rows/markdown
    return {"frame": new_frame, "drill_stack": new_stack}
```

`write_frame` uses `WATCH`/`MULTI` (or a Lua script) keyed on `expected_version` — optimistic
concurrency so two overlapping turns for the same session (e.g. user double-submits) can't
race-corrupt the frame; the loser retries against the fresh version, never silently overwrites.

---

## 7. Memory retrieval algorithm (`memory_read_node`)

```
def memory_read_node(state):
    frame = MemoryStore.read_frame(state["tenant"], state["session_id"])   # O(1) GET
    stack = MemoryStore.read_stack(state["tenant"], state["session_id"])   # O(1) LRANGE
    episodic = MemoryStore.read_episodic(state["tenant"], state["session_id"])  # O(1), ≤3 items
    return {"frame": frame or {}, "drill_stack": stack or [], "history": episodic or []}
    # "history" here is the SHORT capped Tier-2 buffer, purely for the classify prompt —
    # NOT the old unbounded operator.add list.
```

No embedding lookups, no fuzzy search — single deterministic key reads. This satisfies
constraint #6 (no vector retrieval for memory) directly: the existing retrieval system
(`veda_core/retrieval/`, BGE-M3 + BM25) is for SCHEMA/COLUMN grounding, completely separate
from conversational memory, and stays untouched.

---

## 8. Follow-up resolution flow

Today's two-call sequence (`classify_node` SLM call → `resolve_followup_node` SLM call,
confirmed in `chatbot/prompts/supervisor.py` + `chatbot/prompts/followup.py`) is collapsed
into **one** SLM call that classify_node already makes — net latency **reduction**, not
addition.

**New combined classify prompt** (single call, replaces both existing prompts):

```
SYSTEM:
You are a strict classifier for an analytics assistant. You NEVER invent column
names, table names, or filter values — only choose from the CURRENT FRAME shown
below and the user's own words. Output ONLY JSON.

Current frame: {entity: "users_user", metric: "active_users", filters: [], group_by: [], time_range: null}
Last 3 turns (for reference resolution only — do not copy values from here that
aren't ALSO in the user's new message or the current frame):
  user: show active users
  assistant: answered: active_users=5,857

New message: "only Finance"

Classify into EXACTLY one delta_type:
  "new_topic"   - unrelated to the current frame's entity
  "refine"      - adds/overrides a filter or group_by on the SAME entity
  "drill_down"  - narrows into a new dimension level (e.g. named a specific value of a dimension)
  "drill_up"    - "go back" / "zoom out" / "remove that filter"
  "compare"     - asks to compare against a prior time_range or filter value
  "ambiguous"   - cannot be resolved with high confidence from the frame + message alone

If not "ambiguous", extract slot_candidates ONLY using words that appear in the
NEW MESSAGE itself (never invent a value the user didn't say):
{"delta_type": "...", "slot_candidates": {"filters": [{"column_hint": "...", "value": "..."}], "group_by_hint": null, "time_hint": null}, "reason": "<short phrase>"}
```

Note the prompt **never asks the SLM to name a real column** (e.g. `department`) — it asks
for a `column_hint` (the user's own word, "Finance"/"department") which `merge_frame()` then
resolves against `sm["domain_synonyms"]`/`sm["columns"]` **deterministically**, exactly the
way `veda_core/veda/generation.py`'s existing `_term_map` domain-synonym resolution already
works (line ~614-621 in `pipeline.py`) — reusing an existing, already-proven grounding
mechanism instead of inventing a new one.

`context_resolve_node` (replacing `resolve_followup_node`):

```
def context_resolve_node(state):
    delta = state["delta_type"]
    if delta == "ambiguous":
        return {"needs_clarification": True,
                "clarification_question": build_clarifying_question(state["frame"], state["message"])}
    candidate_frame = merge_frame(state["frame"], resolve_slot_candidates(state["delta_slots"], sm))
    if candidate_frame is REJECTED:            # validate_against_schema failed — see §14
        return {"needs_clarification": True,
                "clarification_question": build_clarifying_question(state["frame"], state["message"])}
    resolved_query = render_frame_as_query(candidate_frame)   # deterministic template, NOT LLM prose
    return {"resolved_query": resolved_query, "frame": candidate_frame}
```

`render_frame_as_query()` is a deterministic template (e.g. `"active_users in users_user
where department = Finance"`) fed to the existing engine exactly like any other
`resolved_query` — the engine's own deterministic/LLM SQL-gen path (`veda_core/veda/
pipeline.py` + `generation.py::generate_sql()`) is untouched and still does its own
independent validation (L6a-L6c) regardless of what memory hands it. Memory is a *better
input*, never a bypass of existing SQL validation.

---

## 9. Drill-down management

`DrillStack` = ordered list of `{dimension, value}`, mirrored in `frame.drill_path` for
prompt rendering. Push/pop only — no LLM involvement:

```
def push_drill(stack, harvested):
    if harvested.get("new_dimension") and harvested.get("new_value"):
        stack = stack + [{"dimension": harvested["new_dimension"], "value": harvested["new_value"]}]
    return stack[-10:]                      # capped depth

def pop_drill(stack):
    return stack[:-1] if stack else stack   # "go back" — no-op at root, never errors

def rebuild_frame_from_stack(stack, harvested):
    # Deterministically re-derive filters from the (now shorter) stack — e.g.
    # Revenue -> Region=NA -> City=LA, "go back" -> stack=[Region=NA] -> filters=[region=NA]
    filters = [{"column": s["dimension"], "op": "=", "value": s["value"], "source": "executed_sql"}
               for s in stack]
    return {**harvested, "filters": filters, "drill_path": stack}
```

Example trace for the prompt's drill-down scenario (Revenue → Region → NA → California → LA
→ "go back"):

| Turn | delta_type | Stack after |
|---|---|---|
| "Show revenue" | new_topic | `[]` |
| "By region" | refine (group_by) | `[]` |
| "North America" | drill_down | `[{region: NA}]` |
| "California" | drill_down | `[{region: NA}, {state: CA}]` |
| "Los Angeles" | drill_down | `[{region: NA}, {state: CA}, {city: LA}]` |
| "Go back" | drill_up | `[{region: NA}, {state: CA}]` |

Deterministic, O(1) per turn, zero SLM involvement for the stack mechanics themselves (the
SLM is only asked to classify *which* of these six delta types the utterance is).

---

## 10. Context reset logic (topic switching)

**Zero-cost detector, reusing existing pipeline output — no new component needed.** Every
query, follow-up or not, goes through `call_engine_node` → `veda_core/veda/pipeline.py`'s
retrieval stage (`[L3] Routing ... → primary: <table>`), which independently determines the
query's best-matching table from the 5-signal retrieval engine (BGE-M3 + BM25 + FK + value),
**with no knowledge of the current frame**. Compare:

```
def is_topic_switch(frame, engine_result) -> bool:
    if not frame or not frame.get("entity"):
        return False
    new_primary_table = engine_result["explain"]["data_used"]["table"]   # from business_explain
    return new_primary_table != frame["entity"] and engine_result_confidence_high(engine_result)
```

This runs **after** the engine call, as a `memory_write_node` pre-step, and is the
authoritative override regardless of what `classify_delta()` guessed pre-call — i.e. the
classifier's `delta_type` is a *fast, cheap prior* (used to decide whether `context_resolve_node`
needs to merge a frame into `resolved_query` at all, before the expensive engine call), but the
*actual* reset decision is confirmed deterministically against the engine's own independent
table-routing signal after the fact. If the SLM said "refine" but the engine independently
routed to a completely different table with high confidence, that's treated as a topic switch
and the OLD frame is discarded (archived, not merged) — **never silently blended**. This is
exactly the "Revenue conversation, then compliance incidents" case in the prompt: no explicit
"new topic" phrasing needed, no extra LLM call needed — the existing retrieval engine already
tells us.

---

## 11. Ambiguity handling

Ambiguous means: `classify_delta()` returned `"ambiguous"`, OR `merge_frame()`/
`validate_against_schema()` rejected a slot. In both cases: **no guess, ask.**

```
def build_clarifying_question(frame, message) -> str:
    if not frame or not frame.get("entity"):
        return "Could you tell me which data you'd like to look at?"
    return (f"I have your last question about {humanize(frame['entity'])}"
            f"{format_active_filters(frame)}. Did you mean to filter/change that, "
            f"or ask about something new?")
```

This reuses `ask_clarification_node` (`chatbot/nodes.py:325`, unchanged) as the terminal
node — memory's ambiguity path and the engine's existing refusal path converge on the same
"refuse-over-guess" UX the rest of the platform already uses (the prompt's own example —
"Show revenue" → "Show inactive ones" — is a textbook `ambiguous`: "inactive" has no
antecedent entity/column in the frame, and there's no `inactive` synonym mapped to
`revenue`'s table in `sm["domain_synonyms"]`, so `validate_against_schema` rejects it and
the user is asked, never assumed to mean "inactive users").

---

## 12. Hallucination prevention strategy (the load-bearing section)

Four independent barriers, not one:

1. **Provenance gate** — `memory_write_node` only fires on `status == "answered"`. A refused,
   errored, or clarify-needed turn writes nothing. Memory can only grow from proof.
2. **Vocabulary gate** — the SLM in `classify_delta()` is never shown a blank canvas; it can
   only emit `column_hint`/`value` strings that must **already appear in the user's own
   message**. It is structurally prevented from inventing a value that wasn't typed by the
   user this turn (the prompt says so, and `merge_frame` additionally cross-checks
   `slot_candidates.value` substring-matches `state["message"]` before accepting it — belt +
   suspenders, not trust-the-prompt-alone).
3. **Schema gate** — every `column_hint` must resolve to a real `sm["columns"][...]` entry or
   `sm["domain_synonyms"]` mapping for the frame's OWN table (same lookup
   `veda_core/veda/pipeline.py` already performs for `_term_map`, lines ~614-621). No match →
   rejected, not defaulted.
4. **Independent re-validation gate** — even after memory hands the engine a merged
   `resolved_query`, the engine's own existing L6a (value check), L6b (qualifier
   completeness), L6b+ (IR equivalence), L6c (read-only/parameterized) validation in
   `pipeline.py` still runs unmodified. Memory is a better *input*, never a bypass of
   correctness checks that already exist.

Rejected memory behaviors (per the prompt's instruction to challenge every assumption) and
why:

| Rejected idea | Why |
|---|---|
| Storing the LLM's `resolved_query`/rewritten NL text as memory | Exactly today's bug — text the LLM invented becomes tomorrow's "fact." Replaced by harvesting engine output post-execution. |
| Embedding-based semantic memory recall ("find similar past questions") | Constraint #6 explicitly rejects this without strong justification; deterministic frame+stack fully covers the prompt's use cases with less latency and zero false-positive-recall risk. |
| Letting the classifier SLM output actual SQL fragments or column names directly | Removes the schema gate entirely; rejected. |
| "Fuzzy" confidence scores (e.g. 0.73) driving auto-merge | Enterprise correctness needs a bright line, not a threshold someone will eventually tune wrong. Binary: schema-validated+executed (1.0) or discarded (ask). See §14. |
| Summarizing entire conversation into memory via LLM | Violates "memory is evidence not intelligence" directly; only considered as a LAST resort for the episodic buffer's assistant-side gist, and even that is a **template**, not an LLM call (see §17). |

---

## 13. Validation pipeline

```
validate_against_schema(table, slot_candidates, sm) -> ValidatedSlots | Rejected:

  1. table must be in sm["tables"] (or implicitly valid — same table as frame.entity)
  2. for each filter candidate:
       resolved_col = sm["domain_synonyms"].get(candidate.column_hint.lower())
                      or exact match in sm["columns"][f"{table}.{candidate.column_hint}"]
       if resolved_col is None: REJECT this filter (do not drop silently — mark unresolved)
  3. if ANY filter in this turn's candidates is unresolved -> whole delta is "ambiguous"
     (partial silent application of some filters but not others is WORSE than asking,
     because it produces a plausible-looking but incomplete query — reject the whole turn)
  4. value type-check against sm["columns"][...]["dtype"] if present (e.g. reject a string
     value against a boolean/enum column outside its known value set, when the semantic
     model's column_values/enum metadata is available — same value-existence check
     `veda_core/query/value_resolver.py::column_values_lookup` already performs)
  5. group_by columns: same resolution as filters, must exist in sm for this table
  6. time_range: delegate entirely to query/temporal_parser.py (already deterministic,
     already used by the engine) — memory never parses dates itself
```

Step 3 is a deliberate, opinionated design choice worth calling out: **partial application of
a multi-filter delta is rejected wholesale**, not merged partially. A user who says "Finance,
last month" and only "Finance" resolves cleanly should be asked to clarify the time range
rather than silently getting a query filtered by Finance only — because a plausible-but-
incomplete result is more dangerous in a BI context (it looks authoritative) than an explicit
clarifying question.

---

## 14. Confidence scoring

Deliberately **binary**, not probabilistic, per the reasoning in §12/§13:

```
confidence = 1.0   if  status == "answered" AND all slots passed validate_against_schema
confidence = 0.0   otherwise  →  routed to ask_clarification_node, frame NOT written
```

There is no 0.3-0.9 gray zone stored in Redis. The SLM's own token-probability/uncertainty is
never surfaced as a memory confidence score — it's discarded after `classify_delta()` produces
a `delta_type` + `slot_candidates`; what matters downstream is only whether those candidates
survive deterministic schema validation. This is the single biggest simplification versus
generic chatbot-memory literature (which usually proposes weighted/decaying confidence) — and
it's *correct* for this domain: an enterprise BI answer is either grounded or it isn't.

---

## 15. State lifecycle

```
CREATED  --(first "answered" turn)-->  ACTIVE
ACTIVE   --(refine/drill, "answered")-->  ACTIVE (version++)
ACTIVE   --(topic switch detected, §10)-->  ARCHIVED (old) + CREATED (new)
ACTIVE   --(ambiguous / rejected slot)-->  ACTIVE (unchanged, clarification asked, no write)
ACTIVE   --(TTL idle 4h)-->  EXPIRED (Redis TTL, no explicit archival job needed)
ARCHIVED --(optional, see §16)-->  discarded (not restorable; a resumed old topic re-CREATEs)
```

"ARCHIVED" is not actually persisted anywhere by default — a topic switch simply overwrites
the frame key with a fresh `CREATED` frame. If audit/analytics on topic-switch frequency is
wanted later, `memory_write_node` can optionally append the outgoing frame's final state to a
separate low-priority analytics stream (Celery task via the existing `worker`/`beat`
infrastructure in `docker-compose.yml`) — out of the request's hot path, not a design
requirement for v1.

---

## 16. Memory pruning strategy

Three layers, all already-idiomatic for this stack:

1. **TTL (primary mechanism)** — every key in §4 has a sliding 4h TTL, refreshed
   (`EXPIRE`) on every read AND write. No cron job needed; Redis expires keys natively. This
   alone bounds memory to "active sessions only," unlike today's unbounded `RedisSaver`
   checkpoints.
2. **Stack depth cap** — `LTRIM` to 10 elements on every `push_drill` (§9). A pathological
   50-level drill-down conversation still only costs ~400 bytes.
3. **Episodic buffer cap** — `LTRIM` to 3 elements (§4). Old turns fall off; they were never
   the source of truth anyway (§0), so losing them is harmless — the frame already carries
   forward everything that matters.

No LRU/eviction-under-memory-pressure logic is needed given the ~2KB/session footprint (§4);
this is orders of magnitude below the point where `redis-stack`'s (currently absent, see §4)
`maxmemory` would even become relevant. **Recommendation**: add
`--maxmemory 256mb --maxmemory-policy volatile-ttl` to `docker-compose.yml`'s `redis-stack`
service regardless, as defense-in-depth (mirrors `redis-cache`'s existing
`allkeys-lru` config), since with TTLs everywhere `volatile-ttl` eviction is safe (evicts
soonest-to-expire keys first under pressure, never breaks correctness — it just ages sessions
out slightly early under extreme load).

---

## 17. Conversation summarization strategy (only if truly necessary — and it isn't)

Constraint asked to justify this "only if truly necessary." **Verdict: not necessary, reject
LLM-based summarization entirely.** Reasoning:

- The `QueryFrame` (§3) already IS the summary — structurally, losslessly for everything that
  matters to the planner (entity/filters/metric/time/drill path).
- The episodic buffer's assistant-side entries (§4) are **template-rendered**, not
  LLM-summarized: `templated_gist(engine_result)` is a one-line f-string
  (`f"answered: {metric}={value}"` or `f"answered: {row_count} rows"`), built from the same
  `business_explain.py` output, zero LLM cost.
- If a UI wants a human-readable "conversation so far" recap (e.g. for a session-list preview),
  render it deterministically from the frame (`"Users, active_users, filtered by Finance,
  last month"`) rather than asking an LLM to summarize — cheaper, faster, and can't drift from
  what's actually stored.

The only scenario where LLM summarization would earn its keep is cross-session, long-term
user preference memory ("this user always wants Finance-scoped views") — explicitly out of
scope for this design (the prompt's examples are all single-session analytical continuity),
and if pursued later it should be a background/offline job, never inline in the hot request
path.

---

## 18. Failure cases

| Failure | Behavior |
|---|---|
| Redis (Tier 1/2) unreachable | `memory_read_node` catches, returns empty frame/stack — turn proceeds as if session start (degrades to "always treat as new_topic"), matching `chatbot/checkpointer.py`'s existing "fail loud at startup, soft-degrade per-request" split, but Tier-1/2 reads are NOT LangGraph's own checkpoint (which still fails loud) — only the analytical frame degrades softly. |
| `classify_delta()` SLM call times out / errors | `call_slm()` already raises `RuntimeError` uniformly (`veda_core/slm/_call_slm.py`); `context_resolve_node` catches it and forces `delta_type = "ambiguous"` → clarification, never a guess. |
| Frame version conflict (concurrent double-submit) | Optimistic-lock write rejected; the later write reloads current frame and retries merge once; second conflict → proceeds anyway with a warning log (never blocks the user turn). |
| Engine returns `"answered"` but `business_explain.build_explain()` itself raises (sqlglot parse failure on an edge-case SQL shape) | Already guarded in `pipeline.py` (`logger.exception("business_explain failed — end-user explainability omitted")`) — `harvest_frame()` gets `explain=None` and skips the write entirely rather than harvesting partial/malformed data; user still gets their answer, just no frame update that turn. |
| Drill-down stack references a dimension whose column was since dropped from the semantic model (schema changed mid-session) | `rebuild_frame_from_stack` re-validates every stack entry against current `sm` on `drill_up`/`drill_down`; a now-invalid entry is dropped from the rebuilt filters and flagged, not silently kept. |
| Topic-switch detector (§10) itself wrong (false positive resets a still-relevant frame) | Low blast radius by design: the user's very next follow-up either confirms the new topic (fine) or gets asked to clarify because the merge now fails validation against the new frame's table — never produces a wrong-but-confident answer, only a possibly-unnecessary clarifying question, which is the acceptable failure direction per constraint #1. |

---

## 19. Edge cases

- **"Compare with previous month"** — `delta_type = "compare"`; `merge_frame` clones the
  current frame's `time_range`, computes the prior period deterministically (reuse
  `query/temporal_parser.py`'s existing relative-date resolution, shifted back one period —
  no LLM math), and produces a two-branch resolved_query (or two sequential engine calls,
  implementation choice) — never asks the SLM to compute a date range itself.
- **"Compare with Sales"** (prompt's topic-adjacent comparison, different entity/dimension
  value on the same metric) — resolves as `refine` with a `group_by`/filter alternate value
  from the CURRENT frame's already-known dimension (e.g. `department`), not a topic switch,
  because the retrieval engine's table routing (§10) still lands on the same primary table.
- **First turn ever (no frame)** — `memory_read_node` returns an empty frame;
  `classify_delta()` always short-circuits to `"new_topic"` when `frame.entity` is unset
  (deterministic pre-check in `context_resolve_node`, skipping the SLM call entirely for this
  case — cheap, and matches `chatbot/nodes.py`'s existing pattern of skipping SLM calls when
  a deterministic answer is already known, e.g. `_canned_smalltalk_reply`).
- **User explicitly says "start over" / "new question"** — deterministic keyword fast-path
  (mirrors `_GREETING_RE` etc. in `chatbot/nodes.py`) forces `delta_type="new_topic"` without
  even calling the SLM classifier, same "instant deterministic path" philosophy already used
  for greetings.
- **Drill-up past the root** (`pop_drill` on an empty stack) — no-op, frame unchanged, no
  error; if the user then asks something ambiguous, that's handled by the ambiguity path, not
  a stack-underflow crash.
- **Multi-tenant isolation** — every Redis key is namespaced `veda:mem:{tenant}:{session_id}`
  (mirrors the existing `veda:sm:{source}:{tenant}` convention already used for semantic
  models per `docker-compose.yml`'s `inference` service comment: "read the per-source
  assembled semantic models from redis... so a scoped/cross-source query uses each source's
  OWN model") — memory never leaks across tenants by construction, not by convention.

---

## 20. Latency analysis

Baseline measured live on this deployment this session (external Ollama host, `qwen2.5-coder:7b`):
classify ≈ 3.3s, SQL-gen ≈ 4.4s (see prior debugging trace in this session's transcript — 546
in/22 out tokens for classify, 614 in/21 out for SQL-gen).

| Step | Today | Proposed | Delta |
|---|---|---|---|
| Classify (intent) | 1 SLM call (~3.3s) | 1 SLM call, slightly larger prompt (frame + delta taxonomy instead of raw 6-turn history) (~3.3-3.8s) | ~flat |
| Follow-up resolution | 1 SLM call when history exists (`resolve_followup_node`, ~2-3s typical per this session's traces) | **0 SLM calls** — deterministic `merge_frame`/`validate_against_schema`, sub-millisecond | **-2 to -3s** on every follow-up turn |
| Memory read | none | 3× O(1) Redis GET/LRANGE, <5ms total | +~5ms |
| Memory write | none | 1-2× O(1) Redis SET/LPUSH + 1 optimistic-lock check, <10ms total | +~10ms |
| SQL generation | unchanged | unchanged (better-grounded input, same latency) | flat |

**Net effect: follow-up turns get FASTER (one fewer LLM round trip), fresh-topic turns are
unaffected (~15ms Redis overhead vs. ~7-8s of SLM+retrieval time already spent) —** the memory
subsystem pays for itself in latency, it doesn't add a budget line item.

---

## 21. Redis storage impact

Covered in depth in §4 and §16. Summary: **flat ~2KB/session** (vs. today's unbounded
per-turn growth with a `redis-stack` instance that currently has no `maxmemory` cap set at
all — a real production risk independent of this redesign, flagged as a recommended fix
regardless of whether this memory redesign ships). At 50,000 concurrent active sessions:
~100MB. This is a rounding error next to the `model_cache` volume or Postgres.

---

## 22. Token usage analysis

| Path | Today (tokens, approx) | Proposed |
|---|---|---|
| Classify prompt | ~350-550 in (raw 6-turn history dump, unbounded per turn's content length) | ~250-400 in (frame JSON is fixed-size regardless of conversation depth; episodic buffer capped at 3 short entries — **bounded**, unlike raw history which grows with each turn's message length) |
| Follow-up rewrite prompt | ~300-600 in + ~20-80 out, EVERY follow-up turn | **0** — eliminated |
| Net effect on a 10-turn drill-down session | Classify prompt grows turn-over-turn (more history = more input tokens each time) + N follow-up rewrites | Classify prompt size is **flat** turn 2 through turn N (frame doesn't grow with conversation length, only with distinct active filters, which BI questions bound naturally to a handful) |

This directly satisfies constraint "avoids unnecessary token usage" — and does so by
construction (bounded data structure) rather than by prompt-engineering discipline that could
erode over time.

---

## 23. SLM prompt design

Full `classify_delta()` prompt shown in §8. Design principles applied:

- **Enumerable output space** (`delta_type` is a closed 6-value enum) — 7-8B instruct models
  are reliably good at closed-set classification, materially worse at open-ended generation
  needing to preserve exact grounding (which is what today's `FOLLOWUP_SYSTEM_PROMPT` demands
  and is exactly the risky part being removed).
- **Hints, not authority** — `slot_candidates` are proposals the deterministic layer verifies,
  never applied directly. This means a 7B model's occasional sloppy extraction (e.g. hinting
  `"dept"` instead of `"department"`) is harmless — `validate_against_schema`'s synonym
  resolution (reusing `sm["domain_synonyms"]`, already tuned for this) absorbs it, or rejects
  it cleanly into a clarifying question.
- **No chain-of-thought asked for** — `"reason"` field is one short phrase for logs/debugging,
  not a scratchpad the model needs tokens to think through; keeps `num_predict` small (mirrors
  `generate_sql()`'s existing `num_predict=256`/`num_ctx=2048` discipline in
  `veda_core/veda/generation.py`).
- **Deterministic decoding** — `temperature=0, seed=0` for `classify_delta()`, same rationale
  already documented in `generation.py`'s comment: "SQL generation must be DETERMINISTIC: the
  same question had been returning different WHERE clauses... run-to-run at temperature 0.1."
  Classification needs the same reproducibility guarantee.

---

## 24. API contracts

New internal module, mirrors this repo's existing module-per-concern layout
(`veda_core/query/value_resolver.py`, `fk_path_resolver.py`, etc. as the pattern):

```python
# veda_core/query/memory_frame.py  (or chatbot/memory/ if kept API-tier-side — see §26 note)

def harvest_frame(engine_result: dict) -> dict: ...
def merge_frame(prev: QueryFrame, harvested_or_resolved: dict) -> QueryFrame | Rejected: ...
def validate_against_schema(table: str, slots: dict, sm: dict) -> ValidatedSlots | Rejected: ...
def render_frame_as_query(frame: QueryFrame) -> str: ...
def push_drill(stack: list, harvested: dict) -> list: ...
def pop_drill(stack: list) -> list: ...
def rebuild_frame_from_stack(stack: list, harvested: dict) -> dict: ...

# memory/store.py  (Redis I/O, mirrors chatbot/checkpointer.py's process-wide-singleton style)

class MemoryStore:
    @staticmethod
    def read_frame(tenant: str, session_id: str) -> QueryFrame | None: ...
    @staticmethod
    def write_frame(tenant: str, session_id: str, frame: QueryFrame, expected_version: int) -> bool: ...
    @staticmethod
    def read_stack(tenant: str, session_id: str) -> list[dict]: ...
    @staticmethod
    def write_stack(tenant: str, session_id: str, stack: list[dict]) -> None: ...
    @staticmethod
    def read_episodic(tenant: str, session_id: str) -> list[dict]: ...
    @staticmethod
    def push_episodic(tenant: str, session_id: str, role_gist: dict) -> None: ...

# chatbot/nodes.py additions

def memory_read_node(state: ChatState) -> dict: ...
def context_resolve_node(state: ChatState, config: RunnableConfig) -> dict: ...   # replaces resolve_followup_node
def memory_write_node(state: ChatState) -> dict: ...
```

`classify_delta()` reuses the existing `call_slm()` contract unchanged
(`veda_core/slm/_call_slm.py`) — no new SLM transport, no new backend, `purpose="classify_delta"`
label for tracing (consistent with this session's earlier finding that `purpose` isn't yet
wired to the external monitor — worth doing regardless, independent of this design).

---

## 25. Pseudocode (end-to-end turn)

```
def handle_turn(session_id, tenant, message):
    frame, stack, episodic = memory_read_node(session_id, tenant)          # O(1) Redis

    if is_deterministic_fast_path(message):                                 # greeting/thanks/
        return handle_fast_path(message)                                    # bye/date — unchanged

    delta_type, delta_slots = classify_delta(message, frame, episodic)      # 1 SLM call
    # (folded into classify_node's existing single call — see §6/§8)

    if not frame.entity:
        delta_type = "new_topic"                                            # deterministic override

    if delta_type == "ambiguous":
        return ask_clarification(build_clarifying_question(frame, message))

    validated = validate_against_schema(frame.entity, delta_slots, sm)
    if validated is REJECTED:
        return ask_clarification(build_clarifying_question(frame, message))

    resolved_query = render_frame_as_query(merge_frame(frame, validated)) \
                     if delta_type != "new_topic" else message

    engine_result = call_engine(resolved_query, tenant, session_id)         # unchanged HTTP call
                                                                              # to inference tier

    if engine_result.status == "answered":
        if is_topic_switch(frame, engine_result):                           # deterministic re-check
            frame, stack = fresh_frame(), []
        harvested = harvest_frame(engine_result)                            # no LLM, sqlglot-parsed
        frame, stack = update_frame_and_stack(frame, stack, delta_type, harvested)
        memory_write_node(session_id, tenant, frame, stack, engine_result)  # O(1) Redis, versioned
        return format_reply(engine_result)
    else:
        return ask_clarification(engine_result.feedback)                    # frame untouched
```

---

## 26. LangGraph node placement

```python
# chatbot/graph.py (additions only)
g.add_node("memory_read", memory_read_node)
g.add_node("context_resolve", context_resolve_node)      # renamed from resolve_followup
g.add_node("memory_write", memory_write_node)

g.set_entry_point("classify")
g.add_edge("classify", "memory_read")                    # or fold memory_read INTO classify_node
                                                            # itself if a single node is preferred —
                                                            # both are valid; separate node keeps
                                                            # classify_node's existing unit tests
                                                            # untouched (lower migration risk).
g.add_conditional_edges("memory_read", _route_after_classify, {
    "smalltalk": "smalltalk", "runtime_context": "call_engine",
    "resolve": "context_resolve", "direct": "call_engine",
})
g.add_edge("context_resolve", "call_engine")
g.add_conditional_edges("call_engine", _route_after_engine_with_memory, {
    "answered": "memory_write", "other": "ask_clarification",
})
g.add_edge("memory_write", "format_reply")
g.add_edge("format_reply", END)
g.add_edge("ask_clarification", END)
```

Placement rationale: `memory_write` sits strictly between `call_engine` and `format_reply` —
after validation/execution (evidence exists) but before the user-facing reply is finalized
(so a memory-write failure can be logged without blocking `format_reply`, matching this
codebase's existing "best-effort side channel never blocks the main reply" pattern already
used for `_emit()` progress callbacks in `chatbot/nodes.py:120-130`).

---

## 27. Complete sequence diagram

```
User            api(chatbot)         Redis(mem)        inference tier          Redis(mem)
 |  "only Finance"  |                    |                    |                     |
 |----------------->|                    |                    |                     |
 |                  |--read frame/stack->|                    |                     |
 |                  |<--frame{entity:users_user, metric:active_users}--|            |
 |                  |                    |                    |                     |
 |                  |--classify_delta (1 SLM call, ~3.3s)------------------------->  |
 |                  |<--{"delta_type":"refine","slot_candidates":{"filters":[{"column_hint":"Finance"...}]}}
 |                  |                    |                    |                     |
 |                  |--validate_against_schema("Finance"->department, deterministic)|
 |                  |--render_frame_as_query()-------------->|                     |
 |                  |                    |   "active_users in users_user           |
 |                  |                    |    where department = Finance"          |
 |                  |------------------------------------------------------------->|
 |                  |                    |          [L1..L7 pipeline, unchanged]   |
 |                  |                    |          generate_sql() SLM call (~4.4s)|
 |                  |<---engine_result{status:"answered", explain:{...}}-----------|
 |                  |                    |                    |                     |
 |                  |--harvest_frame(engine_result), no LLM-->|                     |
 |                  |--write frame(v2), push_drill(none, "refine")----------------->|
 |                  |                    |                    |    write frame v2   |
 |<--"Finance active users: 1,204"-------|                    |                     |
```

---

## 28. Trade-offs

| Decision | Trade-off accepted | Why it's the right one here |
|---|---|---|
| Merged classify+delta into one SLM call | Slightly more complex single prompt/schema to maintain | Net latency win (§20) and one fewer failure mode to reason about, versus two separate calls that must agree with each other |
| Binary confidence (no gray zone) | More clarifying questions than a fuzzy-threshold system might ask | Enterprise BI correctness > conversational smoothness (explicit platform requirement) |
| Reject-whole-delta on partial slot validation failure (§13) | User re-asked more often on multi-filter follow-ups | Prevents silently-incomplete-but-confident-looking queries, the worst failure mode in a BI tool |
| No embedding-based memory recall | Can't answer "what did I ask about revenue two topics ago" | Out of scope per explicit constraint #6/#9; a `session_id`-scoped `list_conversations` (already exists, `apps/chat/services.py:72`) covers cross-session recall at the UI level without polluting analytical memory |
| Episodic buffer capped at 3, not semantically retrieved | Loses exact phrasing beyond 3 turns | The frame (not phrasing) is the source of truth; phrasing beyond disambiguating "it"/"that" has no analytical value |
| `redis-stack` kept for LangGraph checkpoints, separate `veda:mem:*` keys for analytical memory (same Redis instance) | Two logical schemas in one Redis, slightly more key-namespace discipline required | Avoids standing up a 4th Redis instance in an already-3-Redis-instance topology (`docker-compose.yml`'s explicit design note about redis-broker/redis-cache/redis-stack separation) — reuses existing infra, adds a naming convention instead of new ops surface |

---

## 29. Production recommendations

1. **Ship in this order** (each independently valuable, lowest-risk first):
   a. Add `maxmemory`/`volatile-ttl` to `redis-stack` in `docker-compose.yml` (§16) — zero
      code change, closes an existing unbounded-growth risk today, independent of everything
      else here.
   b. Introduce `QueryFrame` + `MemoryStore` + `harvest_frame()` write path
      (`memory_write_node`) WITHOUT yet changing `resolve_followup_node` — i.e. start
      recording structured memory passively while the old follow-up mechanism keeps running,
      so the harvested frames can be validated against real traffic before anything depends
      on them.
   c. Only after (b) is observed to produce correct frames in shadow mode, cut over
      `context_resolve_node` to replace `resolve_followup_node` and remove `history` from
      `ChatState`.
2. **Instrument before cutting over** — log `(delta_type, validation_result, topic_switch_bool)`
   for every turn during the shadow period; this is cheap (structured fields, not raw text)
   and gives a concrete before/after clarification-rate metric to justify the cutover.
3. **Thread `purpose` labels through to the external Ollama call** (flagged earlier this
   session as a separate improvement) — makes `classify_delta` calls independently visible in
   the shared monitor, useful for exactly this kind of validation.
4. **Do not let this design creep into a generic "chat memory" library** — every extension
   request going forward should be checked against constraint #2 ("memory is evidence, not
   intelligence"): if a proposed feature requires the memory layer to infer something no
   validated component already proved, it belongs in the classify/merge layer's
   *validation rules*, not as a new trusted LLM output.
