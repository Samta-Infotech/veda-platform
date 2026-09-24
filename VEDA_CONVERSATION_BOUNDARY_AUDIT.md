# VEDA — Conversation State, STM & Context Resolution: architecture audit

Audit only. Nothing in this document was implemented. Every claim below is cited to a
file and line read during the audit, or to a runtime probe run against the live stack on
2026-09-23.

---

## A. Current bug trace — where `financial` enters

### The path, as it actually runs

| # | Where | What happens |
|---|---|---|
| 1 | `apps/chat/views.py:150-176` | RBAC resolved fresh (`resolve_query_scope`, `compute_data_scope`), `ConversationQueryService` built |
| 2 | `chatbot/run.py` → `chatbot/graph.py:118-159` | LangGraph: `memory_read → classify → context_resolve → call_engine → memory_write → format_reply` |
| 3 | `chatbot/nodes.py::memory_read_node` | loads the `QueryFrame` from Redis (`MemoryStore`) |
| 4 | `chatbot/nodes.py::classify_node` | ONE SLM call emits `action` + `delta_type` (merged) |
| 5 | `chatbot/nodes.py:1682 context_resolve_node` | calls `render_frame_as_query(frame, message, delta_type, …)` |
| 6 | **`chatbot/memory/frame.py:1304`** | `return f"{message} (for {ctx})".strip()` |
| 7 | **`chatbot/memory/frame.py:455-458`** | `ctx` is built by `_describe_frame`: |
| 8 | `chatbot/nodes.py:1985` | `query = state["resolved_query"]` — **a single string** is all that crosses |
| 9 | `apps/query/inference_client.py:134-154` | POST `/v1/run_hybrid_query/stream` with `{query, source_id, tenant, source_ids, flags}` |
| 10 | `inference/routes/hybrid.py:263-300` | `run_hybrid_query(req.query, …)` — **`flags` is accepted and never forwarded** |
| 11 | `veda_core/veda/pipeline.py` → `veda_core/veda/validation.py:404` | `qualifier_completeness(query, sql, sm)` |

### The exact line that injects it

```python
# chatbot/memory/frame.py:455
raw_entity, display = frame.get("entity"), frame.get("entity_display")
if display and raw_entity and display.lower() != raw_entity.lower():
    entity = f"{display} ({raw_entity})"
```

`entity_display` is set at `frame.py:287` from `explain.data_used.datasets[0]` — the
engine's own business label. For `accounts_generalledger` that label is
**"Single Financial Transactions"**.

So:

```
user typed   : only the debit ones
sent to engine: only the debit ones (for Single Financial Transactions (accounts_generalledger))
```

### Why the engine then refuses

`qualifier_completeness` (`validation.py:404-413`) states its own contract:

> *"every CONTENT token the user named must appear somewhere in the generated SQL"*

It has no way to know which tokens the user named. Verified live against the running
engine:

```
financial      stripped=False      <- NOT a stopword, so it counts as user content
transactions   stripped=False
single         stripped=True
debit          stripped=False
gate vocabulary size: 271
```

`financial` then substring-matches the real column `financial_year_id` on the queried
table, so `_names_entity_column` classifies it as a **dropped attribute**, and the gate
refuses:

```
I couldn't map 'financial' to any column or value in the data.
Did you mean: financial_year_id
```

**`debit` is real** — `accounts_generalledger.entry_type` has 476 `DEBIT` rows (verified
against `homzhub_prod`). The user's own word was answerable. The refusal is caused
entirely by a word this layer added.

### Scope of the contamination

`frame.py:1267` — `if delta_type in ("new_topic", "ambiguous"): return message`. So the
prefix is added on **refine / replace / remove / drill_down / drill_up / compare** only:
precisely the follow-ups, i.e. exactly the feature under discussion.

---

## B. Current vs required representation

```
CURRENT
  user message ─┐
                ├─► render_frame_as_query ─► ONE natural-language string ─► engine
  QueryFrame  ──┘                            (memory metadata reads as user language)

REQUIRED
  user message ──────────────────────────► engine   (verbatim, never edited)
  QueryFrame (already structured) ───────► engine   (as typed context, not prose)
  delta (already structured) ────────────► applied in Python, before the call
```

**The important finding, and it changes what needs building:** VEDA does **not** need a
structured conversation state — it already has one. `QueryFrame`
(`chatbot/memory/frame.py:106-143`) is typed and holds `entity`, `entity_display`,
`filters: List[FilterFact]`, `group_by`, `measures`, `available_measures`, `order_by`,
`limit`, `source_id`, `route`, `drill_path`, `last_sql`, `last_row_count`, `base_query`.

The contamination is not a missing-state problem. It is a **flattening problem at the
last inch**: structured state is serialized to English at `frame.py:1304`, one function
call before the HTTP boundary, because the boundary accepts nothing else.

---

## C. Existing components

| Requirement | Existing VEDA component | Status | Reuse? | Required change |
|---|---|---|---|---|
| Structured conversation state | `QueryFrame` (`frame.py:106`) | **EXISTS — works** | Yes | none |
| Structured delta vocabulary | `DELTA_TYPES` (`prompts/delta_types.py:34`) — `new_topic, refine, replace, remove, drill_down, drill_up, compare, ambiguous` | **EXISTS — works** | Yes | none |
| Delta applied deterministically | `apply_context_delta` (`frame.py:1004`) — Python mutates, model never does | **EXISTS — works** | Yes | none |
| Delta validation / grounding guards | `frame.py:1019-1027` (field must exist in frame; value must appear verbatim in the message) + `classify.py:64-69` | **EXISTS — works** | Yes | none |
| Drill stack (push/pop/rebuild) | `drill_path`, `pop_drill`, `rebuild_frame_from_stack` | **EXISTS — works** | Yes | none |
| Persistence / STM | Redis `veda:mem:{tenant}:{session}:…`, 7-day sliding TTL, optimistic `expected_version` | **EXISTS — works** | Yes | none |
| Ambiguity as an outcome | `ambiguous` in the closed set; `parse_delta_response` fails closed to it (`classify.py:42-58`); `ask_clarification_node` (`nodes.py:2223`) | **EXISTS — works** | Yes | none |
| RBAC re-check on remembered context | `_frame_still_authorised` (`nodes.py:1516`), `_authorised_episodic` (`store.py:187`) | **EXISTS — works** | Yes | none |
| **Typed transport to the engine** | `HybridRequest` (`hybrid.py:236`) has `flags: dict | None` — **accepted and dropped** | **PARTIAL** | Yes | forward it; define its schema |
| **Engine accepts structured context** | `run_query(..., anchor_hint=...)` (`pipeline.py:189`) already overrides the anchor without NL — but is internal-only, unreachable from HTTP | **PARTIAL** | Yes | widen from one hint to a small typed context |
| **Gate distinguishes user tokens from injected ones** | `qualifier_completeness(query, sql, sm)` (`validation.py:404`) — takes only the string | **MISSING** | — | accept the user's ORIGINAL words as the gate's basis |
| Previous-RESULT reference ("the second one") | frame stores `last_sql`, `last_row_count` — **not the rows** | **MISSING** | — | out of scope of this fix; see §D |
| New conversation DB / memory system | — | **NOT NEEDED** | — | Redis + QueryFrame already cover it |

---

## D. Follow-up capability matrix

Measured on 2026-09-23 over the 27 questions that answer at all (`evaluation/testpy_run/`).

| Capability | Current behaviour | Works? | Why |
|---|---|---|---|
| Basic follow-up | frame anchors, entity retained 13/16 | **Mostly** | `QueryFrame.entity` + `render_frame_as_query` |
| Add filter (`refine`) | 16/16 of the follow-ups that produced SQL added a real `WHERE` | **Yes** | engine grounds the bare VALUE; the frame supplies the entity |
| Replace filter | `replace` + `delta_field`, Python swaps in `apply_context_delta` | **Yes** | field must pre-exist; value must be in the message |
| Remove filter | `remove`; resolved query becomes the restated frame, NOT the user's words | **Yes** | `frame.py:1297-1303` — deliberately avoids re-sending the removed word |
| Reference resolution ("them") | `slot_candidates` vocabulary-gated to substrings of the message | **Yes** | `classify.py:64-69` |
| **Previous-RESULT reference** ("the second one") | frame keeps `last_sql` + `last_row_count`, never the rows; episodic keeps a ≤200-char gist | **No** | nothing stores row identity — see §F |
| Topic switch | `new_topic` returns the message unchanged, no prefix | **Yes** | `frame.py:1267` |
| Clarification round-trip | `ask_clarification_node` → `clarify_reply_node` → engine | **Yes** | engine's own `feedback.text` is reused, never invented |
| Ambiguous query | fails closed to `ambiguous`; never guesses | **Yes** | `classify.py:42-58` |
| Multi-step conversation | drill stack depth 10, reset 25/27 clean | **Mostly** | `_MAX_DRILL_DEPTH = 10` |
| **Shape preserved across a follow-up** | 0 of 8 aggregated bases survived — a `GROUP BY` distribution became a raw row list | **No** | the frame HOLDS `group_by`/`measures`, but `_describe_frame` deliberately drops them from the string (`frame.py:487-496`) because the engine mis-parsed the operation words |
| Cross-source follow-up | `frame.source_id` pins the source; `_frame_still_authorised` re-checks | **Partial** | ~5/27 follow-ups were answered from documents instead of the pinned relational source |

### The two failures share one cause

Both **shape loss** and the **`financial` refusal** come from the same place. The frame
knows `group_by=['furnishing']` and `entity='accounts_generalledger'`. Neither can be
sent as a *fact*; both must be squeezed into prose that the engine re-parses as a
question. So the code faces an impossible trade-off, and its own comments record having
tried both sides of it:

- Include the shape → `frame.py:487` *"the engine read the word 'measuring' as a data term
  and spent 54s before asking 'Could you clarify if measuring is a column name'"* → shape
  was dropped.
- Include the entity label → the `financial` refusal documented above.
- Drop the entity label → `frame.py:448` *"a bare 'go back' re-resolved to a DIFFERENT,
  similarly-named table (`reminders_reminderpaymenttransaction`)"*.

**There is no wording that satisfies all three.** That is the proof that this is a
boundary problem and not a phrasing problem, and it is why the quick fix in §2 of the
brief is correctly rejected.

---

## E. Final architecture

The chain in §12 of the brief is **directionally right but overbuilt for this repo**: it
implies constructing a Conversation Manager, a context resolver, a canonical state and a
validation step. All four already exist. The minimum correct architecture is the existing
one with the last inch changed.

```
USER MESSAGE ─────────────────────────────────────────────┐   EXISTING (verbatim, untouched)
                                                          │
apps/chat/views.py       RBAC / scope / data_scope        │   EXISTING
chatbot/graph.py         memory_read                      │   EXISTING
                         classify   → delta_type          │   EXISTING (closed set, 7B-safe)
                         context_resolve                  │   EXTEND  ── stop flattening here
                           · apply_context_delta (Python) │   EXISTING
                           · build TypedContext           │   NEW (small: a dataclass)
call_engine              POST {query, context, scope}     │   EXTEND  (`flags` already exists)
inference/routes/hybrid  forward context                  │   EXTEND  (currently dropped)
veda_core pipeline       consume context as HINTS         │   EXTEND  (`anchor_hint` precedent)
                         qualifier gate ← USER words only │   EXTEND  ── the actual bug fix
source coordinator → semantic layer → SQL → execution     │   EXISTING (unchanged)
result → memory_write (evidence only)                     │   EXISTING
```

**NEW is one dataclass.** Everything else is EXISTING or a narrow EXTEND.

### What the typed context carries

Only already-proven facts the frame holds — no prose, no invented values:

```
entity_table      accounts_generalledger     -> replaces the "(for …)" prefix entirely
source_id         2                          -> already sent separately today
filters           [{field, operator, value}] -> already structured in the frame
group_by          ["furnishing"]             -> fixes the shape-loss failure
measures/order_by/limit                      -> fixes "make it top 10" needing re-derivation
user_message      "only the debit ones"      -> the gate's basis, verbatim
```

### The one change that actually fixes `financial`

Even with a perfect typed channel, `qualifier_completeness` still gates whatever string
it is given. Its contract is *"every content token **the user named**"*. It must therefore
be given the user's own words as its basis, not the assembled query. That is a signature
change on one function plus its call sites — not a redesign.

---

## F. Enterprise safety analysis

| Risk | Current state | Mitigation |
|---|---|---|
| Incorrect memory injection | **This is the live bug** | typed context; user words never edited |
| Ambiguous references | `ambiguous` is a first-class outcome, fails closed | already correct — keep it |
| Stale state | 7-day sliding TTL; `comparison_is_stale` exists | keep; consider a turn-count bound |
| Wrong source inheritance | `frame.source_id` pins it; `_frame_still_authorised` re-checks each turn | already correct |
| Previous-result references | **not supported** — no rows stored | do NOT fake it from the gist; answer "I'd need to re-run that" |
| RBAC change between turns | `nodes.py:1516` + `store.py:187`, both fail-closed on an empty resolved scope | already correct — *"Authorisation from a previous turn is not authorisation"* |
| Tenant isolation | Redis key is `{tenant}:{session}`; `RequestContext` raises when unset (`context.py:64`) | already correct |
| Cross-source context | ~5/27 follow-ups leaked to documents | separate routing defect, not a memory one |
| Long conversations | history capped at 10 turns (`state.py:24`), drill depth 10 | already bounded |
| Model failure / timeout | `call_slm` returns `None`; parser degrades to `ambiguous` | already correct |
| Invalid structured output | `parse_delta_response` fails closed; slot values must be substrings of the message | already correct |

---

## G. 7B/8B feasibility

**Yes — and the repo already proves it, because the model is already kept out of state
management.**

Evidence:

1. The model's entire output is a closed-set JSON: `{delta_type, slot_candidates,
   delta_field}` (`classify.py:33-93`). It never emits state, SQL, a table, or a column.
2. Python does every mutation (`apply_context_delta`, `frame.py:1007-1014`): *"a 7B model
   asked to restate the whole context correctly every turn will eventually not, whereas
   this cannot drift."*
3. Two independent structural guards survive a wrong classification: the field must
   already exist in the frame, and the value must appear verbatim in the user's message.
4. Unparseable / timed-out output degrades to `ambiguous`, never to a guess.
5. Reset, drill-up, presentation, recall, runtime-context and bare-referential turns
   short-circuit **before any model call** (`nodes.py`).
6. `_SHAPE_PATTERNS` (`frame.py:615+`) fires zero times across all 299 real user messages
   in `evaluation/conversation/corpus_real.jsonl`.

**And there is direct evidence against giving the model more to do.**
`prompts/delta_types.py:12-26` records a measured attempt to compress the 671-token delta
block: 54/54 → 53/55 → 52/55, reverted. *"A 7B asked to do two classifications in one pass
has a budget, and this was over it."* A typed channel **reduces** model load; more
elaborate prompting is measurably the wrong direction.

---

## H. Minimum implementation tasks

Not implemented. Ordered; each is independently shippable.

| # | Task | Purpose | Reuses | New code | Depends on | Risk |
|---|---|---|---|---|---|---|
| 1 | Give `qualifier_completeness` the user's ORIGINAL words as its gating basis | **Fixes `financial` outright** | `validation.py:404` | one parameter + call sites | — | Low. Gate becomes strictly less likely to false-refuse; it can still refuse on the user's own dropped words |
| 2 | Define `ConversationContext` dataclass from existing frame fields | The typed carrier | `QueryFrame` | ~40 lines, no logic | — | Very low — pure data |
| 3 | Forward it over the existing `flags` field | `HybridRequest.flags` is already accepted and dropped | `inference_client.py`, `hybrid.py:236` | wiring only | 2 | Low. Absent/empty = today's behaviour |
| 4 | Engine consumes `entity_table` as an anchor hint | Removes the need for the `(for …)` prefix | `run_query(anchor_hint=…)`, `pipeline.py:1186` | plumb HTTP → `run_query` | 3 | Medium — anchor override already exists but was only ever used for salvage retries |
| 5 | Engine consumes `filters` / `group_by` / `limit` as plan hints | **Fixes shape loss (0/8)** | the existing deterministic branches | hint→plan mapping | 4 | Medium — must not override what the user explicitly asks for this turn |
| 6 | Stop emitting the `(for …)` prefix once 1+4 land | Removes contamination at the source | `frame.py:1304` | deletion | 1, 4 | Low **only after** 4 — until then it is the sole disambiguator |
| 7 | Contract test: resolved query is byte-identical to the user's message | Stops this class of bug returning | `tests/` | one test file | 6 | None |
| 8 | Extend the conversation eval to assert SHAPE survives a drill-down | 0/8 today and nothing catches it | `evaluation/conversation/run_eval.py` | scenarios | 5 | None |
| 9 | Decide previous-result references explicitly | Today it silently cannot | `QueryFrame` | either store row identity, or refuse honestly | — | Medium — storing rows has RBAC/TTL implications |
| 10 | Route follow-ups to the frame's pinned source | ~5/27 leaked to documents | `frame.source_id` | routing guard | — | Medium — interacts with multi-source routing |

**Tasks 1 and 7 alone fix the reported bug** and are low-risk. 2–6 remove the underlying
boundary problem and fix shape loss with it. 9–10 are separate defects found while
auditing.

---

## §14 — what should NOT be built (repo evidence)

| Item | Verdict | Evidence |
|---|---|---|
| New conversation database | **Not needed** | Redis `veda:mem:*` with TTL + optimistic versioning already exists |
| New memory system | **Not needed** | `QueryFrame` + `MemoryStore` cover it |
| Second source router | **Not needed** | `source_coordinator.plan_route` exists |
| Second semantic layer | **Not needed** | one merged semantic model already serves every path |
| Second SQL engine | **Not needed** | `veda/pipeline.py` is the single generator |
| Memory-generated NL enrichment | **Must be removed** | it is the bug — `frame.py:1304` |
| Blind full-history injection | **Not needed** | history already capped at 10 turns (`state.py:24`) |
| Mandatory query rewriting | **Must NOT be reintroduced** | `nodes.py:1717-1736` already removed it for first turns: *"a paraphrase … is a new chance to lose a word the engine parses as data"* |
| LLM-controlled state persistence | **Must NOT be built** | `apply_context_delta` deliberately keeps the model out |
| LLM-controlled RBAC | **Must NOT be built** | RBAC is resolved per-turn in `views.py` and re-checked against memory |
| Vector memory for session STM | **Not needed** | STM is a bounded dict, not a retrieval problem |
