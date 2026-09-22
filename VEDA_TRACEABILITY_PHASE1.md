# VEDA Query Traceability — Phase 1 (Foundation)

**Date:** 2026-09-08 · **Branch:** `feat/multisource-arch` · **Status:** implemented + **LIVE-VERIFIED**, all flags default OFF, nothing committed.

Implements **Phase 1 — Foundation** of the 28-part traceability / explainability / RBAC-aware
lifecycle spec. Builds directly on the inventory in `VEDA_TRACEABILITY_AND_RBAC_AUDIT.md`.

> **The spec message was truncated** mid-Part-28 (`5. Define the internal execution…`). Parts 1–27
> and the "Phase 1 — Foundation" heading were complete, so Phase 1 is what was built. Everything
> else is listed in [§7 Not in this phase](#7-not-in-this-phase) — backlogged explicitly, not
> silently dropped.

---

## 1. Architecture — one fact, three consumers

The spec's Part 27 rule, made structural:

```
                       veda/lifecycle.py :: Timeline.emit()
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
 ExplainTrace section        on_event(phase, msg, extra)   safe_projection
 (internal, persisted)       → existing `thinking` SSE     → explainability payload
```

No layer re-derives "what happened". `source completed` is recorded **once** and projected three
ways. The internal trace stays internal; `veda/safe_projection.py` is the **only** module permitted
to turn it into user-facing output — which is what makes the safety rule auditable: *if a field is
not constructed there, it cannot reach the frontend.*

---

## 2. New modules (`veda_core/veda/`)

| Module | Responsibility | Key safety property |
|---|---|---|
| `lifecycle.py` | 10 business phases (`received → understanding → access_check → source_selection → execution_plan → data_retrieval → cross_source_processing → validation → result_preparation → completed`), 4 statuses, ambient `ContextVar` (same pattern as `explain.py`) | **Closed phase vocabulary** — an unmapped internal stage name emits *nothing* rather than leaking. `details` takes named kwargs, never a raw object dump. |
| `warnings.py` | The missing tier between pass/fail and refuse. 7 stable codes, idempotent per code, streams live *and* lands in the payload | A warning says **that** something was limited, never **which** restricted thing. |
| `exec_records.py` | `SourceExecutionRecord`: status, timestamps, duration, rows, retries, fallback | `as_safe_dict()` is the only user-facing projection — drops `engine`, `error`, `fallback_path`, `error_class`. |
| `source_names.py` | Display names, resolved from the request-scoped profile map | Unknown source → generic `"a data source"`, **never** the numeric id. Only the api tier's already-authorised sources are in the map. |
| `safe_projection.py` | The single trace→user boundary | Has **no reader** for `retrieval` / `rrf` / `reranking` / `graph_expansion` / `slm` / `llm_usage` / `sql_planning` / `join_planning` / `rbac_filter`. |

## 3. Wiring (all flag-gated)

| Where | What now happens |
|---|---|
| `veda_hybrid.run_hybrid_query` | Owns the timeline + recorder for the request; binds both ambiently; emits `received`, `understanding` (**started**, so the phase order matches reality — see §6a), `access_check` (on success **and** on permission refusal), `source_selection` (from the routing decision), and the terminal phase |
| `query/source_coordinator.execute_decision` | Persists the `ExecutionPlan`, opens/closes a per-source record per step, records `MergeResult` policy + conflict, records federation facts, raises partial-failure + conflict warnings |
| `veda/pipeline.run_query` | Completes `understanding`, emits `validation`, and opens/closes a per-source record around `execute_sql` (which also yields `data_retrieval`) — this is the **deterministic** path, which never reaches the coordinator; passes the trace to `build_explain` so v2 can project |
| `veda/execution.execute_sql` | Real DB-side timing (`db_execution_ms`) + raises the truncation warning at the one place that can observe it |
| `apps/chat/turn_events.py` | Accumulates structured lifecycle events; surfaces `trace_id` + `timeline` in `ChatMessage.metadata` |

**Four things that were computed and then discarded** (per the audit) are now recorded:
`ExecutionPlan`, `MergeResult` (incl. detected conflicts), the `partial{}` failure block, and
federation provenance counts.

## 4. Two boundary bugs found and fixed

1. **`X-Veda-Source-Profiles` was sent but never read.** The api tier has forwarded it since the
   multi-source work (`apps/query/inference_client.py:92`), but `inference/main.py` never bound it —
   so `current_source_profiles()` returned `{}` in **every deployed request**. Consequences: the
   coordinator's canonical tie-break could never fire, and nothing could resolve a display name.
   Now bound, **failing open to `{}`** — this is display/tie-break metadata, *not* an authorization
   input (scope was already narrowed by `X-Veda-Source-Ids`), so a malformed header must degrade to
   generic labels, never deny a legitimate query.
2. **`source_profiles_for()` carried no `name`.** Added — the single controlled boundary at which a
   display name crosses into the engine, so no DB join is spread through the pipeline.

## 5. The two audit defects

**Defect 1 — FIXED.** `explain.py:566` and `pipeline.py:274` read flat `"datasets"` / `"check_items"`
keys that `build_explain` has *never* emitted (it returns `data_used.datasets` and
`validation.checks`), so **every** trace recorded `datasets=None, validation_passed=None`. One shared
`summarize_explain_payload()` now reads the real nested shape, also accepts the refusal payload, and
is verified against real payloads.

**Defect 2 — deliberately NOT faked.** `MODE_PARALLEL` still executes sequentially
(`source_coordinator.py:848`). Rather than report a mode that does not happen, the trace stamps
`executed_mode="sequential"` and the projection reports **that**. Real parallel execution needs
cancellation, timeouts, RBAC-before-execute and a deterministic merge — backlogged, see §7.

## 6. Verification

- **39 new tests** in `tests/test_query_traceability.py` — all pass. Includes leak assertions (a raw
  `psycopg2` error with host + password, engine names, `"PARALLEL"`, candidate scores are each
  asserted **absent** from projections), an explicit **default-OFF contract** test, and an
  end-to-end spine test asserting the stream and the persisted payload describe the same run.
- **17 existing suites run per-file** (the suite is order-flaky in aggregate): 16 clean.
- **2 failures, both proven pre-existing by diff:**
  - `test_confident_single_skip::test_flag_on_one_strong_skips_slm` — calls `SC._decision_boundary`,
    which this diff does not touch; matches PM_LOG's own documented 3/4 from the 2026-09-06 RSE work.
  - `test_data_scope::test_staff_bypasses_regardless_of_grants` — `data_scope.py` is untouched here,
    and its own docstring (lines 38, 98) states `is_staff` is **not** a data-scope bypass while the
    test asserts it is. A pre-existing test/code contradiction.

### Flags (all default OFF except the one noted)

All env-overridable, following the repo's existing 73-flag convention
(`_os.environ.get("FLAG", "0") == "1"`), so a rollout or a verification run needs no code edit:

```
LIFECYCLE_EVENTS_ENABLED          default 0
QUERY_WARNINGS_ENABLED            default 0
SOURCE_EXECUTION_RECORDS_ENABLED  default 0
EXECUTION_PLAN_TRACE_ENABLED      default 0
EXPLAIN_V2_ENABLED                default 0
DB_EXECUTION_TIMING_ENABLED       default 0
FEDERATED_JOIN_STATS_ENABLED      default 0
EXPLAIN_EXPOSE_SQL                default 1   # preserves the pre-existing unconditional exposure
```

Env is read at container **create** time, so flipping one needs `up -d`, not `restart` (CLAUDE.md).

`EXPLAIN_EXPOSE_SQL` is the exception on purpose: `business_explain.py:377` hardcoded
`"enabled": True`, so SQL already reached every user. Defaulting it True keeps today's behaviour
byte-identical; setting it False moves SQL behind a technical/admin view.

## 6a. Live verification (2026-09-08)

Run inside the real `inference` container — real Postgres, real retrieval, real pipeline — with the
flags supplied as **env only** (`docker compose exec -e …`), so no default and no `.env` was changed.
Query: *"How many sale negotiations are there?"* → answered `5`, `route=deterministic`.

**Flags ON — streamed `thinking` events (legacy events shown with `-`):**

```
[completed] received          Got your question
[started  ] understanding     Reading your question
[completed] access_check      Verified your access
[-        ] route             Deciding which source can answer…
[completed] source_selection  Found relevant data
[-        ] route             Routed to sql engine
[completed] understanding     Understood what you're asking for
[completed] validation        4 safety checks passed
[started  ] data_retrieval    Retrieving data from Homzhub Property DB
[completed] data_retrieval    Homzhub Property DB completed
[-        ] output            Done — here's your answer
[-        ] answer            SQL query executed
[completed] result_preparation Answer ready
[completed] completed         Done
```

`explain` came back `version: 2.0` with a real per-source record
(`duration_ms: 479, rows_returned: 1`), the resolved display name **"Homzhub Property DB"** (proving
the newly-bound `X-Veda-Source-Profiles` header works end to end), and `support.trace_id`.

**Flags OFF — byte-identical to before:** only the 4 pre-existing legacy events, `version: 1.0`,
exactly the original 9 keys, same answer.

### Four wiring gaps the live run exposed (all fixed)

Unit tests could not have caught these — they are wiring-coverage facts, not logic errors:

1. **The deterministic path produced no `execution` block at all.** It bypasses
   `source_coordinator.execute_decision`, so no `ExecutionPlan` and no per-source record existed —
   on the *most common path of all*. Now a record is opened around `execute_sql` in `pipeline.py`.
2. **`data_retrieval` never fired** on that same path (same cause). Now it does, naming the source.
3. **`access_check` only fired on refusal**, so a successful query never showed "✓ Verified your
   access". Now emitted on success too, from the already-RBAC-narrowed scope.
4. **Phase order was wrong** — `source_selection` completed *before* `understanding` appeared,
   because routing runs at the front door and Tier-1's understanding stage runs later. Fixed by
   opening `understanding` as `started` at the front door (honest: reading the question genuinely is
   the first thing that happens).

### One copy correction

`validation` completes **before** `data_retrieval` — which is correct pipeline behaviour
(validate-then-execute; all four checks are pre-flight on the SQL). The phase title was therefore
changed from *"Checking the result"* to **"Checking the query"**: the original wording described a
result that does not exist yet. The spec's illustrative timeline put "Checking the result" after
"Running the query"; this pipeline's real order is the opposite, and the timeline reports the
real order.

## 6b. Parts 3 / 23 / 10 (2026-09-08, second increment — LIVE-VERIFIED)

### Part 3 — RBAC narrowing is no longer silent
`RESTRICTED_DATA` is raised at **two** places, both reusing an existing decision rather than
re-deriving one:
* `pipeline.py` where `filter_retrieval_results` actually **changed the candidate count** — merely
  running the filter is not newsworthy, removing a relevant candidate is.
* `pipeline._feedback` when `explain_failure` returns `ACCESS_DENIED_WHY` — i.e. the refusal genuinely
  *was* an access denial.

The `before`/`after` counts stay internal and are **not** projected: the number of withheld items is
itself a disclosure. Verified live against a real restricted data-scope — the payload contains no
`rbac_filter`, no `allowed_resources`, and no restricted table name.

### Part 23 — retries and fallback
* `AgentResult.retry_count` is now a real field, stamped by `reliability.execute_reliably` (it
  previously only wrote a prose `reason` string like `"retried x2"`).
* The coordinator copies it onto the per-source record and raises `FALLBACK_USED` once.
* `execute_federated_reliably`'s `retry_attempts` gets the same treatment.
* The **Tier-1 → Tier-2** hand-off raises `FALLBACK_USED` too — a genuine alternate retrieval path.
* The safe projection exposes `retried: true` / `fallback_used: true`, never the attempt count, and
  the warning copy is asserted to name no internal component (no "tier", "slm", "agent", "rag").

### Part 10 — cross-source join match counts
Implemented **only** on the aggregate-pushdown path (`execute_plan`), where the per-source aggregates
are already materialized as DuckDB temp tables — so `_join_stats` is two *local* counts (INNER vs
FULL join cardinality) that touch no source and issue no federated round trip.

Deliberately **not** offered for plain `execute()`: there the SQL is an arbitrary caller-supplied
join, and the same numbers would cost one or two extra federated queries on a path whose measured
median is already tens of seconds. A missing join block is honest; a guessed one is not.

Verified on real DuckDB 1.5.4 in the container: 2 sources → `matched 2 / unmatched 2`; 3 sources →
`matched 1 / unmatched 3`; single table → `None`; flag off → `None`; and the full chain renders
*"…98.0% of relevant records were successfully matched."* with the raw counts stripped.
Flag: `FEDERATED_JOIN_STATS_ENABLED` (default 0).

### Two more honesty bugs the live run caught (both were mine)

1. **"Verified your access" was a premature claim.** It was emitted at scope-resolution time, so a
   query that was *later* refused on permission grounds showed the user "✓ Verified your access"
   followed by "you do not have permission" — a visible contradiction in the live stream. The front
   door now emits `access_check` as **started** ("Checking your access"), and the phase completes in
   `pipeline.py` only once validation has actually passed — which is the first moment we can honestly
   say narrowing did not block *this* query. A real access denial downgrades it to `failed`.
2. **A refused turn carried no warnings and no support reference** — the v2 extension only reached
   the success path, yet a refusal is exactly when a user most needs the caveat and a trace id.
   `build_refusal_explain` now merges `warnings`, `limitations`, `timeline_summary` and `support`.
   It deliberately does **not** merge `routing`/`execution`/`result`, which would describe work that
   produced no answer.

### Pre-existing issue noted, not changed

`veda/rbac_filter.py`'s `narrow_allowed` prints the **entire restricted-table list** (~170 names on
this dataset) to stdout on every narrowed query, so `docker logs inference` accumulates the full
restricted schema. Not introduced here and not touched, but it is a log-side disclosure worth a
separate decision.

## 6c. Full chat-path verification + a pre-existing contextvar bug (2026-09-08)

Flags enabled in `.env` and `inference` recreated (env is read at container create time), then a
**real chat turn** driven through `ConversationQueryService.run_turn(stream=True)` — the actual path
the frontend uses: chatbot LangGraph → inference HTTP/SSE → SSE bridge → `TurnEventAccumulator` →
`ChatMessage.metadata`. User `veda`, scope `[2,3,4,5]`, RBAC mode `enforce`.

**Result — the frontend now receives:**

```
thinking [-        ] supervisor_classify  Understanding your question...
thinking [completed] received             Got your question
thinking [started  ] understanding        Reading your question
thinking [started  ] access_check         Checking your access
thinking [completed] source_selection     Found relevant data
thinking [completed] understanding        Understood what you're asking for
thinking [completed] validation           4 safety checks passed
thinking [completed] access_check         Verified your access
thinking [started  ] data_retrieval       Querying structured data
thinking [completed] data_retrieval       A data source completed
thinking [completed] result_preparation   Answer ready
thinking [completed] completed            Done
content       'The assets salenegotiation count is 5.'
explainability version=2.0  (17 keys)
usage         {...total_tokens: 563, latency_ms: 4392}
```

`ChatMessage.metadata` keys are now `['explainability','thinking','timeline','trace_id','usage']` —
`trace_id` and `timeline` are both new (a `grep -rn "trace_id" apps/` returned **zero** before this
work). `execution.sources[0]` came back as
`{"name": "homzhub", "type": "Database", "duration_ms": 486, "rows_returned": 1}` — the real name
from the Django source registry, resolved across the HTTP boundary.

### A pre-existing bug this uncovered (measured by A/B, not assumed)

The streaming route carried its worker thread's context with
`with_context(try_current(), _run)`, which re-binds **only** the `RequestContext`.
`_source_profiles` is a **separate** ContextVar, so it was silently dropped in that thread.

A/B on the real chat path, same query, only that line changed:

| Thread context carried by | `execution.sources[0]` |
|---|---|
| `with_context(...)` (original) | `{"name": "a data source", "known": false}` |
| `copy_context()` (fixed) | `{"name": "homzhub", "type": "Database", "known": true}` |

Fixed by snapshotting the **whole** context with `copy_context()` — which is what
`inference.concurrency.run_in_threadpool_with_context` already does for the non-streaming route, so
the two paths now behave identically and a future third request-scoped ContextVar cannot
reintroduce this.

**This is wider than display names.** Three modules read `current_source_profiles()`:

| Reader | Consequence on the streaming path |
|---|---|
| `veda/source_names.py` | generic label instead of the real name — **measured above** |
| `veda_hybrid.py:605` → `_is_datalake_source` (line 693) | the datalake-isolated semantic model was never loaded, so a datalake question got the primary source's schema — **inferred from the same measured mechanism, not separately reproduced** |
| `query/datalake_values.py:67` | profile-based datalake value grounding degraded — likewise inferred |

Note `apps/chat/views.py:155-163`'s own comment describes closing exactly this gap by *sending*
profiles from chat. It closed the "not sent" half; the "lost in the worker thread" half survived
until now, so on the SSE path the fix had no effect.

### One correction to my own earlier claim

My first attempt at this attributed the generic label to the ContextVar gap **before** verifying it.
The immediate cause of what I first observed was in fact my own test script, which omitted the
`source_profiles=` argument that `apps/chat/views.py` really passes. Both turned out to be real and
independent: the script was wrong **and** the ContextVar gap was real — but only the A/B above
establishes the second, and it is the A/B, not the initial guess, that the table records.

## 6d. Real HTTP SSE endpoint verified + a re-read gap fixed (2026-09-08)

§6c drove `ConversationQueryService` **in process**. This is the real thing: a JWT-authenticated
`POST /api/v1/conversations/query` with `"stream": true`, over nginx on :8080, `VEDA_JWT_AUTH=1`.

**Stream:** 16 `thinking` + 2 `content` + 1 `explainability` + 1 `usage` + 1 `completed`.
Structured events carry `status`/`title`/`timestamp_ms`/`elapsed_ms`/`details_available`/`details`;
the pre-existing free-text ones (`supervisor_classify`, `route`, `output`, `answer`) still arrive
with just `phase` + `message`, so **an existing client is unaffected**:

```
event: thinking
data: {"phase": "supervisor_classify", "message": "Understanding your question..."}   <- legacy

event: thinking
data: {"status": "started", "title": "Running the query", "elapsed_ms": 3602.1,
       "details": {"source_id": "2"}, "phase": "data_retrieval",
       "message": "Retrieving data from homzhub"}                                     <- structured
```

`explainability` arrived at `version 2.0` carrying `routing`, `execution` (with
`{"name": "homzhub", "type": "Database", "duration_ms": 474, "rows_returned": 1}`), `timeline`,
`timeline_summary`, `warnings`, `limitations`, `result`, `sources`, `support.trace_id`.

**One header note:** `Accept: text/event-stream` is REJECTED by DRF content negotiation
(`{"detail":"Could not satisfy the request Accept header."}`) — the endpoint streams on the
`"stream": true` body flag, not on Accept. A client must send `*/*` or `application/json`.
Pre-existing behaviour, not introduced here, but it will bite a frontend that follows SSE convention.

### The re-read gap (found here, fixed)

The **live stream** and the **persisted row** both had the timeline, but the history endpoint did
not. `_serialize_history_message` (`apps/chat/views.py`) projects metadata through an explicit
3-key allowlist — `thinking` / `explainability` / `usage` — written before these keys existed:

| Read path | Before | After |
|---|---|---|
| `ChatMessage.metadata` (DB) | `trace_id` ✓, `timeline` 11 events ✓ | unchanged |
| `GET /conversations/history` | **`trace_id: None`, `timeline: 0`** | `trace_id` ✓, `timeline` 11 ✓ |

So a user who **reloaded a conversation** lost the execution timeline and the support reference the
live stream had just shown them. The allowlist is deliberately KEPT (never `dict(meta)`, so an
internal key cannot leak by default); the two keys are added explicitly and **omitted when absent**,
per this codebase's "absent, not null" envelope convention. Verified: a pre-existing turn from
chat 28 still re-reads as exactly `['explainability','thinking','usage']`.

## 6e. Governance — Parts 19 / 20 / 21 (2026-09-08, LIVE-VERIFIED)

### Part 20 — real user identity on the audit trail
`QueryLog.user` FK added. **Nullable + `SET_NULL`, deliberately:** nullable because the
user of a historical query cannot be reconstructed and inventing one is worse than recording
"unknown"; `SET_NULL` (never `CASCADE`) because deleting a user must not delete the audit
history of what they ran. `tenant` is left alone — it was populated from `user.username`,
which made it an accidental identity proxy; it now goes back to meaning only tenancy.

### Part 19 — the chat front door is audited, from ONE writer
`QueryLog` was written from exactly one place, so `/api/v1/query` was audited and the chat
front door — the one the product ships — wrote **nothing**. Chat turns existed only as
`ChatMessage` content, which carries no status, route, executed SQL, latency or source.

The fix is **not** a second copy of the write. `apps/query/audit.py::record_query` is the one
writer; both views call it. `QueryLog` also gained the three fields Part 19's questions need:
`participating_sources`, `partial`, `warning_codes` (codes only — messages are display copy).

Everything is read from what the turn ALREADY produced
(`audit_fields_from_explain`), so the audit row and what the user was shown cannot disagree.

**Correlation:** no new column. `request_id` already IS the engine `trace_id`
(`apps.core.middleware` mints X-Request-Id → the api forwards it → `veda/explain.py` adopts it
verbatim), so one id joins the audit row, the engine trace, and the `support.trace_id` the user
saw. A second identifier for the same thing would drift.

**Live proof** — same question through both doors, 118 → 120 rows:

| `request_id` | user | status | route | participating | latency | sql |
|---|---|---|---|---|---|---|
| `gov-chat-001` (chat SSE) | `veda` | answered | deterministic | `[2]` | 6308ms | `SELECT COUNT(DISTINCT "id")…` |
| `gov-query-002` (`/api/v1/query`) | `veda` | answered | deterministic | `[2]` | — | `SELECT COUNT(DISTINCT "id")…` |

#### A drift bug the live run caught

The first run recorded `participating_sources = [2]` on the chat path but `[2,3,4,5]` on the
query path — **the same column, the same query, two different meanings** (what executed vs. the
whole permitted scope). Exactly the drift the shared writer exists to prevent, reintroduced by
getting the merge precedence backwards in one caller. The column means "which sources actually
executed", so the execution records now win on both paths and the scope is only a fallback for a
payload that has no execution block (a v1 payload, or a refusal). Re-verified: both `[2]`.

### Part 21 — authorization decisions are queryable
New `access_management.AuthorizationDecision`, appended by the gate on every denial (and every
shadow-mode would-have-denied, tagged `mode="shadow"` so a rollout is not read as an outage).
Allows are never recorded — a row per permitted request would be enormous and the denominator
comes from `QueryLog` via `request_id`.

**The resource path is deliberately NOT stored.** `db:crm_postgres:employee:salary` names the
schema, table and column of something the caller was just told they may not see; writing it into
a durable table on every denial would build exactly the catalogue the denial exists to protect.
Only the coarse `resource_kind` (`db`/`nosql`/`files`/`lake`) is kept — enough to answer every
question Part 21 names. A test asserts the path never appears in any persisted field.

The `reason_code` is the actionable half, taken from the resolver's own distinction rather than
re-derived: `explicit_deny` (intentional policy — the request is wrong) vs `no_grant` (this role
needs the permission) vs `undeclared` (a view opted into the gate and declared nothing — a code
bug) vs `resolution_error`. `EffectivePermissions.denies()` is deliberately separate from
`not allows()` precisely so these can be told apart; a log line loses that entirely.

#### A gap the tests caught
In **enforce** mode the `undeclared` branch short-circuited on `mode != MODE_ENFORCE` and never
reached the audit — so the one denial caused by a *code bug* was the one invisible in the audit
trail. Fixed; outcome unchanged (enforce still denies, shadow still passes through).

### Migrations
`query.0005` (4 AddField + 1 AddIndex) and `access_management.0011` (CreateModel). Reviewed via
`sqlmigrate` before applying: every added column is nullable or defaulted, so the migration is
safe on existing data. Verified after applying — **all 118 pre-existing `QueryLog` rows preserved
and readable**, oldest row intact with the new defaults (`user=None, partial=False,
warning_codes=[], participating_sources=[]`).

### Flag
`VEDA_AUTHZ_AUDIT` (default **0**). Off by default per this project's rule that new behaviour
ships flag-gated — it adds a DB write to the denial path. An audit table nobody enabled answers
nothing, so turn it on deliberately once the migration has run.

### Noted, not changed
`tenant` differs between the front doors for the same user: chat records `"default"`
(`DEFAULT_TENANT`), `/api/v1/query` records `"veda"` (`_resolve_tenant` uses the username). That
predates this work, and changing tenant semantics could affect isolation/RBAC, so it is flagged
rather than "tidied".

## 7. Not in this phase

Backlogged from the spec, in rough priority order:

| Part | Item | Why deferred |
|---|---|---|
| 25.2 | **Real parallel source execution** | Needs cancellation, timeouts, RBAC-before-execute, deterministic merge. Faking the label was the worse option. |
| 10 | Join **keys** (the predicate itself) for an admin/debug view | Match *counts* are now done for the pushdown path (§6b); the raw predicate stays internal by design, and the plain `execute()` path is excluded on cost grounds. |
| 20 | ~~User FK on `QueryLog`~~ | **Done** — §6e. Nullable + SET_NULL; no backfill (an old row's user is unrecoverable, and inventing one is worse than "unknown"). |
| 21 | ~~Authorization audit model~~ | **Done** — §6e. `AuthorizationDecision`, denials only, resource KIND never the path. |
| 19 | ~~`QueryLog` on the chat path~~ | **Done** — §6e. One shared writer, both front doors, correlated by `request_id`. |
| 23 | ~~Retry / fallback counters~~ | **Done** — see §6b. |
| 3 | ~~RBAC narrowing → `restricted_data` warning~~ | **Done** — see §6b. The product call was resolved by only warning when narrowing actually removed a relevant candidate or caused the refusal. |
| 17 | **Frontend Level-1/Level-2 UI** | `timeline_summary` (Level 1) and the full payload (Level 2) are both produced, streamed AND re-readable from history (§6d); rendering is frontend work. |
| 22 | `prompt_version` | No prompt-versioning scheme exists in the codebase to stamp. |
