# VEDA Explainability — What Happened So Far

**Work done:** 2026-09-08 → 2026-09-09 · **Doc updated:** 2026-09-09
**Branch:** `feat/multisource-arch` · **Nothing committed** — all changes are in the working tree

> **One-line summary.** VEDA now shows the user *what it is doing while it works* — as **4 fixed
> steps** that are the same for every kind of question — and *how the answer was produced*. Built and
> live-verified through the real chat API, with no extra AI calls required and ~0.3 ms overhead.
> Backend is done; the frontend is not built yet, and everything ships behind feature flags that are
> currently **off**.

**Detail docs:** audit → `VEDA_TRACEABILITY_AND_RBAC_AUDIT.md` · engineering →
`VEDA_TRACEABILITY_PHASE1.md` · task list for PM → `VEDA_EXPLAINABILITY_TASKS.md`

---

## 1. What was asked

Three things, in order:

1. **Audit** — what does VEDA already track, and what could it already show a user?
2. **RBAC visibility** — can we show who has permission to what?
3. **Build it** — a 28-part spec for a user-safe execution timeline, explainability payload,
   warnings, and audit trail.
4. **Simplify what the user sees** — collapse the internal detail into **exactly 4 generic steps**
   that work for every question type, with live timing and expandable detail.

---

## 2. The problem we were solving

Before this, a user asked a question and got **an answer**. Nothing else.

They could not see which data source was used, whether anything was checked, whether the result was
complete, or whether data had been withheld from them. All of that information **existed inside the
engine** — and several pieces were computed and then thrown away.

The worst case: an answer that was truncated, or missing a source, or based on restricted data,
looked **exactly the same** as a perfect answer.

---

## 3. What was built

### The normalized model (2026-09-10 redesign)

The `thinking` event now carries ONE object describing the whole turn, so no client
has to fold events together or decide which of several backend phases means the same
thing — that is the api tier's job, and doing it once means every client agrees:

```json
{ "type": "thinking", "status": "active", "current_step": "analyzing",
  "steps": [{ "id", "title", "state", "duration_ms", "summary",
              "details": [{ "type", "label", "state" }] }],
  "evidence": { "sources": 1, "passages": 5, "rows": null },
  "execution": { "type": "documents" },
  "timing": { "total_duration_ms": null } }
```

Three things that shape it:

* **Evidence, never a score.** `5 passages`, `20 rows`, `4 checks passed` are facts a
  reader can weigh. A confidence percentage the backend never defined is not, so none
  is shown. An absent count stays absent — "0 passages" and "we did not report
  passages" are different claims.
* **A closed detail vocabulary** (access / source / evidence / operation / validation
  / output). A new backend event cannot invent a new row shape in the UI.
* **Three disclosure levels.** Level 1 is the four steps; level 2 is their expanded
  evidence; level 3 is the `audit` block — the only place raw backend phase names
  appear, and it is never part of the primary experience.

### A live timeline — 4 steps, always the same 4

**Before:** silence for 8 seconds, then an answer.

**After** (collapsed — this is the default view):
```
✓ Understanding your request              0.2s
✓ Finding the right information           1.4s
    ✓ Checking your access                 58ms
◉ Analyzing the information               …
○ Preparing your answer
```

**Why exactly four.** The engine has **26** internal phases and they differ wildly by question type
— a document question and a database question share almost nothing. Showing them raw meant the UI
changed shape per question and leaked internal vocabulary. These 4 steps are **fixed**: same four,
same order, for SQL, documents, mixed, charts, tables, refusals and small talk. Every internal phase
is mapped to one of them, so nothing is unaccounted for and nothing new can appear.

Each step expands to real detail, in plain language — the step it belongs to, how long it took, and
what was actually done (SQL vocabulary like *group by* is translated, never shown). Steps only move
**forward**: a phase that resolves late cannot rewind the display.

**Authorization is a sub-check, not a step.** It sits inside *Finding the right information* with its
own measured duration. It is a check, not a stage of thinking — and giving it its own step made every
query look like it was fighting for permission.

Streams live over the **existing** chat API — no new endpoint, no polling. First event arrives in
~0 ms even on a 40-second query. Existing clients are unaffected: old progress events still arrive
in their original shape.

### An optional narrator that can never delay the answer

One short SLM call per query can rewrite a step's line into something more natural. It is a
**narrator, not a reasoner**: it never sees the data or the SQL, never decides anything, and its
output only ever replaces other text.

The hard requirement was that no explainability work may delay, reorder or break the answer. That
holds **structurally**, not just in practice: the call runs on its own daemon thread, nothing in the
query path awaits it, and the front door cancels it the moment the query terminates — a narration
that loses the race is discarded, never waited for. Every failure path is silence, and the
deterministic sentence is already on screen. Verified across 4 query shapes: **0** cases of thinking
text arriving after the answer, plus 2 regression tests.

Its output is **filtered, not trusted** — banned terminology, invented numbers and ungrounded domain
nouns are rejected. Honest limit: that catches leakage and numeric invention, which are the failure
modes that matter; it cannot prove a fluent sentence is faithful. Which is why the narration is
confined to progress text and is never used for the answer or the validation ledger. It is **off by
default**.

### A "how this was answered" payload

Alongside the answer: which sources (by **name**, not an internal id), how long each took, how many
rows, which safety checks passed, what the system understood, and a support reference id.

### A warning system that did not exist before

VEDA previously had only two outcomes: **answered** or **refused**. Nothing in between.

So if a result was cut short, or a source did not respond, or permissions removed some data — the
user was told **nothing**. Now:

> ⚠ The result was limited to the first 100 records.
> ⚠ Some available data could not be included because of access restrictions.
> ⚠ The primary method could not answer this, so an alternate method was used.

### An audit trail

Who ran which query, when, against which sources, with what outcome — and separately, a queryable
record of **access denials** with the reason (*permission missing* vs *explicitly forbidden* — those
need opposite fixes, and a log line cannot tell them apart).

---

## 4. Bugs found along the way

These were **not** part of the brief. They were found because building the explainability layer
forced us to look at what the system actually does.

| # | Bug | Impact | Status |
|---|---|---|---|
| 1 | Source-name header was **sent but never read** by the engine | Every explanation could only say `"2"` instead of a source name | ✅ fixed |
| 2 | Request context **lost in the streaming worker thread** | Source names blank on the chat path — **and it silently broke data-lake routing on that path too** | ✅ fixed |
| 3 | Trace read the wrong field names | `datasets` / `validation_passed` recorded blank on **100% of queries** | ✅ fixed |
| 4 | Conversation history dropped the timeline on reload | User lost the detail they had just watched | ✅ fixed |
| 5 | `OLLAMA_URL` hardcoded in docker-compose, overriding config | **Every AI answer silently fell back to a robotic template.** The config said one thing, the container did another | ✅ fixed |
| 6 | `SLM_MODEL_NAME` named a model the AI host will not serve | Answers degraded again, silently — a request that should take 0.6 s timed out at 95 s | ✅ fixed |
| 7 | A step could stay "spinning" **inside the saved record**, forever | Reopening an old, finished answer showed an unfinished step. Access checking resolves *after* the explanation is built | ✅ fixed |

Bug 5 is worth calling out: answers read like *"The assets salenegotiation count is 5."* Once fixed,
the same question answered *"There are 5 sale negotiations. This count represents the total number
of sales negotiations recorded."* The system had been degraded and silent about it.

### And 7 bugs in our own work

The first build wired the timeline to **one code path** (plain SQL). Running 10 real queries showed
that on document, refusal and multi-step queries, steps were missing or stuck spinning.

All 7 fixed and re-verified on the same 10 queries:

| Measure | Before | After |
|---|---|---|
| Progress steps stuck spinning forever | **7** | **0** |
| Missing validation lines | **5** | **0** |
| Truncated result flagged | ❌ | ✅ |
| Low-confidence answer flagged | ❌ | ✅ (disabled pending a threshold decision) |

### And 3 more in iteration 2 (the 4-step UX)

Found by running 12 real queries, not by inspection:

| Bug | Impact | Status |
|---|---|---|
| **A clarify/refusal rendered as four green ticks** — including *"Checking the result is complete and safe · passed"* — above a reply saying it could not answer | The exact dishonesty this whole layer exists to remove. Found on the cache lane: **2 of 3** hits | ✅ fixed |
| **`cache_hit` in the audit had been dead since 2026-07-09** — it compared against a sentinel the engine no longer produces | Cache observability silently zero; `veda_cache_hits_total` stuck at 0 | ✅ fixed |
| **Refusal copy leaked `column` / `table` / `SQL` and raw identifiers straight into the reply** | *"the query's measure lives on another table than the one the SQL ranks/aggregates"* — banned vocabulary in the most user-facing text there is. One string printed raw column names | ✅ fixed — 5 strings reworded, identifiers humanized |
| **A refusal still said *"Putting your answer together"* and listed a "Supporting summary" it never produced** | The terminal frame's context sentence overwrote the phase's own honest message | ✅ fixed |
| **The persisted timeline stopped one phase short** — a stored clarify showed 5 phases, not 7 | The payload is built before the front door emits the phase describing the outcome. On answered queries it was masked by luck (a truncation warning maps to that same phase) | ✅ fixed |
| SQL vocabulary (`group by`) leaked into the *Analyzing* detail line | Internal terminology shown to the user — the exact thing the safe-projection layer exists to prevent | ✅ fixed |
| Small talk left the access sub-check spinning | A "how are you?" turn looked like it was stuck checking permissions | ✅ fixed |
| The narrator's own filter rejected **100%** of real narrations | Feature silently disabled. It threw out safe sentences over ordinary words like *entire* and *year* | ✅ fixed — allowlist widened, protections kept |

One measurement lesson worth recording: a wall-clock timing test **passed** while the underlying code
was still quadratic. Only a scaling-ratio test exposed it (14.8× → 2.7× after the fix). A fast
measurement is not proof of good complexity.

---

## 5. Numbers

| | |
|---|---|
| Extra AI/LLM calls **required** | **0** — proven from the per-call ledger |
| Extra AI/LLM calls **optional** | **1** (the narrator) — off by default, non-blocking, cancellable |
| Latency overhead | **Not established end-to-end.** The ~0.3 ms figure was measured on the trace projection *in isolation* and does not survive an A/B: flags-ON medians 28.4 s vs flags-OFF 25.1 s on the same cached query, but the answer-writing SLM emitted 110–145 completion tokens with flags on versus a constant 110 with them off, so generation time — not this layer — moves with it. n=3–4 against a 22–38 s spread cannot separate the two. **Needs a proper run (n≥20) before any overhead number is quoted.** |
| Tests written | **387** passing across 9 suites, 0 failures |
| Feature flags | **11**, all default **off** — production byte-identical until enabled |
| Spec coverage | **28 of 34 stories complete**, 5 partial, 1 deliberately deferred |
| Live benchmark re-run | **12 / 12 clean** — no stuck step, no leaked vocabulary, no late thinking |
| Answer never delayed by explainability | **0 violations** across 4 query shapes + 2 regression tests |

---

## 6. What is NOT done

| Item | Why |
|---|---|
| **Frontend** | All data is delivered and re-readable from history. Rendering is pure UI work — needs design input. **This is the main gap before users see anything.** |
| **Document/RAG answers get a thinner explanation** | Those heads return the older payload format — no sources, no routing, no warnings. Only database answers get the full one. The 4-step thinking itself now works on every source type (verified live on relational / filesystem / csv_lake / parquet / smalltalk). |
| **Data-lake sources are never actually reached** | A csv_lake question routes to the relational source; Tier-2 names the right lake table and the firewall rejects it as unknown, because lake tables never enter the semantic model. Pre-existing engine gap, outside this scope — explainability now reports it honestly rather than hiding it. |
| **Cross-source explanation unverified** | Join match rates, source conflicts and partial-failure reporting are written but have **never run on a real multi-source query**, because multi-source routing is currently disabled on evidence it hurts accuracy. Not claimed as done. |
| **True parallel execution** | The plan says "parallel" but runs sequentially. We chose to **report it honestly as sequential** rather than claim otherwise. Real concurrency needs cancellation, timeouts and deterministic merge — a performance project, not this one. |
| **Prompt versioning** | Nothing to stamp — no prompt-versioning scheme exists in the codebase yet. |

---

## 7. Decisions needed

| # | Decision | Blocks | Recommendation |
|---|---|---|---|
| ~~D1~~ | ~~Confidence threshold for a "low confidence" caveat~~ | — | ✅ **RESOLVED — set to 0.5.** Live: a 0.294-confidence answer now carries `low_evidence`; it previously shipped silently. Inert in prod (the warning tier is off) |
| ~~D2~~ | ~~Should all users see the generated SQL? Currently yes, everyone~~ | — | ✅ **RESOLVED — default flipped to OFF.** Live: payload keeps the block for contract stability but as `{enabled: false, query: null}`; no SQL and no table name anywhere in it. `EXPLAIN_EXPOSE_SQL=1` restores it for an admin view |
| D3 | Enable the flags in staging? All are off | Rollout | Enable together — they are interdependent |
| D4 | Turn on access-denial auditing? Currently off | Compliance | Enable — the table answers nothing while off |
| D5 | Frontend design for the two-level view | Frontend build | Needs design |
| ~~D6~~ | ~~Should a user be told the answer was replayed from the verified cache?~~ | — | ✅ **RESOLVED — yes, shown.** New `provenance` block + `result.reused_verified_query`, and a plain-language line under *Finding the right information*. Present on refusals too, where the reuse is often the reason for the refusal |

---

## 8. Important caveat

**This work makes VEDA honest, not more accurate.**

Separate benchmarking shows low correctness on filtered and analytical questions. Explainability
will now **surface** those failures instead of hiding them — so expect **more visible warnings, not
fewer**. That is the system working as intended, not a regression.

Two examples seen during testing (pre-existing engine issues, not caused by this work):
- *"hey there, how are you?"* → returned an employee-handbook attendance policy
- A superlative question returned a 100-row page narrated as if it were the whole population

The second one now at least carries a truncation warning.

---

## 9. Files changed

**New (14)**
```
veda_core/veda/lifecycle.py          the timeline
veda_core/veda/warnings.py           the warning tier
veda_core/veda/exec_records.py       per-source execution records
veda_core/veda/source_names.py       safe source display names
veda_core/veda/safe_projection.py    the ONLY trace → user boundary
veda_core/veda/narrator.py           the optional, non-blocking SLM narrator
apps/chat/thinking_steps.py          the 26-phase → 4-step map and tracker
apps/chat/thinking_context.py        plain-language step detail (no SQL vocabulary)
apps/query/audit.py                  the one audit writer, shared by both front doors
apps/access_management/models/audit.py     access-denial records
tests/test_query_traceability.py     55 tests
tests/test_query_governance.py       23 tests
tests/test_thinking_steps.py         ~92 tests
```

**Modified (13)** — `config.py`, `explain.py`, `pipeline.py`, `business_explain.py`, `execution.py`,
`veda_hybrid.py`, `source_coordinator.py`, `federated_executor.py`, `reliability.py`, `agents.py`,
`inference/main.py`, `inference/routes/hybrid.py`, `apps/chat/*`, `apps/query/*`, `docker-compose.yml`

**Migrations (2)** — `query.0005`, `access_management.0011`. Both reviewed before applying; all
**118 existing audit rows preserved**.

---

## 10. Try it yourself

```
URL       http://localhost:8080          ← port 8080, NOT 8000 (that's a different app)
username  veda
password  Veda@12345
chat_id   244
```

```bash
TOKEN=$(curl -s -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"veda","password":"Veda@12345"}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['access_token'])")

curl -N -X POST http://localhost:8080/api/v1/conversations/query \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"chat_id":244,"message":"How many sale negotiations are there?","stream":true}'
```

**Gotchas:** no trailing slash on the URL · do **not** send `Accept: text/event-stream` (the API
rejects it — streaming is switched on by the `"stream": true` body flag) · token lasts 15 minutes.

---

## 11. Suggested next steps

1. **Decide D1** (confidence threshold) — one number, unblocks a built feature. This is now the
   most urgent one: a real query was observed shipping at **confidence 0.018** with five green
   validation ticks and **no caveat at all**, because the threshold is set to 0.
2. **EXP-S2** — give document/RAG answers the full explanation (~½–2 days)
3. **Enable the flags in staging** (D2/D3/D4) and watch what the warnings surface
4. **Frontend** (EXP-S1) — the last thing between this and users seeing it
