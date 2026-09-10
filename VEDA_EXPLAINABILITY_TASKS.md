# EPIC — Query Explainability & Traceability

**Owner:** engineering · **Updated:** 2026-09-09 · **Branch:** `feat/multisource-arch` (uncommitted)

**Goal.** For every query the user can see, in real time, *what the system is doing*, and afterwards
*how the answer was produced* — including honest warnings when the answer is partial, truncated,
restricted or degraded.

**Status.** Backend spine built and live-verified end to end through the real conversation SSE API.
**All 7 known bugs fixed and re-verified on the 10-query benchmark (2026-09-08).** The timeline is
now correct on every query path.

**Iteration 2 (2026-09-09) — the 4-step live thinking UX is built and verified.** The 26 internal
phases are now collapsed into **exactly 4 fixed steps** the user always sees, in the same order, for
every query type (SQL, documents, mixed, refusals, small talk). Authorization appears as a timed
sub-check inside step 2, not as a step of its own. An optional SLM narrator can make one step read
more naturally, and **cannot** delay the answer. Remaining work is the frontend (EXP-S1) and the
thinner explanation on document/RAG answers (EXP-S2).

| Metric | Value |
|---|---|
| Stories complete | **27 / 34** |
| Stories withdrawn | **1 / 34** (EXP-10, 2026-09-10) |
| Stories partial | **5 / 34** |
| Stories not started | **1 / 34** |
| Backend surface built | **~85%** |
| Behaving correctly on *all* query paths | **12 / 12 clean** on the re-run benchmark (was 55% pre-fix) |
| Additional LLM/SLM calls introduced | **0** required · **1 optional**, off by default, non-blocking |
| Measured latency overhead | **Not established** — see the note in `VEDA_EXPLAINABILITY_STATUS.md` §5. The ~0.3 ms figure was an isolated measurement of the trace projection, not an end-to-end result, and an A/B did not reproduce it. |
| Tests | **115** passing (55 traceability + 23 governance + ~92 thinking-step, run per-file) |
| Feature flags | 11, all default **OFF** — prod byte-identical until enabled |

> **Note for PM:** effort estimates are engineering-rough (S = <½ day, M = ½–2 days, L = 2–5 days,
> XL = >1 week). They assume the current codebase and no scope change.

---

# 1. COMPLETED — no further work needed

Each of these was verified against the running stack (real DB, real SSE endpoint), not just unit-tested.

| ID | Story | Verified by |
|---|---|---|
| EXP-01 | Real-time progress streams over the existing chat SSE API — no new endpoint, no polling | First event arrives ~0 ms on a 40.9 s query |
| EXP-02 | Existing clients unaffected — old progress events unchanged on the wire | Raw SSE capture shows legacy + new events side by side |
| EXP-03 | One internal trace reused; no second tracing system | `safe_projection` is the only reader of the trace |
| EXP-04 | Execution plan captured and shown (previously built then discarded) | Present in payload as `execution_plan` |
| EXP-05 | Data sources shown by **name**, not internal id | `"homzhub"` / `"Database"` in the live payload |
| EXP-06 | Per-source execution record — status, duration, rows | `duration_ms: 451, rows_returned: 20` live |
| EXP-07 | Routing explained in business language | `"One data source contains the information this question needs."` |
| EXP-08 | **Zero** extra AI calls for explainability | Per-call SLM ledger: 1 call total, and it was the answer text |
| EXP-09 | Two-level payload produced (summary + full detail) | `timeline_summary` + full `explainability` block |
| EXP-10 | ~~Self-service access endpoint `GET /api/v1/access/me`~~ | **WITHDRAWN 2026-09-10** — product owner decided the endpoint is not needed; it was built and verified, then removed from the codebase. |
| EXP-11 | Query audit row written from **both** front doors | Chat + direct API both produce a `QueryLog` row |
| EXP-12 | Audit records **who** ran the query (real user identity) | `user=veda` on the audit row |
| EXP-13 | Authorization denials are queryable, with cause | `explicit_deny` vs `no_grant` distinguished |
| EXP-14 | True database execution timing | `db_execution_ms` recorded |
| EXP-15 | Fixed: trace recorded blank data on every query | Was `datasets=None` on 100% of queries |

### Iteration 2 — the 4-step live thinking UX (2026-09-09)

| ID | Story | Verified by |
|---|---|---|
| EXP-T1 | **Exactly 4 fixed steps**, same order, every query type — *Understanding your request · Finding the right information · Analyzing the information · Preparing your answer* | 12-query live re-run: 12/12 produced all 4 steps in order |
| EXP-T2 | All **26** real internal phases mapped to a step; the map is derived from the code, not guessed | `PHASE_TO_STEP` covers every phase the engine can emit; a test fails if a phase is unmapped |
| EXP-T3 | Steps only ever move **forward** — a late internal phase cannot rewind the UI | `ThinkingStepTracker` is monotonic; guard test asserts no backwards transition |
| EXP-T4 | Authorization is a **timed sub-check inside step 2**, not a visible step | Live: `Checking your access · 58 ms` on a small-talk turn |
| EXP-T5 | Per-step progressive disclosure — collapsed one-liner + real detail on expand, in plain language | SQL vocabulary (`group by`, `order by`) translated by `_plain_operation`; guard test asserts none leaks |
| EXP-T6 | Optional SLM narrator: **one** call per query, own daemon thread, cancelled at terminal, validated before display | Regression tests + 4 live query shapes: **0** cases of thinking arriving after the answer |
| EXP-T7 | Narrator output is filtered, not trusted — banned terminology, invented numbers, ungrounded domain nouns all rejected | ~415-word general-English allowlist; guard test asserts it holds **no** business metric |
| EXP-T8 | No step is left spinning in the **saved** record — open phases are resolved before the payload is built | Live: `timeline_summary` has **0** unresolved phases (was leaving `access_check: started` forever) |

### Bugs found and fixed as side-effects (worth flagging to stakeholders)

| Finding | Why it mattered |
|---|---|
| Source-name header was **sent but never read** by the engine | Every explanation could only say `"2"` |
| Request context **lost in the streaming worker thread** | Source names blank on the chat path; **also silently broke data-lake routing** on that path |
| Trace read the wrong field names | `datasets` / `validation_passed` blank on every single query |
| Conversation history dropped the timeline on reload | User lost the execution detail they had just watched |
| `OLLAMA_URL` hardcoded in compose, overriding config | **Every AI answer silently fell back to a robotic template** |

---

# 2. CLOSED — bugs (all fixed 2026-09-08)

Re-verified by re-running the same 10-query benchmark and diffing phase coverage
before/after. **Stuck progress steps: 7 → 0. Missing validation lines: 5 → 0.**

| ID | Bug | Fix | Verified |
|---|---|---|---|
| EXP-B1 | Progress step never finished on 6/10 query types | Any phase left open is now closed at the shared end-of-query handler; a genuine denial is never overwritten | 7 stuck steps → **0** |
| EXP-B2 | Truncated results not flagged | Warning moved to where truncation is actually decided (was testing a 1000-row cap while truncation happens at 20) | Live: `⚠ result_truncated — limited to the first 100 records` on the query that previously showed none |
| EXP-B3 | Low-confidence answers looked identical to confident ones | `low_evidence` warning wired at the one place confidence is computed | Verified at threshold 0.5: `0.143 → warning`, `1.0 → none`. **Ships disabled pending decision D1** |
| EXP-B4 | "Running the query" missing for document questions | Dispatch wrapped once (covers all heads); only fills a gap, never double-counts | Step now present on 2 more query types |
| EXP-B5 | No validation line on refused answers | Emitted from the shared exit using the existing ledger; emits **nothing** when no checks ran | 5 missing → **0** |
| EXP-B6 | Misleading "one retrieval method was unavailable" | Reworded — the primary method could not *answer*; nothing was down | Live: new copy confirmed |
| EXP-B7 | Record writing scaled quadratically | Only the changed record is rewritten **and new records are appended, not rebuilt** | 200 records: **654 ms → 8.9 ms**. ⚠ The first fix was INCOMPLETE — see the correction note below |

**Bonus fix found while verifying B3:** a low-confidence answer produced a warning but an empty
`limitations` array. `low_evidence` now appears in both; `fallback_used` deliberately stays
informational-only (an alternate path still produced a complete answer).

**Correction on EXP-B7.** The first fix optimised only the record *close*; record
*open* still re-serialised the whole set, so the code was still quadratic. The original
wall-clock test (`< 200 ms for 200 records`) passed on an idle machine and hid it. Replacing
that assertion with a SCALING ratio (4x the records must not cost ~16x the time) exposed a
remaining **14.8x**, which is now **2.7x**. Lesson recorded: a performance claim asserted
against a wall-clock budget can pass while the complexity is unchanged.

**Test coverage:** 133 tests across three suites (55 traceability + 23 governance +
55 four-step UX), all passing.

# 3. OPEN — remaining scope

## EXP-S1 · Frontend: render the explanation
**Priority: P0 for launch** · **Effort: L** · **Blocked on: nothing — backend is ready**

All data is already delivered and re-readable from conversation history. This is pure UI work.

- [ ] Level 1: collapsed checklist ("How VEDA answered this")
- [ ] Level 2: expandable sections — Understanding / Data used / Execution / Validation / Warnings / Support ref
- [ ] Live progress rendering from the streamed events (group by step, show final state per step)
- [ ] Render warnings prominently — they are the reason this epic exists
- [ ] Handle old conversations gracefully (no timeline present)
- **Design input needed** before build starts.

> **Reading the benchmark:** a few steps still show as absent — e.g. no "Running the query" on a
> question that was refused before any data was fetched, and no validation line where no checks
> actually ran. Those are **correct**: the step genuinely did not happen, and printing a green tick
> there would be false. Absence is the honest representation.

## EXP-S2 · Explanation for document & hybrid answers
**Priority: P1** · **Effort: M**

Document/RAG/smalltalk answers return the **old, thinner** explanation format — no sources, no
routing, no warnings. Only database answers get the full one.

- [ ] Produce the full payload from the document and hybrid heads
- [ ] Reuse the existing projection layer (no new logic)
- [ ] Include document citations as evidence
- **Acceptance:** a document question returns the same payload shape as a database question.

## EXP-S3 · Cross-source explanation — verify against real data
**Priority: P2** · **Effort: M** · **Blocked on: multi-source being re-enabled**

Code is written (join match rates, source conflicts, partial failures) but **has never run on a real
multi-source query**, because multi-source routing is currently disabled on evidence it degrades
accuracy.

- [ ] Verify once multi-source is re-enabled
- [ ] Capture join keys for the technical view
- **Risk:** untested code path. Do not report as done until exercised.

## EXP-S4 · True parallel source execution
**Priority: P3** · **Effort: XL**

The plan is labelled "parallel" but executes sequentially. **We deliberately report it as
sequential rather than claim otherwise.**

- [ ] Concurrency with cancellation and per-source timeouts
- [ ] Permission checks before execution, not during
- [ ] Deterministic result merge
- [ ] Per-source timing and partial-failure capture
- **Recommendation:** defer. This is a performance project, not an explainability one.

## EXP-S5 · Prompt versioning
**Priority: P3** · **Effort: M**

Version stamping records code and schema versions. Prompt version cannot be recorded because **no
prompt-versioning scheme exists in the codebase**. Needs to be introduced first.

---

# 4. Decisions needed from product

| # | Decision | Blocks | Recommendation |
|---|---|---|---|
| D1 | Confidence threshold for a "low confidence" caveat | **EXP-B3 is built but disabled without this** | 0.5 (verified working at that value) |
| D2 | Should end users see the generated SQL? Currently **yes, for everyone** | — | Move behind a technical/admin view; flag already exists |
| D3 | Enable the features in staging? All 8 flags are OFF | rollout | Enable together; they are interdependent |
| D4 | Turn on authorization-denial auditing? Currently OFF | compliance | Enable — the table is useless while off |
| D5 | Frontend design for the two-level view | EXP-S1 | Needs design input |

---

# 5. Risks

| Risk | Note |
|---|---|
| **Nothing is enabled yet** | All 8 flags default OFF. Built and verified, but no user sees it until switched on. |
| **Not committed** | All work is uncommitted in the working tree. |
| **Cross-source path untested** | EXP-S3 code has never executed on real multi-source data. |
| **Answer quality is a separate problem** | This epic makes the system *honest*, not *more accurate*. Benchmarks separately show low correctness on filtered/analytical questions. Explainability will now **surface** those failures — expect more visible warnings, not fewer. |
| **Log-side disclosure (pre-existing)** | An unrelated component prints the full restricted-table list to container logs on every narrowed query. Needs its own decision. |

---

# 6. Suggested sequencing

| Sprint | Content | Outcome |
|---|---|---|
| ~~1~~ | ~~EXP-B1 … B7~~ | ✅ **DONE** — timeline correct on every query path |
| **1** | D1 (set the confidence threshold) + EXP-S2 + D2/D3/D4 · enable in staging | Full explanation on every answer type; live behind flags |
| **2** | EXP-S1 (frontend) | User-facing feature complete |
| **Later** | EXP-S3 (with multi-source), EXP-S4, EXP-S5 | Deferred, dependency-gated |

**Detail / evidence for every claim above:** `VEDA_TRACEABILITY_PHASE1.md`
