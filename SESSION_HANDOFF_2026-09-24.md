# Session handoff — conversation memory & drill-down

Written 2026-09-24 for whoever picks this up next. Everything below was measured on the
local stack; where something was not measured it says so. Nothing has been committed —
the work is staged in the index.

---

## 1. Read this first: traps that cost real time

These are not opinions. Each one produced a wrong conclusion before it was caught.

**`chatbot/state.py` must declare every key a node returns.** LangGraph's `StateGraph` is
typed by that TypedDict and **silently drops** anything not in it. `conversation_context`
went missing from that file (working tree, index AND HEAD, around a merge) and the entire
memory boundary went quiet — `context_resolve_node` still classified correctly and still
passed the user's words through, so every visible signal looked right, while the context
never reached the engine. Every follow-up became a bare fragment, got routed to RAG, and
then overwrote the frame with a document. **If follow-ups start answering from documents,
check this field first.**

**`run_chat_turn()` does not return the whole state.** It omits `resolved_query`,
`conversation_context` and `history`. Reading them from its return value measures nothing.
Read the checkpoint instead:

```python
g = get_graph(); v = g.get_state({"configurable": {"thread_id": session_id}}).values
```

**Filter values are parameterised.** `only the ones in Pune` produces
`WHERE LOWER(CAST("city_name" AS TEXT)) = %s` — the word *Pune* never appears in the SQL.
Grepping for the literal made a working 16/16 look like 0/16. Check the WHERE clause and
the column.

**Never test during container warmup.** After `docker compose restart`, the inference tier
loads BGE-M3, the reranker and the SLM. Queries run during that window return empty
answers and look like total failure. Wait for `curl -sf http://localhost:8001/healthz`.

**The verified cache is keyed on query TEXT.** A bare follow-up fragment is the same three
words whatever it follows, so one cached entry was being replayed for unrelated base
questions. Context-dependent turns are now excluded from that cache, but old poisoned rows
may still exist in `substrate_verifiedquerycache` (delete by `query_text`).

**Every source-DB query opens a NEW connection.** `veda/execution.py::execute_sql` does
`psycopg2.connect()` per call to a DigitalOcean-hosted Postgres — `SELECT 1` costs ~2.4s
steady state. Do not write discovery scripts that loop a query per column; one such script
ran for minutes. Use `pg_stats` instead (see §5).

---

## 2. What is working now (measured)

| Capability | Evidence |
|---|---|
| User's words reach the engine unchanged | contract test + live; `resolved_query == message` on every normal turn |
| Conversation state travels structurally | `flags.conversation_context`, validated at the inference boundary |
| 2-level drill-down, filters ACCUMULATE | `WHERE facing = %s AND city_name = %s`, GROUP BY intact |
| Drill-up steps ONE level | stack 2 → 1 → 0, one predicate dropped each time |
| Clean drill is STABLE | same chain 3 runs, byte-identical |
| A failed question no longer glues onto the next | was `"list 5 ledger… for only the Nagpur ones"` |
| A wrong-lane answer cannot hijack the frame | nonsense turn leaves entity untouched |
| Ungroundable drill refuses and KEEPS state | `only the gated ones` → clear refusal, stack held |
| `start over` | 27/27 clean |
| Ranking (`top 5`) produces a real ORDER BY | matches ground truth |
| Aggregate shape survives a drill | grouped and scalar (SUM/AVG/COUNT) |

## 3. What is still broken (measured)

1. **The turn after a REFUSED turn loses the drill path** — 3 runs out of 3. The follow-up
   comes back `delta=ambiguous`, no context is carried, and the `reset` branch
   (`is_topic_switch`) clears the stack before the prune is reached. **This is the next
   task.** The fix should apply the same principle as `keep_entity_on_lane_change`: a turn
   the classifier could not place must not be allowed to rewrite the conversation's subject.
   Do NOT patch the prune again — the problem is not there.
2. **Level 3+ does not work.** Categorical TEXT values ground; boolean, numeric and
   temporal ones do not (`only the gated ones`, `built in 2026`).
3. **Coverage is 27 of 60** sampled `test.py` questions. The commonest refusal is
   *"I couldn't map '<ordinary English word>' to any column"* — and it is usually the gate
   correctly catching SQL that really did lose the word, not a gate bug. Verified on two
   cases: the anchor or the aggregate was genuinely wrong upstream.
4. **Multi-source and federated drill: NOT VERIFIED.** Never tested end to end.
5. **Summariser invents figures** — measured: *"5737 assets in Pune"* when Pune has 1,642.

---

## 4. Where things live

| Concern | File |
|---|---|
| Structured context handed to the engine | `chatbot/memory/context.py` (`ConversationContext`) |
| Frame, drill stack, deltas, guards | `chatbot/memory/frame.py` |
| Graph nodes, classify, context resolve, memory write | `chatbot/nodes.py` |
| **State schema — additions MUST go here** | `chatbot/state.py` |
| Boundary validation of the context | `inference/routes/hybrid.py::_validated_conversation_context` |
| Ambient context inside the engine | `veda_core/context.py` |
| Context consumption (anchor, filters, shape, gate basis) | `veda_core/veda/pipeline.py` |
| Tier-2 gate (also guards conversation state) | `veda_core/veda_hybrid.py::_tier2_validate` |
| Raw column carried into explain | `veda_core/veda/business_explain.py` |

---

## 5. Useful techniques from this session

**Finding drillable dimensions without scanning tables** — Postgres already stores each
column's commonest value and its frequency:

```sql
SELECT s.tablename, s.attname,
       (s.most_common_vals::text::text[])[1] AS top_value,
       s.most_common_freqs[1] AS share
FROM pg_stats s
WHERE s.schemaname = 'homzhub' AND s.most_common_freqs[1] BETWEEN 0.05 AND 0.60;
```

A value covering 5-60% of the rows is a real narrowing. This is how the verified chains in
`HOMZHUB_DRILLDOWN_CHAINS.md` were chosen.

**Checking the staged tree actually works** — the index went out of sync twice this
session, once leaving it unimportable:

```bash
git checkout-index -a --prefix=/tmp/chk/ && cd /tmp/chk && \
  python -m pytest tests/test_conversation_context_boundary.py -q
```

---

## 6. Documents produced

| File | What it holds |
|---|---|
| `VEDA_CONVERSATION_BOUNDARY_AUDIT.md` | why the boundary was wrong and what the fix had to be |
| `VEDA_DRILLDOWN_10LEVEL_FEASIBILITY.md` | 10-level feasibility, root cause, minimum plan |
| `HOMZHUB_DRILLDOWN_CHAINS.md` | chains verified end to end, with the SQL each produced |
| `DRILLDOWN_FAQ.md` | user-facing: what can be drilled, what cannot, and why |
| `CONVERSATION_MEMORY_TEST_CHARTER.md` | what a tester should try, and how to judge pass/fail |
| `WORKING_QUERIES.md` | the 27 questions that answer, and the refusals grouped by reason |
| `evaluation/testpy_run/` | harnesses and raw JSONL for every run |
| `PM_LOG.md` | one row per task, newest last — the detailed record |

---

## 6a. Older memory documents — read the date first

`VEDA_MEMORY_FULL_AUDIT.md`, `VEDA_STM_IMPLEMENTATION_AUDIT.md` and
`VEDA_MEMORY_GAP_IMPLEMENTATION_REPORT.md` describe the system BEFORE this session. They
are still the best account of how the memory layer was designed and why, but their
statements about what WORKS predate the boundary change, the drill-stack fixes and the
lost-state-field bug — so any pass/fail number in them is stale. Read the design, not the
verdicts, and take the verdicts from §2 and §3 above.

## 7. Next steps, in order

1. **Stop an `ambiguous` turn from rewriting the conversation's subject** (§3 item 1).
2. Ground or explicitly refuse boolean / numeric / temporal qualifiers.
3. Benchmark multi-source and federated drill — currently unmeasured, not proven broken.
4. Run the 1→10 + drill-up + negative/edge matrix in
   `VEDA_DRILLDOWN_10LEVEL_FEASIBILITY.md` §L.
5. Separately: the source DB has no connection pooling and each query pays ~2.4s. That is
   the single biggest latency win available and changes no behaviour.

## 8. Repository state

Nothing is committed. The work is staged, including files that were previously **untracked
but imported by production code** (`chatbot/prompts/delta_types.py`,
`apps/query/data_vocabulary.py`) and 14 memory test files that CI had never seen.
Verify before committing with the `checkout-index` recipe in §5.
