# What a tester can test: conversation, short-term memory, drill-down

Written from a measured run on 2026-09-23 (60 questions from `test.py`, then three
conversation passes over the 27 that answer). Raw data: `evaluation/testpy_run/`.
The list of base questions that actually answer is in `WORKING_QUERIES.md`.

---

## 0. Read this first — how to judge pass/fail

**Judge the SQL, not the sentence.** The summary text is written by a model and has been
measured saying things the data does not support. Every answer has an SQL panel; that is
what actually ran.

**Filter values are parameterised.** A follow-up "only the ones in Pune" produces

    ... WHERE LOWER(CAST("city_name" AS TEXT)) = %s

The word *Pune* will NEVER appear in the SQL. Do not mark a filter as "not applied"
because you cannot find the value — look for the **WHERE clause and the column**. (This
exact mistake was made while measuring, and it turned a 16/16 pass into a false 0/16.)

**Use a fresh chat per scenario.** Memory is per-chat. Re-using one chat across scenarios
means a failure may be leftovers from the previous test, not the thing under test.

**Always record:** the exact messages in order, the SQL for each turn, the answer text,
and the chat id.

---

## 1. Drill down — narrowing an answer

The core feature: ask something, then narrow it without repeating yourself.

| # | Turn 1 | Turn 2 | Expect |
|---|---|---|---|
| D1 | `What is the distribution of properties by facing?` | `only the ones in Pune` | same table, WHERE on the city column, **and the GROUP BY kept** |
| D2 | `List sale listings with status 'APPROVED'.` | `only the ones in Pune` | same table, one more WHERE, previous status filter still there |
| D3 | `Show properties where is gated is true.` | `only the ones in Pune` | BOTH conditions present, not one replacing the other |
| D4 | `List properties with furnishing 'FULL'.` | `only the ones in Mumbai` | filter added; row count drops |
| D5 | `What is the distribution of properties by furnishing?` | `only the ones in Pune` | then `and only gated ones` — three levels deep |

**What to look for**
- The follow-up must query the **same table** as turn 1.
- It must **add** a condition, not replace the earlier one.
- The **shape** must survive: a distribution must stay a `GROUP BY` with a count, an
  average must stay an `AVG`.

**Known failing today — expect these to fail, report only if they CHANGE**
- Shape is lost on every aggregated base measured (0 of 8). D1 and D5 will come back as a
  raw row list instead of a distribution. This is the single biggest open defect.
- About 1 in 5 follow-ups get answered from the **documents** instead of the database
  ("The provided context does not contain… Sources: (site_notes.md)"). Worth recording
  which base questions trigger it.

## 2. Drill up — undoing one level

| # | Steps | Expect |
|---|---|---|
| U1 | base → `only the ones in Pune` → `go back` | back to the base question, filter gone, same table |
| U2 | base → filter → filter → `go back` | only the LAST filter is removed, the first stays |
| U3 | base → `go back` (nothing narrowed yet) | a clear "nothing to go back to", not an error and not a random answer |

U3 was measured behaving correctly: *"There's nothing to go back to — nothing has been
narrowed down yet in this conversation."*

## 3. Reset — making it forget

| # | Steps | Expect |
|---|---|---|
| R1 | base → filter → `start over` → `how many assets are there` | the new answer carries NO trace of the old filter |
| R2 | base → filter → `start over` → `go back` | nothing to go back to |
| R3 | try the phrasings: `start over`, `forget that`, `clear the context`, `let's start fresh`, `never mind` | all should reset |

Measured: 25 of 27 chats forgot the filter cleanly. R3's phrasing coverage is worth
pushing hard — reset is recognised by sentence structure, so unusual phrasings are the
interesting cases.

## 4. Recall — what the assistant remembers about the conversation

| # | Steps | Expect |
|---|---|---|
| M1 | base → `what did I just ask?` | restates the question; **must not re-run a query** (SQL panel empty) |
| M2 | base → filter → `what am I looking at now?` | mentions both the thing and the filter |
| M3 | base → `what did I ask before that?` | earlier turn, or an honest "that's all there is" |

Measured: 27/27 correctly did not re-run a query — but the CONTENT was not verified, and
at least one case answered with a generic *"I'm here for questions about your data"*,
which is a failure. **This is the least-tested area and the best place for a tester to
find things.**

## 5. Presentation — changing how it looks, not what it is

| # | Steps | Expect |
|---|---|---|
| P1 | base → `show it as a chart` | same SQL/entity, now with a chart |
| P2 | base → `as a table` | same data, table form |
| P3 | base → `only the ones in Pune` → `show as chart` | chart of the FILTERED data, filter not dropped |
| P4 | a document/policy question → `show as a chart` | should decline gracefully — there is nothing to chart |

Measured: 25/27 kept the same entity on P1.

## 6. Shape changes — same question, different slice

| # | Steps | Expect |
|---|---|---|
| S1 | `Show the latest 10 ledger entries.` → `just the last 5` | same query, LIMIT 5 |
| S2 | base list → `group them by entry type` | a GROUP BY appears, same table |
| S3 | base → `show me more` | a larger LIMIT, nothing else changed |

## 7. Topic switching — memory must not bleed

| # | Steps | Expect |
|---|---|---|
| T1 | `List sale listings with status 'APPROVED'.` → `only the ones in Pune` → `how many amenities are there` | the amenities answer has NO status/city filter |
| T2 | properties question → filter → ledger question | new table, no inherited WHERE |
| T3 | base → `what about lease listings?` | switches entity, does not silently keep the old one |

This is the one to be strict about: a leaked filter is a **wrong answer that looks right**.

## 8. Ambiguous and adversarial follow-ups

| # | Steps | Expect |
|---|---|---|
| A1 | base → `only the good ones` | should ask what "good" means, not invent a filter |
| A2 | base → `only the ones in Atlantis` | should say the value isn't in the data, not return 0 rows as if it were an answer |
| A3 | base → `that one` | should ask which, not guess |
| A4 | base → `only the debit ones` on a NON-ledger base | should say it can't apply it |
| A5 | send an empty message / only punctuation | graceful |

## 9. Ledger drill-down — currently broken, precise repro

    Turn 1: Show the latest 10 ledger entries.        -> answers, 10 rows
    Turn 2: only the debit ones                        -> FAILS

The failure is worth understanding because it is **not** about the user's word. The
resolved query becomes:

    only the debit ones (for Single Financial Transactions (accounts_generalledger))

and the engine then refuses with *"I couldn't map 'financial' to any column or value in
the data."* — `financial` comes from the table's business label that memory appended, not
from anything the tester typed. `debit` is a real value (`entry_type = 'DEBIT'`, 476 rows).

**Testers should try the variants** and record which get through:
`only the debit ones` · `only DEBIT entries` · `only the ones with entry type DEBIT` ·
`filter to debit` · `debit only`

## 10. Longer conversations

| # | Steps | Expect |
|---|---|---|
| L1 | 10+ turns mixing filters, drill-ups and topic switches | state stays coherent; no filter from turn 2 reappearing at turn 9 |
| L2 | leave a chat, come back later, continue with `only the ones in Pune` | the context is still there (memory has a 7-day window) |
| L3 | two browser tabs on the SAME chat, ask at the same time | no interleaving corruption |
| L4 | a user without access to a source: ask, then filter | memory must never resurface data they cannot see |

---

## Open defects already recorded — do not re-file, but DO tell us if they change

1. **Shape loss on drill-down** — an aggregated/grouped base becomes a raw row list once
   filtered. 0 of 8 survived.
2. **Document leakage** — roughly 1 in 5 follow-ups answered from documents instead of the
   database.
3. **Resolved query leaks internal names** — `(for Single Financial Transactions
   (accounts_generalledger))`; this also causes defect 4.
4. **Ledger drill-down blocked** by the word `financial` injected by defect 3.
5. **Summariser invents figures** — measured: *"5737 assets … in Pune"* where Pune has
   1,642 rows. Any number in the prose that is not in the table is worth reporting.
6. **Coverage**: only 27 of 60 sampled questions answer at all. The commonest refusal is
   *"I couldn't map '<ordinary English word>' to any column"* (database, exist, average,
   property, status, non, exceed, currency).

## What makes a good bug report here

- chat id, and the messages in order
- the SQL for each turn (copy from the SQL panel)
- what you expected the SQL to be
- whether the PROSE and the SQL disagree — that pair is the most valuable thing you can
  send us
