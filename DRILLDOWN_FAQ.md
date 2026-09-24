# Drill-down & follow-ups — FAQ

Every answer here is measured on the local stack, 2026-09-23/24. Where something is
unverified it says so. Raw data: `evaluation/testpy_run/`, per-question table in
`DRILLDOWN_WORKING_LIST.md`.

---

## Can I drill into any question?

**No.** Of 60 questions sampled from `test.py`, **27 answer at all**, and of those **17 can
be drilled into**. The other 10 cannot — not because drill-down is broken, but because
their table has nothing to narrow BY.

## Why can't those 10 be drilled into?

A follow-up like "only the X ones" needs a column whose values actually split the rows.
The rule used to decide this is: does the table have a text column whose commonest value
covers between 5% and 60% of the rows? If not, there is no honest follow-up to offer.

Examples that answer but cannot be drilled:

| Question | Table | Why not |
|---|---|---|
| List all amenities. | `assets_amenity` | no column narrows to 5-60% |
| List all payment types. | `accounts_paymenttype` | only 6 rows |
| List all amenity categories. | `assets_amenitycategory` | only 3 rows |
| What are the names of all projects? | `assets_project` | its commonest value is a geometry blob, not a word |
| How many payment transaction history records exist? | `accounts_paymenttransactionsettlement` | only 3 rows |

## What makes a question GOOD for drill-down?

Three things together:

1. it answers on its own,
2. its table has real dimensions (city, status, furnishing, entry type), and
3. it is a **distribution or an aggregate**, so narrowing changes the figure rather than
   just shortening a list.

**Best measured example — a clean 4-turn chain:**

```
1. What is the distribution of lease listings by status?
2. only the FULL ones        -> same GROUP BY, plus WHERE furnishing
3. what about SEMI           -> FULL replaced by SEMI, not added to it
4. go back                   -> the original question returns
```

## Which follow-up wordings work?

Measured across two bases:

| Works | Does not work |
|---|---|
| `only the X ones` | `filter to X` — the word *filter* is read as data |
| `just X` | `just the top 5` — refused by the ranking check |
| `show only X` | `what did I just ask?` — reaches the engine and refuses |
| `what about Y` (replaces the value) | |
| `remove that filter` | |
| `show it as a chart` | |
| `go back` (see below) | |

The pattern is simple: **name the VALUE, not the operation.** "only the DEBIT ones" works;
"filter to DEBIT" does not, because *filter* is taken as something to look up in the data.

## How many levels deep can I go?

**One level is reliable. A second level sometimes works.** Measured on four bases: all four
accepted the first narrowing; two accepted a second. Treat level 2 as untrusted for now.

## Does "go back" always work?

**No — this is a known open bug.** It works on some chains and on others replies
*"There's nothing to go back to — nothing has been narrowed down yet"* even after two
narrowings had been applied. The drill stack is not being pushed on every turn.

`start over` is reliable (measured 27/27) and is the safe way to reset.

## What happens when a follow-up can't be answered?

It should say why, in plain words. A good example, measured:

> *"I couldn't map 'nagpur' to any column or value in the data."*

on `List sale listings with status 'APPROVED'` — correct, because `assets_salelisting` has
no city column at all.

Messages that are still **not** good enough, and are open issues:

- an answer that comes from the documents (*"The provided context does not contain…"*) when
  the question was about database records
- *"nothing has been narrowed down yet"* after narrowing did happen

## Why did my follow-up get answered from a PDF?

That was a real bug, fixed 2026-09-24. A single field (`conversation_context`) had gone
missing from the conversation state schema, so the remembered table never reached the
engine and every follow-up arrived as a bare fragment — which the router then sent to the
document side. If you see this again, it is a regression worth reporting immediately.

## How long does a follow-up take?

| | median |
|---|---|
| First question | ~22s |
| Follow-up | **~25s** |
| `go back` / `start over` | **0.1s** |

A follow-up is not cheaper than the original question — the whole pipeline runs again.
Only navigation is instant, because it never reaches the engine.

## What should I report as a bug?

The most valuable report is **the SQL and the sentence disagreeing**. Send:

- the chat id and the messages in order,
- the SQL panel contents for each turn,
- what you expected the SQL to be.

Note that filter values are **parameterised**: "only the ones in Pune" produces
`WHERE LOWER(CAST("city_name" AS TEXT)) = %s` and the word *Pune* never appears in the SQL.
Look for the WHERE clause and the column, not the value.

## What is NOT supported yet?

- **Referring to a previous RESULT** — "show me the second one", "compare it with the
  previous result". The system stores the SQL and the row count, not row identity. It does
  not pretend to support this.
- **Drilling a cross-source or CTE-based answer** keeps the filter but loses the grouping.
