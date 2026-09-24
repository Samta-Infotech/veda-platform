# VEDA — 10-level drill-down: feasibility study and audit

Audit only; no code was changed for this study. Runtime evidence gathered 2026-09-24 on
the local stack against `homzhub_prod`. Anything not measured is marked **NOT VERIFIED**
rather than inferred.

---

## A. FEASIBILITY

**FEASIBLE WITH LIMITED EXTENSIONS — for single-source. NOT VERIFIED for multi-source and
federated.**

The reasons are narrow and specific, and none of them is the drill stack:

1. The **drill stack already works and already accumulates**. Measured to depth 2 with both
   filters present in the SQL and both levels on the stack.
2. The **data does not contain a 10-level hierarchy on one table**. The richest homzhub
   business table (`assets_asset`) has ~6 business-meaningful, word-valued dimensions; the
   rest of its 25 statistically-drillable columns are ids, numbers or empty strings. A real
   10-level chain would have to traverse RELATIONSHIPS (project → building → unit →
   tenant → transaction), which is a different capability from filtering one table.
3. The failures observed at level 3 and on drill-up are **two specific defects**, not a
   depth ceiling.

---

## B. WHY CURRENT DRILL-DOWN FAILS — the exact first failure point

Measured chain on `assets_asset`, capturing frame, stack and SQL at every turn:

| Turn | Message | status | frame filters | drill_path | SQL WHERE |
|---|---|---|---|---|---|
| 0 | distribution of properties by furnishing | answered | — | `[]` | — |
| 1 | only the Nagpur ones | answered | Location/city_name/nagpur | `[Location]` | `city_name` |
| 2 | only the EAST facing ones | answered | +Direction Facing/facing/east | `[Location, Direction Facing]` | `facing, city_name` |
| 3 | only the ones built in 2026 | **qualifier_dropped** | unchanged | unchanged | — |
| 4 | go back | answered | **`[]`** | **`[]`** | — |
| 5 | go back | answered | `[]` | `[]` | — |

**Levels 1 and 2 are correct in every respect** — filters accumulate, the stack deepens,
the SQL carries both predicates and keeps the GROUP BY.

**First failure: level 3, in the ENGINE, not in memory.** `qualifier_dropped` is the
qualifier gate refusing SQL that lost "2026". Memory behaved correctly: it did not write, so
the frame and the stack were left untouched. This is a SQL-generation limitation for a
numeric/temporal qualifier, not a drill-down limitation.

**Second failure: drill-up collapses to the root in ONE step.** After two levels, a single
"go back" emptied both the filters and the stack instead of returning to Nagpur.

---

## C. WHY IT APPEARS TO WORK FOR 2 LEVELS

Because it genuinely does. The structural half is sound: `ConversationContext` carries each
filter with its REAL column (harvested from the executed SQL's own IR, not from a label),
so level 2 arrives as `WHERE furnishing = %s AND city_name = %s`. Nothing is being
reconstructed from prose at this point.

What breaks at level 3 is the ENGINE's ability to ground a new kind of qualifier, and what
breaks on the way back is drill-up. Neither is a compounding-state problem.

---

## D. WHAT PREVENTS LEVELS 3-10

Three separate things, in order of how much they cost:

### D1. Drill-up returns to the root instead of one level — a REGRESSION I INTRODUCED

`chatbot/nodes.py:1943`:

```python
elif delta_type in ("remove", "drill_up") and not shape_delta:
    resolved = (frame.get("base_query") or "").strip() or message
```

The rendering this replaced (`chatbot/memory/frame.py:1375`) replayed `base_query` **only
when no filters remained** — the root pop — and otherwise restated the REMAINING context.
My boundary change collapsed that distinction, so every "go back" replays the original
question regardless of depth. The engine then returns an unfiltered result, and
`memory_write_node`'s stack prune —

```python
new_stack = [lvl for lvl in new_stack
             if any(_same_field(lvl["dimension"], f["field"]) for f in new_frame["filters"])]
```

— sees an empty filter list and drops every level. One `go back` therefore erases the whole
path. Without drill-up, depth beyond 2 is not usable even where drilling works.

### D2. The engine cannot ground some qualifier KINDS

Level 3 failed on "built in 2026" (a year), and separately measured, "only the gated ones"
(a boolean) produced `SELECT is_gated … LIMIT 1000` — dropping the GROUP BY and both
existing filters. Categorical text values ground reliably; numeric, temporal and boolean
ones do not. This caps usable depth at however many TEXT dimensions a table has.

### D3. The data does not offer 10 single-table levels

`assets_asset`, the richest business table, offers roughly: `furnishing`, `city_name`,
`facing`, `construction_year`, `postal_code`, `total_floors` — and three booleans that D2
says will fail. Ten levels on one table is not achievable with this data no matter how good
the code is; ten levels needs relationship traversal.

---

## E. SINGLE-SOURCE RESULT

**Verified depth: 2.** Two independent orderings reached level 2 with accumulating filters
and a preserved GROUP BY; both attempts at level 3 failed for the reasons in D2.

| Level | Frame | Drill path | Context | Source | SQL | Result | Failure |
|---|---|---|---|---|---|---|---|
| 1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| 2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| 3 | unchanged (correctly) | unchanged | ✅ | ✅ | ❌ | ❌ | engine: qualifier dropped |
| 4-10 | NOT VERIFIED | | | | | | not reached |

---

## F. MULTI-SOURCE RESULT — **NOT VERIFIED**

No multi-source drill chain was run in this study. What IS known from earlier measurement
this session: a follow-up could previously leak from a pinned relational source into
document retrieval, which is now guarded in two places (`classify()` lane continuity, and
`keep_entity_on_lane_change` which stops a wrong-lane answer becoming the conversation's
subject). Whether drill state survives an intentional multi-source drill is untested.

## G. FEDERATED / COMBINED-SOURCE RESULT — **NOT VERIFIED**

Known related fact, measured earlier: a federated base loses its aggregate shape on a
drill-down, because the shape-preserving branch lives in the single-table deterministic
path and federated SQL is built elsewhere. Join preservation across a drill was not tested.

---

## H. NEGATIVE CASES

| # | Case | Result |
|---|---|---|
| N1 | nonexistent dimension | **PASS** — `only the ones in Atlantis` refused, frame untouched |
| N3 | dimension in another source | **PASS (guarded)** — lane guard holds the entity; not exercised as a deliberate drill |
| N7 | drill after topic switch | **PASS** — `new_topic` resets the stack (`nodes.py`) |
| N9 | drill after filter removal | **PASS by construction** — the prune drops a removed filter's level, with a comment recording the bug it fixed |
| N12 | invalid/timeout SLM | **PASS** — `parse_delta_response` fails closed to `ambiguous` |
| N13/N14 | unauthorized / tenant | **PASS** — `_frame_still_authorised`, `_authorised_episodic`, per-turn RBAC |
| N15 | empty result | NOT VERIFIED |
| N2, N4, N5, N6, N8, N10, N11, N16, N17, N18 | | **NOT VERIFIED** |

---

## I. EDGE CASES

- **Maximum depth**: `_MAX_DRILL_DEPTH = 10` and pushes are capped by slicing — the stack
  can physically hold ten levels. **The drill stack is NOT the limiting factor.**
- **Persistence**: frame and stack survive Redis round-trips with a 7-day sliding TTL and
  optimistic `expected_version`; verified indirectly by reading the frame back between
  turns all session.
- **Concurrency**: `session_turn_lock` wraps the whole `graph.invoke`. Not stress-tested.
- **Repeated drill-up, drill-up from depth 10, high-cardinality dimensions, NULL values,
  duplicate dimension**: NOT VERIFIED.

---

## J. ROOT CAUSE

```
Primary root cause:
  Drill-UP collapses the entire path in one step, because context_resolve_node replays
  base_query for every drill_up rather than only for the root pop, and the stack prune in
  memory_write_node then drops every level whose dimension is no longer in the frame's
  filters. Introduced by my own boundary change; the rendering it replaced had the
  distinction.

Secondary root cause:
  The engine grounds CATEGORICAL TEXT qualifiers reliably and numeric / temporal / boolean
  ones unreliably, so depth is capped by how many text dimensions a table has.

Contributing factor:
  The data itself has no 10-level single-table hierarchy; ten levels requires relationship
  traversal, which is a different capability.

NOT the root cause:
  · the drill stack (holds 10, pushes and accumulates correctly to the depth reached)
  · QueryFrame state (entity, filters with real columns, group_by, measures all survive)
  · natural-language flattening (removed at the boundary; the resolved query is now the
    user's own words and the state travels structurally)
  · memory persistence (frame and stack read back correctly every turn)
  · model limitations (the classifier produced the right delta at every level measured)
```

---

## K. MINIMUM SAFE IMPLEMENTATION

1. **Restore the root-pop distinction** — replay `base_query` only when no filters remain
   after the pop; otherwise send the remaining context. One condition, `nodes.py:1943`.
2. **Make the stack prune tolerant of a turn that returns no filters** — a drill-up that
   pops to depth N should not be pruned against an empty filter list.
3. **Validate the dimension before applying a drill** — reject a drill onto a dimension the
   anchor does not have, instead of letting it produce an unfiltered query that resets the
   path (this is what "only the gated ones" did).
4. **Ground numeric/temporal/boolean qualifiers, or refuse them explicitly** — currently
   they silently produce a query that drops the existing filters.
5. Only then measure multi-source and federated drill, which are untested.

Nothing here needs a new memory system, a second router, or a new semantic layer.

---

## L. TEST PLAN

A chain per source type, capturing at every level: `status`, frame filters (field, column,
value), `drill_path`, and the SQL's WHERE columns — the shape used in section B, which is
what made the two defects visible.

- **Single source**: depth 1→6 using only TEXT dimensions, then drill-up 6→0 one level at a
  time, asserting the stack shortens by exactly one and the SQL loses exactly one predicate.
- **Negative**: N2, N4, N5, N6, N8, N10, N11, N15, N16, N17, N18 — none currently covered.
- **Multi-source / federated**: establish a base on each, then drill, asserting the source
  set and the join survive.
- **Authorization**: drill after a grant is revoked mid-conversation.
