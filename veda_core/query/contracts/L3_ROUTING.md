# L3 — ROUTING / SCHEMA LINKING (contract)

> **Role:** pick the **primary (anchor) table** from the retrieved candidates, then
> vet it against query word-order and grain hints.

Modules: `veda/routing.py` (`route_tables_semantic`, `select_primary_table`,
`vet_primary`), `query/target_selection.py`, `query/schema_linker.py`.

## Consumes

| Input | Source |
|---|---|
| `results` | L2 retrieved columns (reranked). |
| `query` | raw NL. |
| `sm` | semantic model (grain, importance weights). |
| `table_embeddings_v2` | ingestion L4 — for `route_tables_semantic` cosine table ranking. |

## Produces

| Output | Meaning |
|---|---|
| `_cand_tabs` | candidate tables in retrieval order. |
| `_router_primary` | table chosen by `select_primary_table`. |
| `primary` | final anchor after `vet_primary` (may override the router on word-order / grain-hint). |

`target_selection.Target` (evidence-based): `table`, `matched_tokens`,
`lexical_score` (0..1 distinctive name tokens named), `retrieval_score` (0..1 max
column score). Identifies **requested entities only** — it does not plan joins,
resolve keys, or aggregate.

## Guarantees / invariants

- `route_tables_semantic` returns `{table: similarity}` or **`{}`** if
  `table_embeddings_v2` isn't built — graceful fall back to the lexical/column path.
- `vet_primary` is allowed to override the router primary and records the reason to
  the trace.
- Target selection is strictly evidence-in / entities-out: junctions and hubs are
  introduced later by the join planner (L4b), **never here**.

## Failure semantics

- No `primary` selected → `_feedback("no_table")`, return `status="no_table"` (a
  refusal, not a crash).

## Downstream consumers

`primary` + candidate tables → L4 (existence/join decision, `try_multitable`), L5
(single-table vs join SQL), L6 (`allowed_tables`/`allowed_columns` firewall set).
