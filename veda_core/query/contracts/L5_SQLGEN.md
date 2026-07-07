# L5 — SQL GENERATION (contract)

> **Role:** produce the SQL string. Structure (joins, grain) is fixed
> deterministically; the LLM only fills a SELECT/WHERE inside a pinned skeleton —
> and only when the deterministic path can't.

Modules: `veda/generation.py` (`generate_sql`, `generate_join_sql`),
`query/sql_builder.py`, `query/answer_entity.py`, `query/value_resolver.py`,
`query/fk_path_resolver.py`, `query/value_arbiter.py`.

## Consumes

| Input | Source |
|---|---|
| `query`, `primary`, `results` | L2/L3. |
| `intent`, `is_existence`, `mt` join plan | L4. |
| `temporal_filter` | L1 (grounded predicate on canonical temporal column). |
| `sm`, relationship graph, `column_values` | grounding + join keys + value resolution. |

## Produces

- `sql` — the SQL text (pre-validation, pre-parameterization).
- `allowed_tables`, `allowed_columns` — the firewall set handed to L6.

## Paths (in priority order)

| Path | Trigger | LLM? |
|---|---|---|
| Existence semi/anti-join | `is_existence` | no |
| Deterministic pre-aggregation | grain/aggregate plan | no |
| Answer-entity / FK value resolution / multi-hop FK | value-grounded filters resolve to FK targets | no |
| Value-arbiter filter | span classified VALUE / NEGATED_VALUE | no |
| Temporal-only | only a date window, no other predicate | no |
| Single-table select | `SIMPLE` intent | LLM fills SELECT/WHERE |
| Join-skeleton fill | `mt` plan present | LLM fills leaves; **join skeleton fixed** |

## Guarantees / invariants

- **The LLM never writes SQL structure** — joins come from the L4 planner; the LLM
  only fills columns/filters inside a fixed skeleton. Enforced downstream by the
  IR-equivalence firewall (`veda/ir_equivalence.py`) + L6 AST validator.
- Optional `DOMAIN_CONTEXT` primer is **descriptive only** — it forbids inventing
  filters/rules; the IR-equivalence check enforces that no unrequested predicate is
  added.
- Filter values are grounded against `column_values`; an ungroundable value →
  refusal, not a hallucinated literal.
- Temporal predicates reuse the single canonical-temporal chooser
  (`sql_builder._pick_best_temporal`) — one source of truth for event-time.

## Failure semantics

- Ungroundable value / temporal-refuse / no resolvable path → `status="refused"`
  (or `clarify`) with feedback; never emits a guessed query.

## Downstream consumers

`sql` + `allowed_tables`/`allowed_columns` → L6 validation & parameterization.
