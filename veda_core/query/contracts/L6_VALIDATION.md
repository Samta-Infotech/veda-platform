# L6 — VALIDATION (contract)

> **Role:** the security + correctness firewall. AST-validate the generated SQL,
> confirm value grounding, enforce qualifier completeness, and rewrite every literal
> into a bound parameter.

Module: `veda/validation.py` (`validate_and_parameterize`, `value_grounding`,
`qualifier_completeness`).

## Consumes

| Input | Source |
|---|---|
| `sql` | L5. |
| `allowed_tables`, `allowed_columns` | L5 firewall set. |
| `join_constraints`, `fanout_guard` | L4 planner. |
| `sm`, `column_values` | grounding checks. |

## Produces

| Output | Meaning |
|---|---|
| `param_sql` | SQL with all literals replaced by `%s` placeholders. |
| `params` | ordered bound-parameter list. |
| rejection | `err` string → `status="refused"` / validation-rejected. |

## Guarantees / invariants (enforced at the AST level)

- **Single, read-only `SELECT`** — any DML/DDL is rejected at the AST level.
- **Every table referenced exists** → blocks invented tables.
- **Every column referenced exists** → blocks invented columns.
- **All filter literals become `%s` parameters** → no value interpolation (SQLi-safe).
- **Value grounding:** filter values must exist in the sampled value store.
- **Qualifier-completeness gate (L6b):** every named qualifier in the NL question
  must be represented in the SQL — a dropped qualifier is a rejection, not a silent
  broadening.
- Fan-out guard from the planner is honoured so a join can't silently explode row
  counts.

## Failure semantics

Rejection is **terminal for this SQL**: `run_query` prints the rejection and returns
a refusal (`status != "answered"`). It does not retry with a relaxed validator.

## Downstream consumers

`param_sql` + `params` → L7 execution (parameterised, read-only). Rejections are
recorded to the trace (`output.sql`, refusal reason).
