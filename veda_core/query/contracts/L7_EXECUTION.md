# L7 — EXECUTION & NL ANSWER (contract)

> **Role:** run the validated SQL against the client source DB under a read-only
> connection, then (L7b) summarise the rows into a one-line prose answer.

Modules: `veda/execution.py` (`execute_sql`), `veda/runtime.py` (`get_db_config`),
`query/nl_answer.py` (`run_nl_answer`).

## Consumes

| Input | Source |
|---|---|
| `param_sql`, `params` | L6 (parameterised). |
| DB connection | `get_db_config()` — resolved from the request's `Source` row via `storage_adapters.reader.source_connection()`; falls back to the injected primary source outside a request. |
| `query`, `cols`, `rows` | for the NL-back answer. |

## Produces

| Output | Meaning |
|---|---|
| `cols`, `rows` | executed result (fetch ≤ 20 rows). |
| `answer` | one-line NL summary (L7b), or `None`. |
| final `_done(...)` | `{status:"answered", ok:True, cols, rows, answer, sql, table, trace}`. |

## Guarantees / invariants

- **Read-only + isolated:** connection is `set_session(readonly=True, autocommit=True)`
  — no writes can occur even if a DML statement slipped past L6.
- Bounded: 30s timeout, fetch ≤ 20 rows.
- **Connection origin is always the `Source` table** — no hardcoded host/creds, no
  static `.env`; a missing source is a hard fail, never a silent localhost fallback.
  One warm engine serves N sources via the ambient request context.
- **L7b NL-back** (`NL_ANSWER_ENABLED`): local SLM turns rows into prose with a
  deterministic row-count fallback if Ollama is unavailable — failure never blocks
  the answer (the table is still returned).

## Post-execution: verified-query cache

Successful, non-temporal, non-existence, non-fast-path results are saved via
`save_verified_query(query, sql)` so a similar future query skips L2–L5. Fast-path
and temporal results are intentionally **not** cached.

## Failure semantics

- Execution error → printed, `status="error"` returned (no partial rows claimed).
- The route is logged (`log_route`) with latency, table, and row count regardless of
  outcome.

## Downstream consumers

The returned dict flows back to the front door (`veda_hybrid.py`) — for `hybrid`
intent its rows/answer are fused with document evidence; otherwise it is the final
response.
