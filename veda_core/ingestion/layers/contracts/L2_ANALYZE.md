# L2 — ANALYZE (contract)

> **Role:** pure transforms over extracted data. **No source touch, no LLM.**

Module: `ingestion/layers/l2_analyze.py` — wraps `semantic_type_inference`,
`vector_store.store_table_metadata`, `reg_builder`, `join_paths`, and (sequenced
here) L1's `run_value_sampling`.

## Consumes

| Input | Source |
|---|---|
| `ctx: SourceContext` | for `source_id`. |
| `state["scan_result"]` | L1 schema scan (tables, columns, FK edges). |

## Produces

| Stage | `state` key | Side effect | Fatal? |
|---|---|---|---|
| `semantic_types` | `inference_result` (`.stats.avg_confidence`, `.flagged_count`) | — | **FATAL** |
| `table_metadata` | `tm_result` (`.rows_written`) | writes table-metadata / display-columns store | **FATAL** |
| `value_profiling` | `vs_result` | (delegated to `l1_extract.run_value_sampling`, needs `inference_result`) | non-fatal |
| `reg_graph` | `graph` (`.stats.num_table_nodes`, `num_column_nodes`) | — | **FATAL** |
| `join_paths` | `join_paths` (dict of table-pair → shortest FK path) | writes precomputed join-path artifact | non-fatal (optional module) |

## Guarantees / invariants

- Deterministic: same `scan_result` → same outputs. No network/model calls.
- Semantic-type inference must succeed before value sampling (sampling reads
  `inference_result` to know which columns are categorical vs free-text).
- `join_paths` is **additive**: consumed by the query-time `join_planner` only when
  present; its absence never breaks planning (planner falls back to live BFS).

## Failure semantics

- `semantic_types`, `table_metadata`, `reg_graph` → **abort** (types, display
  columns, and the REG graph are load-bearing for every downstream layer).
- `value_profiling`, `join_paths` → logged, build continues.

## Downstream consumers

`inference_result` → L3 (semantic layer input schema), L4 (BGE biencoder embeds the
retrieval docs it produces). `graph` (REG) → L4 graph persist/embed. `join_paths` →
query L3/L4b join planning.
