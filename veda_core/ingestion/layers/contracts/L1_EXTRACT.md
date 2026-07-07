# L1 — EXTRACT (contract)

> **Role:** the *only* layer that touches the tenant's source. After L1 completes,
> the rest of ingestion can finish even if the source goes down.

Module: `ingestion/layers/l1_extract.py` — thin wrappers over the existing
`schema_scanner`, `data_graph`, `vector_store`, `value_sampler` stages.

## Consumes

| Input | Shape / source |
|---|---|
| `ctx: SourceContext` | Resolved source (connection, `source_id`, `exclude_tables`, `schema_filter`). |
| live source | PostgreSQL, read via `schema.real_schema.get_real_schema()`. |
| `state["inference_result"]` | **Only** for `run_value_sampling` — produced by L2, so value sampling is sequenced by L2 after type inference. |

## Produces

Writes into `state` and to the internal store:

| Stage | `state` key | Side effect | Fatal? |
|---|---|---|---|
| `schema_scan` | `scan_result` (`.stats`, `.fk_edges`) | none (read) | **FATAL** |
| `fk_adjacency` | `fk_result` (`.edges_written`, `.backend`) | writes FK-adjacency store | **FATAL** |
| `data_graph` | `dg_result` (`.discovered_edges`, `.stats`) | appends discovered edges to FK-adjacency (HIGH/MED only, `include_soft=False`) | non-fatal |
| `value_profiling` | `vs_result` (`.columns_sampled`, `.total_values`) | writes `column_values` store | non-fatal |

## Guarantees / invariants

- Runs against **one** source (`ctx`), never a re-derived "primary".
- Internal/embedding tables stay in `ctx.exclude_tables` — VEDA never treats its
  own stores as business tables.
- Data-graph edges are only promoted to the FK store at HIGH/MEDIUM certainty;
  SOFT edges are excluded from the hard adjacency.

## Failure semantics

- `schema_scan` or `fk_adjacency` failure → **abort** (no downstream layer can run
  without a schema and key graph).
- `data_graph` / `value_profiling` failure → logged, build continues (undeclared-FK
  discovery and value grounding degrade, but the schema is still queryable).

## Downstream consumers

`scan_result` → L2 (types, REG, join paths), L4 (graph persist). `fk_result` →
join planner at query time. `dg_result` → L4 graph persist. `vs_result` /
`column_values` → query-time value grounding & value-mirror (L5).
