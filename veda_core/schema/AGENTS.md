# veda_core/schema/ — schema access shims

| File | Role | Status |
|------|------|--------|
| `real_schema.py` | One line: `from connectors.relational import get_real_schema`. The production schema accessor used by the relational L1 / L3. | wired |
| `simulate_schema.py` | Synthetic ~66-table real-estate schema (`SIMULATED_SCHEMA`, `get_simulated_schema`). A POC fallback "when no real DB". | **fallback-only** — still imported as a fallback branch by `config.py`, `../ingestion/{reg_builder,schema_scanner,value_sampler,data_graph}.py`. On no query path. Marked for removal in the archived cleanup plan after the fallback branches are cut. |

Not to be confused with `../ingestion/schema_scanner.py` (normalizes a raw schema dict) or
`../ingestion/schema_unifier.py` (connector dataclasses → the legacy dict).
