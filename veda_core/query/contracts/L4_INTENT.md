# L4 — INTENT & PLANNING (contract)

> **Role:** classify what the query asks (intent), detect existence/aggregate
> operators, and — for multi-table questions — build the deterministic join plan.
> **The LLM never writes joins.**

Modules: `query/intent.py` (typed `QueryIntent` + `validate_intent`),
`query_engine/intent_detector.py` (fast-lane classification), `veda/planning.py`
(`existence_mode`, `aggregate_mode`, `try_multitable`).

## Consumes

| Input | Source |
|---|---|
| `query` | raw NL. |
| `results` / `primary` | L2/L3. |
| `sm`, relationship graph | join-key resolution (`veda_relationship_graph.json`). |
| `temporal_filter` | L1 (fast path). |

## Produces

| Output | Meaning |
|---|---|
| `intent` | `SIMPLE` / `MULTI_TABLE` / `AGGREGATE` (falls back to `SIMPLE` on detector error). |
| `is_existence` | `existence_mode(query)` — `exists` / `not_exists` / `how_many_have`, or `None`. |
| `mt` (join plan) | from `try_multitable`: `anchor`, `tables`, `mode`, join path, `confidence`, `max_fanout`. |
| `QueryIntent` | typed, registry-resolved description consumed by deterministic builders. |

## Guarantees / invariants

- **`validate_intent` is the firewall:** every column/metric named must resolve
  against the real schema and every filter value must exist (grounding); an intent
  that doesn't resolve is **DECLINED**, never turned into SQL. `build_sql` contains
  no English and no regex.
- **Existence queries** (`with/without/how-many-have`) are deterministic semi/
  anti-join operators — universal query grammar, not domain vocabulary. They are
  **never** served from the embedding cache (near-identical vectors, opposite SQL).
- Join planning fires for `MULTI_TABLE`/`AGGREGATE` and any existence query; the
  planner pins keys/paths and enforces a **fan-out guard** and `JOIN_CONFIDENCE_FLOOR`.
- The **fast path** (compiled registries) may resolve count/aggregate/dimension-list
  intents here with no retrieval and no LLM.

## Failure semantics

- Intent detector failure → `intent = "SIMPLE"` (safe default).
- Join plan below confidence floor → refusal with actionable feedback, not a guess.

## Downstream consumers

`intent` / `is_existence` / `mt` → L5 (which SQL builder path: existence,
pre-aggregation, single-table, or join-skeleton fill).
