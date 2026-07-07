# L5 — PUBLISH (contract)

> **Role:** derived registries + unified graph — the **atomic activate** point.
> Everything is written, then the query tier is told to rehydrate. Pure transforms
> of the semantic model (LLM-free), so it runs under `skip_llm` too.

Module: `ingestion/layers/l5_publish.py` — wraps `relationship_graph`,
`semantic.compile_semantic_layer`, `value_mirror`, `unified_graph_builder`, plus
per-source HNSW tuning.

## Consumes

| Input | Source |
|---|---|
| `ctx: SourceContext` | `source_id`, `tenant`. |
| `state["scan_result"]` | L1 — table count drives HNSW `ef_search` tuning. |
| semantic model file | written by L3 (read from disk by the registry/graph builders). |
| config | `DERIVED_ARTIFACTS_ENABLED`, `UNIFIED_GRAPH_ENABLED`. |

## Produces

| Stage | Side effect | Gate | Fatal? |
|---|---|---|---|
| `relationship_graph` | `veda_relationship_graph.json` (keys, paths, cardinality, weights) | `DERIVED_ARTIFACTS_ENABLED` | non-fatal |
| `semantic_registry` | `semantic/*.json` (concepts, dimensions, metrics, MANIFEST) — fast-path source | `DERIVED_ARTIFACTS_ENABLED` | non-fatal |
| `hnsw_tune` | `veda_hnsw.json` (`hnsw_ef_search` clamped to [40,200] by table count) + `state["hnsw_ef_search"]` | — | non-fatal |
| `value_mirror` | Redis mirror of `column_values` (Q-5) | — | non-fatal |
| `unified_graph` | unified-graph artifact (nodes/edges) for query-time `GRAPH_EXPAND` | `UNIFIED_GRAPH_ENABLED` | non-fatal |

## Guarantees / invariants

- Registry + relationship-graph builds are **pure transforms of the semantic
  model** → deterministic and LLM-free (valid even on a `skip_llm` run).
- `hnsw_ef_search = clamp(40 + (n_tables // 20) * 20, 40, 200)` — larger schema →
  wider search; 40 is the shipped default. Persisted for the activate step to store
  on `SubstrateVersion.hnsw_ef_search`.
- This is the **atomic-activate** boundary: the query tier only flips to the new
  version once L5 has published all registries.

## Failure semantics

Non-fatal throughout. A failed publish leaves the previous activated version in
place (the query tier keeps serving the last good `SubstrateVersion`).

## Downstream consumers

`veda_relationship_graph.json` → query L3/L4b join planner & fast path.
`semantic/*.json` → query fast path (count/aggregate/dimension-list). `veda_hnsw.json`
→ activate → per-source pgvector search width. Redis value mirror → query L2 value
signal / value grounding. unified graph → query L2g `GRAPH_EXPAND`.
