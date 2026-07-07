# Ingestion Layer Contracts

The ingestion flow is the **offline build** that turns a live source (PostgreSQL
schema, document store, or NoSQL/datalake) into every artifact the query runtime
trusts as fact. The runtime never introspects the source — it only reads what
ingestion publishes.

Entry: `dispatcher.dispatch(ctx)` routes by `source.type`. For `relational`,
`layers.pipeline.run_layered_ingestion(ctx)` composes five layers **L1→L5** in one
process, threading one in-memory `state` dict and honouring per-stage fatal
semantics.

```
SourceContext (from_env: DB Source row → config.get_source)
   └─ L1 EXTRACT   touch the source           (schema, FK, data-graph, values)
        └─ L2 ANALYZE   pure transforms        (types, table metadata, REG, join paths)
             └─ L3 ENRICH   the only LLM layer (semantic layer v2, glossary, concepts)
                  └─ L4 INDEX    embeddings + search  (graph embed, BGE, BM25, enrichment, rerank)
                       └─ L5 PUBLISH  derived registries + unified graph (atomic activate)
```

## The two boundary types (`ingestion/contracts.py`)

| Type | Role |
|---|---|
| `SourceContext` | The one source being ingested, resolved **once** from the injected DB Source row (`from_env`). Carries `source_id`, `tenant`, `type`, `engine`, `connection`, `exclude_tables`, `schema_filter`, `artifact_scope` (tenant, source, version), `skip_llm`, `resume`. Threaded to every layer. |
| `StageOutcome` | Result of one stage: `name`, `ok`, `fatal`, `detail`, `error`. `fatal and not ok` aborts the pipeline; otherwise the pipeline logs and continues. |

## Universal contract every layer honours

- **Signature:** `run(ctx: SourceContext, state: Dict, verbose: bool) -> List[StageOutcome]`
  (L1 also exposes `run_value_sampling`, sequenced by L2).
- **State threading:** each layer reads upstream keys from `state` and writes its
  own outputs back into `state` for the next layer.
- **Fatal vs non-fatal:** a `fatal=True` failure raises and aborts; a non-fatal
  failure records the error and the build proceeds with a degraded (but queryable)
  substrate.
- **Stage events:** the composer emits `[[STAGE]] <layer> <stage> <ok|fail|fatal>`
  on stdout so the Celery orchestrator (`apps/ingestion/tasks.py`) maps them to
  `IngestionStage` rows from real lifecycle events (no stdout-regex).
- **Artifact scope:** file artifacts land under
  `ARTIFACT_ROOT/<tenant>/<source>/<version>/` so N sources never collide.

## Contracts

| Layer | Contract | Touches source? | LLM? | Fatal stages |
|---|---|---|---|---|
| L1 | [L1_EXTRACT.md](L1_EXTRACT.md) | **yes** (only layer) | no | schema_scan, fk_adjacency |
| L2 | [L2_ANALYZE.md](L2_ANALYZE.md) | no | no | semantic_types, table_metadata, reg_graph |
| L3 | [L3_ENRICH.md](L3_ENRICH.md) | no | **yes** (only layer) | none (all non-fatal) |
| L4 | [L4_INDEX.md](L4_INDEX.md) | no | no (model inference) | none (all non-fatal) |
| L5 | [L5_PUBLISH.md](L5_PUBLISH.md) | no | no | none (all non-fatal) |
