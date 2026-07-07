# Query Layer Contracts

The query flow turns one natural-language question into a grounded, executed
answer. A **front door** routes each query to the head best suited to its modality;
the deterministic SQL head is a seven-stage pipeline **L1→L7**.

```
                        veda_hybrid.py  (front door)
                               │
                    query_router.py  → intent ∈ {sql, rag, hybrid, nosql}
        ┌──────────────────────┼───────────────────────┬─────────────────┐
        ▼                      ▼                       ▼                 ▼
   SQL head (L1–L7)      rag_layer.run_rag      rag_layer.run_hybrid   nosql_builder
   [CORRECTNESS]         [BREADTH: docs]        [SQL+docs, RRF]        [native NoSQL]
```

## The deterministic SQL head (`veda/pipeline.py :: run_query`)

| Layer | Stage(s) | Module | Contract |
|---|---|---|---|
| L1 | Temporal parse | `query/temporal_parser.py` | [L1_TEMPORAL.md](L1_TEMPORAL.md) |
| L2 | Retrieval (5-signal RRF) + graph-expand + rerank | `retrieval/*`, `query/reranker.py` | [L2_RETRIEVAL.md](L2_RETRIEVAL.md) |
| L3 | Routing / grain vet / schema linking | `veda/routing.py`, `query/target_selection.py` | [L3_ROUTING.md](L3_ROUTING.md) |
| L4 | Intent + existence/join planning | `query/intent.py`, `veda/planning.py` | [L4_INTENT.md](L4_INTENT.md) |
| L5 | SQL generation (deterministic or LLM) | `veda/generation.py`, `query/sql_builder.py` | [L5_SQLGEN.md](L5_SQLGEN.md) |
| L6 | Validation + qualifier gate + parameterize | `veda/validation.py` | [L6_VALIDATION.md](L6_VALIDATION.md) |
| L7 | Read-only execution + NL-back answer | `veda/execution.py`, `query/nl_answer.py` | [L7_EXECUTION.md](L7_EXECUTION.md) |

## Cross-cutting contract (the whole SQL head)

- **The LLM never writes SQL structure.** Joins are pinned by the deterministic
  planner; the LLM only fills a SELECT/WHERE inside a fixed join skeleton, and even
  that is gated by the IR-equivalence firewall + AST validator (L5/L6).
- **Return shape:** `run_query(query, sm, all_cols, return_result=True)` →
  `{status, ok, cols, rows, answer, sql, table, trace}`. `status ∈ {answered,
  no_table, refused, clarify, error, …}`; `ok == (status == "answered")`.
- **Grounded by construction:** every table/column named must resolve against the
  real schema, and every filter value must exist in the sampled value store —
  otherwise the query is **refused**, never guessed into SQL.
- **Fast paths short-circuit the pipeline:** the verified-query cache and the
  compiled-registry fast path (count/aggregate/dimension-list) answer before L2 with
  no retrieval and no LLM. Existence queries (`with/without/how-many-have`) take a
  deterministic semi/anti-join path and are never cached.
- **Trace:** every stage records to a `trace` object (`veda/explain.py`) surfaced in
  the result for explainability.

## Pre-L1 arbitration & simplification

| Concern | Module | Contract |
|---|---|---|
| Value-vs-column arbitration (runs before retrieval) | `query/value_arbiter.py` | classifies spans as SCHEMA_REF / VALUE / NEGATED_VALUE / ENTITY / UNKNOWN, grounded only by the `column_values` store. |
| Query simplification | `query/nl_simplifier.py` | rewrites verbose NL using sampled value hints before retrieval. |

## The non-SQL heads

| Head | Module | Contract |
|---|---|---|
| RAG (doc synthesis) | `query/rag_layer.py :: run_rag_layer` | [HEADS.md](HEADS.md) |
| Hybrid (SQL + docs, RRF) | `query/rag_layer.py :: run_hybrid_layer` | [HEADS.md](HEADS.md) |
| NoSQL (native Mongo/ES/DynamoDB) | `query/nosql_builder.py` | [HEADS.md](HEADS.md) |
| Front-door router | `query/query_router.py` | [HEADS.md](HEADS.md) |
