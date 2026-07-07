# Front Door & Non-SQL Heads (contract)

The front door routes each query to the head best at its modality. The SQL head
(L1–L7) is documented separately; this file covers the router and the breadth heads.

## Front-door router — `query/query_router.py`

> **Role:** classify a query as `sql | rag | hybrid | nosql` and pick which source
> IDs to query. Called before L1 when `QUERY_ROUTER_ENABLED=True`.

- **Consumes:** `query` + available source types/config.
- **Produces:** `RouteResult{ intent, source_ids, confidence (0..1), reason, stats }`.
- **Contract:** keyword signals first (fast, no model inference); falls back to
  embedding-based classification only when signals are ambiguous
  (`QUERY_ROUTER_CONFIDENCE_THRESHOLD`). Routing is explainable via `reason`.

## RAG head — `query/rag_layer.py :: run_rag_layer`

> **Role:** pure document synthesis for `rag` intent.

- **Consumes:** `query`, L1 `temporal_filter` (restricts chunks by document date),
  value expansion from `value_sampler` (boosts recall), `doc_chunks` store.
- **Pipeline:** MiniLM-embed query (same model as `chunk_embedder`) → cosine top-K
  over `doc_chunks` → single local-SLM synthesis.
- **Produces:** NL answer + cited chunks.
- **Invariant:** on-prem — MiniLM + local SLM, no external API calls.

## Hybrid head — `query/rag_layer.py :: run_hybrid_layer`

> **Role:** fuse structured + unstructured evidence for `hybrid` intent.

- **Consumes:** SQL-signal columns (from the SQL head's retrieval) + doc chunks.
- **Pipeline:** RRF-fuse SQL columns + doc chunks → unified context → single SLM
  call → combined answer.
- **Produces:** one answer grounded in both DB columns and documents.

## NoSQL head — `query/nosql_builder.py`

> **Role:** build a native NoSQL query for `nosql` intent.

- **Consumes:** `query` + `NoSQLCollection` schema list (optionally an `ir_json`).
- **Pipeline:** pick collection → detect `find | count | aggregate` → extract
  field-value filters → emit engine-native dict (MongoDB / Elasticsearch / DynamoDB),
  serialised for `connector.execute_query()`.
- **Invariant:** **no LLM** — deterministic keyword extraction; the IR-JSON path
  translates `filter_tree` directly to native predicates when supplied.

## Multi-result envelope — `query/multi_result.py`

All heads return through `MultiResult` / `SubResult` with a `STATUS_OK /
STATUS_REFUSED / STATUS_ERROR` status so the front door can compose answers from
several heads/sources uniformly.
