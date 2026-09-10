# veda_core/query/ — routing, IR/SLM, the non-SQL heads, multi-source

~53 `.py` files. The SQL head itself is in `../veda/`; this package is the front-door
routing, the LLM seam, the RAG / hybrid / NoSQL heads, and the multi-source machinery.
Full walkthrough: [../../docs/QUERY_ENGINE.md](../../docs/QUERY_ENGINE.md),
[../../docs/MULTI_SOURCE.md](../../docs/MULTI_SOURCE.md). Contract docs: `contracts/`.

## Front door / intent routing
| File | Role |
|------|------|
| `query_router.py` | Keyword-signal intent classifier → `RouteResult{intent∈sql/rag/hybrid/nosql, source_ids, confidence}`. No model inference, **no embedding fallback** (the imported `QUERY_ROUTER_CONFIDENCE_THRESHOLD` is unused). Returns `sql` unconditionally with no doc/nosql sources. |
| `routing_contracts.py` | Typed routing dataclasses (`RoutingDecision`, `CandidateSource`, tiers). |
| `routing_policy.py` | Pure deterministic source-routing decision (`decide()`) over candidates + FK edges (presence tier, `cross_source_fk`, `is_canonical`). |
| `routing_slm.py` | Bounded SLM ambiguity resolver — called ONLY when `routing_policy.decide` returns `AMBIGUOUS`. `resolve_boundary()`. |
| `source_coordinator.py` | The multi-source coordinator: `plan_route()` / `execute_decision()`. **Runs in shadow only** (`MULTISOURCE_ROUTING_SHADOW=1`). Holds the `_ROUTING_QV` embed-once ContextVar. |
| `source_evidence.py` | `SourceEvidence` + `group_evidence_by_source(cols, chunks)`. |
| `operation_classifier.py` | Bounded closed-enum cross-source OPERATION classifier for a MULTI query. |
| `execution_planner.py` | `RoutingDecision` → `ExecutionPlan` (`single` / `federated` / `independent`). |
| `reliability.py` | Bounded transient-retry wrapper + `classify_failure(err) → transient/permanent`. |
| `agents.py` | One thin execution agent per source KIND (relational / datalake / document / nosql); `_AGENT_BY_KIND`. |
| `result_orchestrator.py` | Merges independent multi-source `AgentResult`s (`APPEND` / `CONFLICT_DETECTED` / `CANONICAL_PRIORITY`). |

## IR / SLM layer
| File | Role |
|------|------|
| `slm_layer.py` | **L3.** `run_slm_layer()` — NL → **UUID-only IR JSON** via the local SLM. `_validate_ir` / `_prune_hallucinated_uuids` / `_normalize_ir` guards. Also hosts `run_decomposer()` (**dead** — `QUERY_DECOMPOSE_ENABLED=False`). Delegates to LangGraph when `USE_LANGGRAPH=true` (the default); the ~400-line body below is the fallback. |
| `slm_langgraph.py` | The LangGraph IR pipeline — the default path. `run_langgraph_pipeline()`. |
| `lg_nodes.py` / `lg_prompts.py` | LangGraph node functions + per-node prompts. |
| `envelope_slm.py` / `intent_envelope.py` | Tier-2 "envelope path" — one JSON-constrained call → frozen intent envelope → deterministic `map_envelope_to_intent`. |
| `intent.py` | Typed `QueryIntent` + `validate_intent()` (grounding firewall) + `build_sql()` (the deterministic SQL path from a validated intent). |
| `semantic_layer.py` | **L2** ensemble retrieval / RRF (`JoinEdge`, domain synonyms). Feeds `slm_layer`. |
| `retrieval_select.py` | `select_retrieval()` — single source of truth for which columns/tables/join_path reach L3. The legacy MiniLM/RELGT semantic-layer signal was removed (always returned empty). |
| `retrieval_v2.py` | V2 retrieval: bi-encoder + cross-encoder rerank + bidirectional merge. `RETRIEVAL_V2_ENABLED` (default true). |
| `reranker.py` | Cross-encoder reranker: `rerank_columns` / `rerank_tables` / `rerank_chunks`. Precomputed pair text (`_get_rerank_docs`). |
| `schema_linker.py` | `run_schema_linker()` — spaCy + deterministic schema linking when the query names its objects. |
| `nl_simplifier.py` | Pre-L1 verbose-NL rewrite using sampled value hints. **Not on the hot path** (`NL_SIMPLIFIER_ENABLED=False`). |
| `temporal_parser.py` | **L1** `run_temporal_parser()` → `TemporalFilter{start,end}`. Deterministic. |

## SQL generation / resolvers (called by `../veda/pipeline.py`)
| File | Role |
|------|------|
| `sql_builder.py` | **L4/L5** `run_sql_builder()` — IR JSON → SQL. `_pick_best_temporal`. |
| `value_arbiter.py` | `arbitrate(query, value_lookup)` — classify spans SCHEMA_REF / VALUE / NEGATED_VALUE / ENTITY / UNKNOWN, grounded only by `column_values`. |
| `value_resolver.py` | `resolve_value_filter(...)` — generic 1-hop FK value filter, no hardcoded vocab. |
| `fk_path_resolver.py` | `resolve_fk_path()` — N-hop junction-aware value resolution, refuse-on-ambiguity. |
| `value_filter.py` | Value-aware filter-column retrieval (columns whose sampled values contain a query token). |
| `answer_entity.py` | "who" queries → project a person's display column over the FK. (Still holds a direct-Ollama call.) |
| `entity_resolver.py` | Entity Resolution V1 — flag-gated. |
| `resolution.py` | QSR (Query Semantic Resolution) — one place query tokens resolve to typed schema referents. |
| `target_selection.py` | Stage-1 evidence-based target selection (`select_targets`). |
| `join_planner.py` / `semi_join_planner.py` | Deterministic runtime join planner + bounded cross-source SEMI_JOIN strategy. |
| `superlative_plan.py` / `ratio_plan.py` / `ranking_parser.py` | Deterministic superlative / ratio planners + shared top-N extraction. |
| `fast_path.py` | Deterministic fast paths (count / aggregate / dimension-list) — short-circuit before L2, no LLM. |
| `runtime_context.py` | Layer-0 deterministic answers for pure system-value questions. |

## Non-SQL heads + fusion
| File | Role |
|------|------|
| `rag_layer.py` | `run_rag_layer()` (doc synthesis) + `run_hybrid_layer()` (SQL rows ⊕ doc chunks, one SLM call). **BGE-M3** encode (not MiniLM). Value-expansion for RAG was removed (stale `stats["value_expanded"]`). Still holds a direct-Ollama call. |
| `nosql_builder.py` | `run_nosql_builder()` — deterministic NL → native Mongo/ES/DynamoDB query dict. NO LLM. The `ir_json` path is future/unused. |
| `graph_retriever.py` | `run_graph_retrieval()` — **Personalized PageRank** walk over the unified graph (Tier-2 / datalake / cross-source). The `# BFS expansion` comment is stale. |
| `federated_route.py` / `federated_executor.py` / `cross_source_composer.py` / `cross_source_guard.py` | Cross-source federated NL route (scope ≥ 2 sources, hits > 1 → generate + run a federated DuckDB query), aggregate-then-join execution, hybrid answer composition, and a grounding guard blocking numbers in neither source. |
| `doc_data_planner.py` / `datalake_values.py` | Bounded DOCUMENT_FACT ∩ DATA_GROUNDING; query-time datalake parquet value grounding. |

## Result / answer
| File | Role |
|------|------|
| `multi_result.py` | `MultiResult` / `SubResult` envelope + `STATUS_OK/REFUSED/ERROR`. Compound-query collector (**dormant** with decompose off). |
| `nl_answer.py` | **L7b** `run_nl_answer()` — rows → one-line prose (`NL_ANSWER_ENABLED` true). |
| `result_explainer.py` | Result explanation + `blend_patterns()` (folds analytics patterns into the answer). |

## Gotchas
- The **real front door is `../veda_hybrid.classify()`**, not `query_router.py` — two
  doc-intent overrides run before the router, and `DOC_INTENT_EVIDENCE_ENABLED` defaults ON.
- **Decomposition is dead** in prod (`QUERY_DECOMPOSE_ENABLED=False`): `run_decomposer`,
  `_maybe_split`, `_fan_out`, `multi_result`'s compound path are unreachable.
- **`run_hybrid_layer` is always called with `sql_columns=[]`** — the RRF `w_sql` branch is
  inert; SQL evidence enters as executed rows injected as text.
- The IR/SLM seam emits a **full UUID-only IR envelope**, never SELECT/WHERE-only SQL.
- `contracts/HEADS.md` + `README.md` are the two most stale contract docs (embedding
  router fallback, RAG "MiniLM", hybrid consuming ranked columns — all wrong now).
