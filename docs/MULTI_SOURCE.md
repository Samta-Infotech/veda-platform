# MULTI_SOURCE — routing a question to the right source(s)

How a natural-language question, arriving with a **scope** (the source ids the asker may
see), reaches the correct data source, federates across several, or is refused. Expands
[`ARCHITECTURE.md`](ARCHITECTURE.md) §8. Operator runbook: [`MULTI_SOURCE_DEPLOYMENT.md`](MULTI_SOURCE_DEPLOYMENT.md).

> **Authority.** `veda_core/query/` is authoritative for routing behavior; `apps/query/scope.py`
> for scope resolution; `apps/sources/models.py` for the routing catalog. Where the code is a
> shadow, a skeleton, or dead, it is called out.

---

## 0. The three routing surfaces

They are independent and answer different questions. Only two are live.

| Surface | Question it answers | File(s) | State |
|---|---|---|---|
| **Keyword router** | Which *modality* — SQL / RAG / hybrid / NoSQL — within a scope? | `query/query_router.py`, invoked from `veda_hybrid.classify()` | **live** |
| **Multi-source coordinator** | Which *source(s)* answer this, and how are their results combined? | `query/source_coordinator.py` + `routing_policy.py` + `routing_slm.py` + `source_evidence.py` + `operation_classifier.py` + `execution_planner.py` + `agents.py` + `result_orchestrator.py` + `reliability.py` | **shadow only** — `MULTISOURCE_ROUTING_SHADOW=1` (`config.py:982-983`): computes + traces a `RoutingDecision`, returns `None`, the legacy path answers |
| **Federated route** | Scope spans ≥2 sources and the question genuinely crosses them — generate + run one cross-source query | `query/federated_route.py` + `federated_executor.py` + `cross_source_composer.py` + `cross_source_guard.py` | **live** — fires when the ambient context carries ≥2 `source_ids` |

### Front-door order (`veda_hybrid.run_hybrid_query` → `_l0_dispatch`)

```
run_hybrid_query(query, scope)
  ├─ L0 runtime_context      (pure system value: "current date") ─────────► answer
  ├─ _run_coordinator        MULTISOURCE_ROUTING_ENABLED=1 & SHADOW=1 ────► trace only, returns None
  ├─ _maybe_federated        ambient ctx has ≥2 source_ids ───────────────► federated answer / refusal
  ├─ QUERY_DECOMPOSE_ENABLED=False ──────────────────────────────────────► _dispatch_single(query)   ◄── PRODUCTION PATH
  └─ _dispatch_single(query):
       classify(query) ─► sql | rag | hybrid | nosql
       then head dispatch (see ARCHITECTURE.md §3.4)
```

`_run_coordinator` (`veda_hybrid.py:602`): `On + SHADOW` → traces a decision, `return None`
(`veda_hybrid.py:728-729`); `On + not SHADOW` → the decision drives the answer
(`NO_MATCH` → refuse, `ROUTED/SINGLE` → source agent, `ROUTED/MULTI` → federate / merge).
`Off` → `return None` (`:618`).

---

## 1. Keyword router — modality within a scope (`query/query_router.py`)

`route_query(query, available_sources=None) -> RouteResult{intent, source_ids, confidence, reason}`
(`query_router.py:134`). No model inference; **no embedding fallback** despite the module
docstring — `QUERY_ROUTER_CONFIDENCE_THRESHOLD` / `QUERY_ROUTER_INTENTS` are imported
(`query_router.py:24-26`) and never used.

- `QUERY_ROUTER_ENABLED=False` → hard `intent="sql"`, primary relational source, `confidence=1.0`.
- **No document/nosql source in scope** (`query_router.py:194-200`) → `intent="sql"`,
  `source_ids=sql_ids`, `confidence=1.0`. **This is the common single-DB case — the router
  effectively never runs its keyword logic.**
- Keyword scoring: 4 hand-maintained sets (`_SQL_KEYWORDS` ~40, `_RAG_KEYWORDS` ~30,
  `_TEMPORAL_KEYWORDS` ~25 double-counted into `sql_hits`, `_NOSQL_KEYWORDS` ~12). Matching is
  **substring, not tokenized** — `"per"` matches `"performance"`. Value-filter discount: a query
  token matching a sampled DB column value knocks ~40% off `rag_hits` (`query_router.py:207`).
- Decision cascade: NoSQL (`nosql_score>0.4`) → Hybrid (`sql_hits≥1 and rag_hits≥1 and document_ids`)
  → RAG (`rag_score>sql_score and document_ids`) → SQL (default).

Two doc-intent overrides in `veda_hybrid.classify()` run **before** `route_query` and beat it
in a doc-bearing scope:
1. **Deterministic** (`veda_hybrid.py:252-263`): fixed regex word list (`document`/`policy`/
   `contract`/`clause`/…) ∧ `_scope_has_doc_source()` (a `graph_nodes WHERE node_type='chunk'`
   probe) → `rag`, or `hybrid` if the query also has an aggregation verb.
2. **Evidence-based** (`veda_hybrid.py:265-271`, `DOC_INTENT_EVIDENCE_ENABLED` env default `"1"`):
   reuses the coordinator's cosine evidence; a chunk-backed dominant STRONG source whose chunk
   cosine ≥ its own best column cosine → same `rag`/`hybrid` split.

---

## 2. Multi-source coordinator — which source(s) (shadow only)

Runs on every query (`MULTISOURCE_ROUTING_ENABLED=1`), traces a `RoutingDecision`, then
`return None` — **the legacy federated/single path produces the answer**. Turning
`MULTISOURCE_ROUTING_SHADOW=0` makes it authoritative; `MULTI_SOURCE_DEPLOYMENT.md` §3 warns
the code default (`1`) is the wrong value for a real multi-source deployment.

### Internal pipeline (`source_coordinator.plan_route` / `execute_decision`)

| Step | Component | What it does |
|---|---|---|
| 1 | `source_evidence.group_evidence_by_source(cols, chunks)` | bucket retrieval output (columns + doc chunks) per source into `SourceEvidence` |
| 2 | item prior + dominance retier | `_apply_item_prior` (SourceItem routing embedding), `_dominance_retier` — promote/demote candidate tiers |
| 3 | `routing_policy.decide(candidates, fk_edges)` | **pure deterministic** decision over presence tier + `cross_source_fk` edges + `is_canonical` → `SINGLE` / `MULTI` / `AMBIGUOUS` / `NO_MATCH` |
| 4 | `routing_slm.resolve_boundary()` | **only on `AMBIGUOUS`** — one bounded SLM call picks the SINGLE/MULTI boundary; output is validated against the candidate set |
| 5 | `operation_classifier` | for a `MULTI` query, a bounded closed-enum cross-source OPERATION (or `UNSUPPORTED`) |
| 6 | `execution_planner` | `RoutingDecision` → `ExecutionPlan` with strategy `single` / `federated` / `independent`. Staged A→B **DEPENDENT is explicitly unsupported.** |
| 7 | `agents.py` | one thin agent per source **kind** (`_AGENT_BY_KIND`): `relational`→`DatabaseAgent`, `datalake`→`DataLakeAgent`, `document`→`FileSystemAgent`, `nosql`→`NoSqlAgent`. Each runs that source's existing pipeline, normalizes to `AgentResult`. |
| 8 | `result_orchestrator` | merges `independent` `AgentResult`s under `APPEND` / `CONFLICT_DETECTED` / `CANONICAL_PRIORITY` |
| 9 | `reliability.py` | bounded transient-retry wrapper + `classify_failure(err) -> transient/permanent` around source agents |

`source_coordinator` holds `_ROUTING_QV` — an embed-once ContextVar so the routing-stage
query embedding is reused downstream.

**Gotcha — agent registry is keyed by *kind*, not dialect or connector type.** Passing a raw
connector string (`csv_lake`, `parquet`, `filesystem`) as a source's `source_type` →
`resolve_agent()` returns `None` → `dispatch()` returns `None` → fall through to the legacy
single-source path. The production path is correct (`scope.source_profiles_for` calls
`Source.source_kind()`); this only bites hand-built profiles — see the `doc_bench`/`datalake_bench`
antipattern in [`EVALUATION.md`](EVALUATION.md) and `MULTI_SOURCE_DEPLOYMENT.md` §1.

---

## 3. Federated route — one query across ≥2 sources (`query/federated_route.py`)

`_maybe_federated(query, strict=False)` (`veda_hybrid.py:830`):

- reads the ambient `RequestContext.source_ids`; **`len(sids) < 2` → `return None`** (normal path).
- `run_federated(query, tenant, source_ids)` — if the question does not actually span sources it
  returns `None` → single-source plan → normal path. So the effective trigger is "scope spans
  ≥2 sources **and** retrieval/planning finds a real cross-source join".
- wrapped in `execute_federated_reliably` (bounded transient retry, flag-gated, default a
  single pass-through).
- `strict=True` (set by the coordinator when a `cross_source_fk` edge was DETERMINED — a genuine
  required join): a federation failure is **surfaced** with the involved sources + failure class,
  never silently degraded to a single-source answer that would drop a source.

### Execution

- **`federated_executor.FederatedExecutor.execute_plan`** — **aggregate-then-join**: each metric
  is aggregated per-source first, then the per-source results are joined (never one flat
  cross-database SELECT). Two success shapes: `compose_federated()` flat single-SELECT (`sql`
  key), and the **preferred** `compose_federated_plan()` structured path
  (`plan: {group_by, metrics: [{alias, sql}]}`, each metric aggregated+joined independently).
- **DuckDB** is the federation engine: `src_N` attachments over materialized parquet
  (`data/<source_id>/tables/*.parquet` from L1 `materialize_parquet`, tabular sources) + live
  relational sources.
- **`cross_source_composer.py`** — hybrid cross-source answer composition (structured plan ⊕
  free-form per-metric).
- **`cross_source_guard.py`** — grounding guard on the *synthesized* cross-source answer: blocks
  a number that appears in neither source's rows.

A federated answer returns a `federated` `SubResult`; `build_explain` parses the actual
generated SQL text (`sm=None` degrades labels to humanized names, never crashes) so it still
tables / charts / explains.

---

## 4. Cross-source links — built at ingestion

Routing and federation both rest on links discovered offline. Full mechanics:
[`INGESTION.md`](INGESTION.md).

### Structured ↔ structured (`cross_source_fk` edges)

```
L2  column_sketches.run_sketch_pass       128-perm MinHash over distinct values of
    (per source)                          join-key-shaped columns → column_sketches table
        │
L5  cross_source_graph.discover_and_persist   tenant-wide, EVERY ingest: compare column_sketches
    (tenant-wide)                              across sources (Jaccard + containment) →
        │                                      cross_source_fk col→col edges (HIGH/MED tiers)
        ▼
    graph_edges (type='cross_source_fk')  → federated join planner + routing_policy
```
No-op until ≥2 sources in the tenant have sketches. Backfill without re-ingest:
`scripts/backfill_cross_source.py`.

### Document ↔ structured

- **`entity_linker.link_entities`** (document ingest) — dictionary + pattern + optional-SLM
  entity detection → `entity` nodes bridging `chunk --mentions_entity--> entity --value_of--> column`.
  Replaces the dead `chunk_linker.py`.
- **Semantic bridge (`semantic_linker.py`, Tier A)** — matches chunk M3 vectors against column
  `graph_node_embeddings` → `semantic_about` (chunk→column) edges. **Chunk→column only; never
  authorizes a join.** Traversed by the PPR walk in `query/graph_retriever.py`.
- **Tier B (`value_embedder.py`)** — embeds eligible sampled DISPLAY values → `entity_value_embeddings`
  (HNSW), read back by `semantic_linker` for doc-side span matching. **Runs only in
  `source_dispatcher` step 7d — NOT in the layered L1–L5 relational path**, so for a normal
  relational ingest `entity_value_embeddings` is empty.

---

## 5. The routing catalog — `Source` model fields

`apps/sources/models.py` — `Source` is **global, not tenant-scoped**.

| Field | Role in routing |
|---|---|
| `dialect` (16 values) | the **only** input to `Source.source_kind()` via `_DIALECT_TO_ENGINE`. `postgres/mysql/sqlite/oracle/sqlserver/duckdb→relational`, `mongo/es/dynamo→nosql`, `filesystem/s3_docs→document`, `delta/parquet/csv_lake/iceberg→datalake`. An unknown dialect silently defaults to `("relational","generic")`. |
| `domain_tags` | manual tags, scored by the coordinator's evidence prior |
| `description` | **manual wins**; auto-generated post-ingest when blank (`SOURCE_PROFILER_ENABLED`) |
| `description_generated` | provenance flag (was `description` auto-written?) |
| `is_canonical` | manual tie-break — `result_orchestrator` `CANONICAL_PRIORITY` and `routing_policy` prefer the canonical source on a tie |
| `ready` | query path reads **only** `ready=True`; flipped by ingestion success only |

### SourceItem layer

`SourceItem` / `SourceItemType` (`models.py:226-274`) — uniform per-item routing metadata
(one row per table / dataset / document / collection), global, unique `(source, item_type,
item_key)`. The routing-prior embedding is kept **engine-side** (not a pgvector column here, so
the model migrates on sqlite). Built/backfilled by
`apps/sources/management/commands/{build_source_items,backfill_source_items,profile_source_items}.py`,
gated by `SOURCE_ITEM_PROFILER_ENABLED`. Query-time: `scope.source_profiles_for` folds
SourceItem data into the per-source profile.

### The profilers

`source_profiler.py` (source-level `description`) and `item_profiler.py` (per-item) — flag-gated
post-ingest steps (`tasks.py:295-312`, never-raise). `SOURCE_PROFILER_ENABLED` /
`SOURCE_ITEM_PROFILER_ENABLED`.

---

## 6. Scope resolution — which sources the asker may see

`apps/query/scope.py`, called by `QueryView` and `ConversationQueryView` (imported directly).

```
resolve_query_scope(data, tenant, user, effective_permissions)
   = permitted_source_ids(user, eff)          RBAC source-level check
        ∩  _ready_source_ids()                Source.objects.filter(ready=True)
        ∩  request pin (data["source_id"] / data["source_ids"])   if present
   → ordered list, primary first;  source_id = source_ids[0]
```

- `permitted_source_ids` returns `None` (no narrowing — RBAC off / staff), a `set` (narrowed),
  or `set()` (deny → `SourceAccessDenied` → **403, no `QueryLog` row**).
- No ready source → `NoReadySource` → **503, no `QueryLog` row**.
- An unreadable registry degrades to `[]` → request pin / `VEDA_DEFAULT_SOURCE_ID` fallback.
- `source_profiles_for(source_ids)` builds the per-source profile dict; best-effort `{}`.
- `compute_data_scope` (table/column allow payload) → `serialize_data_scope` → `X-Veda-Data-Scope`
  header → engine `rbac_filter`. All identity no-ops while `VEDA_RBAC_MODE="off"`.

The scope + profiles cross api→inference as headers (`X-Veda-Source-Ids`,
`X-Veda-Data-Scope`, `X-Veda-Source-Profiles`); the inference `_tenant_context` middleware
parses them onto `RequestContext` + `_source_profiles` ContextVar. **The streaming route must
re-capture `copy_context()` or the profiles/scope reach only the non-streaming endpoint** —
see `MULTI_SOURCE_DEPLOYMENT.md` §4.

---

## 7. Status — wired vs shadow vs missing

**Wired:** keyword router + both doc-intent overrides; per-kind agents; federated route
(`_maybe_federated` + `run_federated` + aggregate-then-join executor + composer + grounding
guard); `column_sketches` → `cross_source_graph` → `cross_source_fk`; `entity_linker`;
semantic bridge Tier A; scope resolution chain; SourceItem layer + profilers (flag-gated).

**Shadow:** the entire multi-source coordinator (`MULTISOURCE_ROUTING_SHADOW=1`) — computes and
traces `RoutingDecision`s, does not act on them. `routing_slm.resolve_boundary`,
`routing_policy.decide`, `operation_classifier`, `execution_planner`, `result_orchestrator` all
run but their output is discarded.

**Not the real path / fragile:**
- `query_router` embedding fallback — never implemented; dead config.
- `nosql_builder` IR-JSON `filter_tree` path — labelled "Phase 5+", callers pass only `query`.
- Tier B `value_embedder` — not invoked by the layered relational pipeline.
- `doc_bench.py` / `datalake_bench.py` hand-build source profiles with connector-type strings
  instead of kinds — the vocabulary trap that invalidated 3 benchmark runs
  (`MULTI_SOURCE_DEPLOYMENT.md` §1). Treat their committed results with suspicion.
- `docs/multisource_routing/` (cited ~10× in `config.py`) does not exist; the flag comments
  are the surviving record.
