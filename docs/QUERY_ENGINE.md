# QUERY_ENGINE — the deterministic SQL head

Deep reference for the NL→SQL correctness head: `veda_core/veda_hybrid.py` (front
door + dispatch + Tier‑2) and `veda_core/veda/` (the L1–L7 pipeline + firewall).
Expands [ARCHITECTURE.md](ARCHITECTURE.md) §4. The non‑SQL heads (rag / hybrid /
nosql), the keyword router internals, and the IR/SLM seam are covered in
[ARCHITECTURE.md](ARCHITECTURE.md) §3 and the layer contracts
(`veda_core/query/contracts/`); this doc treats them only where the SQL head calls
them.

> **Authority.** `veda_hybrid.py` + `veda/pipeline.py` are authoritative for engine
> behavior; `veda_core/config.py` is the single source of truth for flags. Written
> from a direct read of the source on 2026‑09‑09 (`master`, tree clean). Line
> numbers drift — treat them as anchors, not addresses.

Path prefix for citations: `veda_core/`.

---

## 1. The front door — `run_hybrid_query`

`run_hybrid_query(query, verbose, on_event, trace_id)` (`veda_hybrid.py:1039`) is
the single public entry. It mints/reuses the **one** `ExplainTrace` for the whole
request, binds it as the ambient trace, runs the inner flow, and finalizes +
persists the trace exactly once. It **always returns a `MultiResult`** (a list of
`SubResult`).

```
run_hybrid_query                                            veda_hybrid.py:1039
  tr = new_trace(query, trace_id)                            :1050
  with use_trace(tr):                                        :1051
    result = _run_hybrid_query_inner(query, …)               :1054
    result.trace_id = tr.trace_id                            :1065
    _clean_refuse_on_empty_error(result)                     :1068
  finally: tr.finalize(_final_status)                        :1072   persist once
```

`_clean_refuse_on_empty_error` (`:982`, `WEAK_EVIDENCE_CLEAN_REFUSE_ENABLED`
default **ON**): a terminal `error` item with **no answer and no SQL** is rewritten
to a `refused` item — `"engine_unavailable"` + retry copy when
`query.reliability.classify_failure` says the failure is *transient*, else
`"no_relevant_data"` + "couldn't find any data relevant to this question".

### 1.1 `_run_hybrid_query_inner` — the pre‑dispatch layers

`_run_hybrid_query_inner` (`veda_hybrid.py:1077`) runs four layers **before**
`_dispatch_single` is ever reached. A query can be answered or refused at any of
them.

| # | Layer | Entry | Behavior at current defaults |
|---|-------|-------|------------------------------|
| 1 | **L0 NL simplifier** | `:1104` | `NL_SIMPLIFIER_ENABLED = False` → skipped. If on, `run_nl_simplifier` may replace `query`; usage folded back via `_merge_l0_usage`. |
| 2 | **L0 runtime context** | `:1133` | `RUNTIME_CONTEXT_ENABLED = True`. `answer_runtime_context(query)` — a pure system‑value question ("what's the current date") returns **immediately** as route `runtime_context`, before any retrieval. |
| 3 | **Multi‑source routing coordinator** | `_run_coordinator` `:602`, called `:1152` | `MULTISOURCE_ROUTING_ENABLED = 1` **and** `MULTISOURCE_ROUTING_SHADOW = 1` → the coordinator computes a `RoutingDecision`, records it to `trace.routing`, and **returns `None`** (observe‑only; the answer path is byte‑identical). Only with `SHADOW = 0` does the decision drive the answer: `NO_MATCH` / `CLARIFICATION_REQUIRED` → refusal `MultiResult`; `ROUTED/SINGLE` → source‑agent dispatch; `ROUTED/MULTI` → doc+data grounding / federated / independent‑merge. Also holds the flag‑gated `ROUTING_PERMISSION_PRECHECK_ENABLED` (default off) that can return a `no_access` refusal. |
| 4 | **Cross‑source federated route** | `_maybe_federated` `:830`, called `:1159` | No‑op unless the ambient `RequestContext` carries **≥ 2 `source_ids`** (`:840`). Then `run_federated` (+ bounded transient retry `execute_federated_reliably`); on `ok` it builds a `federated` `SubResult` with real cross‑source SQL, `build_explain`, and deterministic analytics. A planning/binder failure **degrades** to the normal path (unless `strict=`); a refused/blocked federation is surfaced. |
| 5 | **Decomposition** | `:1163` | `QUERY_DECOMPOSE_ENABLED = False` (config comment: "splits join queries wrongly") → straight to `_dispatch_single(query)` wrapped in a 1‑item `MultiResult` (`:1169`). When on: `classify` → if intent `sql`, run the head as a **probe**; `ok` / `clarify` → return it; else `run_decomposer` → `should_split` → `_fan_out`; `DECOMP_DEPENDENT` → **refuse** with ordered‑part guidance. |

```
run_hybrid_query
  └─ new_trace / use_trace                       (ambient trace for the whole request)
  └─ _run_hybrid_query_inner
       1. NL simplifier          [OFF]                       → (rewrites query)
       2. answer_runtime_context [ON]                        → EXIT MultiResult(runtime_context)
       3. _run_coordinator       [ENABLED=1, SHADOW=1]       → trace only, return None
                                 (authoritative: NO_MATCH→EXIT refuse; SINGLE→source agent;
                                  MULTI→doc_data / federated / merge)
       4. _maybe_federated       [len(ctx.source_ids) >= 2]  → EXIT federated MultiResult
       5. QUERY_DECOMPOSE_ENABLED? ── off ──▶ _dispatch_single(query) ──▶ MultiResult(1 item)   ◄─ PRODUCTION PATH
                                  ── on  ──▶ classify → probe head → run_decomposer
                                              ├ independent ─▶ _fan_out ─▶ MultiResult
                                              ├ dependent   ─▶ guided refusal
                                              └ single      ─▶ _dispatch_single
  └─ _clean_refuse_on_empty_error   (empty error item → engine_unavailable | no_relevant_data)
  └─ tr.finalize(status)            (persist one trace record)
```

---

## 2. `classify` — intent routing

`classify(query)` (`veda_hybrid.py:249`) picks a modality (`sql` / `rag` /
`hybrid` / `nosql`). Two doc‑intent overrides run **before** the keyword router.

1. **Deterministic doc‑intent override** (`:256`): `_DOC_REF_RE` (a fixed word
   list — document / agreement / contract / policy / clause / …) matches **and**
   `_scope_has_doc_source()` (a `graph_nodes WHERE node_type='chunk'` existence
   probe over the scope) → return `hybrid` if `_DB_AGG_RE` also matches, else
   `rag`; `source_ids = None`.
2. **Evidence‑based doc‑intent** (`_doc_intent_by_evidence` `:216`):
   `DOC_INTENT_EVIDENCE_ENABLED` **defaults to `"1"` (ON)** (`config.py:498`).
   Reuses the source‑coordinator's cosine evidence + dominance re‑tiering; a
   chunk‑backed *dominant* source whose chunk cosine ≥ its own best column cosine
   → `rag` / `hybrid`.
3. `QUERY_ROUTER_ENABLED = True` (`config.py:489`) → `query.query_router.route_query`.
   Any exception → `("sql", None)` (safe default).

`route_query` (`query/query_router.py:134`) is a **pure keyword counter** —
`_SQL_KEYWORDS` (+ `_TEMPORAL_KEYWORDS` ×2), `_RAG_KEYWORDS` (discounted ~40% when
a query token matches a sampled DB value), `_NOSQL_KEYWORDS`. It has **no
embedding fallback** — `QUERY_ROUTER_CONFIDENCE_THRESHOLD` and
`QUERY_ROUTER_INTENTS` are imported and never used (contra the module docstring
and `contracts/HEADS.md`). **With no document/nosql sources configured it returns
`sql` unconditionally** (`query_router.py:194`), so in the single‑relational‑source
deployment `classify` is effectively always `sql`.

---

## 3. `_dispatch_single` — per‑modality head

`_dispatch_single` (`veda_hybrid.py:1358`): `intent, source_ids = classify(query)`
(`:1362`), then branch.

| intent | Entry | Notes |
|--------|-------|-------|
| `sql` | `:1367` | `_load_semantic_model()` → `run_query(query, sm, cols, return_result=True)` (`:1402`). **RBAC is not applied to this sm** — it is the same object handed to `get_engine` (`veda_hybrid.py:159`). `sm` narrowed to zero tables → try `run_rag_layer` if the scope has a doc source and `rag.confidence ≥ 0.35`, else `access_denied` (`:1378`). |
| `rag` | `:1504` | `run_rag_layer(query, source_ids, temporal_filter=_temporal(query))`. |
| `hybrid` | `:1517` | `run_query` first (correct‑by‑construction rows), then feed its **executed rows** (as text) + optional unified‑graph chunks into `run_hybrid_layer` (called with `sql_columns=[]`). The SQL head's cols/rows/explain/analytics are attached to the hybrid result. |
| `nosql` | `:1595` | `_run_nosql` → resolve source → `build_connector` → `filter_nosql_collections` (RBAC) → `run_nosql_builder` (deterministic, no LLM) → `execute_query`. |
| default | `:1602` | `run_query` again (safety net). |

### 3.1 The Tier‑2 gate

After `run_query` returns for the `sql` branch (`veda_hybrid.py:1412`):

Tier‑2 fires **iff** all of:

- `res` is **not** ok, **and**
- `res["status"] ∈ {refuse, qualifier_dropped, ungrounded, no_table, exec_error}`, **and**
- `TIER2_LLM_FALLBACK = True` (`config.py:2219`), **and**
- the head took **≤ `TIER2_SKIP_IF_HEAD_OVER_S` = 120.0 s** (`config.py:1324` — the
  call‑site fallback literal in `veda_hybrid.py` still says `60.0`; config wins).

`clarify`, `invalid`, and `ir_mismatch` are **deliberately not Tier‑2‑retried** — a
`clarify` is a grounded question and an LLM retry could override it; `ir_mismatch`
means the LLM already produced wrong semantics once.

Tier‑2 entry: `_tier2_sql(query, sm, cols, deadline = now + TIER2_TIME_BUDGET_S
(=120), execution_state = res["context"])` (`veda_hybrid.py:1966`). The LLM emits a
**UUID‑only IR envelope** (tables/columns by UUID, `filter_tree`, `joins`,
`aggregations`, `group_by`, `order_by`, `limit`) — **never SQL structure**. A
deterministic builder turns the IR into SQL; when the LLM names ≥ 2 entities,
`LANGGRAPH_SHARED_PLANNER = True` routes join construction through
`veda.planning.build_from_entities` (graph‑verified), never the LLM.

Outcomes:

| Tier‑2 result | Route / status |
|---------------|----------------|
| dict returned | route `deterministic`; status derived from `ok` |
| `tier2_rejected` | a firewall gate declined the LLM SQL → **the head's refusal stands**, note attached |
| `tier2_exec_error` | `STATUS_ERROR` |

`_to_subresult` (`veda_hybrid.py:1627`) maps head status → SubResult status: `ok`
→ `STATUS_OK`; `tier2_exec_error` → `STATUS_ERROR`; every other non‑ok →
`STATUS_REFUSED`. Status is **derived from the head's own success signal, never
invented**.

---

## 4. `run_query` — the escalation ladder

`run_query(query, sm, all_cols, return_result=False, anchor_hint=None, on_event=None)`
(`veda/pipeline.py:155`). `_done(rc, status, **kw)` (`:219`) is the single exit
funnel; with `return_result=True` (every front‑door caller) it returns
`{status, ok, trace, explain, usage, latency_ms, context: ExecutionState, **kw}`.
The `int` return path in the docstring is vestigial (benches only).

### 4.1 Understand

- `run_temporal_parser(query)` → `tf` (`:331`).
- **L4 intent is the literal string `"SIMPLE"`** (`:345`). `IntentDetector` is
  **deliberately not wired** (comment `:340`): its keyword classes overlap the
  grammar planners, and flipping intent to `MULTI_TABLE` / `AGGREGATE` here
  re‑opens multi‑table planning latency that `SUPERLATIVE_JOIN_ROUTING`
  deliberately gates off. Query shape comes only from the grammar classifiers:
  `existence_mode` / `aggregate_mode` / `superlative_mode` / `grouped_mode` /
  `ratio_mode` (`veda/planning.py`). `_sup` flips `intent` to `"AGGREGATE"` **only
  if `SUPERLATIVE_JOIN_ROUTING`** (= False), so it stays `SIMPLE` and merely
  records to the trace.

### 4.2 The ladder — first firewall‑passing answer wins

1. **Fast path** (`FAST_PATH_ENABLED = True`, not existence) — `try_fast_path(query, tf)`
   (`:388`). SQL straight from compiled registries; no retrieval, no `get_engine()`,
   no LLM.
2. **Deterministic superlative planner** — `_sup and fp is None and
   SUPERLATIVE_PLAN_ENABLED` (= True): `try_superlative_plan` (`:401`).
   `("clarify", msg)` → `_done("clarify")`.
3. **Deterministic grouped‑breakdown planner** — `GROUPED_PLAN_ENABLED` (= True):
   `try_grouped_plan` (`:423`).
4. **Deterministic ratio planner** — `RATIO_PLAN_ENABLED` (= True):
   `try_ratio_plan` (`:445`).
5. **Fast‑path evidence guard** (`FASTPATH_EVIDENCE_GUARD = True`) — `:465`. If the
   fp pick's tables get **zero typed evidence** (`typed_anchor_evidence <
   QSR_FP_EVIDENCE_FLOOR`), **demote** `fp = None` (fall through to the full
   pipeline), with grounded‑entity and metric‑pick exemptions.
6. **Verified cache** (skipped when existence or fp) — `verified_cache_lookup(query)`
   (`:512`): exact‑hash short‑circuit → BGE‑M3 cosine **≥ 0.85** (`veda/cache.py:25`).
   Then the same evidence guard on the cached tables (`:516`) **and** a fresh
   `qualifier_completeness` re‑check on the cached SQL (`:543`) — either demotes
   the hit and forces a recompute. (Fixes a prod bug where "properties in the UAE"
   replayed for "properties priced above 10,000".)
7. **Full path** (`else`, `:577`):
   - **L2+ enhance** — `QUERY_ENHANCEMENT_ENABLED = False` → `_search = query`.
   - **L2 retrieve** — `get_engine(sm).retrieve(query=_search, intent="SIMPLE",
     top_k=15, use_cache=RETRIEVAL_CACHE_ENABLED(=False))` (`:598`). Six signals →
     RRF(k=60) → intent boost → adaptive cutoff (see [ARCHITECTURE.md](ARCHITECTURE.md) §5).
   - **L2g graph‑expand booster** — `GRAPH_EXPAND_ENABLED` (= True):
     `graph.query_graph.suggest_expansions` appends columns at `final_score=0.0`.
   - **RBAC candidate filter** — `filter_retrieval_results(results, sm, ctx)` (`:642`).
   - **L2b primary cross‑encoder rerank** — `PRIMARY_RERANK_ENABLED` (= True):
     skips on an unambiguous same‑table RRF gap; else `query.reranker` overwrites
     `final_score` (so anchor selection reads reranked order), with a
     `RERANK_NOISE_FLOOR` guard and a scale guard flooring the un‑reranked tail.
   - **L3 anchor** — `_router_primary = select_primary_table(results, query, sm)` (`:755`).
   - **Query‑Understanding layer** — `QUERY_UNDERSTANDING_ENABLED = False` →
     skipped. `ANALYTICAL_SQL_V2 = False` → `_analytical_sql` stays `None`.
   - **Entity Resolution V1** — `ENTITY_RESOLUTION_V1 = True`: `resolve_entities`.
     `AMBIGUOUS` + `ER_GROUNDED_REFUSAL` (flag off) → clarify; `RESOLVED` +
     `pin_eligible` → **pin** `primary = _er.anchor` (bypass `vet_primary`); else
     `primary = vet_primary(...)`.
   - `vet_primary` (`veda/routing.py:119`, `ANCHOR_VET_ROUTER = True`) — grain‑hint
     override, graph‑driven dimension demotion, `score_anchors`, then IDF /
     value‑match / typed‑anchor re‑ranks (all flag‑gated), then
     `ANCHOR_SINGLE_GATE_ENABLED` clarify for sub‑margin disjoint subjects. Can
     return `{"clarify": …}` → `_done("clarify")`.
   - `anchor_hint` (qualifier‑salvage retry only) forces the primary (`:886`).
   - `no primary` → `_done("no_table")` (`:971`).
   - **Join decision** — `needs_join = intent in {MULTI_TABLE, AGGREGATE} or
     is_existence` (`:981`) → **always False except existence**, UNLESS
     `TYPED_MULTITABLE_ROUTE` (= True) with a join phrase / `_agg` / `_grp` /
     `_rat` (`:994`), or `_er_multi` (≥ 2 distinct resolved entity tables, `:1005`).
     `_analytical_sql` forces `needs_join = False`.
   - **join branch** — `_er_multi` → `build_from_entities` per resolved entity as
     anchor (`:1030`); else `mt = try_multitable(query, results, sm, all_cols, tf,
     primary)` if `needs_join` else `{"action":"fallback"}` (`:1046`).

     | `mt["action"]` | Result |
     |----------------|--------|
     | `clarify` | `_done("clarify")` |
     | `refuse` | `_done("refuse")` |
     | `existence` | deterministic EXISTS / NOT EXISTS SQL, **no LLM** (`:1057`) |
     | `aggregate` | deterministic pre‑aggregation CTE SQL, **no LLM**, fan‑out‑free by construction (`:1071`) |
     | `sql` | planner pins the FROM/JOIN skeleton + `join_constraints` (key pairs, qualified pairs, predicate cols) + `fanout_guard`; `_llm_sql = True`; the LLM fills SELECT/WHERE via `generate_join_sql` (`:1087`) |

   - **single‑table sub‑ladder** (`fallback`/else, `:1114`) — each rung
     deterministic, each setting `_llm_sql = False` when it fires:

     | # | Rung | Flag | Emits |
     |---|------|------|-------|
     | 1 | answer‑entity | `ANSWER_ENTITY_DISCOVERY_ENABLED` (T) | display name over FK, or "X and their handler" projection JOIN |
     | 2 | FK‑value resolution | `FK_VALUE_RESOLUTION_ENABLED` (T) | `value → IN (SELECT … FROM related WHERE …)` |
     | 3 | multi‑hop FK | `MULTIHOP_FK_RESOLUTION_ENABLED` (T) | junction‑membership nested IN‑subquery |
     | 4 | value‑arbiter filter | `VALUE_ARBITER_ENABLED` (T) | categorical `=` / negated `!=` on the anchor |
     | 5 | temporal‑only | — | date window on the canonical temporal column |
     | 6 | ranked‑temporal‑only | — | "latest 10 X" with no real range → ORDER BY tcol + LIMIT N |
     | 7 | **temporal‑refuse** | — | `tf` present + anchor has **no** temporal column → `_done("refuse")` (never invent `created_at`) |
     | 8 | **single‑table LLM** | `SINGLE_TABLE_DETERMINISTIC` (T) tries `_deterministic_single_table_sql` first | `generate_sql(query, primary, allowed_columns, tf, …)`; `_llm_sql = True` |

---

## 5. The firewall

Every gate runs **after the SQL string exists** and **on the ORIGINAL `query`** —
`pipeline.py:1563` is a literal `assert` that `query` still equals
`trace.query_understanding.query`. Enhancement (`_search`) only ever reaches
`get_engine().retrieve` and the cross‑encoder rerank text.

| # | Gate | Module / entry | Fail status | Flag (default) | Runs on |
|---|------|----------------|-------------|----------------|---------|
| 1 | **Value grounding (L6a)** | `validation.value_grounding` (`pipeline.py:1551`) | `ungrounded` | always on | all SQL |
| 2 | **Qualifier completeness (L6b)** | `validation.qualifier_completeness` (`:1568`) | `qualifier_dropped` / `access_denied` / `clarify` | always on (`FEEDBACK_ENABLED` gates only the copy) | all SQL |
| 3 | **Grouped‑shape guard** | `validation.grouped_shape_ok` (`:1696`) | `clarify` | `GROUPED_SHAPE_GUARD_ENABLED` env `"1"` (ON) | `_llm_sql` only |
| 4 | **Distinct‑shape guard** | `validation.distinct_shape_ok` (`:1707`) | `clarify` | `DISTINCT_SHAPE_GUARD_ENABLED` env `"1"` (ON) | `_llm_sql` only |
| 5 | **Intent↔SQL referent alignment** | `intent_sql_alignment.alignment_ok` (temporal + entity‑anchor) (`:1722`) | `clarify` | `INTENT_SQL_ALIGNMENT_ENABLED` env `"1"` (ON) | all produced SQL |
| 6 | **Aggregate‑presence guard** | `intent_sql_alignment.aggregate_presence_ok` (`:1734`) | `clarify` | `INTENT_SQL_AGG_PRESENCE_ENABLED` env `"1"` (ON) | all produced SQL |
| 7 | **Filter‑presence guard** | `intent_sql_alignment.filter_presence_ok` (`:1751`) | `clarify` | `INTENT_SQL_FILTER_PRESENCE_ENABLED` via `getattr(cfg, …, True)` (ON — **no such symbol in config.py**) | all produced SQL |
| 8 | **Dimension referent alignment** | `intent_sql_alignment.dimension_alignment` (`:1768`) | `clarify` (`DIM_REFUSE` / `DIM_CLARIFY`) | `INTENT_SQL_DIMENSION_ALIGNMENT_ENABLED` env `"1"` (ON) | SQL with a GROUP BY + a "by <dim>" phrase |
| 9 | **Canonical‑intent shadow** | `canonical_intent_shadow.record_shadow` (`:1783`) | — (observe only) | `CANONICAL_INTENT_SHADOW_ENABLED` (**OFF**) | `fp is None and sql` |
| 10 | **IR equivalence (L6b+)** | `ir_equivalence.validate_ir_equivalence` (`:1794`) | `ir_mismatch` | `IR_EQUIVALENCE_ENABLED` (= True) | **`_llm_generated` SQL only** |
| 11 | **Analytical semantics (advisory)** | `semantic_validation.validate_analytical_semantics` (`:1820`) | — (records `trace.semantic_validation`; **does not block** in Tier‑1) | `SEMANTIC_VALIDATION_ENABLED` (= True); `SEMANTIC_VALIDATION_ENFORCE` (= False) | all SQL, `graph=None` |
| 12 | **RBAC final gate** | `rbac_filter.narrow_allowed` (`:1845`) | — (narrows the allow‑list; a later hit shows as `invalid` / `access_denied`) | no‑op when `ctx.allowed_resources is None` | tables/cols |
| 13 | **AST validate + parameterize (L6c)** | `validation.validate_and_parameterize` + `graph_guard` (`:1860`) | `invalid` (or `access_denied` if a restricted name is referenced) | always on; graph guard behind `GRAPH_GUARD_ENABLED` (= True) | the built SQL |
| 14 | **Execute (L7)** | `execution.execute_sql` (`:1897`) | `exec_error`; `PARAM_MISMATCH_ERROR` for placeholder/param count mismatch | always on | `param_sql` |
| 15 | **NL‑back answer (L7b)** | `pipeline.py:1911`, `query/result_explainer.py` | — (never blocks; row‑count fallback) | `NL_ANSWER_ENABLED` (= True) | rows |
| 16 | **Cache‑back** | `cache.save_verified_query` (`:2078`) | — | — | only when `not from_cache and fp is None and rows and not is_temporal and not is_existence` |

**Gate detail worth knowing:**

- **Qualifier salvage** (gate 2). `QUALIFIER_SALVAGE_ENABLED` (env `"1"`),
  `QUALIFIER_REANCHOR_RETRY = True`: if the missing token's QSR referent table is
  outside the SQL, `run_query` retries **once** with `anchor_hint = that table`
  (≤ `QUALIFIER_REANCHOR_MAX_HEAD_S = 45 s`). Then, in order: a grounded‑clarify
  upgrade (real FK label domains), a referent clarify, an RBAC `access_denied` if
  a restricted table is in the SQL, then `qualifier_dropped`. `strict=True` only
  in the Tier‑2 lane.
- **AST validate** (gate 13). Single read‑only SELECT (reject any DML/DDL node);
  every table exists (CTE aliases allowed); every column exists (SELECT aliases
  allowed); `join_constraints` ON‑integrity; **graph_guard** —
  `verify_joins_against_graph` (every base‑table join key a real FK edge) +
  `check_connectivity` (one connected component, no cartesian); **fan‑out guard**
  (reject `COUNT/SUM/AVG` over a parent‑side column across a 1:N / N:M join unless
  `DISTINCT`); parameterize every literal except `LIMIT` / `INTERVAL` via ordered
  sentinels; `identify=True` quoting; append `LIMIT 100` if absent.
- **`access_denied` reclassification** (gates 12–13). `_restricted_for_sql` is
  captured **before** `narrow_allowed` narrows, so a later `validate_and_parameterize`
  rejection is reclassified `access_denied` if `_sql_references(sql,
  restricted_name)` (quoted **or** bare — the verified‑cache replay path is
  unquoted).

```
                 ┌──────────── run_query ladder (first firewall-passing answer wins) ─────────┐
  L1 temporal_parser ─▶ tf
  L4 intent = "SIMPLE"                       (IntentDetector NOT used)
  grammar: existence / aggregate / superlative / grouped / ratio

  fp ← try_fast_path                                          (FAST_PATH_ENABLED)
  fp ← try_superlative_plan / try_grouped_plan / try_ratio_plan   (if fp is None)
  FASTPATH_EVIDENCE_GUARD: fp with zero typed evidence → demote (fp=None)
  cached ← verified_cache_lookup (cosine ≥ 0.85)  [skip if existence or fp]
           + evidence-guard demote + qualifier re-check demote
  ┌── fp ─────────▶ sql = fp.sql            (no retrieval, no LLM)
  ├── cached ─────▶ sql = cached_sql        (replay)
  └── full path:
        retrieve(top_k=15) → 6 signals → RRF(k=60) → intent boost → adaptive cutoff
        + L2g graph-expand + filter_retrieval_results (RBAC) + L2b cross-encoder rerank
        L3: select_primary_table → Entity Resolution V1 (pin | clarify) → vet_primary (→ clarify)
        no primary → REFUSE no_table
        needs_join = is_existence OR TYPED_MULTITABLE_ROUTE(join/agg/grp/rat) OR _er_multi
        needs_join → try_multitable / build_from_entities
             clarify | refuse | existence(no LLM) | aggregate(no LLM) | sql(LLM SELECT/WHERE)
        else single-table sub-ladder:
             answer-entity → FK-value → multi-hop FK → value-arbiter → temporal-only
             → ranked-temporal-only → temporal-refuse → single-table LLM

  ─────────────── FIREWALL (original query) ───────────────
   1 value_grounding                     → ungrounded
   2 qualifier_completeness              → qualifier_dropped
       └─ salvage: re-anchor retry ; grounded clarify ; referent clarify ; access_denied
   3 grouped_shape_ok      (_llm_sql)    → clarify
   4 distinct_shape_ok     (_llm_sql)    → clarify
   5 alignment_ok (temporal + entity-anchor) → clarify
   6 aggregate_presence_ok               → clarify
   7 filter_presence_ok                  → clarify
   8 dimension_alignment                 → clarify (REFUSE / CLARIFY)
   9 canonical_intent_shadow             (observe only)
  10 validate_ir_equivalence (_llm_sql)  → ir_mismatch
  11 validate_analytical_semantics       (advisory → trace)
  12 narrow_allowed                      (RBAC allow-list narrow)
  13 validate_and_parameterize           → invalid / access_denied
       (AST read-only · table/col exist · ON-integrity · graph_guard FK-edge + connectivity · fan-out · parameterize)
  14 execute_sql (readonly, 30s, fetch EXECUTION_RESULT_LIMIT=1000)  → exec_error
  15 L7b NL-back answer + result_analyzer (never blocks)
  16 save_verified_query  (not cache/fp/temporal/existence, rows>0)
  ─────────▶ _done("answered", cols, rows, answer, sql, table)
```

### 5.1 The Tier‑2 firewall variant

`_tier2_validate` (`veda_hybrid.py:1667`) + inline logic in `_tier2_sql` runs:
`value_grounding` + `qualifier_completeness(strict=True)` +
`validate_ir_equivalence(llm_generated=True)` + (flag) `validate_analytical_semantics`
with `SEMANTIC_VALIDATION_ENFORCE` able to **hard‑fail** — **then** `narrow_allowed`
→ `validate_and_parameterize` (graph‑guarded) → `execute_sql`. Any gate failure →
`{"status":"tier2_rejected"}` and the deterministic head's refusal stands.
`VALIDATION_REPAIR_LOOP_ENABLED` (= False) would otherwise feed `_repair_hint_for(err)`
back into the SLM prompt for up to `VALIDATION_MAX_REPAIR_ATTEMPTS` retries.

---

## 6. Terminal statuses

From `run_query`:

```
answered · no_table · clarify · refuse · ungrounded · qualifier_dropped ·
ir_mismatch · invalid · exec_error · access_denied
```

From Tier‑2: `tier2_rejected` · `tier2_exec_error`.

From the front door: `runtime_context` · `federated` / `federated_failed` /
`federated_refused` · `conflict` / `no_access` / `no_match` / `clarify`
(coordinator, `SHADOW=0` only) · the decomposer's nested‑refuse.

`ok == (status == "answered")`.

---

## 7. RBAC enforcement points (Gate 1)

`ctx.allowed_resources` is populated by `apps.access_management.compute_data_scope`
and forwarded over the `X-Veda-Data-Scope` HTTP header; the inference middleware
parses it (fail‑closed) onto `RequestContext.allowed_resources`. All points are
**pure identity no‑ops when `allowed_resources is None`** (`VEDA_RBAC_MODE` off).

```
ctx.allowed_resources
   │
   ├─ retrieval candidates:  filter_retrieval_results(results, sm, ctx)     pipeline.py:642
   ├─ feedback wording:      restricted_names(sm, ctx) → "_rbac_restricted"  pipeline.py (feedback path)
   ├─ FINAL allow-list gate: narrow_allowed(tables, cols, sm, ctx)          pipeline.py:1845  (+ every Tier-2 validate site)
   │      → validate_and_parameterize rejects any SQL referencing a trimmed name
   │        → invalid → reclassified access_denied if _sql_references(sql, restricted_name)
   ├─ NoSQL schema:          filter_nosql_collections(colls, sid, ctx)      veda_hybrid.py (nosql branch)
   └─ doc chunks:            filter_doc_chunks(chunks, ctx)                 inside retrieve_top_k_chunks
   sm itself is NEVER filtered — the shared per-scope retrieval engine and the
   feedback path both need the full model.
```

`narrow_allowed` (`veda/rbac_filter.py:227`) is the **one centralised allow‑list
choke point**, immediately before `validate_and_parameterize`.

---

## 8. Performance envelope

The engine is CPU‑bound in the dev deployment (Docker‑on‑macOS, no GPU
passthrough); the SLM runs on the host Metal Ollama. Numbers below were measured
in the inference container on 2026‑07‑09 (from `docs/archive/QUERY_PIPELINE_OPTIMIZATION.md`
and `VEDA_Latency_Implementation_Plan.md`) and still describe the current build.

**Bottlenecks, ranked:**

1. **SLM SQL‑gen on CPU** — 239 s/call in‑container. **Fixed** by repointing
   `OLLAMA_URL` to `http://host.docker.internal:11434` (host Metal, model resident
   in VRAM) → ~8.7 s/call. A machine without that compose setting pays the 239 s.
2. **Query‑time re‑encode + rerank of long‑text candidates.** BGE‑M3 encoding 60
   × 512‑token texts is ~98 s on CPU; cross‑encoder reranking 60 long pairs is
   ~101 s. Mitigations shipped:
   - `BIENCODER_CANDIDATE_COLS = 24` (`config.py:1203`, was 80) — fewer texts to
     sparse‑encode and fewer rerank pairs.
   - `RERANKER_MAX_TEXT_LEN = 160` (`config.py:1263`, was 512) — cross‑encoder
     cost is ~quadratic in length.
   - `SPARSE_FIT_MAX_DOCS = 300` (`config.py:217`) — the **critical hang guard**:
     query‑time `sparse_ranker.fit()` no longer live‑encodes ~1900 long
     `retrieval_documents` (~50 min). When the persisted sparse index
     (`column_sparse_v1`) is missing for the scope, retrieval **degrades to
     dense + FK + value** signals.
   - Startup warmup in `inference/loaders.py::hydrate` — BGE‑M3 dense/sparse + SLM
     (reranker falls back to lazy load).
3. **`WORKERS = 1`** on the dev box — one query gets all cores (fastest single
   query), but a slow query blocks the tier. Raise on GPU / multi‑core prod.
4. **Cold model load** on the first request — BGE‑M3 ~22 s, plus the reranker.

**Budgets that gate Tier‑2:** `TIER2_SKIP_IF_HEAD_OVER_S = 120.0`,
`TIER2_TIME_BUDGET_S = 120.0` (`config.py:1324`, `:1329`).
`QUALIFIER_REANCHOR_MAX_HEAD_S = 45.0` bounds the salvage retry.

**Measured warm results (source 2):** deterministic path **0.55 s**; LLM‑IR
fallback path **25.5 s** (HTTP 200, under the 30 s goal and well under
`INFERENCE_TIMEOUT_S = 300`).

**Still open** (`docs/archive/`): populate `column_sparse_v1` for tabular sources;
GPU / host‑native torch for BGE‑M3 + reranker; auto‑federation wiring; a scoped
multi‑source semantic model at query time (a `source_ids:[4]` query today still
retrieves + validates against the single global homzhub sm).

---

## 9. Dormant / dead surface

All of the following are fully built but do not affect a production query today.

| Surface | Files | State |
|---------|-------|-------|
| Enterprise query‑understanding layer | `veda/understanding/` (`__init__`, `orchestrator`, `extractor`, `grounding`, `schema`) | `QUERY_UNDERSTANDING_ENABLED = False` — complete extract → ground → firewall layer, dormant |
| Structured analytical SQL v2 | `veda/analytical_spec.py` | `ANALYTICAL_SQL_V2 = False` — `derive_spec` + `emit_sql` unused |
| Retrieval‑recall sidecar | `veda/query_enhancement.py` | `QUERY_ENHANCEMENT_ENABLED = False` — the whole typo/plural/synonym/alias machinery is built but never consulted; its L0 follow‑up branch is an explicit no‑op "integration seam" |
| Decomposition | `_maybe_split`, `_fan_out`, `run_decomposer`, `_ThreadRouter`, `query/multi_result.py` compound path | `QUERY_DECOMPOSE_ENABLED = False` ("splits join queries wrongly") |
| Tier‑2 repair loop | `_repair_hint_for`, the `for _attempt in range(_max_repairs+1)` loop | `VALIDATION_REPAIR_LOOP_ENABLED = False` → exactly one attempt, no retries |
| Canonical‑intent shadow | `veda/canonical_intent_shadow.py` | observe‑only, `CANONICAL_INTENT_SHADOW_ENABLED = False` — `record_shadow` no‑ops |
| Superlative → join routing | (flag) `SUPERLATIVE_JOIN_ROUTING = False` | superlatives never flip `intent` to `AGGREGATE`; stay single‑table |
| Rule‑based intent classifier | `query_engine/intent_detector.py` | **deliberately not called** by `run_query` (`pipeline.py:340`); only tests import it |
| — | `veda/routing_slm.py` | **empty file (0 bytes)** |
| Multi‑source authoritative branches | `_run_doc_data`, `_datalake_isolated_sm`, `_augment_sm_for_datalake`, coordinator `SINGLE`/`MULTI`/`NO_MATCH` | reached only with `MULTISOURCE_ROUTING_SHADOW = 0` |
| Cross‑source federated route | `_maybe_federated` | live, but only when the ambient scope carries ≥ 2 `source_ids` |

**Self‑contradicting surface to be aware of:**

- `execute_sql` fetches **1000** rows (`EXECUTION_RESULT_LIMIT`), but the
  `pipeline.py` `[L7]` log line says `fetch ≤20`, `_print_rows` prints 20, and
  `record_result_stages` sets `truncated = len(rows) >= 20` — a 900‑row answer is
  reported as "truncated at 20", and the NL summariser can be handed up to 1000
  rows.
- `pipeline.py:591` logs `"5-signal (BGE-M3 + BM25 + …)"`; the retrieval engine is
  **6‑signal** and BM25 was replaced by learned‑sparse M3.
  `retrieval_engine_phase3.py` is internally inconsistent about this
  (`5-signal` in the header and `:231`, `6-signal` at `:408`).
- `filter_presence_ok` reads `INTENT_SQL_FILTER_PRESENCE_ENABLED` via
  `getattr(cfg, …, True)` — **there is no such symbol in `config.py`**; the guard
  is on unless someone adds the flag set to `False`.

---

## 10. Contract drift (`veda_core/query/contracts/`)

The L‑layer contracts sit next to the code but predate it:

- `README.md` / `HEADS.md` name `query_router.py` as the front door — it is
  `veda_hybrid.classify()`, with two doc‑intent overrides ahead of the router.
- `HEADS.md` claims a router "embedding fallback when signals are ambiguous
  (`QUERY_ROUTER_CONFIDENCE_THRESHOLD`)" — **no such code exists**.
- `HEADS.md` names MiniLM for the RAG head — it is BGE‑M3.
- `L6_VALIDATION.md` lists value‑grounding + qualifier + AST only — the code runs
  five more default‑ON alignment guards + IR‑equivalence + graph‑guard, and the
  qualifier gate now re‑anchors and retries once.
- `L7_EXECUTION.md` / `L2_RETRIEVAL.md` say "≤ 20 rows" / "5‑signal RRF" — now
  1000 / 6.

See [ARCHITECTURE.md](ARCHITECTURE.md) §4 for the platform‑level summary and
`veda_core/query/contracts/` for the (stale) layer contracts.
