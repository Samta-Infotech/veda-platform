# Multi-Source Routing — Durable Decisions & Context

> The "why" that isn't in the code. Read this before touching the feature. Facts here are decisions
> already made — don't relitigate without a reason.

## Why this work exists
Current routing = keyword-frequency counting in `query_router.py` (`_SQL_KEYWORDS`/`_RAG_KEYWORDS`).
It's domain-blind (misses "tenant"/"lease"), confuses same-domain sources, and defaults to SQL on
no signal. Goal: evidence-grounded deterministic routing that also handles genuine multi-source
queries — without rebuilding the working pipelines.

## Architecture decisions (locked)
1. **One shared retrieval, group by source_id** — NOT N independent per-source `assess()` calls.
   Reason: `select_retrieval` already searches all `source_ids` at once, stores are source_id-keyed,
   and `partition_subgraph` already groups. N independent retrievals = duplicate work + incomparable
   scores. This is a CORRECTNESS decision, not a cost one.
2. **Never compare raw cross-type scores** — column-cosine vs chunk-cosine vs exact-match have
   different distributions. Reduce each source to a presence-tier STRONG/WEAK/NONE against its own
   type-specific floor. Compare only within a kind, or via the tier.
3. **Agents wrap execute() only** — routing decides WHICH source; the existing pipeline decides HOW
   to answer. Agents are thin `{source_type: AgentClass}` wrappers over Tier-1/2, DuckDB, RAG.
4. **Deterministic-first, SLM bounded** — order: presence-tier → cross_source_fk edge → is_canonical
   → only then SLM (or clarify). SLM never sees non-candidate sources; its output is always validated
   against candidate IDs before execute. Matches VEDA's refuse-over-guess principle.
5. **PARALLEL now, DEPENDENT deferred** — codebase has no staged A→B execution; it's a rare case and
   overlaps the known compound-dependent-query gap. Don't build the autonomous agent graph.
6. **Reuse, don't rebuild** — `scope.py` (security), Tier-1/2, RAG, `run_federated`, `federated_executor`
   are correct; rebuilding = new bugs, no correctness gain. Only routing + profile + multi-source-
   awareness are genuinely new/wrong.

## The 2 spine-blocking bugs (fix in Phase 2 FIRST)
- `retrieval/retrieval_engine_phase3.py:48-64` — `RetrievalResult` has NO `source_id` field.
- `query/graph_retriever.py:543` (and `retrieval_select.py:238,267`) — hardcode `source_id=""`, so
  graph-derived/injected columns get silently DROPPED by `partition_subgraph` (`if not sid: continue`).
  Until fixed, source grouping is unreliable for any graph/injected candidate.

## What's MISSING (net-new, highest design risk)
- Source profile fields (`domain_tags`/`description`/`is_canonical`) — Source model has none.
- Canonical/authoritative source concept — ZERO in codebase today.
- Merge/conflict policy — `cross_source_composer` is provenance-tagging only, no reconciliation.
- Per-source partial-failure — federation is all-or-nothing (single DuckDB conn).
- routing/execution/merge trace sections — not in `_SECTIONS` (`explain.py:43`).

## Highest hallucination risk
Cross-source synthesis (merging two sources' claims) > generated descriptions > SLM source-pick.
Guard the first aggressively (extend `NL_SUMMARY_NUMERIC_GUARD`); prefer clarify over blend.

## Key reuse anchors (file:line)
- Scope/RBAC: `apps/query/scope.py:84-230`, `_ready_source_ids:254`
- Shared retrieval: `query/retrieval_select.py:32,90,112`; source_id path `query/retrieval_v2.py:93,111`
- Doc chunks: `ingestion/chunk_embedder.py:127,529`
- Group-by-source: `query/cross_source_composer.py:31,48-59`
- Federation: `query/federated_route.py:556-696`; `query/federated_executor.py:139-218`
- Relationship edges: `ingestion/cross_source_graph.py:197-353`; hints `federated_route.py:66-113`
- Retry pattern: `veda_hybrid.py:1443-1600`; config `config.py:1782-1783`
- Tracing: `veda/explain.py:74-490`; spans `mlflow_observability/mapper.py:227-256`
- Orchestrator seam to inject coordinator: `veda_hybrid.py::_run_hybrid_query_inner` (classify() ~:228, dispatch ~:516/844/857/912, federated ~:240)
