# Query-engine open items

Residual engineering items lifted from now-archived plans so they aren't lost on the
move. Verify each against `master` before picking it up — some may have shipped since.

Source plans: `docs/archive/ARCHITECTURE_ROOT_CAUSE_PLAN.md`,
`docs/archive/ANCHOR_ROUTING_FIX_PLAN.md`, `docs/archive/VEDA_Latency_Implementation_Plan.md`.

## Anchoring / routing

- **`score_anchors` mechanism retirement.** The root-cause plan's end state folds
  `score_anchors` into QSR (`veda_core/query/resolution.py`) so anchor selection has one
  evidence model. Still two paths today (`vet_primary` re-ranks + `score_anchors`).
- **Strict-gate referent-strength threshold.** The IR-equivalence / qualifier gate keys
  on column-name tokens + entity IDF + values. Flourish words ("market", "across") still
  occasionally over-block a previously-passing Tier-2 answer (tracked sharp edge, e.g. q37).
- **Golden set expansion incl. grain assertions.** `evaluation/golden_homzhub.jsonl` and
  the `evaluation/suite_*.json` sets should carry explicit grain expectations so a
  wrong-grain answer (asked *per property*, answered *per status*) fails CI.

## Fast-path / cache lanes

- **Unconsumed-qualifier discipline on the fast path.** `FASTPATH_EVIDENCE_GUARD` +
  `QSR_FP_EVIDENCE_FLOOR` demote a fast-path pick with zero typed evidence to the full
  pipeline. Extend the same discipline to every fast-path emission, not just the
  zero-evidence case.

## Latency

- **Refusal-lane latency.** Tier-2 still burns 40–90 s before declining on a query it
  cannot answer. `VALIDATION_REPAIR_LOOP_ENABLED` is off (good), but the envelope→IR
  attempt itself is unbudgeted below `TIER2_TIME_BUDGET_S = 120`.
- **`TIER2_SKIP_IF_HEAD_OVER_S`.** Config default is `120.0`; the call-site comment in
  `veda_hybrid.py` still describes a 60 s budget. Reconcile.
- **Heavy-lane budgets.** Per-lane wall-clock budgets (fast / clarify / heavy) are
  asserted only in `evaluation/latency_assert.py`, not enforced in the engine.
- **T9 fast-path coverage**, **T11 async NL-back**, **T14 vLLM query-time backend** —
  from the latency plan, partially addressed.

## Observability

- The engine emits no `tenant` / `session` / `user` id, no `cpu_usage` / `memory_usage`,
  no `rerank_model` / `rerank_latency` into the explain trace — `mlflow_observability`
  has schema slots waiting (`coverage.json`). Capturing them needs pipeline edits.

## ✅ FIXED 2026-09-10 — `storage_adapters/reader.py::ann_search` database target

- **Was:** `reader._connection()` connects to `POSTGRES_DB` (`veda`). `ann_search()` queried
  `column_embeddings_v2`, which `veda_core/ingestion/biencoder.py` writes to
  `VEDA_INTERNAL_DB` (`veda_engine`) and `veda_core/query/retrieval_v2.py` reads from the
  same — a different database on the same Postgres server. `pgbouncer.ini` passes the DB
  name through unchanged. So `ann_search` raised `relation "column_embeddings_v2" does not
  exist`, which `veda_core/retrieval/semantic_search.py:133` caught and logged as
  `Signal 1 adapter unavailable`, falling back to the engine's own `_internal_db_config()`
  connection — which works, but applies **no `source_id` filter**. Net: dense retrieval
  (Signal 1) ran but lost its multi-source scoping (the exact cross-source leak the adapter
  path was added to fix, `semantic_search.py:157-165`).
- **Fix:** added `reader._internal_connection()` (reads `VEDA_INTERNAL_*`, matching
  `writer.sync_from_engine` / `config.VEDA_INTERNAL_DB`, with a graceful fallback to the
  PgBouncer host + `POSTGRES_USER`/`PASSWORD`) and moved only `ann_search`'s vector scan onto
  it. `_resolve_ef_search`'s `substrate_substrateversion` read stays on `_connection()`
  (`veda`), correctly. Log strings in `semantic_search.py` updated.
- **Live-verified 2026-09-10** (stack up, pg16 locally — see the compose note below): calling
  `reader.ann_search` directly hit `relation "column_embeddings_v2" does not exist"` against
  the WRONG-db code and, once pointed at `_internal_connection()`, hit a **second, previously
  masked bug**: `RequestContext.source_ids` are `int` (`context.py` casts every element), but
  `column_embeddings_v2.source_id` is `TEXT` — `WHERE source_id = ANY(%s)` raised
  `operator does not exist: text = integer`. Every other caller of this table
  (`retrieval_engine_phase3.py:193,392`) already stringifies first; `ann_search` now does too
  (`source_ids_str = [str(s) for s in source_ids]`). After both fixes, a real query logged
  `✓ Signal 1 via storage_adapters (engine store, source-scoped): 5 cols` with real cosine
  scores (0.46–0.50) for source 2 — dense retrieval is confirmed working, source-scoped, live.

## Multi-source coordinator authoritative mode — regressed simple queries, fixed via scoping (2026-09-10)

**Status: fixed.** Option 2 below ("only make MODE_MULTI authoritative") is implemented in
`veda_hybrid.py::_run_coordinator` and live-verified — see the bottom of this section. The
original incident + root cause follow, kept for the record.

`docs/MULTI_SOURCE_DEPLOYMENT.md` §"Post-deploy checklist" prescribes
`MULTISOURCE_ROUTING_SHADOW=0` (+ `REQUIRED_SOURCE_ESCALATION_ENABLED=0`) for production. The
dedicated test suite passes (63/65 — see below) and the architecture is genuinely sophisticated:
per-source evidence + dominance tiering (`source_evidence.py`), a deterministic-first policy
with structural edge-driven MULTI and canonical tie-break (`routing_policy.py`), an SLM only at
a genuinely ambiguous boundary (`routing_slm.py`), permission-aware pre-check, and a bounded
doc+data grounding step for compound cross-modal questions (`doc_data_planner.py`,
`DOC_DATA_GROUNDING_ENABLED`, default on).

**Tried it live against this deployment's real data (2026-09-10) and hit a regression**, so it
was reverted (`MULTISOURCE_ROUTING_SHADOW` back to `1`):

- `veda_hybrid._run_coordinator` calls `query/source_coordinator.py::plan_route`, which scores
  candidate sources using its OWN evidence provider — a plain cosine lookup over
  `column_embeddings_v2` / `doc_chunks` (`_default_evidence_provider`), decoupled from the real
  answer engine's retrieval on purpose (its own docstring: "routing tiers on relevance, so it
  needs the raw query↔column cosine — NOT the answer path's reranked score").
- When authoritative (`SHADOW=0`), a `NO_MATCH` or wrongly-ambiguous decision from *that* scoring
  pass causes `_run_coordinator` to return a refusal `MultiResult` **directly** — before
  `veda/pipeline.py::run_query` (6-signal retrieval, fast path, deterministic planners, rerank,
  Tier-2 LLM fallback) ever runs. The routing evidence layer acts as a hard gate in front of a
  strictly more capable engine, using a strictly weaker signal.
- Reproduced live: with `SHADOW=0`, `"how many properties are there"` (pinned to source 2)
  refused `no_match: "unrelated to any source"` — a plain aggregate question the real engine
  answers fine on its own (confirmed by reverting: same query then went through `run_query` and
  got its OWN pre-existing `clarify` outcome, a different and separate anchor-quality issue, not
  a hard NO_MATCH). `"list top 5 properties by monthly rent"` (pinned to source 2) went from
  `status: ok` (wrong-anchor answer, a separate known issue) to `status: refused
  (qualifier_dropped)` — the coordinator dispatched it down a different path than the direct
  `run_query` call the legacy route uses.
- Unpinned cross-source test (`"compare maintenance costs across properties and invoices"`, no
  `source_ids`) also regressed: the SLM boundary resolver picked a single source for a genuinely
  ambiguous case and the `MODE_MULTI` validator rejected it → `CLARIFICATION_REQUIRED
  (INVALID_SLM_DECISION)` — a refusal, where the un-gated legacy path (federated attempt →
  fallback → router) at least produces an answer.

**Test suite result (for context, not sufficient to catch this class of regression):**
`python tests/test_{source_coordinator,routing_policy,source_evidence,routing_slm,
federated_reliability,federated_labelling}.py` (standalone, their intended invocation —
NOT plain `pytest`, which cross-pollutes global module state across these files and produces
2 false failures) → 63/65 pass. The 1 genuine, irrelevant failure is
`test_dispatch_with_adapter_flag_on_matches_flag_off_exactly` (`query.source_adapters` module
doesn't exist — but `SOURCE_ADAPTER_DISPATCH_ENABLED`/`EXECUTION_REQUEST_DISPATCH_ENABLED`
both default off, so that path never runs). None of the 65 tests exercise "authoritative
coordinator vs. a simple single-source query the legacy path already answers well" — the whole
suite is written against multi-source disambiguation scenarios, so it never caught this gap.

**Fix options, not yet attempted:**
1. **Soften the gate** — only let `_run_coordinator` return a refusal authoritatively when its
   own decision is high-confidence (e.g. `RC_SINGLE_CANDIDATE` with a dominant STRONG tier, or a
   structural `RC_RELATIONSHIP_EDGE`/`RC_CANONICAL_SELECTED`); on `NO_MATCH` / low-confidence
   `AMBIGUOUS`-resolved-by-SLM, return `None` (fall through to the legacy path) instead of a
   refusal — closer to "advisory, escalate only when clearly right" than "gate always."
2. **Only make MULTI decisions authoritative**, leave SINGLE/NO_MATCH in shadow — gets the
   genuine cross-source win (federated + doc-data grounding) without gating the single-source
   traffic that the mature engine already handles well. Needs `_run_coordinator` to branch on
   `decision.mode` before checking the shadow flag, not after.
3. **Strengthen the routing evidence provider** so it doesn't disagree with the real engine's own
   retrieval as often — more invasive, requires re-tuning `ROUTING_TIER_*` / dominance floors
   against this deployment's actual schema/data, not just the benchmark fixtures the comments
   reference.
4. Add regression tests exercising this exact gap (authoritative coordinator vs. known-good
   single-source queries) before ever re-enabling in this deployment.

### Fix implemented: option 2 (scope authoritative to `MODE_MULTI`) — 2026-09-10

`veda_hybrid.py::_run_coordinator`: after `decision = plan_route(...)`, compute
`_is_multi_decision = decision.status == "ROUTED" and decision.mode == "MULTI"` and
`_effective_shadow = MULTISOURCE_ROUTING_SHADOW and not _is_multi_decision`; use
`_effective_shadow` (not the raw flag) for the shadow gate, the trace's `shadow=` field, and
the verbose `[shadow]` tag. `.env` unchanged from the reverted state
(`MULTISOURCE_ROUTING_SHADOW=1`) — the new logic makes `MODE_MULTI` authoritative *under that
same setting*, so nothing else needs to flip.

**Verified:**
- Full test suite unchanged: 63/65 (same 2 pre-existing, irrelevant failures as before the fix).
- Isolated test: `SHADOW=True` + a mocked `MODE_MULTI`/`RELATIONSHIP_EDGE` decision →
  `_run_coordinator` returns a real answer (`[multi-authoritative]` in the verbose trace);
  `SHADOW=True` + a mocked `SINGLE` decision → returns `None` (falls through), confirming both
  halves of the scoping.
- Live, against this deployment's real data:
  - The 5 previously-regressed/baseline single-source queries (pinned to source 2) all match
    their pre-incident behavior exactly — no new regression.
  - The explicitly-pinned federated case (`source_ids=[2,3,4,5]`, "compare maintenance costs
    across properties and invoices") still answers via `route: federated`.
  - The **unpinned ambiguous cross-source case that hard-refused under unscoped SHADOW=0**
    (`CLARIFICATION_REQUIRED / INVALID_SLM_DECISION`) now correctly logs `[shadow]` (the
    coordinator's own resolution wasn't `MODE_MULTI`, so it deferred) and gets a real answer
    from the legacy federated path (`status: ok, route: federated`) — the exact "graceful
    fallback instead of hard refusal" option 2 was meant to restore.

**Confirmed live, organically, 2026-09-10 (same day as the fix):** `"which assets have which
amenities"` (no source pin) — the coordinator's own evidence scoring found sources `5`
(`amenities_catalog`, parquet) and `2` (`assets_amenity`, relational) both relevant, matched
the `cross_source_fk` edge between them (`HIGH` tier, discovered at ingestion), and resolved
`ROUTED/MULTI (RELATIONSHIP_EDGE) [multi-authoritative]` — not a mock, not a fallback through
`_maybe_federated`'s own ambient-scope trigger. It then ran `_maybe_federated(..., strict=True)`
inside the coordinator's own MULTI branch, which could not build a validated per-entity join
plan for this "LOOKUP_ENRICH" shape and **surfaced a controlled refusal**
(`"could not build a validated per-entity join plan for this LOOKUP_ENRICH"`) rather than
guessing or silently falling back to one source — exactly the "no silent guessing" behavior
strict federation exists for.

For contrast, the near-identical wording `"list amenities for each asset"` resolved to
`NO_MATCH` from the coordinator (correctly deferred, `[shadow]`), and the *legacy* federated
path answered it anyway because it turned out fully answerable inside source 2 alone (a
same-source join through two junction tables) — no genuine cross-source data was actually
needed despite the surface wording. This is a good example of the coordinator's structural
signal (a real discovered FK edge) correctly distinguishing a genuine cross-source shape from
a same-source one that merely names both entities.

## Fix-list triage 2026-09-10 (external code review: "VEDA — Fix List for Code Agent")

A colleague's code review produced a P0/P1/P2 fix list for cross-source correctness gaps.
Ground rules match `CLAUDE.md` (no commit/push, bind-mount + `up -d` for `.env`, `VEDA_INTERNAL_*`
for engine tables). Executing in the list's suggested order; reporting per step.

### Step 1 (P2-1) — test suite baseline: DONE

Ran the routing-test family the documented way (`python tests/test_X.py`, standalone — plain
`pytest` isn't installed anywhere in these containers and cross-pollutes global module state
across these files if it were, per the existing note above). All 4 tests in
`.pytest_cache/v/cache/lastfailed` triaged:

| Test | Verdict | Action |
|---|---|---|
| `test_federated_labelling.py::test_genuine_join_failure_is_surfaced_not_silent` | pytest cross-module pollution, false failure | none — 5/5 standalone |
| `test_federated_labelling.py::test_genuine_join_success_flows` | same | none |
| `test_source_coordinator.py::test_dispatch_with_adapter_flag_on_matches_flag_off_exactly` | genuine failure, but dead code: `query/source_adapters.py` (imported by `source_coordinator.py:750`'s `_resolve_executable`) does not exist in the tree at all. Unreachable today — `SOURCE_ADAPTER_DISPATCH_ENABLED` and `EXECUTION_REQUEST_DISPATCH_ENABLED` both default `"0"` (`config.py:808,822`) — but flipping either flag on would crash with `ModuleNotFoundError` on the very first single-source dispatch. Left as-is (building a real adapter module is a feature task, not a test fix) but flagging here so it isn't mistaken for a live bug or silently "fixed" by deleting the test. | none, documented |
| `test_authoritative_routing.py::test_single_datalake_augments_sm_and_scopes_relational_does_not` | genuine, and fixable: stale test. `SOURCE_ISOLATED_RETRIEVAL_ENABLED` now defaults **ON** (`config.py:604`), so `veda_hybrid.py`'s datalake SINGLE branch takes the isolated-sm path and never reaches `_augment_sm_for_datalake` — the test's monkeypatch on `_augment_sm_for_datalake` alone can no longer observe anything. | **Fixed**: test now also patches `_datalake_isolated_sm` to return `None` (simulating isolation-unavailable), which exercises the intended augment-fallback path again. `veda_hybrid.py`'s stale "flag-gated, default OFF" comment corrected to "default ON" in the same change. |

Result: the routing family went from the documented 63/65 to **64/65** (one genuine test fixed,
one genuine-but-dead-code failure now explained instead of silently latent, two pytest artifacts
confirmed not real). Did not run the other ~94 unrelated `tests/test_*.py` files — out of scope
for this item, which the fix list itself scopes to the 4 `lastfailed` entries.

### Step 2 (P0-3), first half — biencoder resume skip: FIXED

`ingestion/layers/l4_index.py::_biencoder_embeddings_exist()` ran `SELECT 1 FROM
column_embeddings_v2 LIMIT 1` with **no `source_id` filter** — under `VEDA_RESUME=1` (which
`apps/ingestion/tasks.py::_should_resume` sets automatically after a prior failed job), ingesting
source B would see source A's rows, treat biencoder as already-done, and skip embedding B
entirely, so B ends up with no retrieval Signal 1. Fixed: takes `source_id`, filters
`WHERE source_id = %s` with `str(source_id)` (the column is `TEXT` — same bug class as the
2026-09-10 `ann_search` fix above). Live-verified against the real `veda_engine` store: source 2
(has rows) → `True`, source 999 (no rows) → `False`, and both `str`/`int` source_id inputs work.

### Step 2 (P0-3), second half — L3 semantic-model resume skip: NOT fixed, coupled to P0-1/P0-4/P0-5

`ingestion/layers/l3_enrich.py:21` checks `os.path.exists(SEMANTIC_MODEL_FILE)` for resume-skip.
`SEMANTIC_MODEL_FILE` resolves via `config.artifact_path()`, which is scoped
`<root>/<tenant>/<source>/<version>/<name>` **only if** `VEDA_ARTIFACT_SCOPE` is set for the
process (today, only the ingest subprocess sets it — P0-7); otherwise it's the flat
`data/veda_semantic_model.json` shared by every source. Two ways this could be "fixed" in
isolation, and why neither is safe to do alone:
- Add a source-scoped resume marker independent of the scoping flag → stops the wrong *skip*,
  but L3 would then rerun and **save B's model over A's in the same flat file** on the very next
  line (`save_semantic_model(semantic_model, SEMANTIC_MODEL_FILE)`) — that's P0-4's exact
  overwrite bug, not fixed by touching only the skip condition.
- Give L3 a real per-source *write* path too, bypassing the file → but query-side readers
  (`fast_path._sm()` reads `SEMANTIC_MODEL_FILE` directly, not the assembled per-request `sm` —
  P0-5) still wouldn't know how to find a per-source file, so writing one changes nothing at
  query time; the platform would silently keep serving whatever the old flat model held.
- So this half of P0-3 is genuinely **not isolated** despite the fix list's framing, and needs the
  P0-1/P0-5 per-source artifact resolver landed first (one accessor, both read and write sides,
  as that item specifies) before it can be fixed correctly rather than papered over. Recording
  this here so the next pass doesn't attempt a partial fix that looks green but leaves the
  overwrite live.

### Step 3 (P0-1 + P0-2 + P0-4, relationship graph): DONE — 2026-09-10

The relationship graph was the one artifact worth doing as its own full pass first: it feeds
join planning AND the firewall's `verify_joins_against_graph`/`check_connectivity` — the
highest-stakes correctness surface of the three (P0-1/P0-5's other 7 artifacts are query-quality,
not a "wrong-join could get built and pass the firewall" class of risk).

**P0-1 — unified accessor, per-source persistence.**
- `config.source_artifact_path(name, source_id, tenant)` (new): `<ARTIFACT_ROOT>/<tenant>/<source_id>/<name>`,
  ALWAYS keyed by source_id — unlike `artifact_path()`/`artifact_scope()`, which only scope when
  `VEDA_ARTIFACT_SCOPE` is set for the process (today, only the ingest subprocess — P0-7 still open).
- `veda/runtime.py::get_graph(source_id=None)` is now THE one graph accessor: resolves source_id
  from the ambient request context when not given, caches per `(tenant, source_id)` in one dict
  (`_GRAPH_CACHE`), and a source missing its own graph file gets an EMPTY graph for itself — never
  another source's. `invalidate_graph_cache(source_id=None)` added alongside it.
- `veda/graph_guard._load_graph()` and `query/fast_path._graph()` — previously two MORE
  independent process-global caches of the same unscoped flat file — now both delegate to
  `veda.runtime.get_graph()` instead of loading their own copy. graph_guard keeps a small
  derived-index memo (edge_set/cardinality), invalidated by the underlying graph object's
  identity, so repeated firewall calls don't rewalk the edge list every time.
- `query/join_planner.py::load_graph(path=None, source_id=None, tenant=None)`: explicit `path`
  still wins (back-compat for `ingestion/value_referents.py`'s own call); otherwise resolves
  per-source via `source_artifact_path()`, falling back to the legacy flat file only with
  neither a source_id nor an ambient context (dev-CLI). The three previously-bare `load_graph()` /
  `get_graph()` callers (`veda/pipeline.py:1643`, `query/ratio_plan.py:219`,
  `veda/planning.py`/`routing.py`/`result_analyzer.py`/`retrieval/signal_builder.py`/
  `query/retrieval_v2.py`, all zero-arg) needed NO changes — they become source-scoped for free.

**P0-2 — the writer's DB targeting.** `ingestion/relationship_graph.py::_conn()` and the module's
own `RELATIONSHIP_GRAPH_FILE` constant were the two hardcoded-flat-path/hardcoded-connection
pieces (`build_relationship_graph()` itself calling `get_real_schema()`, which ALSO hardcodes
`get_primary_relational_source()` — the actual first crash point for a tabular source, one layer
before `_conn()`). Fixed:
- `_raw_schema_for(ctx)` mirrors `ingestion/layers/l1_extract.py`'s own connector dispatch: a
  file-backed tabular source (`ctx.engine` in `csv/csv_lake/parquet/xlsx/excel`) builds its schema
  from `TabularFileConnector` directly; everything else uses the legacy `get_real_schema()` shim
  (correct here because this process was launched FOR `ctx.source_id` — single-source-per-process).
- `_conn(ctx)` uses `ctx.connection` when given (byte-identical fallback to
  `get_primary_relational_source()` for the ctx-less dev-CLI call, which in this architecture
  resolves to the SAME source either way).
- PK detection: replaced the Postgres-only `pg_index`/`pg_attribute` catalog query with
  ANSI-standard `information_schema.table_constraints` + `key_column_usage` — works identically on
  Postgres today and is the same query a future non-Postgres relational source would need, so
  there is now one PK-detection path instead of a Postgres-only one plus a hypothetical second.
- Schema: `ctx.schema_filter` (or the connection dict's `schema`) instead of hardcoded `'public'`
  for the `information_schema.columns` filter, and `_conn()` now sets `search_path` to that schema
  (same idiom `veda/runtime.py`'s own query-time connection already uses) so the unqualified
  `"{table}"` identifiers in `_polymorphic_edges`/`_cardinality` resolve correctly too.
- Tabular sources (and any relational engine other than Postgres, since there is no live one to
  verify dialect-specific SQL against — the fix list's own offered fallback): declared-FK-edges
  only, no live SQL introspection (`cardinality="unknown"`, `stats.mode="declared_fk_only"`,
  logged). Crash → real (if coarser) graph.

**P0-4 — no more empty-graph overwrites.** `layers/l5_publish.py` now computes `tables` from THIS
run's own in-memory `state["semantic_model"]` / `state["scan_result"]` and passes it explicitly
into `build_relationship_graph(tables=..., ctx=ctx)` — never a re-read of `SEMANTIC_MODEL_FILE`
from disk (which could be another source's, or stale). `build_relationship_graph()` itself now
also refuses to write when the requested `tables` scope comes back empty but the source's OWN
schema scan found tables (`raise RuntimeError` → non-fatal stage failure, existing graph on disk
untouched) — the empty-graph-overwrite path P0-4 was about.

**P0-6 (graph only)** — both rehydrate paths (`inference/main.py`'s pub/sub subscriber and
`inference/routes/retrieve.py`'s `/v1/rehydrate`) now also call
`veda.runtime.invalidate_graph_cache()`, alongside the existing sm/registry/engine/ef_search/PPR
clears. `build_relationship_graph()` also self-invalidates the one source it just wrote. (The
other P0-5 artifacts — semantic model, rerank docs, synonyms, join paths, enrichment index,
unified graph, registry — are NOT covered by this rehydrate addition; still stale after a
re-ingest until P0-5 lands.)

**Migration.** Before this fix, ALL sources shared one flat file
(`veda_core/data/veda_relationship_graph.json`, 178 tables/609 edges, homzhub's schema, dated
2026-07-08) — deploying the per-source resolver with nothing at the new path would have made
`get_graph()` correctly-but-newly return an EMPTY graph for source 2 (a live regression, caught
by my own end-to-end smoke check before calling this done, not by a user report). Regenerated
source 2's graph for real through the fixed builder (live DB, real `Source(id=2)` connection
info) and confirmed it lands at `veda_core/data/default/2/veda_relationship_graph.json` with
**identical stats to the legacy file** (178 tables, 609 edges, 0 polymorphic) — this is a
regeneration, not a copy, and it is gitignored (`veda_core/data/`) like every other derived
artifact, so it needed no repo change to place.

**Verified, live (all against the real stack, not mocks):**
- Source 2 (homzhub, real Postgres): rebuilt via the fixed builder end-to-end (`_raw_schema_for`
  → `_conn`/search_path → INFORMATION_SCHEMA PK detection → cardinality) — 178 tables / 609 edges,
  byte-for-byte stat match with the pre-fix flat file.
- Source 4 (`invoices_csv`, csv_lake) and source 5 (`catalog_parquet`, parquet): previously
  crashed this stage outright (`ValueError`/`ModuleNotFoundError`, confirmed by reproducing the
  pre-fix crash first). Post-fix: `_raw_schema_for` correctly builds from `TabularFileConnector`
  (2 tables found for source 4: `maintenance`, `vendors`), declared-FK-only mode runs cleanly,
  `stats.mode="declared_fk_only"` — no crash, valid (if edge-less, for these datasets — see gap
  below) graph produced and written to its own per-source path.
- After regenerating source 2's file, re-checked all three accessors in the SAME process context
  the real inference service runs in (cwd `/app/veda_core`, confirmed via `/proc/1/cwd`):
  `veda.runtime.get_graph()`, `veda.graph_guard._load_graph()`, `query.fast_path._graph()` all
  report 178/609 consistently.
- Firewall smoke test against the new per-source graph, live: `verify_joins_against_graph` on a
  real edge (`accounts_generalledger.account_type_id = accounts_generalledgercategory.id`) →
  `(True, None)`; on a fabricated one (`... = users_user.id`) →
  `(False, 'join not backed by a real FK edge...')`. The core safety property survived the
  refactor, not just "it loads without throwing."
- Full routing test suite re-run after all of this: still 64/65 (no new regression).

**New gap found while verifying (NOT one of the fix list's items) — investigated and fixed same
day, 2026-09-10, at the user's request.** Tabular sources got a real, non-crashing graph from the
Step 3 work above, but 0 edges even where a real one exists (source 4's `maintenance`/`vendors`,
joined in reality on `ticket_id`). Root cause: `TabularFileConnector.get_raw_schema_dict()` never
sets `is_fk` (CSVs have no declared-FK concept), so a tabular source's only possible edges come
from `ingestion/data_graph.py`'s undeclared-FK discovery (value-overlap + co-null correlation) —
and that module unconditionally called `get_client_connection()`, which ignores the source_id
it's given (single-source-per-process — `config.get_source`) and fell through to its generic
psycopg2 branch with `host="localhost"` for an engine ("csv"/"parquet") it didn't recognise,
failing closed (`"Continuing without discovered edges"`) for every tabular source, every time.

**Fix:** `run_data_graph()` takes an optional `tabular_connector` param
(`ingestion/layers/l1_extract.py` now passes `state["tabular_connector"]` — the SAME connector
instance L1 already opened, no re-parsing the files); when given, Phase 1 (value overlap) and
Phase 2 (co-null correlation) run against `TabularFileConnector.get_sql_cursor()` (new method) —
a DuckDB connection with every table registered as a view, wrapped in a small DB-API-shaped shim
(`_DuckDBConnShim`/`_DuckDBCursorShim` in `connectors/tabular_files.py`) so the existing SQL in
`_sample_values`/`_count_co_null` (double-quoted identifiers, `%s` params translated to `?`, a
standard `FILTER` clause — nothing Postgres-specific) runs completely unchanged. The
`source_dispatcher.py` "relational only" call site (a genuinely different, lighter pipeline
tabular sources never reach) needed no change.

**Verified, live:** re-ran the same source-4 (`invoices_csv`) reproduction that previously threw
`"Data graph DB connection failed: connection to server at localhost... Connection refused"` —
now completes cleanly (`Eligible from-cols: 8`, `Eligible to-cols: 1`, both phases run, 0 edges
found because of a SEPARATE, pre-existing limitation described below — not a connection failure
this time). Full 64/65 test suite re-confirmed unchanged after this change too.

**Remaining, separate limitation (not fixed, out of scope for this fix)**: `maintenance`/`vendors`
join on `ticket_id`, a STRING natural key — `data_graph._is_eligible_as_target()` only accepts a
PK, an INTEGER `..._id` column, or a UUID as a valid join TARGET, so a string `ticket_id` is never
considered even once the connection works. This is a pre-existing eligibility heuristic shared by
every source (including homzhub), not something specific to tabular sources or introduced by
today's fix — retuning it needs the same before/after eval measurement the fix list's own P1 items
ask for, not a blind change. Recorded here for whoever picks that up.

### Step 4 (P0-5) — triaged, 1 of 7 fixed, scope re-estimated: 2026-09-10

Went through the fix list's own P0-5 table item by item before touching code (per "verify each
item's evidence still holds before fixing").

- **Semantic registry** (`semantic/registry.py`) — the fix list flagged this "verify" rather than
  a confirmed bug. Verified: it's ALREADY correctly per-`(source_id, tenant)`-keyed
  (`_CACHE: dict`, with `_active()` refreshing a live-mirror `_STATE` for external readers). No
  fix needed — closing this row.
- **Domain synonyms** — `query/reranker.py::_domain_synonyms()` had a hardcoded absolute
  `<repo>/data/veda_domain_synonyms.json`, bypassing `config.DOMAIN_SYNONYMS_FILE` entirely —
  every OTHER consumer (`veda/validation.py`, `ingestion/enrichment_index.py`,
  `semantic/compile_semantic_layer.py`, `ingestion/unified_graph_builder.py`) already reads
  through that config constant. **Fixed**: reranker now imports `DOMAIN_SYNONYMS_FILE` from
  config too, so all 5 consumers agree on one path. Live-verified it still resolves to the same
  file in this deployment (`data/veda_domain_synonyms.json`, unchanged content) — no behavior
  change, just removes a real "could silently diverge" landmine. **Still unscoped-by-default**
  like every other item in this table — this fix makes it internally consistent, not per-source.

**Re-estimated the remaining scope before going further, rather than partially converting each
one.** `DOMAIN_SYNONYMS_FILE` and its P0-5 siblings (`SEMANTIC_MODEL_FILE`, rerank docs' index
path, join paths, the enrichment index, the unified graph) are each a **module-level constant**
resolved ONCE at import time via `config.artifact_path()` (P0-7's exact complaint), imported by
name across 5-10 files per artifact:
- Domain synonyms alone: `semantic_layer_v2.py` (writer), `unified_graph_builder.py`,
  `enrichment_index.py`, `veda/validation.py`, `retrieval/query_enrichment.py` (its own SEPARATE
  hardcoded path via `self.data_dir` — a second landmine, not yet fixed), `compile_semantic_layer.py`.
- Rerank docs: writer (`ingestion/rerank_docs.py`) + 2 stacked unscoped caches (its own
  `_RERANK_DOCS_CACHE`, then `query/reranker.py`'s `_RERANK_DOCS` on top).
- The semantic model itself is the largest and most invasive: `query/fast_path.py`'s `_sm()` /
  `_SM_CACHE` reads `SEMANTIC_MODEL_FILE` directly rather than the per-request assembled `sm`
  (`veda_hybrid._SM`, which IS already scope-keyed — the fix list is explicit that this one needs
  fast_path rewired to use that, not a new file-scoping scheme).

Making each of these **genuinely** per-source (matching what the relationship graph now does)
means converting each constant into a function call `(source_id, tenant) -> path` — like
`config.source_artifact_path()`, built for the graph — and updating EVERY one of those 5-10 call
sites per artifact, plus wiring `fast_path._sm()` to `veda_hybrid._SM` for the semantic model
specifically. That is at minimum as large as the P0-1 relationship-graph pass (which touched 9
call sites for ONE artifact) repeated 5 more times, with less of it able to reuse a common
accessor (unlike the graph, most of these have no single "get_X()" chokepoint today). Attempting
it in the same pass as everything else done today risked shipping shallow, unverified changes
across a correctness-sensitive system — the opposite of how every other item in this doc was
handled. Recommending this be its own dedicated pass, one artifact at a time, each with the same
live-verification rigor as the relationship graph got (regenerate for a real source, diff against
the pre-fix behavior, check the test suite) rather than batching all 5 remaining artifacts together.

P0-7 (formalizing/retiring the `VEDA_ARTIFACT_SCOPE` half-mechanism) is naturally next once P0-5
exists to migrate onto it. P1-* (retrieval quality) explicitly wants eval measurement before/after
per the fix list itself. P2-2/2-3 are doc corrections + confirmed-safe deletions.

### Step 4 continued — the 5 remaining P0-5 artifacts + P0-6: DONE, same day, 2026-09-10

User asked to continue rather than stop at the re-estimate above. Did all 5, same rigor as the
relationship graph (regenerate for real against source 2, diff/compare, check the test suite),
using the exact pattern the graph established: `config.source_artifact_path()` for the OUTPUT,
resolve source_id/tenant from ambient context on read, a per-(tenant, source) cache dict with an
`invalidate_*_cache()` wired into both rehydrate paths.

- **Rerank docs** (`ingestion/rerank_docs.py` + `query/reranker.py`): writer now takes
  `tenant` + an optional `semantic_model` (from `l4_index.py`'s own `state["semantic_model"]`,
  P0-4-style — never a re-read of the flat file); reader resolves + caches per-source; removed
  `query/reranker.py`'s SECOND, redundant unscoped cache on top (`_get_rerank_docs()` now purely
  delegates). Live-verified against source 2: 1902 cols / 178 tables, byte-identical content,
  correctly served through `query.reranker._get_rerank_docs()`.
- **Join paths** (`ingestion/join_paths.py` + `query/join_planner.py`): same shape. Removed
  `join_planner._JOIN_PATHS_MAP`'s own unscoped global (now delegates to
  `ingestion.join_paths.load_join_paths()`, which self-caches per source). Live-verified against
  source 2 by rebuilding from a real schema scan: 27,606 pairs over 176 tables, `_is_reachable()`
  correctly consults it.
- **Enrichment index** (`ingestion/enrichment_index.py`): same shape (inputs — domain
  synonyms/concepts/glossary — are still flat until they get their own fix; only this artifact's
  OWN output is now per-source). Live-verified against source 2: 3496 terms.
- **Unified graph** (`graph/query_graph.py` + `ingestion/unified_graph_builder.py`): the biggest of
  the five — the builder fuses 6 separate inputs (semantic model, relationship graph, concept
  graph, domain synonyms, metrics, dimensions), each its own module-level constant. Rather than
  requiring ALL SIX to be migrated before this artifact could be fixed at all,
  `_resolve_input_paths(source_id, tenant)` prefers the per-source path **only where the
  per-source file already exists** (today: just the relationship graph, thanks to P0-1) and falls
  back to the flat legacy path for the other five — a source gets whatever fraction of its own
  data is actually available, never a hard requirement on the other pending P0-5 items, and never
  a regression versus before. The reader (`query_graph.get_graph()`) converted its single
  process-wide `_GRAPH`/`_GRAPH_SIG` slots to dicts keyed by resolved path, so per-source and the
  legacy flat file coexist correctly; a NEW `invalidate_unified_graph_cache()` was added even
  though `get_graph()` already self-invalidates via on-disk mtime/size (belt-and-braces, wired
  into rehydrate for consistency with the other 4). The existing staleness-fingerprint mechanism
  (`stale_inputs()`/`_fingerprint()`) was threaded through with the same source_id/tenant so it
  compares against the RIGHT resolved inputs, not always the flat ones.
  Live-verified against source 2: rebuilt for real (17,134 nodes / 33,551 edges) — `FK_TO` (448),
  `HAS_COLUMN` (1902), and `REFERENCES` (609) counts matched the pre-existing flat unified graph
  EXACTLY (confirming the relationship-graph-derived portion is correct); `METRIC`/`SYNONYM`
  counts were higher than the old flat file simply because the flat registries/synonyms had grown
  since that file was last built — not a bug, just fresher data. Read back correctly through
  `query_graph.get_graph()` under the real request context.
- **Semantic model** (`query/fast_path.py::_sm()`) — the fix list's own hardest item, and the one
  with the most correctness stakes: `_sm()` now delegates to `veda_hybrid._load_semantic_model()`
  (the SAME scoped, Redis-first sm the SQL head and retrieval engine already use for this exact
  request) whenever a request context is set, falling back to the flat-file cache only for a
  ctx-less call. Before this fix, fast_path's checks ran against a DIFFERENT semantic model than
  the rest of the same request whenever the flat file didn't match the request's actual source —
  not just staleness, a genuine silent cross-source correctness gap.
  **Live-verified end-to-end through the real HTTP path** (not a unit test): `curl` against the
  running inference container with `"how many users are there"` (source 2) hit the fast path
  directly (`[FastPath] metric.count (metric users_user_count) — no retrieval / no LLM` in the
  live log), generated `SELECT COUNT(DISTINCT "id") ... FROM "users_user"`, executed, and answered
  correctly (5,944) in 0.2s — `status: ok` end to end. Also tried `"how many properties are
  there"` as a second check: it refused with a `clarify` outcome — traced this via the live trace
  log to a **pre-existing, already-documented anchor-quality issue** (routes to
  `assets_listingvisit` instead of the properties table; this exact query's `clarify` outcome was
  independently confirmed in this same doc's earlier "Multi-source coordinator" section, from
  BEFORE any of today's P0-5 work) — confirmed NOT a regression from this change, since that query
  never reaches fast_path at all (it goes through the full L1–L6 retrieval ladder).

**P0-6, now fully done** (not just the relationship graph): every P0-5 artifact fixed above got
its own `invalidate_*_cache()` wired into BOTH rehydrate paths (`inference/main.py`'s pub/sub
subscriber and `inference/routes/retrieve.py`'s `/v1/rehydrate` route) — rerank docs, join paths,
enrichment index, unified graph, and the relationship graph from Step 3. The semantic model's own
invalidation was already covered by the pre-existing `veda_hybrid._SM.clear()` in both paths,
since `fast_path._sm()` now reads through that same cache.

Full 64/65 test-suite re-confirmed after every one of the changes above, not just once at the end.

**P0-7 status**: still correctly NOT actionable as a full "delete the old mechanism" — the OTHER
five artifacts one level down (semantic model file, glossary, concept graph, domain synonyms,
compiled registries) still only have `config.artifact_path()`/`VEDA_ARTIFACT_SCOPE` as their
scoping option; deleting that mechanism now would remove even the weak, opt-in scoping those
still rely on, with nothing to replace it until each gets its own dedicated P0-5-style fix
(explicitly out of scope for this pass — see the re-estimate above). Leaving both mechanisms in
place, correctly, until that work lands.

### Step 5 (P1-3, P1-4) — DONE; Step 5 (P1-1) — DONE, eval-harness gap found; (P1-2) — deferred: 2026-09-10

**P1-3 (signal_builder FK graph bugs) — all 3 fixed.** `fk_graph` was `str -> str` (last-edge-
wins for a polymorphic column with multiple targets, despite its own comment already saying
"list of referenced column_ids") — now `str -> set`; `_build_table_adjacency` connects to
EVERY target table, not just the last one. `_compute_column_signals`'s
`any(ref == col_id for ref in self.fk_graph.values())` was O(n) per column (O(n²) over the
whole schema) — replaced with a precomputed `_referenced_cols` set, built once. Dead
`from schema.real_schema import get_real_schema` import removed. Live-verified against source
2's real semantic model: 1902 column signals computed, a referenced column correctly scores
`fk_signal=0.7`, a referencing one `0.5`.

**P1-1 (hub-table bias) — fixed, mechanically, not a weight retune.** `rrf_merger.py`'s
`all_candidates` union included Signals 3/4 (subgraph/FK) directly — unlike Signal 6
(table-first prior), which was always boost-only. 25 tables have structural degree ≥ 10
(`users_user`=275), so `min(degree/10,1)` saturated to a virtual rank-1 hit for every column
of those tables regardless of query relevance — both a bias and a pool-size inflation. Fixed:
Signals 3/4 removed from the union, matching Signal 6's shape exactly (still fully contribute
to score, just never introduce a candidate). **Eval gap found**: `scripts/retrieval_eval.py`
(after fixing a real, separate bug in it — see below) measures `query/retrieval_select.py`,
which does not call `RRFMerger` at all; `RRFMerger` is only used by
`retrieval_engine_phase3.py`, the actual production 6-signal spine. The prescribed
before/after eval numbers came back byte-identical (correctly — the harness doesn't exercise
this code), so this fix is verified instead via a live query through the real path (`[L2]
Retrieval ... → RRF` in the running inference container's log) completing cleanly and
producing a sane candidate set — not a quantified recall/MRR number. Flagging the eval-harness
target mismatch itself as worth someone's attention; not fixed here (out of scope for this item).

**Fixed along the way: `scripts/build_golden_set.py` never ran standalone.** `from
apps.substrate.models import VerifiedQueryCache` executed at module import time, before
`django.setup()` (which was called inside `main()`, i.e. too late) — every invocation crashed
with `AppRegistryNotReady`. Moved `django.setup()` before the model import. Used it for real:
built `evaluation/golden_queries.jsonl` from the live `VerifiedQueryCache` (25 rows → 24
queries, 21 gradeable) and ran `scripts/retrieval_eval.py` for a genuine baseline
(recall@5=0.284, recall@15=0.325, mrr=0.568, table_recall@3=0.714) — this is the harness's
FIRST successful run in this environment.

**P1-2 (intent boosting dead on Tier-1) — implemented + measured, 2026-09-11 (user asked for
"a real tuning/eval pass, not blind" as a follow-up to the initial deferral above).**

- **Rescaled the deltas first**, per the fix list's own prerequisite. Measured live (source 2,
  real query, k=60 RRF): top-10 fused scores sit at 0.05-0.06. `intent_boosting.py`'s deltas
  (±0.10 to ±0.40, additive) at that scale don't nudge a ranking, they replace it outright.
  New `config.RETRIEVAL_INTENT_BOOST_SCALE = 0.12` (puts the largest per-intent delta, 0.40, at
  ~0.05 — a felt boost, not an override) applied to `boost_aggregate`/`boost_temporal`/
  `boost_multi_table` only — deliberately NOT to `apply_history_penalty`'s -0.60, which already
  fires today at full strength for every intent and isn't part of this fix's scope.
- **Wired a retrieval-only intent**, exactly as prescribed: `veda/pipeline.py` now computes
  `_retrieval_intent` (TEMPORAL when a real date range parsed; else AGGREGATE when
  `aggregate_mode`/`grouped_mode` fire; else SIMPLE) from the SAME grammar signals already
  computed for the trace, and passes it to `retrieve()` — `intent` itself (which also gates
  `SUPERLATIVE_JOIN_ROUTING` multi-table planning) is untouched, so planning/routing behavior
  is byte-identical to before.
- **A/B'd on the golden set — built a new eval script for it, since the existing one can't
  see this code either.** `scripts/retrieval_eval.py`/`query/retrieval_select.py` doesn't call
  `RRFMerger`/`IntentBooster` at all (same gap P1-1 hit — see that entry). New
  `scripts/eval_p1_2_intent_boost.py` calls `retrieval_engine_phase3.py`'s `retrieve()`
  directly (the actual call site `veda/pipeline.py` uses) with BASELINE (`intent="SIMPLE"`
  always, today's behavior) vs AFTER (the new grammar-derived intent) over the golden set.
  **Result on an 8-query graded sample: byte-identical recall@5/recall@15/mrr/table_recall@3
  in both conditions.** Traced why: only 1 of those 8 queries actually triggered a non-SIMPLE
  intent — `aggregate_mode`/`grouped_mode`'s grammar coverage turned out narrower than this
  golden set's natural phrasings (verified directly: "what is the average payment amount when
  broken down by specific currency configurations" and "what is the total paid amount across
  all payment transactions" — both clearly aggregate queries to a person — return `None` from
  both classifiers). For the one query that DID trigger AGGREGATE, the correct columns were
  apparently already ranking within top-15 under plain RRF, so the boost had nothing to move.
  **This is a genuine "no detectable effect on this small sample," not "proven no benefit" —**
  the sample (8 of 24 golden queries, only 1 with a matching grammar signal) is too thin and
  not diverse enough in aggregate/temporal phrasing to conclude either way; a real verdict
  needs either a larger/curated sample or fixing the grammar-coverage gap first. Full 24-query
  run was attempted first and abandoned after ~40 minutes — CPU-only BGE-M3 (no working Metal
  backend in this environment right now — see the recurring-outage note below) makes each
  `retrieve()` call embed multiple candidate batches at 1-3s/item, so a fair full sweep needs a
  faster environment (working Metal, or a smaller/curated aggregate-heavy query set) than what
  was available this session.
- **Kept the change** — it's mechanically correct, provably no worse than before on the tested
  sample (never regressed, since untriggered queries get `intent="SIMPLE"` exactly as before),
  and the "if no gain, remove" clause presumes an established null result, which an 8-query
  sample with 1 triggering query does not establish either way. Did NOT revert based on
  inconclusive data. Full 64/65 test suite re-confirmed clean after this change too.
- **New, separate finding**: `aggregate_mode`/`grouped_mode`'s narrow phrasing coverage limits
  how often this fix can even fire in production — worth its own look, not fixed here (outside
  P1-2's stated scope, and retuning a grammar classifier without a labelled precision/recall
  check of its own would be exactly the kind of blind change this whole exercise was trying to
  avoid).

**P1-4 (silent degrades invisible) — done.** `inference/loaders.py`'s `_STATE` gained a
`degraded` list, populated (non-fatally) at every existing warm-load try/except plus the
reranker's "returned None without raising" case — surfaced automatically through `/readyz`
(which already spreads `**state` into its response and gates only on `ready`, so this needed
no route change). Added `ingestion/m3_encoder.py::get_embed_backend()` (module-level flag,
updated at each of the 3 encode functions' metal/cpu branches) and
`retrieval_engine_phase3.py`'s `self.sparse_active` (set False only in the "SKIPPING sparse
signal" branch) — both written into a new `retrieval_health` explain-trace section
(`sparse_active`, `reranker_active`, `embed_backend`) at the top of every `retrieve()` call.
Live-verified: a direct `engine.retrieve()` call populated
`{'sparse_active': True, 'reranker_active': True, 'embed_backend': 'unset'}` in a real trace
object. Note `embed_backend` reflects the PREVIOUS encode call's backend (recorded at the
START of `retrieve()`, before this call's own encoding happens) — a deliberate best-effort
trade-off, not a per-call precise audit.

Full 64/65 test suite re-confirmed after every change in this step too.

### Step 6 (P2-2) — doc + comment corrections: DONE, 2026-09-10

Fixed every item in the fix list's P2-2 table, plus 2 more found while doing it:
- `docs/ARCHITECTURE.md` §4/§14: `FkEdge` was documented as "the join engine's FK source of
  truth" — it isn't (no writer found for it at all); the real one is the per-source
  `veda_relationship_graph.json` file. §15: `veda/routing_slm.py — empty file` doesn't exist;
  the real file is `query/routing_slm.py` (244 lines, live, imported by
  `query/source_coordinator.py`) — removed from the dead-code list entirely.
- `docs/INGESTION_AND_QUERY_PIPELINES.md` §A.5: biencoder resume note updated to describe the
  P0-3 fix (now genuinely source-scoped) and explicitly flag L3's still-flat skip check as the
  deliberately-unfixed half; both artifact-table rows for the relationship graph corrected to
  the new per-source path.
- `docs/INGESTION.md`: relationship-graph rows corrected to the per-source path; added a new
  callout on the builder's P0-2 connector-awareness (was Postgres-only, crashed on tabular
  sources); corrected the "atomic activate" callout's "flat global files" claim (5 of those
  artifacts are per-source now).
- `docs/RETRIEVAL.md` §10: removed 4 items now fixed (this session) from the dormant/dead
  table (`retrieval/__init__.py` docstring, `RetrievalEnginePhase3` "5-Signal", the dead
  `signal_builder.py` import, `graph_retriever.py`'s stale BFS comment) with a note on what
  changed; corrected the `semantic_search.py` adapter-path row to name the REAL bug that path
  had (the `ann_search` database-target fix from earlier this same day) instead of vaguely
  blaming "historically returned 0 rows"; §2's intro no longer says the class docstring is
  still wrong (it's fixed); added the P1-1 boost-only writeup with the eval-harness caveat.
- `retrieval/retrieval_engine_phase3.py`: class docstring/module header "5-Signal" → 6,
  signal list corrected (no BM25, added Signal 6, weighted not equal-weight RRF), the
  `[STEP 3/7]` log line.
- `retrieval/__init__.py`: full filename list rewritten to match what's actually in the
  directory today.
- `query/graph_retriever.py:281`: `# BFS expansion` → correctly describes the seed-collection
  step it labels, points at the real PPR section below.
- `ingestion/layers/l4_index.py`: module docstring no longer mentions BM25/ensemble encoder.
- **Also fixed, found while updating the docs above**: `veda/pipeline.py`'s own
  `[L2] Retrieval 5-signal (BGE-M3 + BM25 + ...)` log line — same staleness, just in a log
  string instead of a docstring.

### Step 7 (P2-3) — cleanup candidates: DONE, deleted with user go-ahead, 2026-09-10

All 5 items actioned after the user confirmed:
- `veda_core/data/veda_bm25_index.json` — deleted (gitignored path, doesn't show in `git status`).
- `veda_core/ingestion/chunk_linker.py` — deleted; removed its now-dangling entry from
  `ingestion/AGENTS.md`'s dead-code table.
- `inference/engine.py` — deleted; marked done in `docs/PRODUCTION_READINESS_PLAN.md` (which
  already had this exact item as an open TODO) and corrected the stale reference in
  `docs/ARCHITECTURE.md`'s dormant-code list.
- `apps/ingestion/tasks.py`'s `_MARKER_RE`/`_ENGINE_STEP_TO_STAGE`/`_apply_step_marker` (the
  dead "[N/NN] StageName" monolith-era marker protocol) — removed, along with
  `_consume_engine_output`'s now-unused third return value (`active_marker_stage`) and its
  two call sites. This had a REAL blast radius the fix list's own line item didn't mention:
  `tests/test_apps_layer_refactor.py` imported `_ENGINE_STEP_TO_STAGE` directly and 5 of its
  tests unpacked `_consume_engine_output`'s return as a 3-tuple — all updated (one test whose
  entire subject was the marker protocol removed outright; the rest adjusted to the 2-tuple
  and had their marker-specific assertions dropped). Verified all 5 by hand (pytest still
  isn't installed in these containers) — all pass.
- `.dockerignore` — added `.omc/` (already covered in `.gitignore`).

**Also found during test-suite re-verification after the deletions (unrelated to the
deletions themselves)**: the colleague's Metal embed server (`METAL_EMBED_URL`,
`192.168.1.39:11435`) is unreachable again as of this check — `curl` against it hangs with no
response at all (not even a fast refusal). `ingestion/m3_encoder.py::_metal_post()` already
has a bounded timeout (`METAL_EMBED_TIMEOUT`, default 60s) and does fail over to CPU
correctly — so this is NOT a hang/bug, just a legitimately slow (~60s per call) path whenever
that host is off the network, same class of issue as the earlier "stale IP" fix this session,
now recurring because it's a colleague's personal machine, not a stable service. Worked around
for verification by overriding `METAL_EMBED_URL=` (empty) on the test invocation only — did
not touch the real `.env`. Full 64/65 test suite re-confirmed clean once that 60s-per-call
tax was avoided.

### Step 8 (P0-5 remainder + P0-7) — DONE, 2026-09-14

The last deferred piece: the semantic model file, glossary, concept graph, domain synonyms, and
compiled registries (concepts/dimensions/metrics/MANIFEST) — the "P0-1-sized work × 5 more
artifacts" scope flagged back at Step 4. Same rigor as everything above: real writes to real
per-source paths, live-verified reads, full test suite after every change.

**New shared helper**: `config.resolve_source_artifact(name, source_id=None, tenant=None,
flat_default=None)` — extracted from the "prefer the per-source path IF it exists, else the
flat one" logic `unified_graph_builder.py` had inline for its own 6 inputs (P0-5, Step 4). Now
the ONE read-side resolver every artifact in this step uses, instead of duplicating that
fallback shape 5 more times. `unified_graph_builder.py` itself was refactored to call it too,
for consistency.

- **Semantic model, domain synonyms, concept graph, glossary**: `ingestion/semantic_layer_v2.py::
  run_full_semantic_layer()` gained `domain_synonyms_file`/`concept_graph_file`/`glossary_file`
  params (defaulting to the legacy flat constants — ctx-less calls unchanged);
  `layers/l3_enrich.py` now computes all 4 per-source paths up front and threads them through,
  including **closing the P0-3 gap left open at Step 4**: the resume-skip check
  (`os.path.exists(...)`) now checks THIS source's own semantic-model path, not the flat one
  every source used to share.
- **Compiled registries**: `semantic/compile_semantic_layer.py::compile_all()` gained
  `source_id`/`tenant` params — reads its 2 inputs (semantic model, domain synonyms) via
  `resolve_source_artifact()`, writes all 4 outputs to `source_artifact_path()` unconditionally
  when `source_id` is given. `semantic/registry.py`'s `_load_file()` — the ACTUAL gap, since the
  cache was already correctly keyed by `(source_id, tenant)` but every key loaded from the SAME
  flat file — now takes the resolved scope and calls `resolve_source_artifact()` too. Added
  `registry.invalidate_cache(source_id, tenant)` for a targeted per-source drop (`clear()` — a
  full drop — already existed and is still used by the two rehydrate paths).
- **Read-side consumers fixed to prefer the per-source copy**: `veda_hybrid.py::
  _load_semantic_model()`, `veda/runtime.py::_load_one_sm()` (the ACTUAL fallback `get_engine()`
  uses for every scope — `retrieval_engine_phase3.py`'s OWN `semantic_model_file` param turned
  out to be dead in this call path, since `get_engine()` always passes a concrete `semantic_model`
  and never lets the engine load its own), `retrieval/query_enrichment.py::QueryEnricher`
  (resolves source_id from ambient context itself, since its caller — `RetrievalEnginePhase3.
  __init__` — constructs it with no arguments), `ingestion/enrichment_index.py::
  build_enrichment_index()`'s 3 inputs.
- **`veda/validation.py::_domain_synonyms()`** had the SAME "single unscoped process-global
  cache" bug the relationship graph and every other Step 4 artifact had (not just a flat-path
  issue) — rebuilt as a per-(tenant, source) cache dict + `resolve_source_artifact()`, with a new
  `invalidate_domain_synonyms_cache()`. `query/reranker.py::_domain_synonyms()` (already routed
  through the shared config constant at Step 4, but still its OWN unscoped cache with its OWN
  separate load) now DELEGATES to `validation._domain_synonyms()`'s raw load instead of a second
  independent copy, keeping only its own lowercasing step — one load, one cache lineage, two
  shaped views.
- **Rehydrate (P0-6)**: both paths now also invalidate the 2 domain-synonyms caches above (the
  semantic model's own cache — `veda_hybrid._SM` — and the full registry clear were already
  wired at Step 4/pre-existing).

**Live-verified, real writes**: built a tiny 1-table schema through the REAL
`run_full_semantic_layer()` → `save_semantic_model()` → `compile_all()` chain for a throwaway
source id — all 8 output files (semantic model, domain synonyms, concept graph, glossary,
concepts, dimensions, metrics, MANIFEST) landed at their correct `data/default/<id>/...` paths.
Then confirmed the READ side end-to-end for that same source:
`veda.runtime._load_one_sm()`/`veda_hybrid._load_semantic_model()` returned the tiny model (not
the flat 178-table one), `semantic.registry.active()` returned that source's own 1 concept / 5
metrics, `veda.validation._domain_synonyms()` loaded without error. **No-regression check for
the LIVE source (2, homzhub)**: since none of these artifacts have a per-source copy yet for
source 2, confirmed `resolve_source_artifact()` correctly falls back to the flat file for every
one of them — `_load_semantic_model()` still returns the full 178 tables/1902 columns,
`registry.active()` still returns 167 concepts/726 metrics, unchanged from before this step.
This is the deliberate design difference from P0-1 (the relationship graph): that reader returns
an EMPTY graph when no per-source file exists (a hard requirement for the firewall to never
silently reuse another source's join graph), so P0-1 needed an immediate migration write for
source 2; `resolve_source_artifact()`'s softer "prefer scoped, else flat" contract means every
one of THESE artifacts keeps working exactly as before, with no migration step required — a
source picks up its own copies naturally the next time it's ingested.

**P0-7 — done, by deletion (option 1 of the fix list's two).** `VEDA_ARTIFACT_SCOPING`/
`VEDA_ARTIFACT_SCOPE`/`VEDA_ARTIFACT_{TENANT,SOURCE,VERSION}` and `config.artifact_scope()` were
confirmed genuinely dead in practice (not just theoretically) — `.env` never set
`VEDA_ARTIFACT_SCOPING=1` in this deployment, so the one place that could ever have set
`VEDA_ARTIFACT_SCOPE` (`apps/ingestion/tasks.py::_build_subprocess_env`) never did, and
`SourceContext.artifact_scope` (the field it populated) had zero readers anywhere in the
codebase beyond being stored. Removed entirely rather than building out the fix list's other
option (wiring `SubstrateVersion` in as the version component) — the per-source resolver from
P0-1/P0-5 needs no version segment at all, so resurrecting that complexity would just be
re-adding what this step just finished removing. `config.artifact_path()` is now simply "the
shared flat path," which is exactly what it always resolved to in this deployment and remains
`resolve_source_artifact()`'s correct fallback target. Removed the `artifact_scope` field from
`SourceContext` too (confirmed zero readers). `python manage.py check` and the full 64/65 test
suite (including the `tests/test_apps_layer_refactor.py` tests touched by the earlier P2-3
cleanup) reconfirmed clean after this removal.

**Fix list fully closed** — every P0/P1/P2 item from the original review has now been fixed,
implemented-and-measured, verified-as-already-fine, or explicitly deferred with a documented
reason (P1-2's grammar-classifier follow-up).

## Session-handoff follow-ups (2026-09-15): grammar gap, intent-boost bug, MySQL dialect

Picked up the 3 items the 2026-09-09→15 session handoff left open (its own §"What's
left", items 1–3; item 4 — standing infra risk notes — deliberately left alone).
`METAL_EMBED_URL` (`192.168.1.39:11435`) was reachable and fast again at the start of
this pass (curl `encode_query` round-trip: 0.7s, not the 60s CPU-fallback timeout) —
confirmed live before relying on it for eval runs.

### Item 3 first: full 24-query P1-2 eval, now that Metal is fast

The 2026-09-11 P1-2 write-up above ran only 8/24 golden queries (CPU-only BGE-M3 made
the full sweep take 40+ minutes and it was abandoned). With Metal reachable, one
`retrieve()` call now takes ~1.7s (was multiple seconds/batch on CPU) — the full
24-query (21 gradeable) sweep completes in ~90s. Ran it as a clean baseline BEFORE
touching any code: **BASELINE and AFTER (grammar-derived intent) came back
byte-identical** on all 4 metrics (recall@5=0.1936, recall@15=0.3734, mrr=0.4139,
table_recall@3=0.7619) — confirming the earlier 8-query finding wasn't a sampling
artifact; it reproduces on the full set too. This made items 1 (grammar gap) and the
eval-infra gap moot to investigate separately — they're the same investigation.

### Item 1: the `aggregate_mode`/`grouped_mode` grammar-coverage gap — fixed, data-backed

Per this doc's own P1-2 entry: "retuning a grammar classifier without a labelled
precision/recall check of its own would be exactly the kind of blind change this whole
exercise was trying to avoid." Built that check before touching anything:

- **`evaluation/grouping_grammar_labels.jsonl`** — 25 hand-labelled queries: true
  positives for every existing + candidate grouping phrase (including the real golden-
  set query cited in the P1-2 entry, "...when broken down by specific currency
  configurations"), and true-negative *distractors* that share the surface word "by"
  but must NOT trigger grouping (`increased by`, `sorted by`, `divided by`, `backed
  by`, `measured by`, `followed by`, `given by`, `accompanied by`, `differ by`, plus
  the pre-existing ratio-wording case).
- **`scripts/eval_grouping_grammar.py`** — runs `veda.planning.grouped_mode()` over
  that set against BASELINE `QUERY_GRAMMAR["grouping"]` vs a CANDIDATE list (never
  mutates `config.py` itself). Result: BASELINE precision=1.0 recall=0.417 (7/12 false
  negatives, all "broken down by"/"broken out by"/"break down"/"split by"/"segmented
  by"/"categorized by" phrasings); CANDIDATE (adding exactly those 6 phrasings)
  precision=1.0 recall=1.0 — **zero new false positives against the distractor set**.
- **Applied**: `veda_core/config.py`'s `QUERY_GRAMMAR["grouping"]` now includes
  `"broken down by", "broken out by", "break down", "split by", "segmented by",
  "categorized by"` alongside the original `per/each/grouped by/breakdown`. Bare "by"
  deliberately never added (that's exactly what would catch the distractors).
- **Verified no regression**: hand-invoked all 15 parametrized assertions from
  `tests/test_grouped_aggregation_operators.py` (pytest still isn't installed in these
  containers) — all pass unchanged. Full 64/65 routing suite re-confirmed.
- **Confirmed live**: the exact golden-set query now classifies `AGGREGATE` instead of
  `SIMPLE` (`aggregate_mode`/`grouped_mode` now fire on it), as intended.

### The deeper bug this uncovered: `IntentBooster._get_column_metadata` always returned `{}`

Re-ran the full P1-2 eval after the grammar fix, expecting a change — **got the exact
same byte-identical numbers again.** Traced why by diffing per-column rankings for the
one query that now triggers AGGREGATE: SIMPLE and AGGREGATE intent produced **identical
ranked output**, not just identical aggregate metrics. `retrieval/intent_boosting.py`'s
`IntentBooster._get_column_metadata()` walks `semantic_model["tables"][t]["columns"][c]`
— but no table entry in the real `veda_semantic_model.json` HAS a `"columns"` sub-dict
(confirmed live: a real table entry's keys are `table_name/business_purpose/
primary_entity/table_type/candidate_temporal_columns/candidate_measure_columns`).
Column metadata (`analytics_role`, the field every `boost_*` method reads) actually
lives in the model's own **top-level flat `columns` dict**, keyed `"table.column"`
(1902 entries for source 2 — confirmed both `currency_id`→`IDENTIFIER` and
`paid_amount`→`MEASURE` exist there with the expected roles). So `_get_column_metadata`
always returned `{}`, `role` was always `""`, and `boost_aggregate`/`boost_temporal`/
`boost_multi_table` have been **unconditional no-ops for every intent, always** — not
a grammar-coverage problem at all; the boost could never have fired even with perfect
grammar coverage. This is almost certainly the REAL reason the 2026-09-11 8-query eval
(and this session's own first 24-query re-run) showed zero effect.

**Fixed**: `_get_column_metadata` now reads the real flat `columns` dict first (O(1)
lookup, was an O(tables×columns) nested walk), falling back to the old nested-walk
shape only if the flat lookup misses (back-compat for any differently-shaped model).
No dedicated tests existed for `intent_boosting.py` (checked: zero test files reference
it). Live-verified: re-ran the diagnostic query — `paid_amount` (MEASURE) jumped from
rank 12 to rank 2 under AGGREGATE intent; `currency_id` (IDENTIFIER) dropped out of the
top 15 (the `-0.40`-scaled IDENTIFIER penalty firing as designed — a real trade-off,
not a bug, discussed below).

**Definitive P1-2 eval, both fixes applied, full 24-query set, Metal fast:**

| metric | BASELINE (intent=SIMPLE) | AFTER (grammar-derived intent) | Δ |
|---|---|---|---|
| recall@5 | 0.1936 | 0.2571 | **+33% relative** |
| recall@15 | 0.3734 | 0.3907 | +5% relative |
| mrr | 0.4139 | 0.4790 | **+16% relative** |
| table_recall@3 | 0.7619 | 0.7143 | **−6% relative** |

A genuine, non-identical, mixed result — not an unambiguous win. The regression on
table_recall@3 is explainable, not a bug: `boost_aggregate`'s IDENTIFIER penalty
(`-0.40`, by design — "don't aggregate IDs") demotes columns like `currency_id` even
when the query actually wants that column as a GROUP BY dimension, not something to
sum — a real tension between "IDENTIFIER" and "grouping dimension" that this fix
surfaces but does not resolve. Full 64/65 routing suite re-confirmed clean after this
change too. Given this doc's own standing instruction ("did NOT revert based on
inconclusive data" — the 2026-09-11 entry), and that this result is no longer
inconclusive but genuinely mixed, flagging as **kept, not further tuned this pass** —
the IDENTIFIER-vs-grouping-dimension conflict is a real follow-up, not a blind
revert-or-keep call to make from a 21-query sample.

### Item 2: non-Postgres SQL dialect support — MySQL added and live-verified

The gap: `_can_sql_introspect()` gated full SQL introspection (PK detection,
cardinality, polymorphic-edge correlation) to Postgres only, "since there is no live
[non-Postgres] one to verify dialect-specific SQL against." Discovered
`connectors/relational.py::MySQLConnector` already existed (dialect-correct
information_schema queries, backtick quoting) for **L1 schema extraction** — but
`ingestion/relationship_graph.py` never used it; it hardcoded its own `psycopg2`
connection and Postgres-only SQL (double-quoted identifiers, `::text` casts, `= ANY(%s)`
array binds) for PK/cardinality/polymorphic detection specifically.

Stood up a throwaway MySQL 8 container (`veda-test-mysql`, on `veda-platform_veda_net`,
removed after verification) with a small `customers`/`orders` schema (real FK,
`orders.customer_id → customers.id`, multiple orders per customer) to get the "live
source to verify against" this gap always lacked.

**Found and fixed 3 real bugs along the way, not just added new code:**

1. **`mysql-connector-python` was never installed anywhere in this deployment** —
   `MySQLConnector` has depended on it since it was written, but it's absent from
   every `requirements/*.txt`. Added to `requirements/inference.txt` (used by both
   `inference` and `ingest-worker`, per `docker-compose.yml`'s shared
   `Dockerfile.inference`) and `requirements/host-ingest.txt`.
2. **`connectors/relational.py::RelationalConnector.connect()`'s own health-check
   ping never drained its `SELECT 1` result before closing the cursor.** Harmless on
   psycopg2/sqlite3; fatal on mysql-connector-python's C extension, which leaves the
   whole *connection* (not just that cursor) flagged "has unread result" until
   something fetches it — so the very next `get_schema()` call blew up immediately on
   `self._conn.cursor()` with `InternalError: Unread result found`, before running any
   real query. This is a real, previously-unexercised bug in the shared connector base
   class (not specific to my changes) — never triggered before because MySQL was never
   actually runnable. Fixed: drain via `cur.fetchall()` before `cur.close()`.
3. **`ingestion/relationship_graph.py` dialect support** — added `_engine_of()`,
   `_ident()` (backtick vs double-quote), `_cast_text()` (`CAST(x AS CHAR)` vs
   `x::text`), `_in_clause()` (dialect-neutral `IN (%s,%s,...)`, replacing Postgres-only
   `= ANY(%s)` array binding — not supported by mysql-connector-python or most other
   DB-API drivers) — threaded through `_conn` (now dispatches `mysql.connector.connect`
   vs `psycopg2.connect` on `ctx.engine`), `_table_meta`, `_cardinality`,
   `_polymorphic_edges`. `_can_sql_introspect()` now accepts `mysql` (still declared-
   FK-only for anything else — SQL Server, Oracle, etc. — no live instance to verify
   against yet). Also fixed the schema default: MySQL has no `"public"` schema —
   `information_schema.*`'s "schema" filter IS the database name there (mirrors
   `MySQLConnector`'s own `db = schema or dbname` fallback); Postgres's `"public"`
   default is unchanged.

**Live-verified against the throwaway MySQL container**
(`scripts/verify_mysql_relationship_graph.py`, not a permanent test — a one-off repro
script, kept for reference): `build_relationship_graph()` returned `mode: "sql"` (full
introspection, not the declared-FK-only fallback), found exactly the 1 real FK edge
(`orders.customer_id → customers.id`), and computed `cardinality: "N:1"` correctly from
real data correlation (2 customers have multiple orders, 1 doesn't — the distinctness
check landed on the right answer).

**No regression on the real Postgres path**: rebuilt source 2's (homzhub) graph through
the same patched code — **178 tables / 609 edges / 0 polymorphic, byte-identical** to
the pre-existing stats recorded in this doc's P0-1/P0-2 section above. Full 64/65
routing suite re-confirmed clean.

**Still not done** (out of scope for this pass, same as before): SQL Server, Oracle, or
any other non-Postgres/non-MySQL relational dialect — still declared-FK-only, still no
live instance available to verify against. `mysql-connector-python` is a live pip
install in the running `inference`/`ingest-worker` containers for this session (now
also tracked in `requirements/inference.txt`/`host-ingest.txt` for the next image
rebuild — not yet baked into a rebuilt image, since these containers weren't rebuilt
this pass).

## Query understanding: why new phrasings keep needing individual fixes — see its own doc

A live chat-API investigation the same day (2026-09-15, prompted by a user asking why a
chat follow-up silently dropped a filter) surfaced a bigger, structural finding: SQL
planning (`veda/pipeline.py`) is a fixed chain of narrow regex/keyword-triggered
branches, each covering one surface phrasing — not a real semantic-parsing layer. Every
fix in this doc's P1-2/grammar-gap sections above is an instance of that pattern (one
more trigger added). The codebase already has the right architecture for a real fix
(`veda_core/veda/understanding/` + `veda_core/veda/analytical_spec.py` — LLM concept
extraction → deterministic schema-grounding firewall → the same SQL builder), flag-gated
off and incomplete (no filter/dimension grounding yet), plus a live-tested regression
(enabling it as-is broke 2 working queries while fixing 1). Full write-up, flow diagrams,
the regression evidence table, and a phased roadmap:
**`docs/backlog/QUERY_UNDERSTANDING_GAPS_AND_ROADMAP.md`**.

## Compose note (unrelated, found while verifying the above)

The local `pg_data` volume (494 MB, created 2026-07-05) is PG16-formatted; `docker-compose.yml`
had been bumped to `pgvector/pgvector:pg17`, which refuses to start against an older-major
data directory. Pinned `docker-compose.yml`'s `postgres` image back to `pg16` on
2026-09-10 (user's explicit choice — see the comment at that line) so the existing local data
survives. Re-bump to pg17 via a proper `pg_dumpall`-and-restore (or `pg_upgrade`) when
convenient; `docker-compose.demo.yml:30` still says `pg17` and has the same latent issue if
that override is ever used against this volume.

## M1 close-out pass (2026-09-16): verified-query cache key must become IR-shape-aware

Found while widening the per-source battery (`scripts/eval_per_source_battery.py`, 63
questions / 4 sources with expected SQL shapes): the `VerifiedQueryCache` key is the
normalised question text only. Two consequences observed live, not hypothesised:

1. **Scalar-vs-grouped replay.** A cached scalar answer for "how many maintenance records"
   replayed for "how many maintenance records per vendor"; a cached `LIMIT 1` superlative
   ("which vendor has the highest rating") replayed for "top 3 vendors by rating". The
   three existing demotions (evidence, qualifier, shape) catch some of this; this pass
   **extended the existing shape demotion** to compare grouping ↔ `GROUP BY`, aggregate
   presence, and `LIMIT N` (no fourth heuristic was added — the user's instruction — but
   the extension is a stop-gap and is disclosed here as such).
2. **Poisoning by test traffic.** Wrong-shaped answers produced while the code was in
   flux were cached as "verified" and replayed on every later run. Purged by hand:
   13 rows for sources 4/5 and 2 rows for source 2 created 2026-09-15/16; the 26
   pre-existing source-2 rows (July) were left alone.

**Required in M2/M6:** the cache key must include the grounded IR shape (anchor table,
aggregate, group dimensions, filter columns+ops, ranking N / direction, time bucket), not
the question text — then a scalar and a grouped variant of the same words can never
collide, and demotion heuristics on the SQL text become unnecessary. Until then the
battery carries the scalar-vs-grouped replay case as a regression check (source 4:
"how many maintenance records are there" followed by "… per vendor"; "which vendor has
the highest rating" followed by "top 3 vendors by rating").

### Same pass — Tier-2 envelope contract gate (`veda_hybrid.py::_envelope_inexpressible`)

The frozen intent envelope (`INTENT_ENVELOPE_CONTRACT.md` v1) has no ranking intent and
`eq|ne` filters only. Two battery questions on source 2 were answered with the *nearest
expressible* shape instead of falling through: "list top 5 properties by monthly rent" →
a monthly `DATE_TRUNC` count trend (LIMIT 100), "properties with more than 3 floors" →
`WHERE total_floors = 3`. The envelope path now skips when the question carries a ranking
(`query/ranking_parser`), a numeric threshold, or a negation (the phrase sets
`query/operation_classifier` already uses for the same reason on the cross-source path)
and lets the IR path rank/compare or refuse typed. Comparators and rankings inside the
envelope itself are M2 grounding work (typed filter ops), not a contract v1 patch.

### Same pass — `ingestion/biencoder.py` L4 embed read no semantic model at all

`_load_retrieval_docs()` / `_load_table_purposes()` were called ctx-less; once the flat
`SEMANTIC_MODEL_FILE` fallback was removed (M1), that resolved to no artifact, so every
L4 embedded the structural passage only. Now passed `source_id` + ambient tenant. Takes
effect on the next ingest after `docker compose restart ingest-worker` (prefork child
caches modules); existing embeddings for full-model sources (2) were built before the
fallback removal and are unaffected, lite-model sources (3/4/5) have no
`retrieval_documents` either way.

### Same pass — the shared-planner Tier-2 branch ran without `_tier2_validate`

`veda_hybrid._tier2_sql`'s multi-entity branch (`answered via SHARED planner`) applied only
the AST firewall; the single-table branch below it ran `_tier2_validate` (value grounding,
strict qualifier completeness, IR equivalence). Live consequence: "properties with more
than 3 floors" executed as `… HAVING COUNT(listing reviews) > 3` — a graph-verified join
answering a different question — and "show tickets with high priority" executed with
`high` dropped (an unfiltered ticket list; this copy's `worklists_ticket.priority` holds
only LOW 223 / MEDIUM 8, so the only honest outcome is a typed refusal). Both branches now
run the same gates; the reason feeds the existing repair hint before refusing.

### Same pass — battery semantics added: `expect="refuse"` and `xfail`

`expect="refuse"` makes an *answer* the failure for a question whose value does not exist
in the data; `xfail` carries the verified-cache similarity-replay case ("which vendor has
the highest rating" → cached "top 3 vendors by rating", cosine 0.86 → `LIMIT 3`) as a
documented failing case reported separately — XPASS fires when the IR-shape-aware key
(above) lands, so the marker gets removed rather than forgotten.
