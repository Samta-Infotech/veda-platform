# Architecture review (2026-09-10 snapshot) — reconciled against the live tree, 2026-09-15

**What.** An external review ("every registered source, drill-down, one session") decomposed
the goal into five properties (P1–P5) and a migration order (M0–M6). It was written against
the **2026-09-10 snapshot** — before the P0-1…P0-7 artifact-scoping pass, the P1-1/P1-2/P1-4
retrieval fixes (`docs/backlog/query-engine-open-items.md`), and the 2026-09-15 work
(`SESSION_HANDOFF_2026-09.md` §8–§10). Several of its "✗ Broken" findings were already fixed;
others are confirmed open. This doc records, per claim, what is true *now*, with evidence,
and the amended order of work. The review's own text is not reproduced — its structure is.

**Basis.** Live greps + a per-source behavioural battery against the running stack
(source 2 relational, 3 document, 4 csv, 5 parquet), 2026-09-15. `scripts/
eval_per_source_battery.py` is the battery, kept as the M0 seed.

---

## 0. The five properties — status now

| # | Property | Review said | Now | Evidence |
|---|---|---|---|---|
| **P1** Source isolation | ✗ Broken | **Mostly fixed 2026-09-10/14; two real residuals found and fixed 2026-09-15** | §1 |
| **P2** Uniform safety across SQL heads | ✗ Two safety levels | **Confirmed open** — the federated head calls none of value-grounding / qualifier-completeness / alignment guards; three SLM-authored planners | §2 |
| **P3** Meaning-first planning | ✗ Regex chain | **Confirmed open** — full diagnosis + roadmap in `QUERY_UNDERSTANDING_GAPS_AND_ROADMAP.md` | §3 |
| **P4** Structured conversation state | ✗ Memory drops source | **Confirmed open** — `QueryFrame` has no `source_id`; default scope = all ready sources | §4 |
| **P5** Observable degradation | ✗ Four silent degrades | **Mostly fixed** (P1-4 `degraded` + `retrieval_health`; P1-1 boost-only S3/S4; P1-2 intent boost now live) — `required_for_ready` per source kind still open | §5 |

## 0.1 Registered sources (the review's "what I need" item 3)

| id | name | kind | state | notes |
|---|---|---|---|---|
| 1 | launchpad | relational/postgres | **not ready**, no host | registered only; excluded from every battery |
| 2 | homzhub | relational/postgres | ready | 178 tables / 1902 cols; the source the flat legacy artifacts were built from |
| 3 | docs_contracts | filesystem/document | ready | 5 files → 8 chunks |
| 4 | invoices_csv | csv_lake | ready | `maintenance`, `vendors` (9 cols) |
| 5 | catalog_parquet | parquet | ready | `amenities_catalog` (4 cols) |

No NoSQL source is registered → item 6 resolves itself: relational + tabular + document is
the bar for this milestone.

---

## 1. P1 — what the review saw vs. what's live

Review claims, each checked 2026-09-15:

| Claim (2026-09-10) | Now |
|---|---|
| `relationship_graph.py` / `join_planner.py` hardcode the flat graph path; consumers read homzhub's graph for every source | **Fixed** (P0-1): `config.source_artifact_path()` + `veda.runtime.get_graph()` as the one accessor; a source with no graph gets an EMPTY graph, never another's. Verified 178/609 for source 2, per-source paths for others |
| Builder Postgres/`public`-only; tabular and non-Postgres never get a graph | **Fixed** (P0-2 + 2026-09-15): connector-aware schema fetch, ANSI PK detection, declared-FK-only for unverified dialects, **MySQL dialect live-verified** |
| `fast_path._sm()` reads the flat file | **Fixed** (P0-5): delegates to `veda_hybrid._load_semantic_model()` |
| Resume checks not source-scoped | **Fixed** (P0-3): `WHERE source_id = %s`; L3 checks its own per-source path |
| Rehydrate doesn't clear graph/rerank/joinpath/unified caches | **Fixed** (P0-6): every artifact has an `invalidate_*_cache()` wired into both rehydrate paths |
| `VEDA_ARTIFACT_SCOPING` half-wired | **Fixed by deletion** (P0-7) |
| "DB-backed artifacts are scoped correctly" | **Partly wrong in the other direction**: the *engine-store* tables are scoped and clean (`graph_nodes` for source 4 = 2 table + 9 column nodes), but the **Django mirror** (`apps.substrate.GraphNode/GraphEdge`) holds ~2133/5463 rows under sources 3/4/5 — homzhub's graph written under each id in July. Django-side readers only; not on the query path. Open, low priority |

**Two residuals the review could not have seen, found and fixed 2026-09-15:**

1. **The flat fallback served homzhub's substrate to every non-owner source.**
   `config.resolve_source_artifact()` was "scoped if it exists, else flat". Sources 3/4/5 were
   ingested 2026-07-08, before scoping, so they had no scoped copies — and `_load_semantic_model`
   for source 4 returned homzhub's **178-table** model; every artifact resolved to
   `data/veda_*.json`. Effect, measured by the battery: source 5 planned `FROM "assets_amenity"`
   (a homzhub table) on a parquet source whose only table is `amenities_catalog`; source 4's
   candidate set included `worklists_quote`, `services_valuebundle…`. **Fix**: the flat fallback
   is now owner-only — a source may receive the flat files only if every table it owns in
   `column_embeddings_v2` exists in the flat semantic model (data-derived, cached, permissive
   with a warning on DB error). `_load_semantic_model` degrades a missing model to an empty,
   tagged one; the SQL head reports the new honest status **`not_materialized`** ("re-run
   ingestion") instead of the misleading `access_denied` it used to emit.
   `scripts/backfill_semantic_model.py`'s own docstring already recorded the history: *"the
   tabular/doc ingestion never built a per-source semantic model, so the global homzhub model
   got persisted + published under sources 3/4/5."*
2. **`semantic_layer_v2.py:660`** still reads the flat relationship graph through a process-global
   cache (ingestion-time, single-source-per-process — feeds homzhub's graph into a tabular
   source's enrichment). Not yet changed; small.

**Remediation in flight (M1 exit test = the real thing):** re-ingest sources 3/4/5 under the
scoped pipeline so they get their own artifacts. Source 4 re-ingested 2026-09-15 as the canary
(`IngestionJob` 20); 5 and 3 next. The `ingest-worker` container was created 2026-09-10 with the
stale `METAL_EMBED_URL` (`.43`; `.env` says `.39`) and pays a 60 s timeout per embed call — must be
recreated (`docker compose up -d ingest-worker`) before the next ingest.

**Still open under P1:** `required_for_ready` per source kind (a relational source with no join
graph must not be `ready`); the Django-mirror rows above; the `semantic_layer_v2` reader.

---

## 2. P2 — confirmed open, exactly as described

`grep` for `value_grounding|qualifier_completeness|alignment_ok|aggregate_presence_ok` across
`query/federated_route.py`, `cross_source_composer.py`, `federated_executor.py` → no hits. Three
SLM planners: `_generate_federated_sql` (asks for raw SQL text), `_generate_federated_plan`,
`_generate_structured_plan`. This is the mechanism behind the 2026-09-15 chat finding
(`SESSION_HANDOFF_2026-09.md` §9.2): the follow-up went to the federated head because the chat
request pinned no source (§4), and the filter was dropped with a `for_sale_count` alias left on.

The review's fix shape is right and is adopted as-is: one compiler from a typed intent, two
emitters, one firewall operating on (intent, SQL). Depends on M2 (the intent exists first).

---

## 3. P3 — confirmed open; see the dedicated doc

`docs/backlog/QUERY_UNDERSTANDING_GAPS_AND_ROADMAP.md` (2026-09-15) has the branch-chain
diagram, the four partial intent representations, the nine SQL builders, the live regression
table for the flag-gated understanding layer, and the phased plan. The review adds two leaks
worth recording here: `derive_spec` silently drops an unresolvable GROUP BY
(`analytical_spec.py:117-123`); `numeric=True` on `_resolve_column` is never read; a
`GroundedIntent` overrides ER-V1's anchor even when it then declines to emit SQL. All folded
into M2.

**2026-09-15 addendum — a regression of the point-fix kind the review warns about:** the new
`_bare_count` branch (`pipeline.py`) hijacked *grouped* counts ("how many maintenance records
per vendor") into a scalar `COUNT(*)`; caught by the battery the same day, fixed with a
grouping-word guard. Recorded because it is the pattern: every regex branch added is a new
place for the next phrasing to go wrong. M6 (retire the chain) is the actual fix.

---

## 4. P4 — confirmed open

`chatbot/memory/frame.py::QueryFrame` has no `source_id`/`source_ids`; `render_frame_as_query`
re-stringifies to `"<message> (for <entity, filters>)"`; `apps/query/scope.py:90` defaults an
unpinned request to **all ready sources**. Verified live 2026-09-15 that the frame/classify/
context-resolve machinery itself works correctly (checkpoint-confirmed `delta_type=refine`,
correct composed query) — the defect is what the frame *remembers* (no scope) and how it
*replays* (text, re-routed from scratch). The review's IR-stack design is adopted; depends on M3.

---

## 5. P5 — mostly done; residuals

| Review item | Now |
|---|---|
| Reranker missing → pure RRF, silent | `retrieval_health.reranker_active` in every explain trace; `/readyz.degraded` (P1-4) |
| Signals 3/4 give hub tables a query-independent boost | Boost-only, not candidate-generating (P1-1) |
| Tier-1 hardcodes `intent="SIMPLE"` so intent boosting never fires | Grammar-derived retrieval intent (P1-2) **and** the `IntentBooster` metadata-shape bug fixed 2026-09-15 — boost measurably fires now (recall@5 +33%, MRR +16%, table_recall@3 −6%) |
| Sparse Signal 2 skipped silently | `retrieval_health.sparse_active` |
| Metal unreachable → 60 s CPU fallback | `retrieval_health.embed_backend`; **still bites ingestion** via the stale worker env (§1) |
| Graph build fails non-fatally for non-Postgres | Non-Postgres now builds (MySQL verified); tabular declared-FK-only |
| `required_for_ready` per source kind | **Open** |
| Tune fusion weights with the reranker off | **Open** |

---

## 6. Battery baseline (the M0 seed), 2026-09-15

Same-shaped questions pinned per source, before vs. after the day's hardenings:

| Source | Before | After |
|---|---|---|
| 2 relational | 3/3 answered | 3/3 answered, 0 foreign tables |
| 3 document | RAG fine; a count-shaped question crashed with `OperationalError` (SQL head against a hostless source) | 3/3 answered on RAG, 0 crashes |
| 4 csv | 1/3; `list vendors` → `SELECT FROM "vendors"` (empty projection, homzhub model); homzhub tables in candidates | **re-ingested (job 20)**: 3/3 own-table SQL; the grouped question (`per vendor`) → honest `clarify` |
| 5 parquet | 0/3; SQL against homzhub's `assets_amenity` | **re-ingested (job 21)**: `FROM "amenities_catalog"`, correct columns; grouped AVG → honest `clarify` |

**Final state (after all three re-ingests, same day):** `scripts/eval_per_source_battery.py
--sources 2,3,4,5` → **OK, 0 failures** — 0 foreign tables, 0 crashes, 0 silently-dropped
groupings. Confirmed through the live HTTP chat API with `source_ids` pinned (the warm
server, i.e. the rehydrate fan-out worked): source 5 → 7 rows from `amenities_catalog`,
source 4 → 6 rows from `vendors`. Redis `veda:sm:{3,4,5}:default` published (`2.0-lite`;
source 3's is the correct empty model). Re-ingests on the recreated worker took ≈30 s each
(vs. 197 s for the canary on the stale Metal URL).

**One more generic bug the battery surfaced, fixed the same day:** the verified-query cache
replayed a scalar `COUNT(*)` for a *grouped* question ("…per vendor", similarity 0.88 to the
cached "how many maintenance records are there"). The cache lane had evidence and qualifier
demotion guards but no **shape** guard. Added a third demotion in `pipeline.py` (grouping
phrase ↔ `GROUP BY` mismatch, or scalar-aggregate wording with an aggregate-less cached SQL,
reusing `QUERY_GRAMMAR["grouping"]` + `aggregate_presence_ok`). The battery now asserts the
same shape invariant, so this class can't silently return.

**What "clarify" on the two grouped questions means:** the lite (`2.0-lite`) models built
by the tabular pipeline carry no `semantic_type`/`analytics_role` metadata, so the grouped
planner can't pick a display dimension and refuses rather than guess. That is P3/M2
territory (grounding by type), not an isolation problem — the honest outcome for now.

Hardenings landed 2026-09-15 (all uncommitted): `config.resolve_source_artifact` owner rule +
`_flat_artifact_owner_ok`; `veda_hybrid._load_semantic_model` empty-model path;
`veda_hybrid._scope_has_structured_source` + `hybrid` only with a structured source in scope;
`veda/execution.py` connection acquisition inside the error contract (+ hostless refusal);
`veda/feedback.py` + `veda_hybrid` `not_materialized` status; `pipeline.py` `_bare_count`
grouping guard. Routing suite 64/65 unchanged after each.

### 6.1 M1 close-out, 2026-09-16 — battery widened to 63 questions, shape-asserted

The 2026-09-15 baseline above asserted "no foreign tables / no crash / no dropped
grouping" on 3 questions per source. Widened to ≥15 per source (63 total) with an expected
shape per question — see the script's docstring for the full assertion list. What that
widening found, and what changed (all uncommitted; details in
`query-engine-open-items.md`, "M1 close-out pass"):

| Found by the battery | Root cause (generic) | Fix |
|---|---|---|
| lite sources: value filter answered as an unfiltered list | value mirror / `column_values` not scoped to the source's graph table ids; parquet probe off without HTTP profiles | `value_resolver._scope_table_ids`, `resolution`, `datalake_values` → `resolve_surface` |
| "…per vendor" on csv → clarify | no discovered-FK edge in the relationship graph; no grouped-COUNT branch | `relationship_graph._discovered_fk_edges`, `planning.grouped_count_mode`, `superlative_plan` COUNT |
| "top 3 … by rating" refused on lite models | no ranked-metric deterministic branch | `pipeline` `ranked_metric_only` |
| retrieval saw other sources' embeddings | Signal 1+2 threads lost the request context | `copy_context().run` in `retrieval_engine_phase3` |
| "top 5 by monthly rent" → monthly trend; "more than 3 floors" → `= 3` | Tier-2 envelope contract can't express ranking/thresholds and answered the nearest shape | `veda_hybrid._envelope_inexpressible` gate + `_tier2_validate` dropped-constraint check |
| "more than 3 floors" → "assets with >3 listing reviews" | shared-planner Tier-2 branch skipped `_tier2_validate` | gated |
| "show tickets with high priority" *answered* | there is no HIGH in this copy (LOW 223 / MEDIUM 8); the qualifier was dropped | battery `expect="refuse"`; the gate above refuses |
| "which vendor has the highest rating" → `LIMIT 3` | verified-cache cosine replay (0.86) of "top 3 vendors by rating" | **not** a fourth demotion: battery `xfail`; cache key → IR-shape-aware in M2/M6 |
| scalar replayed for grouped; wrong test answers cached | text-keyed cache; poisoning by test traffic | existing shape demotion extended with `LIMIT N` (disclosed); 15 poisoned rows purged |

**Final line (2026-09-16):** `--sources 2,3,4,5` → `{"summary": "OK", "questions": 63,
"failures": 0, "known_gap_warns": 12, "xfail": 1, "xpass": 0}` — typed-gap WARNs on
the documented M2 items (ungrouped SUM/AVG, COUNT DISTINCT of a dimension, numeric
comparators through the deterministic head, value filters that need type metadata), 1
xfail (cache replay). Routing suite 64/65 (the 1: `test_source_coordinator.py`
`test_dispatch_with_adapter_flag_on_matches_flag_off_exactly` — `ModuleNotFoundError:
query.source_adapters`, dead adapter-flag path, unchanged).

**Note on §6's "owner rule":** superseded the next day — `resolve_source_artifact` now has
**no** flat fallback at all (scoped path or `None`; `_flat_artifact_owner_ok` removed).

---

## 7. Amended migration order

The review's dependency logic holds; the content of M1 and M5 shrinks to residuals.

| Step | Now |
|---|---|
| **M0 Eval harness as the gate** | **Started**: `scripts/eval_per_source_battery.py` (contamination/crash gate, per source); `evaluation/golden_queries.jsonl` + `scripts/eval_p1_2_intent_boost.py` (quality); `scripts/eval_grouping_grammar.py` (grammar). Missing: multi-turn session scripts asserting scope/filters per turn; expected-IR assertions (needs M2). |
| **M1 Source substrate** | **Closed 2026-09-16** (§6.1): resolver scoped-only (no flat fallback), all readers None-safe; 3/4/5 verified clean of homzhub names; lite sources answer the same shapes; Django mirror confirmed non-authoritative and writer scoped; 63-question shape-asserted battery OK. Residuals carried to M2/M5: `required_for_ready`; cache key IR-shape-aware (M2/M6); `docker compose up -d inference` for the `OLLAMA_URL` drift. |
| **M2 Query IR + grounding** | Open. Filter/dimension/time grounding with type + value checks; fix the authority leaks; IR advisory-only. Independent of M1's residuals — can start now. |
| **M3 One compiler, IR firewall** | Open; depends on M2. Closes P2. |
| **M4 IR-stack memory** | Open; depends on M3. Closes P4. |
| **M5 Degradation contract** | Residuals only: `required_for_ready`, fusion weights with reranker off. |
| **M6 Retire the regex chain** | Per branch, when M0 shows the IR path subsumes it. |

**Rule to keep, in the review's words:** *a new decision layer never gates a strictly more
capable path on a strictly weaker signal.* It has now bitten twice (coordinator authoritative
mode; the understanding layer's terminal `Refusal`) and was avoided once (the owner rule is
permissive-with-warning on any error).

---

## 8. The review's "what I need from you" — answered

1. `tests/` — present in the tree (100 files); the routing family runs standalone (no pytest in
   the containers), 64/65 baseline documented in `SESSION_HANDOFF_2026-09.md`.
2. Handoff + open-items docs — `docs/backlog/SESSION_HANDOFF_2026-09.md`,
   `docs/backlog/query-engine-open-items.md`.
3. Source list — §0.1.
4. Failed/wrong-answer queries — §6 plus `SESSION_HANDOFF_2026-09.md` §9 (chat transcripts);
   explain traces are in the inference log for each.
5. The 2026-09-15 working tree — this repo's uncommitted changes; line references in this doc
   are against it.
6. NoSQL scope — none registered; out of this milestone.
7. Eval environment — the bind-mounted clone at `/Users/samta/veda-platform` is the live stack;
   `pg_dump` not taken (not needed for the battery, which runs in place).
