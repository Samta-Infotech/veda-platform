# Session handoff — VEDA platform work log (2026-09-09 → 2026-09-15)

Written so a fresh chat session can pick this repo up with full context. Everything through
§7 below is **already committed** on branch `feat/refinements-pipeline` (the user committed it
personally between turns — commits `db18171` "md files add" and `e4fbc54` "changes"; nothing in
that part of the session was committed by the assistant, per this repo's `CLAUDE.md` rule below).
**§8 and §9 (2026-09-15, a follow-up session) were committed by the user** as `ec15c91` "fixes"
and `3a2d2fa` "pipeline fix" (14:41 IST). **§10 and §11 are NOT committed** — `QUERY_UNDERSTANDING_
GAPS_AND_ROADMAP.md`, `ARCH_REVIEW_2026-09_RECONCILED.md`, `scripts/eval_per_source_battery.py`
(all new), plus §11's edits to `config.py` / `veda_hybrid.py` / `veda/execution.py` /
`veda/feedback.py` / `veda/pipeline.py` and this file — plus a few live DB rows (an RBAC grant, a
DRF token, three re-ingested sources) and one recreated container — still sitting as uncommitted
changes in the
working tree.

## The one rule that matters most

**Never run `git commit` or `git push`**, and never `git reset`/`rebase`/`merge`/`stash`/
`checkout <branch>`/`checkout -- <path>` without being asked for that exact command. Leave
finished work as uncommitted changes and report what changed; committing is the user's call.
This is written into the repo's own `CLAUDE.md` — read it first in any new session.

## What this repo is

VEDA: a Django/DRF platform wrapping a preserved NL→SQL engine (`veda_core`). Two-tier runtime —
a thin Django `api` tier (no `veda_core` import, HTTP only) and a warm FastAPI `inference` tier
(one engine per worker per `(source, tenant)` scope) that actually runs both the ingestion and
query pipelines. See `docs/INGESTION_AND_QUERY_PIPELINES.md` (new this session, see below) for
the full walkthrough — that's the best single starting point for a fresh session.

Key environment facts (also in `CLAUDE.md`):
- Docker bind-mounts `.:/app`; editing a file changes what's live after a restart/reload (the
  inference container runs `uvicorn --reload`, so most edits hot-reload — but reloading is
  triggered on ANY file write anywhere under `/app`, including scratch/eval scripts, and rapid
  successive edits can pile up reload cycles ~40-60s apart while the engine rewarms models on
  CPU; see the "environment gotchas" section below).
- `.env` is gitignored; two settings there matter (`SLM_MODEL_NAME`, `SLM_TEMPERATURE`) plus
  `METAL_EMBED_URL` (see gotchas).
- Two Postgres databases: `veda` (Django tables) and `veda_engine` (the engine's pgvector
  tables — `column_embeddings_v2`, `graph_node_embeddings`, `doc_chunks`, …). Engine tables are
  reached via `VEDA_INTERNAL_*`, never Django's `connection`.
- Docker's postgres image is pinned to `pgvector/pgvector:pg16` (not pg17) because the local
  `pg_data` volume is PG16-formatted — see `CLAUDE.md` for the re-upgrade path if ever needed.

---

## Session timeline (chronological)

### 1. Full repo documentation refresh
Read the whole repo layer by layer and rebuilt the docs tree: `docs/ARCHITECTURE.md`,
`docs/QUERY_ENGINE.md`, `docs/RETRIEVAL.md`, `docs/INGESTION.md`, `docs/RBAC.md`,
`docs/CHAT.md`, `docs/OBSERVABILITY.md`, `docs/EVALUATION.md`, ~29 per-directory `AGENTS.md`
files, `docs/archive/` (moved 19 stale docs there with `git mv` + an index), and
`docs/adr/0001-rbac-resource-path.md`. This was the foundation everything else built on.

### 2. Fixed `storage_adapters/reader.py::ann_search` — wrong database target
**The bug**: `ann_search` queried `column_embeddings_v2` (lives in `veda_engine`) through the
Django-targeted connection (`veda` DB) — the table doesn't exist there, so Postgres raised
"relation does not exist," which was silently swallowed, and the caller got zero rows.
**The fix**: added `_internal_connection()` (a dedicated psycopg2 connection to `veda_engine` via
`VEDA_INTERNAL_*`/PgBouncer) and switched `ann_search` to use it. Also fixed a second, previously
-masked bug the first one was hiding: `source_id` is a `TEXT` column but was bound as `int` —
stringified it. **Live-verified**: `✓ Signal 1 via storage_adapters (engine store, source-scoped):
5 cols` against the real running stack.

### 3. Rebuilt the local docker stack
`docker compose up -d` after the user started Docker Desktop. Hit a Postgres major-version
mismatch (`pg_data` volume was PG16, compose said `pg17`) — user chose "pin compose back to
pg16" over a dump/restore; `docker-compose.yml`'s postgres image was pinned accordingly (see
`CLAUDE.md`'s note on this).

### 4. Fixed a stale `METAL_EMBED_URL`
`.env` pointed at a colleague's Mac at a stale LAN IP (`192.168.1.43`), causing every embed call
to eat a 60s timeout before falling back to CPU (one query even hit nginx's 504). User supplied a
working IP (`192.168.1.39`) from the colleague; verified reachable via curl, updated `.env`,
recreated the inference container. **Confirmed**: zero `metal ... failed` lines in warmup, a real
query dropped from 120s+ timeout to 16.4s.

**⚠ This dependency is flaky by nature** — it's a colleague's personal Mac running
`scripts/metal_embed_server.py`, not a managed service. It went unreachable AGAIN later in this
same session (see gotchas below). Expect to re-diagnose this again in future sessions; it is
never a code bug when it happens, always check reachability first (`curl` the URL in `.env`).

### 5. Made multi-source query routing "intelligent" (the multi-source coordinator)
User asked for an unbiased explanation of how the query pipeline routes across sources, then
asked to make it genuinely intelligent (each source's evidence scored, structural join detection,
permission-aware pre-check, etc.). This surfaced the existing (already-built)
`veda_core/query/source_coordinator.py` + `routing_policy.py` + `source_evidence.py` +
`doc_data_planner.py` machinery, which was running in **shadow mode only**
(`MULTISOURCE_ROUTING_SHADOW=1` — observes and logs, never drives the answer).

**Regression found and fixed**: flipping to fully authoritative (`SHADOW=0`, the previously
documented production setting) regressed plain single-source queries — the coordinator's own
routing-evidence check (a plain cosine lookup, deliberately decoupled from the real answer
engine's retrieval) became a hard gate in front of `veda/pipeline.py::run_query` (far more
capable), refusing questions the real engine answers fine (`"how many properties are there"` →
wrongly `NO_MATCH` under `SHADOW=0`). Reverted immediately, then implemented the user-chosen
fix (**"option 2" of three proposed**): scope authoritative mode to `MODE_MULTI` decisions ONLY
— `veda_hybrid.py::_run_coordinator`:
```python
_is_multi_decision = decision.status == "ROUTED" and decision.mode == "MULTI"
_effective_shadow = bool(MULTISOURCE_ROUTING_SHADOW) and not _is_multi_decision
```
A genuine cross-source `MODE_MULTI` result now drives the answer regardless of `SHADOW`; every
`SINGLE`/`NO_MATCH`/`CLARIFICATION_REQUIRED` decision stays shadow-gated as before. `.env`
unchanged (`SHADOW=1`) — the code now interprets it correctly for the scoped case.

**Live-verified organically** (not mocked): `"which assets have which amenities"` (no source pin)
→ coordinator resolved `ROUTED/MULTI (RELATIONSHIP_EDGE)` on its own via a real discovered
`cross_source_fk` edge, tried strict federation, couldn't build a valid join plan for that
phrasing, and **surfaced a controlled refusal** instead of guessing — intended behavior. Contrast:
`"list amenities for each asset"` resolved `NO_MATCH` (deferred to shadow) and the legacy
federated path answered it via a same-source join — correctly telling apart a genuine
cross-source shape from a same-source one worded similarly.

### 6. Wrote `docs/INGESTION_AND_QUERY_PIPELINES.md` (the flagship doc, ~520 lines)
One self-contained walkthrough of both pipelines end-to-end: ingestion's L1→L5 layers through
every artifact, then the query front door through the firewall and every answering head. Linked
as the top "start here" entry from both `README.md` and `docs/README.md`.

### 7. External code-review fix list — full triage and remediation
A colleague's code review ("VEDA — Fix List for Code Agent") landed with a P0/P1/P2 backlog.
**By the end of this session, every item was fixed, implemented-and-measured,
verified-already-fine, or explicitly deferred with a documented reason.** Full detail —
including exact line numbers, evidence, and live-verification transcripts for every item — is in
**`docs/backlog/query-engine-open-items.md`**, which is now the authoritative record.
Summary by item:

| Item | Outcome |
|---|---|
| P0-1 Relationship graph is process-global | **Fixed.** New `config.source_artifact_path()`; `veda/runtime.py::get_graph()` is now the ONE graph accessor (replacing 3 separate caches in `runtime.py`/`graph_guard.py`/`fast_path.py`). |
| P0-2 Graph builder wrong-DB/Postgres-only | **Fixed** (mostly). Connector-aware schema fetch mirroring L1's own dispatch; ANSI `information_schema` PK detection replacing Postgres-only `pg_index`; declared-FK-only fallback for tabular/unverified-dialect sources instead of crashing. Full non-Postgres SQL dialect support NOT built (no live source to verify against). |
| P0-3 Resume skips not source-scoped | **Fixed**, both halves (biencoder immediately; the semantic-model half once P0-5 made the semantic model itself per-source). |
| P0-4 Graph can be overwritten empty | **Fixed** — tables list now comes from in-memory run state, never re-read from disk; refuses to write an empty graph over a non-empty schema scan. |
| P0-5 Other artifacts also global/flat | **Fixed**, all 7: relationship graph (P0-1), rerank docs, join paths, enrichment index, unified graph, semantic model + domain synonyms + concept graph + glossary, compiled registries (concepts/dimensions/metrics/MANIFEST). New shared `config.resolve_source_artifact()` helper (prefer per-source path if it exists, else flat) used everywhere. |
| P0-6 Rehydrate doesn't invalidate caches | **Fixed** — one `invalidate_*_cache()` per artifact, wired into both rehydrate paths (`inference/main.py` subscriber + `inference/routes/retrieve.py` route). |
| P0-7 `VEDA_ARTIFACT_SCOPING` half-implemented | **Fixed by deletion.** Confirmed genuinely dead (never turned on in this deployment, zero real readers of the field it populated) — removed `VEDA_ARTIFACT_SCOPING`/`VEDA_ARTIFACT_SCOPE` env plumbing and `config.artifact_scope()` entirely rather than building out the alternative (wiring `SubstrateVersion` in as a version component), since the new per-source resolver needs no version segment at all. |
| P1-1 Hub-table bias (Signals 3/4) | **Fixed**, mechanically (not a weight retune) — Signals 3/4 removed from the RRF candidate union, now boost-only like Signal 6. Found `scripts/retrieval_eval.py` measures a different retrieval path that never touches this code (`RRFMerger`) — verified via a live query through the real path instead. |
| P1-2 Intent boosting dead on Tier-1 | **Implemented + measured, not blind.** Rescaled deltas to the real RRF score range (measured live: 0.05-0.06, not the assumed 0.1-0.6); wired a grammar-derived retrieval-only intent into `retrieve()` without touching planning/routing. Built a new eval script (`scripts/eval_p1_2_intent_boost.py`) since the existing harness can't see this code either. Result: no measurable difference on an 8-query sample — traced to `aggregate_mode`/`grouped_mode`'s narrow grammar coverage (see "Known follow-up" below). Kept the fix (provably non-regressing); did not revert on an inconclusive sample. |
| P1-3 signal_builder FK graph bugs | **Fixed**, all 3: `str→str` dict (dropped polymorphic multi-target edges) → `str→set`; O(n²) referenced-column check → precomputed O(1) set; dead import removed. |
| P1-4 Silent degrades invisible | **Fixed** — `/readyz` now reports a `degraded` list; new `retrieval_health` explain-trace section (`sparse_active`/`reranker_active`/`embed_backend`) per query. |
| P2-1 Test suite | **Fixed** — triaged all 4 `pytest`-reported failures (2 were pytest cross-module pollution artifacts, 1 was genuine dead code, 1 was a genuinely stale test) — suite is 64/65 passing via the documented standalone invocation. |
| P2-2 Doc/comment corrections | **Fixed**, every item in the fix list's table plus 2 more found along the way. |
| P2-3 Cleanup candidates | **Fixed** — deleted `veda_bm25_index.json`, `chunk_linker.py`, `inference/engine.py`, the dead `[N/NN]` marker-parsing code in `apps/ingestion/tasks.py` (had to also fix `tests/test_apps_layer_refactor.py`, which imported the removed constant); added `.omc/` to `.dockerignore`. |

**One explicitly open item, by design, not oversight**: P1-2's underlying gap — the grammar
classifiers (`veda_core/veda/planning.py::aggregate_mode`/`grouped_mode`) miss common phrasings
("broken down by" isn't recognized even though "breakdown" and "per" are — confirmed live,
identical query meaning, three phrasings, two work). Retuning these word lists needs its own
labelled precision/recall check before touching them — a wrong classification now measurably
moves retrieval ranking (that's the whole point of P1-2's fix), so blind additions risk the exact
failure mode the fix was closing. Not fixed this session; needs a real eval pass with a larger
labelled query set than was available.

**→ Fixed in §8 below (2026-09-15 follow-up session)**, exactly the way this paragraph
prescribed: labelled precision/recall check first, then the grammar fix — plus a second,
deeper bug the fix uncovered.

---

## 8. Follow-up session (2026-09-15): the P1-2 grammar gap, a real intent-boost bug, and MySQL dialect support

A fresh session picked up the 3 items this handoff's own "What's left" summary named open
(grammar gap / non-Postgres dialect support / thin eval infra), explicitly leaving item 4
(standing infra risk notes, §"Environment gotchas" above) untouched. **`METAL_EMBED_URL` was
reachable and fast again at the start of this session** — confirmed live (`curl .../encode_query`
round-trip: 0.7s, not the 60s CPU-fallback timeout) before relying on it. This one fact is what
made the rest of this section possible: gotcha #4 below (CPU-only eval taking 40+ minutes) simply
didn't apply this time.

### 8.1 Full 24-query P1-2 eval, now that Metal is fast

One `eng.retrieve()` call: ~1.7s (was several seconds/batch on CPU). Full 24-query (21 gradeable)
sweep: ~90s, not 40+ minutes. Ran it clean, before touching any code, as a sanity check on the
2026-09-11 8-query finding above — **BASELINE and AFTER came back byte-identical again** on the
full set (recall@5=0.1936, recall@15=0.3734, mrr=0.4139, table_recall@3=0.7619 both conditions).
Not a sampling artifact; something structural.

### 8.2 The grammar-coverage gap — fixed, data-backed (as prescribed above)

Built the labelled check this doc's §7 said was required before touching the word list, rather
than guessing:
- **`evaluation/grouping_grammar_labels.jsonl`** (new, 25 rows) — true positives for every
  existing + candidate grouping phrase (including the exact golden-set query cited above), and
  true-negative *distractors* sharing the surface word "by" that must NOT trigger grouping
  (`increased by`, `sorted by`, `divided by`, `backed by`, `measured by`, `followed by`,
  `given by`, `accompanied by`, `differ by`, plus the pre-existing ratio-wording case).
- **`scripts/eval_grouping_grammar.py`** (new) — runs `grouped_mode()` over that set,
  BASELINE `QUERY_GRAMMAR["grouping"]` vs a CANDIDATE list, never mutating `config.py` itself.
  BASELINE: precision=1.0, recall=0.417 (7/12 false negatives — every one a "broken down by" /
  "broken out by" / "break down" / "split by" / "segmented by" / "categorized by" phrasing).
  CANDIDATE (adding exactly those 6 phrasings): precision=1.0, recall=1.0 — zero new false
  positives against the distractor set.
- **Applied**: `veda_core/config.py`'s `QUERY_GRAMMAR["grouping"]` now includes those 6 phrasings
  alongside the original `per/each/grouped by/breakdown`. Bare "by" deliberately never added.
- **No regression**: hand-invoked all 15 parametrized assertions from
  `tests/test_grouped_aggregation_operators.py` (pytest still isn't installed) — unchanged. Full
  64/65 routing suite re-confirmed. Live-confirmed: the cited golden-set query now classifies
  `AGGREGATE` instead of `SIMPLE`.

### 8.3 The deeper bug this uncovered: `IntentBooster` was a permanent no-op

Re-ran the P1-2 eval expecting a change after 8.2 — **got the exact same byte-identical numbers a
third time.** Diffed per-column rankings for the query that now triggers AGGREGATE: SIMPLE and
AGGREGATE produced **identical ranked output**, not just identical metrics — meaning the boost
itself wasn't firing, independent of intent classification. Root cause:
`retrieval/intent_boosting.py::IntentBooster._get_column_metadata()` walks
`semantic_model["tables"][t]["columns"][c]` — but no table entry in the real
`veda_semantic_model.json` has a `"columns"` sub-dict at all (confirmed live: a real entry's keys
are `table_name/business_purpose/primary_entity/table_type/candidate_temporal_columns/
candidate_measure_columns`). Column metadata (`analytics_role` — the field every `boost_*` method
reads) actually lives in the model's own **top-level flat `columns` dict**, keyed `"table.column"`
(1902 entries for source 2; confirmed `currency_id`→`IDENTIFIER`, `paid_amount`→`MEASURE` both
present there with the expected roles). So the lookup always returned `{}`, `role` was always
`""`, and `boost_aggregate`/`boost_temporal`/`boost_multi_table` have been **unconditional no-ops
for every intent, always** — not a grammar problem; the boost could never have fired even with
perfect grammar coverage. This is almost certainly the real reason both the 2026-09-11 8-query
eval and this session's first 24-query re-run (8.1) showed zero effect.

**Fixed**: reads the flat `columns` dict first (O(1), was an O(tables×columns) walk), falling
back to the old nested-walk shape only on a miss. No dedicated tests existed for
`intent_boosting.py` (checked — zero references). Live-verified: `paid_amount` (MEASURE) jumped
rank 12→2 under AGGREGATE intent; `currency_id` (IDENTIFIER) dropped out of the top 15 (the
`-0.40`-scaled IDENTIFIER penalty firing as designed).

**Definitive P1-2 eval, both fixes applied, full 24-query set, Metal fast:**

| metric | BASELINE | AFTER | Δ |
|---|---|---|---|
| recall@5 | 0.1936 | 0.2571 | **+33% relative** |
| recall@15 | 0.3734 | 0.3907 | +5% relative |
| mrr | 0.4139 | 0.4790 | **+16% relative** |
| table_recall@3 | 0.7619 | 0.7143 | **−6% relative** |

Genuinely mixed, not a clean win. The table_recall@3 dip is explainable, not a bug: the
IDENTIFIER penalty demotes columns like `currency_id` even when a query wants that column as a
GROUP BY dimension rather than something to sum — IDENTIFIER-vs-grouping-dimension is a real
conflict this fix surfaces but doesn't resolve. **Kept, not further tuned** — flagged as a follow-
up, not reverted from a 21-query sample (matching this doc's own "did NOT revert based on
inconclusive data" precedent above, except this result is no longer inconclusive, just mixed).
Full 64/65 routing suite re-confirmed clean.

### 8.4 Non-Postgres SQL dialect support — MySQL added, live-verified

The P0-2 gap above: `_can_sql_introspect()` gated full SQL introspection to Postgres only, "since
there is no live [non-Postgres] one to verify dialect-specific SQL against." Found
`connectors/relational.py::MySQLConnector` already existed for L1 schema extraction (dialect-
correct information_schema queries, backtick quoting) — but `ingestion/relationship_graph.py`
never used it; it hardcoded its own `psycopg2` connection and Postgres-only SQL (double-quoted
identifiers, `::text` casts, `= ANY(%s)` array binds) for PK/cardinality/polymorphic detection.

Stood up a throwaway MySQL 8 container (`veda-test-mysql`, on `veda-platform_veda_net`, removed
after verification) with a small `customers`/`orders` schema (real FK,
`orders.customer_id → customers.id`, multiple orders per customer) — the live source this gap
always lacked.

**Found and fixed 3 real bugs, not just added new code:**
1. **`mysql-connector-python` was never installed anywhere in this deployment** — `MySQLConnector`
   has depended on it since it was written. Added to `requirements/inference.txt` (shared by
   `inference` + `ingest-worker`) and `requirements/host-ingest.txt`.
2. **`connectors/relational.py::RelationalConnector.connect()`'s own health-check ping never
   drained its `SELECT 1` result before closing the cursor.** Harmless on psycopg2/sqlite3; fatal
   on mysql-connector-python's C extension, which leaves the whole *connection* flagged "has
   unread result" until something fetches it — so the very next `get_schema()` call blew up
   immediately with `InternalError: Unread result found`, before running any real query. A real,
   previously-unexercised bug in the shared connector base class, unrelated to this fix's own
   changes — never triggered before because MySQL was never actually runnable (bug #1). Fixed:
   drain via `cur.fetchall()` before `cur.close()`.
3. **`relationship_graph.py` dialect support** — added `_engine_of()`, `_ident()` (backtick vs
   double-quote), `_cast_text()` (`CAST(x AS CHAR)` vs `x::text`), `_in_clause()` (dialect-neutral
   `IN (%s,%s,...)`, replacing Postgres-only `= ANY(%s)` — unsupported by mysql-connector-python or
   most non-psycopg2 drivers) — threaded through `_conn` (now dispatches on `ctx.engine`),
   `_table_meta`, `_cardinality`, `_polymorphic_edges`. `_can_sql_introspect()` now accepts
   `mysql` (still declared-FK-only for SQL Server/Oracle/etc — no live instance yet). Also fixed
   the schema default: MySQL has no `"public"` schema — `information_schema.*`'s schema filter IS
   the database name there; Postgres's `"public"` default unchanged.

**Live-verified** (`scripts/verify_mysql_relationship_graph.py`, new — a one-off repro script, not
a permanent test): `build_relationship_graph()` returned `mode: "sql"` (full introspection, not
the fallback), found the 1 real FK edge, computed `cardinality: "N:1"` correctly from real data
correlation. **No regression on the real Postgres path**: rebuilt source 2's graph through the
same patched code — **178 tables / 609 edges / 0 polymorphic, byte-identical** to the pre-existing
stats recorded above. Full 64/65 suite re-confirmed clean.

**Still not done**: SQL Server, Oracle, or any other dialect — still declared-FK-only, no live
instance to verify against. `mysql-connector-python` is a live pip install in the running
containers this session; tracked in `requirements/*.txt` for the next image rebuild, not yet
baked into one.

### 8.5 Files touched this session (all uncommitted)

Modified: `veda_core/config.py`, `veda_core/retrieval/intent_boosting.py`,
`veda_core/ingestion/relationship_graph.py`, `veda_core/connectors/relational.py`,
`requirements/inference.txt`, `requirements/host-ingest.txt`,
`docs/backlog/query-engine-open-items.md` (detailed write-up, same content as this section in
more depth).
New: `evaluation/grouping_grammar_labels.jsonl`, `scripts/eval_grouping_grammar.py`,
`scripts/verify_mysql_relationship_graph.py`, this file's §8 itself.

Full 64/65 routing test suite re-confirmed clean after every one of 8.2/8.3/8.4's changes, not
just once at the end — same discipline as the rest of this doc.

Two new resolvable follow-ups, neither fixed this session: the IDENTIFIER-vs-grouping-dimension
tension in `boost_aggregate` (8.3), and full non-Postgres-non-MySQL dialect support (8.4).

---

## 9. Same follow-up session, continued: chat API auth, a live "why did this fail" trace, and a real pipeline gap fixed

The user asked for the chat API to converse with VEDA directly (not just the one-shot
`/api/v1/query` endpoint) and then hit a real refusal live — this section is that whole thread,
kept in one place since each step fed the next.

### 9.1 The chat API, and how auth actually works in this deployment

Endpoint: `POST /api/v1/conversations/query {message, chat_id?, stream?}` (mounted under
`api/v1/` by `apps/chat/urls.py`; `chat_id: null` starts a new conversation, `stream: false` gives
one buffered JSON reply instead of the SSE default — best for curl/terminal checks). Also:
`POST /api/v1/conversations/create`, `GET /api/v1/conversations/list`,
`GET /api/v1/conversations/history?chat_id=...`. Local stack: `http://localhost:8080/...`
(nginx → api container).

**Two real gaps hit getting a login to actually work, not just config trivia:**
1. `admin` (username `admin`/password `admin123`, `user_id=2`, `is_staff=False`) had **no RBAC
   role row** — `apps/authentication/services.py`'s login flow refuses any non-`is_staff` account
   with zero `UserRole` rows (`NO_ROLE_ASSIGNED`, by design — "is_superuser does NOT bypass this
   ... an admin-app account still needs a real role assigned"). Fixed by granting the existing
   `Admin` role (`access_management_role.id=1`) directly via a `access_management_userrole` insert
   (`role_id=1, user_id=2, granted_by_id=NULL`) — a real, reversible RBAC grant on the local DB.
2. **This deployment's `.env` has `VEDA_JWT_AUTH` off** (`config/settings/base.py:172`,
   default `"0"`) — while off, login returns a literal placeholder string
   (`LEGACY_ACCESS_TOKEN = "dummy_access_token"`, `apps/authentication/services.py:83`) that
   authenticates NOTHING; there's also no session-cookie fallback (no `Set-Cookie` on the login
   response, confirmed via `curl -i`). Rather than flip `VEDA_JWT_AUTH=1` (a deployment-wide
   behavior change needing a container recreate), created a real DRF `Token` for `admin` directly
   (`rest_framework.authtoken.models.Token.objects.get_or_create(user=admin)`) — smaller, local,
   reversible (delete the row). **Use `Authorization: Token <key>`, not `Bearer`** — DRF's
   `TokenAuthentication` uses its own scheme. Live-verified: a real `chat_id`/`message_id` got
   persisted through a full `POST /api/v1/conversations/query` call.

### 9.2 "how many properties are there?" — traced live, not guessed

The user's very first real chat query refused (`"I couldn't work out a reliable total for
this..."`). Traced it with the real trace object (`veda_hybrid.run_hybrid_query(..., verbose=True)`
→ `MultiResult.items[0].result["trace"]`), not by assuming the old documented bug still applied:

- **The OLD documented anchor-routing bug (§ "Multi-source coordinator" above, and
  `docs/backlog/query-engine-open-items.md`'s "routes to `assets_listingvisit` instead of the
  properties table") is NOT what's happening now.** Routing correctly picks `assets_asset` (the
  real properties table) — confirmed in the trace's `entity_selection`/`schema_linking` sections.
- **The real cause**: `query_understanding.aggregation` correctly detects a bare COUNT
  (`aggregate_mode()`'s "counting" branch fires on "how many" — this is right). But
  `veda/pipeline.py`'s SQL-planning branch chain (the `if/elif` chain choosing HOW to build the
  SQL — multi-hop FK / arbiter value filter / temporal predicate / ranked-temporal) never
  consulted that signal at all. A bare "how many X are there" — no filter, no date window, no
  ranking — matches none of those specific branches and falls to the generic catch-all, which
  builds a plain row-listing `SELECT` (confirmed: 46 columns, zero aggregate functions —
  `sql_planning: {"action": "single_table", ...}`). `veda/intent_sql_alignment.py::
  aggregate_presence_ok()` then correctly refuses — "how many" intent + an aggregate-less SQL —
  rather than show that row list as if it were the count. **Working safety net, real upstream
  gap.** (`veda/planning.py::build_aggregate_sql`, which exists for exactly this, is never called
  anywhere in `pipeline.py` — confirmed, zero references.)
- Contrast: `"how many users are there"` already worked — but via a completely different path,
  `query/fast_path.py`'s pre-registered metric shortcut (`metric.count`), confirmed from this same
  session's earlier P0-5 work. `assets_asset`/"properties" has no equivalent registered count
  metric, so fast_path never even attempts it and the gap above is what actually answers (or
  refuses) it.
- **Side effect surfaced along the way, unrelated to the refusal itself**: the trace also logged
  `[UnifiedGraph] ⚠ STALE — rebuilt inputs since last build: relationship_graph` — caused by this
  session's own §8.4 MySQL-verification work, which regenerated source 2's relationship graph file
  (same real data, fresh mtime) without also refreshing the unified graph that depends on it.
  **Fixed**: `ingestion/unified_graph_builder.py`'s CLI only rebuilds the flat legacy file
  (`data/veda_unified_graph.json`) — source 2 actually reads its own per-source copy
  (`data/default/2/veda_unified_graph.json`), so rebuilt THAT one directly
  (`write_unified_graph(source_id=2, tenant="default")`, same 17,134 nodes / 33,551 edges as the
  2026-09-10 build). Confirmed live: the staleness warning is gone on the next query.

### 9.3 Fixed: `veda/pipeline.py` now answers bare-count queries deterministically

Per the user's explicit ask ("update pipeline to accept these types of queries as well"), added a
new deterministic branch to the SQL-planning `if/elif` chain (`veda/pipeline.py`, right before the
existing `elif _tpred:` temporal-only branch, so it takes priority and can also compose with a date
window):

```python
_bare_count = bool(_agg) and _agg.get("op") is None and _agg.get("threshold") is None \
    and not _agg.get("top_n") and not _agg.get("ranked")
...
elif _bare_count:
    sql = f'SELECT COUNT(*) AS count FROM "{primary}"' + (f' WHERE {_tpred}' if _tpred else '')
    ...
    _llm_sql = False   # deterministic — skip IR-equivalence
```

`_agg` is the SAME `aggregate_mode(query)` dict already computed near the top of `run_query()` —
the guard deliberately excludes `threshold`/`op`/`top_n`/`ranked` so this never intercepts a
per-anchor child-count ("X with more than one Y") or a ranked count ("top 5 X by count"), which
have their own existing, unrelated handling. Placed BEFORE `_arb_filters`'s/`_mh`'s/`_fk`'s
branches in priority is deliberately NOT how this landed — those still own their own SQL shape;
`_bare_count` only fires when none of them matched, i.e. genuinely no filter at all (plus optional
date window via `_tpred`).

**Live-verified**: `"how many properties are there?"` → `SELECT COUNT(*) AS "count" FROM
"assets_asset" LIMIT 100` → **6,402**, `status: ok`, real persisted answer through the actual chat
API. **Regression-checked**:
- Full 64/65 routing suite — unchanged.
- `"how many users are there"` — still fast_path, untouched (the new branch is in a different code
  path fast_path never reaches).
- `"list top 5 properties by monthly rent"` — still the grouped/ranked SQL, not hijacked into a
  bare count.
- `"show all vendors"` — unaffected plain row list (no aggregate signal at all).

**Found, NOT fixed — flagged, not chased**: `"how many properties were added last month"` still
refuses, but for a completely different, pre-existing, unrelated reason — `aggregate_mode()`
returns `ranked: True` for it (the word "last" ALSO triggers the ranking-word detector; "last N"
vs. "last month" is a genuine, separate ambiguity this session did not touch), so it deliberately
skips the new `_bare_count` branch (which excludes `ranked=True` on purpose, to avoid stepping on
real ranking queries), falls into the existing temporal-only branch, and THAT routes to the wrong
anchor (`reminders_reminder`) and hits an unrelated value-grounding refusal on the literal word
"property". Same discipline as §8.2's grammar work: not touching `QUERY_LANGUAGE["ranking"]`'s
word list without its own labelled precision/recall check first.

**Also noticed, not chased**: `result_explainer.run_nl_answer` logs `SLM unavailable/failed
(RuntimeError: SLM unreachable at http://host.docker.internal:11434/api/generate: HTTP Error 404:
Not Found)` on every query that reaches it, even though a direct `curl` with the CORRECT model
name (`.env`'s real `SLM_MODEL_NAME=qwen2.5-coder:7b`, confirmed present in `ollama list`) succeeds
in under a second. `result_explainer.py` appears to request a different/wrong model name than
`SLM_MODEL_NAME` for this specific call — degrades gracefully to a working fallback answer
("The count is 6,402."), so nothing user-visible broke, but it means every NL-narrated answer in
this deployment is currently using the fallback path, not the LLM one. Not investigated further
this session.

### 9.4 Files touched in §9 (also uncommitted)

Modified: `veda_core/veda/pipeline.py` (the `_bare_count` branch). Data changes, not files:
one `access_management_userrole` row (admin's RBAC grant), one `authtoken_token` row (admin's DRF
token), and a regenerated `data/default/2/veda_unified_graph.json` (gitignored, not part of
`git status` either way). No `docs/backlog/query-engine-open-items.md` entry was added for this
section — ask if you want one written up in that doc's fuller style too.

---

## Test suite — how to actually run it

**Pytest is not installed anywhere in these containers.** The routing/coordinator test suite is
meant to run standalone, per-file, with a `__main__` block:
```bash
docker exec veda-platform-inference-1 bash -lc 'cd /app && python tests/test_authoritative_routing.py'
# one exception needs explicit PYTHONPATH ordering (veda_core first, else Django's own
# `config` package shadows veda_core/config.py):
PYTHONPATH=/app/veda_core:/app python tests/test_federated_reliability.py
```
The 7 files that make up the "64/65" baseline: `test_authoritative_routing.py` (13),
`test_source_coordinator.py` (10/11 — 1 genuine dead-code failure, documented, not a regression),
`test_routing_policy.py` (10), `test_source_evidence.py` (7), `test_routing_slm.py` (8),
`test_federated_labelling.py` (5), `test_federated_reliability.py` (11). Running them under
plain `pytest` (if ever installed) cross-pollutes global module state between files and produces
2 false failures — confirmed by running standalone, which passes clean.

`tests/test_apps_layer_refactor.py` needs Django configured in-process (no DB) —
see its own docstring for the `_setup_django()` helper pattern; run it by hand-invoking its test
functions (no `__main__` block), since pytest isn't available.

---

## Environment gotchas discovered/re-confirmed this session

1. **Two Python module identities for the same file.** `veda_core/__init__.py` inserts its own
   directory onto `sys.path` at import time, so `veda_core/context.py` (and `config.py`) can be
   imported BOTH as bare `context`/`config` AND as `veda_core.context`/`veda_core.config` — two
   SEPARATE module objects with separate contextvars. The real request path always uses the
   `veda_core.`-prefixed form (`inference/main.py`'s middleware, `veda_hybrid.py::_current_ctx()`
   tries both explicitly). **When writing ad-hoc verification scripts, always
   `from veda_core.context import RequestContext, set_context`** — using the bare form will
   silently set a context that the real code never sees, making live tests look broken when
   they aren't (this cost real debugging time this session).
2. **CWD matters for every relative artifact path.** The real ingestion subprocess and the
   inference service both run with `cwd=/app/veda_core` (confirmed via `/proc/1/cwd` for the
   live inference process). Any ad-hoc verification script must `cd /app/veda_core` first, or
   `ARTIFACT_ROOT="data"`-relative paths resolve to the wrong place (`/app/data/...` instead of
   `/app/veda_core/data/...`) and files silently appear "missing."
3. **`METAL_EMBED_URL` is flaky** (it's a colleague's personal Mac, not a managed service). When
   unreachable, `ingestion/m3_encoder.py`'s calls correctly fall back to CPU but only after
   `METAL_EMBED_TIMEOUT` (default 60s) — this makes ANY code path that does real embedding
   (including, surprisingly, `query/source_coordinator.py`'s routing tests, which call real dense
   encoding despite claiming "no DB/SLM/model" in their docstring) take up to 60s per call. For
   fast iteration, override `METAL_EMBED_URL=` (empty) or `METAL_EMBED_TIMEOUT=2` on the specific
   command — never edit the real `.env` for this. **Was unreachable at the end of the 2026-09-09
   session; was reachable and fast again (0.7s round-trip) at the start of the 2026-09-15 follow-
   up (§8)** — exactly the flakiness this note warns about. Always `curl` the URL first; don't
   assume either state.
4. **CPU-only BGE-M3 is slow for bulk eval — but only when Metal is actually down.** With no
   working Metal backend, a full 24-query retrieval eval sweep (2 conditions) took 40+ minutes and
   had to be abandoned/reduced to 8 queries. With Metal reachable (§8), the same full sweep took
   ~90s. Check Metal reachability before assuming an eval needs to be shrunk.
5. **Rapid successive file edits trigger reload storms.** `uvicorn --reload --reload-dir /app`
   watches the WHOLE app directory, including one-off scratch/eval scripts under `scripts/`. Many
   edits in quick succession (each triggering a ~40-60s CPU rewarm) can compound with a
   simultaneously-slow Metal timeout to leave the container "unhealthy" for a while. If this
   happens, `docker restart veda-platform-inference-1` and wait for `hydrate complete` in the
   logs before retrying, rather than fighting the reload queue.
6. **Deleting/adding files inside a container's `/tmp` may fail with "Operation not permitted"**
   if the file was created via `docker cp` (runs as root) but the exec session runs as a
   non-root user. Harmless — these are ephemeral container `/tmp` files, not part of the repo.

---

## 10. Same follow-up session, continued: why new phrasings keep needing individual fixes

Direct continuation of §9's chat-API investigation — the user asked, correctly, "the
query can be anything, we can't keep fixing each of them, there has to be a generic
solution." Investigated whether one exists rather than proposing something new:

- **Diagnosis**: `veda/pipeline.py`'s SQL planning is a fixed chain of 7 narrow, hand-
  written regex/keyword-triggered branches (§9.3's `_bare_count` is one more instance of
  this exact pattern), falling through to an unverified free-text LLM call for anything
  that doesn't match. Not a real semantic-parsing layer — whack-a-mole by construction.
- **Found the codebase already has the right architecture, unused**: `veda_core/veda/
  understanding/` (LLM → typed concept extraction → deterministic, no-LLM schema-
  grounding firewall, refuse-over-guess) + `veda_core/veda/analytical_spec.py`
  (structured spec → the SAME `build_aggregate_sql` every deterministic branch uses).
  Both flag-gated off (`QUERY_UNDERSTANDING_ENABLED`, `ANALYTICAL_SQL_V2`,
  `config.py:1915,1923`).
- **Two concrete, confirmed gaps in that layer**: (1) filter/dimension grounding is
  literally unimplemented — `grounding.py::ground()` hardcodes `dimensions=[],
  filters=[]` — so it could never have fixed the §9.2 "for sale" filter-drop bug even
  fully enabled; (2) **live-tested it scoped-on** (monkeypatched, `.env` untouched)
  against a battery including previously-working queries: fixed 1 case for free
  ("how many properties are there," matching §9.3's own fix), but **regressed 2 working
  queries** ("show all vendors," the §8.2 grouping-grammar case) — because a `Refusal`
  from the new layer is treated as terminal in `pipeline.py`, pre-empting the old,
  proven pipeline. Same failure shape as the multi-source coordinator's authoritative-
  mode incident documented earlier in `query-engine-open-items.md` — a smarter
  subsystem gating in front of a strictly more capable existing one.
- **Not fixed this session** — this is a real, phased engineering project, not a flag
  flip: (1) implement filter/dimension grounding, reusing L6a's existing value-grounding
  machinery; (2) make `Refusal` degrade to the existing pipeline instead of terminal,
  mirroring the coordinator fix's scoping pattern; (3) a labelled eval battery before any
  `.env` default change; (4) retire the regex branches the generic path demonstrably
  subsumes.

Full write-up (flow diagrams for both the SQL-planning chain and the chatbot
conversational layer, the regression evidence table, the phased roadmap, plus the two
smaller open items — the "last N" vs "last month" ranking-word collision and the
federated/multi-source filter-drop bug itself, still not root-caused to a file/line):
**`docs/backlog/QUERY_UNDERSTANDING_GAPS_AND_ROADMAP.md`** (new this session).

No code changed in this part of the session — investigation + documentation only.

---

## 11. Same follow-up session, continued: "every registered source works the same way" — executed

The user shared an external architecture review (2026-09-10 snapshot; five properties P1–P5,
migration order M0–M6) and said: *we need to do this, making sure all registered sources work
the same way.* Reconciled it against the live tree first (it predated §8–§10 and the P0 pass —
P1 and most of P5 were already fixed; P2/P3/P4 confirmed open), then executed M0 + M1's
residuals end-to-end. Full record: **`docs/backlog/ARCH_REVIEW_2026-09_RECONCILED.md`**.

**What was actually wrong** (measured with a per-source battery, not assumed): the four ready
sources did NOT behave the same. Only homzhub (2) worked. Sources 3/4/5 were ingested
2026-07-08, before artifact scoping, so `config.resolve_source_artifact`'s "scoped if it
exists, else flat" contract handed them **homzhub's** semantic model, registries and rerank
docs — source 5 planned `FROM "assets_amenity"` (a homzhub table) on a parquet source whose
only table is `amenities_catalog`; source 4 got an empty projection (`SELECT FROM "vendors"`);
a count-shaped question on the document source entered the SQL head and died with a raw
`OperationalError` (hostless connection). `scripts/backfill_semantic_model.py`'s docstring had
recorded this exact history. The engine-store tables were clean; only the file/Redis substrate
was contaminated.

**Fixed (all uncommitted):**
- `config.resolve_source_artifact` — flat fallback is **owner-only**, ownership derived from
  data (`_flat_artifact_owner_ok`: every table the source owns in `column_embeddings_v2` must
  exist in the flat model; cached; permissive-with-warning on DB error).
- `veda_hybrid._load_semantic_model` — missing model → empty, tagged model; new honest status
  **`not_materialized`** (`veda/feedback.py` + the `if not sm.get("tables")` site) instead of
  the misleading `access_denied`.
- `veda_hybrid._scope_has_structured_source` — `hybrid` intent only when a structured source
  is in scope; a document-only scope stays on RAG.
- `veda/execution.py` — connection acquisition inside the error contract; hostless source →
  typed `exec_error`, never an exception.
- `veda/pipeline.py` — (a) `_bare_count` grouping guard: the §9.3 branch was hijacking grouped
  counts ("…per vendor") into a scalar — my regression, caught by the battery the same day;
  (b) **verified-query cache shape guard** (third demotion): a 0.88-similar cached scalar was
  replayed for a grouped question; now grouping↔`GROUP BY` and aggregate-presence must match.
- `scripts/eval_per_source_battery.py` (new) — the M0 seed: same-shaped questions pinned per
  source, asserting no foreign tables in executed SQL, no crashes, no dropped grouping.
- Re-ingested sources 4 (canary, job 20), 5 (21), 3 (22) under the scoped pipeline; recreated
  `ingest-worker` (it was created 2026-09-10 08:48 with the stale `.43` Metal URL and paid a 60 s
  timeout per embed call — 197 s for the canary vs ≈30 s each afterwards).

**Result:** battery `--sources 2,3,4,5` → OK, 0 failures; confirmed through the live HTTP chat
API with `source_ids` pinned. Routing suite 64/65 after every change. The two grouped questions
on lite-model sources return an honest `clarify` (no type metadata to pick a dimension — P3/M2
work, not isolation).

**Still open, in the review's order:** M2 (IR + filter/dimension/time grounding, fix the
authority leaks) → M3 (one compiler + IR firewall; closes the federated filter-drop) → M4
(IR-stack memory with inherited source scope) → M5 residuals (`required_for_ready`, fusion
weights) → M6 (retire the regex chain). M1 residuals: `semantic_layer_v2.py:660`, Django-mirror
`GraphNode/GraphEdge` rows under 3/4/5, `required_for_ready`.

Files touched in §11: `veda_core/config.py`, `veda_core/veda_hybrid.py`,
`veda_core/veda/execution.py`, `veda_core/veda/feedback.py`, `veda_core/veda/pipeline.py`,
`scripts/eval_per_source_battery.py` (new), `docs/backlog/ARCH_REVIEW_2026-09_RECONCILED.md`
(new), this section. Live changes outside git: three re-ingested sources, one recreated
container.

---

## 12. Close M1 for real (2026-09-16): five ordered items, each with an exit test

The user's pass: *verify each claim against the live tree before changing anything; say so
if a claim is already stale; per item report inspection vs. claim, files, exit test, open
points; no git; no fourth cache-demotion heuristic.* Full itemized record with evidence:
`docs/backlog/query-engine-open-items.md` ("M1 close-out pass") and
`docs/backlog/ARCH_REVIEW_2026-09_RECONCILED.md` §6/§7.

**Item 1 — resolver has no flat fallback.** Claim was *partly stale*: §11 had already
made the flat fallback owner-only. Now removed outright: `config.resolve_source_artifact`
returns the scoped path or `None` (`flat_default` accepted and ignored); every reader is
None-safe and treats "missing" as an empty model / `not_materialized` (`veda/runtime`,
`veda_hybrid`, `query/fast_path`, `query/intent`, `query/entity_resolver` (glossary keyed
per scope), `graph/query_graph`, `query/join_planner`, `veda/validation`,
`semantic/registry`, `inference/loaders`, `ingestion/rerank_docs`, `ingestion/biencoder`,
`ingestion/value_referents`). Exit: grep for flat artifact reads is clean outside the
resolver; source 2 resolves 14/15 artifacts scoped (`veda_entity_aliases.json` has never
existed anywhere); DB down → typed `exec_error`.

**Item 2 — 3/4/5 re-materialised clean.** 0 homzhub table names in any scoped artifact of
3/4/5 (`doc_chunks` "role" hits are English prose). Discovered while checking:
`ingestion/biencoder.py` called its loaders ctx-less, so once the flat fallback went every
L4 embedded the structural passage only — fixed (source_id + ambient tenant), worker
restarted; affects future ingests only.

**Item 3 — lite-model sources answer the same shapes.** Root causes were generic, not
per-source: the value mirror/`column_values` were not scoped by the graph's table ids
(`query/value_resolver._scope_table_ids`, `query/resolution`); the parquet value probe was
silently off without HTTP profiles (`query/datalake_values` now asks `resolve_surface`);
retrieval Signal 1+2 ran in threads without the request context (`copy_context().run`);
the lite model used a `METRIC` role the planner doesn't know (`MEASURE`); data-graph FK
discovery (`fk_adjacency`) was never merged into the relationship graph
(`_discovered_fk_edges`/`_merge_edges`); no grouped-COUNT or ranked-metric deterministic
branch existed (`planning.grouped_count_mode`, `superlative_plan` COUNT branch,
`pipeline` `ranked_metric_only`). Exit, live: "vendors in Kochi" → `WHERE LOWER(city)=%s`;
"amenities in the Sports category" → `WHERE category`; "how many maintenance records per
vendor" → `vendors JOIN maintenance … GROUP BY t0.vendor_id, t0.city` (the discovered
join); grouped COUNT/AVG/SUM/MAX/MIN per category, `top 3 … LIMIT 3`, `above 100 → >`.

**Item 4 — Django mirror is not a query-path authority.** Evidence: `storage_adapters/
assembler.py` reads `Sm*` tables only; no query-path reader of Django `GraphNode/GraphEdge`.
`storage_adapters/writer.py` now selects `graph_nodes/graph_edges WHERE source_id = %s`
and writes to the scoped artifact dir; stale mirror rows cleaned; Redis `veda:sm:*` unchanged.

**Item 5 — battery ≥15/source, shape-asserted.** `scripts/eval_per_source_battery.py`: 63
questions / 4 sources, each with expected route + shape (aggregate, `GROUP BY` dimension,
`WHERE` presence *and* comparator direction, `LIMIT N` = asked N, no foreign tables, no
crash); typed refusals on documented gaps → WARN; **`expect="refuse"`** for "show tickets
with high priority" (this copy has only LOW/MEDIUM — an answer *is* the bug; the shared
planner had been answering it unfiltered); **`xfail`** carrying the verified-cache
similarity-replay case ("which vendor has the highest rating" replays the cached "top 3
vendors by rating", cosine 0.86 → `LIMIT 3`). Per instruction no fourth demotion was added;
disclosed: the *existing* shape demotion was extended with `LIMIT N`; my own wrong test
answers had been cached as "verified" and were purged (13 rows 4/5, 2 rows 2). The cache
key must become IR-shape-aware in M2/M6 (noted in open-items). Routing suite: 64/65 — the
1 is `test_source_coordinator.py::test_dispatch_with_adapter_flag_on_matches_flag_off_exactly`,
`ModuleNotFoundError: query.source_adapters` (`source_coordinator._resolve_executable`
imports a module deleted in P2-3; reachable only with the adapter flag on) — dead code,
unchanged, not a regression.

**Late finds, same pass (generic, all sources):** (a) the Tier-2 **envelope** (frozen
contract v1: no ranking intent, `eq|ne` filters) answered out-of-contract questions with
the *nearest* shape — "top 5 by monthly rent" as a monthly trend, "more than 3 floors" as
`= 3`; now `veda_hybrid._envelope_inexpressible` skips it (ranking / threshold / negation,
reusing `ranking_parser` + `operation_classifier`'s phrase sets); (b) `_tier2_validate`
gained a dropped-constraint check (threshold ⇒ comparison/HAVING; negation ⇒ `<>`/`NOT`);
(c) the **shared-planner** Tier-2 branch ran only the AST firewall, never `_tier2_validate`
— "more than 3 floors" executed as "assets with more than 3 listing reviews"; gated now;
(d) env drift: the inference container holds `OLLAMA_URL=host.docker.internal:11434`
(serves `/api/chat` but 404s `/api/generate`, so the NL explainer falls back) while `.env`
says `192.168.1.35:11500` — the new `env_drift` check reports it on every boot; needs
`docker compose up -d inference` (not done: it would have killed the battery mid-run).

**Result:** battery `--sources 2,3,4,5` → see the final line recorded in
`ARCH_REVIEW_2026-09_RECONCILED.md` §6 (0 failures; typed-gap WARNs and 1 documented
xfail); routing suite 64/65; all changes uncommitted.

**Next:** M2 first checkpoint (grounding of filters/dimensions/time with type + value
checks, `analytical_spec` GROUP BY/numeric enforcement + grounded WHERE, the two authority
leaks in `pipeline.py`), flags staying off; report before M3.

---

## 13. Pre-M2 items (2026-09-16, on the committed baseline `b0bc2eb`) and the M2 first checkpoint

**Item 1 — golden-set retrieval eval: byte-identical to 09-15**, but only after fixing
what the first run exposed. `scripts/eval_p1_2_intent_boost.py --source-id 2` crashed,
then ran blind, for two M1-era reasons: (a) `config.ARTIFACT_ROOT` was CWD-relative — the
battery `chdir`s to `veda_core` and worked, the eval runs from `/app` and every scoped
artifact resolved to `data/default/2/…` relative to the wrong directory (the enricher
loaded no synonyms/glossary/model; the first numbers were recall@5 0.3454 / table_recall@3
0.5238 — meaningless); now anchored at the package dir (`_artifact_root_abs`); (b) the eval
sets scope through the bare `context` module while the resolver reads `veda_core.context`
— both `/app` and `/app/veda_core` are on `sys.path`, so those were two module objects
with two ContextVars (20 vs 25 import sites); `context.py` now makes the second import
adopt the first's ContextVars. One missed reader fixed on the way (`retrieval/
query_enrichment.py` opened a `None` path). Result, both conditions, 21 graded of 24 (3
rows carry no gold columns, as before): baseline recall@5 0.1936 / recall@15 0.3734 / mrr
0.4139 / table_recall@3 0.7619; grammar-intent 0.2571 / 0.3907 / 0.479 / 0.7143 — the
09-15 numbers exactly. Nothing dropped.

**Item 2 — verified-query cache.** (a) `RequestContext.cache_back` (default True):
False = no replay AND no write. Set from `X-Veda-No-Cache: 1`, which the api tier sends
when the request carries `no_cache` (`/api/v1/query` and the chat front door:
`apps/query/views.py`, `apps/chat/views.py` → `ConversationQueryService` →
`chatbot/run.py` → `call_engine_node` → `InferenceClient`); the battery and every eval
script (`eval_per_source_battery`, `eval_p1_2_intent_boost`, `retrieval_eval`,
`tune_fusion_weights`, `parity_suite`, `apps/evaluation/tasks.py`) set it. (b)
`VerifiedQueryCache.substrate_version` (migration `substrate/0009`): written from the
source's current `SubstrateVersion.version`, part of the unique key, and both lookups
(`verified_cache_exact`, cosine `verified_cache_lookup`) match only rows at the source's
CURRENT version — a re-ingest invalidates every replay without a purge. Purged once after
(a): 37 rows (all pre-version; unservable under the new predicate anyway). No new
demotion heuristic. Consequence for the battery's xfail replay case: with `cache_back`
off it can no longer be replayed, so the marker came off (XPASS) — the production replay
hazard is closed by the version key + the M2/M6 IR-shape key, not by the battery.

**Item 3 — RAG content assertions found a real retrieval miss.** `expect_text` (any-of
substrings from the actual chunk texts) + `expect_doc` on all 15 source-3 questions. First
run: 14/15; "what does the site notes document say" answered from the readme with 2
citations — `site_notes.md` ranked **95/177** dense, and the independent sparse scan never
surfaced it either. Root cause (generic, every document source): BGE-M3's lexical vocab is
case-sensitive SentencePiece — `'site notes'` → ids 73048/1764, `'Site Notes'` →
18622/20897, zero overlap — so a lowercase question can never lexically meet a title-cased
heading. Fix: case-normalise chunk text before sparse encoding at ingest
(`ingestion/chunk_embedder.py`) and the query at retrieval (`query/rag_layer.py`); doc
path only, column/table sparse vectors untouched. Re-encoded source 3's 177 sparse rows
in place (dense embeddings unchanged); site_notes is now sparse rank 1. Weak spots the
assertions still tolerate, on record: "list the documents available" names only the
handbook; "notice period for termination" answers the probation clause. M6 retirement
list with battery coverage for the three new deterministic branches: in
`query-engine-open-items.md`.

**Item 4 — adapter branch deleted.** `source_coordinator._resolve_executable` /
`dispatch` no longer import the removed `query/source_adapters`; `_dispatch_flags` and
both config flags gone; the failing test deleted, the two flag-dependent tests rewritten
without the flag. Routing suite **64/64**.

**Also this pass:** the inference container had NOT been recreated (`Created`
2026-09-10); `docker compose up -d inference` is a no-op because compose sets
`OLLAMA_URL: http://host.docker.internal:11434` explicitly for `inference` and
`ingest-worker` — `.env` is not that key's authority, so the env-drift check now excludes
it. The NL explainer's `/api/generate` 404 against that host is a real, separate issue
(the host ollama serves `/api/chat` only for that model) — open.

**M2 first checkpoint — DONE; exit test passed** (flags still OFF by default; env-settable
for the one-process test: `QUERY_UNDERSTANDING_ENABLED=1 ANALYTICAL_SQL_V2=1`).

What landed. `understanding/grounding.py`: dimensions grounded to groupable anchor
columns (L2 semantic_type / lite role / data_type via `column_kind`), filters grounded on
DATA (the value arbiter's typed lookup → `=`/`!=`; numeric literals → comparator read
from the question or the LLM's value text, on a numeric column only; "have a X" →
`IS NOT NULL`), time → the anchor's TEMPORAL column, COUNT-DISTINCT over a named
dimension; anything named that grounds to nothing → typed clarify. Real tables = graph ∪
scoped model (lite sources have no graph edges → 0 graph tables → every concept refused
before). `analytical_spec.py`: grounded dimensions first, unresolvable GROUP BY → None
(never a scalar), `numeric` enforced, grounded WHERE into `build_aggregate_sql`, `list`
specs (predicate only) composed with the pipeline's projection. `pipeline.py`: Refusal
ADVISORY (existing path decides); grounded intent a CANDIDATE — never pins the anchor on
the first pass; the ONE authority moment is re-entry: when a deterministic answer is
REFUSED by a shape guard / planner, run once more with the fast path off and the grounded
candidate first (that is the operational meaning of "no deterministic branch matched").
Two guards fixed on the way: `_named_measure_columns` no longer treats a one-word column
("amount") as named by that word alone; `_boolean_flag_named` no longer reads "paid" as
`is_paid` when `paid_amount` exists on the same tables.

Exit, one process each, flags ON: battery `{"summary": "OK", "questions": 63,
"failures": 0, "known_gap_warns": 3}` — nine of the twelve typed refusals now ANSWER:
"how many properties are in Pune" → `COUNT(*) … WHERE LOWER(city_name) = %s`;
"total paid amount per currency" → `SUM(paid_amount) … GROUP BY currency_id`;
"properties with more than 3 floors" → `WHERE total_floors > 3`; "how many users have a
last login" → `WHERE last_login IS NOT NULL`; "total maintenance amount" →
`SUM(amount)`; "maintenance records with amount above 500" → `WHERE amount > 500`;
"total monthly fee" / "average monthly fee" → `SUM`/`AVG(monthly_fee)`; "how many amenity
categories are there" → `COUNT(DISTINCT category)`. Golden set flags ON: byte-identical
(0.1936/0.3734/0.4139/0.7619 and 0.2571/0.3907/0.479/0.7143). §5 table: 0 regressions —
"how many properties are there" and "show all vendors" answer as before, "average payment
amount broken down by currency" is the same typed clarify as flags OFF (the grounded
GROUP BY is `currency_id`; the guard wants the display column on `generics_currency`,
a join = M3). Flags OFF: `{"summary": "OK", "failures": 0, "known_gap_warns": 12}` —
unchanged. Routing suite 64/64. Still typed with flags on (3): "list properties in
Mumbai" (a list with a value filter: the arbiter's own state-column pick — the fast path
answers before the candidate is consulted and is not refused), "how many properties per
city" (fast path, same), "which project has the highest carpet area" (cross-table
superlative, M3).

---

## 14. Pre-M3 items + M3 checkpoint 1 (one firewall) + multi-source routing made real (2026-09-16 → 18)

**Baseline note:** the pass said "baseline committed as `<sha>`" — it was not; `git log`
still tops at `b0bc2eb` and the M2 pass sat uncommitted (34 files). Everything below is
on top of that, uncommitted.

### Pre-M3 (4 items) — what inspection found vs. the claims
1. **Re-entry eligibility — done.** `ground_entity` now records HOW the anchor grounded
   (`GROUND_EXACT|GLOSSARY|VOCAB|NAME_TOKENS|RETRIEVAL`, `GroundedIntent.anchor_method`);
   `reentry_eligible` is True only for the four name-evidence methods, and `pipeline.py`'s
   re-entry override checks it — a retrieval-top-table grounding stays candidate-only.
   Battery: "how many gizmos/widgets/gadgets are there" per source with `expect="refuse"`.
   **Found on the first flags-on battery: the rule was not enough** — the deterministic
   `_bare_count` branch counts whatever table retrieval ranked first with no name
   evidence at all (`COUNT(*) FROM worklists_ticketuser` for "gizmos"). The layer's
   eligibility gate only governs the candidate; the existing head still guesses. Open.
2. **Business names on source 2 — claim inspected, not stale, but not what it seemed.**
   L3 (`semantic_layer_v2`) emits `business_purpose` / `primary_entity` / `table_type`
   per table; there is NO `business_name` field in the stage or the model, and its
   "glossary" is a generic compliance vocabulary, not schema terms. So `0/178` is by
   design, not a stale model — an ingestion item. Grounding now also reads
   `primary_entity` as table vocabulary (`GROUND_VOCAB`).
3. **Loosened guards bounded by tests.** `total reminder amount` (`expect_table=
   reminders_reminder`) and `how many properties are gated` (filter expected) added.
4. **Explainer — claim stale.** It already went through `call_slm`; the 404 was
   `endpoint="generate"` (the host Metal ollama serves `/api/chat` only) plus
   `NL_SUMMARY_MODEL=qwen2.5:7b-instruct` which that host does not serve. Fixed: chat
   endpoint + `_nl_model()` picks the summary model only when `/api/tags` lists it
   (`OllamaBackend.served_models`). Live: "There are 6 vendors." / "Kochi has two vendors
   with ratings of 4.5 and 3.9 respectively." — LLM-composed, numeric guard intact.

### M3 checkpoint 1 — one firewall over (IR, SQL)
`veda/ir.py` (`QueryIR`, `IRFilter`, `IRMeasure`; adapters `from_grounded_intent`,
`from_query_intent`, `from_branch_state`, `partial`) and `veda/firewall.py`
(`check()` → `FirewallVerdict` ok|ungrounded|qualifier_dropped|shape_mismatch|ir_mismatch|
rbac|invalid; `check_federated()`; `qualifier_only()`). Every head goes through it: the
pipeline (staged — value/qualifier, alignment, IR-equivalence, RBAC+AST — because it
reacts differently per stage: salvage, re-entry), fast path, cache replay, Tier-2
envelope / shared-planner / IR, and the federated head (`_fed_compose` /
`_fed_compose_plan` in `federated_route.py`; `_generate_federated_sql` and the flat-SELECT
fallback DELETED — no free-form SQL reaches the executor). IR-vs-SQL structural checks
run when the IR is complete; text heuristics only for partial slots; `ir_partial` and
`checks_run` are traced (`explain._SECTIONS` gained `firewall` etc.). Exit greps: no
`federated_sql` / `_generate_federated_sql`; the gates are called only from
`veda/firewall.py` (docstring mentions aside; the cache-demotion site reads the message
via the same gate). Routing suite 64/64 after the refactor.

**Batteries (flags OFF / ON, 69 q):** 5 / 4 failures, `ir_partial` 33/56 answered
(`branch.full:SIMPLE` 13/13, `llm_sql`, `cache` — the partial heads checkpoint 2 must
complete). The failures: the three "gizmos" guesses (item 1 above), one adapter bug
("list all users" — `dimension_list` marked DISTINCT but the fast path's `SELECT
DISTINCT` shape wasn't matched by the IR-vs-SQL check), one battery-spec error (source 3
"gizmos" listed as a SQL question). **Golden set: byte-identical.** Host: 22 GB swap
used throughout; each battery ran 4–5 h instead of ~40 min.

### Cross-source battery → the multi-source finding, and the fix
`scripts/eval_cross_source_battery.py`: 32 pairs (source-4/5 questions pinned vs.
unpinned scope 2,3,4,5) + the §9.2 chat sequence. First run: **17/32 DIVERGENT** — every
single-source question ("top 3 vendors by rating", "total monthly fee") ran through the
federated structured planner unpinned and came back as a different shape (7 rows for a
LIMIT 3; a grouped SUM for a scalar). Root cause was upstream of the planner: the routing
coordinator runs in SHADOW, and its decisions were **wrong 9/12** (probe
`route_probe.py`): source 3 (the document source) won on tabular questions and a
`RELATIONSHIP_EDGE` MULTI fired on vendor questions. Three measured causes, three fixes
(`query/source_coordinator.py`, `source_evidence.py`, `routing_slm.py`, `config.py`):
- **Global top-10 evidence** over 1902 (src 2) / 9 (src 4) / 4 (src 5) column embeddings
  → the small sources were invisible. Now per-source top-k over columns AND the table
  embeddings (`table_embeddings_v2`). 3/12 → 7/12.
- **Kind-unfair tiering**: chunk cosine runs ~0.07 hotter than column/table cosine for
  equal relevance here (measured on three questions); the config floors assumed the
  opposite and, applied as a shift, made it worse (7 → 6). `kind_normalised_signal`
  subtracts `ROUTING_CHUNK_KIND_OFFSET` (0.07, measured — re-measure on a new corpus).
  7 → 10/12.
- **Value-overlap `cross_source_fk` edges** at Jaccard 0.001 (`4:asset_id ↔ 2:id`,
  `2:asset_country_id ↔ 4:asset_id`) rated HIGH by containment → `_edge_quality_ok`,
  `ROUTING_EDGE_MIN_JACCARD` 0.05.
- **SLM "NONE" at an evidence boundary** with a tabular leader above the absolute STRONG
  floor → route to the leader (`resolve_boundary`); a nonsense concept stays NO_MATCH.
  10 → **11/12**. The last miss ("how many properties are in Mumbai" → NO_MATCH: best hit
  `assets_asset` at 0.427, the vocabulary gap of item 2) falls to the legacy path under
  SHADOW and is answered/refused exactly as pinned.

**Then the three behaviours the user asked for:**
- (a) **decide the source, run the pinned path there** — `veda_hybrid`: a ROUTED/SINGLE
  decision is now authoritative (the condition the 09-10 note set — a SINGLE-vs-legacy
  regression test — is the cross-source battery); NO_MATCH/CLARIFY stay advisory.
  Live, unpinned: "how many amenities are there" → `ROUTED/SINGLE ['5']`, "There are 7
  amenities." — identical to pinned.
- (a′) **drill-down keeps the source** — `QueryFrame.source_id` (written from
  `engine_result["source_id"]`, which `pipeline._done` and `_agent_to_subresult` now
  stamp); `call_engine_node` narrows a drill/refine/compare follow-up to the frame's
  source. Live: chat 11 turn 2 ran `ROUTED/SINGLE ['5'] (SINGLE_CANDIDATE)` on the
  inherited scope (its filtered-count *answer* needs the M2 flags, off by instruction).
- (c) **several sources each answer → one summary** — `MultiResult.summary`
  (`_summarise_multi_answers`: small NL model over the per-source answers only, numeric
  guard, labelled-join fallback); one item per source is preserved (the routing test pins
  that); the chatbot lifts the summary into the reply and keeps `source_answers`.
  Not yet exercised live (needs Ollama free of the batteries).

**Behaviour (c), continued once Metal was back (same day) — five measured runs on
"how many maintenance records and how many amenities are there", scope 2,3,4,5
(table in `query-engine-open-items.md`):** the wrong federated answer was closed by a
question-IR gate on EVERY federated compose (`_question_gate`: plan-vs-question for
structured plans, SQL-vs-question for semi-join / free-form / per-metric SQL —
aggregation must match the intent, every named entity must be read, two entities + a
plain count is not a join). Then, one leak at a time: the legacy federate-first call in
`run_hybrid_query` ran before the decomposer (`_COMPOUND_HANDOFF` contextvar skips it);
the decomposer is globally off (`QUERY_DECOMPOSE_ENABLED=False`, it mis-splits join
questions) — the coordinator's compound handoff proceeds regardless, because a MULTI
decision with no join relation cannot be that mistake; each part ran on the primary
source ("32 amenities" from homzhub instead of 7) → `_run_sub` routes every part through
the coordinator; part 2 then inherited part 1's narrowed scope → snapshot/restore of the
ambient context around every part. `_fan_out` attaches `MultiResult.summary`
(`_with_summary`). **Run 5 (all of the above), unpinned scope 2,3,4,5:** part 1 →
source 4 *"There are 8 maintenance records."*, part 2 → source 5 *"There are 7
amenities."*, `summary` = *"There are 8 maintenance records and 7 amenities."* — each
figure from the source that owns it, equal to the pinned batteries; the two shared-noun
probes ("records with status open", "average rating") stay honest federated refusals.
Routing suite 64/64 on this code. Golden set on this code: byte-identical (fifth time —
0.1936/0.3734/0.4139/0.7619 and 0.2571/0.3907/0.479/0.7143). Final cross-source, flags-OFF
and flags-ON batteries on this exact code (2026-09-19/20, host swapping — each took 5–20 h):
- cross-source: `{"summary": "OK", "pairs": 32, "divergent": 0, "verdicts": {"same": 24,
  "typed_refusal": 5, "both_typed": 3}, "seq_9_2": "typed_refusal"}`
- flags OFF: `{"summary": "OK", "questions": 69, "failures": 0, "known_gap_warns": 14,
  "ir_partial": 31}`
- flags ON: line below when it lands.

**Cross-source battery after the fixes: `{"summary": "OK", "pairs": 32, "divergent": 0,
"verdicts": {"same": 24, "typed_refusal": 5, "both_typed": 3}, "seq_9_2": "typed_refusal"}`**
— from 17 divergent. 24 unpinned answers are byte-identical in shape and row count to
pinned; the 5 unpinned typed refusals are the flags-off M2 gaps (scalar SUM/AVG, filtered
count) refusing exactly as pinned; §9.2's follow-up refuses honestly (never the unfiltered
`for_sale_count`). Routing suite 64/64 on the final code.

**Bare-count anchor gate (item 1's real fix) + alias stopgap (item 2):** `_bare_count` now
requires the question to NAME the table (name tokens, curated glossary, or L3
`primary_entity`) — "how many gizmos/gadgets are there" → typed clarify; vendors / users /
properties still count. "properties" needed `veda_core/data/default/2/veda_entity_aliases.json`
(the per-source glossary the resolver always looked for and never found — L3 emits no
business names; `assets_asset`'s only vocabulary is "Asset"). Live probe 5/5 as intended.
**That file is NOT in git** (`veda_core/data/` is ignored, like every artifact) — its
content, to recreate on another machine: `{"property": "assets_asset", "properties":
"assets_asset", "payment": "accounts_paymenttransaction", "payments":
"accounts_paymenttransaction", "payment transaction": "accounts_paymenttransaction",
"payment transactions": "accounts_paymenttransaction"}` (keys starting with `_` are
ignored). A tracked seed + an ingestion stage that emits it is the proper fix (open-items).
**Flags-OFF battery on the final code: 2 failures / 69, `ir_partial` 31.** (a) source-3
"gizmos" — my spec listed it as a RAG question; pinned to a document source a count enters
the SQL head and refuses (correct) → spec fixed to `route="sql", expect="refuse"`. (b)
"which vendor has the highest rating" → `LIMIT 3` from head **`cache`** (13 answers via
cache in that run) — the verified-query cache served and WROTE rows (17 since 09-17)
despite `cache_back=False`: the authoritative SINGLE route rebuilt the `RequestContext`
from `source_id`/`tenant` only (`veda_hybrid._constrain_scope_to`; the coordinator's
scoring context likewise), silently dropping `cache_back` — and it would drop an RBAC
data scope the same way. Fixed: `RequestContext.narrowed(source_id)` carries every field;
both sites use it; the 17 rows purged. **Golden set: byte-identical** (fourth time).

**Flags-ON battery on the final code: `{"summary": "OK", "questions": 69, "failures": 0,
"known_gap_warns": 4, "ir_partial": 30}`** — the last exit test. Both batteries OK, cross-
source OK, golden set unchanged, routing suite 64/64.

**Behaviour (c) — measured, and it found the M3 item-4 flaw for real.** Three multi-source
probes unpinned: "how many records have status open" and "what is the average rating" →
`refused` (federated, honest — no per-source plan). "how many maintenance records and how
many amenities are there" → federated head **answered**: *"There are 4 maintenance records
and an average of 10.88 amenities…"* — pinned truth is 5 and 7; the structured planner
chose an AVG for a "how many" and the wrong grain, and the firewall passed it because the
federated IR is built FROM THE PLAN (`_fed_ir_from_structured`), so it validates the
planner against itself. The spec said derive the IR from `understanding.extract` +
grounding across the scope's sources; that is the checkpoint-2 prerequisite, not done
here. The INDEPENDENT/APPEND path (one answer per source → `MultiResult.summary`) was not
reached by any probe: the coordinator routes shared-noun questions to `federated`, not
`independent`. `MultiResult.summary` and the chat lift are in place but unexercised live.

---

## Where to look for more detail

- **`docs/backlog/ARCH_REVIEW_2026-09_RECONCILED.md`** — §11's full record: the review's five
  properties reconciled claim-by-claim, the per-source battery before/after, the amended M0–M6.
- **`docs/backlog/query-engine-open-items.md`** — the complete, itemized record of every fix
  across both sessions (P0-1 through P2-3, then the 2026-09-15 follow-up's grammar/intent-boost/
  MySQL work), each with exact evidence, the fix applied, and live verification transcripts. This
  is the single most detailed source if you need to verify or extend any specific item above.
- **`docs/backlog/QUERY_UNDERSTANDING_GAPS_AND_ROADMAP.md`** — §10's full diagnosis: both
  query-processing flows, why phrasing-specific fixes don't scale, the half-built generic
  understanding layer, live regression evidence, and the phased plan to actually close it.
- **`docs/INGESTION_AND_QUERY_PIPELINES.md`** — the one-read architecture walkthrough.
- **`docs/MULTI_SOURCE.md`** §0/§2/§7 — the coordinator's scoped-authoritative design and why.
- **`docs/RETRIEVAL.md`** §2 — the P1-1 boost-only fix and the eval-harness gap, in place.
- **`CLAUDE.md`** (repo root) — environment facts, the git-safety rule, the two-clones gotcha.

### 2026-09-20 — Metal offline: switched to this machine's own services

`.env` now: `OLLAMA_URL=http://host.docker.internal:11434` (the host's ollama, `qwen2.5-coder:7b`;
the compose `ollama` service holds the same model on CPU as a fallback) and
`METAL_EMBED_URL=` (blank → `ingestion/m3_encoder` and the reranker load BGE-M3 /
bge-reranker-v2-m3 from `/models/hf_cache` on CPU). Previous remote values are kept as
comments in `.env`; a timestamped copy of the old file is in the session scratchpad.
Containers `inference`, `ingest-worker`, `api`, `worker` were recreated (`up -d`, not
`restart`) — the api/worker pair had been carrying a stale `.43` Metal IP since 09-10.
Verified in the recreated `inference` container: BGE-M3 dense 1024-d in 5.9 s cold / sparse 0.2 s,
bge-reranker-v2-m3 loads in 1.6 s, all `backend=cpu`; `call_slm` → host ollama answers in 6.6 s;
no ENV DRIFT line. `.env` had a DUPLICATE `METAL_EMBED_URL` line (the later one overrode the
blank) — removed. The flags-ON battery + chat compound check killed by the recreate were
relaunched on this path. Expect slower embeds/reranks; results are otherwise unchanged (same
models, same weights).
To go back: restore the two lines and `docker compose up -d` the four services.

**Gotcha found right after the recreate (2026-09-20):** `localhost:8080` returned 502. nginx's
`upstream veda_api { server api:8000; }` resolves the `api` hostname ONCE at startup and
caches the IP; recreating the `api` container changed its IP (172.18.0.10 → .12) and nginx
kept proxying to the dead one. `docker compose restart nginx` fixes it. Rule: after ANY
`docker compose up -d` that recreates `api`, restart `nginx` too (or make the upstream
resolve dynamically — see the nginx note in open-items).
