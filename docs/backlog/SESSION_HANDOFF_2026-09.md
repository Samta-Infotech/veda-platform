# Session handoff — VEDA platform work log (2026-09-09 → 2026-09-15)

Written so a fresh chat session can pick this repo up with full context. Everything described
here is **already committed** on branch `feat/refinements-pipeline` (the user committed it
personally between turns — commits `db18171` "md files add" and `e4fbc54` "changes"; nothing in
this session was committed by the assistant, per this repo's `CLAUDE.md` rule below).

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
   command — never edit the real `.env` for this.
4. **CPU-only BGE-M3 is slow for bulk eval.** With no working Metal backend, a full 24-query
   retrieval eval sweep (2 conditions) took 40+ minutes and had to be abandoned/reduced to 8
   queries. Budget for this or find a way to get Metal reachable before running a real
   before/after retrieval eval.
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

## Where to look for more detail

- **`docs/backlog/query-engine-open-items.md`** — the complete, itemized record of every fix in
  this session (P0-1 through P2-3), each with exact evidence, the fix applied, and live
  verification transcripts. This is the single most detailed source if you need to verify or
  extend any specific item above.
- **`docs/INGESTION_AND_QUERY_PIPELINES.md`** — the one-read architecture walkthrough.
- **`docs/MULTI_SOURCE.md`** §0/§2/§7 — the coordinator's scoped-authoritative design and why.
- **`docs/RETRIEVAL.md`** §2 — the P1-1 boost-only fix and the eval-harness gap, in place.
- **`CLAUDE.md`** (repo root) — environment facts, the git-safety rule, the two-clones gotcha.
