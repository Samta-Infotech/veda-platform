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

## Compose note (unrelated, found while verifying the above)

The local `pg_data` volume (494 MB, created 2026-07-05) is PG16-formatted; `docker-compose.yml`
had been bumped to `pgvector/pgvector:pg17`, which refuses to start against an older-major
data directory. Pinned `docker-compose.yml`'s `postgres` image back to `pg16` on
2026-09-10 (user's explicit choice — see the comment at that line) so the existing local data
survives. Re-bump to pg17 via a proper `pg_dumpall`-and-restore (or `pg_upgrade`) when
convenient; `docker-compose.demo.yml:30` still says `pg17` and has the same latent issue if
that override is ever used against this volume.
