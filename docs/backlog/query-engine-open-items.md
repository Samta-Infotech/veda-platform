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

## ⚠ Possible bug — `storage_adapters/reader.py::ann_search` database target (HIGH, unverified)

- `reader._connection()` (`storage_adapters/reader.py:41`) connects to
  `dbname = os.environ.get("POSTGRES_DB", "veda")`. The live `.env` sets `POSTGRES_DB=veda`.
- `ann_search()` (`reader.py:257`) queries `column_embeddings_v2`. `veda_core/config.py:145`
  states that table lives in `VEDA_INTERNAL_DBNAME`, and the live `.env` sets that to
  `veda_engine` — **a different database on the same server**.
- The same `_connection()` also reads Django `substrate_fkedge` / `substrate_glossaryentry`
  / `substrate_verifiedquerycache`, which are in `veda`. One connection cannot see both DBs.
- `writer.py` uses `VEDA_INTERNAL_*` correctly for engine tables; `reader.py` did **not**
  follow suit when it switched off the (now-dropped) `column_embeddings_bge` Django mirror
  to `column_embeddings_v2`.
- **What to check:** run `SELECT count(*) FROM column_embeddings_v2` on the `veda` DB in
  the inference container. If it errors "relation does not exist", dense retrieval (Signal 1)
  has been silently returning zero rows in the deployed system, and retrieval is limping on
  learned-sparse + the structural signals. `CLAUDE.md` warns about exactly this class of
  swallowed cross-database error. `evaluation/benchmark_archive/…/INDEX.md` reports decent
  recall, so either `column_embeddings_v2` was also created in `veda`, or that benchmark ran
  on the CLI path (`VEDA_ANN_VIA_ADAPTER=0`, engine `db_config`), not the served adapter path.
- Fix if confirmed: give `reader.ann_search` (and `_resolve_ef_search`'s
  `SubstrateVersion` read stays on `veda`) its own `VEDA_INTERNAL_*` connection for the
  `column_embeddings_v2` query, the way `writer.sync_from_engine` does.
