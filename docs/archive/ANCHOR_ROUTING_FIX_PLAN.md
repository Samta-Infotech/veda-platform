# Anchor / Routing Fix Plan — "completed payments" mis-anchor class

**Date:** 2026-07-11 · **Status:** PHASES 1–3 IMPLEMENTED (uncommitted) — see §5 Implementation log
**Trigger case:** `"Which category contributes the highest value among all completed payments?"` →
routed to `assets_assetverificationdocumenttype`, refused `qualifier_dropped: "payment"` after 24.5s.

---

## 1. Root-cause chain (verified against the code, deepest first)

The refusal itself is the `qualifier_completeness` gate (`veda_core/veda/validation.py:306`)
working correctly — it caught SQL that was built against the wrong table. Four upstream
defects produced that SQL:

| # | Defect | Where | Evidence |
|---|--------|-------|----------|
| RC1 | **Lexical anchor signal is structurally dead on Django schemas.** Table tokenization splits on `_` only, with whole-token matching, so `payment` never matches `accounts_paymenttransaction` (`{'account','paymenttransaction'}`). | `query/join_planner.py:125,132` · `veda/routing.py:67,77,139,159,180,277` | Verified empirically: matched set is empty. Trace: `lexical: 0.0` for every real candidate. With lexical (w=0.40) + position (w=0.20) dead, anchoring reduces to `0.25·retrieval + 0.15·graph`; graph was 1.0 for all → decision rode on retrieval alone. |
| RC2 | **The retrieval signal that decided the anchor was normalized noise.** All `top_columns` scores are 0.0 (cross-encoder raw ≈1e-4); `score_anchors` normalizes by the max (`join_planner.py:144`), amplifying noise into "1.0 vs 0.912 vs 0.872". Graph-expansion synonyms poured in garbage seeds ("all"→"24/7 access", "completed"→inspection progress). | `veda/pipeline.py:195-206` (zero-score `_RR` rows) · `graph/query_graph.py:258` | Trace `retrieval.top_columns` + `anchor_selection.alternatives`. |
| RC3 | **Intent is hardwired to SIMPLE.** `from query_engine.intent_detector import IntentDetector` (`veda/pipeline.py:106`) — the `query_engine` module **does not exist anywhere in the repo**; the import always throws and `intent = "SIMPLE"`. `aggregate_mode()` (`veda/planning.py:49`) has no superlative-by-dimension grammar ("which X … highest Y"), so `aggregation: null` and `needs_join = False` → single-table planning. | `veda/pipeline.py:105-110,284` | `find . -name intent_detector.py` → nothing. Trace: `intent: SIMPLE, aggregation: null`. |
| RC4 | **The ambiguity abstention gate exists only on the multi-table path.** `ANCHOR_CONFIDENCE_GATE` is enforced at `veda/planning.py:324` but nowhere on the single-table path; `vet_primary` (`veda/routing.py`) uses the margin only to decide router *override*. Margin 0.022 < `ANCHOR_CONFIDENCE_MARGIN` 0.06 committed anyway. | `veda/routing.py:244-268` · `config.py:1058-1059` | Trace: `confidence 0.4, margin 0.022`, proceeded to `single_table`. |
| RC5 | **Value-anchor rerank can't see FK-hop values.** `accounts_paymenttransaction` holds `payment_status_id` (FK → status lookup); "completed" is a value of the *lookup* table, so `VALUE_ANCHOR_RERANK` (enabled, +0.25 — designed for exactly this rescue) never credited the payment table. | `veda/routing.py:230-242` · `query/value_arbiter.py:187` | Semantic model: the table has no status text column, only `payment_status_id`. |

**Key enabling fact for the fix:** the schema's own vocabulary (underscore tokens of all
column names + table names) contains `payment`, `transaction`, `verification`, `document`,
`bundle`, `sale`, `listing`, … — verified. Concatenated Django names can be segmented
deterministically from data the ingestion already has. **No hardcoded word lists needed**,
which matches the project's standing design rule ("schema + data decide, no word lists").

---

## 2. Design principles (from the codebase's own conventions)

- No hardcoded vocabulary or business mappings — everything derived from schema/data.
- Failure-safe: every new signal wrapped so any failure degrades to today's behavior.
- Flag-guarded + retunable from `config.py` without re-ingest.
- Refuse-over-guess is preserved: we make the *right* table winnable, and refuse
  *earlier and cheaper* (clarify at anchor time, not `qualifier_dropped` after 24s).

---

## 3. Phases

### Phase 0 — Baseline (no code changes)
1. Restart stack, confirm healthy: `docker restart veda-platform-inference-1`, wait `healthz`.
2. Run `python3 evaluation/nl_query_suite.py --url http://localhost:8080/api/v1/query --source-id 2 --timeout 120`; stash `nl_query_suite_report.md` as the A-side baseline.
3. Add the trigger query (and 2–3 phrasing variants) to `evaluation/nl_query_suite.py` `QUERIES` so the fix is measured by the same harness.
4. In-container probe (`from veda_core import context` — NOT `import context`): check the
   `column_values` store for `value_norm = 'completed'` → confirm which table.col holds it
   (expected: a payment-status lookup table). This decides Phase 4's exact shape.

### Phase 1 — Schema-vocabulary name tokenizer (fixes RC1; the core fix)
**New module** `veda_core/semantic/name_tokens.py`:
- Lazy per-process vocabulary built from `data/veda_semantic_model.json` (same load
  pattern as `query/intent.py:_table_columns`): all `_`-split tokens of column names,
  table names, and `base_aliases`, singularized, len > 2.
- `segment(token) -> list[str]`: if token ∈ vocab → `[token]`; else greedy
  **longest-match** segmentation over the vocab (min piece length 3, full coverage
  required — partial segmentation returns `[token]` unchanged, failure-safe).
  `paymenttransaction → [payment, transaction]`, `assetverificationdocumenttype →
  [asset, verification, document, type]`.
- `table_name_tokens(table) -> frozenset[str]`: `_`-split → segment each piece →
  singularize → drop connectives (`and/or/of/to/by`) — `lru_cache`d.
- Config flag `NAME_SUBWORD_SPLIT_ENABLED = True`; when False, returns today's tokens.

**Adopt at the anchor-critical call sites only** (bounded blast radius — NOT all 15
`split("_")` sites):
- `query/join_planner.py` `score_anchors` (~:125, :132) — both the specificity
  pre-pass and per-candidate token sets.
- `veda/routing.py` `select_primary_table` (:67 candidate injection, :77 lexical),
  `vet_primary` (:139 grain-exact, :159 named-candidates, :180 dim-demotion),
  `_name_toks` (:277 — also used by planning's `_requested`).
- `veda/planning.py:598` (`ttoks`).

**Explicitly unchanged:** `validation.py` `qualifier_completeness` — its `_accounted`
already does two-way ≥4-char substring matching, so concatenated names are absorbed
there; touching the refusal gate risks admitting dropped filters.

**Expected effect on the trigger query:** `accounts_paymenttransaction` gains lexical
(coverage 1/3 → lex 0.167 → +0.067 composite) + position signal, while
`assetverificationdocumenttype` stays at 0 — flips both the router's lexical term and
`score_anchors`. Also improves the sibling issues already on file (salelisting /
leaselisting ties in the registry fast path use the same token sets).

### Phase 2 — Superlative aggregation grammar + dead-import cleanup (fixes RC3)
1. Delete the dead `query_engine.intent_detector` import (`veda/pipeline.py:105-110`);
   derive `intent` from what actually exists: `existence_mode` / `aggregate_mode` /
   fast-path route — so the trace stops reporting a classifier that never runs.
2. Extend `aggregate_mode` (`veda/planning.py:49`) with the **superlative-by-dimension**
   pattern (grammar-level, no schema vocab):
   `which|what <dim-phrase> … (highest|largest|most|top|maximum|lowest|least|minimum) <measure-phrase>`
   → `{"superlative": "max"|"min", "top_n": 1}`. Wire `needs_join`
   (`veda/pipeline.py:284`) to also fire when `aggregate_mode` returns a superlative,
   so these queries get the join/grain planner (and its confidence gate) instead of
   `single_table`.
3. Fast path: extend the existing superlative/metric-group grammar in
   `query/fast_path.py` (`_metric_entity_group` family) to accept
   "which <dim> … highest <measure>" → `GROUP BY dim ORDER BY SUM(measure) DESC LIMIT 1`
   with grounded filters — the trigger query should ideally never reach retrieval at all.

### Phase 3 — Ambiguity gate on the single-table path (fixes RC4)
In `vet_primary` (`veda/routing.py`, after :244): when `ANCHOR_CONFIDENCE_GATE` and
- top-vs-second margin < `ANCHOR_CONFIDENCE_MARGIN`, **and**
- the *chosen* anchor has `lexical == 0` **and** no value-rerank boost (pure-noise pick,
  exactly the RC1/RC2 failure class),

return a clarify sentinel; pipeline maps it to the existing `clarify` action
(`veda/pipeline.py:287-290`) listing the top-2 candidates. Reuse planning.py's
`subject_clear` exemption so queries that plainly name their subject never regress.
The lexical==0 condition is the regression guard: any query that actually names an
anchor table keeps today's behavior.

### Phase 4 — FK-hop value-anchor credit (fixes RC5)
In the `VALUE_ANCHOR_RERANK` block (`veda/routing.py:230-242`): when a value match
lands on table `V` (e.g. status lookup), also credit candidates that reference `V`
via a **direct FK edge** in the relationship graph, at reduced weight
(`VALUE_ANCHOR_RERANK_FK_WEIGHT = 0.15`, new config). One hop only, and only when `V`
is a lookup/master table (small `table_type != TRANSACTION` or junction-excluded) —
bounded, flag-guarded, try/except like the existing block.
*(Exact shape confirmed by the Phase 0 probe of where 'completed' actually lives.)*

### Phase 5 — Graph-expansion seed hygiene (mitigates RC2, cheap)
`graph/query_graph.py:suggest_expansions`: drop seeds that are gate-strip/filler tokens
(reuse `validation._gate_strip()` vocabulary — "all", "among", "the" must not seed
synonym expansion; today "all" → "24/7 access" pollutes retrieval). Keep zero-score
`_RR` injection as-is; the anchor no longer depends on it once Phase 1 lands.

### Phase 6 — Verification (gate for done)
1. Restart inference container, wait healthz (required after any `veda_core/` edit).
2. **Trigger case E2E:** the query must either answer (grouped SUM over completed
   payments) via fast path/join plan, or at minimum clarify at anchor time in <5s —
   `qualifier_dropped` after 24s is a fail.
3. **A/B suite:** rerun `evaluation/nl_query_suite.py`; diff against Phase 0 baseline.
   Acceptance: no PASS→REFUSED/ERROR regressions; the known-good routes (q25, bare
   counts, MAX/MIN metrics from the current uncommitted stream) unchanged.
4. Unit tests for `name_tokens.segment` (segmentable, unsegmentable, min-length,
   vocab-miss fallback) and for the superlative grammar.
5. `PYTHONHASHSEED` stability spot-check (two fresh processes, same anchor) — the
   sibling-tie hash-order issue on file must not be worsened by new tie surfaces.

---

## 4. Ordering & risk

| Phase | Risk | Rollback |
|-------|------|----------|
| 1 | Low — additive tokens, flag-guarded; biggest win | `NAME_SUBWORD_SPLIT_ENABLED=False` |
| 2 | Low-med — `needs_join` widening sends more queries to the join planner | keep grammar, revert `needs_join` wiring |
| 3 | Med — could increase clarifies; guarded by lexical==0 condition | `ANCHOR_CONFIDENCE_GATE=False` restores today |
| 4 | Low — bounded FK-hop, flag-guarded | new flag off |
| 5 | Low | revert seed filter |

Recommended order: **0 → 1 → 6(partial) → 2 → 3 → 4 → 5 → 6(full)**. Phase 1 alone
likely flips the trigger case's anchor; measure after it before layering the rest.

Everything remains uncommitted per the standing instruction (do not commit unless asked);
this plan file is the handoff artifact.

---

## 5. Implementation log (2026-07-11, all uncommitted)

Constraint honored throughout: **zero hardcoded vocabulary** — every new signal is derived
from the scope's own semantic model, the reranker's calibrated output space, or the
language-layer grammar (config.QUERY_GRAMMAR, which is per-language, never per-schema).

**Done:**
- **Phase 1** — `veda_core/semantic/name_tokens.py` (new): per-(tenant, source-set)-scoped
  vocabulary from column names + generated aliases + domain-synonym phrases (deliberately
  NOT raw table-name tokens — those contain the fused junk being split, and admitting them
  both stops segmentation and risks false splits). Fewest-pieces full-cover segmentation,
  min piece 3. `table_tokens()` + `token_table_idf()`. Flags: `NAME_SUBWORD_SPLIT_ENABLED`,
  `NAME_SUBWORD_MIN_PIECE`. Adopted in `routing._name_toks` (which planning.py shares),
  `select_primary_table`, `vet_primary` (grain-exact / named / dim-demotion),
  `score_anchors` (+ new `sm=` param), `planning._first_pos`. Measured: 114 → 16 opaque
  long tokens across 178 tables; false-split guard holds (transaction/advertisement/… stay
  whole); ~0.08 ms/table cached.
  - **Addition beyond original plan:** IDF weighting of the lexical/position signals in
    `score_anchors` via `token_table_idf` — segmentation exposes generic words ("value",
    "type", app prefixes like "assets"), so matched tokens are weighted by rarity across
    table token sets. Purely schema-derived.
- **RC2 fix (was Phase 5 adjacent, pulled forward)** — `RERANK_NOISE_FLOOR = 0.01`
  (config + pipeline L2b): when the cross-encoder's BEST pair scores under the floor it
  found nothing relevant (measured: irrelevant ≈ 1e-4, weak-real ≈ 0.05, strong ≈ 0.9) —
  keep the RRF consensus order instead of overwriting with noise that anchor
  normalization stretched to 1.0.
- **Phase 2 (partial)** — dead `query_engine.intent_detector` import deleted (it never
  existed; intent was silently always SIMPLE). New `planning.superlative_mode()` (grammar:
  interrogative + QUERY_GRAMMAR superlative_max/min) → intent=AGGREGATE → `needs_join`
  fires, and the trace now reports `superlative`. Kept SEPARATE from `aggregate_mode`,
  whose dict drives the COUNT-based grain planner — a superlative ranks by a MEASURE and
  must not be fed to it.
- **Phase 3** — single-table ambiguity gate in `vet_primary` → `{"clarify": …}` handled in
  pipeline. Fires only when: sub-margin top-2, DISJOINT matched name tokens (two different
  subjects, not entity-vs-sibling), and the winner isn't the sentence-initial subject.
  Flags: `ANCHOR_SINGLE_GATE_ENABLED`, `ANCHOR_SUBJECT_POS_MIN`.
- **Phase 0 probes** — `column_values` has NO `completed` anywhere; payment status is
  `payment_status_id` → `list_of_values_listofvalue` (LOV pattern) and the LIVE values are
  only CAPTURED / AUTHORIZED / CANCELLED. **"Completed payments" is unanswerable from data
  in this source — a refusal/clarify is the CORRECT outcome for q61**, so the goal became
  refusing on the right context, cheaply.
- Suite probes q61–q63 added to `evaluation/nl_query_suite.py`.

**E2E state of the trigger query (all fixes live):** intent AGGREGATE (superlative max),
routes through join planning, refuses with "not confident which entity is requested" —
honest, on-path, no more SQL against a document-type table. Control q63 ("total paid
amount across all payment transactions") **answers correctly** off
`accounts_paymenttransaction` (SUM = 6,097,698.36). q62 ("captured payments") still
refuses safely: the dimension word "category" lexically boosts `reminders_remindercategory`
and the valuebundle family crowds the candidate set.

**Remaining (next session):**
1. **Superlative grain planner** — consume the interrogative dimension token ("which
   CATEGORY") as the GROUP-BY dimension (graph/registry-resolved, not an anchor-name
   match), resolve the superlative measure ("value" → candidate_measure_columns via
   aliases), plan `SELECT dim, SUM(measure) … GROUP BY dim ORDER BY 2 DESC LIMIT 1` on the
   entity table. This is the piece that answers q62 and flips q61 into a value-level
   clarify ("did you mean captured?").
2. **Phase 4** — FK-hop value credit in VALUE_ANCHOR_RERANK (value in a lookup/LOV table
   credits the referencing transaction table). Blocked design note: LOV tables are shared
   (113 rows, many list types), so the credit must key on the FK edge, not the table.
3. **Phase 5** — graph-expansion seed hygiene (filler seeds like "all" still expand).
4. Full-suite A/B: `--tag allfixes` run (in progress at session end) vs the partial
   `--tag phase1` run (q01–q12) — compare before trusting the gate defaults.

### 5.1 Targeted A/B results (2026-07-11/12, `--only` runs)

14 "required" queries (the 10 superlative-affected + q25 + q61–q63), then flag-bisect
runs on the 4 that timed out. Tags: `required` (all fixes on), `supoff`
(SUPERLATIVE_JOIN_ROUTING off), `alloff` (every session flag off = pre-session behavior).

| query | alloff (pre-session) | fixes on | reading |
|-------|----------------------|----------|---------|
| q03/q04/q15/q24/q28 | — | PASS | superlative-class fine |
| q25 (watch-item) | — | PASS (3 rows) | intact |
| q61 (trigger) | qualifier_dropped, wrong table | REFUSED honestly, payment context | goal state pre-grain-planner |
| q62 (captured) | — | REFUSED safely | needs grain planner |
| q63 (control) | — | PASS 0.7–1.1 s | payment table reachable |
| q19 | TIMEOUT 120 s | TIMEOUT 120 s | pre-existing, NOT ours |
| q22 | REFUSED 82 s | 117 s–TIMEOUT | +latency, was refusing anyway |
| q34 | REFUSED 97 s | TIMEOUT | +latency, was refusing anyway |
| q37 | REFUSED 44 s | PASS 117 s (rows=1) | now ANSWERS — trades time for answer |

**Conclusions:** (1) No correctness regressions found; q37 actually flipped refuse→answer.
(2) The session's changes add ~20–70 s on the heavy analytical class (q19/q22/q34/q37 were
already 44–120 s pre-session) — likely the noise floor keeping wider RRF candidate sets
that make downstream tiers do more work, NOT the tokenizer (cached, ~ms). Open item:
profile where the added time goes (trace `total_ms` sections) and consider capping
candidates when the floor keeps RRF order. (3) `SUPERLATIVE_JOIN_ROUTING = False` (new
flag): routing superlatives into join planning bought nothing until the grain planner can
consume them — detection/trace stays on, routing off.
