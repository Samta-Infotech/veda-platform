# VEDA Query Pipeline — Root-Cause Architecture Plan

**Date:** 2026-07-12 · **Status:** AWAITING REVIEW (nothing implemented)
**Scope:** the entire class of anchor/routing/refusal/latency failures, not one query.
**Method:** every claim below is backed by evidence gathered 2026-07-11/12 — live traces,
in-container data probes, flag-bisected A/B suite runs (`required` / `supoff` / `alloff`
tags), and code inventory. Tactical fixes from the same dates are logged in
`ANCHOR_ROUTING_FIX_PLAN.md`; this document is the level above them.

---

## 1. Failure taxonomy (what actually goes wrong, with evidence)

| # | Failure class | Evidence |
|---|---------------|----------|
| F1 | **Wrong-table anchoring.** Dimension/measure words in the query ("category", "value") lexically boost unrelated tables; entity word can't match its fused Django table name. | Trigger query anchored to `assets_assetverificationdocumenttype`, then (post-tokenizer) `services_uservaluebundle`, then `reminders_remindercategory` — never `accounts_paymenttransaction`. |
| F2 | **Late refusal of contradictions knowable in milliseconds.** SQL is fully planned/generated, then a terminal validator rejects it. | q34: 97s → "references unknown column(s): ['currency']". q22/q37: 44–82s → "GROUP BY ['project_name'] not requested". Trigger query originally: 24s → `qualifier_dropped`. |
| F3 | **Confidence computed last, not first.** The entity-confidence number that triggers refusal is produced after full retrieval + rerank + graph expansion. | q54/q61/q62: 50–85s → "not confident which entity (0.63 < accept 0.65)". The inputs to that number exist ~2s into the pipeline. |
| F4 | **Query words never grounded against data domains.** A filter word that isn't a real value produces a generic refusal instead of a grounded alternative. | "completed payments": source's payment statuses are only CAPTURED / AUTHORIZED / CANCELLED (`payment_status_id` → LOV table). `column_values` has NO 'completed' anywhere. User gets "couldn't map 'payment'" instead of "statuses here are captured/authorized/cancelled — which?" |
| F5 | **Answerable aggregation shapes have no builder.** "which DIM has the highest MEASURE" is fully groundable, yet no deterministic plan shape exists for it. | q62 (grounded 'captured' variant) still refuses; superlative detection landed 2026-07-11 but nothing can consume it (`SUPERLATIVE_JOIN_ROUTING=False`). |
| F6 | **No latency governance.** Cold start on first query; no per-stage budgets; timeout instead of anytime-refusal. | q01: 63s (model warm-up inline). q07/q08/q19: 120s timeouts in every run. Heavy class runs 44–120s regardless of outcome. |
| F7 | **Nondeterminism in tie-breaks.** Equal-score siblings resolved by set-iteration order. | q25's anchor historically flips between processes with `PYTHONHASHSEED` (registry `_single_entity` ties; on file from earlier sessions). |

## 2. Root causes (why those failures keep happening)

The seven failure classes reduce to **five architectural root causes**. The mapping matrix
is at the end of this section.

### RC-A — Resolution is fragmented across surface forms
There is no single layer that answers "what does this query token refer to?". Instead:
**6 duplicate `_singularize` implementations, 38 ad-hoc `split("_")` tokenization sites,
19 files privately consulting the value store** (counted 2026-07-12). Table names,
column aliases, business purposes, domain synonyms, sampled values, and embeddings are
each consulted by different stages with different tokenizers and different matching
rules. Every resolution bug of the last months — fused table names, dimension words
boosting wrong tables, 'completed' not matching any value, junction noise — is one
symptom each of this fragmentation. Fixing them one call-site at a time is why the
codebase keeps accreting patches.

### RC-B — Decisions are stacked heuristics, not one scorer over typed evidence
"Which table is this query about?" is currently answered by **eleven interacting
mechanisms**: the router blend (semantic + column-max + lexical), name-candidate
injection, grain-hint override, dimension demotion, `score_anchors` (4 signals), IDF
rerank, value rerank, junction exemption, named-subject protection, the override
margin, and (since 2026-07-11) the single-table gate. Each was added to fix a real
failure; each interacts with the others; the 2026-07-11 session itself had to add IDF
weighting *because* subword segmentation exposed generic-word matches — the stack
generates its own next bug. The deeper defect: **anchor selection is really a
role-assignment problem** — "category" is a GROUP-BY dimension, "value" is a MEASURE,
"payments" is the ENTITY, "completed" is a FILTER value — and no amount of weighting
inside a name-token scorer can recover roles it never sees.

### RC-C — Expensive stages run before cheap contradictions are checked
The pipeline's order is: retrieval → expansion → rerank → routing → planning →
SLM/LLM SQL generation → **then** validation (column existence, value grounding,
qualifier completeness, group-by audit). Every terminal validator that fired in the
evidence (F2) consumed only the query text plus the schema/value index — inputs
available at millisecond cost *before* retrieval. The architecture pays 40–120s to
compute "no". Latency and refusal-quality are the same root cause seen from two sides.

### RC-D — No calibrated confidence contract between stages
The cross-encoder emits calibrated relevance (measured: irrelevant ≈ 1e-4, real ≈
0.05–0.9) — but its noise was normalized to 1.0 and treated as signal (fixed
tactically via `RERANK_NOISE_FLOOR`). The router emits 0.4-confidence anchors that
proceed exactly like 0.9-confidence ones. Margins gate some paths (multi-table) and
not others (single-table, until 2026-07-11). Tie-breaks fall through to hash order
(F7). There is no uniform (decision, confidence, cost) contract an orchestrator could
route on.

### RC-E — Ingestion doesn't finish the job
Ingestion already builds a semantic model, relationship graph, sampled values, aliases,
and domain synonyms — but stops short of three derived artifacts the runtime then
half-rebuilds per query, badly: (1) **name tokens** for fused identifiers (the
2026-07-11 runtime tokenizer is the stopgap); (2) **value→referent closure** — 'captured'
is stored as a row of the LOV table, not as evidence for
`accounts_paymenttransaction.payment_status_id`, so value evidence can't vote for the
transaction table; (3) **per-column value domains** for grounded clarifies (F4). All
three are cheap, deterministic, offline computations.

### Mapping

| | RC-A frag. resolution | RC-B heuristic stack | RC-C late validation | RC-D no confidence contract | RC-E ingestion gaps |
|---|---|---|---|---|---|
| F1 wrong anchor | ✕ | ✕ | | ✕ | ✕ |
| F2 late refusal | ✕ | | ✕ | | ✕ |
| F3 late confidence | | ✕ | ✕ | ✕ | |
| F4 ungrounded values | ✕ | | ✕ | | ✕ |
| F5 missing agg shape | | ✕ | | | |
| F6 latency | | | ✕ | ✕ | |
| F7 nondeterminism | | ✕ | | ✕ | |

## 3. Target architecture

Four components, all evolutions of existing machinery — **no rewrite**:

```
                    ┌──────────────────────────────────────────────┐
   INGESTION        │  existing enrichment  +  (E1) name tokens    │
   (offline)        │  (E2) value→referent FK closure              │
                    │  (E3) per-column value domains               │
                    └───────────────┬──────────────────────────────┘
                                    ▼
                    ┌──────────────────────────────────────────────┐
   1. QSR           │  resolve(query) → typed referents in ~ms:    │
   Query Semantic   │  ENTITY(table) · DIMENSION(col) · MEASURE(col)│
   Resolution       │  VALUE(col≈val, FK-closed) · TEMPORAL ·      │
   (one index,      │  GRAMMAR(operator) · UNRESOLVED(+nearest)    │
   one tokenizer)   └───────────────┬──────────────────────────────┘
                                    ▼
                    ┌──────────────────────────────────────────────┐
   2. Typed plan    │  roles → anchor evidence (ENTITY + VALUE-    │
   contract         │  ownership only; DIM/MEASURE words can no    │
                    │  longer claim subjecthood) → plan SHAPE      │
                    │  (lookup/list/agg/grouped/superlative/       │
                    │  existence/window) → deterministic builders  │
                    └───────────────┬──────────────────────────────┘
                                    ▼
                    ┌──────────────────────────────────────────────┐
   3. Orchestrator  │  every stage: (decision, confidence, cost)   │
   (budgets +       │  high-conf + resolved → deterministic path   │
   staged commit)   │  UNRESOLVED content word → grounded clarify  │
                    │  in <1s (F4)  ·  ambiguous → pay for         │
                    │  retrieval/rerank/LLM under explicit budgets │
                    │  · anytime refusal, never silent timeout     │
                    └───────────────┬──────────────────────────────┘
                                    ▼
                    ┌──────────────────────────────────────────────┐
   4. Golden        │  expected anchor + shape per query (wrong-   │
   harness          │  table answers become failures) · confidence │
                    │  calibration from traces · latency budgets   │
                    │  in CI · PYTHONHASHSEED determinism test     │
                    └──────────────────────────────────────────────┘
```

Key properties:
- **One resolver, everywhere.** `value_arbiter`, the qualifier gate, graph-expansion
  seeds, anchor lexical evidence, and the feedback generator all read QSR. The 6
  singularizers and 38 tokenization sites collapse into one module. A resolution bug
  becomes fixable in exactly one place.
- **Roles before tables.** The trigger query becomes: ENTITY=payments→
  `accounts_paymenttransaction` (via ingestion-time name tokens), DIMENSION=category,
  MEASURE=value→`paid_amount` (alias/candidate_measure_columns), FILTER=completed→
  UNRESOLVED with domain {captured, authorized, cancelled} ⇒ grounded clarify in ~1s.
  Same machinery answers q62 outright (superlative shape over grounded referents).
- **Validation moves to the front.** Column-existence, value-grounding, and
  qualifier-coverage checks run on the QSR output *pre-planning*; the terminal
  validators remain as a second line, not the only line.
- **Refuse-over-guess is preserved and gets cheaper.** Nothing here makes the system
  answer 'completed'≈'captured' by fiat; it makes the refusal grounded, instant, and
  actionable.

## 3.1 Implementation log (running)

**Phase 0 — DONE (2026-07-12), baseline collecting.** Golden-anchor harness in
`evaluation/nl_query_suite.py` (GOLDEN_ANCHORS for 24 queries; wrong-table ANSWERS now
fail; `--recheck` replays goldens over stored results offline). Per-stage `_ms` stamped
into explain-trace sections (`veda/explain.py::_sec`). `evaluation/determinism_check.py`
(two-seed, 10 queries, 0 divergent). **First catch, minutes after landing:** q03
("properties with the most expensive financial records") has been answering
`SELECT COUNT(*) FROM assets_asset` — wrong table AND wrong shape — scored PASS by
answerability in every earlier run.

**Phase A — core DONE (2026-07-12), remainder listed.**
- `ingestion/value_referents.py` (derived pass, LLM-free, no re-understanding): value →
  referent closure with PER-EDGE PRECISION — one `SELECT DISTINCT label FROM R JOIN T`
  per FK edge prunes labels not actually reachable through that FK (30,647 spurious
  pairs pruned; 'captured' closes to `payment_status_id` ONLY). Also emits per-edge
  label domains (`edge_domains`) — the grounded-clarify payload.
- `query/resolution.py` (QSR): `resolve()` types every span
  ENTITY/DIMENSION/MEASURE/VALUE(direct+closed)/GRAMMAR/UNRESOLVED from the semantic
  model's own metadata; `typed_value_lookup()` (direct-only, predicate-safe),
  `closed_value_tables()` (routing evidence), `domain_via()` (FK-scoped domains,
  meta-column pollution excluded). 20-check acceptance suite green
  (`tests/test_qsr_resolution.py`).
- **Major latent defect found and fixed while migrating consumers:** the arbiter-based
  value lookups (`routing.py` VALUE_ANCHOR_RERANK, `pipeline.py` L4c arbiter) passed the
  SOURCE-DB connection to `column_values_typed_lookup` — but `column_values` only exists
  in the INTERNAL store. Verified live: the lookup returned `[]` for every token. **The
  value-evidence channel was silently dead in production** behind fail-open
  try/excepts — a textbook RC-D exhibit. Both call sites now use the QSR lookup;
  closure evidence credits referencing tables at `VALUE_ANCHOR_CLOSED_WEIGHT = 0.15`.
- Expansion-seed hygiene: `graph/query_graph.py::suggest_expansions` seeds content
  words only (language-layer contract shared with the qualifier gate) — kills
  "all" → "24/7 access" pollution.
- Singularizer audit: the "6 duplicates" are mostly canonical-import-with-fallback
  (already consolidated de facto); only ingestion's `relationship_graph.rstrip('s')`
  diverges — left untouched (ingestion-side), noted.
- **Remainder:** qualifier-gate vocab migration to QSR (parity-sensitive — after golden
  baseline lands); value_resolver/fast_path lookups to artifact-first (latency, they're
  mirror-alive); emit name-tokens into the semantic model at ingestion (optimization).

**Phase B — superlative builder DONE (2026-07-12); typed score_anchors rewrite pending
golden results.** `query/superlative_plan.py` (`SUPERLATIVE_PLAN_ENABLED`, wired into the
pipeline before retrieval/cache): grouped ranked aggregation straight from QSR referents,
~230 ms warm. Typed-role discipline proven on the probe family:
- "…highest **paid amount** among **captured** payments" → deterministic PLAN on
  `accounts_paymenttransaction`: GROUP BY `transaction_type` (resolved via the ingestion
  alias 'payment category' / phrase scoring), `payment_status_id IN (SELECT id FROM lov
  WHERE code='CAPTURED')` — executed: CREDIT 5,421,121.36 (90 rows) vs DEBIT 676,577.
- "…highest **value**…" → grounded clarify (which amount: balance/expected/paid/…) —
  'value' is genuinely ambiguous; asking is correct.
- "…**completed** payments" → falls through to the honest full-pipeline refusal
  (unconsumed-qualifier guard: a span with referents the plan doesn't use = a filter it
  would silently drop).
Design rules that emerged (all generic): per-(span, table) evidence counts ONCE (else
shared lookups outvote everything through their own rows); a span whose value FK-closes
elsewhere marks its direct table as a LABEL STORE (demoted); ONE ROLE PER SPAN (words
naming the measure/dimension are spent — 'paid' must not also become order_status='PAID');
the user's own words disambiguate via matched-word scoring for both measures and
dimensions. **Golden harness catches during baseline:** q03 AND q30 both answering from
wrong tables with wrong shapes (bare COUNTs), invisible to answerability scoring.

**Validation cycle 1 (2026-07-12, golden0 baseline 63/63 vs `after` targeted set):**
- Baseline locked: 35 PASS / 14 REFUSED / 9 TIMEOUT / 2 ERROR / **3 GOLDEN-FAIL**
  (q03, q30, q35 — wrong-table answers invisible before the harness), p50 66.3s, p95 120s.
- **Fixed in-cycle:** builder plans now pass the AST validator (full column manifest);
  and `veda_hybrid.py` no longer sends CLARIFY results to Tier-2 — a clarify is a
  grounded question, not a failure; the LLM retry burned 40–80s and could override the
  safe question with guessed SQL. Measured: q62 clarify 41.8s → **3.7s**; the
  answerable paid-amount variant → **2.4s with correct data** (was 86s qualifier_dropped
  pre-batch).
- **Remaining, mapped to plan items:** (1) q03/q30/q35 wrong-table answers persist —
  they are FULL-pipeline anchoring failures: the typed score_anchors rewrite (Phase B
  second half) is their fix; (2) heavy queries drifted +15–25s vs baseline (q25/q28/
  q30/q35) — the RESURRECTED value channel now does real work on every L4c pass;
  budget/caching in Phase C (per-stage `_ms` is live for profiling); (3) q37/q54
  budget-edge flips to TIMEOUT — same Phase C scope; (4) q61 ('completed') still ~88s:
  the REFUSAL path legitimately tries Tier-2 — Phase C's pre-planning value-grounding
  check converts it to a fast grounded clarify instead.

**Phases C/D — landed 2026-07-12 (validation cycle 2 in flight):**
- **Third dead safety rail found:** `_tier2_validate` (value grounding + qualifier gate
  + IR equivalence for LLM answers) was defined and NEVER CALLED — both LLM lanes
  (tier2-IR and envelope) shipped answers with only the AST firewall. That's how
  `SELECT * FROM assets_asset` shipped for "most expensive financial records" (q03)
  and bare COUNTs for q28/q30.
- **Wrong-table blind spot closed:** `qualifier_completeness(strict=True)` — in strict
  mode an unaccounted token with a QSR referent ANYWHERE in the schema is a dropped
  qualifier (lenient mode treated everything the wrong table didn't know as filler).
  Strict is wired into `_tier2_validate`, which is now actually called on BOTH LLM
  lanes (before execution; with one IR repair retry). `tier2_rejected` reclassified
  REFUSED (it's the correctness gate declining, not an infra error).
- **Typed anchor evidence shared:** `resolution.typed_anchor_evidence()` (per-span-max,
  label-store demotion, grammar exclusion) now backs BOTH the superlative builder and a
  new flag-guarded rerank in `vet_primary` (`TYPED_ANCHOR_RERANK_*`) — one scorer, two
  lanes, no drift.
- **Phase D artifacts:** `evaluation/ci_checks.sh` (QSR acceptance → determinism →
  golden recheck → latency assertions), `evaluation/latency_assert.py` (ratcheting SLO
  gate: enforces what's achieved — fast-lane p50 < 5s, clarify p95 < 15s — reports
  distance to the 5s/30s target), `evaluation/calibration_report.py` +
  `calibration_golden0.md`. **Calibration finding:** answered vs refused anchor
  confidence distributions fully overlap (medians 0.53 vs 0.52) — empirical proof that
  threshold tuning cannot fix anchoring; typed evidence is the differentiator (RC-B
  confirmed by data).

**Validation cycle 2 — FINAL (2026-07-12, `after2` 16-query set vs golden0 baseline):**

| outcome | queries |
|---|---|
| Wrong answers now BLOCKED (were shipping silently) | q03 (`SELECT *` tier2), q30 (bare COUNT envelope), q25 (wrong-grain envelope — asked *per property*, answered per status) |
| Latency wins | q62 clarify 86.6s → **3.1s**; q15 46.8s → **1.5s**; q54 114.7 → 81.7s; fast lane p50 **1.9s**, clarify p95 **3.1s** |
| Stable | q02/q04/q24/q28/q63 pass unchanged; q22/q61 same honest refusals; q34 timeout unchanged (heavy lane) |
| Costs, accepted | q37 flourish word 'market' now over-blocks a previously-passing Tier-2 answer (strict-gate sharp edge, tracked); refusal-lane latency unchanged (Tier-2 still burns 40–90s before declining — next batch) |
| Open wrong answer | **q35 only** — comes from the fast-path lane, which bypasses vet_primary and the strict gate; fix = same unconsumed-qualifier discipline on fast-path emissions (tracked) |

`evaluation/ci_checks.sh` runs the whole chain and correctly exits 1 on q35 — the gate
is honest (a stored GOLDEN-FAIL re-fails; found and fixed a recheck loophole that
skipped already-marked failures). `tier2_rejected` reclassified as REFUSED in both the
engine and the suite. **Batch status: Phases 0/A/B/C/D delivered; residuals tracked**
(q35 fast-path guard, score_anchors mechanism retirement, heavy-lane budgets, golden
expansion incl. grain assertions, strict-gate referent-strength threshold).

**Residuals stretch (2026-07-12, cont.):**
- **q35 eliminated — and a FOURTH answer lane discovered.** The wrong-table list was
  served from the **verified-query cache**, replaying SQL "verified" under pre-fix code
  (`intent=SIMPLE | cache` in the serve log). Fixes: (1) evidence guard on the
  fast-path lane — an answer built ONLY on tables with zero typed evidence is
  **DEMOTED to the full pipeline**, not refused (strict-refusing was measured flipping
  good answers on descriptor words); (2) the same guard on the cached lane; (3) the
  poisoned entry purged via Django ORM (`VerifiedQueryCache`, 1 row). The cache guard
  immediately caught a SECOND poisoned near-match (`assets_asset`) live. q35 now
  refuses honestly. Flags: `FASTPATH_EVIDENCE_GUARD`, `QSR_FP_EVIDENCE_FLOOR`.
- **Strict-gate strength rules hardened by measurement.** Replays showed two traps:
  (a) alias word-soup makes prepositions ('across', 'toward', 'along') look like
  referents — strict now keys on column-NAME tokens + entity idf + values only;
  (b) replays must run IN the container — the host lacks the domain-synonyms file and
  produces false refusals. 'between' added to the quantity grammar (was missing).
- **Tier-2 gate failures no longer retry.** A dropped qualifier isn't fixable by IR
  nudging; each retry was a full SLM round (measured >200s on q61). Gate fail →
  immediate `tier2_rejected` (REFUSED).
- **Grounded-clarify conversion** in the pipeline's qualifier-fail branch: a missing
  token with NO value referent + FK label domains on the queried table → clarify with
  the actual domain values, ranked by FK-column/query word overlap. (q61 doesn't hit it
  yet — its missing token is 'payment', which has referents; the statuses-clarify for
  q61 needs the anchor fixed first = mechanism-retirement scope.)
- **Ops root cause for this session's heavy-lane latency:** the host Metal embed
  server (port 11435, BGE-M3 + reranker on `mps`) was DOWN the whole session — every
  rerank/encode ran CPU-fallback inside the container. Restarted per
  `docs/MAC3_APP_HOST_SETUP.md`. All heavy-lane latencies measured this session are
  inflated by this; re-baseline after.
- **Still deferred** (task #6): score_anchors mechanism retirement (build
  `evaluation/anchor_eval.py` offline harness first), q61 anchor quality, per-stage
  budget caps, golden expansion to full suite.

**Question set v2 baseline (2026-07-12, 50 new questions, `v2set` tag, budgets DARK):**
50/50 run: **11 PASS · 17 REFUSED · 22 TIMEOUT · 0 GOLDEN-FAIL**, p50 61.6s.
Reading by section: payments/maintenance (q1–15) — groundable ones answer in 9–37s
(q07 total-collected 9.6s✓golden, q14 spread✓golden, q06/q15/q16); agreement/coverage
(q16–35) — splits between asset/lease-table answers (q31 5.7s, q34 9.7s) and honest
refusals; vendor ratings (q36–50) — **this source has no vendor-rating table** (q49
refuses with "unknown column 'rating'" — correct), so the section's only right outcomes
are refusals, and q37's PASS needs review (what did "average vendor rating" compute?).
**Every timeout is an unanswerable-vocabulary query burning 240s in unbudgeted Tier-2
before giving up** — the exact class the staged TIER2 budgets target. Refusal quality
is visibly upgraded: grounded clarifies ("'subject' doesn't match any value…"),
ambiguous-subject asks, and unknown-column rejections instead of silent wrong answers.

## 4. Phased plan

Phase 0 is deliberately first: **you cannot refactor a ranking system you cannot
measure.**

| Phase | Deliverable | Definition of done | Risk / rollback | Size |
|-------|-------------|--------------------|-----------------|------|
| **0. Golden harness** | Extend `evaluation/nl_query_suite.py`: per-query expected anchor table + plan shape (+ row-count bounds where stable); wrong-table answers FAIL; per-stage `ms` added to explain traces; `PYTHONHASHSEED` two-seed determinism check | Baseline report on current code; every later phase gated on "no golden regressions" | None (test-only) | ~1 session |
| **A. QSR** | (E1–E3) ingestion emits name-tokens + value→referent closure + value domains into the semantic model / value store; new `query/resolution.py` `resolve()` API; `value_arbiter`, qualifier-gate vocab, expansion seeds, anchor lexical migrate to it; delete private lookups + 6 duplicate singularizers | Behavior parity on golden suite; resolution consumers = 1 module; trigger family resolves payments→table, completed→UNRESOLVED{domain} | Parity-first migration behind `QSR_ENABLED`; per-consumer fallback to legacy path | ~2–3 sessions + one re-ingest/republish per source |
| **B. Typed planning** | Role assignment from `resolve()`; `score_anchors` rewritten over typed evidence (ENTITY/VALUE-ownership/FK-hub/calibrated-retrieval); retire IDF hack, dimension demotion, and ≥5 of the 11 stacked mechanisms; superlative + grouped-agg shape builders (answers q62/q54 class) | Trigger family anchors correctly; q62 answers; mechanism count 11→≤5; golden suite green | Single flag `TYPED_ANCHOR_ENABLED`; legacy scorer retained one release | ~2 sessions |
| **C. Orchestrator** | (decision, confidence, cost) contract; per-stage budgets; pre-planning validation (F2/F3 checks at ~ms cost); grounded clarify for UNRESOLVED content words (<1s); model warm-load at deploy behind healthz; candidate-width cap on RRF-kept path (the measured +20–70s) | p50 answered <5s, p95 <30s on suite; zero silent 120s timeouts (explicit budget refusals only); q34-class refuses in <2s with the actual reason | Budgets config-off = today's behavior | ~2 sessions |
| **D. Calibration** | Thresholds (accept 0.65, margins, floors) fitted from trace logs (confidence vs. correctness), not hand-picked; latency assertions in CI | Documented calibration table; CI red on drift | Keep hand values as floor | ~1 session, then continuous |

Sequencing: **0 → A → B → C → D**, each independently shippable. Total ≈ 8–9 working
sessions plus one re-ingestion per source.

## 5. What happens to the 2026-07-11 tactical fixes

| Tactical fix (uncommitted) | Fate in this plan |
|---|---|
| `semantic/name_tokens.py` runtime tokenizer + IDF | Moves to ingestion (E1); runtime module becomes thin reader; IDF *hack in the scorer* retired by typed roles (Phase B) |
| `RERANK_NOISE_FLOOR` | Kept; generalizes into the RC-D confidence contract (Phase C) |
| Single-table clarify gate | Kept; subsumed by orchestrator policy (Phase C) |
| `superlative_mode` detection (+ routing flag off) | Kept; consumed by the superlative shape builder (Phase B), flag then removed |
| Suite probes q61–q63 | Absorbed into golden harness (Phase 0) |

## 6. Rejected alternatives (and why)

1. **Full rewrite of the routing layer.** The validators, builders, graph, and
   ingestion enrichment are sound assets; the defect is *arrangement*, not substance.
   Rewrite risk ≫ refactor risk here.
2. **LLM-first routing** ("let the model pick the table"). Violates the platform's
   determinism/refuse-over-guess contract, un-debuggable regressions, and the evidence
   shows the failures are *resolution* failures the LLM would paper over silently.
3. **Auto-mapping unresolved values to nearest data value** ('completed'→'captured').
   Semantically unsafe (completed ≠ captured in payments); grounded clarify keeps the
   human in the loop at near-zero cost.
4. **Keep patching the heuristic stack.** This session is the counter-evidence: three
   sound patches (tokenizer, floor, gate) each moved the failure to the next
   mechanism. The stack's marginal fix cost now exceeds the layer's replacement cost.

## 7. Decisions (signed off by user, 2026-07-12)

1. **Latency SLO: p50 < 5s, p95 < 30s**, hard 60s anytime-refusal budget. Phase C
   targets these numbers.
2. **Re-ingestion: approved as a DERIVED pass only.** All three Phase-A artifacts
   (name tokens, value→referent closure, value domains) are LLM-free transforms of
   already-ingested artifacts (semantic model + relationship graph + column_values +
   source samples) — they join the existing "post-ingestion derived artifacts"
   family (config.py §~845) + registry republish. **Constraint: if any step turns out
   to require re-running LLM table/column understanding, STOP and ask the user first.**
3. **Grounded clarify: approved** as a first-class answer type. **Added scope:** audit
   the existing SLM feedback/suggestion generator — observed emitting column-name
   guesses (`payment_transaction_id`, `other_payment_type`) where value-domain
   suggestions were needed; replace its content with QSR-grounded suggestions
   (deterministic), SLM at most for phrasing.
4. **Migration: single batch.** Build 0 → A → B → C → D in order with the golden
   harness gating internally, but deliver as ONE body of work for user verification at
   the end — no per-phase sign-off pauses.
