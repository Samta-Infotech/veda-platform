# Phase 2 — Retrieval Decision Layer Audit

Status: **audit complete, no code changed** — this is the report requested before any
implementation. All findings verified against actual code (file:line citations
throughout), not assumed from docstrings/comments.

---

## 1. Current Retrieval Architecture — execution diagram

```
User Query
    │
    ▼
Tier1 5-signal engine (veda_core/retrieval/retrieval_engine_phase3.py)
    ├─ Signal 1: BGE-M3 dense (raw query, semantic_search.py)
    ├─ Signal 2: BGE-M3 learned-sparse ("BM25" — BM25 no longer exists, WP3;
    │            sparse_ranker.py; carries query-enrichment/synonyms)
    ├─ Signal 3: FK subgraph proximity (table-connectivity, signal_builder.py)
    ├─ Signal 4: FK path / join-key signal (signal_builder.py)
    ├─ Signal 5: Value-index match (narrow, compile-time, capped-8-values index,
    │            signal_builder.py — DIFFERENT store than Tier2's value_filter.py)
    ├─ Signal 6 (undocumented): table-first prior (dense+sparse table ANN, WP4)
    ├─ Graph expansion booster (pipeline.py:271-286, suggest_expansions —
    │            purely additive, neutral score 0.0)
    ▼
RRF fusion (rrf_merger.py, k=60, FUSION_WEIGHTS — all currently 1.0/identity)
    ▼
Intent boosting + history-table penalty (intent_boosting.py — deltas ±0.10 to
    ±0.60, DWARFING the entire RRF score range of ~0-0.098)
    ▼
Cross-encoder rerank (pipeline.py:308-334, PRIMARY_RERANK_ENABLED — TWO SEPARATE
    call sites exist, see §2/§4: this one uses BARE "column_name table_name" text)
    ▼
Primary table selection (veda/routing.py::select_primary_table — blends semantic
    table-cosine + importance-weighted retrieval + lexical match; argmax, NO
    confidence/margin recorded here)
    ▼
Grain vet (veda/routing.py::vet_primary — grain-hint override, then
    score_anchors: lexical 0.40 + position 0.20 + retrieval 0.25 + graph 0.15,
    IDF re-rank, value-match re-rank; overrides only if margin ≥ 0.06; THIS is
    where confidence/margin/alternatives get recorded — only when it runs)
    ▼
Routing (veda/planning.py::try_multitable + pipeline.py's deterministic
    evidence cascade — NOT score-based; fixed-priority if/elif on structural
    evidence: answer_entity → fk → multihop → value_arbiter → temporal → ranked
    → LLM fallback)
    ▼
SQL Generation
```

**Key correction to the task's framing**: retrieval is not linear
"Embedding → BM25 → Graph → Value → RRF → Cross Encoder → Primary → Routing."
BM25 doesn't exist (replaced by learned-sparse). Graph appears at *three*
distinct points with different behavior (recall-booster, `score_anchors`'
`graph_sig` feature, and a fully separate PPR-scored retriever used only in
RAG/hybrid). There are two independent cross-encoder call sites with very
different input richness (§4). Confidence is recorded only conditionally, not
universally (§6).

---

## 2. Signal Audit

| Signal | Purpose | Strength | Weakness | Where consumed |
|---|---|---|---|---|
| BGE-M3 dense | Semantic/paraphrase match | Catches synonyms/rephrasing | Score discarded after fusion (only `final_score` survives) | RRF only |
| BGE-M3 learned-sparse ("BM25") | Lexical + enrichment-expansion match | Distinct channel from dense (enrichment routed here specifically) | Same score-discard issue | RRF only |
| FK subgraph (`subgraph_signal`) | Table connectivity | Surfaces join-relevant tables query text doesn't name | **Table-uniform** — every column in a well-connected table gets the identical score; doesn't discriminate columns at all | RRF only, weak discriminator |
| FK path (`fk_signal`) | Is this column a real join key | Genuine column-level signal (0.7/0.5/0.0) | None found | RRF only |
| Value index (Signal 5) | Query literal ↔ column's sampled data | Cheap, in-engine | **Duplicated concept** with Tier2's `query/value_filter.py` — separate index, capped at 8 "safe enum" values, can silently diverge from Tier2's richer live-sampled store | RRF only; Tier2 rebuilds its own separately |
| Table-first prior (6th, undocumented) | Table-level dense+sparse affinity | Real signal, correctly fused | Docstrings/logs still say "5-signal" — stale branding, observability gap | RRF only |
| RRF fusion | Combine all of the above | Formula correct, `k=60` reasonable | **`FUSION_WEIGHTS` are all `1.0`** — the WP6 tuning infrastructure (`scripts/tune_fusion_weights.py`) exists but was never actually run/committed against real weights | Feeds intent-boost → cutoff → rerank |
| Intent boosting / history penalty | Business-logic nudges (recency, entity-type match, etc.) | — | Deltas (±0.10 to ±0.60) are **3-60x larger** than the entire RRF score range (~0-0.098) — in any case where these fire, they (not the carefully-designed RRF fusion) actually decide the ranking | Directly overwrites ranking order |
| Cross-encoder rerank | Precision re-score of top-K | Correctly scale-guards the tail (§ pipeline.py:318-322) | **Two call sites, very different input richness** (§4) | Directly drives `final_score` → primary-table selection |

**Cross-cutting finding, not signal-specific**: `RetrievalResult` (`retrieval_engine_phase3.py:47-63`) declares fields for every individual signal
(`semantic_score`, `sparse_score`, `subgraph_score`, `fk_path_score`,
`value_index_score`, `rrf_score`, `boosted_score`) but the only constructor
(`_results_from_tuples`, lines 433-448) never populates them — they silently
default to `0.0`/`None`. **Every per-signal score is computed and then thrown
away.** No downstream code can ever know *why* a candidate ranked well (exact
value match vs. vague semantic similarity vs. a real FK join key) — only the
single fused `final_score` survives.

---

## 3. Primary Table Selection Audit

Two functions, `veda_core/veda/routing.py`, run in sequence
(`pipeline.py:345-349`):

**`select_primary_table(results, query, sm)`** (`routing.py:31-93`) — first pass:
```python
lex = (1.5 * len(matched) + 2.0 * coverage) if len(matched) >= 2 else 0.5 * coverage
sem = routed.get(t, 0.0)                 # table-embedding cosine
col = (max_score.get(t, 0.0) / hi) if hi else 0.0   # normalized best retrieval score
combined = 1.0 * sem + 0.5 * col + lex
```
Winner = argmax. **Retrieval score is NOT dominant** — its effective weight
(0.5) is smaller than semantic-cosine's (1.0), and lexical match can be larger
still. But **no confidence/margin/runner-up is recorded at this stage at all.**

**`vet_primary(query, primary, results, sm, trace)`** (`routing.py:96-268`) —
second pass, gated on `ANCHOR_VET_ROUTER=True` (default) and the primary being
graph-connected:
1. Explicit grain-hint override ("for each X"/"per X") — unconditional,
   `confidence=1.0`.
2. `score_anchors()` (`query/join_planner.py:79-160`), weights
   `ANCHOR_SCORING = {"lexical": 0.40, "position": 0.20, "retrieval": 0.25,
   "graph": 0.15}` (`config.py:1130-1135`). **Retrieval is only 25% of this
   weight; lexical (40%) dominates by design; graph connectivity is a real,
   independently-weighted signal (15%).**
3. IDF re-rank (rewards distinctive column matches) and value-match re-rank
   (rewards sampled-value hits), both flag-gated, both default-on.
4. Override only if the top candidate beats the router's pick by
   `ANCHOR_CONFIDENCE_MARGIN = 0.06` — a genuine margin gate, not a bare
   argmax.

**This is where `confidence`/`margin`/`alternatives` (top-5 competing
candidates with per-signal breakdown) actually get recorded** —
`routing.py:257-264` — but **only when `vet_primary` runs `score_anchors`** (primary
must be graph-connected, ≥2 candidates). A non-graph-connected primary, or
`ANCHOR_VET_ROUTER=False`, gets **zero confidence recorded anywhere**.

**Verdict**: contrary to a naive assumption, this is already sophisticated,
multi-signal, margin-gated decision logic — not "pick whatever has the highest
retrieval score." The real gaps are (a) `select_primary_table`'s own argmax
never records confidence, and (b) confidence is conditional, not universal.

---

## 4. Metadata Audit

Rich ingestion-time metadata (`business_role`, `business_domain`,
`business_definition`, `aliases`, `negative_aliases`, `user_query_patterns`,
`related_columns`, `value_pattern`/`value_handling`, `sample_values`,
`business_purpose`) is baked into a shared `retrieval_documents` string
(`ingestion/semantic_layer_v2.py:1104-1220`) that **is** correctly used for
both column embedding (`biencoder.py`) and the `query/reranker.py::rerank_columns`/
`rerank_tables` reranker call site (`_col_text`/`_precomputed_rerank_text`,
`reranker.py:148-170`, `290-306`).

**Confirmed dead fields** (generated, never consumed anywhere except their own
generation): `ColumnMetadata.business_examples` (zero downstream readers,
confirmed by repo-wide grep), `ColumnMetadata.null_meaning` (hardcoded to `""`
at construction — never even generated with real content despite the docstring
claiming it's LLM-sourced), `ColumnMetadata.column_domain` (transient
generation input only, its persisted value is never re-read).
`TableMetadata.primary_entity` is genuinely **display/explainability-only**
(`business_explain.py`'s dataset name) — absent from `retrieval_documents`,
table embedding, and table reranking.

**The one substantive waste, directly relevant to accuracy** (not just
cleanup): **`pipeline.py`'s own `PRIMARY_RERANK_ENABLED` block
(lines 308-334) is a *second, separate* cross-encoder call site** —
```python
_pairs = [[_search, f"{r.column_name} {r.table_name}"] for r in _head]
```
This bypasses `_rerank()`/`_col_text()`/the precomputed `rerank_docs` artifact
**entirely**, even though it calls the exact same model (`_get_reranker()`)
that `query/reranker.py`'s properly-enriched call site already uses correctly.
**Two reranker call sites, same model, wildly different input richness** — one
gets `business_definition`/`aliases`/`business_role`/etc., the other gets bare
`column_name table_name`. This second, thinner call site is specifically the
one that drives **primary-table/anchor selection** (§3) — the audit's stated
highest-priority target.

Secondary finding: table reranking (`ingestion/rerank_docs.py:36-43`,
`f"{name}: columns {col_list}"`) omits `business_purpose`, even though
`business_purpose` **is** used in table *embedding* text (`biencoder.py:210-216`)
— an inconsistency between embedding and reranking enrichment for tables.

Minor: `reranker.py:294-301` fetches `get_sampled_columns()` into a `sampled`
dict, but `_col_text()`'s implementation never references that parameter —
dead code from a superseded pre-WP7 approach.

---

## 5. Graph Audit

Two independent graph artifacts exist: `graph.query_graph.UnifiedGraph`
(`veda_unified_graph.json`) and the join-planner relationship graph
(`veda_relationship_graph.json`, loaded once via `veda.runtime.get_graph()` —
genuinely shared/cached as a process-level singleton).

- **Recall boosters are honestly additive-only**: `retrieval_v2.py`'s
  `graph_expand()` and Tier1's `suggest_expansions()` call
  (`pipeline.py:271-286`) both assign graph-sourced candidates a **neutral
  0.0 score** — purely widening the candidate pool, reranker decides.
- **But graph connectivity DOES feed a real score elsewhere**: `score_anchors`'
  `graph_sig` feature (`join_planner.py:145-149`, weight 0.15 in
  `ANCHOR_SCORING`) — FK-reachability between co-mentioned candidate tables —
  genuinely influences primary-table selection (§3). The unified KG also
  drives a discrete dimension-demotion/grain-hint filter in `vet_primary`
  (`routing.py:129-135, 167-184`).
- **A third, heavier mechanism** (`run_graph_retrieval`, PPR-based,
  `graph_retriever.py`) assigns genuine connectivity-derived scores (seed
  cosine, PageRank stationary probability, sibling/chunk decay) that DO survive
  into `RetrievalResult.similarity` — but this path only runs in the
  RAG/hybrid/federated route (`retrieval_select.py` Step 3), not the SQL path.
- **`try_multitable`** (join-vs-single-table routing) reuses the *same*
  relationship-graph singleton for junction-table detection and join-path BFS
  — genuine reuse, not duplication, between anchor-vetting and join routing.
- **Two confirmed waste points**: (1) `select_retrieval`'s documented
  `graph_result` dedup parameter ("pass pre-computed GraphRetrievalResult to
  avoid a double run") has **zero actual callers** — every call site omits it,
  so the expensive PPR retrieval runs fresh every single time. (2) Tier1's and
  Tier2's graph-recall boosters **independently reimplement the identical**
  synonym-resolution + FK-neighbour-reach traversal logic instead of both
  calling `UnifiedGraph.suggest_expansions()`, which already does exactly
  this.

---

## 6. Confidence Audit

Confidence **partially** exists already — this is an extension opportunity,
not a from-scratch build, per the task's own "prefer extending existing logic"
instruction:

- `anchor_selection.confidence` / `.margin` / `.alternatives` (top-5 competing
  candidates with per-signal breakdown) — `routing.py:257-264` — but **only
  populated when `vet_primary` actually runs `score_anchors`.**
  `select_primary_table`'s own first-pass argmax never records anything.
- `join_planning.confidence` (`join_planner.py:309-321`, structural
  hop/polymorphic/inferred-edge product) — separate concept, FK-graph-edge
  quality, not retrieval score.
- `target_selection`'s per-candidate `.confidence` (50/50 lexical+retrieval
  blend, bucketed accept/ambiguous/reject) — separate again, for *which other
  tables* join the anchor.
- The user-facing **"confidence" in `result_explainer.py`/`result_analyzer.py`
  is a DIFFERENT, downstream concept** — answer-trustworthiness, computed as
  `min(anchor_confidence, join_confidence)` (`pipeline.py:932-935`). It reuses
  (a) and (b) as inputs but is conceptually distinct, and **silently defaults
  to a vacuous `1.0`** whenever the upstream vetting step didn't fire (e.g.
  single-table queries where `vet_primary` never ran `score_anchors`) — this
  is misleading: a query that never had ANY confidence check reports maximum
  confidence.

**Recommendation** (matching the task's own request, "smallest possible
implementation"): a lightweight, **internal-only** `RetrievalDecision` object
— `selected_table`, `confidence`, `competing_candidates` (top-N with
per-signal breakdown), `supporting_signals` — populated **unconditionally**
by extending `select_primary_table`/`vet_primary` to always compute and record
this (today they only do so conditionally), not a new retrieval stage or a
parallel confidence system.

---

## 7. Improvement Opportunities (ranked)

| # | Improvement | Accuracy gain | Complexity | Architectural risk |
|---|---|---|---|---|
| 1 | Make `pipeline.py`'s `PRIMARY_RERANK_ENABLED` block use the SAME enriched rerank text (`_col_text`/precomputed `rerank_docs`) that `query/reranker.py` already uses correctly | **High** — the anchor-selection reranker currently sees bare names; this closes that gap with data/code that already exists and is already proven correct elsewhere | **Very low** — swap one line's text construction for an existing helper call | **Very low** — same model, same data, already used identically at another call site |
| 2 | Populate a `RetrievalDecision` object (confidence/margin/alternatives) universally, not just when `vet_primary` happens to run `score_anchors` | **Medium** (doesn't change routing itself, but enables informed downstream tuning/refusal decisions and fixes the misleading vacuous-1.0-confidence issue) | **Low-medium** — extends existing, working code paths | **Low** — purely additive, internal-only, no behavior change to what wins |
| 3 | Populate the already-declared per-signal score fields on `RetrievalResult` instead of discarding them | **Medium** (enables future confidence/decision logic to distinguish "won on exact value match" vs "won on vague similarity"; also unblocks real observability for tuning `FUSION_WEIGHTS`) | **Medium** — touches `rrf_merger.py` + `retrieval_engine_phase3.py` result construction | **Low** — additive fields only, no ranking-behavior change unless something new consumes them |
| — | Add `business_purpose` to table rerank text (`rerank_docs.py`) | Low-medium, same class as #1 but for tables | Very low | Very low |
| — | Wire the `graph_result` dedup parameter to real callers | Latency, not accuracy | Low | Low |
| — | Actually tune `FUSION_WEIGHTS` via the existing (unused) tuning script | Unknown until measured — requires a golden set + experiment, not just a code change | N/A (operational, not engineering) | N/A |
| — | Unify Tier1/Tier2 graph-recall-booster implementations to both call `UnifiedGraph.suggest_expansions()` | Low (dedup/maintainability, not accuracy) | Medium (behavior-preservation risk if the two have subtly diverged) | Medium |
| — | Delete dead metadata fields (`business_examples`, `null_meaning`, `column_domain`) | None (cleanup only) | Low | Low |

---

## 8. Final Recommendation

**1. Is the current 5-signal (really 6-signal) retrieval architecture
fundamentally sound?** Yes. Each signal is well-motivated and non-redundant in
purpose (dense semantic, lexical/enrichment, join-key identity, table
connectivity, value grounding, table-level prior). The architecture doesn't
need replacing.

**2. Is retrieval quality the real problem?** No — not primarily. All four
research threads converge on the same conclusion: the signals themselves are
reasonable; the losses happen **after** they're computed — via discarded
per-signal scores, unweighted (identity) fusion, an under-enriched second
reranker call site, and confidence that's recorded only conditionally.

**3. Is the problem primarily final decision logic (primary table selection /
routing / confidence)?** More precisely: it's an **information-loss problem
at specific, surgical points**, not a wholesale flaw in decision logic.
`vet_primary`/`score_anchors`/`try_multitable` are already sophisticated,
well-designed, multi-signal, margin-gated systems — contradicting a naive
assumption that this is score-only decision-making. The real, specific gaps
are: (a) intent-boosting deltas structurally dwarf the entire RRF score range,
so RRF's carefully-designed fusion is often decorative; (b) Tier1's own
cross-encoder call site is starved of metadata its sibling call site already
uses correctly; (c) confidence is conditional, not universal, and can report a
misleading vacuous `1.0`; (d) per-signal scores are computed and thrown away.

**4. Three highest-ROI improvements** (detailed in §7, table above):
1. Fix `pipeline.py`'s primary-anchor reranker call site to reuse the
   already-existing, already-correct enriched rerank text.
2. Make `RetrievalDecision` (confidence/margin/alternatives) always populated,
   not conditional.
3. Stop discarding per-signal scores on `RetrievalResult`.

All three reuse existing architecture exactly as instructed — no new models,
no new retrieval engine, no planner/prompt/API/SSE changes.

---

**Awaiting approval before any implementation, per instruction.**
