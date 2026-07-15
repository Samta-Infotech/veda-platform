# VEDA — Semantic Bridge Between Unstructured Data and the Structured Semantic Layer

> **Question this doc answers:** today the chunk → structured-column bridge is
> *fixed-value* (exact string containment). Should it be *semantic* (embedding
> similarity) matching against the structured semantic layer — especially when
> the tenant has a large volume of unstructured data alongside structured?
>
> **Short answer:** Yes — as an **additive hybrid**, not a replacement. Add a
> semantic bridge for *retrieval recall*, keep the exact/value-overlap bridge for
> *executable SQL grounding*. The infrastructure to do this already exists; the
> gap is one edge type. Details, trade-offs, and the honest "is it better" verdict
> below.
>
> **Implementation status (Tier A — SHIPPED + LIVE-VERIFIED):** the semantic lane is
> wired into the ingestion pipeline. `veda_core/ingestion/semantic_linker.py` matches
> each chunk's M3 vector against the column vectors already in `graph_node_embeddings`
> (the structured semantic layer) and emits `semantic_about` (chunk→column) edges;
> invoked from `entity_linker.link_entities` after the exact/pattern detectors,
> idempotent via the same scoped delete, loaded tenant-wide by
> `graph_retriever._load_all_edges` so PPR traverses it, tuned by `SEMANTIC_BRIDGE_*`
> in `config.py` (env-overridable; `VEDA_SEMANTIC_BRIDGE=0` disables). No migration, no
> new table (Tier A reuses existing embeddings).
>
> *Live proof (2026-07-15):* a real re-ingest of the contracts doc source (src3, 8
> chunks) against homzhub's 1902 embedded columns emitted 14 `semantic_about` edges in
> 0.26s — e.g. a late-fee contract chunk bridged to `accounts_paymenttransaction.late_fee`
> (cos 0.626) and `accounts_userinvoice.late_fee` (0.624), columns the literal bridge
> never finds. Weights (~0.51–0.56 = 0.9·cos) sit below exact `value_of` (1.5) as designed.
>
> **Tier B (value-level) — BUILT, wiring live-verified:** `ingestion/value_embedder.py`
> embeds eligible sampled DISPLAY values into `entity_value_embeddings` (lazy table +
> HNSW, no migration) at structured ingest (source_dispatcher Step 7d);
> `semantic_linker.build_value_bridge` extracts candidate spans from chunks, ANN-matches
> them against that index, and emits entity nodes + `semantic_value_of` (entity→column,
> weight 1.1) + `mentions_entity`. Wired into `entity_linker`, tuned by `SEMANTIC_VALUE_*`.
> Pure span/matcher functions unit-tested.
>
> *Live proof (2026-07-15):* built the value index for homzhub (4829 value vectors / 276
> cols), re-ingested src3 → 61 `semantic_value_of` edges, e.g. doc spans
> "squash"/"football"/"intercom" → `assets_amenity.name`, "kochi" → `generics_city.name`
> + `generics_location.city_name`, "society charges"/"rent" → `accounts_generalledgercategory.name`.
> Safety verified: 0 column↔column semantic edges (all are chunk→column or entity→column),
> so neither tier can drive a federated SQL join. A prune bug found during verification —
> `_prune_orphan_entities` deleted semantic-only entities because they link via
> `semantic_value_of` not `value_of` — was fixed so the prune honours both link types.

---

## 1. Current behaviour (confirmed against code)

The bridge from the narrative world (chunks) to the structured semantic layer
(columns/values) is built at **ingest time** by
`veda_core/ingestion/entity_linker.py`. It is **purely lexical**:

- **Dictionary detector** (`detect_entities`, `entity_linker.py:190-197`): a
  column's sampled value links to a chunk only when the normalized value string
  **literally appears** in the chunk text — `if value_norm in norm:`.
  Normalization is casefold + strip + NFC + whitespace-collapse
  (`normalize_value` from `column_sketches`). No similarity, no embeddings.
- **Pattern detector** (`entity_linker.py:201-212`): typed regexes (email / money
  / ISO date / phone), and even these require an **exact-value hit** in the value
  index before admitting a link.
- **SLM detector**: documented as the recall-widener, but **not implemented** —
  the header says *"Hook left for the enrichment pass"* (`entity_linker.py:19-20`).

Consequence: a chunk saying **"ACME Corporation"** does **not** link to a column
value **"ACME-CORP"**; synonyms, abbreviations, inflections, and paraphrases are
all missed. Only literal post-normalization co-occurrence creates the
`chunk --mentions_entity--> entity --value_of--> column` bridge.

### What *is* already semantic (important — don't rebuild it)

| Moment | Mechanism | Semantic today? |
|---|---|---|
| Query → columns/chunks (query time) | M3 dense (1024-d) + sparse + PPR seeds | **Yes** |
| **Column/table nodes** embedding | `graph_embedder.embed_graph_nodes` → `graph_node_embeddings`, M3 1024-d, enriched text incl. sampled values (`entity_linker` is *not* involved) | **Yes** |
| **Chunk → column entity bridge** (ingest) | `entity_linker` exact string containment | **No — fixed** |
| Column → column cross-source FK (ingest) | MinHash / exact-containment over *values* | No (value overlap, by design) |

**Key fact for the whole approach:** columns and chunks are *already embedded in
the same M3 space* (`graph_embedder.py:207-208` embeds col/table nodes;
`:219-246` copies chunk vectors — same 1024-d space, explicitly noted at
`:215-217`). Query seeding already does cosine ANN across both
(`retrieve_graph_seeds`, `graph_embedder.py:366`). So a semantic chunk↔column
bridge is **not new infrastructure** — it is a new *edge* over vectors that
already exist.

---

## 2. Why this matters more with lots of unstructured data

The exact-match bridge has a **recall ceiling that scales badly with unstructured
volume**:

- More documents → more phrasings of the same real-world entity ("ACME Corp",
  "ACME", "A.C.M.E.", "the ACME account"). Exact match catches at most one form,
  so the *fraction* of genuine chunk↔column links you recover **falls** as the
  corpus grows.
- The whole value proposition of the cross-source graph — *"traverse from a
  contract clause to the invoice row for the same customer"* — depends on that
  bridge existing. Missed links = silently dropped cross-source context, and the
  composer never even sees the evidence.
- Precision is currently high (literal match rarely wrong), but that precision is
  bought by throwing away recall — the wrong trade when unstructured data is the
  bulk of the corpus.

So the pressure the user is describing is real: **at scale, fixed-value matching
under-links, and the graph's cross-source promise degrades.**

---

## 3. Proposed approach — additive hybrid bridge

Keep the exact/pattern detectors exactly as they are. **Add** a third,
embedding-based detector that produces a *distinct, weaker-tiered* edge.

### 3.1 The core idea

For each chunk, generate candidate spans, embed them (or reuse the chunk vector),
and match them by **cosine similarity** against the **value/column vectors that
already live in `graph_node_embeddings`** (and, optionally, a new per-value
embedding index). Emit a **`semantic_value_of`** edge when similarity ≥ a
threshold. Two granularities, pick per rollout stage:

- **Tier A — column-level (cheapest, ship first).** Match the chunk (or its
  section-heading-prefixed text) against **column node** vectors already in
  `graph_node_embeddings`. Produces `chunk --semantic_about--> column`. Zero new
  storage, reuses `retrieve_graph_seeds(node_types=["column"])`. Good for "this
  paragraph is about the *maintenance cost* column" topical links.
- **Tier B — value/span-level (higher precision bridge).** Extract candidate
  entity spans (noun phrases / SLM-salient proper nouns / capitalized n-grams),
  embed each, and match against **per-value embeddings**. This is the true
  "ACME Corporation" ↔ "ACME-CORP" bridge. Requires a new
  `entity_value_embeddings(col_id, value_norm, embedding)` index built at ingest
  from the same sampled values the sketch pass already reads.

### 3.2 Guardrails (this is what makes it safe)

Mirror the existing `cross_source_fk` tiering philosophy — **semantic edges guide
retrieval but never authorize a SQL join**:

1. **Tiered, never executable.** `semantic_value_of` / `semantic_about` edges get
   a **lower PPR weight** than exact `value_of` (e.g. 0.6 vs 1.5) and are
   **excluded from the join planner / graph-guard**. Federated SQL joins stay
   grounded on exact `value_of` + HIGH `cross_source_fk` only (the firewall
   contract in `CROSSSOURCE_GRAPH.md` §5.1.3 is unchanged). A fuzzy link can
   surface *evidence*; it can never fabricate a *join*.
2. **Threshold + margin.** Admit only cosine ≥ `SEMANTIC_BRIDGE_MIN_SIM` (start
   ~0.62 for M3, tune on the golden set) **and** a top-1 vs top-2 margin, so
   ambiguous spans that are "sort of near" many columns are dropped.
3. **Class/type compatibility gate.** Only bridge a span to a column whose
   semantic type is plausible (don't link a prose phrase to a numeric/id column);
   reuse `_is_metadata_col` / `_is_sensitive_col` exclusions verbatim.
4. **Admission rule preserved.** Same explosion control as today — an entity node
   is created only if it bridges ≥1 chunk AND ≥1 column.
5. **PII guard preserved.** `SENSITIVE_PATTERNS` exclusion and salted-hash email
   ids apply before any node exists — unchanged.
6. **Idempotency preserved.** `semantic_*` edges are stamped with the ingesting
   doc source and swept by the existing `_scoped_delete` + `_prune_orphan_entities`.

### 3.3 Where it plugs in (concrete)

- Add a `_semantic_detector(chunk_text, spans)` path inside
  `entity_linker.detect_entities` (or a sibling `semantic_linker.py`) that calls
  `m3_encoder.encode_dense` on spans and `retrieve_graph_seeds` / a value-vector
  ANN for candidates.
- New config in `veda_core/config.py`: `SEMANTIC_BRIDGE_ENABLED`,
  `SEMANTIC_BRIDGE_MIN_SIM`, `SEMANTIC_BRIDGE_MARGIN`, `SEMANTIC_BRIDGE_TOPK`,
  edge weights in `GRAPH_EDGE_WEIGHTS` (`semantic_value_of`, `semantic_about`).
- `graph_retriever._load_all_edges` already loads bridge edges tenant-wide — add
  the new edge types to that load and to `GRAPH_EDGE_WEIGHTS`; **no PPR change**
  (this is exactly why PPR replaced BFS — new weighted edges just work).
- Tier B only: a `value_embedder` pass alongside `column_sketches` in L2, writing
  `entity_value_embeddings` (lazy `CREATE TABLE`, HNSW cosine index — same pattern
  as `graph_node_embeddings`, so **no migration**).

---

## 4. Is it better than the current approach? — honest verdict

**For a corpus that is mostly unstructured: yes, as a hybrid — clearly.** But the
answer is not "semantic replaces fixed" — it's "semantic *adds a second lane*."

| Dimension | Current (fixed only) | Semantic-only (replace) | **Hybrid (recommended)** |
|---|---|---|---|
| Recall on paraphrase/synonym | Low, degrades with corpus size | High | **High** |
| Precision of bridge | Very high | **Drops** (false links) | High (exact) + tunable (semantic) |
| Safe to drive SQL joins | Yes | **No** — would fabricate joins | Yes (exact lane only drives SQL) |
| Cost at ingest | ~free (string scan) | Embedding per span | Embedding per span (bounded) |
| New storage / migration | None | Value-vector index | None (Tier A) / 1 lazy table (Tier B) |
| Explainability | Exact match = obvious | "≥0.62 cosine" is fuzzier | Both, tier-labeled in provenance |
| Failure mode | Silent under-linking | Silent **wrong** linking | Recall up, wrong-links quarantined to retrieval |

### When it is better
- Large / growing unstructured corpus with entity-name variation → **big recall
  win**, which is the dominant failure mode at scale.
- You want cross-source evidence ("the contract clause for this customer") even
  when the doc doesn't spell the value the way the DB stores it.

### When it is *not* better (and the risks)
- **Precision/false-bridge risk.** Semantic matching *will* create some wrong
  links. This is acceptable **only because** the guardrails quarantine them to
  retrieval (weak PPR weight, excluded from joins). If you let semantic edges into
  the join planner, you would get ungrounded/incorrect cross-source joins — a
  hard regression. Do not do that.
- **Ingest cost.** Embedding spans per chunk is not free. With "a lot of
  unstructured data," bound it: reuse the chunk vector for Tier A (zero extra
  encode), cap spans per chunk, and batch-encode. The value-vector index (Tier B)
  is a one-time cost per ingest, same order as the existing sketch pass.
- **Tuning burden.** The threshold is a real knob. Ship behind
  `SEMANTIC_BRIDGE_ENABLED`, calibrate `MIN_SIM`/`MARGIN` on the golden set before
  trusting it.
- **Not a substitute for the value-overlap FK discovery.** Column↔column join
  discovery (`cross_source_fk`) must stay value-overlap/MinHash based — semantics
  can't prove two columns share *keys*, only that they're *topically* alike.

### Verdict
> **Recommended: implement the hybrid (Tier A first, Tier B if the golden set
> shows the recall gap is value-level).** It strictly dominates the current
> approach for retrieval on a large unstructured corpus, adds no migration for
> Tier A, and — critically — takes on *zero* new risk to SQL correctness because
> the executable join path is untouched. A pure semantic replacement would be
> *worse*, not better: it trades a silent-under-link problem for a silent-wrong-
> link problem and endangers federated SQL. The value is in adding the lane, not
> swapping the lane.

---

## 5. Rollout & verification

1. **Tier A behind a flag.** Add `semantic_about` (chunk→column) using existing
   `graph_node_embeddings`. No new storage. Weight it below exact `value_of`.
2. **Extend the golden set.** In `evaluation/golden_cross_source.jsonl` add
   paraphrase-bridge cases (doc says "ACME Corporation", DB stores "ACME-CORP")
   that the exact bridge currently misses, plus negative cases (near-but-wrong
   spans that must NOT bridge).
3. **Measure with `scripts/retrieval_eval.py --cross-source`.** Track: entity
   bridge recall (should rise), bridge precision (must stay ≥ target — tune
   `MIN_SIM`), and — the safety check — **federated join precision must not move**
   (proves semantic edges never leaked into SQL joins).
4. **Tier B only if** Tier A's recall gain plateaus below target and the misses
   are value-level (need "ACME Corporation"↔"ACME-CORP", not "about the cost
   column"). Then build `entity_value_embeddings` + span matching.
5. **Provenance.** Tag semantic-bridged evidence in the composer's `provenance`
   array with its tier + similarity, so the UI (and audits) can tell an exact
   bridge from a semantic one.

## 6. One-paragraph summary for the reader

Today the doc→structured bridge is exact string matching; it is precise but
under-links, and that under-linking gets worse the more unstructured data you add.
The structured semantic layer is *already* embedded in the same M3 vector space as
your chunks, so the fix is small: add an embedding-similarity **bridge edge** as a
second, lower-weighted lane, keep the exact lane for anything that drives SQL, and
gate the semantic lane with a similarity threshold + type compatibility + the
existing admission/PII controls. This is **better than the current approach for a
large unstructured corpus** — but only as a hybrid; replacing exact matching with
semantic matching outright would hurt precision and endanger federated SQL
correctness.
