# VEDA — Ingestion (detailed)

Ingestion is the **offline build** that turns a live PostgreSQL schema into everything the
runtime trusts as fact: the semantic model, the vector embeddings, and the relationship
graph. The runtime **never** introspects the DB schema or infers relationships — it only
reads these artifacts.

Entry: `main.py` (`run_ingestion`) — the unified pipeline. It builds the semantic model,
the **live** embeddings (`column_embeddings_v2` / `table_embeddings_v2`, written by the BGE
biencoder), **and** the derived artifacts (relationship graph + `semantic/*.json`) in one run.
(`veda_ingestion.py` and `embed_only.py` were retired — their behaviour is folded into `main.py`.)

---

## What it produces

| Artifact | File / table | Consumed by |
|---|---|---|
| Semantic model | `data/veda_semantic_model.json` | routing, value-grounding, qualifier gate, grain |
| Column embeddings | pgvector `column_embeddings` (BGE-M3, 1024-dim, keyed by `col_id="t.c"`) | Signal 1 (semantic search) |
| Table embeddings | pgvector `table_embeddings` (keyed by `table_name`) | table routing (`route_tables_semantic`) |
| Domain synonyms / concept graph | `data/veda_domain_synonyms.json`, `veda_concept_graph.json` | query enrichment |
| Relationship graph | `data/veda_relationship_graph.json` | join planner (keys, paths, polymorphic, cardinality) |

---

## Running it

```bash
# Full ingestion: structural + semantic layer (Qwen) + live embeddings + derived artifacts
python3 main.py --ingestion-only

# Fast LLM-free re-embed: reuse the saved semantic model, skip the Qwen steps
# (semantic layer, synthetic-gen, fine-tune); still refreshes the derived artifacts.
python3 main.py --embed-only
```
The relationship graph (`data/veda_relationship_graph.json`) and the `semantic/*.json`
registry are rebuilt automatically as **Step 12** of `run_ingestion` (gated by
`DERIVED_ARTIFACTS_ENABLED`). To rebuild just those by hand:
```bash
python3 -m ingestion.relationship_graph        # relationship graph only
python3 -m semantic.compile_semantic_layer      # concepts/dimensions/metrics only
```

---

## The 3 steps of `veda_ingestion.py`

### [1/3] Schema load (`load_schema`)
Reads tables + columns from PostgreSQL (`schema/real_schema.py` owns INFORMATION_SCHEMA
access). `--tables N` limits scope; `--all` takes everything. Internal/embedding tables
(`column_embeddings`, `table_embeddings`) must stay in `exclude_tables` so VEDA never
treats its own stores as business tables.

### [2/3] Semantic layer (`run_full_semantic_layer` — see `SEMANTIC_LAYER.md`)
The 5-stage hybrid build (profiling → glossary → table understanding → column
understanding → retrieval docs) + post-processing (domain synonyms, concept graph,
deterministic overrides). LLM (Qwen via Ollama) is used only in stages 2–4; the rest is
deterministic. Writes `veda_semantic_model.json` and the synonym/concept files. This is
the slow step (LLM-bound) — `--embed-only` skips it on re-runs.

### [3/3] BGE-M3 embeddings (`store_bge_embeddings`)
- Embeds each column's **retrieval document** (Stage 5 output) → `column_embeddings`
  (`vector(1024)`, `ivfflat` cosine index, `lists=50`).
- Embeds each table's description (`build_table_texts`: name + business_purpose +
  columns) → `table_embeddings`.
- **Fingerprint caching** (`_fingerprint` / `EMBED_FINGERPRINT_FILE`): if the retrieval
  docs + table texts are unchanged, the (slow) re-embed is skipped.
- Normalizes embeddings (`normalize_embeddings=True`) so cosine = dot product at query time.

---

## Relationship graph build (`ingestion/relationship_graph.py`) — separate, fast (~17s/66 tables)

Builds `data/veda_relationship_graph.json` — the deterministic join foundation. Runs
independently of `veda_ingestion.py` (defaults its table set to whatever the semantic
model covers).

**Edge sources:**
1. **Declared FKs** (`_declared_fk_edges`) — from the live schema.
2. **Polymorphic edges** (`_polymorphic_edges`) — `object_id` + (`object_type`/`model_name`)
   pairs resolved by **data correlation**, not string matching: sample the `object_id`
   values per discriminator value, correlate against candidate key columns. Accept an edge
   **only** if the join key is a **string/business key** OR the discriminator value has
   **name-affinity** with the target table — this defeats the numeric-collision trap
   (small surrogate IDs that coincidentally overlap). Real example:
   `annotation_record.object_id → counterparty_details.counterparty_id` joins on the
   **business key**, not the numeric PK.

**Edge attributes:** `relationship_type`, `weight`, `cardinality` (1:1 / N:1 / N:M from
distinctness), `polymorphic`, `requires_predicate`, `discovery`, `confidence`.

**Edge weighting (load-bearing for join quality):**
| relationship_type | weight |
|---|---|
| `business_core`, `bridge` | 1 |
| `reference`, `lookup`, `polymorphic` | 2 |
| `audit`, `history` | **10** |

`audit` is assigned both by table-name (`_AUDIT_TABLE_RE`: `_history/_log/_audit/_archive`)
**and by column-name** (`_AUDIT_COL_RE`: `*_by_id`, `owned_by*`, `assigned_to*` → `user`).
The high weight is what stops the join planner from routing business joins *through* the
`user` hub ("edited by the same person" is not a relationship). Current 66-table graph:
~65 business_core (w1), 7 polymorphic (w2), 66 audit (w10), 138 edges total.

---

## Build order & dependencies
```
PostgreSQL schema
   └─[1] load_schema
        └─[2] semantic layer (LLM) ──→ veda_semantic_model.json + synonyms + concepts
             └─[3] BGE-M3 embeddings ──→ column_embeddings + table_embeddings
   (separately)
   └─ relationship_graph ──→ veda_relationship_graph.json   (reads semantic model for table set)
```
After a schema or data change: re-run `--all` (or `--embed-only` if only metadata
changed) **and** rebuild the relationship graph.

See also: `SEMANTIC_LAYER.md`, `RETRIEVAL.md`, `../PIPELINE.md`.
