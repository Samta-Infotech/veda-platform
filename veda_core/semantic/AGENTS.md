# veda_core/semantic/ — the deterministic (non-LLM) semantic registries

Compiles and serves the three business-facing registries (concepts / dimensions / metrics)
that power the deterministic fast path, plus the schema-vocabulary tokenizer. All pure
lookup — no model inference. Read at query time by `../query/fast_path.py` and routing.

| File | Role |
|------|------|
| `compile_semantic_layer.py` | **Offline compiler** (L5 `semantic_registry` stage). Pure function of `veda_semantic_model.json` → `concepts.json` (business noun → entity table), `dimensions.json` (groupable cols + values), `metrics.json` (named COUNT/SUM/AVG + grain), `MANIFEST.json`. Stamps `source_hash`. Scope-aware via `config` artifact paths. |
| `registry.py` | Runtime loader + matchers. Per-`(source, tenant)` cache (`_CACHE`), Redis-first (`VEDA_SM_REDIS`), file fallback. Matchers: `match_concepts`, `match_metric_labels`, `match_dimension(s)_in_table`, `match_value(s)_in_table`. `_reg_scope()` tries both `veda_core.context` and bare `context` module identities (dual-import hazard). |
| `name_tokens.py` | Schema-vocabulary table-name tokenizer. Segments opaque Django-style names (`accounts_paymenttransaction` → `[payment, transaction]`) using the source's own column/alias/synonym vocabulary — **no hardcoded word list**. `segment_token`, `table_tokens`, `token_table_idf`. Cached per scope. Flags `NAME_SUBWORD_SPLIT_ENABLED=True`, `NAME_SUBWORD_MIN_PIECE=3`. Consumed by anchor/routing lexical signals. |
| `overrides.json` | Human-declared `display_columns` overrides — highest authority, survives re-ingest. Currently 2 entries. |

## Gotchas
- The intent-boost deltas in `../retrieval/intent_boosting.py` (±0.1–0.6) dwarf the RRF
  score range — the semantic registries' role signals are load-bearing for anchor choice.
- `name_tokens.py` was folded up from the archived `ANCHOR_ROUTING_FIX_PLAN` — the
  data-derived generic-word discount (`token_table_idf`) is what stops a table name from
  drowning out the real subject.
