# veda_core/ — the preserved engine

Moved verbatim from `veda-poc`, with three platform seams added. No Django import
below this line except through the seams. `config.py` is the single source of
truth for engine flags; a query's live behavior is decided almost entirely by the
defaults there.

See [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for the whole system and
[../docs/QUERY_ENGINE.md](../docs/QUERY_ENGINE.md) for the deterministic SQL head.

## Top-level files

| File | What it does |
|------|--------------|
| `veda_hybrid.py` | **The single public front door.** `run_hybrid_query()` owns the one query trace, runs the pre-dispatch layers (runtime-context, multi-source coordinator in shadow, federated route, decomposition-off), `classify()`, `_dispatch_single()` per modality, and the Tier-2 LLM-IR fallback. 2400+ lines, the largest surface here. |
| `config.py` | Every engine flag + constant. `VEDA_<FLAG>` env overrides the file default; 14 flags are bridged into Django settings. Read this before believing any "it's enabled" claim. |
| `context.py` | **Platform seam.** Ambient `RequestContext` (`source`, `tenant`, `source_ids`, `allowed_resources`); `current()` raises when unset (fail-closed). Set per HTTP request by the inference middleware. |
| `main.py` | Thin shim into the layered ingestion pipeline (`ingestion/dispatcher.dispatch`). Not a query entry point. |
| `datalake_bench.py`, `doc_bench.py` | Standalone benches; not imported by the serving path. |
| `__init__.py` | Package marker. |

## Subpackages

| Package | Role | Map |
|---------|------|-----|
| `veda/` | The deterministic L1–L7 head + the ~16-step firewall + Tier-1→Tier-2 handoff. The correctness path and default route. | [veda/AGENTS.md](veda/AGENTS.md) |
| `query/` | Front-door routing, IR/SLM seam, SQL builder + resolvers, the non-SQL heads (rag / hybrid / nosql), multi-source coordinator, federated route. | [query/AGENTS.md](query/AGENTS.md) |
| `query_engine/` | One dead rule-based intent classifier. Not wired. | [query_engine/AGENTS.md](query_engine/AGENTS.md) |
| `retrieval/` | The 6-signal retrieval spine (`retrieval_engine_phase3.py`): BGE-M3 dense + learned-sparse M3 + FK subgraph/path + value index + table-first prior → RRF(k=60) → intent boost → adaptive cutoff. | [retrieval/AGENTS.md](retrieval/AGENTS.md) |
| `ingestion/` | Offline L1 EXTRACT → L2 ANALYZE → L3 ENRICH → L4 INDEX → L5 PUBLISH build. Runs in a subprocess from `apps/ingestion`. | [ingestion/AGENTS.md](ingestion/AGENTS.md), [ingestion/layers/AGENTS.md](ingestion/layers/AGENTS.md) |
| `connectors/` | Source connectors (relational / document / nosql / datalake / tabular files); `build_connector()`. | [connectors/AGENTS.md](connectors/AGENTS.md) |
| `graph/` | Unified KG: `query_graph.suggest_expansions` (Tier-1 synonym/alias + 1-hop FK booster) and the PPR walk used by `query/graph_retriever.py`. | [graph/AGENTS.md](graph/AGENTS.md) |
| `semantic/` | Deterministic concept/dimension/metric registries + the schema-vocabulary tokenizer. | [semantic/AGENTS.md](semantic/AGENTS.md) |
| `schema/` | Schema access shims (`real_schema.py` re-export; `simulate_schema.py` POC fallback). | [schema/AGENTS.md](schema/AGENTS.md) |
| `glossary/` | Ingestion-built domain-synonym JSON artifacts (no `.py`). | — |
| `slm/` | **Platform seam.** `_call_slm.py::call_slm(prompt, *, purpose, …)` — Ollama / vLLM Strategy, one model for every purpose. `SLM_MODEL_NAME` / `SLM_TEMPERATURE` come from `.env` (not `config.py`). `_slm_circuit_breaker` is a pass-through skeleton. | [slm/AGENTS.md](slm/AGENTS.md) |
| `utils/` | Shared helpers. | — |

## The three platform seams

| Seam | File | Why it exists |
|------|------|---------------|
| Ambient context | `context.py` | tenant / source / RBAC scope without threading a param through every signature |
| SLM Strategy | `slm/_call_slm.py` | swap Ollama ↔ vLLM by config; capture token usage per call |
| Gate-1 RBAC | `veda/rbac_filter.py` | engine-side data filtering at retrieval candidates, the `narrow_allowed` choke point, NoSQL collections, doc chunks — identity no-op when `allowed_resources is None` |

Plus the Redis `sm` load in `veda_hybrid.py` (`VEDA_SM_REDIS`) — assembled semantic
model from `redis-cache`, else the on-disk `SEMANTIC_MODEL_FILE`.

## Gotchas

- **`config.py` defaults, not the docstrings, decide behavior.** ~30% of the
  in-scope modules are flag-OFF or observe-only (see each subdir's AGENTS.md).
- **The engine's top-level `config` module is not the Django `config` package.**
  Ingestion runs with `cwd=veda_core` in a subprocess to keep them apart.
- **`.env` gates every SLM call.** `SLM_MODEL_NAME=qwen2.5:7b-instruct`,
  `SLM_TEMPERATURE=0`; the `config.py` defaults differ, so a machine without the
  `.env` override runs a different model and non-deterministic intent.
- Two `.pyc` interpreter versions (3.13 + 3.14) live in `query/__pycache__` —
  confirm the running container's Python.
