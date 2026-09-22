# VEDA Source Capability & Execution Adapter Foundation — Audit

**Status: READ-ONLY AUDIT. No code, config, flags, tests, or benchmark state were touched.**
**Legend:** `[CODE VERIFIED]` = read directly. `[INFERENCE]` = reasoning without a direct read, flagged. `[UNKNOWN]` = not confirmed this pass — named explicitly rather than guessed.

**Note on method:** two of three planned verification passes initially hit the session's API rate limit mid-run; that limit cleared and both were successfully re-run to completion (results folded in below). Everything in this doc is `[CODE VERIFIED]` from this session's audits — nothing is guessed.

---

## 1. Current Architecture Map

**Relational, single-source** `[CODE VERIFIED]`:
```
apps/query/views.py → inference_client.py (HTTP) → inference/main.py → veda_core engine
→ veda/pipeline.py (temporal_parser → fast_path.py keyword/registry match
   → intent.py::build_sql if fast-path claims it, else generation.py's SQL-writing SLM call)
→ SQL executed directly against Postgres via psycopg2 cursor calls inside fast_path.py
   (cur.execute at fast_path.py:366,386,405) — NOT through connectors/base.py
→ intent_sql_alignment.py guards → result_analyzer.py/result_explainer.py synthesis
```

**Datalake, single-source** `[CODE VERIFIED]`:
```
Same entry through pipeline.py, but veda_hybrid.py:335-509 builds a datalake-only
semantic model from column metadata (not the compiled relational registry)
→ execution via cross_source_composer.py's DuckDB-catalog materialization
   (parquet files → DuckDB views) — again, no connectors/base.py involvement
```

**Federated (multi-source, including relational)** `[CODE VERIFIED, new finding]`:
```
source_coordinator.py::execute_decision() → plan_execution() (execution_planner.py)
→ strategy == FEDERATED → _federated_delegate() → federated_route.py
→ federated_executor.py::_connect() → duckdb.connect() directly
   → for a postgres-kind surface: "INSTALL postgres; LOAD postgres;
      ATTACH '<dsn>' AS <catalog> (TYPE POSTGRES, READ_ONLY);"
   → for a datalake-kind surface: CREATE VIEW over parquet paths
```
**Important correction to the task's framing:** DuckDB is not just "the datalake engine" — it is the **actual execution engine for every federated query today, including ones involving Postgres**, via DuckDB's native `ATTACH ... TYPE POSTGRES` mechanism. A future adapter model should treat "federated execution" as a DuckDB-centric concern regardless of which source kinds are involved, not as "SQL adapter + DuckDB adapter running side by side."

**The one live query-time use of `connectors/base.py`** `[CODE VERIFIED]`: `veda_hybrid.py::_run_nosql()` calls `build_connector(src)` twice per query (schema inference, then execution) — this is the only source kind where the connector abstraction is actually exercised outside ingestion.

**Resolved: fast_path.py's single-source relational connection is per-source, not hardcoded** `[CODE VERIFIED]`. `fast_path.py`'s `cur.execute()` calls (lines 366/386/405) live inside `_live_ground_dim_value()` (a flag-gated, default-OFF live-DB probe for filter-value grounding — not the primary SQL execution path, which is `execute_sql`), opened via `conn = _pg()` (`veda/runtime.py:84-105`, imported at `fast_path.py:359`). `_pg()` calls `get_db_config()`, which — when a request context is active — resolves via `storage_adapters/reader.py::source_connection()`, reading the **current request's actual `Source` row** (`host`/`port`/`dbname`/`db_user`/`password_env`/`password_inline`) and setting `search_path` to that source's schema. This is genuinely per-source, matching `Source` model fields exactly — not a single shared/internal DB. (A separate, unrelated cached singleton connection in the same file, `storage_adapters/reader.py::_connection()`, points at VEDA's own internal metadata Postgres for retrieval/glossary lookups — distinct from source execution, not to be conflated.)

**Comparison to `RelationalConnector` (ingestion-tier)** `[CODE VERIFIED]`: same config fields, but `RelationalConnector._get_raw_connection()` additionally sets `connect_timeout=10` and TCP keepalives, and the connector class has retry-with-backoff (3 attempts, doubling delay) and explicit `connect()`/`disconnect()`/`_ensure_connected()` lifecycle — none of which `_pg()` has (a bare one-shot `psycopg2.connect()`, no retry, no timeout). **Neither path pools connections** — both open a fresh connection per call. A future adapter swap to `RelationalConnector` would add resilience but change failure semantics (retry/backoff delay vs. immediate exception) and would need adapting to a stateful-object lifecycle rather than a plain connection factory — a real, if small, design decision for a later phase, not a blocker for Phase A1/A2.

---

## 2. Existing Source Abstractions

| Abstraction | File | Tier | Notes |
|---|---|---|---|
| `BaseConnector` (ABC) + `SourceType` enum + `supports_*` props | `connectors/base.py` | **Mixed** | Ingestion-only for relational/datalake/document; query-time-live only for nosql |
| Concrete connectors | `connectors/{relational,datalake,nosql,document,tabular_files}.py` | Same split | — |
| `Source.source_kind()` | `apps/sources/models.py` | Both | Derived from `dialect`; the one true single source of kind-identity |
| `SourceItem` | `apps/sources/models.py` + `item_profiler.py` | Ingestion-build, query-time-read | Already the source-agnostic metadata catalog |
| `CandidateSource.source_type` | `routing_contracts.py` | Query-time | Bare string — natural attach point for a capability object |
| `source_coordinator.py` | `veda_core/query/` | Query-time | `plan_route`/`build_candidates`/`dispatch`/`execute_decision` |
| `ExecutionStep`/`ExecutionPlan` | `execution_planner.py` | Query-time | See §exact fields below |
| `agents.py::_AGENT_BY_KIND` | `veda_core/query/agents.py` | Query-time | **The closest existing thing to an adapter dispatcher** — relational→`DatabaseAgent`, datalake→`DataLakeAgent`, document→`FileSystemAgent`, nosql→`NoSqlAgent`, each wrapping an existing call (`run_query`/`run_rag_layer`/`_run_nosql`) |
| `federated_route.py`/`semi_join_planner.py`/`federated_executor.py` kind switch | Same files | Query-time | Hardcoded binary `postgres`/`parquet`, 5 duplicated call sites — a real, pre-existing fragility |
| `source_evidence.py` | `veda_core/query/` | Query-time | Two evidence kinds only (`column`/`chunk`) |

**Exact `ExecutionStep`/`ExecutionPlan` fields** `[CODE VERIFIED, fresh read]`:
```python
class ExecutionStep:
    source_id: str
    source_type: str = ""
    depends_on: List[str] = field(default_factory=list)
    required: bool = True

class ExecutionPlan:
    mode: str        # PARALLEL | DEPENDENT
    strategy: str    # single | federated | independent
    steps: List[ExecutionStep]
    reason: str = ""
```
**`depends_on` is a fully dead field** `[CODE VERIFIED via grep]` — it appears nowhere else in the codebase except its own definition; `plan_execution()`'s own step-builder (`_step()`) never sets it. There is zero live producer of a non-empty `depends_on` anywhere. This confirms and sharpens the earlier session's finding that `DEPENDENT` mode is never emitted — not only is the mode unused, the very field that would carry dependency information has no code path that populates it.

**Partial-failure handling — already real and solid** `[CODE VERIFIED, fresh read]`: `execute_decision()`'s `independent` strategy branch iterates `plan.steps`, wraps each agent call in `execute_reliably`, classifies failures (`classify_failure`), and returns `{"partial": {"failures": [...], "any_required_failed": bool, "ok_count": int, "complete": bool}}` — non-required sources can fail without failing the whole query, and this is never silently swallowed. This is a genuinely good existing mechanism a future adapter model should preserve, not replace.

---

## 3. Duplication Analysis

**Reusable as-is:**
- `Source.source_kind()` — must remain the single source of truth for kind; any new capability model derives from it, never re-derives independently.
- `SourceItem` — already the source-agnostic metadata catalog; a capability model is a sibling, not a replacement.
- `agents.py::_AGENT_BY_KIND` — **this is the adapter pattern in embryo.** A `SourceAdapter` interface should be a typed generalization of this exact dispatch table, not a parallel mechanism built from scratch.

**Correction from a follow-up sweep — there are three independent kind-keyed dispatch layers, not one** `[CODE VERIFIED]`. Beyond `agents.py::_AGENT_BY_KIND` (query-time, class-based), two more exist, both at ingestion time:
1. `connectors/base.py::_CONNECTOR_REGISTRY` (`register_connector`/`build_connector`/`get_connector_class`) — keyed `"{source_type}:{engine}"`, returns connector classes.
2. `veda_core/ingestion/source_dispatcher.py::dispatch_ingestion` — a **separate** dict (`{"relational": _dispatch_relational, "datalake": _dispatch_datalake, "document": _dispatch_document, "nosql": _dispatch_nosql}`) keyed on the same `source_type` string, returning pipeline **functions** (not classes), each of which internally calls `build_connector` *again* and runs a distinct multi-step ingestion pipeline. `_dispatch_datalake` even has a *nested third* dispatch inside it, on `engine`, choosing between `TabularFileConnector` directly vs. `build_connector`.

All three maps key on the same `source_type`/`source_kind` string family but are never unified into one adapter object. `apps/sources/serializers.py`'s uppercase vocabulary was confirmed to be pure labeling with no behavioral dispatch — not a fourth real dispatcher, just a fourth *string vocabulary* (per the risk already named below). **This sharpens the consolidation risk**: a future `SourceAdapter` isn't replacing one existing dispatcher, it's a candidate to eventually unify three, and Phase A should not attempt that unification — only build the new capability/adapter layer additively, leaving all three existing dispatchers exactly as they are.

**Overlaps requiring resolution, not duplication:**
`connectors/base.py`'s `supports_*` booleans map directly onto part of the task's requested taxonomy: `schema_discovery`≈`supports_schema`, `structured_query`/`sql`≈`supports_query`, `document_retrieval`≈`supports_chunks`. But `aggregation`/`filtering`/`joining`/`federation`/`temporary_relation_support` have **no existing connector flag** — they are query-execution capabilities, not ingestion-connector capabilities, and belong in a new query-tier model, not bolted onto `BaseConnector`.

**Should NOT be added — no evidence of support in this codebase:**
- `semantic_search`/`vector_search` as distinct from `document_retrieval` — no separate vector-only code path found; RAG retrieval is the only retrieval mechanism. Don't split into two capabilities without evidence.
- `entity_extraction` — exists only as `doc_data_planner.py`'s narrow 2-hop primitive, not a general source capability.
- `api_request` — **zero evidence**: no API-source connector exists anywhere in `connectors/`. Pure speculative addition; do not add.
- `file_metadata_search` — fully overlaps `SourceItem.item_metadata` for documents; don't add a redundant flag for something `SourceItem` already provides.

**Harmful-duplicate risk, explicit:** there are already **two** independent source-kind string vocabularies in the query tier (`source_kind()`'s 4-way lowercase string, and federation's binary `postgres`/`parquet`), plus a third at the API-serialization tier (`serializers.py`'s uppercase `"DATABASE"`/`"DATALAKE"`). Building a capability model on top of `CandidateSource.source_type` without consolidating onto `Source.source_kind()` would create a **fourth** parallel vocabulary. This is the single biggest risk to get right in Phase A1.

---

## 4. Proposed Minimal Source Capability Model

```python
# PROPOSAL — not implemented
class SourceCapability(str, Enum):
    STRUCTURED_QUERY = "structured_query"   # backs supports_query
    SCHEMA_DISCOVERY = "schema_discovery"   # backs supports_schema
    DOCUMENT_RETRIEVAL = "document_retrieval"  # backs supports_chunks
    AGGREGATION = "aggregation"       # NEW — no existing flag
    FILTERING = "filtering"           # NEW
    JOINING = "joining"               # NEW — note: currently DuckDB-mediated even for postgres (§1)
    FEDERATION = "federation"         # NEW — "can this source participate in a DuckDB-federated query"

@dataclass(frozen=True)
class SourceCapabilities:
    source_kind: str   # from Source.source_kind() — never re-derived
    capabilities: frozenset[SourceCapability]
```

Kept deliberately smaller than the task's example list — every capability above either has a direct existing flag to derive from, or is directly evidenced by a real code path found this session (aggregation/filtering exist in fast_path's keyword logic; joining/federation are the DuckDB-ATTACH mechanism from §1). `vector_search`, `entity_extraction`, `api_request`, `file_metadata_search` are explicitly excluded per §3 — adding them now would be speculative architecture, not evidence-based.

---

## 5. Proposed Source Adapter Interface

```python
# PROPOSAL — not implemented
class SourceAdapter(Protocol):
    def get_capabilities(self) -> SourceCapabilities: ...
    def execute(self, query: str, *, source_id: str, source_ids: list[str],
                sm=None, cols=None, evidence=None, execution_context=None,
                on_event=None) -> "AgentResult": ...
```

Deliberately **not** the task's full example (`discover_metadata`/`ground_query`/`plan`/`execute`) — evidence supports only `execute()` as safe to introduce now:

| Method | Input | Output | Existing implementation reused | New code required |
|---|---|---|---|---|
| `get_capabilities()` | none | `SourceCapabilities` | `Source.source_kind()` + a static per-kind capability table (§4) | Small — a lookup function |
| `execute()` | same signature `agents.py`'s agent classes already accept | same `AgentResult` shape agents already return | **Entirely** — this should be a **direct pass-through** to the existing `_AGENT_BY_KIND`-dispatched agent's `.execute()` (`run_query`/`run_rag_layer`/`_run_nosql`) | Only the thin wrapper class itself |

`discover_metadata()`/`ground_query()`/`plan()` are **not proposed for Phase A** — no evidence in this audit that current agents expose these as separable steps; `run_query()` etc. are monolithic calls, not decomposed into ground→plan→execute stages internally. Proposing those methods now would be inventing structure the current implementation doesn't have, exactly what the task warns against.

---

## 6. Migration Strategy

**Phase A1 — Capability representation only.** Add `SourceCapabilities`/`SourceCapability` (§4) as pure dataclasses/enum in a new file (e.g. `veda_core/query/source_capabilities.py`), with a function `capabilities_for(source_kind: str) -> SourceCapabilities` keyed off the existing 4-way `source_kind()` string. Zero call sites anywhere else. Fully safe — cannot affect any running behavior because nothing calls it yet.

**Phase A2 — Adapter interface, wrapping `agents.py`'s existing dispatch.** Introduce `SourceAdapter` (§5) as a thin class per kind, each `__init__` taking the existing agent instance from `_AGENT_BY_KIND`, and `.execute()` delegating verbatim to `agent.execute(...)`. No new call sites in `source_coordinator.py`/`dispatch()`/`execute_decision()` yet — build and unit-test the adapter classes in isolation first.

**Phase A3 — Wrap existing relational execution.** Only after A2 is proven: change `resolve_agent()`'s callers (or add a parallel `resolve_adapter()`) to optionally return the new adapter wrapping the same agent, behind a flag. The underlying `DatabaseAgent`/`run_query()`/fast_path/generation/psycopg2 call chain is **untouched** — the adapter is a call-through layer, not a reimplementation.

**Phase A4 — Compatibility test with existing routing.** Dual-run: for a sample of queries, call both the existing direct-agent path and the new adapter-wrapped path, assert identical `AgentResult`. Do not cut over until this passes on a real query sample, not synthetic ones.

Each phase is independently revertable — Phase A1 by deleting an unused file; A2 by the same; A3 by flipping the flag off; A4 is measurement, not a behavior change at all.

---

## 7. File-Level Change Plan

| File | Change | Blast radius | Tests affected |
|---|---|---|---|
| `veda_core/query/source_capabilities.py` (new) | New dataclasses/enum, Phase A1 | **None** — new file, zero imports elsewhere | None (new file needs new unit tests only) |
| `veda_core/query/source_adapters.py` (new) | New thin adapter classes, Phase A2 | **None** initially — not imported by any live call site until A3 | None until A3 |
| `veda_core/query/agents.py` | Phase A3: optionally expose `resolve_adapter()` alongside existing `resolve_agent()` | **Low** — additive function, existing `resolve_agent()` unchanged | `tests/test_source_agents.py` (exists per repo listing) — run, don't modify |
| `veda_core/query/source_coordinator.py` | Phase A3/A4: flag-gated call to adapter instead of agent, in `dispatch()` only | **Medium** — this is the one file where a real behavior-path decision gets made; must be flag-gated and defaulted off | `tests/test_source_coordinator.py` |

**Explicitly not changed by this phase, per constraints:** `fast_path.py`, `generation.py`, `intent.py`, `intent_sql_alignment.py`, `federated_route.py`, `semi_join_planner.py`, `federated_executor.py`, `retrieval_v2.py`, any semantic-registry file, any benchmark/eval file.

**Open item from the initial pass — now resolved:** `fast_path.py`'s connection-acquisition path was confirmed per-source (`_pg()` → `storage_adapters/reader.py::source_connection()` → reads the current request's actual `Source` row), not a shared internal DB. No remaining open items block any phase of this plan.

---

## 8. Compatibility Matrix

| Existing System | Compatible? | Change Required? | Risk |
|---|---|---|---|
| Fast Path | Yes | None in A1-A2; none in A3 either (call-through only) | **LOW** |
| Current SQL Generation | Yes | None | **LOW** |
| Semantic Registry | Yes | None | **LOW** |
| Postgres (single-source) | Yes, with the §7 open item resolved first | None to execution itself; A3 adds an optional call-through layer | **LOW**, pending the one open verification |
| DuckDB / Datalake | Yes | None — `federated_executor.py`'s DuckDB-ATTACH mechanism is untouched; an adapter would call through to it, not replace it | **LOW** |
| Federated Query | Yes, for A1-A2; A3/A4 should NOT attempt the federated strategy yet | Explicitly deferred — `_federated_delegate()` stays untouched this phase | **LOW** (because deferred) |
| Alignment Guards | Yes | None | **LOW** |
| Source Routing | Yes | Additive optional path only in A3/A4 | **LOW-MEDIUM** |
| Current Benchmarks | Yes | None — no default-on behavior change in any phase | **LOW** |

---

## 9. Go / No-Go Verdict

**Verdict: B — existing architecture requires a small prerequisite step, not a structural refactor.**

The prerequisite is narrow and specific: **before or during Phase A1, explicitly derive the capability model's `source_kind` from `Source.source_kind()` and nowhere else** — because at least three independent kind-keyed dispatch layers and at least three string vocabularies for the same "kind" concept already coexist (§3), and building on `CandidateSource.source_type` in isolation risks adding a fourth of either. This is not a structural blocker (verdict C would require something architecturally incompatible, and nothing found here is) — it is a discipline requirement that costs nothing to satisfy if named explicitly up front, which is what Phase A1 above does.

Everything else evidences A: `agents.py::_AGENT_BY_KIND` already is a proto-adapter dispatcher; `ExecutionStep`/`ExecutionPlan` already carry `source_id`/`source_type`/`required` with real (if not dependency-aware) partial-failure handling; the connector-tier `supports_*` idiom already covers half the needed capability vocabulary. Nothing needs to be torn out or rewritten to introduce a thin, call-through adapter layer.

---

## Compact Summary

1. **What existing architecture can be reused:** `Source.source_kind()` (kind identity), `agents.py::_AGENT_BY_KIND` (adapter dispatch pattern, already exists in embryo), `ExecutionStep`/`ExecutionPlan` (already carries source_id/source_type/required with real partial-failure handling), `connectors/base.py`'s `supports_*` idiom (half the capability vocabulary).
2. **Minimal new abstraction needed:** a small `SourceCapability` enum (6 values, not the task's full 13) + `SourceCapabilities` dataclass, plus a thin `SourceAdapter` wrapping the existing agent dispatch — no new execution logic, no new planning logic.
3. **Is `SourceAdapter` safe?** Yes, as a call-through wrapper over `agents.py`'s existing dispatch (Phase A2) — it adds nothing new to execute, only a typed name for what already runs.
4. **Can relational execution be wrapped without behavior change?** Yes — confirmed. `fast_path.py`'s connection is genuinely per-source (reads the request's actual `Source` row), not hardcoded to one internal DB, and matches `RelationalConnector`'s config fields closely enough that a wrap adds no behavior change (it would only gain resilience — timeout/retry/keepalives — as a later, separate decision).
5. **Does DuckDB fit?** Yes, and better than the task's framing assumed — DuckDB is already the execution engine for *all* federated queries today (including ones involving Postgres, via `ATTACH ... TYPE POSTGRES`), not just datalake. An adapter here is a call-through to `federated_executor.py`, unchanged.
6. **Can federated queries fit later?** Yes, but not in this phase — `ExecutionStep.depends_on` is a fully dead field (confirmed, zero live producers), so DEPENDENT-mode/dependency-aware execution needs its own dedicated future phase, not bolted onto Phase A.
7. **Recommended next implementation increment:** Phase A1 only — the pure `SourceCapabilities`/`SourceCapability` dataclasses, zero call sites, explicitly deriving `source_kind` from `Source.source_kind()`. Stop there, measure, and only propose A2 once A1 is reviewed.
8. **GO / NO-GO: GO**, scoped strictly to Phase A1 as the next concrete step, with the explicit source-kind-vocabulary-consolidation discipline as the one prerequisite condition. All open verification items from the initial pass are now resolved — no blockers remain for A1.
