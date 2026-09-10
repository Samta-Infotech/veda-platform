# Phase C0 — Capability-Based Planning Feasibility Audit

**Status: READ-ONLY AUDIT. No code, config, or tests changed.** Phases A, B1, B2 confirmed untouched.
**Legend:** `[CODE VERIFIED]` = read directly this pass. `[INFERENCE]` = reasoning without a fresh
read, flagged. `[PROPOSAL]` = a design option, not implemented. `[UNKNOWN]` = looked, could not
determine, never guessed.

---

## 1. Existing Routing Flow

`[CODE VERIFIED]` — real call chain, traced this pass in `veda_hybrid.py` and `source_coordinator.py`:

```
veda_hybrid.py::run_hybrid_query() (or the underlying orchestration function, L934+)
    ↓
plan_route(query, sids, profile_provider=...)        [veda_hybrid.py:635]
    → query.source_coordinator.plan_route()
        → group_evidence_by_source() [source_evidence.py] — embeddings only, no schema/requirements
        → build_candidates(evidence_by_source, profiles) → List[CandidateSource]
        → routing_policy.decide(candidates, edge_pairs)   — deterministic tiering (STRONG/WEAK/NONE,
          dominance gap, canonical tiebreak) — see §2
        → IF ambiguous: routing_slm.resolve_boundary() — ONE bounded SLM call, validated
    ↓ returns RoutingDecision (status/mode/source_ids/candidate_sources/reason_code)
    ↓
[CRITICAL — confirmed this pass] sm, cols = _load_semantic_model()   [veda_hybrid.py:664]
    This runs AFTER plan_route() returns and AFTER the SINGLE/source_id is already known — for a
    datalake route it's even source-ISOLATED (built only for that one source_id). The semantic
    model/schema is a POST-ROUTING artifact, never available to the routing decision itself.
    ↓
execute_decision(decision, query, sm=sm, cols=cols, tenant=..., profiles=..., on_event=...)
    [source_coordinator.py]
        → plan_execution(decision) → ExecutionPlan (mode/strategy/steps)
        → strategy == "single"      → dispatch() → _resolve_executable() → agent/adapter.execute(...)
        → strategy == "federated"   → _federated_delegate() → federated_route.run_federated()
        → strategy == "independent" → for-loop over each step's agent, execute_reliably each,
                                       merge_results() at the end (see §7 — NOT a DAG, NOT concurrent)
```

**Query classification** (does this query want COUNT/SUM/filter/temporal?) happens **nowhere in
this chain** — confirmed by tracing `aggregate_mode()`/`temporal_parser.py` call sites (`grep`):
both are called only from `veda/pipeline.py` (`pipeline.py:334` for `aggregate_mode`, `pipeline.py:310`
for `run_temporal_parser`), and `pipeline.py::run_query()` is itself only reached **after** dispatch,
inside `agents.py::_sql_delegate()` — i.e., strictly downstream of source selection. `[CODE VERIFIED]`

**Fallback behavior:** `NO_MATCH`/`CLARIFICATION_REQUIRED` → refuse with `decision.reason`
(`veda_hybrid.py:649-651`); a `SINGLE` route whose agent produces no result does **not** fall through
to federated — it constrains scope to the routed source and returns `None` rather than silently
trying a different source (`veda_hybrid.py:684-693`, explicit comment: "Do NOT fall through... it
would answer from a DIFFERENT source").

---

## 2. Source Selection Model

`[CODE VERIFIED]` — classified against the task's A-G taxonomy:

| Mechanism | Used? | Evidence |
|---|---|---|
| A. Source type | **No, not for selection itself** | `source_type` is attached to `CandidateSource` and used only for post-decision *dispatch* (kind→agent), and for a pooling/scoring-fairness bucket in `routing_policy.py::_kind()` — never as a selection criterion |
| B. Hardcoded source IDs | **No** | `RequestContext.source_ids` scopes the *candidate pool* (RBAC/tenant), but which candidate wins is never ID-hardcoded |
| C. Capabilities | **No — confirmed absent from routing entirely** | Zero references to `source_capabilities`/`SourceCapability` anywhere in `routing_policy.py`, `routing_slm.py`, or `source_coordinator.py`'s decision logic (grep confirms) |
| D. LLM/SLM decision | **Yes, bounded** | `routing_slm.py::resolve_boundary()` — only for the *ambiguous* case (near-tied candidates), validated against the actual candidate set, never freeform |
| E. Rules | **Yes, primary mechanism** | `routing_policy.py::decide()` — dominance-gap tiering, canonical-domain tiebreak, edge-augmented boundary widening (`_augment_with_edges`) |
| F. Embeddings | **Yes, primary mechanism** | `source_evidence.py`'s per-source BGE-M3 cosine tiers (STRONG/WEAK/NONE) feed everything downstream |
| G. Metadata | **Yes, secondary** | `is_canonical`/`domain_tags` (Django `Source` registry, via `profile_provider`) for the canonical tiebreak; `cross_source_fk` graph edges for federated-edge detection |

**Net finding:** source selection today is **evidence (embeddings) + structural graph (FK edges) + rules, with a bounded SLM fallback for genuine ties** — capability-awareness is entirely absent from this stage. This is not a partial gap; it's a clean zero.

---

## 3. Capability Model Reuse

`[CODE VERIFIED]` — auditing Phase A's own `source_capabilities.py`/`source_adapters.py`:

| Capability | Storage | Exposed via | Consumed beyond adapters? | Classification |
|---|---|---|---|---|
| `STRUCTURED_QUERY` | `source_capabilities.py::_PROFILES` (static dict, per `source_kind`) | `SourceAdapter.get_capabilities()` | **No** — zero call sites of `get_capabilities()` outside its own tests (confirmed by the same zero-call-sites pattern Phase A/B's own tests assert) | `UNUSED` (for planning) |
| `SCHEMA_DISCOVERY` | Same | Same | No | `UNUSED` |
| `DOCUMENT_RETRIEVAL` | Same | Same | No | `UNUSED` |
| `AGGREGATION` | Same | Same | No | `UNUSED` |
| `FILTERING` | Same | Same | No | `UNUSED` |
| `JOINING` | Same | Same | No | `UNUSED` — also structurally imprecise: relational/datalake are both marked `JOINING=True`, but real join execution is capped at the binary postgres/parquet semi-join (per the earlier Source Adapter audit) — the capability doesn't yet distinguish "can join" from "can join *this specific pair*" |
| `FEDERATION` | Same | Same | No | `UNUSED` |

**Is capability metadata sufficient for planning?** For **coarse, kind-level filtering** (e.g. "don't consider a document source for a query that clearly needs `AGGREGATION`") — **yes, sufficient as-is**, `READY_FOR_PLANNING` at that granularity. For anything finer (per-pair join feasibility, per-column aggregation support, cost/latency) — `MISSING_REQUIRED_METADATA`, and this audit does **not** propose adding that metadata now (no evidenced need yet — would be speculative, per the strict rules).

**Nothing here is `EXECUTION_ONLY`** — the capability model was deliberately built at Phase A as a static, pre-execution-time lookup (`capabilities_for(source_kind)`), not something computed during execution. It's simply unconsumed by anything except the adapter class itself.

---

## 4. Planning Gap Analysis

**Current:** `Query → routing (embeddings+rules+edges) → source selection → execution`
**Desired:** `Query requirements → capability matching → candidate source selection → execution strategy`

**The smallest gap, stated precisely:** routing already produces a `List[CandidateSource]` (in `build_candidates()`, before `decide()` narrows it) — this is *already* "candidate source selection" in embryo. What's missing is (a) a `QueryRequirements`-shaped object to match against, and (b) one filtering step between candidate generation and the deterministic policy. Neither requires touching `decide()`'s dominance/tiering logic itself.

**Ranked options** (blast radius / backward compat / benchmark risk / cross-source future compat / code reuse):

| Option | Blast radius | Backward compat | Benchmark risk | Cross-source future compat | Code reuse |
|---|---|---|---|---|---|
| **B. Adding a capability filtering stage** (new function, called optionally between `build_candidates()` and `decide()`) | **Lowest** — one new function, one new optional call site | Perfect if flag-gated | **Lowest** — off by default, `decide()` unchanged | Good — same filter works for any future source kind via `SourceCapabilities` | High — reuses `CandidateSource.source_type`, Phase A's `capabilities_for()` |
| E. Reusing federated planner infrastructure | Low-medium | Good | Low | Limited — federated planner is 2-source-specific (postgres/parquet), see §7 | Medium |
| A. Extending existing routing output (`RoutingDecision` gains a `requirements`/`eligible_sources` field) | Medium | Good if fields are optional/defaulted | Low-medium | Good | Medium — touches a shared contract with many readers |
| D. Creating a separate planner (parallel to `source_coordinator.py`) | Medium-high | Perfect (fully additive, zero shared-file risk) | Lowest | Good | Low — doesn't reuse `build_candidates()`'s existing pipeline, risks duplicating evidence-gathering |
| C. Replacing existing source selection | **Highest** | **Breaks everything** | **Highest** | N/A | N/A — explicitly ruled out |

**Recommendation for Phase C design: Option B**, with a light touch of Option A only if a later phase needs the eligibility info to survive into `ExecutionPlan` (currently it wouldn't need to — a filter that runs *before* `decide()` naturally shapes `decide()`'s input, no new field required).

---

## 5. Query Requirements

`[CODE VERIFIED]` — audited `QueryIntent`, `intent_envelope.py`, `RoutingDecision`/`CandidateSource`, `AnalyticalSpec`, `ExecutionContext`:

**None of the existing objects carry query-requirement fields today.** `CandidateSource`/`RoutingDecision` have zero fields resembling "needs aggregation" or "needs join." `QueryIntent`/`AnalyticalSpec` are constructed only **after** a source (and its semantic model) is already selected — confirmed by the same call-chain trace as §1 (`pipeline.py::run_query`, itself downstream of dispatch).

**The one genuinely reusable, source-independent signal that already exists:** `veda/planning.py::aggregate_mode(query)` and `veda/query/temporal_parser.py::run_temporal_parser(query)` are both **pure-text classifiers — no schema, no `sm`/`cols` required** (`aggregate_mode`'s own docstring: "Grammar-level (no schema vocabulary)"). They are simply *invoked too late* today — only inside `pipeline.py`, which only runs after routing. There is nothing structurally preventing these same functions from being called a second time (or hoisted) **before** `plan_route()`, since they need nothing routing doesn't already have (the raw query string).

**This audit explicitly does NOT propose a new canonical IR.** The minimal reusable signal is: call `aggregate_mode()`/`run_temporal_parser()` (or equivalents) once, pre-routing, purely to populate a small `QueryRequirements` object — reusing existing, already-battle-tested classifiers rather than inventing new NL-understanding logic.

---

## 6. Capability Matching Design

`[PROPOSAL]` — the smallest contract, built only from what §3/§5 confirm exists:

```python
# PROPOSAL — not implemented
@dataclass(frozen=True)
class QueryRequirements:
    needs_aggregation: bool = False   # from aggregate_mode(query) is not None
    needs_temporal: bool = False      # from run_temporal_parser(query).temporal_filter is not None
    # Deliberately NOT included yet, no evidenced source: needs_join, needs_semantic_search,
    # needs_document_retrieval — these would require inventing detection logic that doesn't exist
    # today (unlike aggregation/temporal, which reuse real, already-shipped classifiers).

def eligible_candidates(candidates: List[CandidateSource],
                         requirements: QueryRequirements) -> List[CandidateSource]:
    """Filter candidates whose SourceCapabilities can't satisfy the requirements. A candidate with
    no capability data (unknown kind) is never dropped — absence of proof is not proof of absence;
    this filter only REMOVES a candidate it can POSITIVELY confirm is incapable."""
    if not (requirements.needs_aggregation or requirements.needs_temporal):
        return candidates   # no requirement -> no filtering, always safe
    out = []
    for c in candidates:
        caps = capabilities_for(c.source_type)
        if requirements.needs_aggregation and not caps.has(SourceCapability.AGGREGATION):
            continue
        out.append(c)
    return out
```

**Does `QueryRequirements` already exist under another name?** No — confirmed by §5. **Do capabilities need normalization first?** No — Phase A's `capabilities_for(source_kind)` already normalizes via `Source.source_kind()`'s single 4-way vocabulary; no new normalization work needed. **Where should matching happen?** Between `build_candidates()` and `decide()`, inside `source_coordinator.py`, as an optional pre-filter — never inside `routing_policy.decide()` itself (keep the deterministic tiering policy untouched, per the non-goals).

**Future source compatibility (7 kinds):** the design above adds nothing kind-specific — a future API/vector source only needs an entry in `source_capabilities.py::_PROFILES` (already how Phase A was built to extend), never a new routing architecture.

---

## 7. Cross-Source Implications

`[CODE VERIFIED]` — re-confirmed and extended this pass:

- **Federated execution is ONE combined SQL statement, not a DAG.** `federated_route.py::run_federated()` builds one DuckDB-executable cross-source SQL (schema text + join hints → single generated query) — there is no multi-step execution sequence to speak of for the `RELATIONSHIP_EDGE` federated path.
- **The `independent` strategy is a sequential for-loop, not concurrent, not a DAG either.** `source_coordinator.py::execute_decision()`'s independent branch iterates `plan.steps` one at a time (confirmed in the earlier Phase A audit — `execute_reliably` per step, failures collected, `merge_results()` at the end).
- **`ExecutionStep.depends_on` is a fully dead field** — re-confirmed (zero producers, not even `plan_execution()`'s own step-builder sets it). **A DAG exists only as an unused data-model shape**, not as executable machinery.
- **Intermediate results** are never persisted between steps — each independent-strategy step's `AgentResult` is collected into a list and merged once at the end; there is no intermediate-result store to speak of.

**Can capability filtering happen BEFORE federated planning?** **Yes, cleanly.** `federated_route.run_federated()` and `plan_execution()` both consume only `source_ids`/`candidate_sources` from the `RoutingDecision` — a capability pre-filter applied to `build_candidates()`'s output (§6) would already have removed incapable candidates before `decide()` ever chooses SINGLE/MULTI, so federated planning downstream sees an already-filtered set with zero changes to `federated_route.py` itself.

The desired flow (`eligible sources → SINGLE/MULTI decision → federated planner → execution DAG`) is **incrementally introducible exactly at the point Option B (§4) inserts** — no reordering of existing stages required, only a new stage between two that already exist.

---

## 8. Boundaries That Must NOT Change

Confirmed unaffected by every option considered in §4/§6: SQL generation (`generation.py`), fast_path (`fast_path.py`), the semantic layer (`compile_semantic_layer.py`/`registry.py`), current benchmark execution, `SourceAdapter.execute()` (Phase A2, untouched), the `ExecutionRequest` contract (Phase B1/B2, untouched), current SINGLE execution behavior (`dispatch()`'s legacy path, untouched when the new filter is a no-op or flag-off). Any Phase C increment must be additive and flag-gated, exactly like A3/B2 were.

---

## 9. Migration Options

| Option | Files affected | Blast radius | Migration complexity | Rollback difficulty | Benchmark risk | Future cross-source support |
|---|---|---|---|---|---|---|
| **1. Shadow-mode only** (compute `QueryRequirements` + log capability-match info, never filter) | New `query_requirements.py`; one new log/trace call in `source_coordinator.py::plan_route()` | **Lowest** | **Lowest** | Trivial — delete the log call | **None** — no behavior path exists | Establishes the pattern without risk, but doesn't yet gate anything |
| **2. Capability-aware filtering stage** (§4 Option B, §6's `eligible_candidates()`, flag-gated) | New `query_requirements.py`; `source_coordinator.py::build_candidates()` gains one optional call | Low | Low-medium | Flip flag off | Low (off by default; on-path only removes candidates it can *positively* prove incapable) | Good — the exact seam §7 confirms federation needs |
| **3. Capability-aware planner module** (separate `capability_planner.py` wrapping `plan_route`, post-filtering `RoutingDecision.candidate_sources`) | New file only; `veda_hybrid.py` would need ONE call-site swap (behind a flag) to call the wrapper instead of `plan_route` directly | Low (isolated) but touches the orchestration entrypoint | Medium (a wrapper duplicates some of `plan_route`'s decision-status handling to stay safe) | Flip flag off; wrapper file can be deleted | Low | Good, but duplicates logic Option 2 gets for free by living inside `source_coordinator.py` |

**Recommendation: Option 2** — it sits exactly at the seam §4/§7 already identified as the smallest gap, requires no new orchestration-entrypoint call-site changes (unlike Option 3), and only becomes shadow-observable and can be built incrementally starting from Option 1's shape.

---

## 10. Proposed Phase C Increments

**Phase C0 (this document):** audit only. Done.

**Phase C1 — Shadow-mode requirement/capability observation:**
- Goal: prove `QueryRequirements` derivation and capability lookup work correctly on real queries, with zero behavior change.
- Files: `veda_core/query/query_requirements.py` (new — `QueryRequirements` dataclass + a `derive_requirements(query)` function calling `aggregate_mode()`/`run_temporal_parser()`); one new trace/log line in `source_coordinator.py::plan_route()` (observe-only, e.g. attached to the existing trace call already at `veda_hybrid.py:634-640`).
- Behavior change: **none** — pure logging.
- Flag: `CAPABILITY_PLANNING_SHADOW_ENABLED`, default OFF (even the logging is gated, so truly zero risk).
- Tests: unit tests for `derive_requirements()` against known aggregate/temporal query strings (reusing existing `aggregate_mode`/`temporal_parser` test fixtures where possible); a test asserting `plan_route()`'s returned `RoutingDecision` is byte-identical whether the shadow flag is on or off.
- Rollback: delete the log call / flip flag off.
- Success metric: shadow logs show plausible requirement/capability data for a sample of real queries, with zero `RoutingDecision` divergence.

**Phase C2 — Capability-aware candidate filtering:**
- Goal: actually remove capability-incapable candidates before `decide()`, for the two requirement types proven in C1.
- Files: `source_coordinator.py::build_candidates()` (one new optional filter call); reuses `query_requirements.py` and Phase A's `source_capabilities.py` — no new capability code.
- Behavior change: a candidate lacking a required capability is dropped from consideration — **only** when the filter can positively confirm incapability (§6's design principle); everything else (dominance tiering, SLM boundary, canonical tiebreak) unchanged.
- Flag: `CAPABILITY_FILTERING_ENABLED`, default OFF.
- Tests: dual-run equivalence (flag off == pre-C2 behavior, same pattern as Phase A3/B2); a case where filtering changes the outcome (document-only candidate excluded from an aggregation query) proven deliberately, not accidentally.
- Rollback: flag off.
- Success metric: on a benchmark sample, filtering never removes a candidate that was previously the *correct* answer (false-negative rate must be zero, not just low) — this is the one metric that must be measured before ever considering default-on.

**Phase C3 — Capability-aware multi-source strategy hinting:**
- Goal: use capability data to help choose federated vs. independent strategy when `decide()` already says MULTI (not to decide SINGLE-vs-MULTI itself — that stays evidence/edge-driven per §2).
- Files: `execution_planner.py::plan_execution()` gains an optional capability check (e.g. don't attempt `federated` strategy if a selected source lacks `FEDERATION` capability — fall back to `independent`).
- Behavior change: only affects the federated/independent strategy choice, only when C2's flag is also on.
- Flag: `CAPABILITY_STRATEGY_HINTING_ENABLED`, default OFF, depends on C2 being available (not necessarily on).
- Tests: mirrors A3/B2's dual-run pattern applied to `plan_execution()`'s strategy output.
- Rollback: flag off.
- Success metric: strategy choice never changes for a source pair that already federates successfully today (zero regression on the existing federated benchmark cases).

**Phase C4 — Capability-aware federation source selection (deferred, not designed here):**
- Goal: extend capability filtering into which sources a federated query includes, beyond the binary postgres/parquet case.
- Explicitly **not scoped** in this document — depends on the separate, already-flagged fragility (5 duplicated postgres/parquet call sites, per the earlier Source Adapter audit) being addressed first; proposing C4's design now would be speculative ahead of that prerequisite work.

---

## 11. Verdict

**A. Is Phase C needed?** Yes, for the stated long-term goal (capability-aware planning) — but only the narrow, additive slice in §10; nothing here justifies a broad rearchitecture.

**B. Is existing architecture already partially capability-aware?** **No** — confirmed in §2/§3: routing uses zero capability data today; the capability model (Phase A) is fully built but entirely unconsumed outside its own adapter class and tests.

**C. What is the smallest safe next implementation?** Phase C1 (shadow-mode observation) — even smaller than C2, and the natural first increment given C2 depends on proving C1's derivation is trustworthy first.

**D. Should we modify routing or add a capability filtering layer?** Add a filtering layer (§4 Option B / §9 Option 2) — `routing_policy.decide()`'s deterministic tiering logic itself should not change; the filter narrows its *input*, never its internal logic.

**E. Can existing federated infrastructure be reused?** Yes, without modification — §7 confirms `federated_route.py`/`plan_execution()` already consume only `candidate_sources`/`source_ids`, so a pre-filter upstream requires zero changes to either.

**F. What should be implemented immediately after this audit?** Phase C1 only, exactly as scoped in §10 — shadow-mode, zero behavior change, gated behind its own new flag, fully separate from `SOURCE_ADAPTER_DISPATCH_ENABLED`/`EXECUTION_REQUEST_DISPATCH_ENABLED`.

---

## Compact Summary

1. **Current routing architecture:** embeddings (BGE-M3 cosine tiers) + FK-graph edges + deterministic dominance/canonical rules, with a bounded SLM only for genuine ties. The semantic model/schema is loaded strictly *after* a source is chosen — routing never sees query-requirement signals today.
2. **Existing capability model reuse:** Phase A's `SourceCapabilities`/`SourceAdapter.get_capabilities()` is fully built but has zero consumers outside its own class and tests — sufficient for coarse kind-level filtering, not yet for fine-grained matching.
3. **Smallest architectural gap:** one new filtering stage between `build_candidates()` and `decide()` inside `source_coordinator.py` — `build_candidates()` already produces the exact candidate list a filter would narrow; nothing needs reordering.
4. **Recommended Phase C design:** a small `QueryRequirements` object built by reusing two already-shipped, schema-independent text classifiers (`aggregate_mode()`, `run_temporal_parser()`) — no new canonical IR, no new NL-understanding logic.
5. **Exact Phase C increments:** C1 shadow-mode observation (zero behavior change) → C2 flag-gated candidate filtering (proven zero false-negatives before any default-on consideration) → C3 capability-aware federated/independent strategy hinting → C4 deferred (federation source-kind generalization, blocked on separate prerequisite work).
6. **Biggest architecture risk:** none of the proposed increments carry meaningful risk individually (all additive, flag-gated, dual-run-tested per the Phase A/B precedent) — the real risk is skipping C1's shadow-mode validation and jumping straight to C2's behavior-changing filter without first proving the requirement-derivation is trustworthy on real query variety.
7. **Final GO/NO-GO verdict: 🟢 GO — for Phase C1 only.** C2 onward should each be separately re-verified against C1's real shadow-mode data before being greenlit, not pre-approved as a bundle.

---

## Phase C1 — Implementation Record (2026-09-04)

**Status: IMPLEMENTED, tested, verified in both flag states. C2/C3/C4 NOT started, per the hard stop.**

**What was implemented:**
- `veda_core/query/query_requirements.py` — `QueryRequirements` (frozen dataclass: `requires_aggregation`, `aggregate_type`, `requires_temporal`, `temporal_requirement`) + `derive_requirements(query)`.
- `veda_core/query/capability_observation.py` — `CandidateCapabilityObservation` + `observe_candidate_capabilities()` (pure comparison) + `run_capability_planning_shadow()` (flag-gated, logs, swallows all exceptions, never mutates).
- `veda_core/config.py` — new flag `CAPABILITY_PLANNING_SHADOW_ENABLED`, default OFF, separate from the A3/B2 flags.
- `veda_core/query/source_coordinator.py::plan_route()` — one call to `run_capability_planning_shadow(query, candidates)` inserted immediately after `candidates = build_candidates(...)`, before `decide()`. `candidates` itself is never reassigned.

**Reuse-first audit outcome (confirmed vs. assumed in the original C0 document):**
- `aggregate_mode()` was **not** used — on inspection its actual contract is a narrower, different concept (per-anchor child-grouped counting with a threshold), not a general aggregation-intent signal. This corrects an assumption carried from the C0 audit.
- The correct general-purpose signal, confirmed by reading `fast_path.py::_count_intent()`, is `fast_path.py`'s own `_COUNT_TRIGGERS`/`_COUNT_WORDS`/`_SUM_VERBS`/`_AVG_VERBS` tuples — imported directly (not re-derived) so there remains exactly one source of truth for these trigger phrases.
- `run_temporal_parser()` was used exactly as the C0 audit anticipated — no surprises.
- `requires_join`/multi-entity was investigated and **not implemented** — no schema-independent deterministic signal exists anywhere in the codebase (every join-detection mechanism requires a loaded schema or relationship graph). Adding a keyword heuristic would have violated the explicit "no weak heuristics" rule.

**Capability-model gap discovered:** the Phase A `SourceCapability` enum has no temporal capability (only `STRUCTURED_QUERY`/`SCHEMA_DISCOVERY`/`DOCUMENT_RETRIEVAL`/`AGGREGATION`/`FILTERING`/`JOINING`/`FEDERATION`). `requires_temporal` is derived and logged, but **never compared against any capability** — there is nothing to compare it against without inventing a mapping. This is exactly the kind of gap C1 was designed to surface for C2 to address later, not something to patch now.

**Proof C1 does not modify candidates:**
- `test_shadow_run_does_not_reorder_or_filter_or_mutate_candidates` — asserts list identity/membership/order unchanged after the shadow call.
- `test_plan_route_flag_off_decide_receives_identical_candidates` / `test_plan_route_flag_on_decide_still_receives_identical_candidates` — call-spy proving `decide()` receives the **exact same object** `build_candidates()` produced, in both flag states (identity check, `is`, not just equality).
- `test_plan_route_output_identical_shadow_flag_on_vs_off` — full `plan_route()` dual-run equality.
- `test_plan_route_shadow_on_with_incompatible_candidate_still_included` — a document-source candidate the shadow observation would flag incompatible for an aggregation query is still present in the final `RoutingDecision.candidate_sources`.

**Flag OFF verification:** env unset → 116/116 passed (full Phase A+B+C suite).

**Flag ON verification:** `CAPABILITY_PLANNING_SHADOW_ENABLED=1` via real env var → 115/116 passed; the 1 failure (`test_capability_shadow_flag_default_off`) is the default-value assertion correctly firing under an intentional environment override — same classification as Phase A3/B2's verification, not a regression. A manual run with logging enabled confirmed the shadow log fires with exactly the expected shape: `requirements=QueryRequirements(requires_aggregation=True, aggregate_type='count', ...)`, `observations=[{'source_id': '5', 'compatible': True, ...}, {'source_id': '9', 'compatible': False, 'incompatibilities': ['requires_aggregation but source lacks AGGREGATION capability']}]`, and candidates confirmed unmutated after the call.

**Full regression:** 116 tests total across `query_requirements`/`capability_observation`/`execution_request`/`source_capabilities`/`source_adapters`/`routing_policy`/`source_coordinator`/`source_agents`/`execution_and_merge`/`reliability` — green at default flags.

**Explicit confirmation: C1 is shadow-only. No routing behavior has changed.**

**What C2 would need before filtering could be considered (evidence gate, not yet gathered):** real shadow-mode log volume across a representative query sample, specifically to answer the five questions the C1 task posed — which candidates are capability-incompatible in practice, whether the *actually selected* source ever fails its own inferred requirements (a signal of a deeper bug, not a filtering opportunity), what the false-negative rate of a hypothetical C2 filter would have been, and which requirement type (aggregation vs. temporal) produces reliable vs. noisy signal. None of this data exists yet — C1 only just started being capable of producing it.
