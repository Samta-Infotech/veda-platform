# SLM flow and the summary layer — how the small language model moves through the query pipeline

Analysis written 2026-09-20 from a direct read of the source on `feat/refinements-pipeline`
(tree clean at `5180243`), the running containers on this machine, and the 2,158 persisted
explain traces in `veda_core/logs/explain_trace.jsonl`. Line numbers are anchors, not
addresses. Path prefix for engine citations: `veda_core/`.

Companion docs: [QUERY_ENGINE.md](QUERY_ENGINE.md) (the deterministic SQL head),
[ARCHITECTURE.md](ARCHITECTURE.md) §10 (the SLM seam), [CHAT.md](CHAT.md) (the chat
tier). This doc covers what those treat only in passing: **every place the SLM is
consulted during one query, what each call is given, what is done with its output, and
how the final natural-language summary is produced and guarded.**

Contents: §0 summary · §1 the seam · §2 call map · §3 SQL/IR generation sites ·
§4 routing, RAG and federated sites · §5 the summary layer in depth · §6 chat tier and
request path · §7 evidence from traces · §8 findings · §9 config reference.

---

## 0. Executive summary

- **The SLM never decides anything alone.** On the production path it is consulted for
  (a) SQL fill-in inside a deterministically pinned skeleton, (b) the Tier-2 envelope +
  LangGraph IR when the deterministic head refuses, (c) RAG/hybrid prose synthesis,
  (d) the post-execution **summary** of already-executed rows, and (e) bounded
  routing/classification decisions at ambiguous boundaries. Outside `veda/generation.py`
  the model never writes SQL; it emits UUID-only IR, opaque-handle envelopes, closed-enum
  labels, or concepts, and every output is re-validated against the real schema before it
  can execute.
- **The summary layer (L7b) is "facts-in, prose-out."** The model never sees raw rows.
  A deterministic extractor builds a constant-size facts payload (≤ 5 sample rows, true
  row count, per-column aggregates) plus verified findings from `result_analyzer`, and the
  SLM only narrates them. A numeric guard rejects any prose that states a number not
  traceable to that payload; a currency guard strips symbols the data does not carry. The
  chat tier renders the engine's `answer` verbatim and never re-summarises.
- **On this machine the summary model does not exist.** `NL_SUMMARY_MODEL` defaults to
  `qwen2.5:7b-instruct`; the host Ollama serves only `qwen2.5-coder:7b`. The 2026-09-16
  `_nl_model()` fix makes the explainer fall back to the coder model, so summaries work,
  but every one of the 404 post-fix summary calls in the trace log ran on the **coder**
  model, while the trace's `summary.model` field still claims the instruct model, and
  `/readyz` reports the instruct model warm.
- **Temperature is 0.3 here, not 0.** This clone's `.env` sets `SLM_MODEL_NAME` but not
  `SLM_TEMPERATURE`; `config.py` defaults to 0.3. The CLAUDE.md warning about
  non-deterministic intent applies to this environment.
- **A Tier-2 fallback costs 5 SLM calls, not 1.** `USE_LANGGRAPH` defaults to true, so
  the IR is built by four `lg_node` calls after one `envelope` call; the 16-rule
  UUID-contract prompt in `slm_layer.py` is never sent.
- **Two observability channels are silently dead.** The ContextVar usage accumulator is
  never folded (`_fold_usage` has no caller), so no persisted trace has an `llm_usage`
  section and MLflow's token metrics read zero; the `nl_summary` section is not in the
  serialised section list and never reaches disk. Token totals survive only through the
  thread-local `collect_usage` path.
- **Latency:** post-fix, the summary call has a median of ~5.9 s and a p90 of ~11.8 s
  on this CPU/Metal setup and is ~11 % of median end-to-end time. The heavy cost is the
  routing/understanding/generation calls at 5–15 s each, several per query on
  multi-source scopes. Every SLM call, summary included, is synchronous on the request
  path.
- Fifteen concrete defects/inconsistencies are listed in §8 with file anchors.

---

## 1. The SLM seam — one entry point, two backends

`slm/_call_slm.py::call_slm(user_message, *, system, purpose, timeout, temperature=0.0,
num_predict, num_ctx, seed, json_format, endpoint="chat", model)` (`_call_slm.py:376`)
is the single choke point. `purpose` is a label only; it never changes routing.

| Concern | Behaviour | Anchor |
|---|---|---|
| Backend | `SLM_BACKEND` → `OllamaBackend` (`/api/chat`, or `/api/generate` for `endpoint="generate"`, `keep_alive:"24h"`, `format:"json"` when asked) or `VLLMBackend` (`/v1/chat/completions`, `num_predict→max_tokens`, **`num_ctx` silently dropped**). Cached per process. | `:231`, `:297`, `:354` |
| Model | `model=` argument else `SLM_MODEL_NAME` (`config.py:371`, env-overridable; default `qwen2.5-coder:7b`). Only the summary-family calls ever pass `model=`. | `:387` |
| Timeout | per call, else `SLM_TIMEOUT_SECS = 240`. | `:212` |
| Errors | Network/API errors become a uniform `RuntimeError("SLM unreachable/invalid …")`. **Empty content also raises.** Call sites own the degrade. | `:223`, `:288` |
| Circuit breaker | `_slm_circuit_breaker` is a pass-through no-op. | `:124` |
| Per-call ledger | Every call, success **and failure**, is timed and appended to the ambient `ExplainTrace` `slm.calls[]` with `purpose/model/duration_ms/ok/error`. | `:394-413`, `explain.py:129` |
| Token accounting | Two parallel mechanisms. (1) Thread-local `collect_usage()` buffer, summed in `pipeline._done` — **works**. (2) ContextVar accumulator (`reset_usage/_note_usage/_fold_usage/get_usage`) folded into the trace's `llm_usage` at `finish()` — **dead**: `_fold_usage` is never called, and the `/api/generate` branch never calls `_note_usage`. Only successful calls record tokens in either. | `:65-187`, `explain.py:220` |
| `served_models()` | Ollama-only `/api/tags` probe, cached per process; lets a caller pick a secondary model only if it exists. | `:243` |
| `prewarm()` | 1-token call; **never raises**. | `:421` |

**Live environment (this machine, 2026-09-20).** Inference container env:
`SLM_BACKEND=ollama`, `SLM_MODEL_NAME=qwen2.5-coder:7b`,
`OLLAMA_URL=http://host.docker.internal:11434`. Host Ollama `/api/tags` returns exactly one
model: `qwen2.5-coder:7b`. `.env` sets neither `SLM_TEMPERATURE` nor `NL_SUMMARY_MODEL`
nor `CHATBOT_CLASSIFY_MODEL` (all three are in `.env.example`), so the engine runs at
`SLM_TEMPERATURE=0.3` (`config.py:386`), asks for `NL_SUMMARY_MODEL=qwen2.5:7b-instruct`
(`config.py:1469`), which is not served, and the chat classifier also falls back to the
coder model.

**Direct Ollama HTTP.** `slm/AGENTS.md:13` names three modules; two are stale.
`query/slm_layer.py` and `query/rag_layer.py` only carry vestigial `urllib` imports and
call `call_slm`. `query/answer_entity.py:112-121` is the one real remnant: it hand-builds
an `/api/chat` payload (`temperature=0`, `num_predict=8`, `timeout=8`) and posts with
`urlopen`. It bypasses the backend switch, the ledger and token accounting, and is live
(`ANSWER_ENTITY_LLM_FALLBACK_ENABLED=True`, `config.py:2236`). `ingestion/semantic_layer_v2.py:139`
pings `/api/tags` directly, ingestion-only.

The chat tier has its own caller, `chatbot/llm.py` (§6.2), deliberately not an import of
this module.

---

## 2. Where the SLM is consulted during one query

The production path (`run_hybrid_query` → `_dispatch_single` → `run_query`) is
deterministic-first. SLM calls appear only at the points marked below. Purpose labels are
the ones the ledger records.

```
chat tier (chatbot/, separate caller — §6)
  classify | standalone_check | followup | smalltalk | delta-classify   ← 0–2 calls before the engine

run_hybrid_query                                      veda_hybrid.py
  L0 nl_simplify              [NL_SIMPLIFIER_ENABLED=False]     dormant
  L0 runtime_context                                           no SLM
  coordinator plan_route      source_routing                   ← ONLY at an ambiguous evidence boundary;
                                                                 fires even with SHADOW=1 (decision then discarded)
     └ MULTI / federated      doc_data_ground, operation_classify, semi_join_classify,
                              federated_plan, federated_struct_plan, query_understanding
  decomposition               decompose  [QUERY_DECOMPOSE_ENABLED=False] dormant (MULTI bypass live)
  classify → sql | rag | hybrid | nosql
    sql  → run_query          veda/pipeline.py
             fast path / planners / cache                       no SLM
             retrieve → rerank → anchor → ER → vet               no SLM (answer_entity raw HTTP: 1 word)
             join branch "sql"      sql_join           ← SLM fills SELECT/WHERE in a pinned skeleton
             single-table rung 8    sql_single_table   ← only after 7 deterministic rungs miss
             firewall gates 1-13                                 no SLM
             execute                                             no SLM
             L7b summary            nl_answer | insight_engine  ← ONE call, facts-only
             refusal copy           refusal_polish     ← refusals only, FEEDBACK_LLM_POLISH
          Tier-2 (on refuse/…)      envelope, then lg_node ×4   ← IR only, never SQL
             _tier2_finish          nl_answer | insight_engine
    rag  → run_rag_layer            rag_synthesis      ← one prose call over chunks
    hybrid → run_query (+ its nl_answer) + run_hybrid_layer   rag_synthesis over rows-as-text + chunks
    nosql → run_nosql_builder (no SLM) → nl_answer
  ≥2 answered items                 multi_summary      ← one call over the per-source answers
```

Observed per-query call counts from the 2,158 persisted traces: 757 queries made **no**
SLM call at all, 915 made one (almost always the summary), 298 made two, and a long tail
up to 11 (multi-source scopes that pay routing + understanding + federated planning +
per-part summaries).

---

## 3. SQL and IR generation sites

The model writes SQL text in exactly one module, `veda/generation.py`. Everything else
emits structured IR or labels that a deterministic builder turns into SQL.

### 3.1 Tier-1 — `veda/generation.py`

**`generate_sql` → `sql_single_table`** (`generation.py:144`, call `:205`).
`temperature=0, seed=0, num_predict=256, num_ctx=SLM_NUM_CTX(4096), timeout=240`. System:
"PostgreSQL expert, ONE read-only SELECT, no markdown, no semicolon" + `_domain_line()`
(interpret-only domain context). User: question, the single table, a recommended
projection block when it differs from the full column list, the flat column list, an
exact temporal column line ("do not use any other column for the date"), a ranking
line, a column glossary (≤ 12 columns × 80-char definitions, `SQL_COLUMN_GLOSSARY_*`),
phrase→column directives from `domain_synonyms`, and a parsed `LIMIT`. **No sample
values, no few-shot.** Output: strip fences, slice from the first `select`, strip `;`.
**No retries; a `RuntimeError` propagates** into `run_query`. Reached at
`pipeline.py:1818` only after the seven deterministic single-table rungs miss, and
`SINGLE_TABLE_DETERMINISTIC=True` (`config.py:1853`) short-circuits projection/date-range
/temporal-rank shapes inside the function first, so the SLM is left with measure rankings
and categorical/text filters. **LIVE, narrowed.**

**`generate_join_sql` → `sql_join`** (`generation.py:324`, call `:416`). Same model,
`num_predict=320`, **`timeout=120` hard-coded**. System: the FROM/JOIN block is fixed and
must be copied verbatim; the model may add only SELECT (alias-prefixed) and optional
WHERE/GROUP BY/ORDER BY, GROUP BY only on "per/by/each". User: the deterministic skeleton
from `veda/planning.build_skeleton`, per-alias columns capped at 20, entity display
columns resolved from `semantic/overrides.json` then heuristics, a per-table glossary
budget of `max(2, 12 // n_aliases)`, alias-qualified directives, date line, LIMIT. No
retries. Called from `planning.py:1379` inside `try_multitable`, and **also from Tier-2**
via `LANGGRAPH_SHARED_PLANNER → build_from_entities`. **LIVE.**

The incident note at `generation.py:209-216` matters for every other site: requesting a
`num_ctx` the Ollama host does not already serve forces a fresh model instance that blocks
until timeout (1,218 s/query observed; 16.3 s once `SLM_NUM_CTX` was used).

### 3.2 Tier-2 — the envelope + LangGraph IR path (`veda_hybrid.py::_tier2_sql:2286`)

Entered from `veda_hybrid.py:1706` only when the head returned
`refuse | qualifier_dropped | ungrounded | no_table | exec_error`, `TIER2_LLM_FALLBACK=True`,
and the head took ≤ `TIER2_SKIP_IF_HEAD_OVER_S` (120 s); a hard `TIER2_TIME_BUDGET_S`
(120 s) deadline is checked between SLM rounds.

1. **Envelope first** — `query/envelope_slm.py::emit_envelope` (`:117`, call `:123`),
   purpose `envelope`, `temperature=0, num_predict=512, json_format=True, timeout=240`.
   Seven closed intents (`count|measure|ratio|trend|compare|group|dimension_list`), an
   **opaque-handle contract** (`t1`, `c3`; never a real identifier), "you do NOT write
   SQL", 5 few-shot examples. User: alphabetically sorted table candidates (so order
   carries no relevance prior) and columns with **3 hard-coded sample values**. Parse:
   first `{`…last `}`; any failure → `None`. Skipped up front by
   `_envelope_inexpressible` (rankings, thresholds, negation). On success:
   `map_envelope_to_intent → validate_intent → build_sql → firewall.check → execute →
   _tier2_finish(…, "envelope")`. **LIVE** whenever Tier-2 runs.
2. **IR call** — `run_slm_layer` (`slm_layer.py:986`) immediately delegates to
   `query/slm_langgraph.run_langgraph_pipeline` because `USE_LANGGRAPH` defaults to
   `"true"` (`config.py:474`, not overridden in `.env`). Graph: `classify_intent →
   select_entity → select_columns → build_filters → assemble_ir`; the first four are SLM
   calls through `lg_nodes.py::_call_node` (`:102`, call `:110`), all labelled
   **`lg_node`** (per-node attribution impossible), `temperature=SLM_TEMPERATURE(0.3
   here)`, `num_predict=256`, `timeout=240`, **no `num_ctx`, no `json_format`**. Prompts
   (`lg_prompts.py`) are 6–12 lines each; the schema slice narrows per node (`TOP_K_TO_LLM=6`
   tables → columns of the chosen entity → only the selected columns for filters). No
   sample values, no value-filter hints. Any parse failure → `None` → deterministic
   degrade per node (SELECT/SIMPLE; most-frequent table; empty columns; empty AND tree).
   `node_assemble_ir` is LLM-free and re-validates every UUID against the retrieved set.
   **LIVE — the default IR path; 4 calls.**
3. **Dormant linear IR** — `slm_layer.py::_call_ollama` (`:582`, purpose `ir_emit`,
   `num_predict=SLM_IR_MAX_TOKENS=512`, 3 attempts via `SLM_MAX_RETRIES=2`) with the
   75-line `_SYSTEM_PROMPT` (UUID-only contract, 16 rules, forbidden keys), the
   REFERENCE TABLE user message, value-filter hints, `_normalize_ir`,
   `_inject_must_include_cols`, `_prune_hallucinated_uuids`. Runs only with
   `USE_LANGGRAPH=false`. `SLM_PROMPT_INCLUDE_VALUES` / `SLM_PROMPT_MAX_VALUES`
   (`config.py:1391-1392`) are imported here and referenced nowhere else — dead config.
4. **Join decision** — with ≥ 2 entities in the IR and `LANGGRAPH_SHARED_PLANNER=True`
   (`config.py:2286`, used only at `veda_hybrid.py:2519`), joins come from the
   deterministic graph planner `build_from_entities`; the model only *named* entities.
   The flag is misnamed (unrelated to the LangGraph library) and its config comment
   ("production unaffected until both are on") is stale — both are on by default.
5. **Repair loop** — `VALIDATION_REPAIR_LOOP_ENABLED=False` (`config.py:2268`) collapses
   `for _attempt in range(_max_repairs+1)` to one pass and disables the
   seed-from-Tier-1-refusal hint (`veda_hybrid.py:2457`). `_repair_hint_for`
   (`:2000-2027`) classifies firewall errors into IR-level nudges and is therefore unused.
6. **Validation** — one `veda.firewall.check` per branch (RBAC, AST/params, value
   grounding, strict qualifier completeness, IR equivalence, advisory semantics).
   Reject → `tier2_rejected`, head's refusal stands; exec error → `tier2_exec_error`.

### 3.3 Decomposition — `slm_layer.py::_call_ollama_decompose` (`:1264`)

Purpose `decompose`, `temperature=0, num_predict=256`, 3 attempts. Closed enum
`single | independent | dependent_nested`, bare query as user message, "when in doubt
choose single". Every decision is appended to `logs/decompose_log.jsonl`. Gated by
`QUERY_DECOMPOSE_ENABLED=False` (`config.py:2298`, "splits join queries wrongly"); one
live bypass at `veda_hybrid.py:944` for a ROUTED/MULTI decision. **DORMANT** (20 calls in
the log, all from the bypass or older runs).

### 3.4 Site table — generation

| Site | purpose | key params | Live? | Deciding flag | Caller |
|---|---|---|---|---|---|
| `veda/generation.py:205` | `sql_single_table` | T=0 seed=0 np=256 ctx=4096 to=240 | LIVE (narrowed) | `SINGLE_TABLE_DETERMINISTIC` | `pipeline.py:1818` |
| `veda/generation.py:416` | `sql_join` | T=0 seed=0 np=320 ctx=4096 **to=120** | LIVE | — | `planning.py:1379`; Tier-2 via shared planner |
| `query/envelope_slm.py:123` | `envelope` | T=0 np=512 json to=240 | LIVE | `TIER2_LLM_FALLBACK` + `_envelope_inexpressible` | `veda_hybrid.py:2364` |
| `query/lg_nodes.py:110` ×4 | `lg_node` | T=0.3 np=256 to=240 **no ctx** | LIVE, default IR | `USE_LANGGRAPH=true` | `slm_langgraph.py:158` ← `slm_layer.py:990` |
| `query/slm_layer.py:592` | `ir_emit` | T=0.3 np=512 ctx=4096, 3 attempts | DORMANT | `USE_LANGGRAPH=false` only | `slm_layer.py:1054` |
| `query/slm_layer.py:1268` | `decompose` | T=0 np=256, 3 attempts | DORMANT (+1 bypass) | `QUERY_DECOMPOSE_ENABLED=False` | `veda_hybrid.py:1439`, `:944` |
| `query/nl_simplifier.py:100` | `nl_simplify` | endpoint=generate T=0.1 np=64 to=10 | DORMANT | `NL_SIMPLIFIER_ENABLED=False` | `veda_hybrid.py:1318` |

---

## 4. Routing, RAG, federated and presentation sites

| Site | purpose | key params | Prompt gist and guard | Live? | Gate |
|---|---|---|---|---|---|
| `query/routing_slm.py:102` via `source_coordinator.py:621` | `source_routing` | **no np / ctx / timeout** (240 s default), T=0 | Closed `SINGLE\|MULTI\|NONE` over ≤ 3 items / 8 columns / 5 docs per candidate; V2 prompt adds 2 MULTI few-shots (`ROUTING_SLM_MULTI_FEWSHOT_ENABLED=1`). `_normalize_sid` strips the prompt's own `source_id=` label; invalid → `ClarificationRequired`, never a silent source. A valid NONE is overridden to SINGLE for a STRONG tabular leader. | LIVE at an ambiguous boundary. `plan_route` runs **before** the SHADOW gate (`veda_hybrid.py:841-865`), so the call fires and its decision is then discarded. | `MULTISOURCE_ROUTING_ENABLED=1` |
| `query/operation_classifier.py:185` | `operation_classify` | T=0 json, no np/to | Closed operation enum; "do NOT write SQL, do NOT name tables". Post-hoc `_feasible`/`_expressible` refuse negation and count-threshold shapes no planner can express. | LIVE on the federated route | `OPERATION_CLASSIFIER_ENABLED=1` |
| `query/semi_join_planner.py:133` | `semi_join_classify` | T=0 json | Candidate-index answer, deterministic `validate()` re-check | LIVE, federated | `FEDERATED_SEMI_JOIN_STRUCTURED_ENABLED=1` |
| `query/doc_data_planner.py:88` | `doc_data_ground` | T=0 json | Entities must literally appear in the chunk text; data column is an index into candidates | LIVE, doc+data MULTI | `DOC_DATA_GROUNDING_ENABLED=1` |
| `query/federated_route.py:381` / `:513` | `federated_plan` / `federated_struct_plan` | T=0 json | `federated_plan` is the one place the SLM writes cross-source SQL per metric (validated: SELECT-only, must project the group key, 2 attempts with the prior error fed back). `struct_plan` is fields-only. | LIVE with ≥ 2 sources in scope | `FEDERATED_*` all `"1"` |
| `veda/understanding/extractor.py:89` | `query_understanding` | T=0 seed=0 np=320 ctx=4096 to=60 | Concept catalog (≤ 60) + question; closed intents; refuse/clarify/answer rubric | DORMANT via `orchestrator.py:63`; **LIVE via `federated_route.py:692`** (unflagged) | `QUERY_UNDERSTANDING_ENABLED=0` gates only the orchestrator |
| `query/rag_layer.py:506`, `:666`, `:713` | `rag_synthesis` | T=`SLM_TEMPERATURE`(0.3) **np=SLM_MAX_TOKENS=2048** to=240, coder model | Pure RAG: "ONLY the provided context, cite doc+page"; hybrid: `[DB]`/`[DOC]` prefixes, SQL rows are ground truth (≤ `HYBRID_MAX_RESULT_ROWS=20`), chunks ≤ `RAG_TOP_K=5` inlined full-text, sub-0.50 cosine chunks dropped when SQL truth exists. Failure → raw chunk text. | LIVE for intent `rag`/`hybrid` | none |
| `veda/feedback.py:206` | `refusal_polish` | T=0.1 np=96 to=6 | Rephrase `{why, what_needed, suggestions}` into one sentence; `clarify` exempt | LIVE, refusals only | `FEEDBACK_LLM_POLISH=True` |
| `query/answer_entity.py:115` | *(none — raw urlopen)* | T=0 np=8 to=8 | One relation word or `none`; must still match an FK edge | LIVE | `ANSWER_ENTITY_LLM_FALLBACK_ENABLED=True` |
| `veda_hybrid.py:481` | `multi_summary` | T=0.1 np=210 model=`_nl_model()` **no timeout** | see §5.5 | LIVE, ≥ 2 OK items | none |

Two consequences worth stating plainly. First, on a multi-source scope the "shadow"
coordinator is not free: every ambiguous query pays a ~5.7 s `source_routing` call whose
result is thrown away. Second, a hybrid turn pays two summary-class calls — the SQL head's
own `nl_answer` inside `run_query`, then `rag_synthesis` over the same rows as text — and
the deterministic patterns are blended into the RAG prose afterwards
(`veda_hybrid.py:1821-1827`).

---

## 5. The summary layer (L7b) in depth

### 5.1 Entry points and the single-call rule

The summary is produced by `query/result_explainer.py` (`query/nl_answer.py` is a
14-line re-export shim kept for old imports). Five paths call it after execution:

| Head | Call site | What it passes | Notes |
|---|---|---|---|
| Tier-1 SQL | `veda/pipeline.py:2267-2428` | `table`, `semantic_model`, `rank_column`, **all** findings, `result_shape`, `analytical_context` (intent, aggregate operator, ranking column, temporal window, explicit-id request) | the reference implementation |
| Tier-2 SQL | `veda_hybrid.py::_tier2_finish:2041-2225` | `semantic_model`, **top-2** findings, `result_shape` | no `table`, no `rank_column`, no `analytical_context` (§8-3) |
| NoSQL | `veda_hybrid.py:2653-2690` | rows only | analytics computed **after** the summary; patterns blended deterministically |
| Federated | `query/federated_route.py:1110-1131` | `table="federated"`, no semantic model, no timeout arg (→ 45 s) | |
| Hybrid | `veda_hybrid.py:1802-1830` | — | prose comes from `rag_synthesis`; SQL-head patterns blended in afterwards |

The rule "**one post-query SLM call**" is enforced structurally: `INSIGHT_ENGINE_ENABLED`
(default **off**, `config.py:1519`; `.env.example` says `true`, `.env` here says nothing)
selects `run_insight_engine`; otherwise `run_nl_answer`. If the insight engine raises, the
flag is flipped off *for that turn only* and `run_nl_answer` runs instead
(`pipeline.py:2350-2397`), so a failed insight call can cost two calls.

A verified-cache hit does **not** skip the summary: the cache stores `(query → sql)` only
(`veda/cache.py:85-108`), and nothing in the L7b block is gated on `from_cache`. A cached
query skips retrieval and SQL generation but pays the full summary call.

### 5.2 Order of operations inside L7b (Tier-1)

1. `deterministic_fallback_answer` is computed **first** so the result always has an
   answer even if everything below fails (`pipeline.py:2289`).
2. `result_analyzer.analyze_result(query, sql, cols, rows, sm, table, max_rows=200,
   query_intent, confidence_inputs, params)` — pure Python, no SLM, no new SQL (§5.3).
   `analytics_summary(_ictx)` rides the result dict as `analytics` for the API tier.
3. Findings are split: `_all_findings` (every pattern, ≤ 6) goes to the summariser;
   `_pattern_details` (top 2) is the deterministic-blend fallback (`:2325-2330`).
4. `_analytical_ctx` is **reused from this run's own understanding** (aggregate operator
   via `veda.planning.aggregate_operator`, ranking column, temporal window,
   `user_requested_identifier`) — never re-derived by the summary layer (`:2335-2347`).
5. One SLM call (`run_insight_engine` XOR `run_nl_answer`).
6. If no SLM prose wove the patterns in (`_slm_wove_patterns` False), `blend_patterns`
   appends the top-2 as one natural clause (`:2404-2406`).
7. `record_result_stages(...)` writes `execution / result_analysis / summary /
   visualization` sections to the trace (`:2418-2422`); `_done()` later adds
   `nl_summary` (tokens + **actual** model) from the usage buffer (`:268-272`).

### 5.3 What `result_analyzer` hands the narrator (`veda/result_analyzer.py`)

`analyze_result` (`:548-621`) → `InsightContext` (`:238-272`): question, sql, table,
result_type, row_count, columns, entities/dimensions/measures/filters/orderings/limit/
distinct (from `business_explain.extract_sql_facts`, filter values only when `params`
are passed), `column_stats`, `sample_rows`, semantic model, `result_shape`,
`primary_entity`, `related_entities`, `available_measures/dimensions`, `patterns`,
`chart_candidates`.

- **Column stats** (`:275-300`) over `rows[:RESULT_ANALYZER_MAX_ROWS=200]`: kind
  (temporal by name hint or all-date sample; numeric; categorical), role
  (`identifier|dimension|measure|date|boolean|text`, semantic-model type first, then
  suffix/camelCase identifier heuristics), null/distinct counts, min/max/avg/median,
  top-5 values. Role is what keeps identifiers off chart axes and out of patterns.
- **Shape** (`:200-223`): non-multi-row → `SCALAR`; then in order `PIVOT` (≥ 2 measures +
  dimensions), `TREND` (any temporal-kind column + numeric measure), `RANKING` (`limit`
  and `orderings`), `DISTRIBUTION` (dimensions + COUNT-only aggregations), `GROUPED`,
  else `DETAIL_TABLE`. Because `has_temporal` falls back to *any* date column, TREND can
  fire on an unrelated date; and "list all users" with `LIMIT 100` + an ORDER BY is a
  `RANKING` (seen live).
- **Patterns** (`:364-492`, thresholds `:354-361`, ≤ 6 by strength): `missing_values`
  (≥ 30 % null, n ≥ 5), `dominance` (top share ≥ 60 % on a dimension/boolean **not** in
  an SQL equality filter — the one place the SQL AST suppresses a trivial insight),
  `outlier` high/low (> 3 σ, ≥ 8 numeric), `growth/decline` (TREND, ≥ 10 % first→last
  sorted by `str(t)`), `leader/laggard` (unconditional once ranking/grouped shapes have
  ≥ 2 numeric rows), `concentration` (≥ 50 % of total), `distribution` and `spread`
  (**always emitted** for those shapes), `top_gap` (RANKING, ≥ 50 % gap between
  **row-order** positions 0 and 1 — correct only if ORDER BY survived). The pattern
  sweep is bounded by `ANALYSIS_MAX_ROWS=50 000`, the same bound the explainer's metrics
  use, so "vs average" statements are over one population.
- **Grounding vocabulary** from the semantic model: `primary_entity`, up to 10
  measures and 10 dimensions of the **whole table**, up to 8 FK-adjacent entities.
- **Charts**: `CANONICAL_CHART_FOR_SHAPE` = RANKING→bar, TREND→line, DISTRIBUTION→pie,
  GROUPED→bar; SCALAR/DETAIL_TABLE/PIVOT get none. `chart_confidence` is 0.0 for any
  identifier axis, 0.95/0.8 for canonical pairings, 0.6/0.3 otherwise; the gate is
  `VISUALIZATION_CONFIDENCE_THRESHOLD=0.6`.
- `analytics_summary` (`:523-545`) is the JSON-safe projection crossing to the API tier:
  shape, type, row_count, table, entity vocabulary, `display_columns`, column
  name/kind/role only, patterns, chart candidates. It omits `filters`, `orderings`,
  `limit`, `query_intent` and `confidence_inputs`.

### 5.4 `run_nl_answer` — the prompt and its guards (`result_explainer.py:438-610`)

**Input reduction.** `_extract_facts(columns, rows, rank_column)` (`:321`) builds the
*only* data the model sees:

- first 6 columns (+ the ranking column if outside them);
- 1 row → `{"row_count":1,"fields":{…}}`; else ≤ 5 `sample_rows` + `note: "showing 5 of N"`;
- `ranked_by` when the caller resolved an ORDER BY column;
- `metrics`: per numeric column `count/min/max/sum/mean/median` over at most
  `ANALYSIS_MAX_ROWS` rows, skipping `id`/`*_id` columns by name
  (`_numeric_aggregates`, `:284`). If the result exceeds that bound the payload carries
  `metrics_partial / metrics_scanned` and the prompt tells the model to describe totals
  as sample-based.

**Prompt assembly** (`:537-556`), in order: user question → `Extracted data: {facts}` →
column glossary from the semantic model (`business_definition` / `analytics_role` for up
to 6 columns, only when `table` **and** `semantic_model` are passed) → resolved analytical
context → ranking line → `Verified findings already computed` (≤ 5 in analytical mode,
≤ 2 in brief) → partial-metrics note → a currency-neutral style exemplar → role line →
shape guidance (`_SHAPE_GUIDANCE`) → the hard rules (use only numbers shown, never
compute, never infer causes, no markdown, keep ids only if asked, no invented currency).

**Mode.** `_summary_mode` (`:396`) is `analytical` only for
`RANKING/GROUPED/DISTRIBUTION/TREND/PIVOT` with > 1 row; everything else is `brief`.
Analytical gets `NL_SUMMARY_ANALYTICAL_MAX_TOKENS = 320`; brief gets
`NL_SUMMARY_MAX_TOKENS + 50 = 210`.

**The call.** `call_slm(prompt, purpose="nl_answer", temperature=0.1, num_predict=…,
endpoint="chat", timeout=NL_SUMMARY_TIMEOUT_MS/1000 = 45 s, model=_nl_model())`. No
system prompt; everything is in the user message. `_nl_model()` (`:418`) returns
`NL_SUMMARY_MODEL` only if the backend's `/api/tags` lists it, else the backend's own
model — on this machine, always `qwen2.5-coder:7b`.

**Output guards**, in order:

1. empty → `ValueError` → fallback;
2. `NL_SUMMARY_NUMERIC_GUARD` (default on): `_answer_numbers_grounded` (`:189`) parses
   every number in the prose (understands `₹/$`, commas, `%`, `K/L/M/Cr/lakh/crore/bn`)
   and requires each to be within ±2 % (floor ±2) of a number in the facts/metrics/
   findings, **or** an integer ≤ `max(row_count, 12)` (treated as a count/ordinal). One
   ungrounded figure → the whole prose is discarded;
3. `_strip_invented_currency` (`:169`) removes any of `$₹€£¥₩₽` that does not occur in the
   facts JSON.

**Fallback** (`:591-606`): `template_answer` for empty / single scalar / single row
(`"The count is 137."`, `"Result: a 1, b 2."`), else `deterministic_fallback_answer`
(`"Returned N row(s). First: …"`), then `blend_patterns` with the top-2 findings. A
warning with the exception class is logged unconditionally. `slm_used` reports which path
produced the prose.

### 5.5 `run_insight_engine` — the dormant richer variant (`:877-972`)

Same facts extractor and glossary, plus `_stats_block` (per-column stats excluding
identifiers), `_patterns_block` (≤ 6 detected patterns), `_grounding_block` (primary
entity, available measures/dimensions, FK-adjacent entities, filters already applied).
Asks for a JSON object `{summary, insights[], visualization{type,x_axis,y_axis,reason},
follow_up_questions[]}` with `json_format=True`, `num_predict = 280`, timeout
`INSIGHT_ENGINE_TIMEOUT_MS = 45 s`, **`model=NL_SUMMARY_MODEL` directly, not
`_nl_model()`** (`:944`). Output is validated deterministically: `validate_visualization`
(columns must exist, no identifier axes, shape must have a canonical chart, type coerced
to the canonical one, `chart_confidence ≥ 0.6`); `validate_follow_up_questions` drops any
suggestion that names nothing in the result's vocabulary, and `INSIGHT_FOLLOW_UPS_ENABLED`
(default off) empties them anyway. The insight engine's summary has **no numeric guard
and no currency strip** (§8-5).

### 5.6 `_summarise_multi_answers` — the cross-source summary (`veda_hybrid.py:457-490`)

When ≥ 2 items of a `MultiResult` answered (independent merge or compound fan-out),
one `multi_summary` call combines the per-source `answer` strings (never rows) into a
short paragraph, `temperature=0.1`, `num_predict=210`, `model=_nl_model()`, no timeout
argument (→ 240 s). Guard: the set of digit-strings in the output must be a subset of
those in the inputs; otherwise the deterministic `"From source X: … From source Y: …"`
join is used. Only 2 such calls exist in the whole trace log; the path is essentially
unexercised live. The chat tier lifts it over `answer` and keeps `source_answers`, which
nothing renders (§6.3).

### 5.7 What the trace records about the summary

- `slm.calls[]` — the truth: purpose `nl_answer`, the **model actually used**, duration, ok.
- `summary` (from `record_result_stages`) — `engine`, `model=NL_SUMMARY_MODEL` **from
  config, not from the call**, `success=bool(engine)`, `answer_chars`.
- `nl_summary` (from `_done`) — `summary_tokens`, `summary_model` from the usage buffer
  (actual model), `insight_engine_failed/_error`. **Not in `explain._SECTIONS`, so
  `to_dict()` drops it**; 0 of 2,158 persisted traces carry it.
- `llm_usage.per_purpose.nl_answer` — intended token totals; **never populated** (§1).

Post-fix traces show `slm.calls[].model = qwen2.5-coder:7b` and `summary.model =
qwen2.5:7b-instruct` side by side for the same query — the two sections disagree.

---

## 6. The chat tier and the request path

### 6.1 Sequence

1. `POST /api/v1/conversations/query` → `apps/chat/views.py:89`. Auth, RBAC
   (`permitted_source_ids`; empty → synthetic denied turn, engine never called), scope
   resolution, `ConversationQueryService`, user message saved. `stream=True` → SSE.
2. `services.py:197 run_turn` → `_run_streamed:268` runs `chatbot.run.run_chat_turn` on a
   daemon thread and bridges `on_event` through a queue; `business_friendly_message`
   swaps only the displayed phase text.
3. `chatbot/run.py:60` wraps the turn in `collect_usage()`; graph (`chatbot/graph.py:87-120`):
   `memory_read → classify → {smalltalk | runtime_context | context_resolve | call_engine}`,
   then `call_engine → {memory_write → format_reply | ask_clarification}`. Routing after
   classify is history-based, not label-based.
4. `call_engine_node` (`nodes.py:614`) → `apps/query/inference_client.py` HTTP SSE
   `POST {INFERENCE_URL}/v1/run_hybrid_query/stream`, single `INFERENCE_TIMEOUT_S=300`
   for connect and stream. Headers carry source ids, tenant, request id, data scope,
   source profiles, no-cache.
5. `inference/main.py:145` middleware sets `RequestContext` (malformed data scope fails
   closed to "nothing addressable"); `inference/routes/hybrid.py:134` runs
   `run_hybrid_query` on a daemon thread with `copy_context()`; `_serialize` strips
   `{"context","trace","_debug"}` at every depth and converts `Decimal→float`. **The
   trace never leaves the inference tier.**
6. Engine L7b (§5) produces `answer`, `analytics`, optional `insights/visualization/
   follow_up_questions/confidence`, `explain`, `business_intent`, `usage`.
7. `nodes.py:545 _extract_engine_result` reads item 0's nested `result`; when a
   `MultiResult.summary` exists with ≥ 2 items it **replaces** `answer` and stores the
   per-source answers under `source_answers`.
8. `format_reply_node` (`nodes.py:848`): `reply_text = res0.get("answer") or "Here's
   what I found."` — verbatim. `services.py:365 _build_reply_events` emits
   `thinking? → content* → visualization? → explainability → usage → insights?`; the
   summary is content block 1 (`is_summary:true`, insights appended as bullets), the
   markdown table (≤ 20 rows, `—` for null, no currency prefix) is block 2.
   `TurnEventAccumulator` folds the stream into `ChatMessage.content/metadata`.

**The chat tier never re-summarises, truncates, or re-prompts over the engine's prose.**
There is no `call_slm` under `apps/chat/`; the only mutation is the insights-bullet
append.

### 6.2 Chat-tier SLM sites (`chatbot/llm.py::call_slm`, defaults T=0.1, max_tokens=200, timeout=45, returns `None` on any failure)

| Site | purpose | model | max_tokens | Decision | Fallback |
|---|---|---|---|---|---|
| `nodes.py:274` `classify_node` | `classify` | `CHATBOT_CLASSIFY_MODEL` (→ coder here) | 200 | `action ∈ {smalltalk, followup, clarify_reply, answer}` + `delta_type`/`slot_candidates` when a frame exists (one merged call) | `action="answer"` → engine |
| `nodes.py:101` `_depends_on_history` | `standalone_check` | same | **5** | second opinion turning "smalltalk" into `followup` | keep smalltalk |
| `nodes.py:414` `smalltalk_node` | `smalltalk` | same | 60 | reply text beyond canned greetings | `FALLBACK_REPLY` |
| `nodes.py:498` `context_resolve_node` | `followup` | same | 80 | rewrite a follow-up into a standalone query when no frame exists | message unchanged |
| `memory/classify.py:116` `classify_delta` | *(omitted → `"chatbot"`)* | *(omitted → `SLM_MODEL_NAME`)* | 80, T=0 | fallback `delta_type` | `"ambiguous"` |

Deterministic fast paths skip the SLM entirely: greeting/thanks/bye regexes, the
runtime-context regex, drill-up, reset. Each avoided classify call saves ~20 s on this
deployment (`nodes.py:112`).

`chatbot/llm.py` vs `slm/_call_slm.py`: separate by design (the api tier must not import
`veda_core`). Chat: `(system, user)` signature, returns `None` on failure, per-call
backend dispatch, `os.environ` + `load_dotenv`, no `keep_alive`, no `json_format`, no
`num_ctx`, no per-call ledger. Both share the thread-local `collect_usage` shape, and
`chatbot/run.py:88-99` adds the supervisor's tokens onto `engine_result["usage"]` so the
UI figure includes classify/smalltalk/followup spend.

### 6.3 What the chat tier does with each engine field

| Field | Consumer | Outcome |
|---|---|---|
| `answer` | `nodes.py:851` → `services.py:429` | content block 1, verbatim |
| `insights` | `services.py:441`, `:420` | bullets under the summary + `insights` event |
| `follow_up_questions` | `services.py:423` | `insights` event only; gated server-side |
| `cols`/`rows` | `services.py:445-455` | markdown table via `table_rendering.py` |
| `analytics` | `services.py:452,470,489` | `display_columns` projection; recommender roles; 3rd-tier chart fallback |
| `visualization` | `services.py:479` | 2nd-tier chart fallback, rebuilt into real `chart_data` |
| `explain` | `services.py:400` | `explainability` event (fixed-shape fallback) |
| `usage` | `services.py:414` | `usage` event + `latency_ms`; persisted in `ChatMessage.metadata` |
| `trace` | — | stripped at `inference/routes/hybrid.py:61` |
| `business_intent` | — | crosses the wire, **no consumer** in `apps/` or `chatbot/` |
| `MultiResult.summary` / `source_answers` | `nodes.py:576-586` | summary replaces `answer`; `source_answers` **never rendered** |

### 6.4 Startup warm-up (`inference/loaders.py::hydrate`)

Semantic model (gates `ready`) → retrieval engine → BGE-M3 dense+sparse → reranker →
`prewarm()` for `SLM_MODEL_NAME` → a second `prewarm(model=NL_SUMMARY_MODEL)` that sets
`nl_summary_model_warm=True` (`:113-114`). Because `prewarm` never raises, the flag is
set and `nl_summary_slm_unreachable` never recorded even when the model is absent —
live today: `hydrate complete: {... 'nl_summary_model_warm': True, 'degraded': []}`.

### 6.5 Where usage and per-call records end up

- `ChatMessage.metadata["usage"]` — the only durable record of chat-turn token spend.
- `QueryLog` — **not written on the chat path** (`_audit` is invoked only from
  `/api/v1/query`), so `/metrics` sees no chat traffic.
- `ExplainTrace` — `slm.calls[]` (works) and `llm_usage` (never populated, §1), both
  persisted only to `veda_core/logs/explain_trace.jsonl` (`EXPLAIN_TRACE_PERSIST=True`,
  verbose, including SQL text up to 2,000 chars). No DB table.
- MLflow — `mlflow_observability/mapper.py:379-389` maps `sections.llm_usage.*` to
  `total_prompt_tokens / total_completion_tokens / total_tokens / llm_calls`; with
  `llm_usage` never present these metrics are always zero.

---

## 7. Evidence from the persisted traces

`veda_core/logs/explain_trace.jsonl` holds 2,158 verbose records. The last
`/api/generate` 404 on an `nl_answer` call is record 1,479; the 678 records after it are
"post-fix" (the 2026-09-16 chat-endpoint + `_nl_model()` change).

| Metric (post-fix, 678 queries) | Value |
|---|---|
| `nl_answer` calls | 404, all `ok`, all on `qwen2.5-coder:7b` |
| `nl_answer` duration | median 5.9 s, p90 11.8 s, max 42.4 s |
| share of end-to-end time | median 11 %, p90 95 % (fast-path/cache queries where the summary *is* the latency) |
| answered end-to-end | median 66 s, p90 326 s |
| summary length | median 123 chars, p90 282 chars |
| result shapes | SCALAR 123 · DETAIL_TABLE 108 · RANKING 67 · TREND 46 · DISTRIBUTION 43 · none 118 |
| records with `llm_usage` / `nl_summary` section | 0 / 0 |

Whole-log per-purpose view (all 2,158 records):

| purpose | calls | median | p90 | failures | note |
|---|---|---|---|---|---|
| `nl_answer` | 874 | — | 9.3 s | 470 | 469 are the pre-fix `/api/generate` 404 |
| `query_understanding` | 328 | 8.7 s | 17.0 s | 0 | from `federated_route.py:692`, not the dormant L4 layer |
| `rag_synthesis` | 289 | 7.5 s | 18.2 s | 0 | |
| `lg_node` | 269 | 7.4 s | 17.5 s | 0 | Tier-2 LangGraph nodes, 4 per Tier-2 run |
| `source_routing` | 146 | 5.7 s | 10.3 s | 0 | fires in shadow mode |
| `refusal_polish` | 137 | 3.9 s | 6.0 s | 35 timeouts | 6 s budget |
| `operation_classify` | 80 | 4.8 s | 6.7 s | 0 | |
| `envelope` / `federated_struct_plan` | 67 / 67 | 7.2 s / 8.8 s | 15.4 s / 13.4 s | 0 | |
| `sql_single_table` | 55 | 3.8 s | 8.7 s | 0 | Tier-1 rung 8 |
| `sql_join` | 37 | 14.8 s | 25.0 s | 0 | Tier-1 join fill-in |
| `decompose` | 20 | 5.6 s | 8.7 s | 0 | |
| `multi_summary` | 2 | 8.8 s | — | 0 | |
| `ir_emit` / `nl_simplify` / `insight_engine` | 0 | | | | dormant, as expected |

The last six queries show the shapes of a turn: a single-source scalar is `nl_answer`
alone (~3 s); a multi-source scope adds `source_routing` (~5 s) first; a federated probe
adds `query_understanding` (~7 s).

The `summary` section reads `('run_nl_answer', success=True, 'qwen2.5:7b-instruct')` for
**all 879** records that have one — including the 470 whose SLM call failed and whose
prose came from the template fallback. `summary.success` is therefore not a measure of
SLM success (§8-2).

---

## 8. Findings

Ordered by impact. None are fixed by this analysis; anchors are for the fix.

1. **Summary model is silently the coder model on this deployment.**
   `NL_SUMMARY_MODEL=qwen2.5:7b-instruct` is not served by the host Ollama;
   `_nl_model()` (`result_explainer.py:418`) falls back to `qwen2.5-coder:7b`. The prose
   quality rationale in `config.py:1460-1468` does not hold here, and
   `CHATBOT_CLASSIFY_MODEL` is unset so the chat classifier runs on the coder model too.
   Fix is operational (`ollama pull qwen2.5:7b-instruct` on the host, and set the two
   model vars in `.env`), but see 4.

2. **The trace lies about the summary model and about summary success.**
   `record_result_stages(summary_model=NL_SUMMARY_MODEL from config, summary_ok=bool(engine))`
   (`pipeline.py:2415-2421`, `veda_hybrid.py:2214-2221`). The model should come from
   `slm.calls[]`, and `success` should be `nl.slm_used`, which `NLAnswerResult` already
   carries (`result_explainer.py:56`) but neither caller reads.

3. **Tier-2 summary parity gap.** `_tier2_finish` passes only the top-2 findings, no
   `table` (so no column glossary), no `rank_column`, no `analytical_context`
   (`veda_hybrid.py:2176-2179`) versus Tier-1's full set (`pipeline.py:2380-2388`). A
   Tier-2 ranking answer can narrate the wrong column and never enters analytical mode
   with the full findings list. NoSQL and federated paths are thinner still.

4. **`/readyz` reports the summary model warm when it does not exist.** `prewarm()` never
   raises (`_call_slm.py:421-427`), so `hydrate()` sets `nl_summary_model_warm = True`
   (`inference/loaders.py:113-114`) unconditionally; the `nl_summary_slm_unreachable`
   degraded flag can never fire.

5. **`run_insight_engine` is not protected the way `run_nl_answer` is.** It passes
   `model=NL_SUMMARY_MODEL` directly (`result_explainer.py:944`), so on this host it would
   404 on every call if enabled, and its summary skips both the numeric guard and the
   currency strip. Dormant today (`INSIGHT_ENGINE_ENABLED=false`), but `.env.example`
   ships it as `true`.

6. **`llm_usage` is never populated; MLflow token metrics are always zero.**
   `_fold_usage` (`_call_slm.py:97`) has no caller, so `get_usage()` never sees a call
   and `ExplainTrace._stamp` (`explain.py:220-232`) never writes the section (0 of 2,158
   traces). `mlflow_observability/mapper.py:379-389` reads only that section. The
   `/api/generate` branch also skips `_note_usage`. Token totals survive only via the
   thread-local `collect_usage` path into `tr.total_*` and `ChatMessage.metadata`.

7. **`nl_summary` never reaches disk.** Written at `pipeline.py:268-272` and `:2373` but
   absent from `explain._SECTIONS` (`explain.py:46-58`), so `to_dict()` drops it. This is
   the only section that records the *actual* summary model and token count.

8. **Temperature 0.3 in this clone.** `.env` has no `SLM_TEMPERATURE`; `config.py:386`
   defaults to 0.3 with a comment describing the exact non-determinism this causes. The
   summary call pins its own 0.1, but `lg_node`, `ir_emit`, `rag_synthesis` and
   `source_routing` use `config.SLM_TEMPERATURE`.

9. **`lg_nodes.py:110` sends no `num_ctx`** — the only IR/SQL site that omits it, on the
   default Tier-2 path, against two in-repo incident notes (`generation.py:209-216`,
   `nl_simplifier.py:105-107`) documenting that omission as the 1,218 s failure mode.

10. **Shadow routing is not free.** `plan_route` runs before the `MULTISOURCE_ROUTING_SHADOW`
    gate (`veda_hybrid.py:841-865`), so an ambiguous multi-source query pays a ~5.7 s
    `source_routing` call (no `num_predict`, no timeout → 240 s worst case,
    `routing_slm.py:102`) whose decision is discarded.

11. **`answer_entity.py` bypasses the seam.** Its direct `/api/chat` post
    (`answer_entity.py:112-121`) is invisible to the ledger, token totals and the backend
    switch; `slm/AGENTS.md:13` and `ARCHITECTURE.md §10` wrongly list `slm_layer.py` and
    `rag_layer.py` alongside it.

12. **Numeric guard leniency.** Any integer ≤ `max(row_count, 12)` passes as a "count"
    (`result_explainer.py:197-200`). For a 100-row result the model may state any
    integer up to 100, including an invented percentage, without tripping the guard.
    (The live "97%" answer seen today was legitimate — a detected pattern — but an
    invented one would also pass.)

13. **Row-count truncation is narrated as a total.** `execute_sql` fetches up to 1,000
    rows, the facts payload's `row_count` is `len(rows)`, and `record_result_stages` sets
    `truncated = len(rows) >= 20`. A `LIMIT 100` result is narrated as "100 records" with
    no hint that more exist (live example today: *"100 records show various projects…"*).

14. **Dead config and stale comments.** `NL_ANSWER_FAST_TIMEOUT_MS`, `NL_ANSWER_MAX_ROWS`,
    `NL_SUMMARY_MAX_ROWS` (imported or defined, never used); `SLM_PROMPT_INCLUDE_VALUES`
    / `SLM_PROMPT_MAX_VALUES` (imported at `slm_layer.py:52-53`, never referenced);
    `SLM_MAX_RETRIES` honoured only in the dormant linear IR path; `.env.example`
    timeouts (25 000 ms) are half the `config.py` defaults (45 000 ms); the
    `LANGGRAPH_SHARED_PLANNER` comment claims production is unaffected.

15. **Computed but unrendered.** `business_intent` (`veda_hybrid.py:2208`, deterministic
    from the executed SQL) and `source_answers` (`nodes.py:579`) cross the wire and are
    dropped by the chat UI. Chat turns also write no `QueryLog` row, so `/metrics` does
    not observe chat traffic.

---

## 9. Configuration reference (summary-relevant)

| Flag | Default | Live here | Effect |
|---|---|---|---|
| `SLM_BACKEND` | `ollama` | `ollama` | backend strategy |
| `SLM_MODEL_NAME` | `qwen2.5-coder:7b` | same (via `.env`) | every non-summary call |
| `SLM_TEMPERATURE` | `0.3` | **0.3** (not in `.env`) | generation/routing/IR calls |
| `SLM_TIMEOUT_SECS` | 240 | | default per-call timeout |
| `SLM_NUM_CTX` | 4096 | | must match what the host serves |
| `USE_LANGGRAPH` | `true` | | Tier-2 IR via 4 `lg_node` calls |
| `TIER2_LLM_FALLBACK` / budgets | True / 120 s / 120 s | | Tier-2 gate |
| `VALIDATION_REPAIR_LOOP_ENABLED` | False | | single Tier-2 pass |
| `NL_ANSWER_ENABLED` | True | | gates L7b entirely |
| `NL_SUMMARY_MODEL` | `qwen2.5:7b-instruct` | **falls back to coder** | summary model |
| `CHATBOT_CLASSIFY_MODEL` | → `SLM_MODEL_NAME` | **coder** (not in `.env`) | chat classify/smalltalk/followup |
| `NL_SUMMARY_TIMEOUT_MS` | 45 000 | | summary call timeout (covers a cold model swap) |
| `NL_SUMMARY_MAX_TOKENS` | 160 (+50 at call) | | brief-mode budget |
| `NL_SUMMARY_ANALYTICAL_MAX_TOKENS` | 320 | | analytical-mode budget |
| `NL_SUMMARY_MAX_FINDINGS` | 5 | | findings handed to analytical mode |
| `NL_SUMMARY_NUMERIC_GUARD` | true | | reject ungrounded numbers |
| `ANALYSIS_MAX_ROWS` | 50 000 | | bound for metrics + pattern sweep |
| `RESULT_ANALYZER_MAX_ROWS` | 200 | | rows profiled for column stats and sample rows |
| `INSIGHT_ENGINE_ENABLED` | false | | JSON summary+insights+viz call |
| `INSIGHT_FOLLOW_UPS_ENABLED` | false | | surface follow-ups |
| `INSIGHT_ENGINE_TIMEOUT_MS` | 45 000 | | |
| `VISUALIZATION_CONFIDENCE_THRESHOLD` | 0.6 | | chart gate |
| `FEEDBACK_ENABLED` / `FEEDBACK_LLM_POLISH` | True / True | | refusal copy (`refusal_polish`) |
| `MULTISOURCE_ROUTING_ENABLED` / `_SHADOW` | 1 / 1 | | coordinator runs, decision observe-only |
| `INFERENCE_TIMEOUT_S` | 300 | | Django → inference HTTP + stream |
| `EXPLAIN_TRACE_ENABLED/VERBOSE/PERSIST` | True/True/True | | full traces to `veda_core/logs/explain_trace.jsonl` |
