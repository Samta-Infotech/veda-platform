"""VEDA · The L1→L7 orchestrator (run_query)."""
import os, re, sys, time, json, logging, threading
from query.ranking_parser import parse_ranking
from veda.cache import save_verified_query, verified_cache_lookup
from veda.execution import execute_sql
from veda.generation import generate_sql
from veda.planning import existence_mode, try_multitable
from veda.routing import recommended_projection, select_primary_table, vet_primary
from veda.runtime import get_engine
from veda.validation import qualifier_completeness, validate_and_parameterize, value_grounding
from utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_temporal_column(table, sm):
    """Canonical temporal column of `table` — schema-metadata driven (semantic_type=
    TEMPORAL). When several exist, REUSE the existing canonical-temporal chooser
    (query.sql_builder._pick_best_temporal) so there's ONE source of truth for the
    event-time preference, not a second hardcoded name list. None if no temporal column."""
    temporal = [k.split(".", 1)[1] for k, m in (sm.get("columns", {}) or {}).items()
                if k.startswith(table + ".") and (m or {}).get("semantic_type") == "TEMPORAL"]
    if not temporal:
        return None
    if len(temporal) == 1:
        return temporal[0]
    try:
        from query.sql_builder import _pick_best_temporal
        return _pick_best_temporal(temporal, {c: {"col_name": c} for c in temporal})
    except Exception:
        return sorted(temporal)[0]


_VAGUE_RECENCY_RE = re.compile(
    r'\b(?:recently|lately|latest|newest|most\s+recent)\b', re.IGNORECASE)


def _is_vague_recency_only(raw_expressions):
    """True when every temporal span the L1 parser matched is a bare vague-recency
    word (latest/newest/most recent/recently/lately) — i.e. the derived 30-day
    BETWEEN window is a heuristic guess, not something the user explicitly asked
    for. Used to prefer ORDER BY + LIMIT (an explicit top-N ranking request like
    'latest 10 ledger entries') over a hard date-range filter that would silently
    exclude rows outside an arbitrary 30-day window and ignore the requested N."""
    return bool(raw_expressions) and all(_VAGUE_RECENCY_RE.search(e) for e in raw_expressions)


def _resolve_rank_metric_column(table, sm):
    """The anchor's single unambiguous measure column, if the schema names exactly
    one (veda_semantic_model.json tables[...].candidate_measure_columns) — lets a
    'top N'/'highest N' style ranking pick an ORDER BY column without guessing
    among several measures."""
    candidates = (sm.get("tables", {}).get(table, {}) or {}).get("candidate_measure_columns") or []
    return candidates[0] if len(candidates) == 1 else None


def _rank_sort_column(rank, table, sm, tcol):
    """The single column (if any) `_rank_order_limit_sql` below will actually
    ORDER BY for this ranking request — extracted as its own function so
    callers that need to know (recommended_projection's must_include, so a
    "latest 10 X" result is never sorted by a column it doesn't also show)
    don't duplicate this basis/temporal/metric branching. None means no
    ranking was requested (plain LIMIT, no ORDER BY)."""
    if rank.basis == "temporal" and tcol:
        return tcol
    if rank.basis == "metric":
        return _resolve_rank_metric_column(table, sm)
    return None


def _rank_order_limit_sql(rank, table, sm, tcol, alias=None):
    """Shared ' ORDER BY ... LIMIT ...' tail for every hand-built single-table SQL
    branch (FK / multi-hop / value-filter / temporal-only / plain listing) — so
    'latest 10 X' / 'top 5 Y' / 'bottom 3 Z' are honored everywhere a single-table
    SELECT is constructed deterministically, not only on the LLM-generated path.
    Falls back to the historical ' LIMIT 100' when the query named no ranking, so
    an unrelated query's SQL is byte-for-byte unaffected.

    `alias`: some branches FROM the anchor under an alias (answer-entity's `a`/`t`
    join) — the ORDER BY column must be qualified there to stay unambiguous."""
    limit = rank.top_n if rank.top_n is not None else 100
    prefix = f"{alias}." if alias else ""
    sort_col = _rank_sort_column(rank, table, sm, tcol)
    if sort_col:
        direction = "ASC" if rank.direction == "asc" else "DESC"
        return f' ORDER BY {prefix}"{sort_col}" {direction} LIMIT {limit}'
    return f' LIMIT {limit}'


def _temporal_predicate(table, sm, tf):
    """Grounded BETWEEN/>=/<= predicate on the anchor's canonical temporal column, or ''
    when there's no window or no temporal column. Literals are parameterised downstream by
    validate_and_parameterize (same as every other deterministic-path literal)."""
    if not tf or not (tf.start or tf.end):
        return ""
    col = _resolve_temporal_column(table, sm)
    if not col:
        return ""
    q = f'"{col}"'
    if tf.start and tf.end:
        return f"{q} BETWEEN '{tf.start}' AND '{tf.end}'"
    if tf.start:
        return f"{q} >= '{tf.start}'"
    return f"{q} <= '{tf.end}'"


def run_query(query, sm, all_cols, return_result=False, anchor_hint=None, on_event=None):
    """Run one NL→SQL→result. Reuses the shared engine; never closes it.

    Returns an int status code (0 ok / 1 error) by default — backward-compatible.
    With return_result=True, returns a dict {status, ok, cols, rows, answer, sql, …}
    so callers (the hybrid fusion, the Tier-2 fallback) can use the executed rows and
    distinguish 'answered' from 'refused'/'clarify'/error (the int code can't).

    anchor_hint (internal, qualifier salvage): force this table as the primary anchor
    — set only by the salvage retry after a first pass refused with a dropped
    qualifier whose QSR referent lives in a table retrieval never surfaced. Every
    downstream correctness gate still judges the plan; the hint also marks the run as
    a retry so salvage can never recurse."""
    start = time.time()
    join_constraints = None
    fanout_guard = None
    _llm_sql = False          # True only when the SQL's SELECT/WHERE was LLM-written
    from veda.explain import new_trace
    from veda.execution_state import ExecutionState
    from slm._call_slm import collect_usage, usage_totals
    tr = new_trace(query)
    es = ExecutionState()
    _usage = collect_usage()
    _usage.__enter__()  # closed in _done() — the single funnel for every exit below

    _ticks: list = []   # passive (phase, message) record of every _tick() below,
                        # for build_explain()'s "timeline" — see _done()'s return.

    def _tick(phase, message):
        """Fire a live, user-facing thinking event. Static string only — no LLM/SLM
        call, no extra DB round-trip. A no-op when on_event is None or itself raises,
        so progress reporting can never fail a query."""
        _ticks.append((phase, message))
        if on_event is None:
            return
        try:
            on_event(phase, message, {})
        except Exception:
            logger.exception("_tick: on_event callback raised for phase=%s", phase)

    def _feedback(status, **ctx):
        """Build + print actionable failure guidance (why / what's needed / suggestions).
        Returns the feedback dict (or None). Never raises — falls back to silence."""
        try:
            from config import FEEDBACK_ENABLED
        except Exception:
            FEEDBACK_ENABLED = True
        if not FEEDBACK_ENABLED:
            return None
        try:
            from veda.feedback import explain_failure
            fb = explain_failure(status, sm, **ctx)
            print("\n" + fb["text"] + "\n")
            return fb
        except Exception:
            return None

    def _done(rc, status, **kw):
        if status != "answered":
            _refusal = kw.get("msg") or kw.get("error") or kw.get("missing")
            tr.set("output", refusal=_refusal)
            es.refusal_reason = _refusal
        else:
            _tick("output", "Done — here's your answer")
        _calls = _usage.calls()
        _totals = usage_totals(_calls)
        tr.total_prompt_tokens = _totals["prompt_tokens"]
        tr.total_completion_tokens = _totals["completion_tokens"]
        tr.total_tokens = _totals["total_tokens"]
        _sql_calls = [c for c in _calls if c["purpose"] in ("sql_single_table", "sql_join")]
        if _sql_calls:
            tr.set("output", prompt_tokens=_sql_calls[-1]["prompt_tokens"],
                   completion_tokens=_sql_calls[-1]["completion_tokens"],
                   sql_model=_sql_calls[-1]["model"])
        _nl_calls = [c for c in _calls if c["purpose"] in ("nl_answer", "insight_engine")]
        if _nl_calls:
            tr.set("nl_summary",
                   summary_tokens=_nl_calls[-1]["prompt_tokens"] + _nl_calls[-1]["completion_tokens"],
                   summary_model=_nl_calls[-1]["model"])
        _usage.__exit__(None, None, None)
        tr.finish(status)
        # Tier2 continuation context (Tier1→Tier2 propagation) — deliberately NOT the
        # full trace (that stays below, for debugging); just what Tier2 needs to avoid
        # recomputing temporal parsing / query understanding / retrieval / primary table.
        es.sql_planning = dict(tr.sections.get("sql_planning", {}))
        if return_result:
            explain = None
            if status == "answered":
                _confidence = None
                try:
                    from query.result_explainer import synthesize_confidence
                    _anchor_conf = tr.sections.get("anchor_selection", {}).get("confidence")
                    _join_conf = tr.sections.get("join_planning", {}).get("confidence")
                    _conf_inputs = {k: v for k, v in
                                   (("anchor", _anchor_conf), ("join", _join_conf)) if v is not None}
                    _confidence = synthesize_confidence(_conf_inputs)
                except Exception:
                    logger.exception("synthesize_confidence failed — result confidence omitted")
                try:
                    from veda.business_explain import build_explain
                    explain = build_explain(
                        sql=kw.get("sql") or "", table=kw.get("table") or "", sm=sm,
                        checks=tr.sections.get("validation", {}).get("checks", []),
                        visualization=kw.get("visualization"),
                        params=params,
                        timeline=_ticks,
                        confidence=_confidence,
                    )
                except Exception:
                    logger.exception("business_explain failed — end-user explainability omitted")
            elif kw.get("feedback"):
                # Refusal path: same structured-explainability CONTRACT as a
                # success, built from the feedback _feedback() already computed
                # above (why/what_needed/suggestions) — not from SQL, which
                # doesn't exist for a refusal. None when no feedback dict is
                # available (invalid/exec_error's _done() calls don't build
                # one — see those call sites), same as before this change.
                try:
                    from veda.business_explain import build_refusal_explain
                    explain = build_refusal_explain(status, kw.get("feedback"))
                except Exception:
                    logger.exception("build_refusal_explain failed — refusal explainability omitted")
            # business_intent (advisory, Phase-1 business-aware output): the
            # deterministic one-sentence business reading of the EXECUTED SQL —
            # build_explain's understanding.summary, surfaced as a convenience
            # top-level key. Derived from the SQL that actually ran (source of
            # truth), never from an LLM's own claim about what it meant to do.
            if status == "answered" and explain:
                kw.setdefault("business_intent",
                              (explain.get("understanding") or {}).get("summary"))
            return {"status": status, "ok": (status == "answered"),
                    "trace": tr.to_dict(), "explain": explain,
                    "usage": {"prompt_tokens": tr.total_prompt_tokens,
                              "completion_tokens": tr.total_completion_tokens,
                              "total_tokens": tr.total_tokens},
                    "latency_ms": tr.total_ms,
                    "context": es, **kw}
        return rc

    def _rec_plan(p):
        tr.set("join_planning", confidence=p.get("confidence"),
               max_fanout=p.get("max_fanout"),
               join_path=[f"{e['source_table']}.{e['source_column']}→"
                          f"{e['target_table']}.{e['target_column']}"
                          for e in p.get("join_path", [])],
               unreachable=p.get("unreachable") or [],
               ambiguous=[a.get("target") for a in p.get("ambiguous", [])])
        for w in p.get("why", []):
            tr.note("join_planning", w)

    # Which column a "top N"/"latest N" ranking request was actually ordered by —
    # set only on the single-table path below; stays None (and the NL-answer SLM
    # gets no ranking hint) for every other route, unchanged from before.
    _rank_column_for_nl = None

    from query.temporal_parser import run_temporal_parser
    _tp_result = run_temporal_parser(query)
    es.temporal_result = _tp_result
    tf = _tp_result.temporal_filter
    if tf and (tf.start or tf.end):
        print(f"  [L1] Temporal     {tf.start}  →  {tf.end}")
    else:
        print("  [L1] Temporal     (no date range)")

    # Intent comes from the grammar signals that actually exist (existence_mode /
    # aggregate_mode / superlative_mode below). The rule-based
    # query_engine.IntentDetector present in this tree is deliberately NOT wired in:
    # its keyword classes overlap the grammar planners, and flipping intent to
    # MULTI_TABLE/AGGREGATE here re-opens the multi-table planning latency that
    # SUPERLATIVE_JOIN_ROUTING deliberately gates off (see config.py).
    intent = "SIMPLE"
    print(f"  [L4] Intent       {intent} (grammar-derived below)")

    # Existence queries (with/without/how-many-have) are deterministic + fast, and the
    # embedding cache CAN'T tell "with" from "without" (near-identical vectors, opposite
    # SQL) — so never cache or serve them from the verified-query cache.
    is_existence = existence_mode(query) is not None
    if is_existence:
        print(f"  [L4a] Existence    semi/anti-join operator detected → {existence_mode(query)}")

    from veda.planning import (aggregate_mode as _agg_mode, grouped_mode as _grp_mode,
                               ratio_mode as _rat_mode, superlative_mode as _sup_mode)
    _agg, _sup, _grp, _rat = (_agg_mode(query), _sup_mode(query), _grp_mode(query),
                              _rat_mode(query))
    if _sup:
        # Routing a superlative into join planning is gated: until the grain planner
        # can actually CONSUME a superlative (group-by dim + measure), the extra
        # multi-table planning is pure latency on wide schemas (measured: 4 suite
        # queries pushed past the 120s budget). The trace still records the
        # superlative either way.
        try:
            from config import SUPERLATIVE_JOIN_ROUTING
        except Exception:
            SUPERLATIVE_JOIN_ROUTING = False
        if SUPERLATIVE_JOIN_ROUTING:
            intent = "AGGREGATE"
        print(f"  [L4] Intent       {intent} (superlative: {_sup['term']} → {_sup['superlative']})")
    _qu = dict(query=query, intent=intent,
               temporal=({"start": tf.start, "end": tf.end}
                         if tf and (tf.start or tf.end) else None),
               existence=existence_mode(query), aggregation=_agg, superlative=_sup,
               grouped=_grp, ratio=_rat)
    tr.set("query_understanding", **_qu)
    es.query_understanding = _qu

    # Deterministic fast path: count / aggregate / dimension-list questions resolve
    # straight from the compiled registries — no retrieval, no planner, no LLM (and
    # they never touch get_engine(), so they're fast even on a cold process). Existence
    # already has its own deterministic path. Conservative match → falls through on miss.
    from config import FAST_PATH_ENABLED
    from query.fast_path import try_fast_path, log_route
    fp = None
    if FAST_PATH_ENABLED and not is_existence:
        try:
            fp = try_fast_path(query, tf)
        except Exception as _fpe:
            print(f"  [FastPath] warning: {_fpe} — falling through")
            fp = None

    # Deterministic superlative-by-dimension planner (QSR-backed, Phase B): grouped
    # ranked aggregation straight from resolution artifacts — no retrieval, no LLM.
    # May also return a grounded clarify (ambiguous dimension/measure listed).
    if fp is None and not is_existence and _sup:
        try:
            from config import SUPERLATIVE_PLAN_ENABLED
        except Exception:
            SUPERLATIVE_PLAN_ENABLED = False
        if SUPERLATIVE_PLAN_ENABLED:
            try:
                from query.superlative_plan import try_superlative_plan
                _sp = try_superlative_plan(query, sm)
            except Exception as _spe:
                print(f"  [SupPlan] warning: {_spe} — falling through")
                _sp = None
            if isinstance(_sp, tuple) and _sp and _sp[0] == "clarify":
                fb = _feedback("clarify", msg=_sp[1])
                log_route("clarify", query, (time.time() - start) * 1000)
                return _done(0, "clarify", msg=_sp[1], feedback=fb)
            if _sp is not None:
                fp = _sp

    # Deterministic grouped-breakdown planner (same QSR machinery, non-ranked
    # sibling): "how much does each <dim> contribute" → GROUP BY dim, SUM(measure).
    # Same clarify/fall-through contract as the superlative planner above.
    if fp is None and not is_existence and _grp:
        try:
            from config import GROUPED_PLAN_ENABLED
        except Exception:
            GROUPED_PLAN_ENABLED = False
        if GROUPED_PLAN_ENABLED:
            try:
                from query.superlative_plan import try_grouped_plan
                _gp = try_grouped_plan(query, sm)
            except Exception as _gpe:
                print(f"  [GrpPlan] warning: {_gpe} — falling through")
                _gp = None
            if isinstance(_gp, tuple) and _gp and _gp[0] == "clarify":
                fb = _feedback("clarify", msg=_gp[1])
                log_route("clarify", query, (time.time() - start) * 1000)
                return _done(0, "clarify", msg=_gp[1], feedback=fb)
            if _gp is not None:
                fp = _gp

    # Deterministic ratio planner: "ratio of X to Y" → single-scan divided sums
    # on the measure-owning anchor; ungroundable side → grounded clarify with the
    # anchor's real value domain (terminal — never retried by Tier-2).
    if fp is None and not is_existence and _rat:
        try:
            from config import RATIO_PLAN_ENABLED
        except Exception:
            RATIO_PLAN_ENABLED = False
        if RATIO_PLAN_ENABLED:
            try:
                from query.ratio_plan import try_ratio_plan
                _rp = try_ratio_plan(query, sm)
            except Exception as _rpe:
                print(f"  [RatioPlan] warning: {_rpe} — falling through")
                _rp = None
            if isinstance(_rp, tuple) and _rp and _rp[0] == "clarify":
                fb = _feedback("clarify", msg=_rp[1])
                log_route("clarify", query, (time.time() - start) * 1000)
                return _done(0, "clarify", msg=_rp[1], feedback=fb)
            if _rp is not None:
                fp = _rp

    # FAST-PATH EVIDENCE GUARD: the fast path bypasses anchor vetting, so an answer
    # built entirely on tables the query gives NO typed evidence for (no entity
    # word, no value, no closure) is the wrong-pick signature — "annual sum of
    # financial records…" answered from users_userpreference via the
    # financial_year_id accident. DEMOTE, don't refuse: fall through to the full
    # pipeline, which has anchor vetting and its own gates. (Measured on the golden
    # baseline: refusing here flipped good answers on descriptor words; demotion
    # only costs latency on the rare zero-evidence picks.)
    if fp is not None and not isinstance(fp, dict):
        try:
            from config import FASTPATH_EVIDENCE_GUARD, QSR_FP_EVIDENCE_FLOOR
            if FASTPATH_EVIDENCE_GUARD and fp.tables:
                from query.resolution import typed_anchor_evidence
                _ev, _ = typed_anchor_evidence(query, sm)
                if not any(_ev.get(t, 0.0) >= QSR_FP_EVIDENCE_FLOOR for t in fp.tables):
                    print(f"  [FastPath] demoted: no typed evidence for "
                          f"{sorted(fp.tables)[:3]} — full pipeline")
                    tr.note("schema_linking",
                            f"fast-path pick {sorted(fp.tables)[:3]} demoted (zero typed evidence)")
                    fp = None
        except Exception:
            pass

    cached_sql, sim = (None, 0.0) if (is_existence or fp) else verified_cache_lookup(query)
    # Same evidence guard for the CACHED lane — the fourth answer-producing lane,
    # which replays SQL verified under OLDER code: a cached answer whose tables get
    # zero typed evidence from the query is a stale wrong pick → recompute.
    if cached_sql:
        try:
            from config import FASTPATH_EVIDENCE_GUARD, QSR_FP_EVIDENCE_FLOOR
            if FASTPATH_EVIDENCE_GUARD:
                from query.resolution import typed_anchor_evidence
                _ct = set(re.findall(r'(?:FROM|JOIN)\s+"?([A-Za-z_][A-Za-z0-9_]*)',
                                     cached_sql))
                _ev, _ = typed_anchor_evidence(query, sm)
                if _ct and not any(_ev.get(t, 0.0) >= QSR_FP_EVIDENCE_FLOOR for t in _ct):
                    print(f"  [cache] demoted: no typed evidence for cached tables "
                          f"{sorted(_ct)[:3]} — recompute")
                    tr.note("schema_linking", "verified-cache hit demoted (zero typed evidence)")
                    cached_sql = None
        except Exception:
            pass
    if cached_sql:
        # QUALIFIER re-check (distinct from the table-level evidence guard above):
        # the evidence guard only asks "is THIS query plausibly about the cached
        # SQL's table(s)" — it can't catch a same-table cache entry whose WHERE
        # clause answers a DIFFERENT question (found in production: a cached
        # "properties in the UAE" answer replayed verbatim for "properties priced
        # above 10,000" — same table, unrelated filter, similarity ≥0.85 anyway).
        # Reuses the SAME gate the main pipeline already applies to freshly-built
        # SQL (below, ~line 1090) — a cache hit must clear the identical bar a
        # fresh answer would, not a lesser one just because it was pre-verified
        # once under possibly-older code.
        try:
            ok_cache_q, missing_cache_q = qualifier_completeness(query, cached_sql, sm)
            if not ok_cache_q:
                print(f"  [cache] demoted: cached SQL drops qualifier {missing_cache_q!r} "
                      f"for THIS query — recompute")
                tr.note("schema_linking",
                        f"verified-cache hit demoted (dropped qualifier {missing_cache_q!r})")
                cached_sql = None
        except Exception:
            pass
    if fp:
        print(f"  [FastPath] {fp.route}  ({'; '.join(fp.why)})  — no retrieval / no LLM")
        sql, primary, from_cache = fp.sql, fp.primary, False
        allowed_tables, allowed_columns = set(fp.tables), list(fp.columns)
    elif cached_sql:
        print(f"  [cache] verified-query hit (sim={sim:.2f}) — skipping retrieval + SLM")
        sql, from_cache = cached_sql, True
        import sqlglot
        from sqlglot import exp
        try:
            ct = sqlglot.parse_one(sql, read="postgres")
            allowed_tables = {t.name for t in ct.find_all(exp.Table) if t.name}
        except Exception:
            allowed_tables = set()
        # The real table name, derived from the cached SQL text itself — NOT the
        # literal string "(cached)" this used to be (a display-only leftover that
        # ended up as engine_result["table"], then as the QueryFrame's "entity"
        # via harvest_frame(), poisoning memory/topic-switch detection on every
        # cache-hit turn). Multi-table cached query: pick deterministically
        # (first alphabetically) rather than guess — never crashes downstream,
        # which only special-cases an empty/unknown primary already.
        primary = (next(iter(allowed_tables)) if len(allowed_tables) == 1
                   else (sorted(allowed_tables)[0] if allowed_tables else ""))
        allowed_columns = [k.split(".", 1)[1] for k in all_cols
                           if k.split(".", 1)[0] in allowed_tables]
    else:
        from config import QUERY_ENHANCEMENT_ENABLED
        enh = None
        if QUERY_ENHANCEMENT_ENABLED:
            try:
                from veda.query_enhancement import enhance_query
                enh = enhance_query(query, sm)
            except Exception:
                enh = None
        _search = enh.search_query if enh else query
        tr.set("query_understanding", enhancement=(enh.to_dict() if enh else None))
        if enh and _search != query:
            print(f"  [L2+] Enhance      +{len(enh.search_terms) + len(enh.expanded_aliases)} "
                  f"search terms  ({'; '.join(enh.enhancement_trace[:3])})")
        print("  [L2] Retrieval     5-signal (BGE-M3 + BM25 + FK subgraph/path + value) → RRF")
        try:
            from config import RETRIEVAL_CACHE_ENABLED as _RC
        except Exception:
            _RC = False
        # Pass THIS (source, tenant)'s semantic model so the engine for this scope is built
        # from the right source's BM25/signals (P5 multi-source); Signal-1 store is source-scoped.
        results = get_engine(sm).retrieve(query=_search, intent=intent, top_k=15, use_cache=_RC)

        # ── Unified-graph recall booster (Phase 4): ADD columns the 5-signal engine may
        # have missed, via synonym/alias resolution + FK-neighbour reach. Purely additive
        # (the cross-encoder rerank below re-scores everything), flag-guarded, and fully
        # try/except'd → on ANY failure retrieval is byte-identical to before. col_id here
        # is the "table.col" string the engine already uses, so no UUID lookup is needed.
        try:
            from config import GRAPH_EXPAND_ENABLED, GRAPH_EXPAND_MAX
        except Exception:
            GRAPH_EXPAND_ENABLED, GRAPH_EXPAND_MAX = False, 12
        if GRAPH_EXPAND_ENABLED and results is not None:
            try:
                from graph.query_graph import suggest_expansions
                from retrieval.retrieval_engine_phase3 import RetrievalResult as _RR
                _have_cols = {r.col_id for r in results}
                _have_tabs = {r.table_name for r in results}
                _seeds, _added, _syn = suggest_expansions(
                    query, _have_cols, _have_tabs, max_add=GRAPH_EXPAND_MAX)
                for _name in _added:
                    _tt, _cc = _name.split(".", 1)
                    results.append(_RR(col_id=_name, column_name=_cc,
                                       table_name=_tt, final_score=0.0))
                if _added:
                    print(f"  [L2g] Graph expand  +{len(_added)} cols "
                          f"(seeds={_seeds[:4]}): {_added[:5]}")
                tr.set("graph_expansion", seeds=_seeds, synonyms=_syn, added=_added)
            except Exception as _ge:
                print(f"  [L2g] graph expand skipped: {type(_ge).__name__}: {str(_ge)[:80]}")

        # ── PRIMARY cross-encoder rerank (Step 2): the precision ranker now runs on the
        # PRIMARY path (not only Tier-2). Reorders candidates + updates final_score so anchor
        # selection ranks off reranked scores — directly tightening the near-tie RRF margins
        # that caused mis-anchoring. Generic: reranker no longer carries a hardcoded business
        # map (it uses the generated domain_synonyms). Graceful: any failure keeps RRF order.
        try:
            from config import (PRIMARY_RERANK_ENABLED, RERANKER_BATCH_SIZE,
                                 RERANK_SKIP_GAP, RERANK_MAX_CANDIDATES, RERANKER_MAX_TEXT_LEN)
        except Exception:
            PRIMARY_RERANK_ENABLED = False

        def _rrf_gap_unambiguous(_results) -> bool:
            """True when candidate #1 clearly leads #2 AND both are the same table —
            reranking would not change the anchor, so skip it (F4)."""
            if len(_results) < 2:
                return True
            s0, s1 = _results[0].final_score, _results[1].final_score
            same_table = _results[0].col_id.split(".")[0] == _results[1].col_id.split(".")[0]
            return same_table and (s0 - s1) >= RERANK_SKIP_GAP

        _rk_before = _rk_after = None   # top-5 col_ids around the rerank (trace only)
        if PRIMARY_RERANK_ENABLED and results and not _rrf_gap_unambiguous(results):
            try:
                from query.reranker import _get_reranker, _precomputed_rerank_text
                _rk = _get_reranker()
                if _rk is not None:
                    # F4: cap candidate width — the tail never wins anchor selection.
                    _head = results[:RERANK_MAX_CANDIDATES]
                    _tail = results[RERANK_MAX_CANDIDATES:]
                    # Same enriched cross-encoder text query/reranker.py's own _col_text()
                    # uses (business definition/aliases/role/etc., precomputed at ingestion,
                    # WP7) — not bare column_name+table_name. This is the SAME model as
                    # rerank_columns()/rerank_tables(); it was just seeing less context here
                    # than at that other call site. Falls back to the bare name pair when no
                    # precomputed doc exists for a column (identical fallback _col_text uses).
                    _pairs = [
                        [_search, (_precomputed_rerank_text(r.col_id, is_table=False)
                                   or f"{r.column_name} {r.table_name}")[:RERANKER_MAX_TEXT_LEN]]
                        for r in _head
                    ]
                    _enriched_n = sum(1 for r in _head
                                      if _precomputed_rerank_text(r.col_id, is_table=False) is not None)
                    print(f"  [L2b] Enriched rerank input: {_enriched_n}/{len(_head)} candidates "
                          f"used precomputed metadata, {len(_head) - _enriched_n} fell back to bare name")
                    _sc = _rk.predict(_pairs, batch_size=RERANKER_BATCH_SIZE)
                    # NOISE FLOOR: the cross-encoder's output is calibrated (sigmoid) — when
                    # its BEST pair is near zero it is affirmatively saying NO candidate is
                    # relevant to this query. Overwriting final_score then replaces the RRF
                    # consensus (BM25+embedding+graph) with pure noise that downstream anchor
                    # normalization stretches to 1.0 — mis-anchoring on garbage. Keep the RRF
                    # order instead; the floor is in the model's own output space, no schema
                    # or vocabulary assumption.
                    try:
                        from config import RERANK_NOISE_FLOOR
                    except Exception:
                        RERANK_NOISE_FLOOR = 0.0
                    _smax = max((float(s) for s in _sc), default=0.0)
                    if _smax < RERANK_NOISE_FLOOR:
                        print(f"  [L2b] Primary rerank UNINFORMATIVE (max {_smax:.5f} < "
                              f"{RERANK_NOISE_FLOOR}) — keeping RRF order")
                    else:
                        _rk_before = [r.col_id for r in results[:5]]   # pre-rerank order (trace)
                        _ranked = sorted(zip(_sc, _head), key=lambda x: float(x[0]), reverse=True)
                        for _s, _r in _ranked:
                            _r.cross_encoder_score = float(_s)   # keep the CE score visible (trace)
                            _r.final_score = float(_s)   # anchor reads final_score → now reranked
                        # SCALE GUARD (H-0): reranked head carries cross-encoder scores, the tail
                        # keeps RRF scores — incomparable, so floor the tail below the head to keep
                        # it from hijacking anchor selection. (Verified NOT the count-for-sale
                        # regression culprit; the anchor ambiguity there is pre-existing.)
                        if _ranked and _tail:
                            _floor = min(float(_s) for _s, _ in _ranked)
                            for _i, _r in enumerate(_tail):
                                _r.final_score = _floor - 1.0 - _i * 1e-6
                        results = [_r for _, _r in _ranked] + _tail
                        _rk_after = [r.col_id for r in results[:5]]   # post-rerank order (trace)
                        print(f"  [L2b] Primary rerank (cross-encoder, top {RERANK_MAX_CANDIDATES}) → top: {results[0].col_id}")
            except Exception as _rr_e:
                print(f"  [L2b] primary rerank skipped: {type(_rr_e).__name__}: {str(_rr_e)[:100]}")
        elif PRIMARY_RERANK_ENABLED and results:
            print(f"  [L2b] Primary rerank SKIPPED (unambiguous RRF gap) → top: {results[0].col_id}")

        _cand_tabs = []
        for r in results:
            _t = r.col_id.split(".")[0]
            if _t not in _cand_tabs:
                _cand_tabs.append(_t)
        _router_primary = select_primary_table(results, query, sm, trace=tr)
        primary = vet_primary(query, _router_primary, results, sm, trace=tr)
        if anchor_hint and anchor_hint in (sm.get("tables") or {}):
            # Qualifier-salvage retry: the first pass refused with a dropped qualifier
            # whose QSR referent lives in anchor_hint — retrieval/vetting never
            # surfaced it (single-table planning takes its columns from all_cols, not
            # from retrieval, so the miss doesn't matter). Overrides a clarify verdict
            # too: this run exists to test the hinted anchor against the full gates.
            if primary != anchor_hint:
                print(f"  [L3] Anchor hint   "
                      f"{(_router_primary if isinstance(primary, dict) else primary)!r}"
                      f" → {anchor_hint!r} (qualifier salvage)")
                tr.note("schema_linking", f"anchor_hint override → {anchor_hint}")
            primary = anchor_hint
        if isinstance(primary, dict):
            # single-table ambiguity gate: two sub-margin, differently-named subjects —
            # ask which grain the user means instead of silently picking one.
            _cmsg = primary.get("clarify")
            fb = _feedback("clarify", msg=_cmsg)
            log_route("clarify", query, (time.time() - start) * 1000)
            return _done(0, "clarify", msg=_cmsg, feedback=fb)
        if primary != _router_primary:
            print(f"  [L3] Grain vet     router primary {_router_primary!r} → {primary!r} "
                  f"(word-order / grain-hint)")
        from config import PRIMARY_TABLE_SEED_BOOST
        es.primary_table = primary
        es.candidate_tables = list(_cand_tabs)
        # Plain {table_name, col_name, score} dicts — connector-agnostic, and reused
        # as-is by select_retrieval()'s seed-candidate merge (no second DB lookup).
        # Fields from the VETTED primary table get a small score boost — this is how
        # `primary_table` actually influences Tier2 (not just an inert log flag): Tier1
        # already spent a whole retrieval+grain-vet pass deciding this table is the
        # anchor, so Tier2's reranker should start from that prior, not from zero.
        # Small and additive — never overrides the cross-encoder's own judgment.
        # Enriched with retrieval PROVENANCE (RC-5): each entry keeps the raw RRF
        # score and the cross-encoder score separately (and a `reranked` flag), plus
        # the field's semantic_type from the model — so Tier2 can tell a resolved
        # MEASURE from a DIMENSION from an IDENTIFIER and knows which score is raw vs
        # reranked, instead of receiving one flattened number. The first three keys
        # are unchanged, so every existing consumer keeps working.
        _sm_cols = (sm or {}).get("columns", {})
        es.candidate_fields = [
            {"table_name": (_t := r.col_id.split(".", 1)[0]),
             "col_name":   r.col_id.split(".", 1)[1] if "." in r.col_id else r.column_name,
             "score":      float(getattr(r, "final_score", 0.0)) + (PRIMARY_TABLE_SEED_BOOST
                                                                     if _t == primary else 0.0),
             "semantic_type": (_sm_cols.get(r.col_id, {}) or {}).get("semantic_type"),
             "rrf_score":   float(getattr(r, "rrf_score", 0.0)),
             "cross_encoder_score": (float(r.cross_encoder_score)
                                     if getattr(r, "cross_encoder_score", None) is not None
                                     else None),
             "reranked":    getattr(r, "cross_encoder_score", None) is not None}
            for r in results[:15]
        ]
        # The exact text the cross-encoder reranked against (the ENHANCED query when
        # enhancement ran, else the raw query) — recorded so Tier2, which reranks
        # against the RAW query, can tell whether its scores are comparable to these.
        es.rerank_query = _search if _rk_after is not None else None
        tr.set("retrieval", candidate_tables=_cand_tabs[:8], n_columns=len(results))
        for r in results[:15]:
            # Per-signal scores (semantic_score/sparse_score/subgraph_score/fk_path_score/
            # value_index_score) are now actually populated (retrieval_engine_phase3.py) —
            # surface them here so the trace explains WHY a candidate ranked well, not just
            # that it did. "type" used to read `semantic_type`, a field RetrievalResult
            # never had (always None) — replaced with real signal-level evidence.
            tr.cand("retrieval", "top_columns",
                    {"col": r.col_id, "score": round(getattr(r, "final_score", 0.0), 3),
                     "signals": {
                         "semantic": round(getattr(r, "semantic_score", 0.0), 3),
                         "sparse":   round(getattr(r, "sparse_score", 0.0), 3),
                         "subgraph": round(getattr(r, "subgraph_score", 0.0), 3),
                         "fk_path":  round(getattr(r, "fk_path_score", 0.0), 3),
                         "value":    round(getattr(r, "value_index_score", 0.0), 3),
                     }})
        tr.set("schema_linking", selected_table=primary,
               router_primary=_router_primary, candidate_tables=_cand_tabs[:8])
        if primary:
            _tick("schema_linking", f"Using {primary} for this")
        print(f"  [L3] Routing       {len(results)} cols across {len(_cand_tabs)} tables "
              f"({', '.join(_cand_tabs[:4])}…) → primary: {primary}")
        if not primary:
            fb = _feedback("no_table", candidates=_cand_tabs)
            log_route("no_table", query, (time.time() - start) * 1000)
            return _done(1, "no_table", feedback=fb,
                         msg="no single table confidently matched the question")
        from_cache = False

        # Multi-table: deterministic join plan (LLM never writes joins). Fires for
        # MULTI_TABLE / AGGREGATE, and for any existence query (with/without/how-many-have)
        # — negation like "without" isn't tagged MULTI_TABLE, so detect it directly.
        needs_join = intent in ("MULTI_TABLE", "AGGREGATE") or is_existence
        mt = try_multitable(query, results, sm, all_cols, tf, primary=primary) if needs_join else {"action": "fallback"}

        if mt["action"] == "clarify":
            fb = _feedback("clarify", msg=mt.get("msg"))
            log_route("clarify", query, (time.time() - start) * 1000)
            return _done(0, "clarify", msg=mt.get("msg"), feedback=fb)
        if mt["action"] == "refuse":
            fb = _feedback("refuse", msg=mt.get("msg"))
            log_route("refuse", query, (time.time() - start) * 1000)
            return _done(0, "refuse", msg=mt.get("msg"), feedback=fb)
        if mt["action"] == "existence":
            # Deterministic EXISTS / NOT EXISTS — no LLM, no fan-out, no join skeleton.
            p = mt["plan"]
            _rec_plan(p)
            tr.set("sql_planning", action="existence", anchor=mt["anchor"],
                   mode=mt["mode"], tables=sorted(mt["tables"]))
            _tick("sql_planning", "Checking which records match")
            print(f"  [L4b] Existence    {mt['mode']}  {mt['anchor']} ⟕ "
                  f"{' '.join(t for t in mt['tables'] if t != mt['anchor'])}")
            for w in p["why"]:
                print(f"        ↳ {w}")
            sql = mt["sql"]
            allowed_tables, allowed_columns = mt["tables"], mt["columns"]
            print("  [L5] SQL           deterministic (no LLM)")
        elif mt["action"] == "aggregate":
            # Deterministic pre-aggregation CTEs — no LLM, fan-out-free by construction.
            p = mt["plan"]
            _rec_plan(p)
            tr.set("sql_planning", action="aggregate", anchor=mt["anchor"],
                   measures=mt.get("metrics"), dimension=mt.get("group_col"),
                   threshold=mt.get("threshold"), top_n=mt.get("top_n"))
            _tick("sql_planning", "Calculating the numbers")
            thr = mt.get("threshold")
            print(f"  [L4c] Grain plan   {mt['anchor']} ⟕ {', '.join(mt['metrics'])}"
                  + (f"  (filter {thr}+)" if thr is not None else ""))
            for w in p["why"]:
                print(f"        ↳ {w}")
            sql = mt["sql"]
            allowed_tables, allowed_columns = mt["tables"], mt["columns"]
            print("  [L5] SQL           deterministic pre-aggregation (no LLM)")
        elif mt["action"] == "sql":
            p = mt["plan"]
            _rec_plan(p)
            tr.set("sql_planning", action="sql", tables=sorted(mt["tables"]))
            _tick("sql_planning", "Building the query")
            _llm_sql = True
            print(f"  [L4b] Join plan    {' ⋈ '.join(sorted(mt['tables']))}  "
                  f"(conf {p['confidence']}, fan-out {p['max_fanout']})")
            for w in p["why"]:
                print(f"        ↳ {w}")
            t_sql = time.time()
            sql = mt["sql"]
            allowed_tables, allowed_columns = mt["tables"], mt["columns"]
            # ON-integrity constraints: the LLM must keep these exact join keys + predicates
            pred_cols = set()
            for e in p["join_path"]:
                if e.get("requires_predicate"):
                    m = re.search(r"\.(\w+)\s*=", e["requires_predicate"])
                    if m:
                        pred_cols.add(m.group(1))
            join_constraints = {
                "key_pairs": [frozenset({e["source_column"], e["target_column"]}) for e in p["join_path"]],
                "qualified_pairs": mt.get("qualified_key_pairs") or [],
                "predicate_cols": pred_cols}
            fanout_guard = {"parent_aliases": mt.get("parent_aliases", set()),
                            "parent_only_cols": mt.get("parent_only_cols", set())}
            print(f"  [L5] SQL gen       {time.time()-t_sql:.1f}s (join skeleton fixed)")
        else:
            # Single-table path
            allowed_tables = {primary}
            allowed_columns = [c.split(".", 1)[1] for c in all_cols if c.startswith(primary + ".")]

            # "latest 10 X" / "top 5 Y" / "bottom 3 Z" — an explicit ranking request
            # the deterministic branches below must honor (ORDER BY + LIMIT N),
            # instead of every branch hardcoding "LIMIT 100" with no ordering.
            _rank = parse_ranking(query)

            # FK→label resolution (task #6): if a query VALUE grounds (EXACT) to a related
            # table reachable by an FK, build the filter THROUGH that FK deterministically
            # (subquery) — never let the LLM match a name against a foreign-key id. The
            # resolver returns None on cross-table / ambiguous / no-FK-path values, so we
            # fall through to the normal LLM path (refuse-over-guess preserved downstream).
            _fk = None
            try:
                from config import FK_VALUE_RESOLUTION_ENABLED
            except Exception:
                FK_VALUE_RESOLUTION_ENABLED = True
            if FK_VALUE_RESOLUTION_ENABLED:
                try:
                    from query.value_resolver import resolve_value_filter, column_values_lookup
                    from veda.runtime import get_graph
                    from veda.runtime import _pg as _pgc
                    _qtoks = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2]
                    # anchor's own column names: a token naming one ("email"→user.email) is a
                    # column to project, not a cross-table value filter — pass so it's skipped.
                    _anchor_cols = {c.split(".", 1)[1] for c in sm.get("columns", {})
                                    if c.split(".", 1)[0] == primary}
                    _fk = resolve_value_filter(primary, _qtoks, get_graph(),
                                               column_values_lookup(_pgc), anchor_cols=_anchor_cols)
                except Exception:
                    _fk = None

            # Multi-hop FK resolution (OFF by default): only when 1-hop found nothing, try a
            # junction-membership path (e.g. tags on a document via document_tags). Fires
            # ONLY for a single unambiguous path; multiple paths (RBAC direct+role) or shared
            # dimensions → None → falls to the LLM. Never guesses/unions.
            _mh = None
            try:
                from config import MULTIHOP_FK_RESOLUTION_ENABLED
            except Exception:
                MULTIHOP_FK_RESOLUTION_ENABLED = False
            if _fk is None and MULTIHOP_FK_RESOLUTION_ENABLED:
                try:
                    from query.fk_path_resolver import resolve_fk_path
                    from veda.runtime import get_graph as _gg_mh
                    from veda.runtime import _pg as _pgc_mh
                    from query.value_resolver import column_values_lookup as _cvl
                    from retrieval.query_enrichment import _singularize as _sg_mh
                    _qtoks_mh = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2]
                    # Anchor attribute tokens: a query word naming one ("state"→workflow_state)
                    # is projection, not a cross-table value filter — never fabricate a join for it.
                    _anchor_col_toks_mh = {_sg_mh(tok) for c in all_cols
                                           if c.split(".", 1)[0] == primary
                                           for tok in c.split(".", 1)[1].split("_") if len(tok) > 2}
                    _mh = resolve_fk_path(primary, _qtoks_mh, _gg_mh(), _cvl(_pgc_mh),
                                          anchor_col_toks=_anchor_col_toks_mh)
                    if _mh:
                        print(f"  [L4d] multi-hop FK  {' → '.join(_mh['path'])}  — deterministic, no LLM")
                except Exception:
                    _mh = None

            # Value-vs-Column Arbitration (runs before SQL generation): classify query
            # spans against the sampled column_values store. Business adjectives that
            # match a categorical value ("critical", "open") are grounded as VALUE
            # filters on the anchor table — never as columns — and negations
            # ("unresolved" -> status != resolved) become structured filters the LLM
            # would otherwise miss. Data-driven (EXACT value match); no word lists.
            _arb_filters = []
            try:
                from config import VALUE_ARBITER_ENABLED
            except Exception:
                VALUE_ARBITER_ENABLED = False
            if VALUE_ARBITER_ENABLED:
                try:
                    from query.value_arbiter import (arbitrate, anchor_filters,
                                                     build_schema_terms)
                    # QSR typed lookup — the old column_values_typed_lookup(runtime._pg)
                    # pointed at the SOURCE DB (no column_values there) and silently
                    # returned [] for every token; the arbiter never saw a value.
                    from query.resolution import typed_value_lookup
                    _arb = arbitrate(query, typed_value_lookup(),
                                     build_schema_terms(sm))
                    _arb_filters = anchor_filters(_arb, primary)
                    if _arb.value_filters:
                        for _ln in _arb.explain().splitlines():
                            print("  [L4c] " + _ln)
                        tr.set("value_arbitration", table=primary, filters=[
                            (f["column"], f["op"], f["value"]) for f in _arb_filters])
                except Exception:
                    _arb_filters = []

            # Temporal window → grounded predicate on the anchor's canonical temporal
            # column, applied DIRECTLY in the deterministic SQL (FK / arbiter / temporal-
            # only). A date filter is never silently dropped, and we never fall back to the
            # LLM just to add a BETWEEN. Schema-metadata driven (_resolve_temporal_column).
            #
            # An explicit ranking request needing a temporal sort ("latest 10 X") also
            # needs the canonical column resolved even when L1 found no date RANGE at
            # all (e.g. "last 10 ledger entries" — "last" without a time unit sets no
            # temporal_filter), so ORDER BY has something to sort by.
            _want_rank_order = _rank.top_n is not None and _rank.basis == "temporal"
            _tcol = (_resolve_temporal_column(primary, sm)
                    if (tf and (tf.start or tf.end)) or _want_rank_order else None)
            # "latest 10 X" ALSO makes L1 match the vague-recency word and derive a
            # synthetic last-30-days BETWEEN window (query/temporal_parser.py) — but
            # "the latest 10" means ORDER BY ... DESC LIMIT 10, not "some rows from an
            # arbitrary 30-day window, silently dropping the requested count". When the
            # ranking's own count is present and the ONLY temporal signal was that bare
            # vague-recency wording (never an explicit range like "last month"/"since
            # January"), skip the synthetic window and let ORDER BY + LIMIT do the job.
            _skip_vague_window = (_want_rank_order
                                  and _is_vague_recency_only(_tp_result.raw_expressions))
            _tpred = (_temporal_predicate(primary, sm, tf)
                     if _tcol and not _skip_vague_window else "")
            _rank_tail = _rank_order_limit_sql(_rank, primary, sm, _tcol)
            _rank_tail_a = _rank_order_limit_sql(_rank, primary, sm, _tcol, alias="a")
            # Whatever column the tail above actually ORDER BY's on (if any) must
            # always be part of what's shown — a "latest 10 X" result sorted by a
            # date the recommendation happened not to pick would be confusing.
            # Computed once, reused as recommended_projection's must_include below.
            _rank_sort_col = _rank_sort_column(_rank, primary, sm, _tcol)
            # Postgres requires SELECT DISTINCT's ORDER BY expressions to appear in the
            # select list — the WHO/distinct-name branch below projects only the display
            # column, so it can honor an explicit count but not an ORDER BY on a column
            # (the temporal/metric column) that isn't part of that projection.
            _limit_only_tail = f' LIMIT {_rank.top_n if _rank.top_n is not None else 100}'
            # Which column the ranking actually sorted by — passed to the L7b NL
            # summarizer (query/result_explainer.py) so it narrates the right field
            # (e.g. "amount") instead of guessing (e.g. an id column) for "top N"/
            # "latest N" style questions.
            if _rank.basis == "temporal" and _tcol:
                _rank_column_for_nl = _tcol
            elif _rank.basis == "metric":
                _rank_column_for_nl = _resolve_rank_metric_column(primary, sm)

            # Answer-Entity Discovery (OFF by default): a WHO question projects the
            # person's display column reached over a FK, not the raw id. Reuses
            # concept_graph["PERSON"] + the FK graph + _resolve_display_column. Deterministic
            # JOIN. anchor value filters (e.g. "high priority") apply, alias-prefixed.
            _ans = None
            try:
                from config import ANSWER_ENTITY_DISCOVERY_ENABLED
            except Exception:
                ANSWER_ENTITY_DISCOVERY_ENABLED = False
            if ANSWER_ENTITY_DISCOVERY_ENABLED:
                try:
                    from query.answer_entity import find_answer_entity
                    from veda.runtime import get_graph as _gg
                    _ans = find_answer_entity(query, primary, _gg(), sm)
                    if _ans:
                        print(f"  [L4e] answer-entity  {_ans['reason']}")
                        tr.set("sql_planning", action="answer_entity_detect",
                               table=primary, fk_col=_ans["fk_col"],
                               target=_ans["target_table"], display=_ans["display_col"])
                except Exception:
                    _ans = None

            if _ans and not _tpred:
                # Project the person over the FK (display name, not the raw id). Defer to the
                # normal path when a temporal window is present so the date filter isn't dropped.
                from query.value_arbiter import where_clause as _arb_where
                _disp, _tt = _ans["display_col"], _ans["target_table"]
                _fkc, _tpk = _ans["fk_col"], _ans["target_pk"]
                _w = _arb_where(_arb_filters, alias="a") if _arb_filters else ""
                if _ans.get("mode") == "projection":
                    # "incidents and their handler" → show the anchor rows + the handler NAME.
                    # LEFT JOIN so anchor rows with no related person still appear.
                    _rl = _ans.get("rel_label", _disp)
                    _anchor_cols = [c.split(".", 1)[1] for c in all_cols
                                    if c.split(".", 1)[0] == primary]
                    # Business-facing anchor projection instead of `a.*` (every anchor
                    # column) — same recommended_projection() every other deterministic
                    # branch uses; allowed_columns below is UNCHANGED (still every
                    # anchor column, for validation) so this only narrows what's shown.
                    _ans_proj_cols = recommended_projection(primary, _anchor_cols, results,
                                                            sm, query)
                    _ans_proj = ", ".join(f'a."{c}"' for c in _ans_proj_cols) or "a.*"
                    sql = (f'SELECT {_ans_proj}, t."{_disp}" AS "{_rl}" '
                           f'FROM "{primary}" a LEFT JOIN "{_tt}" t ON a."{_fkc}" = t."{_tpk}"'
                           + (f' WHERE {_w}' if _w else '') + _rank_tail_a)
                    allowed_columns = (_anchor_cols + [_disp, _fkc, _tpk]
                                       + [f["column"] for f in _arb_filters])
                    _proj_desc = f"{_ans_proj} + {_tt}.{_disp} AS {_rl}"
                else:
                    # WHO → just the person's distinct name.
                    sql = (f'SELECT DISTINCT t."{_disp}" '
                           f'FROM "{primary}" a JOIN "{_tt}" t ON a."{_fkc}" = t."{_tpk}"'
                           + (f' WHERE {_w}' if _w else '') + _limit_only_tail)
                    allowed_columns = ([_disp, _fkc, _tpk]
                                       + [f["column"] for f in _arb_filters])
                    _proj_desc = _disp
                allowed_tables = {primary, _tt}
                _llm_sql = False                 # deterministic — skip IR-equivalence
                tr.set("sql_planning", action="answer_entity", table=primary,
                       target=_tt, project=_proj_desc, via=_fkc,
                       filters=[(f["column"], f["op"], f["value"]) for f in _arb_filters])
                _tick("sql_planning", "Looking up who's involved")
                print(f"  [L4e] answer-entity  {_ans.get('mode','who')}: JOIN {_tt} → {_proj_desc}"
                      f"{(' + ' + str(len(_arb_filters)) + ' filter(s)') if _arb_filters else ''}"
                      "  — deterministic, no LLM")
            elif _fk and _fk.get("kind") == "subquery":
                # OR across exactly the exact-grounded columns (no preference, no LIKE).
                _or = " OR ".join(
                    f"""lower("{c}"::text) = lower('{str(v).replace("'", "''")}')"""
                    for c, v in _fk["pairs"])
                # Business-facing SELECT list — NOT the validation allow-list.
                # allowed_columns (extended a few lines below with WHERE/JOIN-only
                # helper columns like _tcol) stays exactly what it was: the AST
                # firewall's allow-list. This is a SEPARATE, smaller list for what
                # actually gets projected — composed from metadata VEDA already
                # computed at ingestion + this query's own retrieval relevance
                # (veda/routing.py::recommended_projection — never re-ranked here).
                _proj_cols = recommended_projection(primary, allowed_columns, results, sm, query,
                                                    must_include=[_rank_sort_col] if _rank_sort_col else None)
                _proj = ", ".join(f'"{c}"' for c in _proj_cols) or "*"
                _wparts = [f'"{_fk["anchor_col"]}" IN '
                           f'(SELECT "{_fk["target_col"]}" FROM "{_fk["target"]}" WHERE {_or})']
                if _tpred:
                    _wparts.append(_tpred)        # FK + temporal stays deterministic
                sql = (f'SELECT {_proj} FROM "{primary}" WHERE '
                       + " AND ".join(_wparts) + _rank_tail)
                allowed_tables = {primary, _fk["target"]}
                allowed_columns = (allowed_columns + [c for c, _ in _fk["pairs"]]
                                   + [_fk["target_col"], _fk["anchor_col"]]
                                   + ([_tcol] if _tcol else []))
                _llm_sql = False                 # deterministic — skip IR-equivalence
                tr.set("sql_planning", action="fk_value_resolution", table=primary,
                       target=_fk["target"], via=_fk["anchor_col"],
                       filters=[c for c, _ in _fk["pairs"]], temporal=_tcol)
                _tick("sql_planning", "Matching that to the right record")
                print(f"  [L4d] FK value     {primary}.{_fk['anchor_col']} → "
                      f"{_fk['target']}.({', '.join(c for c, _ in _fk['pairs'])}) "
                      f"= '{_fk['pairs'][0][1]}'"
                      + (f" + {_tcol} window" if _tcol else "") + "  — deterministic, no LLM")
            elif _mh:
                # Multi-hop junction membership: WHERE anchor_pk IN (<nested IN-subquery>).
                # Business-facing SELECT list — NOT the validation allow-list.
                # allowed_columns (extended a few lines below with WHERE/JOIN-only
                # helper columns like _tcol) stays exactly what it was: the AST
                # firewall's allow-list. This is a SEPARATE, smaller list for what
                # actually gets projected — composed from metadata VEDA already
                # computed at ingestion + this query's own retrieval relevance
                # (veda/routing.py::recommended_projection — never re-ranked here).
                _proj_cols = recommended_projection(primary, allowed_columns, results, sm, query,
                                                    must_include=[_rank_sort_col] if _rank_sort_col else None)
                _proj = ", ".join(f'"{c}"' for c in _proj_cols) or "*"
                _wparts = [f'"{_mh["anchor_col"]}" IN ({_mh["subquery"]})']
                if _tpred:
                    _wparts.append(_tpred)
                sql = (f'SELECT {_proj} FROM "{primary}" WHERE '
                       + " AND ".join(_wparts) + _rank_tail)
                allowed_tables = {primary, *_mh["path"]}
                allowed_columns = allowed_columns + [_mh["anchor_col"]] + ([_tcol] if _tcol else [])
                _llm_sql = False                 # deterministic — skip IR-equivalence
                tr.set("sql_planning", action="multihop_fk_resolution", table=primary,
                       path=_mh["path"], anchor_col=_mh["anchor_col"], temporal=_tcol)
                _tick("sql_planning", "Tracing the connection through related records")
            elif _arb_filters:
                # Deterministic single-table SQL with arbiter-grounded categorical
                # filters (= for VALUE, != for NEGATED_VALUE). All columns belong to the
                # anchor table by construction (anchor_filters filtered on `primary`).
                # where_clause compares lower(col) to value_norm, so High/high/HIGH match.
                from query.value_arbiter import where_clause as _arb_where
                # Business-facing SELECT list — NOT the validation allow-list.
                # allowed_columns (extended a few lines below with WHERE/JOIN-only
                # helper columns like _tcol) stays exactly what it was: the AST
                # firewall's allow-list. This is a SEPARATE, smaller list for what
                # actually gets projected — composed from metadata VEDA already
                # computed at ingestion + this query's own retrieval relevance
                # (veda/routing.py::recommended_projection — never re-ranked here).
                _proj_cols = recommended_projection(primary, allowed_columns, results, sm, query,
                                                    must_include=[_rank_sort_col] if _rank_sort_col else None)
                _proj = ", ".join(f'"{c}"' for c in _proj_cols) or "*"
                _wparts = [_arb_where(_arb_filters)]
                if _tpred:
                    _wparts.append(_tpred)        # value filter + temporal stays deterministic
                sql = (f'SELECT {_proj} FROM "{primary}" WHERE '
                       + " AND ".join(_wparts) + _rank_tail)
                allowed_columns = (allowed_columns + [f["column"] for f in _arb_filters]
                                   + ([_tcol] if _tcol else []))
                _llm_sql = False                 # deterministic — skip IR-equivalence
                tr.set("sql_planning", action="value_arbiter_filter", table=primary,
                       filters=[(f["column"], f["op"], f["value"]) for f in _arb_filters],
                       temporal=_tcol)
                _tick("sql_planning", "Applying your filters")
                print(f"  [L4c] value filter {primary} WHERE {' AND '.join(_wparts)}"
                      "  — deterministic, no LLM")
            elif _tpred:
                # Temporal-only deterministic projection ("users created last month") — no
                # value/FK filter, just the date window on the canonical temporal column.
                # Business-facing SELECT list — NOT the validation allow-list.
                # allowed_columns (extended a few lines below with WHERE/JOIN-only
                # helper columns like _tcol) stays exactly what it was: the AST
                # firewall's allow-list. This is a SEPARATE, smaller list for what
                # actually gets projected — composed from metadata VEDA already
                # computed at ingestion + this query's own retrieval relevance
                # (veda/routing.py::recommended_projection — never re-ranked here).
                _proj_cols = recommended_projection(primary, allowed_columns, results, sm, query,
                                                    must_include=[_rank_sort_col] if _rank_sort_col else None)
                _proj = ", ".join(f'"{c}"' for c in _proj_cols) or "*"
                sql = f'SELECT {_proj} FROM "{primary}" WHERE {_tpred}' + _rank_tail
                allowed_columns = allowed_columns + [_tcol]
                _llm_sql = False                 # deterministic — skip IR-equivalence
                tr.set("sql_planning", action="temporal_only", table=primary, temporal=_tcol)
                _tick("sql_planning", "Narrowing to that time period")
                print(f"  [L4e] Temporal     {primary} WHERE {_tcol} in window"
                      "  — deterministic, no LLM")
            elif _want_rank_order and _tcol:
                # "latest 10 X" / "last 10 X" — the only temporal signal was a bare
                # recency word (or none at all), so there's no real date-RANGE filter
                # to apply (see _skip_vague_window above) — just ORDER BY the anchor's
                # canonical temporal column + LIMIT N. Deterministic, no LLM.
                # Business-facing SELECT list — NOT the validation allow-list.
                # allowed_columns (extended a few lines below with WHERE/JOIN-only
                # helper columns like _tcol) stays exactly what it was: the AST
                # firewall's allow-list. This is a SEPARATE, smaller list for what
                # actually gets projected — composed from metadata VEDA already
                # computed at ingestion + this query's own retrieval relevance
                # (veda/routing.py::recommended_projection — never re-ranked here).
                _proj_cols = recommended_projection(primary, allowed_columns, results, sm, query,
                                                    must_include=[_rank_sort_col] if _rank_sort_col else None)
                _proj = ", ".join(f'"{c}"' for c in _proj_cols) or "*"
                sql = f'SELECT {_proj} FROM "{primary}"' + _rank_tail
                allowed_columns = allowed_columns + [_tcol]
                _llm_sql = False                 # deterministic — skip IR-equivalence
                tr.set("sql_planning", action="ranked_temporal_only", table=primary,
                       temporal=_tcol, top_n=_rank.top_n, direction=_rank.direction)
                _tick("sql_planning", "Sorting and picking the top results")
                print(f"  [L4e] Ranked       {primary} ORDER BY {_tcol} "
                      f"{_rank.direction.upper()} LIMIT {_rank.top_n or 100}"
                      "  — deterministic, no LLM")
            else:
                # ENFORCEMENT: a temporal question on an anchor with NO date column cannot
                # be answered — refuse, rather than hand the LLM an impossible "date-filter
                # a dateless table" task (it invents a column like created_at, which only
                # gets caught downstream as a confusing 'unknown column' error). The
                # deterministic temporal path (elif _tpred) already serves anchors that DO
                # have one; this guards the LLM fallback. Refuse-over-guess.
                if tf and (tf.start or tf.end) and _tcol is None:
                    _msg = (f"'{primary}' has no date/time column, so the requested time "
                            f"filter ('{tf.start or ''}'..'{tf.end or ''}') can't be applied")
                    print(f"  [L5] Temporal refuse  {_msg}")
                    fb = _feedback("refuse", msg=_msg)
                    log_route("refuse", query, (time.time() - start) * 1000)
                    return _done(0, "refuse", msg=_msg, feedback=fb)
                tr.set("sql_planning", action="single_table", table=primary)
                _tick("sql_planning", "Building the query")
                _llm_sql = True
                # in-scope column glossary (business_definition + aliases) → SQL prompt hint
                _gloss = {}
                for _c in allowed_columns:
                    _m = sm.get("columns", {}).get(f"{primary}.{_c}", {}) or {}
                    if _m.get("aliases") or _m.get("business_definition"):
                        _gloss[_c] = {"aliases": _m.get("aliases") or [],
                                      "def": _m.get("business_definition") or ""}
                # domain_synonyms-driven phrase→column directives: a query phrase that the
                # model maps to a SPECIFIC in-scope column ("last logged in" → last_logged_in)
                # tells the LLM the exact column, so it can't pick a sibling (last_login).
                # Word-boundary, len≥4 phrases only (avoid 'in'/'log' over-matching). This is
                # what enforces correct column choice WITHOUT weakening qualifier_completeness.
                _term_map, _allowed_set = [], set(allowed_columns)
                for _phrase, _cks in (sm.get("domain_synonyms", {}) or {}).items():
                    if len(_phrase) < 4 or not re.search(rf"\b{re.escape(_phrase.lower())}\b", query.lower()):
                        continue
                    for _ck in (_cks or []):
                        _pt, _, _pc = _ck.partition(".")
                        if _pt == primary and _pc in _allowed_set:
                            _term_map.append((_phrase, _pc))
                t_sql = time.time()
                _proj_cols = recommended_projection(primary, allowed_columns, results, sm, query,
                                                    must_include=[_rank_sort_col] if _rank_sort_col else None)
                sql = generate_sql(query, primary, allowed_columns, tf,
                                   col_glossary=_gloss, term_map=_term_map, time_col=_tcol,
                                   recommended_projection=_proj_cols)
                print(f"  [L5] SQL gen       {time.time()-t_sql:.1f}s"
                      + (f"  (+{len(_gloss)} col defs)" if _gloss else "")
                      + (f"  (+{len(_term_map)} term→col)" if _term_map else ""))

    # Value validation: reject fabricated filter values (e.g. 'failed_review' on a
    # column that has no such value). Resolve each column to its table via the SQL's
    # own aliases; skip our deterministic polymorphic-predicate values.
    import sqlglot as _sg
    from sqlglot import exp as _exp
    skip_values = set()
    if join_constraints:
        # Only a literal sitting ON a polymorphic-predicate column (e.g.
        # object_type = 'counterparty') is OUR deterministic value — skip just that
        # one from value-grounding. A literal on ANY other column is a user filter and
        # must still be grounded. (Previously this skipped every string literal whenever
        # a predicate existed, silently disabling value-grounding on all polymorphic
        # joins.)
        _pred_cols = {pc.lower() for pc in join_constraints.get("predicate_cols", set())}
        if _pred_cols:
            try:
                for eqp in _sg.parse_one(sql, read="postgres").find_all(_exp.EQ):
                    _c = eqp.this if isinstance(eqp.this, _exp.Column) else (
                        eqp.expression if isinstance(eqp.expression, _exp.Column) else None)
                    _l = eqp.expression if isinstance(eqp.expression, _exp.Literal) else (
                        eqp.this if isinstance(eqp.this, _exp.Literal) else None)
                    if (_c is not None and _l is not None and _l.is_string
                            and _c.name.lower() in _pred_cols):
                        skip_values.add(_l.name)
            except Exception:
                pass
    alias_to_table = {}
    try:
        for tnode in _sg.parse_one(sql, read="postgres").find_all(_exp.Table):
            if tnode.alias:
                alias_to_table[tnode.alias.lower()] = tnode.name
            alias_to_table[tnode.name.lower()] = tnode.name
    except Exception:
        pass
    _default_tbl = primary if (not from_cache and len(allowed_tables) == 1) else (
        next(iter(allowed_tables)) if len(allowed_tables) == 1 else None)
    _cols_meta_map = sm.get("columns", {})

    def _owning_table(col_name):
        # Unqualified column in a multi-table query: resolve only when exactly one
        # in-scope table owns a column of this name (unambiguous). Otherwise return
        # None so value-grounding safely skips it — never guesses a table.
        owners = [t for t in allowed_tables if f"{t}.{col_name}" in _cols_meta_map]
        return owners[0] if len(owners) == 1 else None

    def _resolve(colexp):
        if colexp.table:
            return alias_to_table.get(colexp.table.lower())
        return _default_tbl or _owning_table(colexp.name)

    _route = (fp.route if fp else "cache" if from_cache else
              "existence" if is_existence else f"full:{intent}")

    tr.set("schema_linking", selected_table=primary)
    ok_val, bad = value_grounding(sql, _resolve, sm.get("columns", {}), skip_values)
    tr.check("value_grounding", ok_val, "" if ok_val else str(bad))
    if not ok_val:
        colname, val = bad
        fb = _feedback("ungrounded", column=colname, value=val)
        log_route(_route + ".ungrounded", query, (time.time() - start) * 1000)
        return _done(0, "ungrounded", detail=bad,
                     msg=f"value '{val}' not present in {colname}", feedback=fb)
    print("  [L6a] Value check  ✓  filter literals exist in the data")

    # HARD GUARD: the correctness gates MUST validate against the ORIGINAL query,
    # never the enhanced retrieval search string. Enhancement is a recall-only sidecar.
    assert query == getattr(tr, "sections", {}).get("query_understanding", {}).get("query", query), \
        "validation must run on the original query, not the enhanced search string"

    # Unified qualifier-completeness gate (all paths): refuse if the user named a
    # qualifier the SQL doesn't account for (a dropped filter → broader answer).
    ok_q, missing = qualifier_completeness(query, sql, sm)
    tr.check("qualifier_completeness", ok_q, "" if ok_q else str(missing))
    if not ok_q:
        _sql_tabs = set(re.findall(r'(?:FROM|JOIN)\s+"?([A-Za-z_][A-Za-z0-9_]*)', sql))
        # QUALIFIER SALVAGE (generic, schema-agnostic): before refusing, ask QSR what
        # `missing` IS in this scope. Referent tables entirely OUTSIDE the SQL mean
        # the ANCHOR was wrong, not the query ("payment" refused against a
        # document-type table while a payment table exists) — retry ONCE with that
        # table forced as primary; the retried plan faces every gate above, including
        # this one (anchor_hint marks the retry, so salvage can't recurse). Runs only
        # on would-be refusals — an answered query can never regress through here.
        _refs = []
        try:
            from config import (QUALIFIER_SALVAGE_ENABLED, QUALIFIER_REANCHOR_RETRY,
                                QUALIFIER_REANCHOR_MAX_HEAD_S)
        except Exception:
            QUALIFIER_SALVAGE_ENABLED, QUALIFIER_REANCHOR_RETRY = True, True
            QUALIFIER_REANCHOR_MAX_HEAD_S = 45.0
        if QUALIFIER_SALVAGE_ENABLED and anchor_hint is None:
            try:
                from query.resolution import referent_tables
                _refs = [r for r in referent_tables(missing, sm)
                         if r["table"] not in _sql_tabs]
                # Anchor preference: a table backed by ENTITY/COLUMN-NAME evidence
                # beats a value-only home — the latter is usually a shared label
                # store ('payment' exists as a ROW in list_of_values), which is
                # filter evidence, not an anchor. Measured on the trigger query:
                # retrying against the label store refuses; retrying against the
                # entity table lands the grounded domain clarify.
                _refs.sort(key=lambda r: (
                    not any("entity" in w or "column-name" in w for w in r["why"]),
                    -r["score"], r["table"]))
            except Exception:
                _refs = []
            if (_refs and QUALIFIER_REANCHOR_RETRY
                    and (time.time() - start) <= QUALIFIER_REANCHOR_MAX_HEAD_S):
                print(f"  [L6b] Qualifier salvage  '{missing}' → {_refs[0]['table']} "
                      f"({'; '.join(_refs[0]['why'][:2])}) — re-anchored retry")
                tr.note("validation", f"qualifier salvage retry → {_refs[0]['table']}")
                try:
                    _retry = run_query(query, sm, all_cols, return_result=True,
                                       anchor_hint=_refs[0]["table"])
                except Exception:
                    _retry = None
                # Surface the retry when it ANSWERED, or when it refused with
                # something MORE grounded than the original qualifier_dropped:
                # a clarify (grounded question) or an ungrounded value on the
                # re-anchored table ("'completed' is not a value of
                # payment_status — did you mean captured/authorized/…") — the
                # value-level diagnosis on the RIGHT anchor is strictly more
                # actionable than an entity-level clarify. Any other refusal
                # falls through to the referent clarify below.
                if isinstance(_retry, dict) and (_retry.get("ok")
                        or _retry.get("status") in ("clarify", "ungrounded")):
                    try:
                        _retry.setdefault("salvage", {
                            "reanchored_to": _refs[0]["table"],
                            "dropped_qualifier": missing})
                    except Exception:
                        pass
                    log_route(_route + ".salvage_reanchor", query,
                              (time.time() - start) * 1000)
                    return _retry if return_result else 0
        # Grounded clarify upgrade: when the dropped token is NOT a real data value
        # anywhere (no direct/closed referent) and the queried table exposes FK label
        # domains, the honest answer is the domain, not a generic refusal —
        # "'completed' doesn't match any payment status; statuses here are captured /
        # authorized / cancelled." Clarify is terminal (never retried by Tier-2).
        try:
            from query.resolution import value_referents, domain_via
            from query.join_planner import load_graph
            _vr = value_referents(missing)
            if not _vr["direct"] and not _vr["closed"]:
                _qw = set(re.findall(r"[a-z]+", query.lower()))
                _doms = []
                for _e in load_graph().get("edges", []):
                    if _e.get("source_table") in _sql_tabs and _e.get("cardinality") == "N:1":
                        _d = domain_via(_e["source_table"], _e["source_column"], limit=6)
                        if len(_d) > 1:
                            # rank by how much the FK column's own words overlap the
                            # query ("payment_status_id" for "…completed payments")
                            _ov = len(_qw & {w for w in _e["source_column"].split("_")
                                             if len(w) > 2})
                            _doms.append((-_ov, _e["source_column"], _d))
                if _doms:
                    _doms.sort()
                    _, _col, _d = _doms[0]
                    _msg = (f"'{missing}' doesn't match any value in this data — "
                            f"did you mean one of {', '.join(_d[:5])} ({_col})?")
                    fb = _feedback("clarify", msg=_msg)
                    log_route(_route + ".grounded_clarify", query, (time.time() - start) * 1000)
                    return _done(0, "clarify", msg=_msg, feedback=fb)
        except Exception:
            pass
        if _refs:
            # Referent clarify: the retry didn't rescue it (or is off/over budget),
            # but QSR knows what the token IS here — tell the user, grounded in the
            # schema's own vocabulary, instead of "couldn't map 'X'".
            try:
                from query.superlative_plan import _human
                _names = list(dict.fromkeys(_human(r["table"], sm) for r in _refs[:2]))
            except Exception:
                _names = list(dict.fromkeys(r["table"] for r in _refs[:2]))
            _msg = (f"'{missing}' here refers to {' / '.join(_names)}, which this "
                    f"answer never touched — ask about {_names[0]} directly, or say "
                    f"how '{missing}' relates to your question.")
            fb = _feedback("clarify", msg=_msg)
            log_route(_route + ".salvage_clarify", query, (time.time() - start) * 1000)
            return _done(0, "clarify", msg=_msg, feedback=fb)
        fb = _feedback("qualifier_dropped", missing=missing)
        log_route(_route + ".qualifier_dropped", query, (time.time() - start) * 1000)
        return _done(0, "qualifier_dropped", missing=missing, feedback=fb)
    print("  [L6b] Qualifier    ✓  every named qualifier is represented in the SQL")

    # IR equivalence — refuse LLM SQL that introduced semantics the query never asked
    # for (extra filters/grouping/ordering/joins/DISTINCT). Deterministic builds skip it.
    from veda.ir_equivalence import validate_ir_equivalence
    _skip_pred = (join_constraints or {}).get("predicate_cols", set())
    _tcols = ({k.split(".", 1)[1] for k, m in sm.get("columns", {}).items()
               if k.split(".", 1)[0] in allowed_tables
               and (m or {}).get("semantic_type") == "TEMPORAL"}
              if (tf and (tf.start or tf.end)) else set())
    ok_ir, ir_viol = validate_ir_equivalence(
        query, sql, sm, allowed_tables=allowed_tables,
        skip_predicate_cols=_skip_pred, temporal_cols=_tcols, llm_generated=_llm_sql)
    tr.check("ir_equivalence", ok_ir, "; ".join(ir_viol))
    if not ok_ir:
        fb = _feedback("ir_mismatch", msg="; ".join(ir_viol))
        log_route(_route + ".ir_mismatch", query, (time.time() - start) * 1000)
        return _done(0, "ir_mismatch", msg="; ".join(ir_viol), feedback=fb)
    if _llm_sql:
        print("  [L6b+] IR check    ✓  no unrequested filters / joins / grouping / ordering")

    # ── Shared analytical-semantics check (advisory): the SAME generic, metadata-
    # driven invariants Tier-2/LangGraph use (veda/semantic_validation.py) — requested
    # operator preserved, group-by present, user-facing dimension not an unnecessary
    # identifier. Recorded to the trace for observability + "why" provenance; it does
    # NOT block (deterministic Tier-1 SQL is already correct by construction; the LLM
    # branch gets an early signal). graph=None here — join grounding is already
    # enforced upstream by join_constraints + ir_equivalence, so we skip the graph
    # load. Fully try/except'd: a check failure can never fail a query.
    try:
        from config import SEMANTIC_VALIDATION_ENABLED as _SV_ON
    except Exception:
        _SV_ON = False
    if _SV_ON:
        try:
            from veda.semantic_validation import validate_analytical_semantics
            _sv = validate_analytical_semantics(query, sql, sm, graph=None)
            if _sv:
                tr.set("semantic_validation", findings=_sv)
                for _f in _sv:
                    tr.note("semantic_validation",
                            f"{_f['severity']}: {_f['code']} — {_f['detail']}")
                print(f"  [L6d] Semantics    {len(_sv)} advisory finding(s): "
                      f"{', '.join(sorted({f['code'] for f in _sv}))}")
        except Exception:
            logger.debug("semantic_validation (Tier-1) skipped", exc_info=True)

    param_sql, params, err = validate_and_parameterize(sql, allowed_tables, allowed_columns,
                                                        join_constraints=join_constraints,
                                                        fanout_guard=fanout_guard)
    tr.check("ast_readonly_parameterized_fanout", not err, err or "")
    if not err and getattr(tr, "enabled", False):
        _jc = tr.sections.get("join_planning", {}).get("confidence")
        tr.set("output", sql=param_sql, params=[str(x) for x in (params or [])],
               confidence=_jc if _jc is not None else 1.0)
    if err:
        print(f"\n❌ [L6] Validation rejected the SQL: {err}\n      raw: {sql}\n")
        log_route(_route + ".invalid", query, (time.time() - start) * 1000, error=err)
        return _done(1, "invalid", error=err)
    _np = len(params) if params else 0
    print(f"  [L6c] Validate     ✓  read-only · parameterized ({_np} bound value"
          f"{'' if _np == 1 else 's'}) · AST/coverage/fan-out checked")

    print("\n  Generated SQL (parameterized):")
    print("  " + "-" * 74)
    for line in param_sql.splitlines():
        print(f"    {line}")
    if params:
        print(f"    -- params: {params}")
    print("  " + "-" * 74)

    print("  [L7] Execute       read-only connection · 30s timeout · fetch ≤20")
    cols, rows, err = execute_sql(param_sql, params)
    if err:
        print(f"\n❌ [L7] Execution error: {err}\n")
        log_route(_route + ".exec_error", query, (time.time() - start) * 1000, error=err)
        return _done(1, "exec_error", error=err)

    print(f"\n  Result: {len(rows)} rows (showing up to 20)\n")
    if cols:
        print("    " + " | ".join(str(c) for c in cols))
        print("    " + "-" * 74)
        for row in rows:
            cells = [("" if v is None else str(v))[:22] for v in row]
            print("    " + " | ".join(cells))

    # L7b — NL-back summarisation: turn the result rows into a one-line prose answer
    # (the SQL path otherwise returns only a table). Gated by NL_ANSWER_ENABLED; uses
    # the local SLM with a deterministic row-count fallback if Ollama is unavailable.
    # execute_sql returns tuples → zip to the dicts run_nl_answer expects.
    try:
        from config import (NL_ANSWER_ENABLED, NL_ANSWER_FAST_TIMEOUT_MS,
                            NL_SUMMARY_TIMEOUT_MS,
                            INSIGHT_ENGINE_ENABLED, RESULT_ANALYZER_MAX_ROWS)
    except Exception:
        NL_ANSWER_ENABLED = False
        NL_ANSWER_FAST_TIMEOUT_MS = 800
        NL_SUMMARY_TIMEOUT_MS = 10000
        INSIGHT_ENGINE_ENABLED = False
        RESULT_ANALYZER_MAX_ROWS = 200
    nl_answer_text = None
    _insight_extra = {}
    if NL_ANSWER_ENABLED and cols is not None:
        row_dicts = [dict(zip(cols, r)) for r in rows]
        # F6: don't block on the SLM prose call. Compute the safe fallback now;
        # the caller (run_query) still returns promptly even if the SLM is slow.
        from query.nl_answer import deterministic_fallback_answer
        nl_answer_text = deterministic_fallback_answer(query, list(cols), row_dicts)

        # Deterministic analytics (ALWAYS, not flag-gated): the ONE post-execution
        # analysis pass — column stats/roles, result shape, business patterns,
        # chart candidates, grounding metadata. Pure Python over ≤RESULT_ANALYZER_
        # MAX_ROWS sampled rows, zero LLM, zero new SQL. Its JSON-safe summary
        # rides the result dict (same channel as `explain`) so every downstream
        # consumer — api-tier visualization included — reads this single
        # computation instead of re-deriving its own. Enrichment only: cols/rows
        # are never modified. Only the SLM narrative (Insight Engine) below stays
        # gated behind INSIGHT_ENGINE_ENABLED.
        _ictx = None
        try:
            from veda.result_analyzer import analyze_result, analytics_summary
            # Reuse metadata already computed upstream this same run — never a
            # second reasoning pass: intent (L4 IntentDetector) and the anchor/
            # join gating confidence already surfaced into the trace.
            _anchor_conf = tr.sections.get("anchor_selection", {}).get("confidence")
            _join_conf = tr.sections.get("join_planning", {}).get("confidence")
            _conf_inputs = {k: v for k, v in
                           (("anchor", _anchor_conf), ("join", _join_conf)) if v is not None}
            _ictx = analyze_result(query, param_sql, list(cols), row_dicts, sm=sm,
                                   table=str(primary), max_rows=RESULT_ANALYZER_MAX_ROWS,
                                   query_intent=intent, confidence_inputs=_conf_inputs,
                                   params=params)
            _insight_extra["analytics"] = analytics_summary(_ictx)
        except Exception as _ae:
            print(f"  [L7b] Analytics    (skipped: {type(_ae).__name__}: {_ae})")

        # Insight Engine (additive, flag-gated): ONE combined SLM call producing
        # summary + insights + visualization suggestion + follow-ups, REPLACING
        # (not layering on top of) the plain NL-answer SLM call below — never both,
        # so there is still only one post-query SLM call either way. Consumes the
        # SAME InsightContext computed above — never a second analysis pass.
        # The top-2 detected patterns, handed to whichever summary SLM runs so it
        # WEAVES them into the prose (natural insight, not a bolted-on suffix).
        _pattern_details = ([p.detail for p in _ictx.patterns[:2]]
                            if (_ictx is not None and getattr(_ictx, "patterns", None)) else [])
        # ALL verified findings for the summarizer (it selects how many to narrate by
        # mode); the top-2 above are only the deterministic-fallback blend.
        _all_findings = ([p.detail for p in _ictx.patterns]
                         if (_ictx is not None and getattr(_ictx, "patterns", None)) else [])
        # Resolved analytical context — REUSED from this run's own understanding
        # (operation via the canonical aggregate normalizer, ranking column, temporal
        # window, explicit-id request), never re-derived in the summary layer. Lets the
        # narrator speak to the user's actual intent + preserve explicit id requests.
        _analytical_ctx = None
        try:
            from veda.planning import aggregate_operator as _agg_op
            from veda.semantic_validation import user_requested_identifier as _uri
            _analytical_ctx = {
                "intent": intent,
                "operation": _agg_op(query),
                "ranking": _rank_column_for_nl,
                "temporal": (f"{tf.start} to {tf.end}" if (tf and (tf.start or tf.end)) else None),
                "explicit_identifier": _uri(query),
            }
        except Exception:
            _analytical_ctx = None
        _slm_wove_patterns = False   # did a summary SLM already phrase the findings?

        if INSIGHT_ENGINE_ENABLED and _ictx is not None:
            try:
                from query.result_explainer import run_insight_engine
                insight = run_insight_engine(_ictx, rank_column=_rank_column_for_nl)
                if getattr(insight, "answer", None):
                    nl_answer_text = insight.answer
                    _slm_wove_patterns = True   # its prompt already grounds on the patterns_block
                    print(f"\n  [L7b] Insight      {insight.answer}")
                _insight_extra.update({          # update, not reassign — keeps "analytics"
                    "insights": insight.insights,
                    "follow_up_questions": insight.follow_up_questions,
                    "visualization": insight.visualization,
                    "confidence": insight.confidence,
                })
            except Exception as _ie:
                print(f"  [L7b] Insight Engine unavailable ({type(_ie).__name__}: {_ie}) "
                      f"— falling back to plain NL answer")
                INSIGHT_ENGINE_ENABLED = False   # this turn only — fall through below

        if not INSIGHT_ENGINE_ENABLED:
            try:
                from query.nl_answer import run_nl_answer
                nl = run_nl_answer(query, list(cols), row_dicts,
                                   # 7B instruct summary (NL_SUMMARY_MODEL) needs the
                                   # full summary budget, not the 1.5B-era fast timeout.
                                   timeout=NL_SUMMARY_TIMEOUT_MS / 1000.0,
                                   table=str(primary), semantic_model=sm,
                                   rank_column=_rank_column_for_nl,
                                   patterns=_all_findings,
                                   result_shape=getattr(_ictx, "result_shape", None),
                                   analytical_context=_analytical_ctx)
                if getattr(nl, "answer", None):
                    nl_answer_text = nl.answer
                    # run_nl_answer wove the patterns only when the SLM actually ran;
                    # on its deterministic fallback it already blended them itself.
                    _slm_wove_patterns = True
                    print(f"\n  [L7b] Answer       {nl.answer}")
            except Exception as _nle:
                print(f"  [L7b] Answer       (summarisation skipped: {type(_nle).__name__})")

        # Fold the deterministic analytics into the FINAL summary ONLY if no summary
        # SLM already wove them in (2026-07-17: was an unconditional "Analysis: …"
        # suffix — mechanical, and doubly-stated on SLM answers that already
        # discussed the patterns). Now a natural clause, and only as the last-resort
        # path when the summary SLM was disabled/absent. Top 2 only.
        if _pattern_details and not _slm_wove_patterns:
            from query.result_explainer import blend_patterns
            nl_answer_text = blend_patterns(nl_answer_text, _pattern_details)

    is_temporal = bool(tf and (tf.start or tf.end))
    # Don't cache fast-path results — they're already instant and the fast path always
    # wins ahead of the cache, so a cached copy would never be served.
    if not from_cache and fp is None and rows and not is_temporal and not is_existence:
        save_verified_query(query, sql)

    log_route(_route, query, (time.time() - start) * 1000, table=str(primary), rows=len(rows))
    tag = "cache" if from_cache else f"table={primary}"
    print("\n" + "=" * 78)
    print(f"✅ Done in {time.time()-start:.1f}s   |   intent={intent}   |   {tag}")
    print("=" * 78 + "\n")
    return _done(0, "answered", cols=list(cols) if cols else [], rows=rows,
                 answer=nl_answer_text, sql=param_sql, table=str(primary), **_insight_extra)