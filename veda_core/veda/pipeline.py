"""VEDA · The L1→L7 orchestrator (run_query)."""
import os, re, sys, time, json, logging, threading
from query.ranking_parser import parse_ranking
from veda.cache import save_verified_query, verified_cache_lookup
from veda.execution import execute_sql
from veda.generation import generate_sql
from veda.planning import existence_mode, try_multitable
from veda.routing import select_primary_table, vet_primary
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
    if rank.basis == "temporal" and tcol:
        direction = "ASC" if rank.direction == "asc" else "DESC"
        return f' ORDER BY {prefix}"{tcol}" {direction} LIMIT {limit}'
    if rank.basis == "metric":
        metric_col = _resolve_rank_metric_column(table, sm)
        if metric_col:
            direction = "ASC" if rank.direction == "asc" else "DESC"
            return f' ORDER BY {prefix}"{metric_col}" {direction} LIMIT {limit}'
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


def run_query(query, sm, all_cols, return_result=False):
    """Run one NL→SQL→result. Reuses the shared engine; never closes it.

    Returns an int status code (0 ok / 1 error) by default — backward-compatible.
    With return_result=True, returns a dict {status, ok, cols, rows, answer, sql, …}
    so callers (the hybrid fusion, the Tier-2 fallback) can use the executed rows and
    distinguish 'answered' from 'refused'/'clarify'/error (the int code can't)."""
    start = time.time()
    join_constraints = None
    fanout_guard = None
    _llm_sql = False          # True only when the SQL's SELECT/WHERE was LLM-written
    from veda.explain import new_trace
    from veda.execution_state import ExecutionState
    tr = new_trace(query)
    es = ExecutionState()

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
        tr.finish(status)
        # Tier2 continuation context (Tier1→Tier2 propagation) — deliberately NOT the
        # full trace (that stays below, for debugging); just what Tier2 needs to avoid
        # recomputing temporal parsing / query understanding / retrieval / primary table.
        es.sql_planning = dict(tr.sections.get("sql_planning", {}))
        if return_result:
            explain = None
            if status == "answered":
                try:
                    from veda.business_explain import build_explain
                    explain = build_explain(
                        sql=kw.get("sql") or "", table=kw.get("table") or "", sm=sm,
                        checks=tr.sections.get("validation", {}).get("checks", []),
                        visualization=kw.get("visualization"),
                    )
                except Exception:
                    logger.exception("business_explain failed — end-user explainability omitted")
            return {"status": status, "ok": (status == "answered"),
                    "trace": tr.to_dict(), "explain": explain, "context": es, **kw}
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

    try:
        from query_engine.intent_detector import IntentDetector
        ir = IntentDetector().detect(query)
        intent = ir.intent.value if hasattr(ir.intent, "value") else str(ir.intent)
    except Exception as e:
        logger.warning("IntentDetector unavailable (%s: %s) — defaulting intent=SIMPLE, "
                        "join/aggregate detection degraded to existence-only", type(e).__name__, e)
        intent = "SIMPLE"
    print(f"  [L4] Intent       {intent}")

    # Existence queries (with/without/how-many-have) are deterministic + fast, and the
    # embedding cache CAN'T tell "with" from "without" (near-identical vectors, opposite
    # SQL) — so never cache or serve them from the verified-query cache.
    is_existence = existence_mode(query) is not None
    if is_existence:
        print(f"  [L4a] Existence    semi/anti-join operator detected → {existence_mode(query)}")

    from veda.planning import aggregate_mode as _agg_mode
    _qu = dict(query=query, intent=intent,
               temporal=({"start": tf.start, "end": tf.end}
                         if tf and (tf.start or tf.end) else None),
               existence=existence_mode(query), aggregation=_agg_mode(query))
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

    cached_sql, sim = (None, 0.0) if (is_existence or fp) else verified_cache_lookup(query)
    if fp:
        print(f"  [FastPath] {fp.route}  ({'; '.join(fp.why)})  — no retrieval / no LLM")
        sql, primary, from_cache = fp.sql, fp.primary, False
        allowed_tables, allowed_columns = set(fp.tables), list(fp.columns)
    elif cached_sql:
        print(f"  [cache] verified-query hit (sim={sim:.2f}) — skipping retrieval + SLM")
        sql, primary, from_cache = cached_sql, "(cached)", True
        import sqlglot
        from sqlglot import exp
        try:
            ct = sqlglot.parse_one(sql, read="postgres")
            allowed_tables = {t.name for t in ct.find_all(exp.Table) if t.name}
        except Exception:
            allowed_tables = set()
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
                                 RERANK_SKIP_GAP, RERANK_MAX_CANDIDATES)
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

        if PRIMARY_RERANK_ENABLED and results and not _rrf_gap_unambiguous(results):
            try:
                from query.reranker import _get_reranker
                _rk = _get_reranker()
                if _rk is not None:
                    # F4: cap candidate width — the tail never wins anchor selection.
                    _head = results[:RERANK_MAX_CANDIDATES]
                    _tail = results[RERANK_MAX_CANDIDATES:]
                    _pairs = [[_search, f"{r.column_name} {r.table_name}"] for r in _head]
                    _sc = _rk.predict(_pairs, batch_size=RERANKER_BATCH_SIZE)
                    _ranked = sorted(zip(_sc, _head), key=lambda x: float(x[0]), reverse=True)
                    for _s, _r in _ranked:
                        _r.final_score = float(_s)   # anchor reads final_score → now reranked
                    # SCALE GUARD: the reranked head now carries CROSS-ENCODER scores (raw
                    # logits, often negative), while the tail still carries RRF scores (small
                    # positive). select_primary_table/score_anchors pick the anchor by
                    # max(final_score) across BOTH — so without this a weak tail column (RRF
                    # 0.05) would outrank a reranked head column scored -3 and hijack the
                    # anchor. Floor every tail score strictly below the lowest reranked head
                    # score (preserving the tail's own relative order) so the tail can never
                    # win anchoring — matching this block's stated intent.
                    if _ranked and _tail:
                        _floor = min(float(_s) for _s, _ in _ranked)
                        for _i, _r in enumerate(_tail):
                            _r.final_score = _floor - 1.0 - _i * 1e-6
                    results = [_r for _, _r in _ranked] + _tail
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
        _router_primary = select_primary_table(results, query, sm)
        primary = vet_primary(query, _router_primary, results, sm, trace=tr)
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
        es.candidate_fields = [
            {"table_name": (_t := r.col_id.split(".", 1)[0]),
             "col_name":   r.col_id.split(".", 1)[1] if "." in r.col_id else r.column_name,
             "score":      float(getattr(r, "final_score", 0.0)) + (PRIMARY_TABLE_SEED_BOOST
                                                                     if _t == primary else 0.0)}
            for r in results[:15]
        ]
        tr.set("retrieval", candidate_tables=_cand_tabs[:8], n_columns=len(results))
        for r in results[:15]:
            tr.cand("retrieval", "top_columns",
                    {"col": r.col_id, "score": round(getattr(r, "final_score", 0.0), 3),
                     "type": getattr(r, "semantic_type", None)})
        tr.set("schema_linking", selected_table=primary,
               router_primary=_router_primary, candidate_tables=_cand_tabs[:8])
        print(f"  [L3] Routing       {len(results)} cols across {len(_cand_tabs)} tables "
              f"({', '.join(_cand_tabs[:4])}…) → primary: {primary}")
        if not primary:
            fb = _feedback("no_table", candidates=_cand_tabs)
            log_route("no_table", query, (time.time() - start) * 1000)
            return _done(1, "no_table", feedback=fb)
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
                                                     column_values_typed_lookup,
                                                     build_schema_terms)
                    from veda.runtime import _pg as _pgc_arb
                    _arb = arbitrate(query, column_values_typed_lookup(_pgc_arb),
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
                    sql = (f'SELECT a.*, t."{_disp}" AS "{_rl}" '
                           f'FROM "{primary}" a LEFT JOIN "{_tt}" t ON a."{_fkc}" = t."{_tpk}"'
                           + (f' WHERE {_w}' if _w else '') + _rank_tail_a)
                    _anchor_cols = [c.split(".", 1)[1] for c in all_cols
                                    if c.split(".", 1)[0] == primary]
                    allowed_columns = (_anchor_cols + [_disp, _fkc, _tpk]
                                       + [f["column"] for f in _arb_filters])
                    _proj_desc = f"a.* + {_tt}.{_disp} AS {_rl}"
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
                print(f"  [L4e] answer-entity  {_ans.get('mode','who')}: JOIN {_tt} → {_proj_desc}"
                      f"{(' + ' + str(len(_arb_filters)) + ' filter(s)') if _arb_filters else ''}"
                      "  — deterministic, no LLM")
            elif _fk and _fk.get("kind") == "subquery":
                # OR across exactly the exact-grounded columns (no preference, no LIKE).
                _or = " OR ".join(
                    f"""lower("{c}"::text) = lower('{str(v).replace("'", "''")}')"""
                    for c, v in _fk["pairs"])
                _proj = ", ".join(f'"{c}"' for c in allowed_columns) or "*"
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
                print(f"  [L4d] FK value     {primary}.{_fk['anchor_col']} → "
                      f"{_fk['target']}.({', '.join(c for c, _ in _fk['pairs'])}) "
                      f"= '{_fk['pairs'][0][1]}'"
                      + (f" + {_tcol} window" if _tcol else "") + "  — deterministic, no LLM")
            elif _mh:
                # Multi-hop junction membership: WHERE anchor_pk IN (<nested IN-subquery>).
                _proj = ", ".join(f'"{c}"' for c in allowed_columns) or "*"
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
            elif _arb_filters:
                # Deterministic single-table SQL with arbiter-grounded categorical
                # filters (= for VALUE, != for NEGATED_VALUE). All columns belong to the
                # anchor table by construction (anchor_filters filtered on `primary`).
                # where_clause compares lower(col) to value_norm, so High/high/HIGH match.
                from query.value_arbiter import where_clause as _arb_where
                _proj = ", ".join(f'"{c}"' for c in allowed_columns) or "*"
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
                print(f"  [L4c] value filter {primary} WHERE {' AND '.join(_wparts)}"
                      "  — deterministic, no LLM")
            elif _tpred:
                # Temporal-only deterministic projection ("users created last month") — no
                # value/FK filter, just the date window on the canonical temporal column.
                _proj = ", ".join(f'"{c}"' for c in allowed_columns) or "*"
                sql = f'SELECT {_proj} FROM "{primary}" WHERE {_tpred}' + _rank_tail
                allowed_columns = allowed_columns + [_tcol]
                _llm_sql = False                 # deterministic — skip IR-equivalence
                tr.set("sql_planning", action="temporal_only", table=primary, temporal=_tcol)
                print(f"  [L4e] Temporal     {primary} WHERE {_tcol} in window"
                      "  — deterministic, no LLM")
            elif _want_rank_order and _tcol:
                # "latest 10 X" / "last 10 X" — the only temporal signal was a bare
                # recency word (or none at all), so there's no real date-RANGE filter
                # to apply (see _skip_vague_window above) — just ORDER BY the anchor's
                # canonical temporal column + LIMIT N. Deterministic, no LLM.
                _proj = ", ".join(f'"{c}"' for c in allowed_columns) or "*"
                sql = f'SELECT {_proj} FROM "{primary}"' + _rank_tail
                allowed_columns = allowed_columns + [_tcol]
                _llm_sql = False                 # deterministic — skip IR-equivalence
                tr.set("sql_planning", action="ranked_temporal_only", table=primary,
                       temporal=_tcol, top_n=_rank.top_n, direction=_rank.direction)
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
                sql = generate_sql(query, primary, allowed_columns, tf,
                                   col_glossary=_gloss, term_map=_term_map, time_col=_tcol)
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
                            INSIGHT_ENGINE_ENABLED, RESULT_ANALYZER_MAX_ROWS)
    except Exception:
        NL_ANSWER_ENABLED = False
        NL_ANSWER_FAST_TIMEOUT_MS = 800
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

        # Insight Engine (additive, flag-gated): ONE combined SLM call producing
        # summary + insights + visualization suggestion + follow-ups, REPLACING
        # (not layering on top of) the plain NL-answer SLM call below — never both,
        # so there is still only one post-query SLM call either way.
        if INSIGHT_ENGINE_ENABLED:
            try:
                from veda.result_analyzer import analyze_result
                from query.result_explainer import run_insight_engine
                # Reuse metadata already computed upstream this same run — never a
                # second reasoning pass: intent (L4 IntentDetector) and the anchor/
                # join gating confidence already surfaced into the trace.
                _anchor_conf = tr.sections.get("anchor_selection", {}).get("confidence")
                _join_conf = tr.sections.get("join_planning", {}).get("confidence")
                _conf_inputs = {k: v for k, v in
                               (("anchor", _anchor_conf), ("join", _join_conf)) if v is not None}
                ctx = analyze_result(query, param_sql, list(cols), row_dicts, sm=sm,
                                     table=str(primary), max_rows=RESULT_ANALYZER_MAX_ROWS,
                                     query_intent=intent, confidence_inputs=_conf_inputs)
                insight = run_insight_engine(ctx, rank_column=_rank_column_for_nl)
                if getattr(insight, "answer", None):
                    nl_answer_text = insight.answer
                    print(f"\n  [L7b] Insight      {insight.answer}")
                _insight_extra = {
                    "insights": insight.insights,
                    "follow_up_questions": insight.follow_up_questions,
                    "visualization": insight.visualization,
                    "confidence": insight.confidence,
                }
            except Exception as _ie:
                print(f"  [L7b] Insight Engine unavailable ({type(_ie).__name__}: {_ie}) "
                      f"— falling back to plain NL answer")
                INSIGHT_ENGINE_ENABLED = False   # this turn only — fall through below

        if not INSIGHT_ENGINE_ENABLED:
            try:
                from query.nl_answer import run_nl_answer
                nl = run_nl_answer(query, list(cols), row_dicts,
                                   timeout=NL_ANSWER_FAST_TIMEOUT_MS / 1000.0,
                                   table=str(primary), semantic_model=sm,
                                   rank_column=_rank_column_for_nl)
                if getattr(nl, "answer", None):
                    nl_answer_text = nl.answer
                    print(f"\n  [L7b] Answer       {nl.answer}")
            except Exception as _nle:
                print(f"  [L7b] Answer       (summarisation skipped: {type(_nle).__name__})")

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
