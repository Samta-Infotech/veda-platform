#!/usr/bin/env python3
"""
veda_hybrid.py — VEDA unified front door (the hybrid architecture).

Routes each query to the engine that is BEST at it — composing the two pipelines
that already live on this branch, without reimplementing or clobbering either:

  sql    -> veda/ DETERMINISTIC engine   (joins pinned by the planner, value
                                          grounding + AST + fan-out firewall)   [CORRECTNESS]
  rag    -> integrated RAG layer         (doc retrieval + LLM synthesis)        [BREADTH]
  hybrid -> integrated hybrid layer      (SQL signals + docs, RRF-fused)        [BREADTH]
  nosql  -> integrated NoSQL builder     (native Mongo/etc. query)              [BREADTH]

Decision: the LLM never writes SQL structure (that's the deterministic head's
job); the router + RAG/graph/NoSQL give multi-modal reach. The router classifies;
each head owns its modality. Each head works once its own stores are populated
(SQL: the deterministic semantic model; RAG/graph: doc + graph ingestion).

Usage:
    python3 veda_hybrid.py "how many incidents are escalated"      # -> deterministic SQL
    python3 veda_hybrid.py "what does the SLA policy say about RFIs" # -> RAG
"""

import os
import sys
import json
import time

# Zero-egress on-prem: force HuggingFace/transformers OFFLINE before ANY model-loading
# import. This entry point loads the BGE retrieval model (query.retrieval_engine) before
# veda/__init__ runs, so the offline flags must be set here too — otherwise transformers
# tries to reach huggingface.co and fails even though the model is cached locally.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from query.multi_result import (
    MultiResult, SubResult, STATUS_OK, STATUS_REFUSED, STATUS_ERROR,
)

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger("veda.veda_hybrid")

_SM = {}   # {(source_id, tenant): {"sm": dict, "cols": list}} — scope-keyed (P5)


def _current_ctx():
    """The ambient RequestContext, read from whichever context MODULE holds it.

    The engine is imported both as bare `context` (cwd=veda_core) and as
    `veda_core.context` (inference tier, PYTHONPATH=/app) — Python loads these as TWO
    module objects with SEPARATE thread-locals. The inference middleware sets the scope
    on `veda_core.context`, so a read of bare `context` alone silently misses it and the
    SQL head falls back to source 1 (global model). Try both so the request scope is
    seen regardless of which name set it."""
    import importlib
    for modname in ("veda_core.context", "context"):
        try:
            ctx = importlib.import_module(modname).try_current()
            if ctx is not None:
                return ctx
        except Exception:
            continue
    return None


def _sm_scope():
    """(source_id, tenant) for the semantic-model cache/Redis key. Prefers the
    ambient per-request context (set by the inference middleware from headers),
    falling back to the env pin (single-source dev / bare-metal runs).

    `source_id` here is the PRIMARY source: the SQL head's per-source model is loaded
    from the primary today; the multi-source merge for the SQL head arrives with
    federated naming (Phase 5). The cache is keyed by the full scope (`_sm_cache_key`)
    so a `{A}` request and an `{A,B}` request never share an sm entry."""
    ctx = _current_ctx()
    if ctx is not None:
        return (str(ctx.source_id), str(ctx.tenant))
    return (os.environ.get("VEDA_SM_SOURCE_ID", "1"),
            os.environ.get("VEDA_SM_TENANT", "default"))


def _sm_cache_key():
    """Scope-unique key for the inference-tier `_SM` cache: the full source SET +
    tenant (P5), so distinct scopes over the same primary don't collide."""
    ctx = _current_ctx()
    if ctx is not None:
        return (frozenset(int(s) for s in ctx.source_ids), str(ctx.tenant))
    sid, tenant = _sm_scope()
    return (frozenset({int(sid)}), tenant)


def _load_sm_from_redis(scope=None):
    """Load the Django-assembled `sm` from redis-cache (§3.6, §8a).

    The SemanticModelAssembler (running in a Django tier) rebuilds `sm` from the
    normalized substrate and publishes it to `veda:sm:{source}:{tenant}`. The
    inference tier reads it here — no Django/ORM dependency in this process. The
    key is the ambient (source, tenant) scope, so one warm worker can serve N
    ready sources (P5). Returns the sm dict, or None to fall back to the on-disk
    file (dev / cache miss).
    """
    if os.environ.get("VEDA_SM_REDIS", "").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    try:
        import redis as _redis
        url = os.environ.get("REDIS_CACHE_URL", "redis://redis-cache:6379/0")
        source_id, tenant = scope or _sm_scope()
        raw = _redis.Redis.from_url(url).get(f"veda:sm:{source_id}:{tenant}")
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _load_semantic_model():
    """Load the deterministic engine's semantic model once per (source, tenant)
    scope (for the SQL head).

    Prefers the Django-owned substrate via the assembler's Redis publication
    (§3.6); falls back to the on-disk `SEMANTIC_MODEL_FILE` when Redis is not
    configured or misses. The front-door signature is unchanged; the cache is
    scope-keyed so multiple ready sources are queryable from one warm worker (P5),
    and the rehydrate subscriber clears it on re-ingest.
    """
    scope = _sm_scope()               # primary (source, tenant) — Redis sm key for the SQL head
    cache_key = _sm_cache_key()       # full scope SET — cache identity (P5)
    entry = _SM.get(cache_key)
    if entry is None:
        sm = _load_sm_from_redis(scope)
        if sm is None:
            from config import SEMANTIC_MODEL_FILE
            with open(SEMANTIC_MODEL_FILE) as f:
                sm = json.load(f)
        entry = {"sm": sm, "cols": list(sm.get("columns", {}).keys())}
        _SM[cache_key] = entry
    return entry["sm"], entry["cols"]


import re as _re_mod

# A document is referenced (nouns) + the utterance is asking what it SAYS (verbs). Used to
# route doc-scope queries to the fast RAG/hybrid lanes instead of the SQL head + 30s LLM-IR
# fallback (which ignores the document and dumps DB rows).
_DOC_REF_RE = _re_mod.compile(
    r"\b(document|agreement|contract|policy|policies|msa|sla|clause|section|terms|"
    r"report|readme|notes?|memo|memorandum|pdf|docx?|paper|letter|manual|handbook)\b",
    _re_mod.I)
_DB_AGG_RE = _re_mod.compile(
    r"\b(how many|count|total|sum|average|avg|per |group by|number of|top \d|highest|"
    r"lowest|most|least|ranked?)\b", _re_mod.I)


def _scope_has_doc_source() -> bool:
    """True when the request scope includes a source with document chunks (cheap indexed
    lookup)."""
    ctx = _current_ctx()
    sids = [str(s) for s in (getattr(ctx, "source_ids", ()) or ())] if ctx is not None else []
    if not sids:
        return False
    try:
        from ingestion.db_abstraction import (
            get_internal_connection, release_internal_connection)
        conn = get_internal_connection()
        try:
            with conn.cursor() as cur:
                ph = ",".join(["%s"] * len(sids))
                cur.execute(f"SELECT 1 FROM graph_nodes WHERE node_type='chunk' "
                            f"AND source_id IN ({ph}) LIMIT 1", sids)
                return cur.fetchone() is not None
        finally:
            release_internal_connection(conn)
    except Exception:
        return False


def classify(query, verbose=False):
    """Return (intent, source_ids). Falls back to 'sql' if the router is off/unavailable
    — the deterministic SQL head is the safe default."""
    # Deterministic doc-intent override (fast, before the SLM router): a document-referencing
    # question over a scope that actually has doc chunks routes to the RAG lane (pure doc ask)
    # or the HYBRID lane (doc + a DB clause), never the SQL head that would ignore the doc.
    q = query or ""
    if _DOC_REF_RE.search(q) and _scope_has_doc_source():
        # Prefer the fast RAG lane (retrieve chunks + one synthesis call, ~6-8s). Only take
        # the heavier HYBRID lane (RAG ⊕ deterministic SQL head) when the utterance clearly
        # needs a DB aggregation (count/total/per-group) the documents can't supply.
        intent = "hybrid" if _DB_AGG_RE.search(q) else "rag"
        if verbose:
            print(f"  [router] doc-intent override → {intent}")
        return intent, None

    try:
        from config import QUERY_ROUTER_ENABLED
    except Exception:
        QUERY_ROUTER_ENABLED = False
    if not QUERY_ROUTER_ENABLED:
        return "sql", None
    try:
        from query.query_router import route_query
        r = route_query(query, verbose=verbose)
        return r.intent, r.source_ids
    except Exception as e:
        if verbose:
            print(f"  [router] unavailable ({type(e).__name__}: {e}) — defaulting to sql")
        return "sql", None


def _temporal(query):
    try:
        from query.temporal_parser import run_temporal_parser
        return run_temporal_parser(query).temporal_filter
    except Exception:
        return None


def _emit(on_event, phase, message, **extra):
    """Fire the optional SSE progress callback (see run_hybrid_query's on_event contract):
    ``on_event(phase, message, extra: dict)``. A no-op when on_event is None, and it never
    raises into the pipeline — progress reporting must not be able to fail a query."""
    if on_event is None:
        return
    try:
        on_event(phase, message, extra)
    except Exception:
        pass


def _maybe_federated(query, verbose=False):
    """If the request scope spans ≥2 sources, try the cross-source federated route.
    Returns a MultiResult on a federated answer/refusal, or None to use the normal path."""
    ctx = _current_ctx()
    sids = list(getattr(ctx, "source_ids", ()) or ()) if ctx is not None else []
    if len(sids) < 2:
        return None
    from slm._call_slm import collect_usage, usage_totals
    _fed_t0 = time.time()
    _fed_calls = []
    try:
        with collect_usage() as _fed_usage:
            from query.federated_route import run_federated
            payload = run_federated(query, tenant=str(getattr(ctx, "tenant", "default")),
                                    source_ids=sids, verbose=verbose)
            # MUST read calls() INSIDE the with block — collect_usage().__exit__()
            # clears the thread-local buffer on exit (it's the outermost scope
            # here), so reading it after the block always returns empty. This
            # was the actual root cause of every federated-route usage=0 report:
            # call_slm() genuinely ran and recorded real tokens (confirmed from
            # prod logs — federated_struct_plan/federated_answer calls with
            # real prompt/completion counts), but by the time this function
            # read _fed_usage.calls(), the buffer had already been reset.
            _fed_calls = _fed_usage.calls()
    except Exception as e:
        if verbose:
            print(f"  [federated] route error ({type(e).__name__}: {e}) — normal path")
        return None
    _fed_usage_totals = usage_totals(_fed_calls)
    _fed_latency_ms = round((time.time() - _fed_t0) * 1000, 2)
    logger.debug("_maybe_federated status=%s calls_captured=%d purposes=%s tokens=%s",
                payload.get("status") if payload else None, len(_fed_calls),
                [c["purpose"] for c in _fed_calls], _fed_usage_totals)
    if payload is None:
        return None                      # single-source plan → normal path
    if payload.get("status") == "ok":
        r = payload.get("result") or {}
        # Two success shapes (federated_route.py::run_federated): compose_federated()'s
        # flat single-SELECT path has "sql" directly; compose_federated_plan()'s
        # structured/free-form per-metric path (the PREFERRED one — see that
        # function's own "PREFERRED: DETERMINISTIC join-path planner" comment) has
        # no single "sql" key at all, only "plan": {group_by, metrics: [{alias, sql}]}
        # — each metric aggregated+joined independently, never one flat statement.
        # Join the per-metric SQL fragments so explain/the SQL panel show the real
        # generated queries instead of an empty string (which made build_explain()
        # invent a generic "Federateds" table name from nothing — payload.get("sql")
        # was always None on this path, never a bug in build_explain() itself).
        plan = payload.get("plan") or {}
        metric_sqls = [m.get("sql") for m in (plan.get("metrics") or []) if m.get("sql")]
        sql = payload.get("sql") or "\n\n".join(metric_sqls) or ""
        # group_table is DuckDB-qualified (src_2.public."assets_asset") — strip to the
        # bare table name for display; _business_table_name() would otherwise humanize
        # the dots/quotes verbatim into garbage.
        _group_table = plan.get("group_table") or ""
        table = _group_table.rsplit(".", 1)[-1].strip('"') if _group_table else "federated"
        # No single-source semantic model applies across a federated query, but
        # build_explain() parses the SQL text itself (sm=None degrades table/column
        # labels to humanized raw names, never crashes) — so a federated answer gets
        # real explainability (entities/filters/operations from the ACTUAL cross-source
        # SQL that ran) instead of apps/chat/services.py's generic _NO_EXPLAIN
        # placeholder, which previously fired for every federated answer even though
        # cols/rows/sql were all genuinely available.
        explain = None
        try:
            from veda.business_explain import build_explain
            # Plan-path payloads carry a per-metric `plan` (group key + one aggregate
            # SELECT per metric/source), NOT a single `sql` — so build_explain on an
            # empty string produced the generic "List records" / "Federateds". Fall back
            # to a representative metric SELECT so explainability shows the REAL group-by
            # + aggregate + source table(s) the federated answer computed. The flat-SQL
            # path already carries `sql`, so it is preferred when present.
            _fed_sql = payload.get("sql")
            if not _fed_sql and isinstance(payload.get("plan"), dict):
                _mets = payload["plan"].get("metrics") or []
                _fed_sql = next((m.get("sql") for m in _mets if m.get("sql")), None)
            explain = build_explain(sql=_fed_sql or "", table="federated", sm=None)
        except Exception as e:
            if verbose:
                print(f"  [federated] business_explain failed ({type(e).__name__}: {e}) — explainability omitted")
        result = {"ok": True, "status": "answered", "route": "federated",
                  "sql": sql, "cols": r.get("columns"), "rows": r.get("rows"),
                  "table": table, "answer": payload.get("answer"), "explain": explain,
                  "provenance": payload.get("provenance"), "sources": payload.get("sources"),
                  "usage": _fed_usage_totals, "latency_ms": _fed_latency_ms}
        # Deterministic analytics for the federated result too — degraded mode
        # (no single-table semantic model, so grounding fields stay empty), but
        # column stats, result shape, chart candidates and detected patterns all
        # work off (cols, rows) alone. Same "Analysis:" fold-in as Tier-1/Tier-2.
        try:
            from veda.result_analyzer import analyze_result, analytics_summary
            _fc, _fr = result.get("cols") or [], result.get("rows") or []
            if _fc and _fr:
                _frd = [row if isinstance(row, dict) else dict(zip(_fc, row)) for row in _fr]
                _fctx = analyze_result(query, sql, list(_fc), _frd)
                result["analytics"] = analytics_summary(_fctx)
                if _fctx.patterns:
                    from query.result_explainer import blend_patterns
                    result["answer"] = blend_patterns(result.get("answer") or "",
                                                       [p.detail for p in _fctx.patterns[:2]])
        except Exception as _fae:
            if verbose:
                print(f"  [federated] analytics skipped ({type(_fae).__name__}: {_fae})")
        return MultiResult(items=[_to_subresult(query, "federated", result)])
    # A federated EXECUTION/planning FAILURE (the generated cross-source SQL was invalid —
    # binder error, hallucinated column, unparseable, no plan) means the LLM mis-planned,
    # NOT that the question genuinely spans sources. Degrade to the normal single-source
    # path rather than hard-refusing the whole query. Only a PRINCIPLED refusal (truly
    # can't be answered across the scoped sources) is surfaced.
    _reason = str(payload.get("reason") or "").lower()
    if any(k in _reason for k in (
            "binder error", "does not have a column", "unparseable", "syntax error",
            "exec_error", "could not build", "no select generated", "does not exist",
            "unknown column", "not exist", "catalog error", "referenced column",
            # Infra gaps are never a principled refusal: a missing executor
            # dependency (e.g. "duckdb not installed" — observed live 2026-07-16,
            # stale inference image predating requirements/inference.txt's duckdb
            # line) means WE can't federate right now, not that the question
            # can't be answered — degrade to the normal single-source path.
            "not installed", "unavailable", "no module named")):
        if verbose:
            print(f"  [federated] plan failed ({_reason[:80]}) — falling back to single-source")
        return None
    # refused/blocked federation is a real, explained outcome — surface it, don't silently
    # fall back to a single-source answer that would drop a source.
    result = {"ok": False, "status": "federated_refused",
              "error": payload.get("reason") or "federation refused", "sql": payload.get("sql"),
              "usage": _fed_usage_totals, "latency_ms": _fed_latency_ms}
    return MultiResult(items=[_to_subresult(query, "federated", result)])


def run_hybrid_query(query, verbose=False, on_event=None):
    """Single entry point. Returns a MultiResult ALWAYS — a one-item MultiResult for a
    plain query, N items for a compound one. Callers branch on MultiResult, never on
    "is this compound", so everything downstream of here stays single-intent-dumb.

    on_event(phase, message, extra: dict), optional: fired at real stage transitions
    (classify, decompose, sub-query dispatch, per-modality routing, tier2 fallback,
    answer produced) so an SSE caller can stream genuine progress instead of blocking
    silently until the whole pipeline returns. Never required — None is a no-op.

    Compound handling (flag QUERY_DECOMPOSE_ENABLED): the DETERMINISTIC head
    self-certifies completeness (qualifier_completeness inside the fast path) — a clean
    SQL answer is known to cover the WHOLE utterance, so we skip the decomposer entirely
    (zero added latency on the hot path). A non-deterministic head (RAG/hybrid/NoSQL)
    CANNOT cheaply self-certify — it could answer one clause of a compound query and
    silently drop the rest — so there we decompose FIRST. A deterministic refusal also
    triggers decomposition (the utterance may have been several questions).

    L0 — the NL simplifier runs HERE (flag-gated by NL_SIMPLIFIER_ENABLED) so every
    consumer (CLI, inference API, demo) shares one simplification pass instead of each
    caller applying it (or not) itself. Off by default → zero added hot-path latency."""
    # L0 — NL simplifier (shared front-door step). No-op when the flag is off or the
    # simplifier is unavailable, so the original query flows through unchanged.
    # Its own call_slm() usage (purpose="nl_simplify") would otherwise never be
    # captured — it runs before any collect_usage() scope opens below — so it
    # gets its own small scope here, merged into whatever result is finally
    # returned via _merge_l0_usage() at every return point past this.
    from slm._call_slm import collect_usage as _collect_usage_l0, usage_totals as _usage_totals_l0
    _l0_calls = []
    try:
        from config import NL_SIMPLIFIER_ENABLED
    except Exception:
        NL_SIMPLIFIER_ENABLED = False
    if NL_SIMPLIFIER_ENABLED:
        try:
            from query.nl_simplifier import run_nl_simplifier
            with _collect_usage_l0() as _l0_usage:
                _l0 = run_nl_simplifier(query, verbose=verbose)
                _l0_calls = _l0_usage.calls()
            if getattr(_l0, "was_simplified", False):
                print(f"  [L0] Simplified: {_l0.simplified_query!r} ({_l0.duration_ms}ms)")
                query = _l0.simplified_query
        except Exception:
            pass  # fall back to the original query silently
    _l0_usage_totals = _usage_totals_l0(_l0_calls)

    # Runtime Context Provider (L0): pure system-value questions ("what's the
    # current date") need no table/SQL/LLM — answer directly before retrieval
    # ever runs, so a stray lexical match (e.g. "current" -> an is_current
    # column) can never select a table for a question that references no data.
    try:
        from config import RUNTIME_CONTEXT_ENABLED
    except Exception:
        RUNTIME_CONTEXT_ENABLED = False
    if RUNTIME_CONTEXT_ENABLED:
        from query.runtime_context import answer_runtime_context
        _rc = answer_runtime_context(query)
        if _rc is not None:
            # No on_event/"thinking" emit here — same as classify_node's smalltalk
            # fast path: an instant, deterministic answer has nothing to narrate.
            print(f"  [L0] Runtime context: {_rc['answer']!r}")
            return _merge_extra_usage(
                MultiResult(items=[_to_subresult(query, "runtime_context", _rc)]),
                _l0_usage_totals)

    # Cross-source federated route (MS-6): when the scope spans ≥2 sources and retrieval
    # selects columns from more than one, no single-DB head can join them — generate + run
    # a federated DuckDB query instead. Returns None (→ normal path) when not applicable.
    fed = _maybe_federated(query, verbose=verbose)
    if fed is not None:
        return _merge_extra_usage(fed, _l0_usage_totals)

    try:
        from config import QUERY_DECOMPOSE_ENABLED
    except Exception:
        QUERY_DECOMPOSE_ENABLED = False

    if not QUERY_DECOMPOSE_ENABLED:
        route, res = _dispatch_single(query, verbose=verbose, on_event=on_event)
        return _merge_extra_usage(
            MultiResult(items=[_to_subresult(query, route, res)]), _l0_usage_totals)

    _emit(on_event, "classify", "Classifying query intent...")
    intent, _source_ids = classify(query, verbose=verbose)

    # Deterministic head: try it directly; a clean answer is complete-by-construction.
    if intent == "sql":
        import io, contextlib
        _emit(on_event, "sql_probe", "Trying deterministic SQL...")
        sm, cols = _load_semantic_model()
        from veda.pipeline import run_query
        # Capture the probe's trace: if the head answers we replay it (hot path); if it
        # refuses and we then DECOMPOSE, the probe's "couldn't identify the entity" chatter
        # is misleading (the query was simply compound), so it must NOT reach the user.
        probe = io.StringIO()
        with contextlib.redirect_stdout(probe):
            det = run_query(query, sm, cols, return_result=True)
        if isinstance(det, dict) and (det.get("ok") or det.get("status") == "clarify"):
            # A CLARIFY is terminal, same as an answer: the head UNDERSTOOD the
            # utterance and asked a grounded question back. Running the decomposer
            # after it burns an SLM round (~30s on a busy SLM, measured: 2.8s head
            # → 39s total) and could only override the safe question with a
            # mis-split. Same contract as the Tier-2 clarify exemption.
            sys.stdout.write(probe.getvalue())
            _emit(on_event, "answer",
                  "Deterministic SQL answered the query" if det.get("ok")
                  else "Asked a clarifying question")
            return _merge_extra_usage(
                MultiResult(items=[_to_subresult(query, "deterministic", det)]),
                _l0_usage_totals)
        # Deterministic couldn't fully answer → maybe it was several questions.
        return _merge_extra_usage(
            _maybe_split(query, verbose=verbose, precomputed_sql=det,
                        probe_trace=probe.getvalue(), on_event=on_event),
            _l0_usage_totals)

    # RAG/hybrid/NoSQL self-certify nothing → decompose before dispatching (silent-drop guard).
    return _merge_extra_usage(
        _maybe_split(query, verbose=verbose, on_event=on_event), _l0_usage_totals)


def _maybe_split(query, verbose=False, precomputed_sql=None, probe_trace=None, on_event=None):
    """Run the decomposer, then either fan out independent sub-queries or fall back to
    the single-query pipeline. dependent_nested → refuse (out of scope for v1).

    probe_trace: the captured stdout of the deterministic probe (SQL intent only). Shown
    only on the single fallback (where it explains the refusal); discarded when we split
    or refuse-as-nested (there it would be a misleading 'couldn't answer' message)."""
    import io, contextlib
    from query.slm_layer import run_decomposer, DECOMP_DEPENDENT
    from slm._call_slm import collect_usage as _collect_usage_dc, usage_totals as _usage_totals_dc
    _emit(on_event, "decompose", "Checking whether this is a compound question...")
    # Capture the decomposer's own chatter so the on-screen order stays CHRONOLOGICAL. The
    # deterministic probe ran FIRST (its trace is in probe_trace); the decomposer runs AFTER.
    # Without capture, the decomposer prints live and appears BEFORE the replayed probe trace
    # — the scramble. We replay buffers in the order things actually happened.
    _dbuf = io.StringIO()
    _dc_calls = []
    with contextlib.redirect_stdout(_dbuf):
        with _collect_usage_dc() as _dc_usage:
            decomp = run_decomposer(query, verbose=verbose)
            # MUST read calls() INSIDE the with block — see _maybe_federated() for why.
            _dc_calls = _dc_usage.calls()
    _decomp_trace = _dbuf.getvalue()
    _dc_usage_totals = _usage_totals_dc(_dc_calls)

    if decomp.should_split:
        # Compound: the probe trace is a misleading "couldn't answer" for a query that was
        # simply several questions — suppress it; show the split decision + its reasoning.
        sys.stdout.write(_decomp_trace)
        print(f"\n  [Hybrid] compound query → {len(decomp.sub_queries)} independent sub-queries")
        _emit(on_event, "decompose", f"Split into {len(decomp.sub_queries)} sub-queries",
              sub_queries=list(decomp.sub_queries))
        return _merge_extra_usage(
            _fan_out(decomp.sub_queries, verbose=verbose, on_event=on_event), _dc_usage_totals)

    if decomp.type == DECOMP_DEPENDENT:
        # One part needs another's RESULT — these recompose into one query, which v1
        # doesn't build. Refuse rather than mis-split into wrong independent answers —
        # but GUIDE the user with the ordered parts the decomposer identified, so the
        # refusal is a path forward, not a dead end. (No recomposition engine needed.)
        steps = " ; then: ".join(f'"{s}"' for s in decomp.sub_queries) \
            if len(decomp.sub_queries) >= 2 else None
        reason = ("this is a nested question — one part depends on another part's result, "
                  "which v1 doesn't combine into one query.")
        if steps:
            reason += f" Ask the parts in order: {steps}"
        else:
            reason += " Ask the parts separately."
        return MultiResult.single(query, STATUS_REFUSED, "none", refuse_reason=reason)

    # single → run the FULL single-query pipeline (incl. Tier-2 / RAG) on the whole query,
    # reusing the deterministic result already computed when we have it. Emit in the order
    # things ACTUALLY ran: the probe (chronologically first) THEN the decomposer's "treated
    # as single" chatter — so the trace reads top-to-bottom as it happened (no scramble).
    if probe_trace:
        sys.stdout.write(probe_trace)
    sys.stdout.write(_decomp_trace)
    route, res = _dispatch_single(query, verbose=verbose, precomputed_sql=precomputed_sql,
                                   on_event=on_event)
    return _merge_extra_usage(
        MultiResult(items=[_to_subresult(query, route, res)]), _dc_usage_totals)


def _run_sub(sq, verbose=False, on_event=None, index=None, total=None):
    """Dispatch one sub-query, never raising — a crash becomes an error SubResult so one
    bad sub-query can't sink the others."""
    if index is not None and total is not None:
        _emit(on_event, "sub_query", f"Running sub-query {index}/{total}: {sq}",
              index=index, total=total, sub_query=sq)
    try:
        route, res = _dispatch_single(sq, verbose=verbose, on_event=on_event)
    except Exception as e:
        print(f"  [Hybrid] sub-query crashed: {type(e).__name__}: {e}")
        route, res = "none", None
    return _to_subresult(sq, route, res)


def _fan_out(sub_queries, verbose=False, on_event=None):
    """Run independent sub-queries and assemble the MultiResult IN QUERY ORDER.

    Default (QUERY_DECOMPOSE_MAX_WORKERS == 1): SEQUENTIAL with LIVE output — each
    sub-query prints its trace as it runs (no buffering, no stdout games).

    Concurrent (workers > 1): the DB layer is safe (fresh connection per call), but
    contextlib.redirect_stdout is process-global and NOT thread-safe — using it inside
    worker threads corrupts stdout (one sub-query's whole trace vanishes). So we install
    a thread-ROUTING stdout that sends each thread's writes to its own buffer, then replay
    buffers in query order. (Concurrency is opt-in pending model thread-safety checks.)"""
    try:
        from config import QUERY_DECOMPOSE_MAX_WORKERS as _MAXW
    except Exception:
        _MAXW = 1
    workers = max(1, min(_MAXW, len(sub_queries)))

    # Pre-warm shared read-only singletons ONCE so concurrent first access can't race.
    _load_semantic_model()

    total = len(sub_queries)
    if workers == 1:
        items = []
        for i, sq in enumerate(sub_queries, start=1):
            print(f"\n  [Hybrid] ── sub-query: {sq!r}")
            items.append(_run_sub(sq, verbose=verbose, on_event=on_event, index=i, total=total))
        return MultiResult(items=items)

    import io, sys, threading
    from concurrent.futures import ThreadPoolExecutor
    real_stdout = sys.stdout
    buffers = {}                       # thread id → that worker's capture buffer

    class _ThreadRouter:
        def write(self, s):
            (buffers.get(threading.get_ident()) or real_stdout).write(s)
        def flush(self):
            real_stdout.flush()

    # Carry the ambient (source, tenant) into the fan-out threads — worker threads
    # start with an empty contextvars context, so storage_adapters would otherwise
    # fail-closed / read the wrong tenant (§4.1). Captured in the parent, set per child.
    from veda_core.context import set_context as _set_ctx, try_current as _try_ctx
    _parent_ctx = _try_ctx()

    def _one(indexed_sq):
        i, sq = indexed_sq
        if _parent_ctx is not None:
            _set_ctx(_parent_ctx)
        buffers[threading.get_ident()] = io.StringIO()
        item = _run_sub(sq, verbose=verbose, on_event=on_event, index=i, total=total)
        return item, buffers[threading.get_ident()].getvalue()

    sys.stdout = _ThreadRouter()
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            pairs = list(ex.map(_one, enumerate(sub_queries, start=1)))  # ex.map preserves input order
    finally:
        sys.stdout = real_stdout

    items = []
    for (item, out), sq in zip(pairs, sub_queries):
        print(f"\n  [Hybrid] ── sub-query: {sq!r}")
        if out.strip():
            print(out.rstrip("\n"))
        items.append(item)
    return MultiResult(items=items)


def _dispatch_single(query, verbose=False, precomputed_sql=None, on_event=None):
    """The single-query pipeline: classify → best head → (Tier-2 for SQL). Returns
    (route, head_result). This is the UNCHANGED per-modality dispatch — every sub-query
    of a compound query runs through here exactly as a standalone query would."""
    intent, source_ids = classify(query, verbose=verbose)
    print(f"\n  [Hybrid] intent = {intent}   sources = {source_ids or 'default'}")
    _emit(on_event, "route", f"Routed to {intent} engine", intent=intent)

    # ── SQL → DETERMINISTIC engine (the correctness brain) ────────────────────
    if intent == "sql":
        sm, cols = _load_semantic_model()
        from veda.pipeline import run_query
        _head_t0 = time.time()
        res = precomputed_sql if isinstance(precomputed_sql, dict) \
            else run_query(query, sm, cols, return_result=True, on_event=on_event)
        _head_s = time.time() - _head_t0
        # Tier-2 fallback: if the deterministic head couldn't answer (refuse / dropped
        # qualifier / ungrounded / no table), let the LLM emit IR → deterministic
        # builder → GRAPH-GUARDED firewall → execute. Flag-gated (needs Ollama); the
        # graph guard (now live in the firewall) keeps LLM-proposed joins honest.
        # A CLARIFY is deliberately NOT retried: it is a grounded QUESTION the
        # deterministic head chose to ask (refuse-over-guess) — an LLM retry both
        # burns 40–80s and can override the safe question with guessed SQL.
        if isinstance(res, dict) and not res.get("ok") and res.get("status") in (
                "refuse", "qualifier_dropped", "ungrounded", "no_table",
                "exec_error"):
            try:
                from config import TIER2_LLM_FALLBACK
            except Exception:
                TIER2_LLM_FALLBACK = False
            # TIME BUDGET (heavy-lane governance): a slow deterministic head means
            # retrieval/grounding already struggled — Tier-2 rarely rescues those and
            # each SLM round is 30–120s. Skip Tier-2 when the head overspent, and give
            # Tier-2 itself a hard deadline (enforced between SLM rounds). Measured:
            # ungroundable maintenance-vocab queries burned 240s+ without this.
            try:
                from config import TIER2_SKIP_IF_HEAD_OVER_S, TIER2_TIME_BUDGET_S
            except Exception:
                TIER2_SKIP_IF_HEAD_OVER_S, TIER2_TIME_BUDGET_S = 60.0, 120.0
            if TIER2_LLM_FALLBACK and _head_s > TIER2_SKIP_IF_HEAD_OVER_S:
                print(f"  [Tier2] SKIPPED (head took {_head_s:.0f}s > "
                      f"{TIER2_SKIP_IF_HEAD_OVER_S:.0f}s budget) — refusal stands")
            elif TIER2_LLM_FALLBACK:
                print("  [Tier2] deterministic head couldn't answer → LLM-IR fallback")
                _emit(on_event, "tier2", "Deterministic head couldn't answer — trying LLM-assisted SQL...")
                from slm._call_slm import collect_usage, usage_totals
                _t2_t0 = time.time()
                _t2_calls = []
                with collect_usage() as _t2_usage:
                    t2 = _tier2_sql(query, sm, cols, verbose=verbose,
                                    deadline=time.time() + TIER2_TIME_BUDGET_S,
                                    execution_state=res.get("context") if isinstance(res, dict) else None,
                                on_event=on_event)
                    _t2_calls = _t2_usage.calls()  # read INSIDE the with — see _maybe_federated()
                if isinstance(t2, dict) and "usage" not in t2:
                    # Combine with Tier-1's own already-attempted usage (res["usage"],
                    # from run_query() above) — this fallback only fires because Tier-1
                    # tried and failed, so its SQL-gen tokens were genuinely spent on
                    # THIS query too, not just Tier-2's. Reporting Tier-2-only would
                    # undercount every query that fell through to this path.
                    _head_usage = (res.get("usage") if isinstance(res, dict) else None) or {}
                    _t2_totals = usage_totals(_t2_calls)
                    t2["usage"] = {
                        "prompt_tokens": _head_usage.get("prompt_tokens", 0) + _t2_totals["prompt_tokens"],
                        "completion_tokens": _head_usage.get("completion_tokens", 0) + _t2_totals["completion_tokens"],
                        "total_tokens": _head_usage.get("total_tokens", 0) + _t2_totals["total_tokens"],
                    }
                if isinstance(t2, dict) and "latency_ms" not in t2:
                    t2["latency_ms"] = round((_head_s + (time.time() - _t2_t0)) * 1000, 2)
                if t2 is not None:
                    if isinstance(t2, dict) and t2.get("status") == "tier2_rejected":
                        # Tier-2 exists to RESCUE a refusal; a candidate its own
                        # correctness gates killed is not an answer. The head's
                        # refusal (with its user-facing feedback) stands — the
                        # gate error ("fan-out risk: SUM(t2.list_value)…") is an
                        # internal note about SQL the user never saw.
                        print(f"  [Tier2] candidate rejected by gates — head refusal "
                              f"stands ({str(t2.get('error'))[:80]})")
                        if isinstance(res, dict):
                            res["tier2_note"] = t2.get("error")
                    else:
                        _emit(on_event, "answer", "Tier-2 SQL answered the query")
                        return "deterministic", t2
        elif isinstance(res, dict) and res.get("ok"):
            _emit(on_event, "answer", "SQL query executed")
        else:
            # Not tier-2-retried (status outside the retry list above, e.g.
            # clarify/invalid/ir_mismatch) AND not ok — the pipeline did NOT
            # actually answer. Emitting "SQL query executed" here would lie
            # to the caller about what happened (audit fix: this progress
            # event used to fire unconditionally in this branch).
            _emit(on_event, "answer", "SQL query could not be answered")
        return "deterministic", res

    # ── RAG → integrated document retrieval + synthesis ───────────────────────
    if intent == "rag":
        from query.rag_layer import run_rag_layer
        _emit(on_event, "rag", "Retrieving relevant documents...")
        rag = run_rag_layer(query, source_ids=source_ids,
                            temporal_filter=_temporal(query), verbose=verbose, on_event=on_event)
        if getattr(rag, "error", None):
            print(f"  [RAG] ✗ {rag.error}")
        else:
            print(f"\n  [RAG] {rag.answer}\n  citations: {rag.citations}")
            _emit(on_event, "answer", "Synthesized answer from retrieved documents")
        return "rag", rag

    # ── HYBRID → DETERMINISTIC SQL rows ⊕ document fusion ─────────────────────
    if intent == "hybrid":
        import types
        from veda.pipeline import run_query
        from query.rag_layer import run_hybrid_layer
        sm, cols = _load_semantic_model()
        _emit(on_event, "hybrid", "Running SQL and document fusion...")
        # Run the DETERMINISTIC SQL head first and feed its EXECUTED rows into the
        # fusion (the correct-by-construction numbers), instead of letting the fusion
        # rely on LLM-written SQL. (Also supplies the previously-missing sql_columns.)
        sqlres = run_query(query, sm, cols, return_result=True, on_event=on_event)
        sql_result = None
        if isinstance(sqlres, dict) and sqlres.get("ok"):
            _c, _r = sqlres.get("cols", []), sqlres.get("rows", [])
            sql_result = types.SimpleNamespace(
                columns=_c, rows=[dict(zip(_c, row)) for row in _r],
                row_count=len(_r), error=None)
        hy = run_hybrid_layer(query, sql_columns=[], source_ids=source_ids,
                             temporal_filter=_temporal(query),
                             sql_result=sql_result, verbose=verbose, on_event=on_event)
        if isinstance(sqlres, dict) and sqlres.get("ok"):
            # Attach the SQL head's OWN executed rows + already-built explain
            # (Tier1's _done() computed both from real, validated SQL) — never
            # recomputed here — so a hybrid answer with real tabular rows gets
            # the same chart/table/explainability apps/chat/services.py already
            # gives a plain SQL answer, instead of silently losing them because
            # HybridResult previously had no field to carry them.
            hy.cols = sqlres.get("cols") or []
            hy.rows = sqlres.get("rows") or []
            hy.explain = sqlres.get("explain")
            # Analytics + "Analysis:" fold-in on the SQL head's OWN executed rows —
            # same deterministic pass as Tier-1/Tier-2/federated (never a second
            # analysis; works off the already-attached cols/rows). Best-effort: a
            # failure here must never sink the hybrid answer.
            try:
                from veda.result_analyzer import analyze_result, analytics_summary
                if hy.cols and hy.rows:
                    _hrd = [row if isinstance(row, dict) else dict(zip(hy.cols, row))
                            for row in hy.rows]
                    _hctx = analyze_result(query, sqlres.get("sql") or "", list(hy.cols),
                                           _hrd, sm=sm, table=sqlres.get("table"))
                    hy.analytics = analytics_summary(_hctx)
                    if _hctx.patterns:
                        from query.result_explainer import blend_patterns
                        hy.answer = blend_patterns(hy.answer or "",
                                                   [p.detail for p in _hctx.patterns[:2]])
            except Exception as _hae:
                if verbose:
                    print(f"  [Hybrid] analytics skipped ({type(_hae).__name__}: {_hae})")
        if getattr(hy, "error", None):
            print(f"  [Hybrid] ✗ {hy.error}")
        else:
            print(f"\n  [Hybrid] {hy.answer}")
            _emit(on_event, "answer", "Fused SQL and document results into an answer")
        return "hybrid", hy

    # ── NoSQL → integrated native-query builder + execution ───────────────────
    if intent == "nosql":
        _emit(on_event, "nosql", "Querying document store...")
        result = _run_nosql(query, source_ids, verbose=verbose, on_event=on_event)
        _emit(on_event, "answer", "NoSQL query executed")
        return "nosql", result

    # ── default safety net ────────────────────────────────────────────────────
    sm, cols = _load_semantic_model()
    from veda.pipeline import run_query
    return "deterministic", run_query(query, sm, cols, return_result=True, on_event=on_event)


def _merge_extra_usage(mr, extra_usage):
    """Fold token counts spent BEFORE any collect_usage() scope opened (currently
    just L0's nl_simplify — see run_hybrid_query()) into the first sub-result's
    "usage", so a query-wide token total is never silently short by whatever ran
    at the very front door. No-op when extra_usage is zero or the first item's
    result isn't dict-shaped (RAG/hybrid/NoSQL results are objects with no usage
    key today — unaffected, not regressed)."""
    if not extra_usage.get("total_tokens") or not mr.items:
        return mr
    result = mr.items[0].result
    if not isinstance(result, dict):
        return mr
    base = result.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    result["usage"] = {
        "prompt_tokens": base.get("prompt_tokens", 0) + extra_usage["prompt_tokens"],
        "completion_tokens": base.get("completion_tokens", 0) + extra_usage["completion_tokens"],
        "total_tokens": base.get("total_tokens", 0) + extra_usage["total_tokens"],
    }
    return mr


def _to_subresult(sub_query, route, result):
    """Map a head result (dict for SQL/Tier-2, object for RAG/hybrid/NoSQL) to a typed
    SubResult. status is derived from the head's own success signal — NEVER invented."""
    if result is None:
        return SubResult(sub_query, STATUS_ERROR, route or "none", None, "no result")
    if isinstance(result, dict):
        if result.get("ok"):
            return SubResult(sub_query, STATUS_OK, route, result)
        st = result.get("status")
        reason = result.get("error") or st or "could not answer"
        # Tier-2 exec failures are infra errors; tier2_rejected is the CORRECTNESS GATE
        # declining an unsafe LLM answer (dropped qualifier / ungrounded value / wrong
        # semantics) — a refusal, same contract as deterministic declines.
        status = STATUS_ERROR if st == "tier2_exec_error" else STATUS_REFUSED
        return SubResult(sub_query, status, route, result, str(reason))
    # object-shaped head result (RAG / hybrid / NoSQL)
    err = getattr(result, "error", None)
    if err:
        return SubResult(sub_query, STATUS_ERROR, route, result, str(err))
    return SubResult(sub_query, STATUS_OK, route, result)


def _print_rows(cols, rows, sql=None):
    """Render Tier-2 result rows like the deterministic path — the rows are executed but
    were never shown, so the trace printed 'answered … N rows' with no table below it.
    Also surfaces the generated SQL so the chosen join/relationship is inspectable."""
    if sql:
        print("\n  Generated SQL (parameterized):")
        print("  " + "-" * 74)
        print(f"    {sql}")
        print("  " + "-" * 74)
    print(f"\n  Result: {len(rows)} rows (showing up to 20)\n")
    if cols:
        print("    " + " | ".join(str(c) for c in cols))
        print("    " + "-" * 74)
        for row in rows[:20]:
            cells = [("" if v is None else str(v))[:22] for v in row]
            print("    " + " | ".join(cells))


def _tier2_validate(query, raw_sql, sm, allowed_tables, allowed_cols, llm_written, tf):
    """The SAME correctness gates run_query applies (value_grounding + qualifier_completeness
    + ir_equivalence), run on a Tier-2 candidate BEFORE execution. Tier-2 fires precisely
    when the deterministic head REFUSED — often because a gate tripped — so re-answering
    with only the AST firewall (as before) let dropped-filter / fabricated-value / unrequested-
    semantics answers through. Returns (ok, reason). Mirrors veda/pipeline.py:579-619."""
    from veda.validation import value_grounding, qualifier_completeness
    from veda.ir_equivalence import validate_ir_equivalence
    import sqlglot
    from sqlglot import exp

    cols_meta = sm.get("columns", {})
    allowed_tables = set(allowed_tables)
    amap = {}
    try:
        tree = sqlglot.parse_one(raw_sql, read="postgres")
        for t in tree.find_all(exp.Table):
            if t.alias:
                amap[t.alias.lower()] = t.name
    except Exception:
        pass
    _default_tbl = next(iter(allowed_tables)) if len(allowed_tables) == 1 else None

    def _resolve(colexp):
        if colexp.table:
            return amap.get(colexp.table.lower())
        owners = [t for t in allowed_tables if f"{t}.{colexp.name}" in cols_meta]
        return owners[0] if len(owners) == 1 else _default_tbl

    ok_val, bad = value_grounding(raw_sql, _resolve, cols_meta)
    if not ok_val:
        return False, f"ungrounded value {bad}"
    # STRICT: LLM-lane answers face the QSR-aware gate — an unaccounted token with a
    # referent anywhere in the schema is a dropped qualifier, closing the wrong-table
    # blind spot (SELECT * FROM assets_asset for "most expensive financial records").
    ok_q, missing = qualifier_completeness(query, raw_sql, sm, strict=True)
    if not ok_q:
        return False, f"dropped qualifier {missing!r}"
    _tcols = ({k.split(".", 1)[1] for k, m in cols_meta.items()
               if k.split(".", 1)[0] in allowed_tables
               and (m or {}).get("semantic_type") == "TEMPORAL"}
              if (tf and (getattr(tf, "start", None) or getattr(tf, "end", None))) else set())
    ok_ir, ir_viol = validate_ir_equivalence(query, raw_sql, sm, allowed_tables=allowed_tables,
                                             temporal_cols=_tcols, llm_generated=llm_written)
    if not ok_ir:
        return False, f"ir_mismatch: {'; '.join(ir_viol)}"

    # ── Shared analytical-semantics check — the SAME generic, metadata-driven
    # invariants Tier-1 uses (veda/semantic_validation.py). This is the common
    # boundary for BOTH Tier-2 IR SQL and LangGraph SQL (run_langgraph_pipeline's
    # output is validated through this same function). Advisory by default (logged);
    # with SEMANTIC_VALIDATION_ENFORCE a hard operator-loss finding (the LLM ignored
    # the requested AVG/SUM/…) drives the EXISTING repair/retry loop by returning a
    # reason, instead of executing SQL that answers a different question. Never raises.
    try:
        from config import SEMANTIC_VALIDATION_ENABLED as _SV_ON, SEMANTIC_VALIDATION_ENFORCE as _SV_ENF
    except Exception:
        _SV_ON, _SV_ENF = False, False
    if _SV_ON:
        try:
            from veda.semantic_validation import validate_analytical_semantics
            _sv = validate_analytical_semantics(query, raw_sql, sm, graph=None)
            _hard = [f for f in _sv if f.get("code") in
                     ("operator_mismatch", "operator_dropped", "missing_group_by")]
            if _sv:
                print(f"  [Tier2] Semantics  {len(_sv)} finding(s): "
                      f"{', '.join(sorted({f['code'] for f in _sv}))}"
                      + (" (enforced)" if (_SV_ENF and _hard) else " (advisory)"))
            if _SV_ENF and _hard:
                return False, f"semantic: {_hard[0]['code']} — {_hard[0]['detail']}"
        except Exception:
            pass
    return True, ""


def _repair_hint_for(error: str) -> str:
    """Turn a firewall/execution error into a corrective instruction appended to the SLM
    prompt on the NEXT IR attempt (execution-feedback self-repair, IR-level).

    The LLM emits IR, never SQL, so the hint steers IR choices (columns/joins/grain) — it
    never asks the model to 'fix SQL'. Classified for a targeted nudge; generic fallback
    otherwise."""
    e = (error or "").lower()
    if "column" in e and any(k in e for k in ("unknown", "not exist", "does not exist", "no such")):
        cls = ("The previous attempt referenced a column that does not exist. Use ONLY the "
               "column UUIDs provided above — never invent column names.")
    elif any(k in e for k in ("join", "fk", "cartesian", "edge", "not directly related")):
        cls = ("The previous attempt proposed a join that is not a real foreign-key edge. "
               "Only join tables that share a provided FK relationship; otherwise answer "
               "with a single table.")
    elif "ungrounded" in e or "value" in e:
        cls = ("The previous attempt filtered on a value that is not present in the data. "
               "Only filter on values that actually exist in the named column.")
    elif "qualifier" in e or "dropped" in e:
        cls = ("The previous attempt dropped a condition the question asked for. Represent "
               "every filter/grouping/ordering the question mentions.")
    elif "ambiguous" in e:
        cls = "The previous attempt was ambiguous about which column or table was meant — be explicit."
    elif "ir_mismatch" in e or "syntax" in e:
        cls = "The previous attempt did not match the question's intent. Produce a simpler, faithful IR."
    else:
        cls = "The previous attempt failed validation/execution. Produce a simpler, correct IR."
    return f"[REPAIR] {cls} (error: {str(error)[:180]})"


def _is_param_mismatch(err) -> bool:
    """True when an exec error is the classified placeholder/param-count mismatch
    (veda.execution.PARAM_MISMATCH_ERROR) — OUR param assembly failed, so the SLM
    repair loop can't fix it and the caller should keep the deterministic refusal."""
    try:
        from veda.execution import PARAM_MISMATCH_ERROR
        return PARAM_MISMATCH_ERROR in str(err or "")
    except Exception:
        return "parameter mismatch" in str(err or "")


def _tier2_finish(query, sm, cols, rows, sql, source, business_intent=None):
    """Bring a Tier-2 result to STRUCTURAL PARITY with the deterministic
    (Tier-1, veda/pipeline.py's _done()) response shape — same bug/fix as the
    NoSQL path (_run_nosql), extended: rows were always correct, but
    previously none of Tier-2's three success returns ever computed a
    natural-language answer, a "table" key, or a real "explain" — the latter
    two were EITHER MISSING ENTIRELY (table) OR only present when
    INSIGHT_ENGINE_ENABLED (explain), so with that flag at its default (off)
    every Tier-2 answer had a visibly different shape than Tier-1: no table,
    and explainability always fell back to the empty placeholder. Tier-1
    never gates table/explain on that flag, so Tier-2 shouldn't either — only
    insights/follow_up_questions/visualization/confidence are flag-gated.
    Never raises — a summarization/analysis failure still returns the
    (correct) rows, just without prose/insights."""
    result = {"status": "answered", "ok": True, "cols": cols, "rows": rows,
              "sql": sql, "source": source}

    # table: derived from the SQL's own primary entity (AST, zero LLM) — Tier-2
    # SQL may join multiple tables, so this is the FIRST referenced table,
    # matching Tier-1's single-table `table` field as closely as this
    # multi-table-capable path allows.
    table = None
    try:
        from veda.business_explain import extract_sql_facts
        facts = extract_sql_facts(sql or "")
        table = facts["entities"][0] if facts["entities"] else None
    except Exception:
        pass
    result["table"] = table

    try:
        from config import (NL_ANSWER_ENABLED, NL_ANSWER_FAST_TIMEOUT_MS,
                            NL_SUMMARY_TIMEOUT_MS,
                            INSIGHT_ENGINE_ENABLED, RESULT_ANALYZER_MAX_ROWS)
    except Exception:
        NL_ANSWER_ENABLED, NL_ANSWER_FAST_TIMEOUT_MS = True, 2500
        NL_SUMMARY_TIMEOUT_MS = 10000
        INSIGHT_ENGINE_ENABLED, RESULT_ANALYZER_MAX_ROWS = False, 200

    visualization = None
    _ictx = None
    _confidence = None
    if NL_ANSWER_ENABLED and cols:
        row_dicts = [r if isinstance(r, dict) else dict(zip(cols, r)) for r in rows]
        # Deterministic analytics (ALWAYS, not flag-gated) — same single
        # post-execution analysis pass Tier-1 (veda/pipeline.py L7b) computes:
        # column stats/roles, result shape, patterns, chart candidates,
        # grounding metadata. Zero LLM; attached to the result so downstream
        # consumers read one computation. Only the SLM narrative below stays
        # gated behind INSIGHT_ENGINE_ENABLED.
        try:
            from veda.result_analyzer import analyze_result, analytics_summary
            _ictx = analyze_result(query, sql, list(cols), row_dicts, sm=sm, table=table,
                                   max_rows=RESULT_ANALYZER_MAX_ROWS)
            result["analytics"] = analytics_summary(_ictx)
        except Exception as _ae:
            print(f"  [Tier2] Analytics (skipped: {type(_ae).__name__}: {_ae})")
        # Safe default FIRST, same as veda/pipeline.py's L7b (the Tier-1 path) —
        # previously this function only set result["answer"] on a SUCCESSFUL
        # insight/NL-answer call; run_insight_engine/run_nl_answer already
        # degrade to a deterministic fallback internally on an SLM failure, but
        # an exception from ANALYSIS itself (e.g. analyze_result()/extract_sql_
        # facts() raising, before the SLM call is even attempted) skipped both
        # try blocks entirely, leaving "answer" absent from the result dict —
        # worse than Tier-1's raw-text fallback: format_reply_node's own
        # generic "Here's what I found." masked that a Tier-2 turn had silently
        # produced NO grounded summary at all. Always overwritten below by a
        # real summary when either call succeeds.
        from query.nl_answer import deterministic_fallback_answer
        result["answer"] = deterministic_fallback_answer(query, list(cols), row_dicts)
        got_real_answer = False

        if INSIGHT_ENGINE_ENABLED and _ictx is not None:
            try:
                from query.result_explainer import run_insight_engine
                insight = run_insight_engine(_ictx)   # same ctx as above — one analysis pass
                if getattr(insight, "answer", None):
                    result["answer"] = insight.answer
                    got_real_answer = True
                result["insights"] = insight.insights
                result["follow_up_questions"] = insight.follow_up_questions
                result["visualization"] = insight.visualization
                _confidence = insight.confidence
                visualization = insight.visualization
            except Exception as _ie:
                print(f"  [Tier2] Insight Engine unavailable ({type(_ie).__name__}: {_ie}) "
                      f"— falling back to plain NL answer")
                # Record the attempt even though it failed — call_slm() only records
                # usage on a SUCCESSFUL backend.call() return, so an exception here
                # means zero tokens get attributed to this attempt even if the SLM
                # was actually contacted. Without this, usage.total_tokens == 0 is
                # indistinguishable from "no LLM call was needed this turn." Tier-2
                # has no ExplainTrace to record onto (unlike Tier-1's tr.set("nl_summary",
                # ...)), so this lives under "_debug" — a key inference/routes/hybrid.py's
                # _INTERNAL_ONLY_KEYS strips at every nesting depth, same guarantee as
                # "trace"/"context", so it never reaches the client-facing response.
                result.setdefault("_debug", {})["insight_engine_failed"] = True
                result["_debug"]["insight_engine_error"] = f"{type(_ie).__name__}: {str(_ie)[:200]}"

        # got_real_answer (not "answer" in result — that key is ALWAYS present
        # now, see the deterministic default set above): still tries plain
        # run_nl_answer whenever insight-engine didn't produce a genuine
        # summary, exactly as before this fix.
        # The insight engine's prompt already grounds on the patterns_block, so a
        # genuine insight answer has ALSO woven the findings in.
        _pattern_details = ([p.detail for p in _ictx.patterns[:2]]
                            if (_ictx is not None and getattr(_ictx, "patterns", None)) else [])
        _slm_wove_patterns = got_real_answer

        if not got_real_answer:
            try:
                from query.nl_answer import run_nl_answer
                nl = run_nl_answer(query, list(cols), row_dicts,
                                   timeout=NL_SUMMARY_TIMEOUT_MS / 1000.0, semantic_model=sm,
                                   patterns=_pattern_details,
                                   result_shape=getattr(_ictx, "result_shape", None))
                if getattr(nl, "answer", None):
                    result["answer"] = nl.answer
                    _slm_wove_patterns = True   # SLM prose wove them; fallback blended them itself
            except Exception as _nle:
                print(f"  [Tier2] Answer (summarisation skipped: {type(_nle).__name__})")

        # Fold the deterministic analytics into the final summary ONLY when no
        # summary SLM already phrased them — same natural-blend / no-double-statement
        # rule as Tier-1 (veda/pipeline.py L7b, 2026-07-17). Top 2 only.
        if _pattern_details and not _slm_wove_patterns:
            from query.result_explainer import blend_patterns
            result["answer"] = blend_patterns(result.get("answer") or "", _pattern_details)

    try:
        from veda.business_explain import build_explain
        result["explain"] = build_explain(sql=sql or "", table=table or "", sm=sm,
                                          visualization=visualization,
                                          confidence=_confidence)
    except Exception:
        print("  [Tier2] explainability skipped")
    # business_intent (advisory): deterministic reading of the EXECUTED SQL
    # first (explain.understanding.summary — the source of truth); the SLM's
    # own advisory claim (`business_intent` param, from the Tier-2 IR envelope)
    # only fills in when the deterministic one is unavailable. Presentation
    # metadata only — never feeds validation or SQL.
    _det_intent = ((result.get("explain") or {}).get("understanding") or {}).get("summary")
    if _det_intent or business_intent:
        result["business_intent"] = _det_intent or business_intent
    return result


def _tier2_sql(query, sm, all_cols, verbose=False, deadline=None, execution_state=None, on_event=None):
    """Tier-2 SQL fallback (only when the deterministic head can't answer).

    LLM emits IR → deterministic sql_builder makes the SQL (LLM never writes SQL) →
    the GRAPH-GUARDED firewall validates (every join must be a real FK edge, no
    cartesian, value-grounded) → execute. Returns a result dict or None. Needs Ollama
    + the integrated retrieval stores; any failure → None (caller keeps the refusal).

    deadline: optional absolute time.time() cutoff — checked between SLM rounds so
    an expired Tier-2 budget returns the head's refusal instead of burning minutes
    (TIER2_TIME_BUDGET_S at the call site).

    execution_state: optional veda.execution_state.ExecutionState from Tier1's own
    run_query() call (see _dispatch_single) — when given, reuses Tier1's temporal
    parse and seeds retrieval with Tier1's candidate fields instead of starting cold.
    None (default) preserves this function's exact prior behavior."""
    try:
        from query.retrieval_select import select_retrieval
        from query.slm_layer import run_slm_layer
        from query.sql_builder import run_sql_builder
        from veda.validation import validate_and_parameterize, value_grounding
        from veda.execution import execute_sql
        from query.temporal_parser import run_temporal_parser

        if execution_state is not None and execution_state.temporal_result is not None:
            tf = execution_state.temporal_result.temporal_filter
        else:
            tf = run_temporal_parser(query).temporal_filter

        _seeds = execution_state.candidate_fields if execution_state is not None else None
        if verbose and execution_state is not None:
            # Only claim what's ACTUALLY functionally reused below — Temporal (the tf
            # computed above) and Candidate Fields (seed_candidates, passed to
            # select_retrieval right below; primary_table is folded into these fields'
            # scores in pipeline.py, not used standalone here). query_understanding and
            # sql_planning are carried on ExecutionState for future use but are NOT yet
            # consumed by any Tier2 decision — deliberately left out of this log so it
            # doesn't overstate what this function does.
            _reused = []
            if execution_state.temporal_result is not None: _reused.append("Temporal")
            if _seeds:                                       _reused.append("Candidate Fields")
            if execution_state.primary_table and _seeds:
                _reused.append(f"Primary Table ({execution_state.primary_table!r}, "
                                f"biased in Candidate Fields)")
            if _reused:
                print(f"  [Tier2] Tier1 completed. Reusing: {', '.join('✓ ' + r for r in _reused)}")
            print("  [Tier2] continuing execution...")
        sel = select_retrieval(query=query, intent="sql", verbose=verbose, seed_candidates=_seeds)

        # ── ENVELOPE PATH (D): one-call SLM → intent envelope → deterministic build_sql.
        # Single-table analytical shapes (count/measure/ratio/trend/compare/group/list) go
        # through the ONE shared builder + value-grounding + AST firewall. On None (multi-
        # entity, unresolved handle, grain_suspect, or Ollama down) it falls through to the
        # IR→sql_builder path below — additive, never replaces the existing fallback.
        try:
            from query.envelope_slm import emit_envelope
            from query.intent_envelope import map_envelope_to_intent
            from query.intent import validate_intent, build_sql
            _env, _hmap = emit_envelope(query, sel.columns, verbose=verbose)
            _qi = map_envelope_to_intent(_env, _hmap, tf) if _env else None
            if _qi is not None and validate_intent(_qi)[0] == "ok":
                _sql, _tbls, _cols, _route, _why = build_sql(_qi)
                ok_val, bad = value_grounding(_sql, lambda _c: _qi.subject_table,
                                              sm.get("columns", {}))
                if not ok_val:
                    print(f"  [Tier2] envelope value ungrounded {bad} — fallback to IR")
                else:
                    psql, params, err = validate_and_parameterize(_sql, _tbls, _cols)
                    if err:
                        print(f"  [Tier2] envelope firewall rejected ({err}) — fallback to IR")
                    elif not (_ev := _tier2_validate(query, psql, sm, set(_tbls), _cols,
                                                     llm_written=True, tf=tf))[0]:
                        print(f"  [Tier2] envelope gated ({_ev[1]}) — fallback to IR")
                    else:
                        ecols, erows, eerr = execute_sql(psql, list(params))
                        if eerr:
                            print(f"  [Tier2] envelope exec error ({eerr}) — fallback to IR")
                        else:
                            print(f"  [Tier2] answered via ENVELOPE ({_qi.query_type}) — {len(erows)} rows")
                            _print_rows(ecols, erows, sql=psql)
                            return _tier2_finish(query, sm, ecols, erows, psql, "envelope")
        except Exception as _ee:
            print(f"  [Tier2] envelope path skipped: {type(_ee).__name__}: {str(_ee)[:120]}")

        # ── RECOMMENDED PROJECTION (2026-07) ───────────────────────────────────────────
        # Reuse veda/routing.py::recommended_projection() — the same business-facing
        # SELECT-list composer Tier1 already uses (default display column + this
        # query's retrieval relevance + HIGH-importance columns) — so the IR path's
        # SLM gets the same "what should be displayed" guidance instead of self-judging
        # relevance from a flat retrieval list alone. Computed ONCE here, threaded into
        # BOTH run_slm_layer's non-langgraph and langgraph (default) branches as plain
        # data — neither branch recomputes it.
        #
        # Primary table: Tier1's own vetted choice when this call is reusing Tier1's
        # ExecutionState (the common case — Tier2 only runs after Tier1), else
        # sel.tables[0] (select_retrieval's own top-ranked table — already computed,
        # no new ranking here).
        #
        # NOTE: sel.columns are ingestion.vector_store.RetrievalResult (Tier2's own
        # retrieval shape: col_id is a UUID). recommended_projection()'s "this query's
        # retrieval relevance" signal expects retrieval.retrieval_engine_phase3.RetrievalResult
        # (Tier1's shape: col_id is "table.col", plus .final_score) — that one signal
        # silently no-ops here (caught by its own try/except in routing.py), but the
        # display-column and HIGH-importance signals — the two that actually exclude
        # audit columns — read only `primary`/`sm`/`allowed_columns` and apply in full.
        _rec_proj_cols = None
        try:
            from veda.routing import recommended_projection
            _t2_primary = (execution_state.primary_table if execution_state is not None else None) \
                          or (sel.tables[0] if sel.tables else None)
            if _t2_primary:
                _t2_allowed = [k.split(".", 1)[1] for k in all_cols
                               if k.split(".", 1)[0] == _t2_primary]
                if _t2_allowed:
                    _rec_names = recommended_projection(_t2_primary, _t2_allowed, sel.columns, sm, query)
                    _rec_proj_cols = [r for r in sel.columns
                                      if r.table_name == _t2_primary and r.col_name in _rec_names] or None
        except Exception as _pe:
            print(f"  [Tier2] recommended projection skipped: {type(_pe).__name__}: {str(_pe)[:120]}")
            _rec_proj_cols = None

        # ── IR PATH with bounded EXECUTION-FEEDBACK REPAIR loop ───────────────────────
        # The LLM emits IR (never SQL); on a firewall rejection or execution error we feed
        # the classified error back into the SLM prompt (via _repair_hint_for) and retry a
        # corrected IR, instead of refusing on the first miss. Bounded by
        # VALIDATION_MAX_REPAIR_ATTEMPTS; on exhaustion the original rejection stands. The
        # hint is appended to the QUERY so it reaches the prompt regardless of which
        # run_slm_layer branch runs (both build the prompt from `query`) — no SLM-internal
        # edits. Flag-gated, off by default: on any config miss the loop runs 0 extra times
        # and behaves exactly as before.
        try:
            from config import VALIDATION_REPAIR_LOOP_ENABLED, VALIDATION_MAX_REPAIR_ATTEMPTS
        except Exception:
            VALIDATION_REPAIR_LOOP_ENABLED, VALIDATION_MAX_REPAIR_ATTEMPTS = False, 0
        _max_repairs = int(VALIDATION_MAX_REPAIR_ATTEMPTS) if VALIDATION_REPAIR_LOOP_ENABLED else 0
        # Seed attempt 0 with WHY Tier1 refused (when known) instead of starting cold —
        # reuses the existing repair-hint mechanism, not a second retry framework.
        _repair_hint = (_repair_hint_for(execution_state.refusal_reason)
                         if (execution_state is not None and execution_state.refusal_reason
                             and _max_repairs > 0) else None)
        from config import LANGGRAPH_SHARED_PLANNER

        for _attempt in range(_max_repairs + 1):
            # hard deadline between SLM rounds — an expired budget returns the
            # deterministic refusal instead of starting another 30–120s generation
            if deadline is not None and time.time() > deadline:
                print(f"  [Tier2] time budget exhausted before attempt {_attempt} — "
                      f"keeping deterministic refusal")
                return None
            _q_ir = query if not _repair_hint else f"{query}\n\n{_repair_hint}"
            if _repair_hint and _attempt == 0:
                print("  [Tier2] seeded with Tier1's refusal reason")
            elif _repair_hint:
                print(f"  [Tier2] repair attempt {_attempt}/{_max_repairs}")
            l3 = run_slm_layer(query=_q_ir, temporal_filter=tf, top_k_columns=sel.columns,
                               join_path=sel.join_path, verbose=verbose,
                               recommended_projection=_rec_proj_cols, on_event=on_event)
            if getattr(l3, "error", None) or not getattr(l3, "ir_json", None):
                print(f"  [Tier2] no usable IR from SLM "
                      f"({getattr(l3, 'error', None) or 'empty ir_json'}) — keeping refusal")
                return None

            # ── ONE JOIN ENGINE (Phase 2) ─────────────────────────────────────────
            # If the LLM identified MULTIPLE entities, build the joins with the
            # deterministic graph planner (plan_join_tree), NOT sql_builder's retrieval
            # join_path. The LLM only NAMES entities; the graph-verified planner builds
            # (or refuses) the joins — same engine the deterministic head uses.
            ents = (l3.ir_json or {}).get("entities", []) or []
            id2name = {r.table_id: r.table_name for r in sel.columns}
            ent_names = [n for n in dict.fromkeys(id2name.get(e.get("table_id")) for e in ents) if n]
            if LANGGRAPH_SHARED_PLANNER and len(ent_names) >= 2:
                from veda.planning import build_from_entities
                act = build_from_entities(query, sm, all_cols, tf, ent_names[0], ent_names[1:],
                                          results=sel.columns)
                if isinstance(act, dict) and act.get("sql"):
                    a_tables = set(act.get("tables", []))
                    a_cols = act.get("columns") or [k.split(".", 1)[1] for k in all_cols
                                                    if k.split(".", 1)[0] in a_tables]
                    psql, params, err = validate_and_parameterize(act["sql"], a_tables, a_cols)
                    if err:
                        if _attempt < _max_repairs:
                            _repair_hint = _repair_hint_for(err); continue
                        print(f"  [Tier2] shared-planner firewall rejected (kept safe): {err}")
                        return {"status": "tier2_rejected", "ok": False, "error": err}
                    cols, rows, eerr = execute_sql(psql, list(params))
                    if eerr:
                        if _is_param_mismatch(eerr):
                            # our own param assembly failed — not SLM-repairable; keep
                            # the deterministic head's clean refusal, never a raw crash
                            print(f"  [Tier2] shared-planner exec degraded: {eerr}")
                            return None
                        if _attempt < _max_repairs:
                            _repair_hint = _repair_hint_for(eerr); continue
                        return {"status": "tier2_exec_error", "ok": False, "error": eerr}
                    print(f"  [Tier2] answered via SHARED planner (graph-verified joins)"
                          f"{' after repair' if _repair_hint else ''} — {len(rows)} rows")
                    _print_rows(cols, rows, sql=psql)
                    return _tier2_finish(query, sm, cols, rows, psql, "tier2_shared_planner",
                                         business_intent=getattr(l3, "business_intent", None))
                # planner refused/clarified the multi-table join → respect it (refuse-over-guess)
                if isinstance(act, dict) and act.get("action") in ("refuse", "clarify"):
                    if verbose:
                        print(f"  [Tier2] shared planner declined join: {act.get('msg','')}")
                    return None
                # otherwise fall through to single-table sql_builder below

            l4 = run_sql_builder(ir_json=l3.ir_json, top_k_columns=sel.columns,
                                 join_path=sel.join_path, verbose=verbose)
            if getattr(l4, "error", None) or not getattr(l4, "sql", None):
                return None
            allowed_tables = set(getattr(l4, "tables_used", []) or [])
            allowed_cols = [k.split(".", 1)[1] for k in all_cols
                            if k.split(".", 1)[0] in allowed_tables]
            # firewall — graph_guard (live) verifies LLM-proposed joins against the FK graph
            psql, params, err = validate_and_parameterize(l4.sql, allowed_tables, allowed_cols)
            if err:
                if _attempt < _max_repairs:
                    _repair_hint = _repair_hint_for(err); continue
                print(f"  [Tier2] firewall rejected (kept safe): {err}")
                return {"status": "tier2_rejected", "ok": False, "error": err}
            # Correctness gates (value grounding + STRICT qualifier + IR equivalence) —
            # previously defined but never called on this path, which is how a bare
            # SELECT * from the wrong table shipped as an "answer". NO repair retry on
            # a gate failure: a dropped qualifier isn't fixable by IR nudging (the
            # schema doesn't change between attempts), and each retry is a full SLM
            # round — measured blowing the q61-class past 200s on CPU fallback.
            _ok2, _why2 = _tier2_validate(query, psql, sm, allowed_tables, allowed_cols,
                                          llm_written=True, tf=tf)
            if not _ok2:
                print(f"  [Tier2] gated (kept safe): {_why2}")
                return {"status": "tier2_rejected", "ok": False, "error": _why2}
            cols, rows, eerr = execute_sql(psql, list(params))
            if eerr:
                if _is_param_mismatch(eerr):
                    print(f"  [Tier2] IR exec degraded: {eerr}")
                    return None
                if _attempt < _max_repairs:
                    _repair_hint = _repair_hint_for(eerr); continue
                return {"status": "tier2_exec_error", "ok": False, "error": eerr}
            print(f"  [Tier2] answered via LLM-IR (graph-verified)"
                  f"{' after repair' if _repair_hint else ''} — {len(rows)} rows")
            _print_rows(cols, rows, sql=psql)
            return _tier2_finish(query, sm, cols, rows, psql, "tier2",
                                 business_intent=getattr(l3, "business_intent", None))
        return None   # repair attempts exhausted → keep the refusal
    except Exception as e:
        # Always surface WHY Tier-2 bailed (Ollama down, retrieval store missing, etc.) —
        # otherwise the path silently no-ops and looks like it never ran.
        print(f"  [Tier2] unavailable: {type(e).__name__}: {str(e)[:140]}")
        return None


def _run_nosql(query, source_ids, verbose=False, on_event=None):
    """Compact NoSQL path: resolve the source, infer schema, build + execute.

    on_event: optional progress callback — previously schema inference + the
    LLM-based query-building step (run_nosql_builder) were a silent black box
    between the caller's outer "Querying document store..."/"NoSQL query
    executed" ticks (_dispatch_single)."""
    from config import get_source, SQL_DEFAULT_LIMIT
    from connectors.base import build_connector
    from query.nosql_builder import run_nosql_builder
    for sid in (source_ids or []):
        try:
            src = get_source(sid)
            if src.get("type") != "nosql":
                continue
            conn = build_connector(src)
            if not conn.connect().ok:
                continue
            collections = conn.get_nosql_schema()
            conn.disconnect()
            _emit(on_event, "nosql_build", "Figuring out how to query your data")
            nb = run_nosql_builder(query=query, source_id=sid,
                                   engine=src.get("engine", "mongodb"),
                                   collections=collections, verbose=verbose)
            if nb.error:
                print(f"  [NoSQL] ✗ {nb.error}"); continue
            conn2 = build_connector(src); conn2.connect()
            res = conn2.execute_query(query=nb.query_json,
                                      row_limit=SQL_DEFAULT_LIMIT, timeout_sec=30)
            conn2.disconnect()
            print(f"  [NoSQL] {getattr(res,'row_count','?')} docs")
            # NL-back summary, parity with the SQL path (gated; graceful fallback).
            # F6-equivalent: bound worst-case latency with NL_SUMMARY_TIMEOUT_MS instead
            # of falling through to the SLM's full default timeout.
            try:
                from config import NL_ANSWER_ENABLED, NL_SUMMARY_TIMEOUT_MS
            except Exception:
                NL_ANSWER_ENABLED = False
                NL_SUMMARY_TIMEOUT_MS = 1000
            cols = getattr(res, "columns", None)
            rows = getattr(res, "rows", None)
            if NL_ANSWER_ENABLED and cols and rows is not None:
                try:
                    from query.nl_answer import run_nl_answer
                    row_dicts = [r if isinstance(r, dict) else dict(zip(cols, r)) for r in rows]
                    nl = run_nl_answer(query, list(cols), row_dicts,
                                       timeout=NL_SUMMARY_TIMEOUT_MS / 1000.0)
                    if getattr(nl, "answer", None):
                        res.answer = nl.answer
                        print(f"  [NoSQL] Answer  {nl.answer}")
                except Exception:
                    pass
            # Deterministic analytics + "Analysis:" fold-in — parity with
            # Tier-1/Tier-2/federated. Degraded mode (no single-table semantic
            # model / no SQL string for a document store, so grounding fields stay
            # empty), but column stats, result shape, chart candidates and patterns
            # all work off (cols, rows) alone. Best-effort: never sink the answer.
            if cols and rows:
                try:
                    from veda.result_analyzer import analyze_result, analytics_summary
                    _nrd = [r if isinstance(r, dict) else dict(zip(cols, r)) for r in rows]
                    _nctx = analyze_result(query, "", list(cols), _nrd,
                                           connector_type="nosql")
                    res.analytics = analytics_summary(_nctx)
                    if _nctx.patterns:
                        from query.result_explainer import blend_patterns
                        res.answer = blend_patterns(getattr(res, "answer", None) or "",
                                                    [p.detail for p in _nctx.patterns[:2]])
                except Exception as _nae:
                    if verbose:
                        print(f"  [NoSQL] analytics skipped ({type(_nae).__name__}: {_nae})")
            return res
        except Exception as e:
            print(f"  [NoSQL] source {sid} failed: {type(e).__name__}: {e}")
    print("  [NoSQL] no usable NoSQL source")
    return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    debug = "--debug" in sys.argv
    if not args:
        print('usage: python3 veda_hybrid.py "<your question>" [--verbose] [--debug]')
        return 1
    if debug:
        # --debug → capture the full explainability trace (incl. candidate lists)
        import config as _cfg
        _cfg.EXPLAIN_TRACE_ENABLED = True
        _cfg.EXPLAIN_TRACE_VERBOSE = True
    res = run_hybrid_query(" ".join(args), verbose="--verbose" in sys.argv)
    _render_multi(res)
    if debug:
        from veda.explain import render_trace
        for it in res.items:
            if isinstance(it.result, dict) and it.result.get("trace"):
                print("\n" + render_trace(it.result["trace"]))
    return 0


def _render_multi(mr):
    """Per-head output already printed as each sub ran; this adds the compound recap
    (which sub-query → which route → answered/refused) and surfaces a single refusal."""
    if not mr.is_compound:
        it = mr.items[0]
        if it.status != STATUS_OK and it.refuse_reason:
            tag = "refused" if it.status == STATUS_REFUSED else "error"
            print(f"\n  [Hybrid] {tag}: {it.refuse_reason}")
        return
    print("\n  " + "=" * 74)
    print(f"  Compound query — {len(mr.items)} sub-queries "
          f"({sum(1 for i in mr.items if i.status == STATUS_OK)} answered)")
    print("  " + "=" * 74)
    marks = {STATUS_OK: "✓", STATUS_REFUSED: "✗ refused", STATUS_ERROR: "✗ error"}
    for i, it in enumerate(mr.items, 1):
        print(f"  [{i}] {marks.get(it.status, it.status)}  ({it.route})  {it.sub_query}")
        if it.status != STATUS_OK and it.refuse_reason:
            print(f"        → {it.refuse_reason}")


if __name__ == "__main__":
    sys.exit(main())
