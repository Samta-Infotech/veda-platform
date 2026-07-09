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

_SM = {}   # {(source_id, tenant): {"sm": dict, "cols": list}} — scope-keyed (P5)


def _emit(on_event, phase: str, message: str, **extra):
    """Best-effort progress callback for SSE streaming (never breaks the pipeline —
    a callback error must not sink the query it's merely reporting on)."""
    if on_event is None:
        return
    try:
        on_event(phase, message, extra)
    except Exception:
        pass


def _sm_scope():
    """(source_id, tenant) for the semantic-model cache/Redis key. Prefers the
    ambient per-request context (set by the inference middleware from headers),
    falling back to the env pin (single-source dev / bare-metal runs).

    `source_id` here is the PRIMARY source: the SQL head's per-source model is loaded
    from the primary today; the multi-source merge for the SQL head arrives with
    federated naming (Phase 5). The cache is keyed by the full scope (`_sm_cache_key`)
    so a `{A}` request and an `{A,B}` request never share an sm entry."""
    try:
        from context import try_current
        ctx = try_current()
        if ctx is not None:
            return (str(ctx.source_id), str(ctx.tenant))
    except Exception:
        pass
    return (os.environ.get("VEDA_SM_SOURCE_ID", "1"),
            os.environ.get("VEDA_SM_TENANT", "default"))


def _sm_cache_key():
    """Scope-unique key for the inference-tier `_SM` cache: the full source SET +
    tenant (P5), so distinct scopes over the same primary don't collide."""
    try:
        from context import try_current
        ctx = try_current()
        if ctx is not None:
            return (frozenset(int(s) for s in ctx.source_ids), str(ctx.tenant))
    except Exception:
        pass
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


def classify(query, verbose=False):
    """Return (intent, source_ids). Falls back to 'sql' if the router is off/unavailable
    — the deterministic SQL head is the safe default."""
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

    L0 — NL Simplifier runs first, here, so every consumer (CLI, inference API, demo)
    shares identical pre-processing regardless of how the query arrived. Previously
    only the CLI (main.py) applied this, so a query answered via the API could route
    differently than the same text answered via `main.py --query`."""
    try:
        from query.nl_simplifier import run_nl_simplifier
        _l0 = run_nl_simplifier(query, verbose=False)
        if _l0.was_simplified:
            print(f"  [L0] Simplified: '{_l0.simplified_query}' ({_l0.duration_ms}ms)")
            _emit(on_event, "simplify", f"Simplified query to: {_l0.simplified_query}")
        query = _l0.simplified_query
    except Exception:
        pass  # fall back to the original query silently

    try:
        from config import QUERY_DECOMPOSE_ENABLED
    except Exception:
        QUERY_DECOMPOSE_ENABLED = False

    if not QUERY_DECOMPOSE_ENABLED:
        route, res = _dispatch_single(query, verbose=verbose, on_event=on_event)
        return MultiResult(items=[_to_subresult(query, route, res)])

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
        if isinstance(det, dict) and det.get("ok"):
            sys.stdout.write(probe.getvalue())
            _emit(on_event, "answer", "Deterministic SQL answered the query")
            return MultiResult(items=[_to_subresult(query, "deterministic", det)])
        # Deterministic couldn't fully answer → maybe it was several questions.
        return _maybe_split(query, verbose=verbose, precomputed_sql=det,
                            probe_trace=probe.getvalue(), on_event=on_event)

    # RAG/hybrid/NoSQL self-certify nothing → decompose before dispatching (silent-drop guard).
    return _maybe_split(query, verbose=verbose, on_event=on_event)


def _maybe_split(query, verbose=False, precomputed_sql=None, probe_trace=None, on_event=None):
    """Run the decomposer, then either fan out independent sub-queries or fall back to
    the single-query pipeline. dependent_nested → refuse (out of scope for v1).

    probe_trace: the captured stdout of the deterministic probe (SQL intent only). Shown
    only on the single fallback (where it explains the refusal); discarded when we split
    or refuse-as-nested (there it would be a misleading 'couldn't answer' message)."""
    import io, contextlib
    from query.slm_layer import run_decomposer, DECOMP_DEPENDENT
    _emit(on_event, "decompose", "Checking whether this is a compound question...")
    # Capture the decomposer's own chatter so the on-screen order stays CHRONOLOGICAL. The
    # deterministic probe ran FIRST (its trace is in probe_trace); the decomposer runs AFTER.
    # Without capture, the decomposer prints live and appears BEFORE the replayed probe trace
    # — the scramble. We replay buffers in the order things actually happened.
    _dbuf = io.StringIO()
    with contextlib.redirect_stdout(_dbuf):
        decomp = run_decomposer(query, verbose=verbose)
    _decomp_trace = _dbuf.getvalue()

    if decomp.should_split:
        # Compound: the probe trace is a misleading "couldn't answer" for a query that was
        # simply several questions — suppress it; show the split decision + its reasoning.
        sys.stdout.write(_decomp_trace)
        print(f"\n  [Hybrid] compound query → {len(decomp.sub_queries)} independent sub-queries")
        _emit(on_event, "decompose", f"Split into {len(decomp.sub_queries)} sub-queries",
              sub_queries=list(decomp.sub_queries))
        return _fan_out(decomp.sub_queries, verbose=verbose, on_event=on_event)

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
    return MultiResult(items=[_to_subresult(query, route, res)])


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
        res = precomputed_sql if isinstance(precomputed_sql, dict) \
            else run_query(query, sm, cols, return_result=True)
        # Tier-2 fallback: if the deterministic head couldn't answer (refuse / dropped
        # qualifier / ungrounded / no table), let the LLM emit IR → deterministic
        # builder → GRAPH-GUARDED firewall → execute. Flag-gated (needs Ollama); the
        # graph guard (now live in the firewall) keeps LLM-proposed joins honest.
        if isinstance(res, dict) and not res.get("ok") and res.get("status") in (
                "refuse", "qualifier_dropped", "ungrounded", "no_table", "clarify",
                "exec_error"):
            try:
                from config import TIER2_LLM_FALLBACK
            except Exception:
                TIER2_LLM_FALLBACK = False
            if TIER2_LLM_FALLBACK:
                print("  [Tier2] deterministic head couldn't answer → LLM-IR fallback")
                _emit(on_event, "tier2", "Deterministic head couldn't answer — trying LLM-assisted SQL...")
                t2 = _tier2_sql(query, sm, cols, verbose=verbose)
                if t2 is not None:
                    _emit(on_event, "answer", "Tier-2 SQL answered the query")
                    return "deterministic", t2
        else:
            _emit(on_event, "answer", "SQL query executed")
        return "deterministic", res

    # ── RAG → integrated document retrieval + synthesis ───────────────────────
    if intent == "rag":
        from query.rag_layer import run_rag_layer
        _emit(on_event, "rag", "Retrieving relevant documents...")
        rag = run_rag_layer(query, source_ids=source_ids,
                            temporal_filter=_temporal(query), verbose=verbose)
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
        sqlres = run_query(query, sm, cols, return_result=True)
        sql_result = None
        if isinstance(sqlres, dict) and sqlres.get("ok"):
            _c, _r = sqlres.get("cols", []), sqlres.get("rows", [])
            sql_result = types.SimpleNamespace(
                columns=_c, rows=[dict(zip(_c, row)) for row in _r],
                row_count=len(_r), error=None)
        hy = run_hybrid_layer(query, sql_columns=[], source_ids=source_ids,
                             temporal_filter=_temporal(query),
                             sql_result=sql_result, verbose=verbose)
        if getattr(hy, "error", None):
            print(f"  [Hybrid] ✗ {hy.error}")
        else:
            print(f"\n  [Hybrid] {hy.answer}")
            _emit(on_event, "answer", "Fused SQL and document results into an answer")
        return "hybrid", hy

    # ── NoSQL → integrated native-query builder + execution ───────────────────
    if intent == "nosql":
        _emit(on_event, "nosql", "Querying document store...")
        result = _run_nosql(query, source_ids, verbose=verbose)
        _emit(on_event, "answer", "NoSQL query executed")
        return "nosql", result

    # ── default safety net ────────────────────────────────────────────────────
    sm, cols = _load_semantic_model()
    from veda.pipeline import run_query
    return "deterministic", run_query(query, sm, cols, return_result=True)


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
        # Tier-2 firewall/exec failures are infra errors; deterministic declines are refusals.
        status = STATUS_ERROR if st in ("tier2_rejected", "tier2_exec_error") else STATUS_REFUSED
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
    ok_q, missing = qualifier_completeness(query, raw_sql, sm)
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


def _tier2_sql(query, sm, all_cols, verbose=False):
    """Tier-2 SQL fallback (only when the deterministic head can't answer).

    LLM emits IR → deterministic sql_builder makes the SQL (LLM never writes SQL) →
    the GRAPH-GUARDED firewall validates (every join must be a real FK edge, no
    cartesian, value-grounded) → execute. Returns a result dict or None. Needs Ollama
    + the integrated retrieval stores; any failure → None (caller keeps the refusal)."""
    try:
        from query.retrieval_select import select_retrieval
        from query.slm_layer import run_slm_layer
        from query.sql_builder import run_sql_builder
        from veda.validation import validate_and_parameterize, value_grounding
        from veda.execution import execute_sql
        from query.temporal_parser import run_temporal_parser

        tf = run_temporal_parser(query).temporal_filter
        sel = select_retrieval(query=query, intent="sql", verbose=verbose)

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
                    else:
                        ecols, erows, eerr = execute_sql(psql, list(params))
                        if eerr:
                            print(f"  [Tier2] envelope exec error ({eerr}) — fallback to IR")
                        else:
                            print(f"  [Tier2] answered via ENVELOPE ({_qi.query_type}) — {len(erows)} rows")
                            _print_rows(ecols, erows, sql=psql)
                            return {"status": "answered", "ok": True, "cols": ecols,
                                    "rows": erows, "sql": psql, "source": "envelope"}
        except Exception as _ee:
            print(f"  [Tier2] envelope path skipped: {type(_ee).__name__}: {str(_ee)[:120]}")

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
        _repair_hint = None
        from config import LANGGRAPH_SHARED_PLANNER

        for _attempt in range(_max_repairs + 1):
            _q_ir = query if not _repair_hint else f"{query}\n\n{_repair_hint}"
            if _repair_hint:
                print(f"  [Tier2] repair attempt {_attempt}/{_max_repairs}")
            l3 = run_slm_layer(query=_q_ir, temporal_filter=tf, top_k_columns=sel.columns,
                               join_path=sel.join_path, verbose=verbose)
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
                act = build_from_entities(query, sm, all_cols, tf, ent_names[0], ent_names[1:])
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
                        if _attempt < _max_repairs:
                            _repair_hint = _repair_hint_for(eerr); continue
                        return {"status": "tier2_exec_error", "ok": False, "error": eerr}
                    print(f"  [Tier2] answered via SHARED planner (graph-verified joins)"
                          f"{' after repair' if _repair_hint else ''} — {len(rows)} rows")
                    _print_rows(cols, rows, sql=psql)
                    return {"status": "answered", "ok": True, "cols": cols, "rows": rows,
                            "sql": psql, "source": "tier2_shared_planner"}
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
            cols, rows, eerr = execute_sql(psql, list(params))
            if eerr:
                if _attempt < _max_repairs:
                    _repair_hint = _repair_hint_for(eerr); continue
                return {"status": "tier2_exec_error", "ok": False, "error": eerr}
            print(f"  [Tier2] answered via LLM-IR (graph-verified)"
                  f"{' after repair' if _repair_hint else ''} — {len(rows)} rows")
            _print_rows(cols, rows, sql=psql)
            return {"status": "answered", "ok": True, "cols": cols, "rows": rows,
                    "sql": psql, "source": "tier2"}
        return None   # repair attempts exhausted → keep the refusal
    except Exception as e:
        # Always surface WHY Tier-2 bailed (Ollama down, retrieval store missing, etc.) —
        # otherwise the path silently no-ops and looks like it never ran.
        print(f"  [Tier2] unavailable: {type(e).__name__}: {str(e)[:140]}")
        return None


def _run_nosql(query, source_ids, verbose=False):
    """Compact NoSQL path: resolve the source, infer schema, build + execute."""
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
            try:
                from config import NL_ANSWER_ENABLED
            except Exception:
                NL_ANSWER_ENABLED = False
            cols = getattr(res, "columns", None)
            rows = getattr(res, "rows", None)
            if NL_ANSWER_ENABLED and cols and rows is not None:
                try:
                    from query.nl_answer import run_nl_answer
                    row_dicts = [r if isinstance(r, dict) else dict(zip(cols, r)) for r in rows]
                    nl = run_nl_answer(query, list(cols), row_dicts)
                    if getattr(nl, "answer", None):
                        print(f"  [NoSQL] Answer  {nl.answer}")
                except Exception:
                    pass
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
