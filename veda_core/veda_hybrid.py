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
import importlib
from slm._call_slm import collect_usage, usage_totals
from veda.explain import new_trace, use_trace
from slm._call_slm import collect_usage as _collect_usage_l0, usage_totals as _usage_totals_l0
import io, contextlib
from query.slm_layer import run_decomposer, DECOMP_DEPENDENT
from slm._call_slm import collect_usage as _collect_usage_dc, usage_totals as _usage_totals_dc
from concurrent.futures import ThreadPoolExecutor
from veda_core.context import set_context as _set_ctx, try_current as _try_ctx
from veda.explain import (current_trace as _cur_trace, bind_trace as _bind_trace,
                          record_result_stages, render_trace)
# (value_grounding / qualifier_completeness now run inside veda.firewall — M3 checkpoint 1)
from veda.ir_equivalence import validate_ir_equivalence
import sqlglot
from sqlglot import exp
from connectors.base import build_connector
from query.nosql_builder import run_nosql_builder
from veda.rbac_filter import filter_nosql_collections
import types

_SM = {}   # {(source_id, tenant): {"sm": dict, "cols": list}} — scope-keyed (P5)


def _current_ctx():
    """The ambient RequestContext, read from whichever context MODULE holds it.

    The engine is imported both as bare `context` (cwd=veda_core) and as
    `veda_core.context` (inference tier, PYTHONPATH=/app) — Python loads these as TWO
    module objects with SEPARATE thread-locals. The inference middleware sets the scope
    on `veda_core.context`, so a read of bare `context` alone silently misses it and the
    SQL head falls back to source 1 (global model). Try both so the request scope is
    seen regardless of which name set it."""
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
    # MULTI-SOURCE scope: hand the SQL head the MERGED model, not the primary's alone.
    # The retrieval engine has always used the merge for a multi-source scope
    # (veda/runtime.py::_load_scoped_sm, via get_engine), so the two disagreed: retrieval
    # ranked columns across every source in scope while the SQL head could only plan
    # against ONE source's schema. A cross-source follow-up ("which properties do these
    # maintenance records belong to") therefore had no table for the other half of the
    # question and either refused or silently answered from the one schema it could see.
    # This is the merge veda_hybrid._sm_scope()'s own docstring anticipated.
    if entry is None:
        _ctx = _current_ctx()
        _ids = [str(s) for s in (getattr(_ctx, "source_ids", ()) or ())] if _ctx else []
        if len(_ids) > 1:
            try:
                from veda.runtime import _load_one_sm, _merge_scoped_sms
                _tenant = str(getattr(_ctx, "tenant", "default"))
                _merged = _merge_scoped_sms([(sid, _load_one_sm(sid, _tenant)) for sid in _ids])
                if _merged.get("tables"):
                    print(f"  [sm] multi-source scope {_ids} → merged model "
                          f"({len(_merged['tables'])} tables)")
                    entry = {"sm": _merged, "cols": list(_merged.get("columns", {}).keys())}
                    _SM[cache_key] = entry
            except Exception as _me:
                print(f"  [sm] multi-source merge failed ({type(_me).__name__}: {_me}) "
                      f"— falling back to the primary source's model")
    if entry is None:
        sm = _load_sm_from_redis(scope)
        if sm is None:
            # P0-5 (2026-09-11): prefer THIS source's own on-disk model over the
            # flat SEMANTIC_MODEL_FILE every source used to share — falls back to
            # the flat file automatically when no per-source copy exists yet
            # (resolve_source_artifact's own contract).
            from config import resolve_source_artifact
            sm_path = resolve_source_artifact("veda_semantic_model.json", scope[0], scope[1])
            if not sm_path or not os.path.exists(sm_path):
                # 2026-09-15: a source that is not the flat files' owner and has no scoped
                # semantic model yet (never ingested under the scoped pipeline, Redis copy
                # gone) now resolves to a MISSING path instead of another source's model
                # (config.resolve_source_artifact's owner rule). An empty model makes the
                # SQL head refuse with "no table" rather than plan SQL against a foreign
                # schema — the honest answer until the source is (re-)ingested.
                print(f"  [sm] no semantic model materialized for scope {scope} "
                      f"(expected {sm_path}) — empty model, SQL head will refuse")
                # `_not_materialized` lets the SQL head's empty-model refusal say the
                # TRUE reason ("not ingested yet") instead of misreporting it as an
                # RBAC access_denied — see the `if not sm.get("tables")` site below.
                sm = {"tables": {}, "columns": {}, "_not_materialized": True}
            else:
                with open(sm_path) as f:
                    sm = json.load(f)
        entry = {"sm": sm, "cols": list(sm.get("columns", {}).keys())}
        _SM[cache_key] = entry
    # NOTE (Gate 1, Task 16): this `sm` is the SAME object handed to
    # `get_engine(sm)` inside `veda.pipeline.run_query` — filtering it here would
    # bake whichever user's request misses the cache first into the shared,
    # per-scope retrieval engine (`_ENGINES` in veda.runtime), permanently, for
    # every other user in that scope. RBAC narrowing is applied downstream in
    # `run_query` instead, to the retrieval CANDIDATES (`filter_retrieval_results`
    # in `veda.rbac_filter`), never to this cached model. Stays byte-identical.
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


def _scope_has_structured_source() -> bool:
    """True when the request scope includes a source with structured columns (relational or
    tabular) — the mirror of ``_scope_has_doc_source`` (2026-09-15). The HYBRID lane (RAG ⊕
    deterministic SQL head) only makes sense when there is a structured source to run the
    SQL half against; a document-ONLY scope asked a count/total-shaped question used to be
    routed to HYBRID anyway, and the SQL head then retrieved columns from whatever semantic
    model the flat fallback served (another source's) and tried to execute against a source
    with no SQL endpoint at all. Found live on source 3 (`docs_contracts`)."""
    ctx = _current_ctx()
    sids = [str(s) for s in (getattr(ctx, "source_ids", ()) or ())] if ctx is not None else []
    if not sids:
        return False
    try:
        from config import BIENCODER_COL_TABLE
        from ingestion.db_abstraction import (
            get_internal_connection, release_internal_connection)
        conn = get_internal_connection()
        try:
            with conn.cursor() as cur:
                ph = ",".join(["%s"] * len(sids))
                cur.execute(f"SELECT 1 FROM {BIENCODER_COL_TABLE} WHERE source_id IN ({ph}) LIMIT 1",
                            sids)
                return cur.fetchone() is not None
        finally:
            release_internal_connection(conn)
    except Exception:
        return False


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


def _doc_intent_evidence_on() -> bool:
    """True when DOC_INTENT_EVIDENCE_ENABLED is set — classify() also consults the coordinator's
    cosine evidence for the sql-vs-rag decision (see config). Default OFF → byte-identical."""
    try:
        import config as _cfg
        return bool(getattr(_cfg, "DOC_INTENT_EVIDENCE_ENABLED", False))
    except Exception:
        return False


def _doc_intent_by_evidence(query) -> bool:
    """Data-driven doc-intent: reuse the SAME retrieval evidence + tiering the source coordinator uses,
    and return True when a DOCUMENT source is the dominant STRONG source (its chunk cosine is at least
    its own best column cosine, i.e. it is chunk-backed, and it tiers STRONG after dominance re-tiering).
    Catches document questions the fixed _DOC_REF_RE word list misses. No keywords. Flag-gated."""
    if not _doc_intent_evidence_on():
        return False
    ctx = _current_ctx()
    sids = [int(s) for s in (getattr(ctx, "source_ids", ()) or ())] if ctx is not None else []
    if not sids:
        return False
    try:
        import query.source_coordinator as _SC
        from query.source_evidence import group_evidence_by_source
        cols, chunks = _SC._default_evidence_provider(query, sids)
        ev = group_evidence_by_source(cols, chunks)
        if not ev:
            return False
        _SC._apply_item_prior(query, sids, ev)
        _SC._dominance_retier(ev)
        strong = [e for e in ev.values() if getattr(e, "presence_tier", "") == "STRONG"]
        if not strong:
            return False
        # the dominant STRONG source, ranked by its best raw cosine (chunk or column)
        best = max(strong, key=lambda e: max(getattr(e, "top_chunk_score", 0.0),
                                             getattr(e, "top_column_score", 0.0)))
        # chunk-backed (a document source) and not out-scored by its own column evidence
        return (getattr(best, "top_chunk_score", 0.0) > 0.0
                and getattr(best, "top_chunk_score", 0.0) >= getattr(best, "top_column_score", 0.0))
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
        # HYBRID needs a structured source to run its SQL half against (2026-09-15) — a
        # document-only scope stays on the RAG lane even for count/total wording; see
        # _scope_has_structured_source().
        intent = ("hybrid" if (_DB_AGG_RE.search(q) and _scope_has_structured_source())
                  else "rag")
        if verbose:
            print(f"  [router] doc-intent override → {intent}")
        return intent, None

    # Evidence-based doc-intent (flag-gated): catches document questions the fixed word list misses,
    # by consulting the coordinator's cosine evidence. OFF ⇒ this is False ⇒ byte-identical.
    if _doc_intent_by_evidence(q):
        # HYBRID needs a structured source to run its SQL half against (2026-09-15) — a
        # document-only scope stays on the RAG lane even for count/total wording; see
        # _scope_has_structured_source().
        intent = ("hybrid" if (_DB_AGG_RE.search(q) and _scope_has_structured_source())
                  else "rag")
        if verbose:
            print(f"  [router] doc-intent (evidence) → {intent}")
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


def _agent_to_subresult(query, ar):
    """AgentResult → SubResult, reusing the same status contract as _to_subresult. An OK result is
    re-shaped into the dict head-result form ({ok, cols, rows, answer, sql, ...}) so downstream
    explain/table rendering treats it exactly like a normal answer."""
    engine = getattr(ar, "engine", "") or "routed"
    status = getattr(ar, "status", "")
    if status == "ok":
        payload = {"ok": True, **(getattr(ar, "data", {}) or {})}
        # the answering source travels with the answer (P4, 2026-09-18) so the chat frame
        # can keep a drill-down on it
        if getattr(ar, "source_id", None) not in (None, "") and "source_id" not in payload:
            payload["source_id"] = ar.source_id
        return _to_subresult(query, engine, payload)
    if status == "refused":
        # Pass the agent's diagnostic payload through instead of None. It carries the pipeline's
        # own feedback/answer for this refusal (see agents.py::_from_sql_dict), and dropping it
        # left every consumer with nothing but the reason string — which is why a refused
        # datalake query surfaced as the contentless "Could you clarify what you're asking
        # about?" in chat. Empty dict → None, so a refusal with genuinely nothing to say behaves
        # exactly as before.
        return SubResult(query, STATUS_REFUSED, engine, (getattr(ar, "data", None) or None),
                         getattr(ar, "reason", "") or "refused")
    return SubResult(query, STATUS_ERROR, engine, None, getattr(ar, "error", "") or "failed")


def _multi_to_multiresult(query, out, on_event=None):
    """Convert an independent-MULTI execution (agents run separately, then merged) into a MultiResult,
    honouring the explicit merge policy and surfacing per-source partial failure.

    - CONFLICT_DETECTED (same metric differs, no canonical) → a single REFUSED result that surfaces the
      conflict; the values are NEVER silently blended.
    - CANONICAL_PRIORITY (conflict, one canonical) → the canonical source's answer wins.
    - APPEND (independent facts) → one SubResult PER source, preserving per-source identity + status;
      a failed source becomes an error/refused item, so an incomplete answer is visibly incomplete.
    """
    from query.result_orchestrator import (
        POLICY_CONFLICT_DETECTED, POLICY_CANONICAL_PRIORITY)
    merge = out.get("merge")
    results = out.get("results") or []
    partial = out.get("partial") or {}

    if merge is not None and merge.policy == POLICY_CONFLICT_DETECTED:
        vals = "; ".join(f"source {v['source_id']} = {v['value']}"
                         for v in (merge.conflict or {}).get("values", []))
        reason = ("Sources disagree on the same value and no canonical source is set — "
                  f"cannot choose safely: {vals}. Please clarify which source is authoritative.")
        _emit(on_event, "answer", reason)
        return MultiResult.single(query, STATUS_REFUSED, "conflict", refuse_reason=reason)

    if merge is not None and merge.policy == POLICY_CANONICAL_PRIORITY:
        winner = next((r for r in results
                       if getattr(r, "source_id", "") == merge.winner_source_id
                       and getattr(r, "status", "") == "ok"), None)
        if winner is not None:
            return MultiResult(items=[_agent_to_subresult(query, winner)])

    # APPEND (or canonical winner not found) — one item per source, failures labelled.
    items = [_agent_to_subresult(query, r) for r in results]
    seen = {getattr(r, "source_id", "") for r in results}
    for f in partial.get("failures", []):
        if f.get("source_id") not in seen:
            items.append(SubResult(query, STATUS_ERROR, "routed", None,
                                   f"source {f.get('source_id')}: {f.get('error')}"))
    if not items:
        return None
    # APPEND (independent facts from ≥2 sources): lead with ONE summary that states each
    # source's answer, so the reply reads as a single answer over several sources rather
    # than a list of disconnected blocks (2026-09-18). Composed by the small NL model from
    # the per-source answers ONLY (no new facts); deterministic fallback = the answers
    # joined, each labelled with its source. Per-source items stay for provenance/drill.
    _summary = None
    if len([it for it in items if it.status == STATUS_OK]) >= 2:
        try:
            _summary = _summarise_multi_answers(query, items)
        except Exception:
            _summary = None
    return MultiResult(items=items, summary=_summary)


def _with_summary(mr):
    """Attach MultiResult.summary when ≥2 items answered (compound fan-out / independent
    multi-source merge) — behaviour (c), 2026-09-18. The per-source items are untouched."""
    try:
        oks = [it for it in (getattr(mr, "items", None) or []) if it.status == STATUS_OK]
        if len(oks) >= 2 and not getattr(mr, "summary", None):
            q = " ; ".join(it.sub_query for it in oks)
            mr.summary = _summarise_multi_answers(q, oks)
    except Exception:
        pass
    return mr


def _summarise_multi_answers(query, items):
    """One sentence-or-three over the per-source answers (APPEND merge). The SLM sees only
    the answers already produced (with their source ids) and must not add figures; the
    numeric guard the explainer uses applies. Falls back to a labelled join."""
    parts = []
    for it in items:
        if it.status != STATUS_OK or not isinstance(it.result, dict):
            continue
        sid = it.result.get("source_id") or (it.result.get("sources") or [""])[0]
        ans = str(it.result.get("answer") or "").strip()
        if ans:
            parts.append((str(sid), ans))
    if len(parts) < 2:
        return None
    fallback = " ".join(f"From source {sid}: {ans}" for sid, ans in parts)
    try:
        from slm import call_slm
        from config import NL_SUMMARY_MAX_TOKENS
        from query.result_explainer import _nl_model
        prompt = ("Combine the following answers, each from a different data source, into ONE short "
                  "summary that answers the question. State each source's figure; do not invent or "
                  "compute new numbers; if they disagree, say so.\n\n"
                  f"Question: {query}\n" + "\n".join(f"- source {sid}: {ans}" for sid, ans in parts)
                  + "\n\nSummary:")
        out = call_slm(prompt, purpose="multi_summary", temperature=0.1,
                       num_predict=NL_SUMMARY_MAX_TOKENS + 50, endpoint="chat", model=_nl_model()).strip()
        import re as _re
        nums_in = set(_re.findall(r"\d[\d,]*\.?\d*", " ".join(a for _, a in parts)))
        nums_out = set(_re.findall(r"\d[\d,]*\.?\d*", out))
        if out and nums_out <= nums_in:      # no invented figures
            return out
    except Exception:
        pass
    return fallback


def _constrain_scope_to(source_id):
    """Narrow the ambient request scope to a SINGLE source (the coordinator's authoritative SINGLE
    decision). This makes `_maybe_federated` a no-op (it requires ≥2 in-scope sources) so a SINGLE
    decision can NEVER be silently overridden by a cross-source federated answer from a DIFFERENT
    source — the src_5.amenities_catalog mis-execution (docs/multisource_routing/ANSWER_E2E_ROOTCAUSE.md).
    Best-effort: any failure leaves the scope untouched (falls back to legacy behaviour). Only ever
    reached on the flag-gated authoritative-routing path, so production (flag OFF) is byte-identical."""
    try:
        ctx = _current_ctx()
        if ctx is None:
            return
        sid = int(source_id)
        # narrowed(): EVERY other field travels (allowed_resources, cache_back) — a fresh
        # RequestContext here dropped cache_back=False and the cross-source battery wrote
        # 17 verified-cache rows through the authoritative SINGLE route (2026-09-18)
        _set_ctx(ctx.narrowed(sid))
    except Exception:
        pass


def _is_datalake_source(source_id, decision, profiles):
    """True when the routed source is a tabular datalake (parquet/CSV) — its schema lives in
    column_embeddings_v2 + parquet files, NOT the relational semantic model."""
    sid = str(source_id)
    prof = (profiles or {}).get(sid, {}) or {}
    if str(prof.get("source_type", "")).lower() == "datalake":
        return True
    for c in getattr(decision, "candidate_sources", []) or []:
        if str(c.source_id) == sid and str(getattr(c, "source_type", "")).lower() == "datalake":
            return True
    return False


def _augment_sm_for_datalake(sm, cols, source_id):
    """Return a COPY of the semantic model with the datalake source's tables/columns merged in
    (from column_embeddings_v2). The relational sm knows only the primary DB, so a correct datalake
    SQL (e.g. `SELECT monthly_fee FROM amenities_catalog`) is otherwise rejected by the validator as
    "references unknown column(s)" and never reaches the DuckDB executor. Scoped to ONE source so it
    cannot bleed the datalake schema into an unrelated relational query. Best-effort: on any failure
    the original (sm, cols) is returned unchanged, so the caller degrades to today's behaviour."""
    try:
        from ingestion.db_abstraction import get_internal_connection, release_internal_connection
        conn = get_internal_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT table_name, col_name, semantic_type FROM column_embeddings_v2 "
                            "WHERE source_id = %s", [str(source_id)])
                rows = cur.fetchall()
        finally:
            release_internal_connection(conn)
        if not rows:
            return sm, cols
        new_cols = dict(sm.get("columns", {}) or {})
        new_tables = dict(sm.get("tables", {}) or {})
        # analytics_role is a DIFFERENT vocabulary from semantic_type and must be derived, not
        # copied. semantic_type is MONETARY / METRIC / CATEGORY / FREE_TEXT / TEMPORAL /
        # IDENTIFIER; analytics_role is MEASURE / DIMENSION / TIME_DIMENSION / IDENTIFIER /
        # ATTRIBUTE (see ingestion/deterministic_metadata.py::compute_analytics_role, the one
        # producer of it, and _SQL_USAGE which keys groupable/filterable/sortable off it).
        # Copying semantic_type in put strings like "MONETARY" in the role field — not a valid
        # role, so every consumer silently fell through _SQL_USAGE's ATTRIBUTE default:
        #   * monthly_fee/amount/rating (MONETARY/METRIC) were never seen as MEASUREs, so
        #     intent_sql_alignment.py:56's `analytics_role != "MEASURE": continue` skipped them
        #     and _wants_time_bucket read "monthly fee" as a per-month breakdown -> the
        #     temporal-alignment guard refused amenity-fee questions outright;
        #   * category/status (CATEGORY) were never DIMENSIONs, so they were not groupable.
        # Same derivation the relational ingestion path uses, so datalake columns now carry the
        # same role vocabulary as every other source.
        from ingestion.deterministic_metadata import compute_analytics_role
        for tbl, col, stype in rows:
            st = stype or "DIMENSION"
            new_cols[f"{tbl}.{col}"] = {
                "col_name": col, "table_name": tbl, "semantic_type": st,
                "analytics_role": compute_analytics_role(col, st),
                "business_definition": f"{col} from datalake dataset {tbl}"}
            new_tables.setdefault(tbl, {
                "table_name": tbl, "business_purpose": f"datalake dataset {tbl}",
                "primary_entity": tbl, "table_type": "datalake",
                "candidate_temporal_columns": [], "candidate_measure_columns": []})
        merged = dict(sm)
        merged["columns"] = new_cols
        merged["tables"] = new_tables
        return merged, list(new_cols.keys())
    except Exception:
        return sm, cols


def _run_doc_data(query, decision, profiles, on_event=None):
    """Bounded DOCUMENT_FACT + DATA_GROUNDING (flag-gated). For a MULTI spanning exactly one DOCUMENT
    source and one RELATIONAL data source: extract the entities the document names (grounded), query a
    candidate data column's values, and deterministically INTERSECT. Returns a MultiResult on success,
    or None to defer to the existing MULTI logic. Best-effort — any failure returns None."""
    try:
        from query.doc_data_planner import doc_data_enabled, classify, intersect
        if not doc_data_enabled():
            return None

        def _kind(sid):
            return str((profiles or {}).get(str(sid), {}).get("source_type", "")).lower()
        sids = [str(s) for s in (decision.source_ids or [])]
        doc = [s for s in sids if _kind(s) in ("document", "filesystem")]
        dat = [s for s in sids if _kind(s) == "relational"]
        if len(doc) != 1 or len(dat) != 1:
            return None                     # v1: exactly one doc + one relational data source
        doc_sid, dat_sid = doc[0], dat[0]

        # 1. document chunks
        from query.rag_layer import _encode_rag_query
        from ingestion.chunk_embedder import retrieve_top_k_chunks
        qv = _encode_rag_query(query)
        if qv is None:
            return None
        chunks = retrieve_top_k_chunks(query_vector=qv, source_ids=[doc_sid], top_k=6) or []
        chunk_texts = [getattr(c, "text", "") or "" for c in chunks]
        if not any(chunk_texts):
            return None

        # 2. candidate data columns (from retrieval over the data source)
        from query.retrieval_select import select_retrieval
        sel = select_retrieval(query=query, source_ids=[dat_sid], intent="sql", verbose=False)
        data_cols = []
        seen = set()
        for c in (getattr(sel, "columns", []) or []):
            t, col = getattr(c, "table_name", "") or "", getattr(c, "col_name", "") or ""
            if t and col and (t, col) not in seen:
                seen.add((t, col))
                data_cols.append({"source_id": dat_sid, "table": t, "col": col})
        if not data_cols:
            return None

        # 3. bounded SLM: grounded entities + chosen data column
        res = classify(query, chunk_texts, data_cols[:12])
        if res is None:
            return None
        entities, col = res

        # 4. query DISTINCT values of the chosen data column (existing execution path)
        from veda.execution import execute_sql
        sql = f'SELECT DISTINCT "{col["col"]}" FROM "{col["table"]}" WHERE "{col["col"]}" IS NOT NULL'
        _cols, rows, err = execute_sql(sql, [])
        if err or rows is None:
            return None
        data_values = [(r[0] if not isinstance(r, dict) else list(r.values())[0]) for r in rows]

        # 5. deterministic intersection
        matched = intersect(entities, data_values)
        cites = sorted({(getattr(c, "doc_name", "") or "") for c in chunks if getattr(c, "doc_name", "")})
        if matched:
            answer = (f"{len(matched)} of the items the document names are in our data: "
                      f"{', '.join(matched)}.")
        else:
            answer = ("None of the items the document names appear in our data "
                      f"(document named: {', '.join(entities)}).")
        _emit(on_event, "answer", answer)
        return MultiResult.single(query, STATUS_OK, "doc_data", result={
            "answer": answer, "rows": [[m] for m in matched], "cols": [col["col"]],
            "sql": sql, "citations": cites, "operation": "DOCUMENT_FACT_DATA_GROUNDING",
            "doc_entities": entities, "matched": matched})
    except Exception:
        return None


def _datalake_isolated_sm(source_id):
    """Build a DATALAKE-ONLY semantic model for a datalake SINGLE route (source isolation, flag-gated).

    Unlike _augment_sm_for_datalake (which MERGES the datalake schema into the homzhub sm, so the
    shared retrieval engine then mixes ~1900 homzhub tables into a datalake query), this returns an sm
    containing ONLY the routed source's tables/columns — plus on-demand parquet sample_values so the
    value-arbiter grounds datalake filter values (e.g. "Kochi") with NO homzhub column competing. The
    caller has already _constrain_scope_to(source), so get_engine() builds a SEPARATE per-scope engine
    (keyed on frozenset(source_ids)); the homzhub engine is never touched. Returns (sm, cols) or None
    on any failure (caller then falls back to the merge path)."""
    try:
        from ingestion.db_abstraction import get_internal_connection, release_internal_connection
        conn = get_internal_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT table_name, col_name, semantic_type FROM column_embeddings_v2 "
                            "WHERE source_id = %s", [str(source_id)])
                rows = cur.fetchall()
        finally:
            release_internal_connection(conn)
        if not rows:
            return None
        # Per-column parquet sample values (invert the datalake value sampler: token -> [(tbl,col,..,orig)]).
        col_samples = {}
        try:
            from query.datalake_values import _sample_source, _sample_limit
            _tenant = "default"
            _ctx = _current_ctx()
            if _ctx is not None:
                _tenant = str(getattr(_ctx, "tenant", "default"))
            vidx = _sample_source(str(source_id), _tenant, _sample_limit()) or {}
            for _tok, hits in vidx.items():
                for (tbl, col, _st, orig) in hits:
                    col_samples.setdefault((tbl, col), [])
                    if orig not in col_samples[(tbl, col)]:
                        col_samples[(tbl, col)].append(orig)
        except Exception:
            col_samples = {}
        tables, columns = {}, {}
        # Derive analytics_role rather than copying semantic_type — see the same fix in
        # _augment_sm_for_datalake above for why (the two vocabularies are not interchangeable,
        # and an invalid role silently degrades to ATTRIBUTE).
        from ingestion.deterministic_metadata import compute_analytics_role
        for tbl, col, stype in rows:
            st = stype or "DIMENSION"
            svals = col_samples.get((tbl, col), [])
            columns[f"{tbl}.{col}"] = {
                "col_name": col, "table_name": tbl, "semantic_type": st,
                "analytics_role": compute_analytics_role(col, st),
                "business_definition": f"{col} from datalake dataset {tbl}",
                "sample_values": svals}
            tables.setdefault(tbl, {
                "table_name": tbl, "business_purpose": f"datalake dataset {tbl}",
                "primary_entity": tbl, "table_type": "datalake",
                "candidate_temporal_columns": [], "candidate_measure_columns": []})
        iso = {"version": 1, "tables": tables, "columns": columns,
               "retrieval_documents": {}, "domain_synonyms": {}, "concept_graph": {},
               # Marker read ONLY by the retrieval engine's Signal-1 isolation filter (retrieve()):
               # the shared BGE searcher queries the GLOBAL column store and is the one signal not
               # bound to this per-engine sm, so it must be filtered back to this source's columns.
               # Present only on this flag-gated isolated sm → the normal path never sees it.
               "__source_isolated__": True}
        return iso, list(columns.keys())
    except Exception:
        return None


from contextvars import ContextVar as _CtxVar
# set by _run_coordinator when a MULTI decision on a compound question is handed to the
# decomposer; read by run_hybrid_query to skip the legacy federate-first call (behaviour c)
_COMPOUND_HANDOFF: "_CtxVar[bool]" = _CtxVar("veda_compound_handoff", default=False)


def _run_coordinator(query, verbose=False, on_event=None):
    """Multi-source routing coordinator entry (Phase 3.6 + authoritative wiring).

    Off  (MULTISOURCE_ROUTING_ENABLED=0)  → returns None, no work (prod byte-identical).
    On + SHADOW                           → computes + traces a RoutingDecision, returns None
                                            (answer path unchanged — observe only).
    On + not SHADOW (authoritative)       → the decision DRIVES the answer:
        NO_MATCH / CLARIFICATION_REQUIRED → a refusal MultiResult (NO answer is generated — closes
                                            the 'no silent guessing' gap).
        SINGLE                            → dispatch via the source agent, mapped to a MultiResult.
        MULTI                             → returns None so the existing federated path handles it.
    Always best-effort: any failure returns None and the legacy path proceeds.
    """
    try:
        from config import MULTISOURCE_ROUTING_ENABLED, MULTISOURCE_ROUTING_SHADOW
    except Exception:
        return None
    if not MULTISOURCE_ROUTING_ENABLED:
        return None
    try:
        ctx = _current_ctx()
        sids = [str(s) for s in (getattr(ctx, "source_ids", ()) or ())] if ctx is not None else []
        if not sids:
            return None
        try:
            from veda_core.context import current_source_profiles as _csp
        except Exception:
            try:
                from context import current_source_profiles as _csp
            except Exception:
                _csp = lambda: {}
        _profiles = _csp() or {}
        _tenant = str(getattr(ctx, "tenant", "default"))
        try:
            # The scope reaching here is ALREADY RBAC-narrowed by the api tier
            # (resolve_query_scope -> permitted_source_ids), so a non-empty scope is
            # a positive access fact worth showing. Says only THAT access was
            # verified — never which sources, and never what was excluded.
            from veda import lifecycle as _lc1
            # STARTED, not completed: at scope-resolution time we know the caller
            # has *a* permitted scope, but not yet whether narrowing will block
            # THIS query — that only surfaces after narrow_allowed + validation.
            # Claiming "Verified your access" here and then refusing on access
            # grounds was a visible contradiction in the live stream. The phase is
            # completed in pipeline.py once validation actually passes.
            _lc1.current_timeline().started(_lc1.PHASE_ACCESS_CHECK)
        except Exception:
            pass
        from query.source_coordinator import plan_route, execute_decision
        _emit(on_event, "route", "Deciding which source can answer…")

        # Permission-aware routing pre-check (flag-gated, default OFF). Decide the best source over ALL
        # ready sources; if the strict winner is one the user has NO access to, refuse with a clear
        # permission message rather than mis-routing to a weaker permitted source. Only match SCORES of
        # the inaccessible source are read (pre-computed vectors) — its content is never fetched.
        try:
            from config import ROUTING_PERMISSION_PRECHECK_ENABLED as _perm_pc
        except Exception:
            _perm_pc = False
        if _perm_pc:
            try:
                from query.source_coordinator import (all_ready_source_ids,
                                                      best_matching_scored)
                _permitted = {str(s) for s in sids}
                _denied = set(all_ready_source_ids()) - _permitted
                if _denied:
                    _all_hit = best_matching_scored(query, sorted(_permitted | _denied), _profiles)
                    _best = _all_hit[0] if _all_hit is not None else None
                    # ...but WHICH source won is not enough on its own, and deciding on identity
                    # alone failed in both directions:
                    #   * refuse whenever the global winner is inaccessible → a datalake-only
                    #     caller asking "How many maintenance records are repairs?" was told
                    #     "you don't have permission", because maintenance_policy.docx (source 3,
                    #     denied) matched best — even though the permitted `maintenance` dataset
                    #     answers it directly;
                    #   * proceed whenever ANY permitted source is STRONG → homzhub (178 tables)
                    #     is STRONG for almost any wording, so a homzhub-only caller asking
                    #     "Which city has the highest-rated vendor?" got a bare "could you
                    #     clarify", never learning that the answer exists but is not theirs.
                    # So compare the two winners' SCORES. When the inaccessible source is ahead by
                    # the same margin the router itself calls dominant (ROUTING_DOMINANT_GAP, the
                    # gap _decision_boundary uses for a confident SINGLE), the caller genuinely has
                    # no access to the data that answers this → refuse WITH the permission message.
                    # When a permitted source is within that margin it is a real answer, so answer
                    # it. Costs one extra scoring pass, and only on this rare path (some ready
                    # source is inaccessible AND wins).
                    #
                    # The margin is its OWN knob, not ROUTING_DOMINANT_GAP, because the two
                    # populations overlap and the trade-off is genuinely tight. Measured on this
                    # deployment (item-prior scores, permission-agnostic scoring):
                    #   datalake-only caller, MUST NOT refuse (it is entitled):
                    #     "How many maintenance records are repairs?"          gap 0.105
                    #     "Show amenities in the Security category."           gap 0.033
                    #   homzhub-only caller, SHOULD refuse with the permission message:
                    #     "Which city has the highest-rated vendor?"           gap 0.215
                    #     "Show vendors with a rating above 4.2."              gap 0.142
                    #     "Show unpaid maintenance tickets with their IDs."    gap 0.089
                    #     "Show every maintenance ticket ..."                  gap 0.047
                    # No single cutoff satisfies both ends (0.105 sits above 0.047-0.089), so this
                    # is deliberately set ABOVE the highest must-not-refuse case: never tell an
                    # entitled caller "you don't have permission" — that is the harmful error —
                    # and accept that a narrow-margin unauthorised case falls through to the
                    # pipeline's own (vaguer) refusal instead. It still refuses, and still leaks
                    # nothing; it just says less. Raising this only makes it more conservative.
                    # A principled fix needs more than a scalar gap (e.g. domain/type agreement
                    # between the query and the winning source) and is not attempted here.
                    _perm_hit = (best_matching_scored(query, sorted(_permitted), _profiles)
                                 if (_best is not None and str(_best) in _denied) else None)
                    try:
                        from config import ROUTING_PERMISSION_DENY_GAP as _deny_gap
                    except Exception:
                        _deny_gap = 0.12
                    if (_best is not None and str(_best) in _denied
                            and (_perm_hit is None
                                 or (_all_hit[1] - _perm_hit[1]) >= float(_deny_gap))):
                        # Deliberately says nothing about WHICH source, or that a
                        # source able to answer exists at all: naming it tells an
                        # unauthorised caller what data the platform holds. Same
                        # wording as veda/feedback.py's ACCESS_DENIED_WHY/_WHAT, the
                        # canonical denial text every other path already uses.
                        _deny_msg = ("You don't have permission to access this data. "
                                     "Contact your Admin to request access.")
                        # RESOLVE access_check NEGATIVELY before returning.
                        #
                        # Regression re-fix: this emit existed on this branch and was
                        # LOST when the branch was rewritten for the deny-gap logic
                        # (the merge diff shows the removed line). Without it the
                        # timeline keeps the optimistic `started`/`completed` from
                        # scope-resolution time, so a real denial shipped as
                        # "Checking access permissions ✓ — You have permission to
                        # access the required information" directly above an answer
                        # saying "You don't have permission to access this data".
                        # Telling someone they have permission they do not have is
                        # the worst thing this layer can get wrong, so the emit
                        # belongs next to the denial it describes, not in a sweep.
                        try:
                            from veda import lifecycle as _lc2
                            _lc2.current_timeline().failed(_lc2.PHASE_ACCESS_CHECK)
                        except Exception:
                            pass
                        try:
                            from veda import warnings as _vw2
                            _vw2.add(_vw2.RESTRICTED_DATA)
                        except Exception:
                            pass
                        _emit(on_event, "answer", _deny_msg)
                        # Built AFTER the warning above, so the payload carries it.
                        # Without this the denial arrived with every block empty and
                        # no `support.trace_id` — the one thing a user needs to hand
                        # to support about a permission problem (measured live).
                        _deny_ex = _explain_from_trace_only()
                        return MultiResult.single(
                            query, STATUS_REFUSED, "no_access",
                            refuse_reason=_deny_msg,
                            result=({"explain": _deny_ex} if _deny_ex else None))
            except Exception:
                pass

        # The conversation's previous scope, stated to the boundary SLM as a prior. This
        # is what keeps a WIDENED session scope from drifting: the chat tier widens so a
        # cross-source follow-up can reach another source, and this tells routing which
        # source the thread is actually on so it only moves when the question warrants it.
        _prior = None
        try:
            _pids = [str(s) for s in (getattr(ctx, "session_prior", ()) or ())]
            if _pids:
                _prior = {"source_ids": _pids,
                          "anchor": str(getattr(ctx, "session_anchor", "") or "") or None}
        except Exception:
            _prior = None
        decision = plan_route(query, sids, profile_provider=lambda _s: _profiles,
                              prior=_prior)
        try:  # user-safe source-selection event, from the decision the router made
            from veda import lifecycle as _lc3
            _tl3 = _lc3.current_timeline()
            if decision.status == "ROUTED":
                _tl3.completed(_lc3.PHASE_SOURCE_SELECTION,
                               source_count=len(decision.source_ids or []))
            elif decision.status == "NO_MATCH":
                _tl3.failed(_lc3.PHASE_SOURCE_SELECTION)
            else:
                _tl3.warning(_lc3.PHASE_SOURCE_SELECTION)
        except Exception:
            pass

        # Scoped authoritative rollout (2026-09-10, see docs/backlog/query-engine-open-items.md):
        # live-testing MULTISOURCE_ROUTING_SHADOW=0 unscoped against this deployment's data showed
        # the coordinator's OWN routing-evidence check (a plain cosine lookup, decoupled from the
        # real answer engine's retrieval on purpose) hard-refusing plain single-source questions
        # that veda/pipeline.py::run_query answers fine on its own — it was gating a strictly more
        # capable engine with a strictly weaker signal. A genuine cross-source MULTI decision does
        # not have that failure mode (there is no single-source legacy answer to defeat), and is
        # exactly the case the coordinator was built for (structural join detection, doc+data
        # grounding, federated execution). So: MULTI decisions are authoritative whenever routing
        # is enabled at all, independent of SHADOW; every other decision (SINGLE / NO_MATCH /
        # CLARIFICATION_REQUIRED / anything else) stays gated by SHADOW as before. SHADOW=1 in
        # .env today — flip to 0 only once a SINGLE-vs-legacy-engine regression test exists.
        _is_multi_decision = decision.status == "ROUTED" and decision.mode == "MULTI"
        # 2026-09-18: a ROUTED/SINGLE decision is authoritative too. The condition the 09-10
        # note set for this ("a SINGLE-vs-legacy-engine regression test") exists now —
        # scripts/eval_cross_source_battery.py — and the routing evidence it was gated on has
        # been fixed to be per-source and kind-fair (source_coordinator, 3/12 → 10/12 on the
        # probe). A SINGLE route runs the SAME pinned path the source's own battery runs, so
        # an unpinned single-source question can no longer land in the federated planner.
        # NO_MATCH / CLARIFICATION_REQUIRED stay advisory under SHADOW: the legacy engine is
        # strictly more capable than "no source", so they fall through instead of refusing.
        _is_single_decision = decision.status == "ROUTED" and decision.mode == "SINGLE"
        _effective_shadow = (bool(MULTISOURCE_ROUTING_SHADOW)
                             and not _is_multi_decision and not _is_single_decision)

        try:
            # slm_consulted / slm_decision_discarded make the routing SLM's COST
            # auditable against its effect. The plan for this pass was to skip the
            # source_routing call entirely under SHADOW because "the decision will be
            # discarded" — that was true before 2026-09-18, but is not true now: the
            # scoped-authoritative wiring just above makes BOTH MULTI and SINGLE
            # decisions authoritative regardless of SHADOW, and whether a decision is
            # one of those is only knowable AFTER the call. Skipping it would not save
            # a wasted call, it would delete the routing. The genuinely discarded case
            # is narrow — an SLM-decided NO_MATCH/CLARIFICATION_REQUIRED under
            # SHADOW=1 — so it is measured here instead of guessed at.
            _slm_used = decision.decision_method == "slm"
            _cur_trace().set(
                "routing", status=decision.status, mode=decision.mode,
                source_ids=decision.source_ids, reason_code=decision.reason_code,
                decision_method=decision.decision_method,
                slm_consulted=_slm_used,
                slm_decision_discarded=bool(_slm_used and _effective_shadow),
                shadow=_effective_shadow, shadow_flag=bool(MULTISOURCE_ROUTING_SHADOW))
        except Exception:
            pass
        if verbose:
            print(f"  [routing] {decision.status}/{decision.mode} sources={decision.source_ids} "
                  f"({decision.reason_code}){' [shadow]' if _effective_shadow else ''}"
                  f"{' [multi-authoritative]' if _is_multi_decision and MULTISOURCE_ROUTING_SHADOW else ''}")

        if _effective_shadow:
            return None   # observe only

        # ---- authoritative: the decision drives the answer ----
        if decision.status in ("NO_MATCH", "CLARIFICATION_REQUIRED"):
            route = "no_match" if decision.status == "NO_MATCH" else "clarify"
            _emit(on_event, "answer", decision.reason or "Could not determine a source.")
            return MultiResult.single(query, STATUS_REFUSED, route, refuse_reason=decision.reason)

        if decision.status == "ROUTED" and decision.mode == "SINGLE":
            _sid = decision.source_ids[0] if decision.source_ids else None
            _stype = str((_profiles.get(str(_sid), {}) or {}).get("source_type", "") or "").strip()
            _emit(on_event, "route",
                  f"Routing to the {_stype} source…" if _stype else "Routing to the matched source…",
                  source_ids=decision.source_ids, mode="single")
            _is_dl = bool(_sid) and _is_datalake_source(_sid, decision, _profiles)
            if _is_dl:
                # Datalake SINGLE route: constrain retrieval to this source (so a datalake dataset
                # whose name overlaps a relational concept — e.g. "maintenance" — does not bleed into
                # DB tables and produce an impossible cross-source join), and make its parquet columns
                # known to the SQL validator. See docs/multisource_routing/ANSWER_E2E_ROOTCAUSE.md.
                _constrain_scope_to(_sid)
            sm, cols = _load_semantic_model()
            if _is_dl:
                # Source isolation (flag-gated, default ON since config.py:604 — changed from the
                # original OFF default after it proved safe): run over a DATALAKE-ONLY sm so
                # retrieval/planning/validation/value-grounding see ONLY this source (no
                # homzhub-table mixing, no shared-value collision). On OFF or any failure, fall
                # back to the merge path below (_augment_sm_for_datalake) — byte-identical to the
                # pre-isolation behaviour.
                _iso = None
                try:
                    from config import SOURCE_ISOLATED_RETRIEVAL_ENABLED as _iso_on
                except Exception:
                    _iso_on = False
                if _iso_on:
                    _iso = _datalake_isolated_sm(_sid)
                if _iso is not None:
                    sm, cols = _iso
                else:
                    sm, cols = _augment_sm_for_datalake(sm, cols, _sid)
            out = execute_decision(decision, query, sm=sm, cols=cols, tenant=_tenant,
                                   profiles=_profiles, on_event=on_event)
            ar = (out or {}).get("result")
            if ar is None:
                # Authoritative SINGLE[s], but the source agent produced no result. Do NOT fall
                # through to the cross-source federated path — it would answer from a DIFFERENT
                # source (the src_5.amenities_catalog mis-execution). Constrain the scope to the
                # routed source so only the single-source legacy path can answer.
                if decision.source_ids:
                    _constrain_scope_to(decision.source_ids[0])
                return None
            return MultiResult(items=[_agent_to_subresult(query, ar)])

        if decision.status == "ROUTED" and decision.mode == "MULTI":
            # Behaviour (c), 2026-09-18: a COMPOUND question over several sources ("how many
            # maintenance records and how many amenities are there") is not a join — it is
            # one sub-question per source. Let the decomposer split it (the normal path
            # below _run_coordinator) so each part routes SINGLE on its own and the fan-out
            # returns one answer per source plus a summary. Federation stays for questions
            # that genuinely relate the sources. Measured: this question went to the
            # federated planner and came back "4 … and an average of 10.88" (truth 5 and 7).
            try:
                _dc = run_decomposer(query, verbose=False)
                if getattr(_dc, "should_split", False) and len(getattr(_dc, "sub_queries", []) or []) >= 2:
                    if verbose:
                        print(f"  [routing] MULTI but compound ({len(_dc.sub_queries)} parts) → "
                              f"decompose; each part routes on its own")
                    try:
                        _cur_trace().set("routing", compound=True, parts=list(_dc.sub_queries),
                                         handoff="decomposer")
                    except Exception:
                        pass
                    # tell run_hybrid_query to skip the legacy federate-first call and go
                    # straight to the decomposer (contextvar: request-scoped, thread-safe)
                    _COMPOUND_HANDOFF.set(True)
                    return None
            except Exception:
                pass
            _emit(on_event, "route",
                  f"Combining data across {len(decision.source_ids)} sources…",
                  source_ids=decision.source_ids, mode="multi")
            # DOCUMENT_FACT + DATA_GROUNDING (flag-gated, default OFF). A doc+data MULTI can't federate
            # (a document isn't column-bearing); the existing path merges the two independently and
            # can't intersect. Try the bounded grounding first; on None fall through unchanged.
            _dd = _run_doc_data(query, decision, _profiles, on_event=on_event)
            if _dd is not None:
                return _dd
            # Genuine join (cross_source_fk edge) → strategy 'federated': defer to the existing
            # federated route (_maybe_federated), which already builds cross-source SQL + a MultiResult.
            # No edge (SLM-resolved) → strategy 'independent': run each source and merge/conflict here.
            from query.execution_planner import plan_execution, STRATEGY_INDEPENDENT
            plan = plan_execution(decision)
            if plan.strategy != STRATEGY_INDEPENDENT:
                _emit(on_event, "route", "Joining data across sources…", mode="federated")
                # Genuine join (cross_source_fk edge). Run the federated route in STRICT mode: a
                # failure is a surfaced controlled failure of a required join, never a silent
                # single-source fallback (gaps #2/#3). None → federation genuinely not applicable
                # (retrieval didn't span sources) → defer to the legacy path.
                return _maybe_federated(query, verbose=verbose, strict=True)
            # SLM-resolved MULTI (no structural edge): the planner defaults to 'independent' (run each
            # source + merge), which CANNOT join — so a genuine cross-source query fails. The federated
            # executor self-discovers join hints independently of the routing edge (verified: it answers
            # these directly), so try it FIRST (non-strict); on None fall back to independent-merge.
            # Flag-gated, default OFF → byte-identical.
            try:
                from config import FEDERATE_SLM_MULTI_ENABLED as _fed_slm
            except Exception:
                _fed_slm = False
            if _fed_slm:
                fed = _maybe_federated(query, verbose=verbose, strict=False)
                if fed is not None:
                    return fed
            sm, cols = _load_semantic_model()
            out = execute_decision(decision, query, sm=sm, cols=cols, tenant=_tenant,
                                   profiles=_profiles, on_event=on_event)
            if not out or out.get("kind") != "independent":
                return None
            return _multi_to_multiresult(query, out, on_event=on_event)

        # anything else → legacy path
        return None
    except Exception as _e:
        if verbose:
            print(f"  [routing] skipped ({type(_e).__name__}: {_e})")
        return None


def _maybe_federated(query, verbose=False, strict=False):
    """If the request scope spans ≥2 sources, try the cross-source federated route.
    Returns a MultiResult on a federated answer/refusal, or None to use the normal path.

    ``strict`` (routing gaps #2/#3): set by the routing coordinator when a cross_source_fk edge was
    DETERMINED (a genuine join). Then a federation failure is a CONTROLLED failure of a required
    join and is SURFACED (with the involved sources + transient/permanent class), never silently
    degraded to a single-source answer that would drop a source. Default False = legacy behaviour."""
    ctx = _current_ctx()
    sids = list(getattr(ctx, "source_ids", ()) or ()) if ctx is not None else []
    if len(sids) < 2:
        return None
    _fed_t0 = time.time()
    _fed_calls = []
    try:
        with collect_usage() as _fed_usage:
            from query.federated_route import run_federated
            from query.reliability import execute_federated_reliably
            # Bounded transient-retry (flag-gated, default-OFF): same hardening the coordinator's
            # _federated_delegate applies — this direct call-site must not be the one branch that
            # skips it. OFF → single pass-through, byte-identical.
            payload = execute_federated_reliably(
                lambda: run_federated(query, tenant=str(getattr(ctx, "tenant", "default")),
                                      source_ids=sids, verbose=verbose))
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
            # Record WHICH sources the federation ran over, then build WITH the
            # trace. Both were missing: this composer called build_explain with no
            # trace at all, so no v2 extension was assembled — and nothing wrote
            # the federation section, so even a trace would have had no source
            # evidence to project. A real cross-source answer therefore shipped
            # `sources: null` and `execution: {"type": "sql"}` while its own text
            # compared figures from two sources (measured live: "There are 7
            # invoices compared to 96 assets").
            #
            # `catalogs` is the strongest proof available — the catalogs the
            # EXECUTED SQL actually referenced, as returned by the federated
            # executor (`src_<id>`). `payload["sources"]`, the source ids among
            # the selected columns, is the fallback for the shapes that carry no
            # catalog list.
            _tr_f, _tid_f = None, ""
            try:
                from veda.explain import current_trace as _ct_f
                _tr_f = _ct_f()
                if _tr_f is not None and getattr(_tr_f, "enabled", False):
                    _cats = [str(c)[4:] for c in (r.get("catalogs") or [])
                             if str(c).startswith("src_")]
                    _fids = _cats or [str(x) for x in (payload.get("sources") or [])]
                    if _fids:
                        _tr_f.set("federation", used=True, operation="combined",
                                  source_ids=_fids, result_status="complete")
                    _tid_f = getattr(_tr_f, "trace_id", "") or ""
                else:
                    _tr_f = None
            except Exception:
                _tr_f = None
            explain = build_explain(sql=_fed_sql or "", table="federated", sm=None,
                                    trace=_tr_f, trace_id=_tid_f)
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
        if strict:
            # Authoritative MULTI: a cross_source_fk edge was determined, so these sources genuinely
            # join and are all required — a federation failure is a CONTROLLED failure, surfaced with
            # its sources + class, NOT a silent single-source fallback (routing gaps #2/#3).
            if verbose:
                print(f"  [federated/strict] failed ({_reason[:80]}) — surfacing controlled failure")
            result = {"ok": False, "status": "federated_failed",
                      "error": payload.get("reason") or "federation failed",
                      "sources": payload.get("sources"),
                      "failure_class": payload.get("failure_class"),
                      "retryable": payload.get("retryable"),
                      "usage": _fed_usage_totals, "latency_ms": _fed_latency_ms}
            return MultiResult(items=[_to_subresult(query, "federated", result)])
        # M3 checkpoint 1: a plan the FIREWALL refused (ungrounded literal, dropped IR
        # slot, out-of-scope table) is a real, explained refusal — never a silent
        # single-source fallback that would answer a different question.
        if isinstance(payload.get("firewall"), dict) and payload["firewall"].get("verdict") not in (None, "ok"):
            if verbose:
                print(f"  [federated] firewall {payload['firewall'].get('verdict')} — surfacing refusal")
            result = {"ok": False, "status": "federated_refused",
                      "error": payload.get("reason") or "federation refused", "sql": payload.get("sql"),
                      "firewall": payload.get("firewall"),
                      "usage": _fed_usage_totals, "latency_ms": _fed_latency_ms}
            try:
                _cur_trace().set("firewall", **payload["firewall"])
            except Exception:
                pass
            return MultiResult(items=[_to_subresult(query, "federated", result)])
        if verbose:
            print(f"  [federated] plan failed ({_reason[:80]}) — falling back to single-source")
        try:
            _cur_trace().set("federated", status=payload.get("status"), reason=str(_reason)[:200],
                             fallback="single_source")
        except Exception:
            pass
        return None
    # refused/blocked federation is a real, explained outcome — surface it, don't silently
    # fall back to a single-source answer that would drop a source.
    result = {"ok": False, "status": "federated_refused",
              "error": payload.get("reason") or "federation refused", "sql": payload.get("sql"),
              "firewall": payload.get("firewall"),
              "usage": _fed_usage_totals, "latency_ms": _fed_latency_ms}
    try:
        if isinstance(payload.get("firewall"), dict):
            _cur_trace().set("firewall", **payload["firewall"])
    except Exception:
        pass
    return MultiResult(items=[_to_subresult(query, "federated", result)])


def _clean_refuse_on_empty_error(result) -> None:
    """A terminal 'error' that produced NEITHER an answer NOR SQL is a "couldn't answer this" — surface
    a clean refusal instead of a bare error. Evidence-of-failure only (no answer + no SQL); never probes
    a source the user cannot access, so no existence disclosure. Flag-gated, default ON; a no-op on any
    item that has an answer or SQL (real results are untouched)."""
    try:
        import config as _cfg
        if not bool(getattr(_cfg, "WEAK_EVIDENCE_CLEAN_REFUSE_ENABLED", True)):
            return
    except Exception:
        return
    _msg = "I couldn't find any data relevant to this question in the sources available to you."
    # An INFRASTRUCTURE failure must not be repainted as "the data isn't there". "no answer and
    # no SQL" is also exactly what an unreachable LLM host produces, and blaming the data then
    # sends the user — and whoever debugs it — hunting a retrieval or permission problem that
    # does not exist. Observed today: with the SLM host refusing connections, every SQL query
    # came back "I couldn't find any data relevant to this question" while the rows sat right
    # there. So reuse the engine's OWN transient/permanent classifier (query/reliability.py, the
    # same one execute_reliably retries on) and say what actually happened when it is transient.
    _infra_msg = ("I couldn't reach the service needed to answer this, so I don't have an answer "
                  "yet. Please try again in a moment.")

    def _is_transient(err) -> bool:
        if not err:
            return False
        try:
            from query.reliability import classify_failure
            return classify_failure(str(err)) == "transient"
        except Exception:
            _e = str(err).lower()
            return any(m in _e for m in ("timeout", "timed out", "504", "unreachable",
                                         "connection refused", "connectionerror"))

    try:
        for it in (getattr(result, "items", None) or []):
            if getattr(it, "status", "") != "error":
                continue
            r = it.result if isinstance(getattr(it, "result", None), dict) else None
            if r is not None and (r.get("answer") or r.get("sql")):
                continue                                   # a real result — leave it alone
            _err = getattr(it, "refuse_reason", None) or (r or {}).get("error")
            _transient = _is_transient(_err)
            _text = _infra_msg if _transient else _msg
            _reason = "engine_unavailable" if _transient else "no_relevant_data"
            it.status = "refused"
            it.refuse_reason = getattr(it, "refuse_reason", None) or _reason
            if r is not None:
                r["ok"] = False
                r["status"] = "refused"
                r["answer"] = r.get("answer") or _text
            else:
                it.result = {"ok": False, "status": "refused", "answer": _text,
                             "refuse_reason": _reason}
    except Exception:
        pass


def _emit_terminal_lifecycle(timeline, final_status: str) -> None:
    """Close the timeline with the phase that matches the actual outcome.

    Only emitted for a terminal state we can describe safely: `answered` completes,
    anything else is a warning on result_preparation rather than a failure, because
    a refusal is a CORRECT outcome (the refuse-over-guess contract) and must not be
    presented to the user as the system breaking."""
    try:
        from veda import lifecycle as lc
        if not getattr(timeline, "enabled", False):
            return
        # EXP-B1: close any phase left OPEN (started, never resolved). Measured on the
        # 10-query benchmark: access_check hung at "started" on 6 of 10 query types,
        # because its completion lived on the deterministic SQL path only — document,
        # hybrid and refusal paths never reach it, so the user watched a spinner that
        # never finished. Resolving here means EVERY path closes it, whichever head ran.
        #
        # A phase already resolved NEGATIVELY is left alone: a real permission denial
        # emits access_check=failed earlier (pipeline._feedback), and overwriting that
        # with "verified" would be the contradiction this whole fix is about.
        # Shared with veda/pipeline.py::_done, which sweeps FIRST so the persisted
        # payload never records an unresolved phase. Idempotent, so running twice
        # is harmless.
        timeline.close_open_phases(failed=(final_status not in
                                           ("answered", "refused", "clarify")))

        if final_status == "answered":
            timeline.completed(lc.PHASE_RESULT_PREPARATION)
            timeline.completed(lc.PHASE_COMPLETED)
        elif final_status in ("refused", "clarify"):
            # WARNING, not completed. The message here was already honest ("could not
            # answer") but the STATUS contradicted it, and the status is what the UI
            # renders: a refusal came out as four green ticks above a reply saying the
            # question could not be answered. A refusal is not a system failure — so
            # not `failed` either. `warning` is the state that exists for exactly this.
            timeline.warning(lc.PHASE_RESULT_PREPARATION,
                             "Could not answer this from the available data")
            timeline.completed(lc.PHASE_COMPLETED)
        else:
            timeline.failed(lc.PHASE_RESULT_PREPARATION,
                            "Could not complete this question")
    except Exception:
        pass


def _emit_terminal_lifecycle(timeline, final_status: str) -> None:
    """Close the timeline with the phase that matches the actual outcome.

    Only emitted for a terminal state we can describe safely: `answered` completes,
    anything else is a warning on result_preparation rather than a failure, because
    a refusal is a CORRECT outcome (the refuse-over-guess contract) and must not be
    presented to the user as the system breaking."""
    try:
        from veda import lifecycle as lc
        if not getattr(timeline, "enabled", False):
            return
        # EXP-B1: close any phase left OPEN (started, never resolved). Measured on the
        # 10-query benchmark: access_check hung at "started" on 6 of 10 query types,
        # because its completion lived on the deterministic SQL path only — document,
        # hybrid and refusal paths never reach it, so the user watched a spinner that
        # never finished. Resolving here means EVERY path closes it, whichever head ran.
        #
        # A phase already resolved NEGATIVELY is left alone: a real permission denial
        # emits access_check=failed earlier (pipeline._feedback), and overwriting that
        # with "verified" would be the contradiction this whole fix is about.
        # Shared with veda/pipeline.py::_done, which sweeps FIRST so the persisted
        # payload never records an unresolved phase. Idempotent, so running twice
        # is harmless.
        timeline.close_open_phases(failed=(final_status not in
                                           ("answered", "refused", "clarify")))

        if final_status == "answered":
            timeline.completed(lc.PHASE_RESULT_PREPARATION)
            timeline.completed(lc.PHASE_COMPLETED)
        elif final_status in ("refused", "clarify"):
            # WARNING, not completed. The message here was already honest ("could not
            # answer") but the STATUS contradicted it, and the status is what the UI
            # renders: a refusal came out as four green ticks above a reply saying the
            # question could not be answered. A refusal is not a system failure — so
            # not `failed` either. `warning` is the state that exists for exactly this.
            timeline.warning(lc.PHASE_RESULT_PREPARATION,
                             "Could not answer this from the available data")
            timeline.completed(lc.PHASE_COMPLETED)
        else:
            timeline.failed(lc.PHASE_RESULT_PREPARATION,
                            "Could not complete this question")
    except Exception:
        pass


def run_hybrid_query(query, verbose=False, on_event=None, trace_id=None):
    """Public front door. Owns the ONE query trace for the whole request.

    Mints a trace_id (reusing the caller's request id when one is passed —
    §1 "don't introduce a redundant identifier"), binds it as the AMBIENT trace
    so every downstream stage — Tier-1, Tier-2, retrieval, the SLM choke-point,
    summary, visualization — records into the SAME ExplainTrace without threading
    a `tr` parameter through every signature, then finalizes + persists it exactly
    once (Tier-1's own finish() becomes a checkpoint while this scope owns the
    trace — see explain.ExplainTrace.finish). Observability only: the returned
    MultiResult is byte-identical except for the added trace_id field."""
    tr = new_trace(query, trace_id=trace_id)
    # The user-safe execution timeline + per-source recorder for THIS query. Both
    # are _Null* objects unless their flag is on, so with the flags off this is two
    # attribute lookups and the answer path is byte-identical. Bound ambiently (the
    # same ContextVar pattern as the trace) so a stage deep in the pipeline can emit
    # without threading a parameter through every signature.
    from veda import lifecycle as _lc
    from veda import exec_records as _er
    _tl = _lc.new_timeline(on_event=on_event, trace=tr)
    _rec = _er.new_recorder(trace=tr, timeline=_tl)
    with use_trace(tr), _lc.use_timeline(_tl), _er.use_recorder(_rec):
        _final_status = "error"
        _tl.completed(_lc.PHASE_RECEIVED)
        # Ordering (live-verification finding): routing runs BEFORE Tier-1's
        # understanding stage, so without this the timeline showed
        # source_selection completing before understanding had even appeared.
        # The phase order comes from first appearance, so opening understanding
        # here restores the intended narrative without misstating anything —
        # reading the question genuinely is the first thing that happens.
        _tl.started(_lc.PHASE_UNDERSTANDING)
        try:
            result = _run_hybrid_query_inner(query, verbose=verbose, on_event=on_event)
            try:  # a final status for the trace's one-glance summary
                if getattr(result, "ok", False):
                    _final_status = "answered"
                elif getattr(result, "items", None):
                    _final_status = result.items[0].status or "unknown"
                else:
                    _final_status = "unknown"
            except Exception:
                _final_status = "unknown"
            try:
                result.trace_id = getattr(tr, "trace_id", "") or None
            except Exception:
                pass
            _clean_refuse_on_empty_error(result)
            # Decide the access-check outcome from the TURN's terminal feedback,
            # before the timeline is closed and re-read into the payload.
            # BEFORE the backfill: the warning it raises has to be in the trace by
            # the time the explainability payload is assembled from it.
            _mark_empty_results(result)
            _backfill_missing_explain(result)   # before the timeline refresh reads it
            _sync_reported_row_count(result)
            _reconcile_access_check(result, _tl)
            _emit_terminal_lifecycle(_tl, _final_status)
            # The payload was built before the line above ran, so the PERSISTED
            # timeline was one phase short. Refresh it now that the timeline is
            # genuinely final (F30).
            _refresh_persisted_timeline(result, _tl)
            return result
        finally:
            try:
                tr.finalize(_final_status)
            except Exception:
                pass




#: The key every head's payload carries once the outcome is decided, and the ONLY
#: thing the api tier reads to answer "did this turn actually produce anything?".
EMPTY_RESULT_KEY = "_no_results"


def _mark_empty_results(result) -> None:
    """Decide, once, whether a turn found ANYTHING — and say so on the payload.

    Measured live on the mainstream SQL path: the reply read "No results found.",
    `result.row_count` was 0, `warnings` was empty, and all four progress steps were
    green with "Checks passed" and "Relevant information available". The panel said
    the work succeeded; the answer said nothing was found. The same shape was
    confirmed on Tier-2, NoSQL, federated, cache replay, RAG-with-zero-chunks and
    hybrid — seven heads, one missing fact.

    The fact was missing because no head recorded it: `ok`/`status` describe whether
    the PIPELINE ran, not whether it found anything, and the two were being read as
    the same thing. So this decides it from the only evidence that means "found
    something" — rows for a tabular answer, passages for a document one.

    Sited HERE, at the one exit every head passes through, for the reason
    `_raise_low_confidence_caveat`, `_backfill_missing_explain` and
    `_sync_reported_row_count` are: wiring this per head is the recurring
    "wired to one path" bug that produced EXP-B1/B4/B5, and a head added next year
    gets this for free.

    DELIBERATELY CONSERVATIVE. It marks a turn empty only on POSITIVE evidence of
    emptiness — a rows list that is present and empty, or a reported passage count
    of zero. A head that reports neither (a refusal, small talk, a plain text
    answer) is left alone, because "we cannot tell" must not be rendered as "nothing
    was found". It never touches a turn that already refused: a refusal has its own,
    better explanation of why there is no answer.
    """
    try:
        for item in (getattr(result, "items", None) or []):
            if getattr(item, "status", None) != STATUS_OK:
                continue                      # a refusal explains itself
            payload = getattr(item, "result", None)
            if payload is None:
                continue
            _get = (payload.get if isinstance(payload, dict)
                    else lambda k, d=None: getattr(payload, k, d))
            rows = _get("rows")
            # BOTH NAMES. The RAG head calls them `chunks`, the HYBRID head calls
            # them `doc_chunks` — reading only the first meant a hybrid turn looked
            # like it had retrieved nothing, so its SQL half's zero rows decided the
            # outcome and an answer written from 5 passages was reported as "nothing
            # matched". `thinking_steps._absorb_evidence` already reads both names
            # for the same reason.
            chunks = _get("chunks")
            if chunks is None:
                chunks = _get("doc_chunks")
            empty = None
            # THE MODEL'S OWN DECLARATION, where it exists. The document head is the
            # one case with no countable signal: retrieval succeeded, passages came
            # back, and the model then said the passages do not answer the question
            # — a fact that lived only in the English prose until `no_answer` was
            # added to RAGResult. It is authoritative when true, and says nothing
            # when false, so the counted checks below still decide every other path.
            _declared = _get("no_answer") is True
            if isinstance(rows, list):
                empty = len(rows) == 0
            if not _declared and isinstance(chunks, (list, int)):
                _n = len(chunks) if isinstance(chunks, list) else chunks
                # Passages found means the document half had something, even when
                # the SQL half returned no rows — a HYBRID turn is not empty then.
                # This must be able to OVERRIDE the row check above; gating it on
                # `empty is not True` broke exactly the case it exists for, and a
                # hybrid answer written from 5 passages was reported as "nothing
                # matched" because its SQL half returned zero rows.
                empty = False if _n > 0 else True
            if _declared:
                # THE MODEL'S OWN DECLARATION WINS. It is the only signal on the one
                # path with nothing countable: retrieval succeeded, passages came
                # back, and the model then said they do not answer the question.
                empty = True
            if empty is not True:
                continue
            if isinstance(payload, dict):
                payload[EMPTY_RESULT_KEY] = True
            else:
                try:
                    setattr(payload, EMPTY_RESULT_KEY, True)
                except Exception:
                    pass
            try:
                from veda import warnings as _vwe
                _vwe.add(_vwe.NO_RESULTS)
            except Exception:
                pass
    except Exception:
        pass


def _reconcile_access_check(result, timeline) -> None:
    """Record an access-check FAILURE only if the turn actually refused on access.

    `access_check` is emitted optimistically at scope-resolution time (we know the
    caller has *a* permitted scope, not whether narrowing will block THIS query), and
    pipeline.py completes it once validation passes. The failure case is the hard one:
    a permission problem surfaces late, but so do plenty of failures that are NOT
    permission problems — and Tier-1 refusing does not mean the TURN refused.

    So the decision is made HERE, once, from the terminal feedback the user actually
    receives. `ACCESSED_DENIED_WHY` is feedback's own classification, not a
    re-derivation — one source of truth for "was this an access failure?".

    This runs BEFORE _emit_terminal_lifecycle and before _refresh_persisted_timeline,
    so the mark reaches both the live stream and the persisted payload. It only ever
    ADDS a negative resolution that the evidence supports; it never flips one away.
    """
    try:
        if not getattr(timeline, "enabled", False):
            return
        from veda import lifecycle as lc
        from veda.feedback import ACCESS_DENIED_WHY
        for item in (getattr(result, "items", None) or []):
            if getattr(item, "status", None) == STATUS_OK:
                continue
            payload = getattr(item, "result", None)
            fb = (payload or {}).get("feedback") if isinstance(payload, dict) else None
            if isinstance(fb, dict) and fb.get("why") == ACCESS_DENIED_WHY:
                timeline.failed(lc.PHASE_ACCESS_CHECK)
                try:
                    from veda import warnings as vw
                    vw.add(vw.RESTRICTED_DATA)
                except Exception:
                    pass
                return
    except Exception:
        pass


def _raise_low_confidence_caveat(confidence) -> None:
    """Raise the low-confidence caveat if the evidence does not clear the floor.

    ONE helper, called from EVERY answer-producing path. There are three — Tier-1
    (veda/pipeline.py::_done), Tier-2, and federated — and EXP-B3 originally landed
    on Tier-1 alone. Measured live: "probabtion period days in samta" took the
    FEDERATED route, was answered from amenity monthly fees relabelled "Total Days",
    and carried confidence=null with ZERO warnings. That is the same "wired to one
    path" mistake EXP-B1/B4/B5 were, so this is a shared function rather than a
    third copy of the check.

    `None` does NOT count as low. That was tried and reverted: the federated path
    computes no confidence at all and Tier-2 only has one when the Insight Engine
    ran, so treating None as low put a "limited matching data" caveat on every
    answer from those paths, including correct ones. A caveat on everything is a
    caveat on nothing. An absent confidence signal is a gap in the SIGNAL, and the
    fix for it is to compute one — not to warn unconditionally.

    Must be called BEFORE build_explain — safe_projection reads warnings from the
    trace at payload-build time. Never raises.
    """
    try:
        import config as _cfg
        floor = float(getattr(_cfg, "LOW_CONFIDENCE_WARNING_BELOW", 0.0) or 0.0)
        if confidence is None:
            return
        if floor > 0 and float(confidence) < floor:
            from veda import warnings as _vw
            _vw.add(_vw.LOW_EVIDENCE)
    except Exception:
        pass



#: The v1 blocks are DERIVED FROM SQL. Handed an empty SQL string, build_explain
#: still produces content — measured: `understanding: "List records."`,
#: `operations: [{"type": "list", "summary": "List records"}]`, and worst of all
#: `validation: {"passed": true, "checks": []}`. On a DOCUMENT answer all three are
#: fabrications: nothing listed any records, and claiming validation PASSED when
#: nothing was checked is the exact kind of false assurance this layer exists to
#: remove. The keys stay (a client reads them — CHAT_API_CONTRACT.md §1e) but the
#: content becomes the honest empty value: `None` where a fact is unknown, `[]`
#: where nothing happened.
_V1_WITHOUT_SQL = {
    "understanding": {"summary": None, "breakdown": []},
    "operations": [],
    "validation": {"passed": None, "checks": []},
    "data_used": {"datasets": [], "fields": []},
}


def _strip_invented_v1(explain: dict) -> dict:
    """Replace the SQL-derived v1 blocks with their honest empty form."""
    try:
        for key, empty in _V1_WITHOUT_SQL.items():
            if key in explain:
                explain[key] = json.loads(json.dumps(empty))
    except Exception:
        pass
    return explain


def _document_evidence(payload) -> dict:
    """Document names and passage count off a RAG/hybrid head result.

    These are REAL facts the head already computed and then threw away:
    `citations` carries "doc_name (p.N)" strings and `chunks` is the retrieved
    passage list. Before this, a document answer's `data_used.datasets` was empty
    even though the head knew exactly which handbook it had read.

    The page suffix is dropped and underscores become spaces — a display name, the
    same treatment a relational source's name gets. Names are DEDUPED preserving
    order, because five passages from one document are one document.
    """
    out = {"documents": [], "passages": 0}
    try:
        cites = list(getattr(payload, "citations", None) or [])
        seen = set()
        for c in cites:
            name = str(c).split(" (p.")[0].strip()
            if not name:
                continue
            name = name.rsplit(".", 1)[0] if "." in name[-6:] else name
            name = name.replace("_", " ").strip()
            if name and name not in seen:
                seen.add(name)
                out["documents"].append(name)
        chunks = getattr(payload, "chunks", None)
        if chunks is not None:
            out["passages"] = len(chunks)
    except Exception:
        pass
    return out


def _resync_flow_operations(explain: dict) -> None:
    """Rebuild the flow's "Operations applied" stage from the FINAL v1 operations.

    `flow` is assembled by build_explain_extension while the v1 operations are still
    the SQL-derived ones, and the document operations are written afterwards, here.
    So the two disagreed: v1 `operations` correctly said retrieval / read /
    synthesis while the flow the reader follows still said "List records" — measured
    live on a contract question answered entirely from a PDF.

    Only the operations stage is touched. "Combined across sources" carries the same
    stage name but is a separate, federation-only statement and is left alone, as is
    every other stage.
    """
    try:
        stages = (explain.get("flow") or {}).get("stages")
        if not isinstance(stages, list):
            return
        ops = [o.get("summary") for o in (explain.get("operations") or [])
               if isinstance(o, dict) and o.get("summary")]
        for st in stages:
            if isinstance(st, dict) and st.get("stage") == "operations" \
                    and st.get("label") == "Operations applied":
                if ops:
                    st["items"] = ops[:8]
                else:
                    st.pop("items", None)
                return
    except Exception:
        pass


def _apply_document_v1(explain: dict, ev: dict) -> dict:
    """Fill the v1 blocks with what a DOCUMENT answer actually did.

    Keeps the v1 shape exactly (`operations` entries stay `{type, summary}` — a
    client reads `summary`, so a differently-named field would be a silent break).

    `understanding.summary` states what was DONE, not an interpretation of the
    question. "Answered from the Employee Handbook, using 5 relevant passages" is
    checkable against the citations; "Question about employee separation notice
    period" would be the system's reading of the user's intent, which the document
    path never actually computes — and inventing one is the thing this layer exists
    to prevent.

    `validation` stays unknown: no query checks ran, and `passed: true` with an
    empty check list is a false assurance.
    """
    try:
        docs, n = ev.get("documents") or [], int(ev.get("passages") or 0)
        if docs:
            explain["data_used"] = {"datasets": list(docs), "fields": []}
        ops = []
        if n:
            ops.append({"type": "retrieval",
                        "summary": f"Retrieved {n} relevant passage{'' if n == 1 else 's'}"})
            ops.append({"type": "read", "summary": "Read the relevant passages"})
            ops.append({"type": "synthesis",
                        "summary": "Synthesized the retrieved information"})
        if ops:
            explain["operations"] = ops
            _resync_flow_operations(explain)
        if docs or n:
            where = docs[0] if len(docs) == 1 else f"{len(docs)} documents"
            bits = [f"Answered from {where}"] if docs else []
            if n:
                bits.append(f"using {n} relevant passage{'' if n == 1 else 's'}")
            explain["understanding"] = {"summary": " ".join(bits) + ".",
                                        "breakdown": [o["summary"] for o in ops]}
    except Exception:
        pass
    return explain


def _merge_document_evidence(explain: dict, ev: dict) -> dict:
    """ADD the document half to a payload that already describes the SQL half.

    A hybrid answer fuses SQL rows with document passages. When the SQL head
    succeeded, the hybrid result inherits ITS payload — which describes the SQL and
    says nothing about the documents. Observed live: a maintenance-policy answer
    carried `understanding: "Count all maintenances, grouped by Asset Id"` with
    `data_used.datasets` naming a relational table, while the answer the user read
    came from a policy document.

    Purely ADDITIVE — nothing the SQL head wrote is replaced, because that half of
    the answer really did happen. The document names and passage count are appended,
    and the summary is extended rather than rewritten, so the payload finally
    describes BOTH halves instead of half the answer.
    """
    try:
        docs, n = ev.get("documents") or [], int(ev.get("passages") or 0)
        if not (docs or n):
            return explain
        du = explain.get("data_used")
        if isinstance(du, dict):
            existing = list(du.get("datasets") or [])
            du["datasets"] = existing + [d for d in docs if d not in existing]
        ops = explain.get("operations")
        if isinstance(ops, list) and n:
            have = {o.get("summary") for o in ops if isinstance(o, dict)}
            for kind, text in (("retrieval",
                                f"Retrieved {n} relevant passage"
                                f"{'' if n == 1 else 's'}"),
                               ("read", "Read the relevant passages"),
                               ("synthesis", "Synthesized the retrieved information")):
                if text not in have:
                    ops.append({"type": kind, "summary": text})
        un = explain.get("understanding")
        if isinstance(un, dict) and docs:
            where = docs[0] if len(docs) == 1 else f"{len(docs)} documents"
            tail = f"Also drew on {where}"
            if n:
                tail += f" ({n} relevant passage{'' if n == 1 else 's'})"
            base = (un.get("summary") or "").rstrip()
            if tail not in base:
                un["summary"] = (base + (" " if base else "") + tail + ".").strip()
    except Exception:
        pass
    return explain

def _explain_from_trace_only() -> dict | None:
    """An explainability payload for a turn that never reached the pipeline.

    Built from the trace alone, with the v1 blocks emptied — there is no SQL, no
    operations and no filters, and inventing them would be worse than saying
    nothing. What survives is the part that matters on a refusal: the warnings,
    where we looked, and `support.trace_id`.

    Used by the permission-denial branch, which returns EARLY and so never reaches
    the front door's shared post-processing. The explainability belongs next to the
    denial it describes for the same reason the negative `access_check` emit does:
    a sweep at the exit is what lost that emit once already.
    """
    try:
        from veda.business_explain import build_explain
        from veda.explain import current_trace
        tr = current_trace()
        if tr is None or not getattr(tr, "enabled", False):
            return None
        return _strip_invented_v1(build_explain(
            sql="", table="", sm=None, trace=tr,
            trace_id=getattr(tr, "trace_id", "") or ""))
    except Exception:
        return None


def _backfill_missing_explain(result) -> None:
    """Give an ANSWERED turn an explainability payload when its head produced none.

    The document and hybrid heads never build one: `rag` has no SQL to describe, and
    `hybrid` inherits the SQL head's payload — which does not exist when that head
    refused or failed validation. The api tier then ships its empty `_NO_EXPLAIN`
    fallback, so a perfectly good answer arrives with NO explanation at all
    (observed live: a correct employee-handbook answer with `version: "1.0"` and
    every block empty).

    Built from the TRACE, once, HERE — at the one exit every head passes through —
    rather than per head. Wiring this kind of thing per path is the mistake that
    produced EXP-B1/B4/B5, the terminal step frame, the low-confidence caveat and
    the refusal timeline_summary, all in this same codebase.

    The v1 blocks come out EMPTY and that is deliberate: there is no SQL, no
    operations and no filters to describe, and inventing them would be worse than
    saying nothing. What the reader gains is the v2 half — routing, execution,
    sources, warnings, flow, audit — assembled from what actually happened. On a
    refusal or a denial those blocks are mostly absent too, because each one
    appears only where the trace holds evidence for it; what survives is the part
    that matters there — the warnings, where we looked, and the support trace id.

    ADDITIVE and safe for a client: a payload that was all-empty gains keys; no v1
    key changes shape (CHAT_API_CONTRACT.md §1e — "never require a block"). An item
    that already carries a payload is never touched, so a refusal that built its own
    keeps it.
    """
    try:
        from veda.business_explain import build_explain
        from veda.explain import current_trace
        tr = current_trace()
        if tr is None or not getattr(tr, "enabled", False):
            return
        for item in (getattr(result, "items", None) or []):
            # ANY item with no explainability of its own, not only an answered one.
            # A permission DENIAL and the federated `refuse` path both return
            # without building a payload, so the api tier shipped its empty
            # fallback — measured live: a denied csv_lake question arrived with
            # every block empty, no warning code, and NO `support.trace_id`, which
            # is the one thing a user needs to hand to support. An item that
            # already has a payload is left alone.
            payload = getattr(item, "result", None)
            if isinstance(payload, dict):
                if not payload.get("explain"):
                    payload["explain"] = _strip_invented_v1(build_explain(
                        sql="", table="", sm=None, trace=tr,
                        trace_id=getattr(tr, "trace_id", "") or ""))
            elif payload is not None and not getattr(payload, "explain", None):
                try:
                    _ex = _strip_invented_v1(build_explain(
                        sql="", table="", sm=None, trace=tr,
                        trace_id=getattr(tr, "trace_id", "") or ""))
                    payload.explain = _apply_document_v1(
                        _ex, _document_evidence(payload))
                except Exception:
                    pass
    except Exception:
        pass


def _sync_reported_row_count(result) -> None:
    """Make `result.row_count` agree with the rows that were actually returned.

    No path sets `execution.row_count` in the trace for a federated answer, so
    build_result_meta read nothing and the payload said `row_count: null` — while
    the thinking model's `evidence.rows` said 5 and the rendered table had exactly
    5 rows. Two numbers for one fact, one of them wrong.

    Filled HERE, at the one exit every head passes through, from the rows the head
    actually returned. Only ever fills a null: a count the engine did record is
    left alone, because that one came from the execution itself.
    """
    try:
        for item in (getattr(result, "items", None) or []):
            payload = getattr(item, "result", None)
            rows = None
            if isinstance(payload, dict):
                rows = payload.get("rows")
                ex = payload.get("explain")
            else:
                rows = getattr(payload, "rows", None)
                ex = getattr(payload, "explain", None)
            if not isinstance(ex, dict) or not isinstance(rows, list):
                continue
            res = ex.get("result")
            if isinstance(res, dict) and res.get("row_count") is None:
                res["row_count"] = len(rows)
    except Exception:
        pass

def _refresh_persisted_timeline(result, timeline) -> None:
    """Re-read the timeline into the payload AFTER the terminal phase is emitted.

    WHY (F30). `pipeline._done` builds the explainability payload, and only THEN
    does the front door emit the phase that describes the outcome
    (`result_preparation`). So the payload — the thing that gets PERSISTED and
    re-read from conversation history — was missing the final phase. Measured on a
    clarify: the live thinking steps correctly showed 4 steps ending in a warning,
    but the stored `timeline_summary` had 5 phases and stopped at validation. On an
    answered query it was masked by luck: a truncation warning happens to map to
    `result_preparation`, so the phase appeared for an unrelated reason.

    An omission, not a false claim — but a stored record that ends one phase early
    is exactly the class of thing this layer exists to remove.

    Only keys ALREADY PRESENT are refreshed. A v1 payload must not silently gain a
    v2 block here: the additive-by-contract promise runs in one direction only.
    """
    try:
        if not getattr(timeline, "enabled", False):
            return
        from veda import safe_projection as sp
        from veda.explain import current_trace
        tr = current_trace()
        if tr is None or not getattr(tr, "enabled", False):
            return
        # Each key is refreshed WHERE IT LIVES. The phase-carrying blocks live under
        # `audit` (level 3, §9); `warnings` is a top-level, user-facing block. The
        # distinction matters: build_explain's own v1 payload ALSO has a `timeline`
        # key, holding its stage ticks — refreshing that one with the lifecycle list
        # quietly replaced user-safe content with raw backend phase names, and
        # leaked `source_selection` / `data_retrieval` into the normal UX.
        fresh = {}
        for key, fn, where in (("timeline", sp.build_timeline, "audit"),
                               ("timeline_summary", sp.build_timeline_summary, "audit"),
                               # RESTRICTED_DATA is raised by
                               # _reconcile_access_check, which by design runs after
                               # pipeline._done has already built the payload.
                               ("warnings", sp.build_warnings, "top"),
                               # `limitations` is PROJECTED FROM the same warnings,
                               # so refreshing one without the other makes the two
                               # blocks disagree about the same turn. Measured on a
                               # zero-row answer: `warnings: ["no_results"]` beside
                               # `limitations: []`. Latent since the refresh was
                               # added — every warning raised after _done (
                               # RESTRICTED_DATA, NO_RESULTS) had the same split.
                               ("limitations", sp.build_limitations, "top")):
            try:
                fresh[key] = (fn(tr), where)
            except Exception:
                pass
        if not fresh:
            return
        for item in (getattr(result, "items", None) or []):
            payload = getattr(item, "result", None)
            if not isinstance(payload, dict):
                continue
            ex = payload.get("explain")
            if not isinstance(ex, dict):
                continue
            audit = ex.get("audit") if isinstance(ex.get("audit"), dict) else None
            for key, (value, where) in fresh.items():
                target = audit if where == "audit" else ex
                if target is not None and key in target:
                    target[key] = value   # refresh-only: never introduce a key
    except Exception:
        pass

def _run_hybrid_query_inner(query, verbose=False, on_event=None):
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
                try:  # record the rewrite so downstream knows what retrieval actually got
                    _cur_trace().set("query_understanding",
                                 original_query=query,
                                 effective_query=_l0.simplified_query,
                                 rewrite_reason="nl_simplifier")
                except Exception:
                    pass
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

    # Multi-source routing coordinator (docs/multisource_routing/). Flag-gated default-OFF; in shadow
    # it only traces (answer path byte-identical), and when authoritative it can drive the answer —
    # NO_MATCH/clarify refuse WITHOUT generating an answer, SINGLE routes via its source agent. Returns
    # None to defer to the legacy path (off / shadow / MULTI / any failure).
    _COMPOUND_HANDOFF.set(False)
    _routed = _run_coordinator(query, verbose=verbose, on_event=on_event)
    if _routed is not None:
        return _merge_extra_usage(_routed, _l0_usage_totals)

    # Cross-source federated route (MS-6): when the scope spans ≥2 sources and retrieval
    # selects columns from more than one, no single-DB head can join them — generate + run
    # a federated DuckDB query instead. Returns None (→ normal path) when not applicable.
    # Behaviour (c), 2026-09-18: NOT when the coordinator just handed a COMPOUND question
    # to the decomposer — federating first re-created the wrong "4 … average 10.88" answer.
    fed = None if _COMPOUND_HANDOFF.get() else _maybe_federated(query, verbose=verbose)
    if fed is not None:
        return _merge_extra_usage(fed, _l0_usage_totals)

    try:
        from config import QUERY_DECOMPOSE_ENABLED
    except Exception:
        QUERY_DECOMPOSE_ENABLED = False

    # Behaviour (c), 2026-09-18: the global decomposer flag is OFF (it mis-splits join
    # questions), but when the ROUTING COORDINATOR handed a MULTI decision off as compound
    # the split is safe by construction — it found ≥2 sources and no join relation, and the
    # decomposer produced ≥2 parts — so the handoff proceeds to the split regardless.
    if not QUERY_DECOMPOSE_ENABLED and not _COMPOUND_HANDOFF.get():
        route, res = _dispatch_single(query, verbose=verbose, on_event=on_event)
        return _merge_extra_usage(
            MultiResult(items=[_to_subresult(query, route, res)]), _l0_usage_totals)

    _emit(on_event, "classify", "Classifying query intent...")
    intent, _source_ids = classify(query, verbose=verbose)

    # Deterministic head: try it directly; a clean answer is complete-by-construction.
    if intent == "sql":
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
    # Each part starts from the FULL request scope: the coordinator's SINGLE route narrows
    # the ambient context to the chosen source (_constrain_scope_to), and without a reset
    # part 2 inherited part 1's source (run 4, 2026-09-18: "how many amenities" ran on the
    # maintenance source and refused). Snapshot → route → restore, every part.
    _saved_ctx = None
    try:
        from veda_core.context import try_current as _tc, set_context as _sc
        _saved_ctx = _tc()
    except Exception:
        _saved_ctx = None
    try:
        # Behaviour (c), 2026-09-18: each part of a compound question DECIDES ITS OWN
        # SOURCE — route it through the coordinator first (a SINGLE decision answers on
        # that source, the same pinned path its battery runs); None → the legacy
        # single dispatch on the ambient scope, exactly as before.
        _routed = None
        try:
            _routed = _run_coordinator(sq, verbose=verbose, on_event=on_event)
        except Exception:
            _routed = None
        if _routed is not None and getattr(_routed, "items", None):
            _it = _routed.items[0]
            return SubResult(sq, _it.status, _it.route, _it.result, _it.refuse_reason)
        route, res = _dispatch_single(sq, verbose=verbose, on_event=on_event)
    except Exception as e:
        print(f"  [Hybrid] sub-query crashed: {type(e).__name__}: {e}")
        route, res = "none", None
    finally:
        if _saved_ctx is not None:
            try:
                _sc(_saved_ctx)
            except Exception:
                pass
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
        return _with_summary(MultiResult(items=items))

    import io, sys, threading
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
    _parent_ctx = _try_ctx()
    # Same carry for the ambient query trace — worker threads start with an empty
    # contextvars context, so without this each parallel sub-query's SLM calls
    # (call_slm ledger) and stage records would fall on a NullTrace instead of the
    # one query trace. bind_trace in the child re-attaches the parent's trace.
    _parent_trace = _cur_trace()

    def _one(indexed_sq):
        i, sq = indexed_sq
        if _parent_ctx is not None:
            _set_ctx(_parent_ctx)
        _bind_trace(_parent_trace)
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
    return _with_summary(MultiResult(items=items))


#: route -> the engine label used when this dispatch has to open a record itself
#: (EXP-B4). Only routes that do NOT go through the SQL head need one; the SQL head
#: opens its own record around execute_sql, with true DB timing.
_ROUTE_ENGINE = {"rag": "rag", "hybrid": "rag", "nosql": "nosql"}


def _dispatch_single(query, verbose=False, precomputed_sql=None, on_event=None):
    """Thin wrapper over the real dispatch, adding the data_retrieval record for the
    heads that do not produce one themselves (EXP-B4).

    Measured on the 10-query benchmark: document and hybrid answers showed NO
    "Running the query" step at all, because the record was opened around execute_sql
    — which those heads never call. Wrapping the dispatch ONCE covers every head and
    every one of its seven return points; touching those individually is the very
    per-path mistake that caused this.

    Only fills a GAP: if the head already registered a record (the SQL path does, with
    real DB timing), nothing is added here — no double counting. Note the duration
    recorded for a document head is whole-retrieval time, not database time; there is
    no database in that path, so the two numbers legitimately mean different things.
    """
    from veda import exec_records as _erd
    # COMPLETE THE UNDERSTANDING PHASE for the heads that never reach pipeline.py.
    # Only Tier-1 completes it with facts, so on a document answer the first step
    # sat on its generic fallback sentence with nothing inside it on every single
    # turn — structurally, not by accident.
    #
    # The one thing this path genuinely knows about the QUESTION is the period the
    # user asked for: `_temporal(query)` is already parsed here (twice, further
    # down) and is a local parse, no network. Nothing else is available without
    # inventing it — the doc path computes no intent and no grouping, and a route
    # name like "rag" is not a fact about the question.
    #
    # Emitted from the same single seam that already adds the data_retrieval record
    # for these heads, and AFTER the dispatch because that is when the route is
    # known — classifying a second time here would be both wasteful and capable of
    # disagreeing with the route actually taken. Arriving late costs nothing: the
    # api tier's step model is monotonic (a late phase cannot reopen a finished
    # step) and the FACT is absorbed independently of which step the event lands on.
    _rec_before = _erd.current_recorder().has_records()
    route, res = _dispatch_single_inner(query, verbose=verbose,
                                        precomputed_sql=precomputed_sql,
                                        on_event=on_event)
    try:
        if _ROUTE_ENGINE.get(route):
            _tfu = _temporal(query)
            if _tfu is not None and (getattr(_tfu, "start", None)
                                     or getattr(_tfu, "end", None)):
                from veda import lifecycle as _lcu
                _lcu.current_timeline().completed(
                    _lcu.PHASE_UNDERSTANDING,
                    period=f"{getattr(_tfu, 'start', None) or '?'} to "
                           f"{getattr(_tfu, 'end', None) or '?'}")
    except Exception:
        pass
    try:
        _recorder = _erd.current_recorder()
        _engine = _ROUTE_ENGINE.get(route)
        if _engine and not _rec_before and not _recorder.has_records():
            # Prefer the source the ROUTER actually chose over the request
            # context's default. Measured on a document question: routing picked
            # source 3 (the document store) and the answer really did come from a
            # PDF, but this stamped the ambient context's source_id — 2, the
            # relational database — so the user was told "Retrieving data from
            # homzhub" for an answer that came from the employee handbook.
            # STRONGEST EVIDENCE FIRST: the passages the answer was written from
            # carry the id of the source they were retrieved out of. On the HYBRID
            # route the router picks the relational source (the SQL half is tried
            # first), so preferring the routing decision named `homzhub` for an
            # answer that came entirely from the employee handbook — measured live
            # on "How many casual leaves do employees get per year?", which returned
            # `rows: 0` from SQL, 5 passages from the documents, and reported
            # `sources: ["homzhub"]`. The chunks know better than the router did.
            _sid = None
            try:
                _chunks = (getattr(res, "chunks", None)
                           or getattr(res, "doc_chunks", None) or [])
                for _c in _chunks:
                    _cs = getattr(_c, "source_id", None)
                    if _cs:
                        _sid = _cs
                        break
            except Exception:
                _sid = None
            if _sid is None:
                try:
                    _routed = (_cur_trace().sections.get("routing") or {}).get("source_ids")
                    if _routed:
                        _sid = _routed[0]
                except Exception:
                    _sid = None
            if _sid is None:
                _ctx = _current_ctx()
                _sid = getattr(_ctx, "source_id", None) if _ctx is not None else None
            _r = _recorder.open(_sid if _sid is not None else "", engine=_engine)
            _ok = not getattr(res, "error", None) if res is not None else False
            _rows = getattr(res, "rows", None)
            _recorder.close(_r, _erd.COMPLETED if _ok else _erd.FAILED,
                            rows=(len(_rows) if _rows is not None else None),
                            error=(str(getattr(res, "error", "")) or None) if not _ok else None)
    except Exception:
        pass
    return route, res


def _dispatch_single_inner(query, verbose=False, precomputed_sql=None, on_event=None):
    """The single-query pipeline: classify → best head → (Tier-2 for SQL). Returns
    (route, head_result). This is the UNCHANGED per-modality dispatch — every sub-query
    of a compound query runs through here exactly as a standalone query would."""
    intent, source_ids = classify(query, verbose=verbose)
    print(f"\n  [Hybrid] intent = {intent}   sources = {source_ids or 'default'}")
    _emit(on_event, "route", f"Routed to {intent} engine", intent=intent)

    # ── SQL → DETERMINISTIC engine (the correctness brain) ────────────────────
    if intent == "sql":
        sm, cols = _load_semantic_model()

        # Gate 1 shortcut: this request's primary source has NO queryable table at
        # all (RBAC narrowed it to zero relational tables — e.g. a role granted
        # only a filesystem/datalake source — or the scope is genuinely
        # non-relational). Retrieval/SQL-gen can only ever fail here (there is
        # nothing for it to anchor to), so running the full pipeline just to reach
        # the same "no" wastes 5-40s of embedding/LLM calls and, worse, surfaces a
        # confusing "'X' doesn't match any value in this data" instead of the real
        # reason.
        if not sm.get("tables"):
            # classify()'s doc-intent override (_DOC_REF_RE) is a FIXED word list —
            # it can never be complete (a new document's own subject-matter vocabulary,
            # e.g. "maintenance" for maintenance_policy.docx, isn't a generic word like
            # "policy"/"document" and was never going to be hardcoded in advance).
            # Rather than add words one incident at a time, ask the real signal
            # instead: if this scope HAS a document source, actually retrieve against
            # it and trust content similarity, not a keyword guess. Only fall through
            # to the clean access_denied refusal below if that ALSO finds nothing
            # genuinely relevant.
            if _scope_has_doc_source():
                from query.rag_layer import run_rag_layer
                rag = run_rag_layer(query, source_ids=None, verbose=verbose, on_event=on_event)
                _MIN_SIM = 0.35  # a real topical match, not a coincidental near-miss
                if not getattr(rag, "error", None) and rag.confidence >= _MIN_SIM:
                    return "rag", rag

            from veda.feedback import explain_failure
            if sm.get("_not_materialized"):
                # Empty because the source was never materialized under the scoped
                # pipeline (2026-09-15, see _load_semantic_model) — NOT because RBAC
                # narrowed it to nothing. Say so; "access denied" sent people to the
                # wrong fix (roles) when the real one is "ingest this source".
                fb = explain_failure("not_materialized", sm)
                _emit(on_event, "answer", "This source has no schema model yet")
                return "deterministic", {"ok": False, "status": "not_materialized", "feedback": fb}
            fb = explain_failure("access_denied", sm)
            _emit(on_event, "answer", "No permitted database in scope")
            return "deterministic", {"ok": False, "status": "access_denied", "feedback": fb}

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
                try:
                    # Part 23: an ALTERNATE retrieval path is being taken. Worth
                    # telling the reader — the primary, fully-deterministic path
                    # could not answer, so the result came a different way. Names
                    # no internal component (never "Tier-2", never an agent class);
                    # veda/warnings.py's FALLBACK_USED copy is deliberately generic.
                    from veda import warnings as _vwt
                    _vwt.add(_vwt.FALLBACK_USED)
                except Exception:
                    pass
                # TIER-1 → TIER-2 boundary snapshot: exactly what Tier-1 knew when it
                # handed off (read from the ExecutionState it already built — no
                # recompute). One of the most important trace events for debugging
                # false multi-table planning / refusals.
                try:
                    _es_snap = res.get("context") if isinstance(res, dict) else None
                    _tr_snap = _cur_trace()
                    _cf_snap = getattr(_es_snap, "candidate_fields", None) or [] if _es_snap else []
                    _tr_snap.set(
                        "tier1",
                        handoff="tier1_refusal",
                        handoff_status=res.get("status") if isinstance(res, dict) else None,
                        primary_table=getattr(_es_snap, "primary_table", None),
                        candidate_tables=getattr(_es_snap, "candidate_tables", None),
                        candidate_field_count=len(_cf_snap),
                        rerank_query=getattr(_es_snap, "rerank_query", None),
                        refusal_reason=getattr(_es_snap, "refusal_reason", None))
                    for _c_snap in _cf_snap[:15]:      # verbose-only heavy list
                        _tr_snap.cand("tier1", "candidate_fields", _c_snap)
                except Exception:
                    pass
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
        from veda.pipeline import run_query
        from query.rag_layer import run_hybrid_layer
        sm, cols = _load_semantic_model()
        _emit(on_event, "hybrid", "Running SQL and document fusion...")
        # Run the DETERMINISTIC SQL head first and feed its EXECUTED rows into the
        # fusion (the correct-by-construction numbers), instead of letting the fusion
        # rely on LLM-written SQL. (Also supplies the previously-missing sql_columns.)
        # summarise=False: run_hybrid_layer below synthesises the answer the user
        # actually sees, over these same executed rows. Letting the SQL head also
        # write prose meant two summary-class SLM calls per hybrid turn and the
        # first one's output was never read (only cols/rows/explain/table are taken
        # from sqlres). One prose call per turn.
        sqlres = run_query(query, sm, cols, return_result=True, on_event=on_event,
                           summarise=False)
        sql_result = None
        if isinstance(sqlres, dict) and sqlres.get("ok"):
            _c, _r = sqlres.get("cols", []), sqlres.get("rows", [])
            sql_result = types.SimpleNamespace(
                columns=_c, rows=[dict(zip(_c, row)) for row in _r],
                row_count=len(_r), error=None)
        # Unified-graph chunk retrieval for the fusion context. Without this the
        # hybrid answer saw ONLY cosine-retrieved chunks — the graph's PPR walk
        # (which bridges chunks to the query's entities, and reaches chunks in
        # other sources via cross_source_fk) never influenced the synthesised
        # text at all. Purely additive downstream + fully guarded: on any failure
        # the fusion behaves exactly as before.
        _graph_chunks = None
        try:
            from config import (UNIFIED_GRAPH_ENABLED, GRAPH_RETRIEVAL_ENABLED,
                                GRAPH_EMBED_ENABLED)
            if UNIFIED_GRAPH_ENABLED and GRAPH_RETRIEVAL_ENABLED and GRAPH_EMBED_ENABLED:
                from query.graph_retriever import run_graph_retrieval
                _gr = run_graph_retrieval(query=query, source_ids=source_ids,
                                          verbose=verbose)
                _graph_chunks = list(getattr(_gr, "chunks", None) or [])
                if verbose:
                    print(f"  [Hybrid] graph chunks: {len(_graph_chunks)}")
        except Exception as _gce:
            if verbose:
                print(f"  [Hybrid] graph chunk retrieval skipped "
                      f"({type(_gce).__name__}: {_gce})")

        hy = run_hybrid_layer(query, sql_columns=[], source_ids=source_ids,
                             temporal_filter=_temporal(query),
                             sql_result=sql_result, verbose=verbose, on_event=on_event,
                             graph_chunks=_graph_chunks)
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
            # The inherited payload describes the SQL half only. Add the document
            # half so it describes the answer the user actually read.
            if hy.explain:
                hy.explain = _merge_document_evidence(
                    hy.explain, _document_evidence(hy))

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

    # M3 checkpoint 1: value grounding + STRICT qualifier completeness through the ONE
    # firewall (veda.firewall) — same gates, same order, one implementation. RBAC and
    # parameterisation already ran on this SQL before _tier2_validate is called, so
    # only the semantic gates are requested here (run_rbac/ast are the caller's).
    from veda.firewall import check as _fw_check, UNGROUNDED as _FW_UNGROUNDED
    from veda.ir import partial as _ir_partial
    _v = _fw_check(_ir_partial("tier2"), raw_sql, sm, query=query, allowed_tables=allowed_tables,
                   allowed_columns=allowed_cols, resolve_table=_resolve, strict_qualifier=True,
                   llm_generated=llm_written, tf=tf, run_alignment=False,
                   run_ir_equivalence=False, run_rbac=False, head="tier2",
                   trace=_cur_trace(), _semantic_only=True)
    if not _v.ok:
        return False, (f"ungrounded value {_v.detail}" if _v.verdict == _FW_UNGROUNDED
                       else f"dropped qualifier {_v.detail!r}")
    # Constraint-class check (2026-09-16, M1 close-out battery): "properties with more
    # than 3 floors" came back as SELECT total_floors … LIMIT 1000 — the threshold was
    # dropped, and qualifier_completeness can't see it because "3" is not a categorical
    # value with a referent. A numeric threshold needs a comparison (or HAVING); a
    # negation needs <> / NOT. Reason wording "dropped" feeds _repair_hint_for's existing
    # "represent every condition" hint, so the IR loop gets a retry before refusing.
    _ck = _constraint_kind(query)
    if _ck and not _sql_keeps_constraint(raw_sql, _ck):
        return False, f"dropped {_ck} constraint (no {'comparison' if _ck == 'threshold' else 'negation'} predicate in SQL)"
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
    facts = None
    try:
        from veda.business_explain import extract_sql_facts
        facts = extract_sql_facts(sql or "")
        table = facts["entities"][0] if facts["entities"] else None
    except Exception:
        pass
    result["table"] = table
    # M4: the IR for session memory, reverse-engineered from the SQL that actually ran.
    # Tier-2 builds no structured intent of its own (its firewall IR is a bare
    # `partial("tier2")`), so the AST facts extracted just above are the only structural
    # description of this answer that exists. Always ir_partial — see ir.from_sql_facts.
    try:
        from veda.ir import from_sql_facts as _ir_from_facts
        result["ir"] = _ir_from_facts(facts or {}, head=f"tier2.{source}").to_dict()
    except Exception:
        result["ir"] = None
    # sql_generation — every Tier-2 answer funnels through here, so this one place
    # records the final SQL shape for all Tier-2 return paths (envelope / shared
    # planner / IR). Reads the AST facts already extracted above; no re-parse.
    try:
        _trs = _cur_trace()
        _f = facts or {}
        _trs.set("sql_generation",
                 source=source,
                 tables_used=_f.get("entities"),
                 filter_count=len(_f.get("filters") or []),
                 aggregation_count=len(_f.get("aggregations") or []),
                 group_by=_f.get("groupings"),
                 order_by=_f.get("orderings"),
                 limit=_f.get("limit"),
                 distinct=_f.get("distinct"))
        _trs.cand("sql_generation", "sql", (sql or "")[:2000])   # verbose-only
    except Exception:
        pass

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
    _summary_engine = None            # which summariser produced the prose (trace)
    # Entity COVERAGE (flag-gated, never refuses) — the same check Tier-1 runs at L6c.
    # Computed HERE, before the summariser runs, because its `not_covered` terms are an
    # INPUT to that summary: the narrator sees only the question and the numbers, so
    # without them it reads the question's own entity list back as something the figures
    # cover. The explainability block at the end of this function reuses the result.
    _cov_ok, _cov_missing, _cov_terms = True, [], []
    try:
        from veda.intent_sql_alignment import entity_coverage
        _cov_ok, _cov_missing, _cov_terms = entity_coverage(query, sql or "", sm)
    except Exception:
        _cov_ok, _cov_missing, _cov_terms = True, [], []
    # Function scope, not the NL_ANSWER_ENABLED block below: record_result_stages()
    # at the end of this function reads it, and that call sits outside the block.
    _truncated_t2 = False
    _fetch_limit_t2 = None
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

        # rank_column: the column Tier-2's own SQL ordered by. Tier-1 takes this from
        # its ranking plan; Tier-2 has no such plan, but its ORDER BY is the same fact.
        # Read off the AST facts already extracted at the top of _tier2_finish — no
        # re-parse. Derived HERE because both summarisers below need it.
        _rank_column_t2 = None
        try:
            _ord = (facts or {}).get("orderings") or []
            _rank_column_t2 = _ord[0][0] if _ord else None
        except Exception:
            _rank_column_t2 = None
        # Did this result fill its fetch limit? Then row_count is a FLOOR and the
        # summariser must say "at least N". Same AST facts, no re-parse.
        _fetch_limit_t2 = (facts or {}).get("limit")
        _truncated_t2 = bool(_fetch_limit_t2 and len(rows or []) >= int(_fetch_limit_t2))  # noqa: F841 — read below

        if INSIGHT_ENGINE_ENABLED and _ictx is not None:
            try:
                from query.result_explainer import run_insight_engine
                # rank_column, like the run_nl_answer call below: Tier-1 passes it and
                # Tier-2 did not, so a Tier-2 "top N" narrative had to guess the sorted field.
                insight = run_insight_engine(_ictx, rank_column=_rank_column_t2)   # same ctx as above — one analysis pass
                if getattr(insight, "answer", None):
                    result["answer"] = insight.answer
                    got_real_answer = True
                    _summary_engine = "run_insight_engine"
                result["insights"] = insight.insights
                result["follow_up_questions"] = insight.follow_up_questions
                result["visualization"] = insight.visualization
                _confidence = insight.confidence
                visualization = insight.visualization
            except Exception as _ie:
                print(f"  [Tier2] Insight Engine unavailable ({type(_ie).__name__}: {_ie}) "
                      f"— falling back to plain NL answer")
                # Record the attempt even though it failed — call_slm() only records
                # token usage on a SUCCESSFUL backend.call() return, so an exception here
                # means zero tokens get attributed to this attempt even if the SLM was
                # actually contacted. Without this, usage.total_tokens == 0 is
                # indistinguishable from "no LLM call was needed this turn." (The
                # per-CALL SLM ledger in slm/_call_slm.py now DOES record this failed
                # call — purpose/model/duration/ok=False — onto the ambient query trace;
                # this _debug flag is the parallel token-accounting note.) It lives under
                # "_debug" — a key inference/routes/hybrid.py's _INTERNAL_ONLY_KEYS strips
                # at every nesting depth, same guarantee as "trace"/"context", so it never
                # reaches the client-facing response.
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

        # Summary-input parity with Tier-1 (veda/pipeline.py L7b). Tier-2 was calling
        # the SAME summariser with strictly less to work with: no `table`, no
        # `rank_column`, no `analytical_context`, and only the top-2 pattern details
        # instead of all verified findings. The visible effect was a Tier-2 "top N"
        # answer narrating an arbitrary column (often an id) because nothing told it
        # which field the ranking had actually sorted by — the same question answered
        # by Tier-1 got that right. Every value below is read off work this path has
        # ALREADY done (the SQL AST facts extracted at the top of _tier2_finish, the
        # single analytics pass, the caller's business_intent); nothing is re-derived
        # and no new LLM call is made.
        _all_findings = ([p.detail for p in _ictx.patterns]
                         if (_ictx is not None and getattr(_ictx, "patterns", None)) else [])
        _analytical_ctx_t2 = None
        try:
            from veda.planning import aggregate_operator as _agg_op
            from veda.semantic_validation import user_requested_identifier as _uri
            _analytical_ctx_t2 = {
                "intent": business_intent,
                "operation": _agg_op(query),
                "ranking": _rank_column_t2,
                "temporal": None,   # Tier-2 carries no resolved temporal window here
                "explicit_identifier": _uri(query),
            }
        except Exception:
            _analytical_ctx_t2 = None

        if not got_real_answer:
            try:
                from query.nl_answer import run_nl_answer
                nl = run_nl_answer(query, list(cols), row_dicts,
                                   timeout=NL_SUMMARY_TIMEOUT_MS / 1000.0, semantic_model=sm,
                                   table=str(table) if table else None,
                                   rank_column=_rank_column_t2,
                                   patterns=_all_findings,
                                   result_shape=getattr(_ictx, "result_shape", None),
                                   analytical_context=_analytical_ctx_t2,
                                   truncated=_truncated_t2, fetch_limit=_fetch_limit_t2,
                                   not_covered=_cov_terms,
                                   sql=sql)   # LIMIT-page awareness, see result_explainer
                if getattr(nl, "answer", None):
                    result["answer"] = nl.answer
                    _slm_wove_patterns = True   # SLM prose wove them; fallback blended them itself
                    _summary_engine = "run_nl_answer"
            except Exception as _nle:
                print(f"  [Tier2] Answer (summarisation skipped: {type(_nle).__name__})")

        # Fold the deterministic analytics into the final summary ONLY when no
        # summary SLM already phrased them — same natural-blend / no-double-statement
        # rule as Tier-1 (veda/pipeline.py L7b, 2026-07-17). Top 2 only.
        if _pattern_details and not _slm_wove_patterns:
            from query.result_explainer import blend_patterns
            result["answer"] = blend_patterns(result.get("answer") or "", _pattern_details)

    # Entity COVERAGE (flag-gated, never refuses) — the same check Tier-1 runs after its
    # qualifier gate (veda/pipeline.py L6c). Tier-2 answers the same shape of question, so
    # a partial answer must be as visible here: the entities left out are named in the
    # panel and the confidence is capped, instead of the answer reading as complete.
    try:
        if not _cov_ok:
            _cur_trace().check("entity_coverage", False, "not covered: " + ", ".join(_cov_missing))
            from config import ENTITY_COVERAGE_CONFIDENCE
            _confidence = (ENTITY_COVERAGE_CONFIDENCE if _confidence is None
                           else min(_confidence, ENTITY_COVERAGE_CONFIDENCE))
            print(f"  [Tier2] Coverage  ⚠  partial — not covered: {', '.join(_cov_missing)}")
        else:
            _cur_trace().check("entity_coverage", True, "")
    except Exception:
        pass
    _raise_low_confidence_caveat(_confidence)
    try:
        from veda.business_explain import build_explain
        result["explain"] = build_explain(sql=sql or "", table=table or "", sm=sm,
                                          visualization=visualization,
                                          confidence=_confidence,
                                          not_included=_cov_missing or None)
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
    # Record the shared post-execution stages (execution / result_analysis / summary /
    # visualization / explainability) into the ONE query trace so a Tier-2 answer tells
    # the same structured story as a Tier-1 one — reading only what was already computed.
    try:
        try:
            from config import NL_SUMMARY_MODEL as _nl_model
        except Exception:
            _nl_model = None
        _cur_trace().set("tier2", answered_via=source, row_count=len(rows or []))
        record_result_stages(
            engine=_summary_engine, cols=cols, row_count=len(rows or []),
            truncated=_truncated_t2, ictx=_ictx, answer=result.get("answer"),
            summary_model=_nl_model, summary_ok=bool(_summary_engine),
            visualization=visualization, explain_payload=result.get("explain"))
    except Exception:
        pass
    return result


class _EnvelopeSkip(Exception):
    """Control-flow only: the envelope contract can't express this question's shape."""


def _constraint_kind(query):
    """"threshold" | "negation" | "" — the constraint CLASS the question carries that a
    planner can silently drop or invert. Phrase sets are the ones query/operation_classifier
    already refuses on for the cross-source path (same rule, same words), plus the
    number-anchored "above/below/under/over N" form those sets miss."""
    try:
        from query.operation_classifier import _COUNT_THRESHOLD, _NEGATION
    except Exception:
        return ""
    q = " " + (query or "").lower().strip() + " "
    if any(s in q for s in _COUNT_THRESHOLD) or _re_mod.search(r"\b(above|below|under|over)\s+\d", q):
        return "threshold"
    if any(s in q for s in _NEGATION):
        return "negation"
    return ""


def _sql_keeps_constraint(sql, kind):
    """True when the SQL carries a predicate of that class: a comparison / BETWEEN / HAVING
    for a threshold, a <> / NOT for a negation. Unparseable SQL → True (never refuse on a
    parser hiccup; the AST firewall has already run)."""
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return True
    if kind == "threshold":
        kinds = (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between, exp.Having)
    else:
        kinds = (exp.NEQ, exp.Not)
    return any(tree.find(k) is not None for k in kinds)


def _envelope_inexpressible(query):
    """Reason string when the frozen intent envelope (INTENT_ENVELOPE_CONTRACT v1:
    count/measure/ratio/trend/compare/group/dimension_list, filters eq|ne) cannot
    represent the question, else "". Deterministic; reuses query/ranking_parser and
    _constraint_kind — the same "never drop or invert a constraint silently" rule the
    cross-source path already enforces. Temporal rankings ("last"/"first") are NOT
    gated: the envelope's count + time filter handles those."""
    try:
        from query.ranking_parser import parse_ranking
        rk = parse_ranking(query or "")
        if rk.top_n is not None or rk.basis == "metric":
            return f"ranking (top_n={rk.top_n}, basis={rk.basis})"
    except Exception:
        pass
    kind = _constraint_kind(query)
    if kind == "threshold":
        return "numeric threshold (envelope filters are eq|ne only)"
    if kind == "negation":
        return "negation (no anti-join / ne-only filters)"
    return ""


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
        # Tier-2 INPUT snapshot — what Tier-2 starts from (already computed above).
        try:
            _cur_trace().set(
                "tier2",
                available_column_count=len(getattr(sel, "columns", []) or []),
                candidate_tables=list(getattr(sel, "tables", []) or [])[:12],
                reused_tier1=execution_state is not None,
                seed_field_count=len(_seeds or []))
        except Exception:
            pass

        # ── ENVELOPE PATH (D): one-call SLM → intent envelope → deterministic build_sql.
        # Single-table analytical shapes (count/measure/ratio/trend/compare/group/list) go
        # through the ONE shared builder + value-grounding + AST firewall. On None (multi-
        # entity, unresolved handle, grain_suspect, or Ollama down) it falls through to the
        # IR→sql_builder path below — additive, never replaces the existing fallback.
        try:
            from query.envelope_slm import emit_envelope
            from query.intent_envelope import map_envelope_to_intent
            from query.intent import validate_intent, build_sql
            # Contract gate (2026-09-16, M1 close-out battery): the frozen envelope has NO
            # ranking intent and filters are eq|ne only. A question that carries a shape the
            # contract can't express ("top 5 X by Y", "more than 3 floors", "X without Y")
            # was still being answered — with the NEAREST expressible shape (a monthly
            # trend, a `= 3` filter). Refuse-over-guess: skip the envelope and let the IR
            # path (which can rank/compare or refuse typed) handle it. Reuses the
            # ranking parser and the cross-source expressiveness phrases — no new grammar.
            _inexp = _envelope_inexpressible(query)
            if _inexp:
                print(f"  [Tier2] envelope skipped (shape outside contract: {_inexp}) — fallback to IR")
                raise _EnvelopeSkip()
            _env, _hmap = emit_envelope(query, sel.columns, verbose=verbose)
            _qi = map_envelope_to_intent(_env, _hmap, tf) if _env else None
            if _qi is not None and validate_intent(_qi)[0] == "ok":
                _sql, _tbls, _cols, _route, _why = build_sql(_qi)
                # M3 checkpoint 1: the envelope's QueryIntent → IR → ONE firewall call
                # (value grounding, qualifier, alignment, IR-equivalence, RBAC, AST+params).
                from veda.firewall import check as _fw_check
                from veda.ir import from_query_intent as _ir_from_qi
                _fv = _fw_check(_ir_from_qi(_qi, head="tier2.envelope"), _sql, sm, query=query,
                                allowed_tables=_tbls, allowed_columns=_cols, ctx=_current_ctx(),
                                resolve_table=lambda _c: _qi.subject_table, strict_qualifier=True,
                                llm_generated=True, tf=tf, head="tier2.envelope", trace=_cur_trace())
                if not _fv.ok:
                    print(f"  [Tier2] envelope firewall {_fv.verdict} ({_fv.reason[:100]}) — fallback to IR")
                else:
                    psql, params = _fv.sql, _fv.params
                    if True:
                        ecols, erows, eerr = execute_sql(psql, list(params))
                        if eerr:
                            print(f"  [Tier2] envelope exec error ({eerr}) — fallback to IR")
                        else:
                            print(f"  [Tier2] answered via ENVELOPE ({_qi.query_type}) — {len(erows)} rows")
                            _print_rows(ecols, erows, sql=psql)
                            return _tier2_finish(query, sm, ecols, erows, psql, "envelope")
        except _EnvelopeSkip:
            pass                                          # already reported above
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
        # PROJECTION funnel (Tier-2 stage): the recommended SELECT-list handed to the
        # SLM. Later compared against IR-selected + SQL SELECT to reveal where extra
        # columns entered. Reuses the _rec_proj_cols already computed above.
        try:
            _cur_trace().set(
                "projection",
                recommended_count=(len(_rec_proj_cols) if _rec_proj_cols else 0),
                recommended=[getattr(r, "col_name", None) for r in (_rec_proj_cols or [])][:30],
                source="tier2")
        except Exception:
            pass

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
                try:
                    _cur_trace().set("tier2", attempt=_attempt,
                                     ir_error=getattr(l3, "error", None) or "empty ir_json")
                except Exception:
                    pass
                return None
            # IR GENERATION — what the SLM actually produced (the intermediate rep the
            # deterministic builder turns into SQL). Read straight off l3.ir_json.
            try:
                _ir = l3.ir_json or {}
                _ir_ents = _ir.get("entities", []) or []
                _ir_sel_cols = [c for e in _ir_ents
                                for c in (e.get("columns") or e.get("select") or [])]
                _cur_trace().set(
                    "tier2",
                    attempt=_attempt,
                    ir_intent=_ir.get("intent"),
                    ir_entity_count=len(_ir_ents),
                    ir_selected_column_count=len(_ir_sel_cols),
                    ir_has_filters=bool(_ir.get("filter_tree") or _ir.get("filters")),
                    ir_aggregation_count=len(_ir.get("aggregations") or []),
                    ir_group_by=_ir.get("group_by"),
                    ir_order_by=_ir.get("order_by"),
                    ir_limit=_ir.get("limit"),
                    ir_confidence=_ir.get("confidence"),
                    repaired=bool(_repair_hint))
                # projection funnel: IR-selected vs the recommended projection above
                _cur_trace().set("projection", ir_selected_count=len(_ir_sel_cols))
            except Exception:
                pass

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
                    # M3 checkpoint 1: ONE firewall call (RBAC + AST/params + the semantic
                    # gates, in the pipeline's order). A refusal feeds the repair hint
                    # first, the typed refusal last — the head reacts, the firewall decides.
                    from veda.firewall import check as _fw_check
                    from veda.ir import partial as _ir_partial
                    _fv = _fw_check(_ir_partial("tier2.shared_planner", ent_names[0]), act["sql"], sm,
                                    query=query, allowed_tables=a_tables, allowed_columns=a_cols,
                                    ctx=_current_ctx(), strict_qualifier=True, llm_generated=True,
                                    tf=tf, head="tier2.shared_planner", trace=_cur_trace())
                    if not _fv.ok:
                        if _attempt < _max_repairs:
                            _repair_hint = _repair_hint_for(_fv.reason); continue
                        print(f"  [Tier2] shared-planner firewall {_fv.verdict} (kept safe): {_fv.reason[:120]}")
                        return {"status": "tier2_rejected", "ok": False, "error": _fv.reason,
                                "firewall": _fv.trace_dict()}
                    psql, params = _fv.sql, _fv.params
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
            # M3 checkpoint 1: ONE firewall call. Same reaction contract as before: an
            # AST/allow-list failure ('invalid'/'rbac') is repairable → retry with a hint;
            # a semantic gate failure (ungrounded / dropped qualifier / mismatch) is NOT
            # (the schema doesn't change between attempts) → refuse at once. The SLM's
            # IR names entities, not grounded slots → partial IR (text heuristics apply).
            from veda.firewall import check as _fw_check, INVALID as _FW_INVALID, RBAC as _FW_RBAC
            from veda.ir import partial as _ir_partial
            _fv = _fw_check(_ir_partial("tier2.ir", next(iter(allowed_tables), None)), l4.sql, sm,
                            query=query, allowed_tables=allowed_tables, allowed_columns=allowed_cols,
                            ctx=_current_ctx(), strict_qualifier=True, llm_generated=True, tf=tf,
                            head="tier2.ir", trace=_cur_trace())
            if not _fv.ok:
                if _fv.verdict in (_FW_INVALID, _FW_RBAC):
                    try:
                        _cur_trace().check("tier2_firewall", False, _fv.reason[:200])
                        if _attempt < _max_repairs:
                            _cur_trace().repair("firewall", f"attempt {_attempt}", "retry")
                    except Exception:
                        pass
                    if _attempt < _max_repairs:
                        _repair_hint = _repair_hint_for(_fv.reason); continue
                print(f"  [Tier2] firewall {_fv.verdict} (kept safe): {_fv.reason[:120]}")
                return {"status": "tier2_rejected", "ok": False, "error": _fv.reason,
                        "firewall": _fv.trace_dict()}
            psql, params = _fv.sql, _fv.params
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
            # Gate 1 (User Story 3, 2026-08-08 audit finding): the relational
            # path's narrow_allowed doesn't apply here — a NoSQL source's schema
            # is connector-native (NoSQLCollection), not sm['tables']/['columns'].
            # filter_nosql_collections is its mirror: a no-op without a forwarded
            # data scope.
            collections = filter_nosql_collections(collections, int(sid), _current_ctx())
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
