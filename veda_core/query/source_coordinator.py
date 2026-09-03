"""query/source_coordinator.py — the multi-source routing coordinator (routing Phase 3.1).

Ties the pieces together WITHOUT owning any source-specific logic:

    shared retrieval → group evidence by source → build candidates (tier + profile)
      → deterministic routing policy → (ambiguous → bounded SLM, validated) → RoutingDecision
      → dispatch to the selected source agent(s)

Everything the coordinator touches the outside world through is an injectable provider, so the
routing brain is unit-testable with no DB / SLM / model. The default providers wire the real
engine pieces (select_retrieval, graph_edges, agents). Profiles (is_canonical / domain_tags /
description) come from the Django Source registry via the api-tier; until that is wired the default
profile provider returns {} and the canonical tie-break simply doesn't fire (safe — ambiguity then
routes through the SLM/clarify path rather than guessing).
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Callable, Dict, List, Optional, Set, Tuple

from query.routing_contracts import (
    CandidateSource, RoutingDecision,
    STATUS_ROUTED, STATUS_NO_MATCH, STATUS_CLARIFY,
    MODE_SINGLE, MODE_MULTI,
)
from query.routing_policy import decide
from query.routing_slm import resolve_boundary


# ── request-scoped query embedding (P3: embed once) ──────────────────────────────────────────────
# A ContextVar (request-safe, concurrency-safe — NOT global mutable state): the query embedding is
# computed once per routing request and reused by BOTH the evidence provider and the item-prior
# provider, which previously each embedded the query separately. Reset at the top of plan_route.
_ROUTING_QV: ContextVar[tuple] = ContextVar("veda_routing_qv", default=(None, None))


def _query_embedding(query: str):
    """The BGE-M3 embedding of ``query`` for THIS routing request — computed once, then cached on the
    request-scoped ContextVar and returned on subsequent calls (embed-once). Same model/backend as
    before, so vectors are unchanged."""
    cached_q, cached_v = _ROUTING_QV.get()
    if cached_q == query and cached_v is not None:
        return cached_v
    try:
        from query.rag_layer import _encode_rag_query
        v = _encode_rag_query(query)
    except Exception:
        v = None
    _ROUTING_QV.set((query, v))
    return v
from query.source_evidence import group_evidence_by_source


# ── semantic decision boundary (P1) ─────────────────────────────────────────────────────────────
def _attach_item_summaries(query, candidates, per_source=3):
    """For the boundary candidates only, fetch each source's top-matching ITEM summaries (name +
    one-line summary from `source_item_embeddings`) and attach them to `evidence_summary["items"]`.
    This is the "relevant item descriptions/summaries" the SLM contract calls for — it lets the SLM
    connect an indirect query ("who are our suppliers") to the right item ("vendors: contact details").
    Best-effort; a failure just leaves the candidates as-is."""
    sids = [c.source_id for c in candidates]
    if not sids:
        return
    try:
        qv = _query_embedding(query)
        if qv is None:
            return
        vec = "[" + ",".join(f"{v:.8f}" for v in qv.tolist()) + "]"
        from ingestion.db_abstraction import get_internal_connection, release_internal_connection
        conn = get_internal_connection()
        try:
            by_source = {}
            with conn.cursor() as cur:
                for sid in sids:
                    cur.execute(
                        "SELECT name, summary, 1-(embedding <=> %s::vector) AS s "
                        "FROM source_item_embeddings WHERE source_id=%s AND embedding IS NOT NULL "
                        "ORDER BY embedding <=> %s::vector LIMIT %s",
                        [vec, str(sid), vec, per_source])
                    by_source[str(sid)] = [{"name": n, "summary": (sm or "")[:160]}
                                           for n, sm, _ in cur.fetchall()]
        finally:
            release_internal_connection(conn)
        for c in candidates:
            items = by_source.get(c.source_id)
            if items:
                c.evidence_summary = {**(c.evidence_summary or {}), "items": items}
    except Exception:
        pass


def _boundary_params():
    """(dominant_gap, compete_window) from config. dominant_gap: how far ahead the top source must be
    for a deterministic SINGLE (skip the SLM). compete_window: how close a runner-up must be to count
    as genuine competition that gets included in the SLM candidate set."""
    try:
        import config as _cfg
        return (float(getattr(_cfg, "ROUTING_DOMINANT_GAP", 0.10)),
                float(getattr(_cfg, "ROUTING_COMPETE_WINDOW", 0.08)))
    except Exception:
        return 0.10, 0.08


def _augment_with_edges(boundary, all_candidates, edge_pairs):
    """Add to the boundary candidate set any source that is (a) joined by a cross_source_fk edge to a
    source already in the set AND (b) has evidence CLOSE to that set's top (within COMPETE_WINDOW).
    This is the fix for the federated case (benchmark: cross-source 0%): the genuinely-needed second
    source ("...where we own assets" → the assets DB) is often only WEAK by tier but scores nearly as
    high as the primary (maintenance datalake) — a real join partner. It must be CLOSE, not merely
    above a noise floor, so a clean dominant single-source query (whose edge-neighbours score far
    lower) is NOT dragged into the SLM and stays deterministic + fast. Purely evidence+edge — no
    keywords."""
    if not edge_pairs or not boundary:
        return boundary
    _, compete_window = _boundary_params()
    top_in_set = max(getattr(c, "top_score", 0.0) for c in boundary)
    in_set = {c.source_id for c in boundary}
    by_id = {c.source_id: c for c in all_candidates}
    add = set()
    for pair in edge_pairs:
        if pair & in_set:
            for sid in (pair - in_set):
                c = by_id.get(sid)
                if c is not None and getattr(c, "top_score", 0.0) >= top_in_set - compete_window:
                    add.add(sid)
    return boundary + [by_id[s] for s in add if s in by_id]


def _rse_enabled():
    """Required-Source Escalation feature flag (default OFF → dominant-SINGLE path byte-identical)."""
    try:
        from config import REQUIRED_SOURCE_ESCALATION_ENABLED
        return bool(REQUIRED_SOURCE_ESCALATION_ENABLED)
    except Exception:
        return False


def _required_secondary(top, ranked, edge_pairs):
    """The best secondary candidate that MAY be required alongside the dominant ``top`` — using two
    orthogonal, already-computed signals (no new threshold, no keywords):
      SIGNAL 1 (structural): edge-connected to ``top`` via a discovered cross_source_fk pair.
      SIGNAL 2 (semantic):   item-prior-positive (query ↔ its dataset summaries cosine > 0), i.e. the
                             query is semantically ABOUT this source — not a bare shared-column match.
    Returns the highest item-prior such candidate, or None. The SLM (not this function) decides whether
    the secondary is actually required; this only gates WHEN to ask.
    """
    top_id = top.source_id
    connected = set()
    for pair in (edge_pairs or set()):
        if top_id in pair:
            connected |= (set(pair) - {top_id})
    cands = [c for c in ranked
             if c.source_id != top_id
             and c.source_id in connected                       # Signal 1: structural join path
             and getattr(c, "top_item_score", 0.0) > 0.0        # Signal 2: semantic item support
             and c.presence_tier in ("STRONG", "WEAK")]         # already a real candidate, not NONE
    return max(cands, key=lambda c: getattr(c, "top_item_score", 0.0)) if cands else None


def _decision_boundary(candidates, decision, ambiguous, edge_pairs=None):
    """Decide whether the bounded SLM should adjudicate SINGLE|MULTI|NONE, and on which compact
    candidate set. Returns (at_boundary, boundary_candidates). Evidence/tier/edge-based only — NO
    keywords, NO phrase matching, NO fixed similarity cutoff.

    NOT a boundary (stay deterministic):
      - the policy already produced a structural MULTI (edge co-leaders) — a computed relationship;
      - a clearly dominant SINGLE: exactly one STRONG source whose top_score is >= dominant_gap ahead
        of every other source.
    A boundary (let the SLM decide), because evidence alone can't answer SINGLE vs MULTI vs NONE:
      - the policy returned AMBIGUOUS; or
      - two or more sources are STRONG (genuine competition — maybe the query needs several); or
      - the top source is STRONG but a runner-up is within compete_window (meaningful competition —
        the "compare X with Y" case a structural edge can't catch, G1); or
      - no source is STRONG at all (nothing convincing — could be out-of-scope NONE, or a weak-but-
        valid single; the SLM decides, G2).
    """
    edge_pairs = edge_pairs or set()

    def _boundary(subset):
        return True, _augment_with_edges(subset, candidates, edge_pairs)[:6]

    if ambiguous:
        pooled = [c for c in candidates if c.source_id in set(getattr(decision, "source_ids", []))]
        return _boundary(pooled or list(candidates))

    # Confident DETERMINISTIC resolutions stay deterministic (never reach the SLM):
    #   - structural MULTI (relationship edge) and canonical tie-break are computed facts;
    #   - a genuine NO_MATCH with NO candidates at all is a real empty-evidence refusal.
    from query.routing_contracts import RC_RELATIONSHIP_EDGE, RC_CANONICAL_SELECTED
    if decision.reason_code in (RC_RELATIONSHIP_EDGE, RC_CANONICAL_SELECTED):
        return False, []
    if not candidates:
        return False, []

    dominant_gap, compete_window = _boundary_params()
    ranked = sorted(candidates, key=lambda c: getattr(c, "top_score", 0.0), reverse=True)
    strong = [c for c in ranked if c.presence_tier == "STRONG"]
    top = ranked[0] if ranked else None
    runner = ranked[1] if len(ranked) > 1 else None
    top_s = getattr(top, "top_score", 0.0) if top else 0.0
    runner_s = getattr(runner, "top_score", 0.0) if runner else 0.0

    # nothing convincing → SLM (may be NONE / out-of-scope, or a weak valid single)
    if not strong:
        window = [c for c in ranked if getattr(c, "top_score", 0.0) >= top_s - compete_window] or ranked
        return _boundary(window[:5])
    # genuine multi-candidate competition → SLM
    if len(strong) >= 2:
        return _boundary(strong[:5])
    # one STRONG but a close runner-up → SLM (meaningful competition; the compare/relate case)
    if runner is not None and (top_s - runner_s) < compete_window:
        window = [c for c in ranked if getattr(c, "top_score", 0.0) >= top_s - compete_window]
        return _boundary(window[:5])
    # exactly one STRONG, clearly dominant → deterministic SINGLE, skip the SLM. We do NOT edge-augment
    # here: a dominant homzhub query ("count properties by city") has datalake edge-neighbours that
    # score close on shared dimension words ("city"/"category"), and pulling them in would send every
    # clean single-source query to the SLM and destroy the fast deterministic path (measured
    # regression). Edge-augmentation is applied only to cases that ALREADY reach the boundary above
    # (ambiguous / genuine competition), where adding an edge-connected join partner is free.
    if (top_s - runner_s) >= dominant_gap or runner is None:
        # Required-Source Escalation (flag-gated, default OFF). A dominant SINGLE can still be a
        # cross-source query whose join partner is merely WEAKER, not absent — "primary more relevant
        # than secondary" is NOT "primary alone is sufficient". Escalate to the SAME bounded SLM ONLY
        # when a secondary is BOTH edge-connected AND item-prior-positive (see _required_secondary).
        # Every non-qualifying dominant SINGLE stays on the deterministic fast path unchanged.
        if _rse_enabled() and top is not None:
            sec = _required_secondary(top, ranked, edge_pairs)
            if sec is not None:
                return True, _augment_with_edges([top, sec], candidates, edge_pairs)[:6]
        return False, []
    window = [c for c in ranked if getattr(c, "top_score", 0.0) >= top_s - compete_window]
    return _boundary(window[:5])


# ── deterministic edge-driven MULTI override (flag-gated) ───────────────────────────────────────
def _tbl_tokens(name):
    """Entity tokens of a table name, singularized: 'assets_asset' → {asset}, 'vendors' → {vendor}.
    Language-layer normalisation of the SCHEMA's own table names — not a keyword list."""
    import re as _re
    return {t[:-1] if t.endswith("s") and len(t) > 3 else t
            for t in _re.findall(r"[a-z]+", (name or "").lower()) if len(t) > 2}


def _edge_multi_pair(query, source_ids):
    """Deterministic edge-driven MULTI (flag-gated, default OFF). Return [src_a, src_b] (sorted) when
    the QUERY names the entity tokens of BOTH endpoint tables of a HIGH-tier cross_source_fk edge whose
    two sources are BOTH in scope — a genuine cross-source JOIN the tuned policy under-routes to SINGLE
    when retrieval surfaced the wrong sibling table and one side reads WEAK ("which assets have
    maintenance tickets": the HIGH edge is maintenance↔assets_asset, but retrieval surfaced
    assets_leaselisting etc., so an entity-match on the surfaced tables misses). No keywords/thresholds:
    the tokens are the schema's own table names matched against the user's own words. Returns None when
    the flag is off, no such edge exists, an endpoint source is out of scope, or only one endpoint is
    named."""
    try:
        from config import CROSS_SOURCE_EDGE_MULTI_ENABLED as _on
    except Exception:
        _on = False
    if not _on:
        return None
    scope = {str(s) for s in (source_ids or [])}
    if len(scope) < 2:
        return None
    ql = " " + (query or "").lower() + " "

    def _named(table):
        return any(tok in ql for tok in _tbl_tokens(table))

    try:
        from ingestion.db_abstraction import (
            get_internal_connection, release_internal_connection)
        conn = get_internal_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ns.source_id, ns.table_name, nd.source_id, nd.table_name "
                    "FROM graph_edges e "
                    "JOIN graph_nodes ns ON ns.node_id = e.src_node_id "
                    "JOIN graph_nodes nd ON nd.node_id = e.dst_node_id "
                    "WHERE e.edge_type = 'cross_source_fk' "
                    "AND (e.attrs::jsonb->>'tier') = 'HIGH'")
                rows = list(cur.fetchall())
        finally:
            release_internal_connection(conn)
    except Exception:
        return None

    for sa, ta, sb, tb in rows:
        sa, sb = str(sa), str(sb)
        if sa in scope and sb in scope and _named(ta) and _named(tb):
            return sorted([sa, sb])
    return None


# ── candidate assembly ────────────────────────────────────────────────────────────────────────
def build_candidates(evidence_by_source: dict, profiles: dict) -> List[CandidateSource]:
    """Merge per-source evidence (tier, summary) with its registry profile (type, canonical, domain)."""
    out: List[CandidateSource] = []
    for sid, ev in evidence_by_source.items():
        prof = (profiles or {}).get(sid, {})
        summary = ev.summary() if hasattr(ev, "summary") else dict(ev or {})
        if prof.get("description"):
            summary = {**summary, "description": prof["description"]}
        top_score = max(getattr(ev, "top_item_score", 0.0),
                        getattr(ev, "top_column_score", 0.0),
                        getattr(ev, "top_chunk_score", 0.0))
        out.append(CandidateSource(
            source_id=sid,
            source_type=prof.get("source_type", ""),
            presence_tier=getattr(ev, "presence_tier", "NONE"),
            top_score=float(top_score),
            top_item_score=float(getattr(ev, "top_item_score", 0.0) or 0.0),
            is_canonical=bool(prof.get("is_canonical", False)),
            domain_tags=list(prof.get("domain_tags", []) or []),
            evidence_summary=summary,
        ))
    return out


# ── default providers (lazy, guarded — real engine pieces) ─────────────────────────────────────
def all_ready_source_ids() -> list:
    """Every source that has a routing prior (one row per item in source_item_embeddings) — i.e. all
    ingested/ready sources, permission-agnostic. Used only by the permission pre-check to tell whether a
    BETTER source than the user's permitted set exists. No content is read — just the source_id list."""
    try:
        from ingestion.db_abstraction import get_internal_connection, release_internal_connection
        conn = get_internal_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT source_id FROM source_item_embeddings")
                return [str(r[0]) for r in cur.fetchall()]
        finally:
            release_internal_connection(conn)
    except Exception:
        return []


def best_matching_source(query: str, source_ids, profiles=None):
    """The single STRICTLY-best-matching source over ``source_ids`` (permission-agnostic), by the same
    evidence + dominance tiering the router uses. Returns a source_id, or None when there is no clear
    winner. Only pre-computed match SCORES are read — no source content is fetched or returned.

    Runs the evidence retrieval under a PERMISSION-AGNOSTIC scope (the ambient RBAC data-scope would
    otherwise filter out the very sources this pre-check exists to detect) — restored immediately after.
    """
    _saved = None
    try:
        from veda_core.context import try_current, RequestContext, set_context
        _cur = try_current()
        if _cur is not None:
            _saved = _cur
            # same tenant/source_id, but NO allowed_resources and ALL sources in scope, for scoring only
            set_context(RequestContext(source_id=_cur.source_id, tenant=_cur.tenant,
                                       source_ids=tuple(source_ids), allowed_resources=None))
    except Exception:
        _saved = None
    try:
        cols, chunks = _default_evidence_provider(query, source_ids)
        ev = group_evidence_by_source(cols, chunks)
        if not ev:
            return None
        _apply_item_prior(query, source_ids, ev)
        _dominance_retier(ev)
        cands = build_candidates(ev, profiles or {})
        strong = [c for c in cands if c.presence_tier == "STRONG"]
        if not strong:
            return None
        top = max(strong, key=lambda c: getattr(c, "top_score", 0.0))
        # must be the strict top over EVERY candidate (a genuine single winner, not a tie)
        if any(c is not top and getattr(c, "top_score", 0.0) >= getattr(top, "top_score", 0.0) for c in cands):
            return None
        return top.source_id
    except Exception:
        return None
    finally:
        if _saved is not None:
            try:
                from veda_core.context import set_context
                set_context(_saved)                        # restore the caller's real (RBAC) scope
            except Exception:
                pass


def _default_evidence_provider(query: str, source_ids) -> Tuple[list, list]:
    """(columns, chunks) for ROUTING, each carrying a CLEAN bi-encoder COSINE per source.

    Routing tiers on relevance, so it needs the raw query↔column / query↔chunk cosine — NOT the
    answer path's reranked score. Benchmark finding: `select_retrieval` reranks columns and its
    `similarity` field collapses to ~0.07 (a homzhub property column's true cosine is ~0.5), and its
    `graph_result.chunks` carry a PPR score (~0.01). Tiering on those made every tabular source WEAK
    and dropped doc sources. So here we go straight to the cosine stores: `_cosine_search_v2` over
    `column_embeddings_v2` and `retrieve_top_k_chunks` over `doc_chunks` — both source_id-tagged with
    a genuine cosine. This is decoupled from the answer path (which still runs its own retrieval)."""
    sids = [str(s) for s in (source_ids or [])]
    columns, chunks = [], []
    try:
        from config import RAG_TOP_K as _K
    except Exception:
        _K = 5
    qv = _query_embedding(query)   # embed-once (P3): reused with the item-prior provider
    if qv is None:
        return columns, chunks
    # columns — clean cosine over the bi-encoder store, per source
    try:
        from ingestion.db_abstraction import get_internal_connection, release_internal_connection
        from query.retrieval_v2 import _cosine_search_v2
        from config import BIENCODER_COL_TABLE
        conn = get_internal_connection()
        try:
            columns = list(_cosine_search_v2(conn, BIENCODER_COL_TABLE, qv, max(_K, 10), sids) or [])
        finally:
            release_internal_connection(conn)
    except Exception:
        columns = []
    # chunks — clean cosine over the doc store
    try:
        from ingestion.chunk_embedder import retrieve_top_k_chunks
        chunks = list(retrieve_top_k_chunks(query_vector=qv, source_ids=sids, top_k=_K) or [])
    except Exception:
        chunks = []
    return columns, chunks


def _default_edge_provider(source_ids) -> Set[frozenset]:
    """cross_source_fk edge pairs among the scoped sources, from graph_edges (reuses the same
    relationship metadata the federated route uses)."""
    sids = [str(s) for s in (source_ids or [])]
    if len(sids) < 2:
        return set()
    try:
        from ingestion.db_abstraction import get_internal_connection, release_internal_connection
        from ingestion.graph_persist import GRAPH_EDGES_TABLE, GRAPH_NODES_TABLE
        conn = get_internal_connection()
        try:
            ph = ",".join(["%s"] * len(sids))
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT DISTINCT ns.source_id, nd.source_id FROM {GRAPH_EDGES_TABLE} e "
                    f"JOIN {GRAPH_NODES_TABLE} ns ON ns.node_id = e.src_node_id "
                    f"JOIN {GRAPH_NODES_TABLE} nd ON nd.node_id = e.dst_node_id "
                    f"WHERE e.edge_type = 'cross_source_fk' "
                    f"AND ns.source_id IN ({ph}) AND nd.source_id IN ({ph})",
                    sids + sids)
                pairs = set()
                for a, b in cur.fetchall():
                    if a and b and str(a) != str(b):
                        pairs.add(frozenset({str(a), str(b)}))
                return pairs
        finally:
            release_internal_connection(conn)
    except Exception:
        return set()


def _default_profile_provider(source_ids) -> Dict[str, dict]:
    """Profiles (type/canonical/domain/description) come from the Django Source registry, passed in
    by the api-tier. Engine-side default is empty until that is wired (Phase 3.6)."""
    return {}


def _default_item_prior_provider(query: str, source_ids) -> Dict[str, float]:
    """Per-source routing PRIOR: the top cosine of query ↔ this source's ITEM descriptions
    (`source_item_embeddings`). A source whose table/document is semantically ABOUT the query scores
    here even when its raw columns don't — the robust source-level signal the benchmark showed we
    need. Returns {source_id: top_item_cosine}. Best-effort; empty when the store/embedding is absent."""
    sids = [str(s) for s in (source_ids or [])]
    if not sids:
        return {}
    try:
        qv = _query_embedding(query)   # embed-once (P3): reused with the evidence provider
        if qv is None:
            return {}
        vec = "[" + ",".join(f"{v:.8f}" for v in qv.tolist()) + "]"
        from ingestion.db_abstraction import get_internal_connection, release_internal_connection
        conn = get_internal_connection()
        try:
            ph = ",".join(["%s"] * len(sids))
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT source_id, max(1 - (embedding <=> %s::vector)) FROM source_item_embeddings "
                    f"WHERE source_id IN ({ph}) AND embedding IS NOT NULL GROUP BY source_id",
                    [vec] + sids)
                return {str(sid): float(sim) for sid, sim in cur.fetchall()}
        finally:
            release_internal_connection(conn)
    except Exception:
        return {}


# ── the coordinator ─────────────────────────────────────────────────────────────────────────
def plan_route(query: str, source_ids, *,
               evidence_provider: Callable = None,
               edge_provider: Callable = None,
               profile_provider: Callable = None,
               item_prior_provider: Callable = None,
               slm_call: Callable = None,
               query_id: str = "", trace_id: str = "") -> RoutingDecision:
    """Produce a RoutingDecision for a query over an authorized source scope — WITHOUT executing.
    Deterministic-first; the SLM is consulted only for a genuinely ambiguous candidate set and its
    output is validated before it can route.
    """
    evidence_provider = evidence_provider or _default_evidence_provider
    edge_provider = edge_provider or _default_edge_provider
    profile_provider = profile_provider or _default_profile_provider
    _ROUTING_QV.set((None, None))   # reset embed-once cache for this routing request

    columns, chunks = evidence_provider(query, source_ids)
    evidence_by_source = group_evidence_by_source(columns, chunks)
    _apply_item_prior(query, source_ids, evidence_by_source, item_prior_provider)
    _dominance_retier(evidence_by_source)
    profiles = profile_provider(source_ids)
    candidates = build_candidates(evidence_by_source, profiles)
    edge_pairs = edge_provider(source_ids)

    decision, ambiguous = decide(candidates, edge_pairs)

    # Deterministic edge-driven MULTI override (flag-gated, default OFF → no-op, byte-identical). The
    # tuned policy routes SINGLE when exactly one source is STRONG, so a genuine cross-source JOIN whose
    # secondary reads WEAK (retrieval surfaced the wrong sibling table) is mis-routed. When the QUERY
    # names both endpoints of a HIGH cross_source_fk edge AND both sources are in scope, override that
    # SINGLE to a structural MULTI on exactly those sources. Only fires on a SINGLE that qualifies —
    # every other decision (existing MULTI, canonical, NONE) is untouched, and the boundary below skips
    # the SLM for this RC_RELATIONSHIP_EDGE decision (a computed relationship, not a guess).
    if decision.mode == MODE_SINGLE:
        _emp = _edge_multi_pair(query, source_ids)
        if _emp is not None:
            override = [c for c in candidates if c.source_id in set(_emp)]
            if len(override) >= 2:
                from query.routing_contracts import RC_RELATIONSHIP_EDGE, METHOD_DETERMINISTIC
                decision = RoutingDecision(
                    status=STATUS_ROUTED, mode=MODE_MULTI,
                    source_ids=[c.source_id for c in override],
                    candidate_sources=list(override),
                    evidence_summary=[c.evidence_summary for c in override if c.evidence_summary],
                    decision_method=METHOD_DETERMINISTIC, reason_code=RC_RELATIONSHIP_EDGE,
                    reason="Query names both endpoints of a cross-source relationship; routed MULTI.",
                    relationship_basis={"edge": "cross_source_fk", "pairs": [sorted(_emp)]})
                ambiguous = False

    # Semantic decision boundary (P1, G1+G2). The deterministic policy decides the confident cases;
    # when the evidence alone cannot answer "SINGLE vs MULTI vs NONE" (genuine competition, or nothing
    # convincing), hand the boundary to the bounded SLM — NOT a keyword/threshold heuristic. High-
    # confidence SINGLE and structural MULTI never reach the SLM.
    at_boundary, boundary_candidates = _decision_boundary(candidates, decision, ambiguous, edge_pairs)
    if at_boundary:
        # Deterministic document-preference tie-break (flag-gated). When an ambiguous doc+tabular set
        # reaches the boundary, the SLM was mis-picking the tabular source even though the document
        # source carried the strictly-higher signal (a "monthly Society Charges fee" answer lives in the
        # policy doc). If a DOCUMENT source is the strict top-signal candidate, route SINGLE to it and
        # skip the SLM. Uses the structural source_type + the already-computed max signal — no keywords.
        _doc_win = _doc_boundary_winner(boundary_candidates) if _doc_boundary_pref_on() else None
        if _doc_win is not None:
            from query.routing_contracts import METHOD_DETERMINISTIC, RC_SINGLE_CANDIDATE
            decision = RoutingDecision(
                status=STATUS_ROUTED, mode=MODE_SINGLE, source_ids=[_doc_win.source_id],
                candidate_sources=list(boundary_candidates),
                evidence_summary=[_doc_win.evidence_summary] if _doc_win.evidence_summary else [],
                decision_method=METHOD_DETERMINISTIC, reason_code=RC_SINGLE_CANDIDATE,
                reason="Document source carried the strict top signal at an ambiguous boundary.")
        else:
            _attach_item_summaries(query, boundary_candidates)   # give the SLM the matched item summaries
            decision = resolve_boundary(query, boundary_candidates, slm_call=slm_call,
                                        query_id=query_id, trace_id=trace_id)

    decision.query_id = decision.query_id or query_id
    decision.trace_id = decision.trace_id or trace_id
    return decision


def _apply_item_prior(query, source_ids, evidence_by_source, item_prior_provider=None):
    """Fold the source-level item-description prior into the evidence. A source that has NO surfaced
    column/chunk evidence but whose ITEM description matches the query is ADDED as a candidate (its
    table/document is on-topic even if retrieval didn't rank its columns). Sets `top_item_score`."""
    from query.source_evidence import SourceEvidence
    provider = item_prior_provider or _default_item_prior_provider
    try:
        priors = provider(query, source_ids) or {}
    except Exception:
        priors = {}
    for sid, score in priors.items():
        ev = evidence_by_source.get(sid)
        if ev is None:
            ev = SourceEvidence(source_id=sid)
            evidence_by_source[sid] = ev
        ev.top_item_score = max(getattr(ev, "top_item_score", 0.0), float(score))


_DOC_SOURCE_TYPES = frozenset({"filesystem", "document", "docs", "doc"})


def _doc_boundary_pref_on() -> bool:
    """True when ROUTING_DOC_BOUNDARY_PREF_ENABLED is set — at an ambiguous boundary, a document source
    that is the strict top-signal candidate is routed to deterministically instead of the SLM (which was
    mis-picking the tabular source). Default OFF → byte-identical (the SLM boundary resolver runs)."""
    try:
        import config as _cfg
        return bool(getattr(_cfg, "ROUTING_DOC_BOUNDARY_PREF_ENABLED", False))
    except Exception:
        return False


def _doc_boundary_winner(candidates):
    """Return the document (chunk-backed) candidate iff it is the STRICT top-signal source among the
    boundary set — i.e. its top_score is higher than every non-document candidate's. Structural
    (source_type + already-computed top_score); no keywords. None when a tabular source leads."""
    if not candidates:
        return None
    top = max(candidates, key=lambda c: getattr(c, "top_score", 0.0))
    if (top.source_type or "").lower() not in _DOC_SOURCE_TYPES:
        return None
    # strictly ahead of every non-document candidate (no tie with a table)
    for c in candidates:
        if c is top:
            continue
        if (c.source_type or "").lower() not in _DOC_SOURCE_TYPES \
                and getattr(c, "top_score", 0.0) >= getattr(top, "top_score", 0.0):
            return None
    return top


def _dominance_retier(evidence_by_source, gap=None, floor=None):
    """Re-tier sources RELATIVE to the field (benchmark finding). Absolute per-source floors fail on
    a large multi-column DB, which always has *some* column matching any query at ~0.6 — a spurious
    STRONG. Dominance fixes it: a source is STRONG only when its best cosine is within ``gap`` of the
    best source overall AND above a low ``floor`` (noise reject); clearly-behind sources drop to WEAK,
    noise to NONE. So a genuine winner (doc chunk 0.79) makes a spuriously-matched table (0.58) WEAK,
    and only genuinely-close sources stay STRONG together → SINGLE when one dominates, ambiguous when
    several tie. Both cosines are BGE-M3, so cross-kind comparison here is the same metric."""
    try:
        import config as _cfg
        gap = float(getattr(_cfg, "ROUTING_DOMINANCE_GAP", 0.12)) if gap is None else gap
        floor = float(getattr(_cfg, "ROUTING_DOMINANCE_FLOOR", 0.35)) if floor is None else floor
    except Exception:
        gap = 0.12 if gap is None else gap
        floor = 0.35 if floor is None else floor
    # Prefer the item-prior as the tiering signal when it's present for the field: item-description
    # cosines are directly comparable across sources (all query↔summary), whereas raw column vs chunk
    # cosines live in different distributions and mixing them re-introduces noise. Fall back to the
    # column/chunk max only for sources without an item prior (or when no source has one).
    # When set, the tiering signal is the MAX of (item prior, column cosine, chunk cosine) — the item
    # prior ADDS a source but must not SUPPRESS a strong raw cosine: a document source with a dominant
    # chunk hit (0.76) was being flattened to its weaker item-summary score (0.36), tying it with
    # spuriously-close tabular sources → AMBIGUOUS → SLM → NO_MATCH (doc questions refused). Default OFF
    # → the item-prior-preferred behaviour below is byte-identical.
    try:
        import config as _cfg2
        _max_signal = bool(getattr(_cfg2, "ROUTING_TIER_MAX_SIGNAL_ENABLED", False))
    except Exception:
        _max_signal = False
    have_item = any(getattr(e, "top_item_score", 0.0) > 0 for e in evidence_by_source.values())
    if _max_signal:
        tops = {sid: max(getattr(e, "top_item_score", 0.0),
                         e.top_column_score, e.top_chunk_score)
                for sid, e in evidence_by_source.items()}
    elif have_item:
        tops = {sid: (getattr(e, "top_item_score", 0.0)
                      or max(e.top_column_score, e.top_chunk_score))
                for sid, e in evidence_by_source.items()}
    else:
        tops = {sid: max(e.top_column_score, e.top_chunk_score)
                for sid, e in evidence_by_source.items()}
    if not tops:
        return
    mx = max(tops.values())
    for sid, e in evidence_by_source.items():
        t = tops[sid]
        if t >= floor and t >= mx - gap:
            e.presence_tier = "STRONG"
        elif t >= floor * 0.5:
            e.presence_tier = "WEAK"
        else:
            e.presence_tier = "NONE"


def _kind_of(sid, decision, profiles):
    prof = (profiles or {}).get(sid, {}) or {}
    if prof.get("source_type"):
        return prof["source_type"]
    for c in decision.candidate_sources:
        if c.source_id == sid and c.source_type:
            return c.source_type
    return ""


def _federated_delegate(query, tenant, source_ids):
    """Real cross-source federation (reuses the existing federated route). Monkeypatched in tests.

    Wrapped in bounded transient-retry (flag-gated, default-OFF): the federated route is the one
    dispatch branch the single/independent agents' `execute_reliably` never covered, so a transient
    infra blip (DuckDB ATTACH timeout, postgres reset) failed it permanently. `execute_federated_reliably`
    re-runs it ONLY on a transient failure the payload itself labels; OFF → single pass-through."""
    from query.federated_route import run_federated
    from query.reliability import execute_federated_reliably
    return execute_federated_reliably(
        lambda: run_federated(query, tenant=tenant, source_ids=source_ids))


def _build_execution_context(decision, query):
    """Assemble the request-scoped ExecutionContext threaded into agent.execute (P2/G5) — the routing
    decision + the query embedding already computed during routing (reused, not recomputed)."""
    from query.routing_contracts import ExecutionContext
    cached_q, cached_v = _ROUTING_QV.get()
    return ExecutionContext(
        query=query, query_id=getattr(decision, "query_id", ""),
        trace_id=getattr(decision, "trace_id", ""),
        selected_source_ids=list(getattr(decision, "source_ids", []) or []),
        routing_decision=decision,
        query_embedding=cached_v if cached_q == query else None)


def _dispatch_flags():
    """(use_adapter, use_execution_request) from config — two SEPARATE flags, one per architectural
    change (Phase A3 / Phase B2). use_execution_request implies adapter resolution even if
    SOURCE_ADAPTER_DISPATCH_ENABLED itself is off, because execute_request() only exists on
    SourceAdapter (see _resolve_executable's docstring)."""
    try:
        import config as _cfg
        use_adapter = bool(getattr(_cfg, "SOURCE_ADAPTER_DISPATCH_ENABLED", False))
        use_execution_request = bool(getattr(_cfg, "EXECUTION_REQUEST_DISPATCH_ENABLED", False))
    except Exception:
        use_adapter = False
        use_execution_request = False
    return use_adapter, use_execution_request


def _resolve_executable(source_kind: str):
    """Phase A3 (Source Adapter Foundation) / Phase B2 (Execution Request): return the object whose
    ``.execute(...)`` (or, Phase B2, ``.execute_request(...)``) runs this source kind — either
    query/source_adapters.py::SourceAdapter (a thin call-through wrapper, same signature/return as
    the bare agent) when SOURCE_ADAPTER_DISPATCH_ENABLED or EXECUTION_REQUEST_DISPATCH_ENABLED is on,
    or the bare agent from query.agents.resolve_agent() otherwise. Both flags OFF -> resolve_agent()
    is called exactly as before either phase existed — byte-identical. See
    docs/architecture/VEDA_SOURCE_CAPABILITY_ADAPTER_AUDIT.md and
    docs/architecture/VEDA_CANONICAL_EXECUTION_REQUEST_AUDIT.md."""
    use_adapter, use_execution_request = _dispatch_flags()
    if use_adapter or use_execution_request:
        from query.source_adapters import resolve_adapter
        return resolve_adapter(source_kind)
    from query.agents import resolve_agent
    return resolve_agent(source_kind)


def dispatch(decision: RoutingDecision, query: str, *, sm=None, cols=None,
             profiles: dict = None, evidence=None, on_event=None):
    """Execute a SINGLE-mode routing decision via its source agent, returning the AgentResult.
    NO_MATCH / CLARIFICATION_REQUIRED / MULTI return None (use execute_decision for MULTI).

    Phase B2 (EXECUTION_REQUEST_DISPATCH_ENABLED, default OFF): when on, this function's own
    legacy kwargs are unchanged — no caller has to change — but internally it normalizes them into
    ONE query/execution_request.py::ExecutionRequest and calls the resolved SourceAdapter's
    execute_request(request) instead of the legacy agent.execute(...) call. execute_request() is a
    pure unpack-and-delegate back to execute(...) (Phase B1, proven equivalent by test), so this is
    a boundary-shape change only — never a behavior change."""
    from query.reliability import execute_reliably
    if decision.status != STATUS_ROUTED or decision.mode != MODE_SINGLE:
        return None
    sid = decision.source_ids[0]
    executable = _resolve_executable(_kind_of(sid, decision, profiles))
    if executable is None:
        return None
    exec_ctx = _build_execution_context(decision, query)
    _, use_execution_request = _dispatch_flags()
    if use_execution_request:
        from query.execution_request import ExecutionRequest
        request = ExecutionRequest(query=query, source_id=sid, source_ids=[sid], sm=sm, cols=cols,
                                   execution_context=exec_ctx, on_event=on_event)
        return execute_reliably(lambda: executable.execute_request(request, evidence=evidence))
    return execute_reliably(lambda: executable.execute(
        query, source_id=sid, source_ids=[sid], sm=sm, cols=cols,
        evidence=evidence, execution_context=exec_ctx, on_event=on_event))


def execute_decision(decision: RoutingDecision, query: str, *, sm=None, cols=None,
                     tenant: str = "default", profiles: dict = None, evidence=None, on_event=None):
    """Execute a ROUTED decision end-to-end (Phase 4.2). Returns a dict:
        {"kind": "single"|"federated"|"independent"|"none", ...}
    - SINGLE       → the source agent's AgentResult.
    - MULTI + federated (join edge)  → run_federated payload (existing cross-source SQL).
    - MULTI + independent (no edge)  → each agent run, merged by result_orchestrator with an
                                       explicit policy (APPEND / CANONICAL_PRIORITY / CONFLICT).
    NO_MATCH / CLARIFY / NONE → {"kind": "none"} (caller surfaces the decision itself).
    """
    from query.execution_planner import (
        plan_execution, STRATEGY_SINGLE, STRATEGY_FEDERATED, STRATEGY_INDEPENDENT)
    if decision.status != STATUS_ROUTED or decision.mode not in (MODE_SINGLE, MODE_MULTI):
        return {"kind": "none", "decision": decision}

    plan = plan_execution(decision)

    if plan.strategy == STRATEGY_SINGLE:
        return {"kind": "single", "plan": plan,
                "result": dispatch(decision, query, sm=sm, cols=cols, profiles=profiles,
                                   evidence=evidence, on_event=on_event)}

    if plan.strategy == STRATEGY_FEDERATED:
        return {"kind": "federated", "plan": plan,
                "result": _federated_delegate(query, tenant, list(decision.source_ids))}

    # independent: run each source's agent (with bounded transient retry), then merge under an
    # explicit policy. Per-source partial failure (Phase 5.3) is surfaced, never hidden.
    from query.agents import resolve_agent
    from query.result_orchestrator import merge_results
    from query.reliability import execute_reliably, classify_failure
    exec_ctx = _build_execution_context(decision, query)
    results, failures = [], []
    for step in plan.steps:
        agent = resolve_agent(step.source_type or _kind_of(step.source_id, decision, profiles))
        if agent is None:
            failures.append({"source_id": step.source_id, "required": step.required,
                             "error": "no agent for source kind", "failure_class": "permanent"})
            continue
        res = execute_reliably(lambda a=agent, s=step: a.execute(
            query, source_id=s.source_id, source_ids=[s.source_id], sm=sm, cols=cols,
            evidence=evidence, execution_context=exec_ctx, on_event=on_event))
        results.append(res)
        if getattr(res, "status", "") == "failed":
            failures.append({"source_id": step.source_id, "required": step.required,
                             "error": res.error, "failure_class": classify_failure(res.error)})
    canonical_ids = {c.source_id for c in decision.candidate_sources if c.is_canonical}
    any_required_failed = any(f["required"] for f in failures)
    return {"kind": "independent", "plan": plan, "results": results,
            "merge": merge_results(results, canonical_ids=canonical_ids),
            "partial": {"failures": failures, "any_required_failed": any_required_failed,
                        "ok_count": sum(1 for r in results if getattr(r, "status", "") == "ok"),
                        "complete": not failures}}
