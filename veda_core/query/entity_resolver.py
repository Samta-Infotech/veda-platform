# =============================================================================
# query/entity_resolver.py — Entity Resolution V1 (flag-gated experiment)
#
# WHAT business entities is the user talking about?  (The join planner still owns
# HOW they connect.)  This is the missing FUSION step the feasibility audit found:
# the resolvers already exist (target_selection, resolution.referent_tables), but
# their name-coverage domination — the logic that already makes `user` beat
# `user_profile` inside select_targets — is only applied to SECONDARY targets, never
# to the ANCHOR. Wrong-anchor routing (assets_asset→assets_salelisting,
# accounts_paymenttransaction→…settlement, users_user→users_userpreference) is the
# proven dominant failure (VEDA_ADVERSARIAL_FAILURE_MAP.md).
#
# V1 = apply name-coverage domination to ANCHOR selection, with table_type=MASTER as
# a *tiny* tie-break only (MASTER must NOT auto-win — an explicit detail noun like
# "sale listings" must keep assets_salelisting). Confidence-gated: RESOLVED /
# AMBIGUOUS / UNGROUNDED. Only RESOLVED changes behavior; everything else falls back
# to the current pipeline UNCHANGED. No new SLM call, no GNN, no synonym dependency
# (audit proved domain synonyms are column-level and mislead here). Deterministic.
# =============================================================================
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

RESOLVED = "RESOLVED"
AMBIGUOUS = "AMBIGUOUS"
UNGROUNDED = "UNGROUNDED"


@dataclass
class ResolvedEntities:
    anchor: Optional[str] = None
    secondaries: List[str] = field(default_factory=list)   # distinct entity tables (not the anchor)
    confidence: float = 0.0
    status: str = UNGROUNDED
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def distinct_tables(self) -> int:
        return len({self.anchor, *self.secondaries} - {None})


def _thresholds():
    try:
        from config import (ER_COVERAGE_MIN, ER_MARGIN_MIN, ER_PIN_CONFIDENCE)
        return ER_COVERAGE_MIN, ER_MARGIN_MIN, ER_PIN_CONFIDENCE
    except Exception:
        return 0.5, 0.4, 0.7


def _table_type(sm, t):
    return ((sm.get("tables", {}) or {}).get(t, {}) or {}).get("table_type", "")


# Entity-likelihood ordinal by table_type — a SMALL tie-break among name-coverage ties
# (a real entity beats a bridge/reference of the same name). MASTER does NOT auto-win:
# the term is 0.02–0.10, dwarfed by matched-count+coverage (~0–4), so an explicit detail
# noun (higher count/coverage) always beats a MASTER prior.
_TYPE_ORDINAL = {"MASTER": 1.0, "TRANSACTION": 0.8, "EVENT": 0.6, "REFERENCE": 0.4, "BRIDGE": 0.2}


_GLOSSARY = {"m": None}


def _entity_glossary():
    """Curated business-noun → canonical entity TABLE map (data/veda_entity_aliases.json).
    Resolves UNNAMED entities the query text doesn't lexically name (owner→users_user,
    property→assets_asset) — the completeness gap the probe proved metadata can't fill
    deterministically. Empty on any load failure (rule then no-ops)."""
    if _GLOSSARY["m"] is None:
        import os, json
        g = {}
        # cwd/data (engine runs with cwd=veda_core) OR module-relative veda_core/data
        # (pytest / other cwds) — robust regardless of where the process started.
        _here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # veda_core/
        for _p in (os.path.join(os.getcwd(), "data", "veda_entity_aliases.json"),
                   os.path.join(_here, "data", "veda_entity_aliases.json")):
            try:
                if os.path.exists(_p):
                    raw = json.load(open(_p))
                    g = {k.lower(): v for k, v in raw.items() if not str(k).startswith("_")}
                    break
            except Exception:
                g = {}
        _GLOSSARY["m"] = g
    return _GLOSSARY["m"]


def _score(count, coverage, table_type, retrieval):
    # matched-count + coverage (~0..4) dominate; type ordinal + retrieval are small
    # tie-breaks (they only decide among candidates that TIE on name coverage — e.g.
    # accounts_paymenttransaction vs reminders_reminderpaymenttransaction, where retrieval
    # knowing the query is about payments breaks it). Never large enough to let a wide
    # detail table out-rank a better-named entity.
    return count + coverage + 0.10 * _TYPE_ORDINAL.get(table_type, 0.5) + 0.15 * retrieval


def _retrieval_scores(results):
    """{table: normalized 0..1 max retrieval score} from this run's results."""
    raw = {}
    for r in (results or []):
        cid = getattr(r, "col_id", "")
        if "." in cid:
            t = cid.split(".", 1)[0]
            raw[t] = max(raw.get(t, 0.0), float(getattr(r, "final_score", 0.0)))
    hi = max(raw.values()) if raw else 1.0
    return {t: (v / hi if hi else 0.0) for t, v in raw.items()}


def _score_named_tables(qtoks, sm, retr):
    """Score every table the query lexically NAMES: distinctive-name-token coverage blended
    with table type and retrieval score. Returns a list of candidate dicts (unsorted)."""
    from veda.routing import _name_toks
    scored = []
    for t in (sm.get("tables", {}) or {}):
        toks = _name_toks(t, sm)
        if not toks:
            continue
        matched = toks & qtoks
        if not matched:
            continue                         # not named by the query → not an entity candidate
        coverage = len(matched) / len(toks)
        ttype = _table_type(sm, t)
        s = _score(len(matched), coverage, ttype, retr.get(t, 0.0))
        scored.append({"table": t, "matched": sorted(matched), "coverage": round(coverage, 3),
                       "count": len(matched), "table_type": ttype, "master": ttype == "MASTER",
                       "retrieval": round(retr.get(t, 0.0), 3), "score": round(s, 4)})
    return scored


def _add_glossary_candidates(scored, query, qtoks, sm, retr):
    """Append UNNAMED business entities (owner→users_user, property→assets_asset) the query
    text doesn't lexically name, via the curated glossary — each phrase present adds its
    canonical table at coverage 1.0. The completeness fix. Mutates `scored` in place."""
    import re
    from retrieval.query_enrichment import _singularize
    _ql = f" {query.lower()} "
    tables = sm.get("tables", {}) or {}
    for _phrase, _tbl in _entity_glossary().items():
        if _tbl not in tables:
            continue
        # single-word phrase → match on SINGULARIZED query tokens ("owners"→"owner");
        # multi-word → whole-phrase word-boundary match on the raw query.
        if len(_phrase.split()) == 1:
            _hit = _singularize(_phrase) in qtoks or _phrase in qtoks
        else:
            _hit = re.search(r"\b" + re.escape(_phrase) + r"\b", _ql) is not None
        if _hit and not any(d["table"] == _tbl for d in scored):
            _pw = max(1, len([w for w in _phrase.split() if len(w) > 2]))
            _tt = _table_type(sm, _tbl)
            scored.append({"table": _tbl, "matched": [_phrase], "coverage": 1.0,
                           "count": _pw, "table_type": _tt, "master": _tt == "MASTER",
                           "retrieval": round(retr.get(_tbl, 0.0), 3),
                           "score": round(_score(_pw, 1.0, _tt, retr.get(_tbl, 0.0)), 4),
                           "glossary": True})


def _drop_junction_candidates(scored, sm):
    """Drop VEDA-classified junction/bridge tables (planner-introduced, never the REQUESTED
    entity) — but keep them when they are the sole candidates. Best-effort. NOTE: content-
    carrying link tables (assets_salelistinguser) are NOT flagged by this heuristic (a
    name-only rule to catch them would also demote legitimate compound entities like
    users_userpreference), so it is deliberately not applied; glossary + retrieval handle
    the common cases."""
    try:
        from veda.runtime import get_graph
        from veda.planning import _junction_tables
        _junc = _junction_tables(get_graph(), sm)
        _non_junc = [d for d in scored if d["table"] not in _junc]
        return _non_junc or scored
    except Exception:
        return scored


def _pick_secondaries(scored, anchor, anchor_matched, cov_min):
    """Secondaries = other resolved entities naming a DISTINCT concept (matched tokens NOT a
    subset of the anchor's — a subset competes for the SAME slot). Among candidates sharing a
    matched-token set, keep only the best (users_user beats users_useraddress on {user}), so
    siblings don't flood the join. Glossary entities are always distinct secondaries."""
    _best_by_set = {}
    for d in scored:
        if d["table"] == anchor:
            continue
        _mset = frozenset(d["matched"])
        if not _mset or _mset.issubset(anchor_matched):
            continue
        if not (d.get("glossary") or d["coverage"] >= cov_min):
            continue
        if _mset not in _best_by_set or d["score"] > _best_by_set[_mset]["score"]:
            _best_by_set[_mset] = d
    return [d["table"] for d in
            sorted(_best_by_set.values(), key=lambda x: x["score"], reverse=True)][:3]


def resolve_entities(query, results, sm, all_cols):
    """Deterministic canonical entity resolution. Returns ResolvedEntities.

    Anchor = the table whose DISTINCTIVE name tokens are most fully covered by the
    query (matched-count primary, coverage tie-break — the same evidence select_targets
    uses), MASTER a hair's tie-break. UNGROUNDED when the query names no table (e.g.
    "properties" → nothing, since no table is literally named 'property' and synonyms
    are column-level) → caller falls back to existing routing, unchanged. Orchestrates
    the scoring/glossary/junction/classify/secondary helpers above."""
    import re
    from retrieval.query_enrichment import _singularize
    cov_min, margin_min, pin_conf = _thresholds()
    qtoks = {_singularize(w) for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 2}
    retr = _retrieval_scores(results)

    scored = _score_named_tables(qtoks, sm, retr)
    _add_glossary_candidates(scored, query, qtoks, sm, retr)
    scored = _drop_junction_candidates(scored, sm)

    if not scored:
        return ResolvedEntities(status=UNGROUNDED,
                                evidence={"reason": "no_table_named_by_query", "qtoks": sorted(qtoks)})

    scored.sort(key=lambda d: d["score"], reverse=True)
    top = scored[0]
    second = scored[1] if len(scored) > 1 else None
    margin = top["score"] - (second["score"] if second else 0.0)

    # AMBIGUOUS: the runner-up matches the SAME token SET (a true competitor for the same
    # entity slot — not merely a shared partial token, which happens for two DISTINCT
    # entities like leaselisting {lease,listing} vs leasetransaction {lease,transaction})
    # and is within a tight epsilon on the blended score. A coin-flip → fall back.
    _AMB_EPS = 0.06
    ambiguous = bool(second and set(top["matched"]) == set(second["matched"])
                     and margin < _AMB_EPS)

    status, confidence, anchor = UNGROUNDED, 0.0, None
    secondaries: List[str] = []
    if not ambiguous and top["coverage"] >= cov_min:
        status = RESOLVED
        # confidence blends how completely the query named the entity + the lead over #2
        confidence = round(min(1.0, 0.6 * top["coverage"] + 0.4 * min(1.0, margin)), 3)
        anchor = top["table"]
        secondaries = _pick_secondaries(scored, anchor, set(top["matched"]), cov_min)
    elif ambiguous:
        status = AMBIGUOUS
        confidence = round(top["coverage"], 3)

    ev = {
        "anchor_candidate": top["table"], "anchor_matched": top["matched"],
        "anchor_coverage": top["coverage"], "anchor_master": top["master"],
        "margin": round(margin, 3), "ambiguous": ambiguous,
        "candidates": [{k: d[k] for k in ("table", "matched", "coverage", "master", "retrieval")}
                       for d in scored[:6]],
        "pin_eligible": bool(status == RESOLVED and confidence >= pin_conf),
    }
    return ResolvedEntities(anchor=anchor, secondaries=secondaries, confidence=confidence,
                            status=status, evidence=ev)


def _resolve_secondaries(query, anchor, scored, sm, results, qtoks):
    """Reuse target_selection.select_targets to pick DISTINCT secondary entities the
    query also names, given the resolved anchor. Junctions/bridges are the planner's
    job (never returned here). Best-effort — any failure yields no secondaries."""
    try:
        from query.target_selection import select_targets
        from veda.routing import _name_toks
        from config import TARGET_SELECTION
        from veda.runtime import get_graph
        from veda.planning import _junction_tables
        graph = get_graph()
        junctions = _junction_tables(graph, sm) if graph else set()
        anchor_toks = _name_toks(anchor, sm)
        cols = (sm.get("columns", {}) or {})
        anchor_col_toks = set()
        from retrieval.query_enrichment import _singularize
        for cid in cols:
            if cid.split(".", 1)[0] == anchor and "." in cid:
                for w in cid.split(".", 1)[1].split("_"):
                    if len(w) > 2:
                        anchor_col_toks.add(_singularize(w))
        others = [d["table"] for d in scored if d["table"] != anchor]
        retr = _retrieval_scores(results)
        tr = select_targets(anchor, others, qtoks=qtoks, anchor_toks=anchor_toks,
                            anchor_col_toks=anchor_col_toks, retrieval=retr,
                            junctions=junctions, name_toks=lambda t: _name_toks(t, sm),
                            cfg=TARGET_SELECTION)
        # only confidently-requested, DISTINCT tables — ambiguity/uncertainty → no secondary
        if tr.ambiguous:
            return []
        return [t.table for t in tr.requested if t.table != anchor]
    except Exception:
        return []
