"""VEDA · L2/L3 — semantic table routing + primary-table selection."""
import os, re, sys, time, json, logging, threading
from veda.runtime import IMPORTANCE_WEIGHTS, TABLE_EMB_TABLE, _encode_query, _pg


def route_tables_semantic(query, top_n=6):
    """Rank tables by cosine of the query against the table_embeddings store.

    Returns {table_name: similarity}. Empty dict if the table isn't built yet
    (graceful fallback to the lexical/column path).
    """
    try:
        qv = _encode_query(query).tolist()
        vec = "[" + ",".join(str(x) for x in qv) + "]"
        conn = _pg()
        with conn.cursor() as cur:
            cur.execute(f"SELECT to_regclass('{TABLE_EMB_TABLE}')")
            if cur.fetchone()[0] is None:
                conn.close(); return {}
            cur.execute(
                f"SELECT table_name, 1 - (embedding <=> %s::vector) "
                f"FROM {TABLE_EMB_TABLE} ORDER BY embedding <=> %s::vector LIMIT %s",
                (vec, vec, top_n))
            rows = cur.fetchall()
        conn.close()
        return {r[0]: float(r[1]) for r in rows}
    except Exception:
        return {}


def select_primary_table(results, query, semantic_model):
    """Choose the target table by blending three signals:

      1. SEMANTIC ROUTING (primary) — cosine of the query vs the per-table
         embeddings (built from name + business_purpose + columns). This finally
         uses the rich table understanding, vector-based not keyword-based.
      2. COLUMN MAX score — best single-column retrieval score per table (so a
         small correct MASTER table isn't drowned by a wide table on count).
      3. LEXICAL name match (singularized) — tiebreaker / fallback when the
         table_embeddings store isn't built yet.
    """
    from retrieval.query_enrichment import _singularize

    routed = route_tables_semantic(query, top_n=8)   # {table: cosine} (may be empty)

    # importance_class → weight. Lives in the ranking layer (retunable, no re-ingest).
    # Down-weights audit/surrogate-id columns so they stop inflating the wrong table.
    cols_meta = semantic_model.get("columns", {})
    max_score = {}
    for r in results:
        t = r.col_id.split(".")[0]
        imp = (cols_meta.get(r.col_id, {}) or {}).get("importance_class", "MEDIUM")
        w = IMPORTANCE_WEIGHTS.get(imp, 0.6)
        max_score[t] = max(max_score.get(t, 0.0), r.final_score * w)

    candidates = set(max_score) | set(routed)

    hi = max(max_score.values()) if max_score else 1.0
    q_tokens = {_singularize(w) for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 2}

    # A table the query NAMES must always be a candidate, even if 5-signal retrieval
    # missed it: retrieval can rank a richer sibling higher ("sla config" lost to
    # sla_instance because the latter matches "workflow state" on more columns), which
    # silently routes to the wrong table. Name-matched tables enter scoring; only a
    # strong lexical match (below) actually wins, so this adds candidates, not noise.
    for t in semantic_model.get("tables", {}):
        if q_tokens & {_singularize(w) for w in t.split("_") if len(w) > 2}:
            candidates.add(t)

    if not candidates:
        return None

    best, best_score = None, -1e9
    for t in candidates:
        # Strip structural connectives so a wide name can't match on glue words
        # (investigation_AND_research_counter_party must not match query token "and").
        name_tokens = {_singularize(w) for w in t.split("_")
                       if len(w) > 2 and w not in _NAME_CONNECTIVES}
        matched = q_tokens & name_tokens
        coverage = (len(matched) / len(name_tokens)) if name_tokens else 0.0
        # Lexical evidence outranks a higher semantic cosine ONLY when the query genuinely
        # names the table: matched-token COUNT primary, COVERAGE the tiebreaker. A SINGLE
        # name-token match is often coincidental ("case" inside ml_use_case; one token of a
        # wide counterparty table) and must NOT override the learned reranker — so a lone
        # token contributes only coverage and defers to sem/col. Two+ matched tokens
        # ("document categories" → document_category_master) keep the full lexical weight.
        lex = (1.5 * len(matched) + 2.0 * coverage) if len(matched) >= 2 else 0.5 * coverage
        sem = routed.get(t, 0.0)                 # 0..1 cosine, the strongest signal
        col = (max_score.get(t, 0.0) / hi) if hi else 0.0   # 0..1 normalized
        combined = 1.0 * sem + 0.5 * col + lex
        if combined > best_score:
            best_score, best = combined, t
    return best


def vet_primary(query, primary, results, semantic_model, trace=None):
    """Vet the router's primary with the signals the router lacks: grain-hint
    ("for each X"), word ORDER (the subject in "X with their Y"), and junction
    awareness — via the multi-signal score_anchors. Overrides the router ONLY when
    score_anchors wins by ≥ ANCHOR_CONFIDENCE_MARGIN, so the router still decides
    every case where it's already right. Returns the (possibly corrected) primary.

    Applied once after the router, so BOTH the single-table and join paths get the
    corrected grain."""
    try:
        from config import ANCHOR_VET_ROUTER, ANCHOR_CONFIDENCE_MARGIN
    except Exception:
        return primary
    if not ANCHOR_VET_ROUTER or not primary:
        return primary
    from query.join_planner import score_anchors
    from veda.runtime import get_graph
    graph = get_graph()
    gtabs = set(graph.get("tables", []))
    if primary not in gtabs:
        return primary

    # Explicit grain declaration ("for each incident" / "per role") names the grain
    # outright — it overrides the router regardless of score margin. BUT if the grain phrase
    # is a group-by DIMENSION the UNIFIED GRAPH maps to a column ("for each workflow state"
    # → workflow_state), it's NOT the anchor entity — skip the override so "SLA hours for each
    # workflow state" stays on sla_config (the dimension demotion below then drops `workflow`).
    from retrieval.query_enrichment import _singularize
    gm = re.search(r"\b(?:for each|for every|for all|per|each|every)\s+([a-z]+(?:\s+[a-z]+)?)",
                   query.lower())
    _grain_is_dim = False
    if gm:
        try:
            from graph.query_graph import get_graph as _ugg
            _g2 = _ugg()
            if _g2 is not None and " " in gm.group(1):
                _grain_is_dim = any((_g2.node(c) or {}).get("type") == "COLUMN"
                                    for c in _g2.resolve_term(gm.group(1).strip()))
        except Exception:
            _grain_is_dim = False
    if gm and not _grain_is_dim:
        grain = _singularize(gm.group(1).split()[0])
        exact = [t for t in gtabs
                 if {_singularize(w) for w in t.split("_") if len(w) > 2} == {grain}]
        if len(exact) == 1:
            if trace is not None:
                trace.set("anchor_selection", anchor=exact[0], confidence=1.0,
                          router_primary=primary, overrode_router=(exact[0] != primary),
                          source="grain-hint")
                trace.note("anchor_selection",
                           f"explicit grain '{grain}' → anchor {exact[0]!r}")
            return exact[0]

    score = {}
    for r in results:
        t = r.col_id.split(".")[0]
        score[t] = max(score.get(t, 0.0), r.final_score)
    cand = [t for t in sorted(score, key=score.get, reverse=True) if t in gtabs][:6]
    # Ensure name-matched tables are candidates even if retrieval ranked them low —
    # else a grain-hint table ("for each incident" → incident) can't be scored/pinned.
    from retrieval.query_enrichment import _singularize
    qtok = {_singularize(w) for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 2}
    named = [t for t in gtabs
             if {_singularize(w) for w in t.split("_") if len(w) > 2} & qtok]
    cand = list(dict.fromkeys(cand + named + [primary]))

    # Graph-driven dimension demotion: a multi-word phrase the UNIFIED GRAPH maps to a COLUMN
    # ("workflow state" → workflow_state) is a group-by DIMENSION, not the anchor entity. Drop
    # candidates whose ONLY name-match is consumed by such a phrase, so "SLA hours for each
    # workflow state" anchors on sla_config, not the `workflow` table. Never drops the router's
    # own pick. Structural (real column names via the graph), no hardcoding.
    try:
        from graph.query_graph import get_graph as _ug
        _g = _ug()
        if _g is not None:
            _words = re.findall(r"[a-z]+", query.lower())
            _dim_toks = set()
            for _i in range(len(_words) - 1):
                _phrase = _words[_i] + " " + _words[_i + 1]
                if any((_g.node(c) or {}).get("type") == "COLUMN"
                       for c in _g.resolve_term(_phrase)):
                    _dim_toks |= {_singularize(_words[_i]), _singularize(_words[_i + 1])}
            if _dim_toks:
                _filtered = [t for t in cand if t == primary or not (
                    (m := {_singularize(w) for w in t.split("_") if len(w) > 2} & qtok)
                    and m <= _dim_toks)]
                if len(_filtered) >= 2:
                    cand = _filtered
    except Exception:
        pass

    if len(cand) < 2:
        return primary

    ranked = score_anchors(query, cand, score, graph=graph)
    if not ranked:
        return primary

    # IDF re-rank (downstream of score_anchors — never touches the scorer): nudge each
    # candidate by how RARE its best query-relevant column is. A table matched on a distinctive
    # column (sla_hours, incident_no) outranks one matched only on shared columns (status, id).
    # Bounded, flag-guarded, try/except → on any issue the original ranking stands.
    try:
        from config import IDF_RERANK_ENABLED, IDF_RERANK_WEIGHT
    except Exception:
        IDF_RERANK_ENABLED, IDF_RERANK_WEIGHT = False, 0.3
    if IDF_RERANK_ENABLED and len(ranked) > 1:
        try:
            from query.column_idf import col_idf_norm
            _tbl_idf = {}
            for r in results[:12]:                       # top reranked = query-relevant cols
                _t = r.col_id.split(".")[0]
                _cn = r.col_id.split(".", 1)[1] if "." in r.col_id else ""
                _tbl_idf[_t] = max(_tbl_idf.get(_t, 0.0), col_idf_norm(_cn))
            for a in ranked:
                a.score = round(a.score + IDF_RERANK_WEIGHT * _tbl_idf.get(a.table, 0.0), 4)
            ranked.sort(key=lambda a: a.score, reverse=True)
        except Exception:
            pass

    # Value-match re-rank (downstream of score_anchors, same rationale as the IDF re-rank
    # above): a query token that exact-matches a sampled CATEGORY value on a candidate table
    # ("debit" -> accounts_generalledger.entry_type) is boosted, since that's direct evidence
    # the query is ABOUT that table's rows — evidence the lexical/position/retrieval/graph
    # signals in score_anchors never see (they score table NAMES and column retrieval, not
    # sampled cell values). Without this, a name-lexical tie is broken arbitrarily, the value
    # filter ends up scoped to the wrong (unchosen) table, and qualifier_completeness
    # correctly but unhelpfully refuses the whole query later instead of the anchor ever
    # being fixed. Bounded, flag-guarded, try/except → on any issue the original ranking
    # stands.
    try:
        from config import VALUE_ANCHOR_RERANK_ENABLED, VALUE_ANCHOR_RERANK_WEIGHT
    except Exception:
        VALUE_ANCHOR_RERANK_ENABLED, VALUE_ANCHOR_RERANK_WEIGHT = False, 0.25
    if VALUE_ANCHOR_RERANK_ENABLED and len(ranked) > 1:
        try:
            from query.value_arbiter import arbitrate, column_values_typed_lookup
            from veda.runtime import _pg as _pgc_anchor
            _arb = arbitrate(query, column_values_typed_lookup(_pgc_anchor))
            _value_tables = {t.table for t in _arb.value_filters if t.table}
            if _value_tables:
                for a in ranked:
                    if a.table in _value_tables:
                        a.score = round(a.score + VALUE_ANCHOR_RERANK_WEIGHT, 4)
                ranked.sort(key=lambda a: a.score, reverse=True)
        except Exception:
            pass

    top = ranked[0]
    pscore = next((r.score for r in ranked if r.table == primary), 0.0)
    chosen, overrode = primary, False
    # Protect a NAMED, non-junction subject from a pure-score override toward a table the
    # query names LESS: "document categories" → document_category_master must not lose to the
    # richer `documents`. Junctions are EXEMPT — they should still yield to the base entity
    # ("roles with their permissions": role_permissions → role). Structural, no hardcoding.
    from veda.planning import _junction_tables
    _junctions = _junction_tables(graph, semantic_model)
    _protect = (primary not in _junctions
                and len(qtok & _name_toks(primary)) > len(qtok & _name_toks(top.table)))
    if top.table != primary and not _protect and (top.score - pscore) >= ANCHOR_CONFIDENCE_MARGIN:
        chosen, overrode = top.table, True
    if trace is not None:
        second = ranked[1].score if len(ranked) > 1 else 0.0
        trace.set("anchor_selection", anchor=chosen, confidence=round(top.score, 3),
                  margin=round(top.score - second, 3), router_primary=primary,
                  overrode_router=overrode, source=("score_anchors" if overrode else "router"))
        for r in ranked[:5]:
            trace.cand("anchor_selection", "alternatives",
                       {"table": r.table, "score": r.score, "signals": r.signals})
        if overrode:
            trace.note("anchor_selection",
                       f"router {primary!r} → {chosen!r} (margin {round(top.score - pscore, 3)})")
    return chosen


_NAME_CONNECTIVES = {"and", "or", "of", "to", "by"}

_TABLE_VOCAB_CACHE = {"v": None}


def _table_vocab():
    """Domain vocabulary harvested from column-name tokens across the WHOLE semantic
    model — schema-driven, not hardcoded, so it generalizes to any schema with this
    pattern (not just Homzhub). Column names use normal snake_case even when the table
    names they belong to don't (verification_document_type_id on
    assets_assetverificationdocument), so this gives good coverage for segmenting the
    compound table-name blobs in _name_toks. Cached for the process lifetime."""
    if _TABLE_VOCAB_CACHE["v"] is None:
        vocab = set()
        try:
            from config import SEMANTIC_MODEL_FILE
            with open(SEMANTIC_MODEL_FILE) as f:
                sm = json.load(f)
            for col_id in sm.get("columns", {}):
                _, _, col = col_id.partition(".")
                for tok in col.split("_"):
                    tok = tok.lower()
                    if len(tok) > 2 and tok not in _NAME_CONNECTIVES:
                        vocab.add(tok)
        except Exception:
            vocab = set()
        _TABLE_VOCAB_CACHE["v"] = vocab
    return _TABLE_VOCAB_CACHE["v"]


def _segment_compound(word, vocab, min_len=3, max_len=20):
    """DP word-break: decompose `word` into known `vocab` words, or None if no full
    decomposition exists. Minimizes piece count (dynamic-programming shortest token
    sequence) so 'leaselisting' -> ['lease', 'listing'], not spurious short chunks."""
    n = len(word)
    best = [None] * (n + 1)
    best[0] = []
    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):
            if best[j] is None:
                continue
            seg = word[j:i]
            if len(seg) < min_len or seg not in vocab:
                continue
            cand = best[j] + [seg]
            if best[i] is None or len(cand) < len(best[i]):
                best[i] = cand
    return best[n]


def _name_toks(table_name):
    """Singularized entity tokens of a TABLE NAME, minus structural connectives.

    ADDITIVE compound-segmentation: Django auto-generates table names by concatenating
    a multi-word model name with NO internal underscore (assets_assetverificationdocument,
    assets_leaselisting) — the plain split("_") below can never recover "document" or
    "listing" from that blob. For any long (>8 char) segment, also try a vocabulary-driven
    word-break (_segment_compound) and UNION its pieces in — the original blob token is
    always kept too, so this can only add new matches, never remove an existing one."""
    from retrieval.query_enrichment import _singularize
    toks = set()
    for tok in table_name.split("_"):
        if len(tok) <= 2 or tok in _NAME_CONNECTIVES:
            continue
        toks.add(_singularize(tok))
        if len(tok) > 8:
            parts = _segment_compound(tok, _table_vocab())
            if parts:
                toks.update(_singularize(p) for p in parts)
    return toks
