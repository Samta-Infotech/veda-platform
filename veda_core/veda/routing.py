"""VEDA · L2/L3 — semantic table routing + primary-table selection."""
import os, re, sys, time, logging, threading
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


def select_primary_table(results, query, semantic_model, trace=None):
    """Choose the target table by blending three signals:

      1. SEMANTIC ROUTING (primary) — cosine of the query vs the per-table
         embeddings (built from name + business_purpose + columns). This finally
         uses the rich table understanding, vector-based not keyword-based.
      2. COLUMN MAX score — best single-column retrieval score per table (so a
         small correct MASTER table isn't drowned by a wide table on count).
      3. LEXICAL name match (singularized) — tiebreaker / fallback when the
         table_embeddings store isn't built yet.

    `trace`: optional ExplainTrace (veda/explain.py) — when given, records the
    candidate scoring under "table_routing" (a DIFFERENT section than
    vet_primary's own "anchor_selection", since that runs after this and may
    override the choice — both stay visible in the trace, not overwritten).
    None (the default) is a no-op, same as every other trace-accepting call
    in this module."""
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
        if q_tokens & _name_toks(t, semantic_model):
            candidates.add(t)

    if not candidates:
        if trace is not None:
            trace.set("anchor_selection", anchor=None, source="router",
                      confidence=0.0, margin=0.0, alternatives=[])
        return None

    best, best_score, scored = None, -1e9, {}
    for t in candidates:
        # Strip structural connectives so a wide name can't match on glue words
        # (investigation_AND_research_counter_party must not match query token "and").
        name_tokens = _name_toks(t, semantic_model)
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
        scored[t] = combined
        if combined > best_score:
            best_score, best = combined, t
    if trace is not None:
        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second_score
        # Non-tautological confidence (docs/RETRIEVAL_DECISION_LAYER_AUDIT.md): NOT
        # best_score/best_score (always 1.0 regardless of how close the runner-up
        # was) — margin relative to the runner-up, so a near-tie scores low and a
        # clear win scores high, always in (0, 1].
        confidence = margin / (margin + second_score) if (margin + second_score) > 0 else 1.0
        trace.set("anchor_selection", anchor=best, source="router",
                  confidence=round(min(max(confidence, 0.0), 1.0), 3),
                  margin=round(margin, 3),
                  alternatives=[{"table": t, "score": round(s, 3)} for t, s in ranked[:8]])
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
        exact = [t for t in gtabs if _name_toks(t, semantic_model) == {grain}]
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
    named = [t for t in gtabs if _name_toks(t, semantic_model) & qtok]
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
                    (m := _name_toks(t, semantic_model) & qtok)
                    and m <= _dim_toks)]
                if len(_filtered) >= 2:
                    cand = _filtered
    except Exception:
        pass

    if len(cand) < 2:
        return primary

    ranked = score_anchors(query, cand, score, graph=graph, sm=semantic_model)
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
            # QSR (query/resolution.py): artifact-first typed lookup — the old
            # column_values_typed_lookup(runtime._pg) queried the SOURCE DB, which
            # has no column_values table, so this whole channel silently returned
            # [] in production. Also new: FK-CLOSURE evidence — a value living one
            # join away ('captured' in a status lookup) credits the REFERENCING
            # table (accounts_paymenttransaction.payment_status_id) at reduced
            # weight. Anchor evidence only; predicates still use direct referents.
            from query.value_arbiter import arbitrate
            from query.resolution import typed_value_lookup, closed_value_tables
            try:
                from config import VALUE_ANCHOR_CLOSED_WEIGHT
            except Exception:
                VALUE_ANCHOR_CLOSED_WEIGHT = 0.15
            _arb = arbitrate(query, typed_value_lookup())
            _value_tables = {t.table for t in _arb.value_filters if t.table}
            _closed_tables = set()
            for _t in _arb.value_filters:
                _closed_tables.update(closed_value_tables(_t.span))
            _closed_tables -= _value_tables
            if _value_tables or _closed_tables:
                for a in ranked:
                    if a.table in _value_tables:
                        a.score = round(a.score + VALUE_ANCHOR_RERANK_WEIGHT, 4)
                    elif a.table in _closed_tables:
                        a.score = round(a.score + VALUE_ANCHOR_CLOSED_WEIGHT, 4)
                ranked.sort(key=lambda a: a.score, reverse=True)
        except Exception:
            pass

    # QSR typed-evidence re-rank (downstream of score_anchors, same pattern as the
    # IDF/value re-ranks): role-disciplined evidence — entity words vote (grammar
    # words don't), values vote for their owning tables (FK-closed at reduced
    # weight, label stores demoted). This is the signal that anchors "annual sum of
    # financial records by transaction years" on a *transaction table instead of
    # whatever retrieval noise ranked first. Bounded, flag-guarded, failure-safe.
    try:
        from config import TYPED_ANCHOR_RERANK_ENABLED, TYPED_ANCHOR_RERANK_WEIGHT
    except Exception:
        TYPED_ANCHOR_RERANK_ENABLED, TYPED_ANCHOR_RERANK_WEIGHT = False, 0.2
    if TYPED_ANCHOR_RERANK_ENABLED and len(ranked) > 1:
        try:
            from query.resolution import typed_anchor_evidence
            _ev, _ = typed_anchor_evidence(query, semantic_model)
            _mx = max(_ev.values()) if _ev else 0.0
            if _mx > 0:
                for a in ranked:
                    a.score = round(a.score + TYPED_ANCHOR_RERANK_WEIGHT
                                    * (_ev.get(a.table, 0.0) / _mx), 4)
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

    # ── Single-table ambiguity gate (counterpart of planning.py's multi-table gate,
    # which this path never reaches). Fires ONLY when (a) the top two anchors are
    # within the commit margin, (b) each is named by DISJOINT query tokens — two
    # different subjects, not one entity vs its wider sibling (those share tokens
    # and stay with the existing rules), and (c) the winner is not the query's
    # plain grammatical subject (its name token both earliest among the pair and
    # genuinely early in the sentence). Refuse-over-guess: a coin-flip between two
    # differently-named subjects should ask, not silently pick a grain.
    try:
        from config import ANCHOR_SINGLE_GATE_ENABLED, ANCHOR_SUBJECT_POS_MIN
    except Exception:
        ANCHOR_SINGLE_GATE_ENABLED, ANCHOR_SUBJECT_POS_MIN = False, 0.7
    if ANCHOR_SINGLE_GATE_ENABLED and len(ranked) > 1:
        a, b = ranked[0], ranked[1]
        if (a.score - b.score) < ANCHOR_CONFIDENCE_MARGIN:
            am = qtok & _name_toks(a.table, semantic_model)
            bm = qtok & _name_toks(b.table, semantic_model)
            if am and bm and not (am & bm):
                _qw = [_singularize(w) for w in re.findall(r"[a-z]+", query.lower())
                       if len(w) > 2]

                def _fpos(mset):
                    for _i, _w in enumerate(_qw):
                        if _w in mset:
                            return _i
                    return None

                fa, fb = _fpos(am), _fpos(bm)
                subject_clear = (fa is not None
                                 and (fb is None or fa < fb)
                                 and (1.0 - fa / (len(_qw) or 1)) >= ANCHOR_SUBJECT_POS_MIN)
                if not subject_clear:
                    msg = (f"ambiguous subject — should this be about {a.table} or "
                           f"{b.table}? (confidence {a.score} vs {b.score})")
                    if trace is not None:
                        trace.set("anchor_selection", anchor=None,
                                  confidence=round(a.score, 3),
                                  margin=round(a.score - b.score, 3),
                                  router_primary=primary, overrode_router=False,
                                  source="single-table-gate")
                        trace.note("anchor_selection",
                                   f"sub-margin disjoint subjects {a.table!r} vs "
                                   f"{b.table!r} → clarify")
                    return {"clarify": msg, "candidates": [a.table, b.table]}

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


def _name_toks(table_name, sm=None):
    """Singularized entity tokens of a TABLE NAME, minus structural connectives.
    Schema-vocabulary segmentation (semantic/name_tokens) opens up concatenated
    names ("paymenttransaction" → payment, transaction); falls back to the plain
    underscore split on any failure or when the flag is off."""
    from retrieval.query_enrichment import _singularize
    try:
        from semantic.name_tokens import table_tokens
        return {t for t in table_tokens(table_name, sm) if t not in _NAME_CONNECTIVES}
    except Exception:
        return {_singularize(tok) for tok in table_name.split("_")
                if len(tok) > 2 and tok not in _NAME_CONNECTIVES}


def recommended_projection(primary, allowed_columns, results, sm, query, must_include=None):
    """Deterministic, LLM-free business-facing SELECT list for `primary` —
    composes THREE signals VEDA already computes elsewhere, never re-ranking
    columns from scratch:

      1. Entity identity  — veda/generation.py::_resolve_display_column(), the
         SAME governed label-column resolver (human-override file, then a
         name/title/label/business_role/*_no heuristic) already used by every
         OTHER caller that needs "the column that represents this entity"
         (generate_join_sql's per-alias hints, the grouped-breakdown planner,
         build_aggregate_sql's dimension display — planning.py:197,430,
         generation.py:273). Reusing it here (instead of independently
         reading the compiled concept registry's OWN default_display_columns)
         means there is exactly ONE mechanism answering "what's this table's
         display column" across the whole codebase, not two that could
         silently disagree. Falls back to the concept registry only if this
         returns nothing (e.g. no metadata at all for `primary`) — a
         secondary source, never a competing decision for the same table.
      2. User intent       — THIS query's own per-column retrieval relevance
         (`results[].final_score`, already computed by the 5-signal retrieval
         engine before this function ever runs).
      3. Business importance — columns whose ingestion-computed
         `importance_class` is "HIGH" (veda/ingestion/deterministic_metadata.py).

    Plus two safety overrides, both added before RECOMMENDED_PROJECTION_MAX_COLS
    trims the list so they always survive the cap:
      a. `must_include` — columns the CALLER already knows are structurally
         required for this SQL to read sensibly (e.g. the column a ranked
         result is actually ORDER BY'd on) — never invented here, only
         accepted from the caller, which already computed it for the WHERE/
         ORDER BY clause.
      b. a column the user's query literally names (by column name or a
         known business alias).

    `allowed_columns` (the validation allow-list) is read-only input here and
    is NEVER itself modified — this function only decides what to SELECT,
    never what SQL is allowed to reference. Every candidate is intersected
    with `allowed_columns` so a stale/mismatched registry entry can never
    introduce a column that hasn't been validated.

    Falls back to `allowed_columns` verbatim (today's behavior, unchanged) if
    every signal above comes up empty — a missing registry, empty retrieval
    results, or a semantic model with no importance metadata must never
    produce a degraded (or empty) SELECT list."""
    from config import RECOMMENDED_PROJECTION_MAX_COLS

    allowed = list(allowed_columns or [])
    allowed_set = set(allowed)
    cols_meta = (sm or {}).get("columns", {})
    picked: list = []

    def _add(col):
        if col in allowed_set and col not in picked:
            picked.append(col)

    # a. MUST-INCLUDE — structurally required by the caller (e.g. the active
    # ORDER BY column), first so it always survives the cap.
    for c in (must_include or []):
        _add(c)

    # b. SAFETY OVERRIDE — an explicitly user-named column (by column name or a
    # known business alias) must appear regardless of importance/relevance
    # rank. Same "does the user's own wording name this column" idea
    # qualifier_completeness (veda/validation.py) already applies to
    # generated SQL — applied here to candidate columns instead, not a new
    # matching scheme.
    query_l = (query or "").lower()
    if query_l:
        for c in allowed:
            meta = cols_meta.get(f"{primary}.{c}", {}) or {}
            names = [c.replace("_", " ")] + [str(a).lower() for a in (meta.get("aliases") or [])]
            if any(len(n) > 3 and n in query_l for n in names):
                _add(c)

    # 1. Entity identity — ONE canonical resolver (see docstring); the concept
    # registry's own default_display_columns is only a fallback for when
    # _resolve_display_column has nothing at all for this table.
    try:
        from veda.generation import _resolve_display_column
        dc = _resolve_display_column(primary, sm)
        if dc:
            _add(dc)
        else:
            from semantic import registry as reg
            concept = reg.active().get("concepts", {}).get(primary) or {}
            for cid in concept.get("default_display_columns") or []:
                _, _, col = str(cid).partition(".")
                if col:
                    _add(col)
    except Exception:
        pass

    # 2. User intent — this query's own retrieval relevance for primary's
    # columns (RetrievalResult.final_score, retrieval/retrieval_engine_phase3.py)
    # — read, not recomputed.
    try:
        candidates = sorted(
            (r for r in (results or []) if getattr(r, "col_id", "").split(".")[0] == primary),
            key=lambda r: r.final_score, reverse=True,
        )
        for r in candidates[:RECOMMENDED_PROJECTION_MAX_COLS]:
            _add(r.column_name)
    except Exception:
        pass

    # 3. Business importance — HIGH-importance columns (ingestion-computed,
    # veda/ingestion/deterministic_metadata.py::compute_importance_class).
    for c in allowed:
        meta = cols_meta.get(f"{primary}.{c}", {}) or {}
        if meta.get("importance_class") == "HIGH":
            _add(c)

    if not picked:
        return allowed   # every signal was empty — unchanged behavior, never degrade
    return picked[:RECOMMENDED_PROJECTION_MAX_COLS]
