# =============================================================================
# query/superlative_plan.py
# VEDA — deterministic superlative-by-dimension planner (Phase B of
# ARCHITECTURE_ROOT_CAUSE_PLAN.md; the builder Phase 2 detection was waiting for).
#
# "Which <dimension> has the highest <measure> among <filters>?" is a RANKED
# AGGREGATION: GROUP BY dim ORDER BY AGG(measure) LIMIT 1. Every referent is
# resolvable from QSR artifacts alone (query/resolution.py) — no retrieval, no
# reranker, no LLM — so this runs fast-path style, BEFORE the expensive stages.
#
# Typed-role discipline (the fix for the mis-anchor class):
#   · anchor evidence = ENTITY spans + VALUE ownership (direct 1.0 / FK-closed 0.6)
#   · the interrogative DIMENSION span ("which CATEGORY…") contributes ZERO anchor
#     evidence — it's the GROUP BY target
#   · grammar-flagged spans ("value" is language-layer) contribute ZERO anchor
#     evidence — their measure reading is used for measure resolution instead
# Refuse-over-guess: an unresolvable/ambiguous dimension or measure yields a
# GROUNDED CLARIFY listing the anchor's actual options — never a guessed grouping.
#
# Returns:  FastPathResult  (plan built)
#         | ("clarify", msg)  (grounded question)
#         | None  (not a superlative / not confidently plannable → full pipeline)
# =============================================================================
from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

ANCHOR_MARGIN = 0.15          # top table must lead #2 by this in typed evidence
DIRECT_W, CLOSED_W = 1.0, 0.6
LABEL_STORE_W = 0.3           # a span's direct table when the same span FK-closes
                              # elsewhere: the value's home is a label store serving
                              # other tables — weak evidence for anchoring on it


def _q(ident: str) -> str:
    return '"' + str(ident).replace('"', '""') + '"'


def _words(query: str):
    return re.findall(r"[a-z0-9]+", query.lower())


def _human(table: str, sm=None) -> str:
    """'accounts_generalledgercategory' → 'general ledger category': drop the app
    prefix, segment fused words via the schema vocabulary (name_tokens). The user
    re-asks in THESE words ('reminder categories'), which is exactly what entity
    resolution consumes — the raw table name is developer-speak."""
    tail = table.split("_", 1)[1] if "_" in table else table
    try:
        from semantic.name_tokens import segment_token
        segs = []
        for piece in tail.split("_"):
            segs += segment_token(piece, sm)
        return " ".join(segs) or tail
    except Exception:
        return tail.replace("_", " ")


def _explicit_measure_operator(query: str):
    """The aggregate operator a superlative EXPLICITLY names for its rank measure,
    restricted to the NON-DIRECTIONAL operators (AVG / SUM) — or None. "which project
    has the highest AVERAGE carpet area" ranks by AVG, not SUM. MIN/MAX are excluded
    on purpose: their trigger words ('highest'/'lowest'/'largest') ARE the superlative
    direction, so treating them as a per-group operator would silently reinterpret a
    plain "highest <measure>" ranking. Generic SQL semantics (config.AGGREGATE_
    OPERATORS), no schema vocabulary."""
    from config import AGGREGATE_OPERATORS
    ql = f" {query.lower()} "
    for op in ("AVG", "SUM"):
        for w in AGGREGATE_OPERATORS.get(op, []):
            hit = (w in ql) if " " in w else bool(re.search(rf"\b{re.escape(w)}\b", ql))
            if hit:
                return op
    return None


def try_superlative_plan(query: str, sm=None):
    try:
        from veda.planning import superlative_mode
        sup = superlative_mode(query)
        if not sup:
            return None
        return _try(query, sm, {"kind": "superlative",
                                "direction": sup.get("superlative", "max"),
                                "term": sup["term"],
                                # PRESERVE an explicitly-named rank operator (AVG/SUM);
                                # default SUM keeps every existing superlative unchanged.
                                "op": _explicit_measure_operator(query) or "SUM"})
    except Exception:
        return None                      # failure-safe: fall through to full pipeline


def try_grouped_plan(query: str, sm=None):
    """Non-ranked sibling of the superlative shape: "how much does each <dim>
    contribute" / "total <measure> per <dim>" → GROUP BY dim, SUM(measure) on the
    typed-evidence anchor. Same machinery, same refuse-over-guess guards; only the
    tail differs (full breakdown instead of LIMIT 1)."""
    try:
        from veda.planning import grouped_mode
        _gm = grouped_mode(query)
        if not _gm:
            return None
        return _try(query, sm, {"kind": "grouped", "op": _gm.get("op", "SUM")})
    except Exception:
        return None                      # failure-safe: fall through to full pipeline


def _try(query: str, sm=None, mode=None):
    if not mode:
        return None

    # Requested aggregate operator — PRESERVED from the query (generic SQL semantics,
    # config.AGGREGATE_OPERATORS), never assumed. Both the grouped breakdown
    # (grouped_mode → AVG/SUM/MIN/MAX) and the LIMIT-1 superlative rank
    # (_explicit_measure_operator → AVG/SUM) carry their operator on mode["op"];
    # default SUM keeps every prior plan byte-for-byte.
    agg = (mode.get("op") or "SUM").upper()

    # ── negation/exclusion scope (BOTH shapes): "excluding paid entries" must
    # INVERT the paid filter — dropping it broadens the answer, applying it
    # positively silently inverts it. Generic scope rule (grammar-level, no
    # schema vocabulary): exactly ONE marker in the query + exactly ONE
    # anchor-filterable value span (established after anchor selection below) →
    # that span's predicates are negated, NULL-preserving. Zero markers → all
    # positive (today's behavior). Two+ markers, or a marker whose target can't
    # be pinned to exactly one value span → scope is ambiguous → fall through
    # to the full pipeline (refuse-over-guess).
    from config import QUERY_GRAMMAR
    ql = f" {query.lower()} "
    _markers = 0
    for _g in ("negation", "exclusion"):
        for _w in QUERY_GRAMMAR.get(_g, []):
            if " " in _w:
                _markers += ql.count(_w)
            else:
                _markers += len(re.findall(rf"\b{re.escape(_w)}\b", ql))
    if _markers > 1:
        return None
    negated = _markers == 1

    # ── frequency superlative: "appears most frequently" / "most common" ranks
    # by COUNT(*), not by a measure column; "what percentage of the dataset"
    # adds a share-of-total projection. Grammar words only — and their spans are
    # CONSUMED by the shape: they must neither vote for an anchor ('percentage'
    # coincides with sampled lookup values) nor trip the unconsumed-span guard.
    _qwords = set(re.findall(r"[a-z]+", ql))
    freq_words = {w for w in QUERY_GRAMMAR.get("frequency", []) if w in _qwords}
    portion_words = {w for w in QUERY_GRAMMAR.get("portion", []) if w in _qwords}
    count_rank = mode["kind"] == "superlative" and bool(freq_words)
    wants_pct = count_rank and bool(portion_words)
    _shape_spans = (freq_words | portion_words) if count_rank else set()

    from query.resolution import resolve, domain_via
    from query.fast_path import FastPathResult
    if sm is None:
        from veda.runtime import _load_scoped_sm
        sm = _load_scoped_sm()

    spans = {tr.span: tr for tr in resolve(query, sm)}
    words = _words(query)

    # ── the dimension span: names the GROUP BY target, never the anchor.
    # GROUPED prefers the grouping-word path — "per CATEGORY" is the explicit
    # group-by declaration, while "what is the total …" openers make the
    # interrogative scan grab a measure word. SUPERLATIVE stays interrogative-
    # first ("which CATEGORY has the highest …"). One content word either way.
    def _dim_after_grouping():
        for i, w in enumerate(words):
            if w in ("per", "each", "by"):
                for nxt in words[i + 1:]:
                    tr = spans.get(nxt)
                    if tr is None or tr.grammar:
                        continue
                    if tr.dimensions:
                        return nxt
                    break              # only the immediate next content word
        return None

    def _dim_after_interrogative():
        try:
            wi = next(i for i, w in enumerate(words)
                      if w in ("which", "what", "who", "whose"))
        except StopIteration:
            return None
        for w in words[wi + 1:]:
            tr = spans.get(w)
            if tr is None or tr.grammar:
                continue
            if tr.dimensions:
                return w
            break                      # only the immediate next content word
        return None

    if mode["kind"] == "grouped":
        dim_span = _dim_after_grouping() or _dim_after_interrogative()
    else:
        dim_span = _dim_after_interrogative()

    # ── the dimension PHRASE ("transaction type"): consecutive words after the
    # dim span that keep a dimension reading anywhere. The WHOLE phrase is
    # excluded from anchor evidence — trailing phrase words ("type") carry
    # entity readings (accounts_paymenttype, assets_assettype, …) that crowd
    # the anchor margin with lookup tables the user made no subject claim about.
    dim_phrase = {dim_span} if dim_span else set()
    if dim_span:
        wi = words.index(dim_span)
        for w in words[wi + 1:]:
            tr = spans.get(w)
            if tr is None or tr.grammar or not tr.dimensions:
                break
            dim_phrase.add(w)

    # ── typed anchor evidence (shared scorer — see resolution.typed_anchor_evidence
    # for the role-discipline rules). The dimension phrase contributes nothing: it
    # names the GROUP BY, and its incidental lookup/entity hits are noise. Shape
    # spans (frequency/portion words) likewise contribute nothing.
    from query.resolution import typed_anchor_evidence
    evidence, why_ev = typed_anchor_evidence(
        query, sm, exclude_spans=dim_phrase | _shape_spans)
    why_ev = defaultdict(list, why_ev)

    # ── measure-aggregation anchor recovery: the typed-evidence scorer credits
    # ENTITY and VALUE spans only (by design — a measure word must not mis-anchor).
    # But a pure "<op> <measure> per <dim>" question has NO entity/value span for its
    # own anchor: its only anchor signal is that ONE table owns BOTH the requested
    # measure column AND the grouping dimension column ("carpet area" votes for the
    # carpet-area-UNIT lookup entity, leaving the asset table that actually holds the
    # measure with zero evidence). When exactly one table co-owns a matched measure
    # and the dimension AND the evidence winner does NOT own the dimension (so the
    # normal flow would mis-anchor or bail), adopt that co-owner. Schema-agnostic:
    # column ownership comes from the semantic model (METRIC/MONETARY measures, the
    # CATEGORY dimension — identifiers are already excluded from tr.dimensions),
    # never from names. Only relaxes anchoring for this shape; every other query is
    # unchanged.
    _co_owner = None
    if dim_span and not count_rank and mode.get("kind") in ("grouped", "superlative"):
        _meas_tabs = {cid.rpartition(".")[0] for sp, tr in spans.items()
                      for cid in tr.measures}
        _dim_tabs = {cid.rpartition(".")[0] for cid in spans[dim_span].dimensions}
        _co = _meas_tabs & _dim_tabs
        _ev_top = max(evidence, key=evidence.get) if evidence else None
        if len(_co) == 1 and (_ev_top is None or _ev_top not in _dim_tabs):
            _co_owner = next(iter(_co))

    if _co_owner:
        anchor = _co_owner
        why_ev[anchor].append("owns the requested measure + grouping dimension")
    else:
        if not evidence:
            # Dimension named but ZERO entity/value evidence anywhere ("which
            # category appears most frequently in the records" — records of WHAT?):
            # ask, grounded in the schema. Strictly better than falling through —
            # the full pipeline's dim-word anchor trap ends in a wrong-table gate
            # refusal after 30–60s (measured twice on this shape).
            if dim_span and spans[dim_span].entities:
                opts = [_human(t, sm) for t, _ in spans[dim_span].entities][:5]
                return ("clarify",
                        f"'{dim_span}' of which records? This data has: "
                        f"{', '.join(opts)} — name the entity to count.")
            return None
        ranked = sorted(evidence.items(), key=lambda kv: (-kv[1], kv[0]))
        anchor, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if top - second < ANCHOR_MARGIN:
            # dim-ownership tie-break: entity families tie on shared name words
            # (paymenttransaction / settlement / settlementlog all match 'payment');
            # only candidates owning EVERY word of the asked dimension phrase
            # ('transaction type' → transaction_type) stay in the running.
            owners = [(t, s) for t, s in ranked
                      if dim_phrase and all(
                          any(cid.rpartition(".")[0] == t for cid in spans[w].dimensions)
                          for w in dim_phrase)]
            if owners and (len(owners) == 1
                           or owners[0][1] - owners[1][1] >= ANCHOR_MARGIN):
                anchor, top = owners[0]
            else:
                return None                          # ambiguous subject → full pipeline

    cols_meta = sm.get("columns", {})
    anchor_cols = {c.split(".", 1)[1] for c in cols_meta if c.split(".", 1)[0] == anchor}

    # ── refuse-over-guess guard (BEFORE any clarify/plan): every referent-bearing
    # content span must be CONSUMED by this plan (dim span, anchor value/entity/
    # measure word). A span whose only referents live on OTHER tables ("completed"
    # → inspection completion dims, but NOT a payment value) is a qualifier this
    # plan would silently drop — bail to the full pipeline, which refuses/clarifies
    # honestly. Truly unresolved spans ("contributes", "among") are connective filler.
    used = dim_phrase | {dim_span} | _shape_spans
    for span, tr in spans.items():
        if (any(r["table"] == anchor for r in tr.values["direct"] + tr.values["closed"])
                or any(t == anchor for t, _ in tr.entities)
                or any(cid.rpartition(".")[0] == anchor
                       for cid in tr.measures + tr.dimensions)):
            used.add(span)
    for span, tr in spans.items():
        if tr.grammar or tr.unresolved or span in used:
            continue
        return None

    # ── negation target: the single anchor-filterable value span the marker
    # scopes to. More or fewer than one → can't pin the negation → full pipeline.
    neg_span = None
    if negated:
        _filterable = [span for span, tr in spans.items()
                       if span != dim_span
                       and any(r["table"] == anchor and r["column"] in anchor_cols
                               for r in tr.values["direct"] + tr.values["closed"])]
        if len(_filterable) != 1:
            return None
        neg_span = _filterable[0]

    # ── measure: candidates = measure-typed spans ∩ anchor, SCORED by how many
    # of the user's measure words each column matches ("paid amount" → paid_amount
    # matches two words, balance_amount one) — the user's own words disambiguate.
    # A genuine tie ("value") → clarify with the real options. A frequency
    # superlative ranks by COUNT(*) and needs no measure at all.
    measures, measure = [], None
    if not count_rank:
        mscore: dict = defaultdict(int)
        for span, tr in spans.items():
            if span == neg_span:
                continue          # one role per span: the negated value ("paid entries
            for cid in tr.measures:  # excluded") must not also name the measure column
                t, _, c = cid.rpartition(".")
                if t == anchor:
                    mscore[c] += 1
        cand = sorted(mscore, key=lambda c: (-mscore[c], c))
        if cand and len(cand) > 1 and mscore[cand[0]] > mscore[cand[1]]:
            cand = [cand[0]]                       # user's words name one column best
        named = bool(cand)
        if not cand:
            # No user-named measure → the schema's own declared candidates.
            cand = [c for c in (sm.get("tables", {}).get(anchor, {})
                                .get("candidate_measure_columns") or []) if c in anchor_cols]
        if not cand:
            return None
        if len(cand) > 1:
            if mode["kind"] == "grouped" and not named:
                # "how much does each contribute" names NO amount column — the
                # complete, guess-free answer is EVERY declared measure per group;
                # the user reads the column they meant. A superlative can't do this
                # (LIMIT 1 needs ONE rank key) and a user who DID name an amount but
                # tied ("amount") gets asked which one — both clarify below.
                cand = cand[:5]
            else:
                _verb = (f"aggregated ({agg})" if mode["kind"] == "grouped" else "ranked")
                return ("clarify",
                        f"which amount should be {_verb} — {', '.join(cand[:4])}?")
        measures = cand
        measure = measures[0]      # rank/sort key; sole measure when len == 1

    # ── dimension: the dim PHRASE ("transaction type", not just "transaction")
    # scores each of the anchor's dimension columns by matched words — the user's
    # own words disambiguate, exactly as with measures. Consume consecutive words
    # while each keeps a dimension reading on the anchor. Unique best → pick;
    # genuine tie → clarify with the top options; no match → clarify with the
    # anchor's real categorical columns (refuse-over-guess, never a guessed group).
    dim_col = None
    dim_words = []
    if dim_span:
        wi = words.index(dim_span)
        for w in words[wi:]:
            tr = spans.get(w)
            if tr is None or tr.grammar or not any(
                    cid.rpartition(".")[0] == anchor for cid in tr.dimensions):
                break
            dim_words.append(w)
        dscore: dict = defaultdict(int)
        for w in dim_words:
            for cid in spans[w].dimensions:
                t, _, c = cid.rpartition(".")
                if t == anchor and c in anchor_cols:
                    dscore[c] += 1
        ranked_dims = sorted(dscore, key=lambda c: (-dscore[c], c))
        if ranked_dims and (len(ranked_dims) == 1
                            or dscore[ranked_dims[0]] > dscore[ranked_dims[1]]):
            dim_col = ranked_dims[0]
        elif ranked_dims:
            return ("clarify", f"group by which of {', '.join(ranked_dims[:5])}?")
        else:
            opts = sorted(c.split(".", 1)[1] for c, m in cols_meta.items()
                          if c.startswith(anchor + ".")
                          and (m or {}).get("semantic_type", "").upper()
                          in ("CATEGORY", "CATEGORICAL"))
            if opts:
                return ("clarify",
                        f"'{dim_span}' doesn't name a specific field here — group the "
                        f"result by which of: {', '.join(opts[:5])}?")
            return None
    if dim_col is None:
        return None

    # ── one role per span: words that NAMED the chosen measure or dimension are
    # spent — their incidental value readings must not also become filters
    # ("paid amount" names the measure; 'paid' must not add order_status='PAID').
    spent = set(dim_words)
    for span, tr in spans.items():
        if span == neg_span:
            continue      # the negation target's role is fixed: it IS the filter —
                          # even when the fallback measure is a column it names
                          # ('paid' names paid_amount; the exclusion must survive)
        if any(f"{anchor}.{m}" in tr.measures for m in measures):
            spent.add(span)

    # ── grounded filters on the anchor: direct values as predicates; FK-closed
    # values as label subqueries (the label is NOT a legal literal for the FK) ──
    where, why = [], []
    ftables = {anchor}
    fcols = [dim_col, *measures]      # FULL column manifest — the AST validator
    for span, tr in spans.items():    # rejects any reference not declared here
        if span in spent:
            continue
        # A value living in SEVERAL anchor domains ('cancelled' ∈ order_status
        # AND payment_status) is genuinely ambiguous — ANDing the readings
        # silently under-counts (rows cancelled in only one sense drop out).
        # Same-dimension discipline as the ratio builder: ask, don't guess.
        _cols_hit = {r["column"] for r in tr.values["direct"] + tr.values["closed"]
                     if r["table"] == anchor and r["column"] in anchor_cols}
        if len(_cols_hit) > 1:
            return ("clarify",
                    f"'{span}' matches more than one field "
                    f"({', '.join(sorted(_cols_hit))}) — which did you mean?")
        # Inverted predicates for the negation target are NULL-PRESERVING: a row
        # with no value in the column is not the excluded value (bare <>/NOT IN
        # silently drops NULL rows — an invisible under-count).
        _neg = neg_span is not None and span == neg_span
        for r in tr.values["direct"]:
            if r["table"] == anchor and r["column"] in anchor_cols:
                _p = (f"LOWER(CAST(a.{_q(r['column'])} AS TEXT)) = "
                      f"LOWER('{str(r['value_raw']).replace(chr(39), chr(39)*2)}')")
                if _neg:
                    _p = f"(a.{_q(r['column'])} IS NULL OR NOT ({_p}))"
                where.append(_p)
                fcols.append(r["column"])
                why.append(f"{'exclude' if _neg else 'filter'} "
                           f"{r['column']}='{r['value_raw']}'")
        for r in tr.values["closed"]:
            if r["table"] == anchor and r["column"] in anchor_cols:
                lit = str(r["value_raw"]).replace("'", "''")
                _sub = (f"(SELECT v.{_q(r['via_pk'])} "
                        f"FROM {_q(r['via_table'])} v "
                        f"WHERE LOWER(CAST(v.{_q(r['via_column'])} AS TEXT)) = LOWER('{lit}'))")
                if _neg:
                    where.append(f"(a.{_q(r['column'])} IS NULL OR "
                                 f"a.{_q(r['column'])} NOT IN {_sub})")
                else:
                    where.append(f"a.{_q(r['column'])} IN {_sub}")
                ftables.add(r["via_table"])
                fcols += [r["column"], r["via_pk"], r["via_column"]]
                why.append(f"{'exclude' if _neg else 'filter'} {r['column']} via "
                           f"{r['via_table']}.{r['via_column']}='{r['value_raw']}'")

    if count_rank:
        # frequency superlative: rank groups by COUNT(*); optional share-of-total
        # via a window over the grouped result — same WHERE scope as the counts,
        # single scan, no self-join. ORDER BY the expression, never the ordinal.
        order = "DESC" if mode.get("direction") == "max" else "ASC"
        projs = [f"a.{_q(dim_col)} AS {_q(dim_col)}", 'COUNT(*) AS "frequency"']
        if wants_pct:
            projs.append('COUNT(*) * 100.0 / NULLIF(SUM(COUNT(*)) OVER (), 0) '
                         'AS "pct_of_total"')
        sql = (f"SELECT {', '.join(projs)} FROM {_q(anchor)} a"
               + (f" WHERE {' AND '.join(where)}" if where else "")
               + f" GROUP BY a.{_q(dim_col)} ORDER BY COUNT(*) {order} LIMIT 1")
        why = ([f"frequency superlative '{mode['term']}' → COUNT(*) rank {order}"
                + (" + % of total" if wants_pct else ""),
                f"anchor {anchor} (typed evidence: {'; '.join(why_ev[anchor][:3])})",
                f"group by {dim_col}"] + why)
        return FastPathResult(sql=sql, tables=ftables,
                              columns=list(dict.fromkeys(fcols)), primary=anchor,
                              route=f"superlative.frequency.{order.lower()}", why=why)

    if mode["kind"] == "grouped":
        # Full breakdown: every group, sorted by its own measure (presentation only —
        # the deterministic lane, so no IR gate involved), bounded like other lanes.
        # ORDER BY repeats the aggregate EXPRESSION, never the ordinal: the
        # parameterizer lifts a bare `ORDER BY 2` into a bind value — Postgres
        # then sorts by the constant 2 (i.e. not at all), which under LIMIT 1
        # returns an arbitrary group.
        tail, route = (f" GROUP BY a.{_q(dim_col)} ORDER BY {agg}(a.{_q(measure)}) "
                       f"DESC LIMIT 100"), f"grouped.breakdown.{agg.lower()}"
        head = [f"grouped breakdown → {agg}({', '.join(measures)}) per {dim_col}"]
    else:
        order = "DESC" if mode.get("direction") == "max" else "ASC"
        tail = (f" GROUP BY a.{_q(dim_col)} ORDER BY {agg}(a.{_q(measure)}) "
                f"{order} LIMIT 1")
        route = f"superlative.group.{order.lower()}"
        head = [f"superlative '{mode['term']}' → {order} rank"]

    # grouped also projects the group size — "what categories remain and how
    # much does each contribute" asks for the groups AND their weight. With no
    # user-named measure, EVERY declared measure is aggregated with the requested
    # operator (see `measures`); the alias records which operator was applied.
    _cnt = ', COUNT(*) AS "row_count"' if mode["kind"] == "grouped" else ""
    _aggs = ", ".join(f"{agg}(a.{_q(m)}) AS {_q(agg.lower() + '_' + m)}" for m in measures)
    sql = (f"SELECT a.{_q(dim_col)} AS {_q(dim_col)}, {_aggs}{_cnt} "
           f"FROM {_q(anchor)} a"
           + (f" WHERE {' AND '.join(where)}" if where else "")
           + tail)

    why = head + [f"anchor {anchor} (typed evidence: {'; '.join(why_ev[anchor][:3])})",
                  f"group by {dim_col}, measure {agg}({', '.join(measures)})"] + why
    return FastPathResult(sql=sql, tables=ftables,
                          columns=list(dict.fromkeys(fcols)), primary=anchor,
                          route=route, why=why)
