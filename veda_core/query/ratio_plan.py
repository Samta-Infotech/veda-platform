# =============================================================================
# query/ratio_plan.py
# VEDA — deterministic ratio planner (QSR-backed, fast-path style).
#
# "What is the ratio of X to Y?" is a DIVIDED PAIR of aggregates on ONE anchor:
#   · a side that names a grounded VALUE ("paid", "cancelled") becomes a
#     CASE-WHEN-filtered SUM over the anchor's measure(s)
#   · a side that names a MEASURE COLUMN ("paid amount") becomes a plain SUM
# Single table scan — no join, structurally fan-out-free (the Tier-2 failure
# this replaces was SUM over a 1:N join of a lookup's label column).
#
# Typed-role discipline:
#   · the anchor must OWN a declared measure (candidate_measure_columns) —
#     value-evidence alone must not anchor a ratio on a lookup/label table
#   · viable-anchor ties break by how many of the user's measure words the
#     table's columns match; still tied → fall through (refuse-over-guess)
#   · with no user-named measure, EVERY declared measure is ratio'd (cap 3) —
#     complete, not guessed (same policy as the grouped planner)
# Refuse-over-guess: a side whose words ground on NO anchor value/measure is
# unanswerable HERE — return a grounded clarify listing the anchor's real FK
# value domain ("'completed' isn't a payment status; statuses here are
# attempted, cancelled, created, paid") in milliseconds, instead of letting
# Tier-2 burn 60s to fail a firewall gate.
#
# Returns:  FastPathResult | ("clarify", msg) | None (→ full pipeline)
# =============================================================================
from __future__ import annotations

import re
from collections import defaultdict

ANCHOR_MARGIN = 0.15          # evidence lead required unless measure-hits break the tie
MEASURE_NAME_MIN = 2          # side words must match a measure column ≥2 times to
                              # claim the MEASURE role ("paid amount" yes, "paid" no —
                              # single hits stay eligible as value filters)
MAX_MEASURES = 3


def _q(ident: str) -> str:
    return '"' + str(ident).replace('"', '""') + '"'


def _slug(v) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(v).lower()).strip("_")[:24] or "side"


def _words(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def try_ratio_plan(query: str, sm=None):
    try:
        return _try(query, sm)
    except Exception:
        return None                      # failure-safe: fall through to full pipeline


def _try(query: str, sm=None):
    from veda.planning import ratio_mode
    rm = ratio_mode(query)
    if not rm:
        return None

    # positive predicates only — same rule as the grouped/superlative builders
    from config import QUERY_GRAMMAR
    ql = f" {query.lower()} "
    for _g in ("negation", "exclusion"):
        for _w in QUERY_GRAMMAR.get(_g, []):
            if (" " in _w and _w in ql) or re.search(rf"\b{re.escape(_w)}\b", ql):
                return None

    from query.resolution import resolve, typed_anchor_evidence, domain_via
    from query.fast_path import FastPathResult
    if sm is None:
        from veda.runtime import _load_scoped_sm
        sm = _load_scoped_sm()

    spans = {tr.span: tr for tr in resolve(query, sm)}
    tables_meta = sm.get("tables", {})

    # ── anchor: highest typed evidence among tables that OWN a measure ──────
    evidence, why_ev = typed_anchor_evidence(query, sm)
    why_ev = defaultdict(list, why_ev)
    mhits: dict = defaultdict(int)
    for tr in spans.values():
        for cid in tr.measures:
            mhits[cid.rpartition(".")[0]] += 1
    viable = [(t, s) for t, s in evidence.items()
              if tables_meta.get(t, {}).get("candidate_measure_columns")]
    if not viable:
        return None
    viable.sort(key=lambda kv: (-kv[1], -mhits[kv[0]], kv[0]))
    anchor, top = viable[0]
    if len(viable) > 1:
        t2, s2 = viable[1]
        if top - s2 < ANCHOR_MARGIN and mhits[anchor] <= mhits[t2]:
            return None                  # ambiguous subject → full pipeline
    cols_meta = sm.get("columns", {})
    anchor_cols = {c.split(".", 1)[1] for c in cols_meta
                   if c.split(".", 1)[0] == anchor}
    anchor_toks = set(re.findall(r"[a-z]+", anchor))
    try:
        from semantic.name_tokens import table_tokens
        anchor_toks |= set(table_tokens(anchor))
    except Exception:
        pass

    def _is_entity_word(w) -> bool:
        tr = spans.get(w)
        return (w in anchor_toks
                or bool(tr and any(t == anchor for t, _ in tr.entities)))

    # ── resolve each side to a MEASURE column or grounded value FILTERS ─────
    # Filters are grouped BY COLUMN ("by_col") so the same-dimension rule below
    # can reconcile the two sides: 'cancelled' exists in BOTH order_status and
    # payment_status domains — ANDing them silently under-counts.
    def _resolve_side(text):
        side = {"measure": None, "by_col": {}, "why": [], "unmatched": [],
                "ftables": set(), "fcols": []}
        swords = [w for w in _words(text)]
        # measure claim: score anchor measures by NON-entity side words — the
        # entity word ("payment") aliases every amount column, which must not
        # let 'paid payment value' read as the paid_amount measure and silently
        # drop the PAID status filter.
        ms: dict = defaultdict(int)
        for w in swords:
            tr = spans.get(w)
            if tr is None or _is_entity_word(w):
                continue
            for cid in tr.measures:
                t, _, c = cid.rpartition(".")
                if t == anchor and c in anchor_cols:
                    ms[c] += 1
        best = sorted(ms, key=lambda c: (-ms[c], c))
        if best and ms[best[0]] >= MEASURE_NAME_MIN \
                and (len(best) == 1 or ms[best[0]] > ms[best[1]]):
            side["measure"] = best[0]
            side["why"].append(f"measure {best[0]} (named)")
            return side
        # value filters on the anchor (positive), grouped by column; dedup by
        # (column, value) so a direct+closed double-hit yields one predicate
        seen = set()

        def _add(col, pred, why, ftabs=(), fcs=()):
            e = side["by_col"].setdefault(col, {"preds": [], "why": []})
            e["preds"].append(pred)
            e["why"].append(why)
            side["ftables"] |= set(ftabs)
            side["fcols"] += [col, *fcs]

        for w in swords:
            tr = spans.get(w)
            if tr is None or tr.grammar:
                continue
            hit = False
            for r in tr.values["direct"]:
                key = (r["column"], str(r["value_raw"]).lower(), "d")
                if r["table"] == anchor and r["column"] in anchor_cols and key not in seen:
                    seen.add(key)
                    lit = str(r["value_raw"]).replace("'", "''")
                    _add(r["column"],
                         f"LOWER(CAST(a.{_q(r['column'])} AS TEXT)) = LOWER('{lit}')",
                         f"filter {r['column']}='{r['value_raw']}'")
                    hit = True
            for r in tr.values["closed"]:
                key = (r["column"], str(r["value_raw"]).lower(), "c")
                if r["table"] == anchor and r["column"] in anchor_cols and key not in seen:
                    seen.add(key)
                    lit = str(r["value_raw"]).replace("'", "''")
                    _add(r["column"],
                         f"a.{_q(r['column'])} IN (SELECT v.{_q(r['via_pk'])} "
                         f"FROM {_q(r['via_table'])} v WHERE "
                         f"LOWER(CAST(v.{_q(r['via_column'])} AS TEXT)) = LOWER('{lit}'))",
                         f"filter {r['column']} via {r['via_table']}.{r['via_column']}"
                         f"='{r['value_raw']}'",
                         ftabs=[r["via_table"]], fcs=[r["via_pk"], r["via_column"]])
                    hit = True
            if not hit and not tr.unresolved and not _is_entity_word(w):
                side["unmatched"].append(w)
        return side

    sides = [_resolve_side(rm["side_a"]), _resolve_side(rm["side_b"])]
    labels = [_slug(rm["side_a"]), _slug(rm["side_b"])]

    # ── same-dimension discipline: a ratio compares two slices of ONE dimension.
    # A value living in several domains ('cancelled' ∈ order_status AND
    # payment_status) must not AND across columns (silent under-count). When both
    # sides carry filters, restrict to their SHARED column(s); a side still
    # spanning several columns is genuinely ambiguous → clarify. Multiple values
    # within one column OR together (IN-list semantics).
    if sides[0]["by_col"] and sides[1]["by_col"]:
        shared = set(sides[0]["by_col"]) & set(sides[1]["by_col"])
        if shared:
            for s in sides:
                s["by_col"] = {c: v for c, v in s["by_col"].items() if c in shared}
    for s, txt in zip(sides, (rm["side_a"], rm["side_b"])):
        if len(s["by_col"]) > 1:
            return ("clarify",
                    f"'{txt}' matches more than one field "
                    f"({', '.join(sorted(s['by_col']))}) — which should the ratio use?")
        s["preds"] = []
        for _c, e in s["by_col"].items():
            p = e["preds"][0] if len(e["preds"]) == 1 \
                else "(" + " OR ".join(e["preds"]) + ")"
            s["preds"].append(p)
            s["why"] += e["why"]

    # ── ungroundable side → grounded clarify with the anchor's REAL domain ──
    bad = [w for s in sides if not s["measure"] and not s["preds"]
           for w in s["unmatched"]]
    if any(not s["measure"] and not s["preds"] for s in sides):
        if not bad:
            return None                  # nothing to name → full pipeline
        dom, domcol = [], None
        try:
            from query.join_planner import load_graph
            qw = set(_words(query))
            cands = []
            for e in load_graph().get("edges", []):
                if e.get("source_table") == anchor and e.get("cardinality") == "N:1":
                    d = domain_via(anchor, e["source_column"], limit=6)
                    if len(d) > 1:
                        ov = len(qw & {t for t in e["source_column"].split("_")
                                       if len(t) > 2})
                        cands.append((-ov, len(d), e["source_column"], d))
            if cands:
                cands.sort()
                _, _, domcol, dom = cands[0]
        except Exception:
            pass
        who = " / ".join(f"'{w}'" for w in dict.fromkeys(bad))
        if dom:
            return ("clarify",
                    f"{who} doesn't match any value in this data — "
                    f"{domcol} values here are {', '.join(dom[:5])}; which did you mean?")
        return ("clarify", f"{who} doesn't match any value in this data — "
                           f"rephrase with a value that exists.")

    # ── measures: named per side, else the shared declared candidates ───────
    cand = [c for c in (tables_meta.get(anchor, {})
                        .get("candidate_measure_columns") or []) if c in anchor_cols]
    measures = []
    if sides[0]["measure"] and sides[1]["measure"]:
        pass                             # each side sums its own column
    else:
        shared = [m for m in [sides[0]["measure"], sides[1]["measure"]] if m]
        measures = shared or cand[:MAX_MEASURES]
        if not measures:
            return None

    # ── single-scan SQL ──────────────────────────────────────────────────────
    sels, ftables, fcols, why = [], {anchor}, [], []
    for s in sides:
        ftables |= s["ftables"]
        fcols += s["fcols"]
        why += s["why"]

    def _sum(side_i, m):
        preds = sides[side_i]["preds"]
        col = f"a.{_q(m)}"
        if preds:
            return f"SUM(CASE WHEN {' AND '.join(preds)} THEN {col} ELSE 0 END)"
        return f"SUM({col})"

    if sides[0]["measure"] and sides[1]["measure"]:
        ma, mb = sides[0]["measure"], sides[1]["measure"]
        fcols += [ma, mb]
        sels = [f"{_sum(0, ma)} AS {_q('total_' + ma)}",
                f"{_sum(1, mb)} AS {_q('total_' + mb)}",
                f"{_sum(0, ma)} / NULLIF({_sum(1, mb)}, 0) AS {_q('ratio')}"]
        why.insert(0, f"ratio → SUM({ma}) / SUM({mb})")
    else:
        fcols += measures
        for m in measures:
            sels += [f"{_sum(0, m)} AS {_q(labels[0] + '_' + m)}",
                     f"{_sum(1, m)} AS {_q(labels[1] + '_' + m)}",
                     f"{_sum(0, m)} / NULLIF({_sum(1, m)}, 0) AS {_q(m + '_ratio')}"]
        why.insert(0, f"ratio per measure ({', '.join(measures)}): "
                      f"[{rm['side_a']}] / [{rm['side_b']}]")

    sql = f"SELECT {', '.join(sels)} FROM {_q(anchor)} a"
    why.insert(1, f"anchor {anchor} (typed evidence: {'; '.join(why_ev[anchor][:3])})")
    return FastPathResult(sql=sql, tables=ftables,
                          columns=list(dict.fromkeys(fcols)), primary=anchor,
                          route="ratio.scan", why=why)
