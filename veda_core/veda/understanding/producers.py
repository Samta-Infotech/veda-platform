"""veda.understanding.producers — the deterministic parsers, demoted to FRAME PRODUCERS.

Each producer reads the question and returns a frame FRAGMENT (a dict of the slots it is
sure about) or None. They emit no SQL and pick no table. They are the same classifiers
the old chain branched on — `parse_ranking`, `run_temporal_parser`, `aggregate_operator`
/ `grouped_mode` / `grouped_count_mode`, `parse_numeric_predicates`, `existence_mode` —
plus a value-glossary producer over the Stage 1 vocabulary.

Placeholder concepts a producer may emit (grounding resolves them on the chosen entity):
    "@date"     the entity card's business date
    "@measure"  the entity card's first key measure
    "@id"       the table's primary key

Merge rule (Stage 2.5): a fragment slot that the SLM left EMPTY is filled; one the SLM
filled identically is CONFIRMED (provenance "slm+producer:<name>"); one that DIFFERS is a
CONFLICT recorded under `frame.provenance["conflict:<slot>"]` with the producer's value
kept as an alternative — frame_grounding resolves it by which candidate grounds, never by
branch order.

`config.FRAME_PRODUCERS_DISABLED` disables producers by name (the Stage 6 ablation knob).
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from veda.understanding.frame import Frame, FrameFilter, FrameOrder, FrameTime, slot_values

_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "twenty": 20, "fifty": 50, "hundred": 100}


def _disabled() -> frozenset:
    try:
        import config
        return frozenset(getattr(config, "FRAME_PRODUCERS_DISABLED", frozenset()) or ())
    except Exception:
        return frozenset()


# ── individual producers ─────────────────────────────────────────────────────────────
_UPDATE_VERB = re.compile(r"\b(modified|updated|changed|edited)\b")
_CREATE_VERB = re.compile(r"\b(created|added|put up|listed|registered|signed up|joined)\b")
_PAID_VERB = re.compile(r"\b(paid|settled)\b")
_ID_ORDER = re.compile(r"\b(internal|system|processing|record)\s+(?:system\s+)?(?:processing\s+)?id\b|\bby\s+(?:their\s+)?id\b")


def p_ranking(q: str, vocab=None) -> Optional[Dict[str, Any]]:
    """ranking words → order (+ limit). Temporal basis → "@date" (or the verb-bound date:
    'most recently modified' → "updated"); metric basis → the 'by <X>' phrase if the
    question names one, else "@measure"."""
    from query.ranking_parser import parse_ranking
    r = parse_ranking(q)
    ql = q.lower()
    if not (r.ranked or r.sort_requested or _ID_ORDER.search(ql)):
        return None
    frag: Dict[str, Any] = {}
    if r.top_n:
        frag["limit"] = int(r.top_n)
    if _ID_ORDER.search(ql):
        frag["order"] = {"concept": "@id", "dir": r.direction or "desc", "explicit": True}
        return frag
    # an explicit "sort/order … by X" names the order outright — it beats a recency word
    ms = re.search(r"\b(?:sort(?:ed)?|order(?:ed)?|arrange(?:d)?|rank(?:ed)?)\s+(?:\w+\s+){0,4}?by\s+(?:their\s+|the\s+|its\s+)?(?:respective\s+)?([a-z][a-z ]{2,30}?)(?:\s+(?:in|for|with|and|descending|ascending|order)\b|[?.,]|$)", ql)
    if ms and not re.search(r"\b(date|time|recen|latest|newest|oldest)", ms.group(1)):
        d = "desc" if re.search(r"\b(descending|reverse|highest|largest|most)\b", ql) else "asc"
        frag["order"] = {"concept": ms.group(1).strip(), "dir": d, "explicit": True}
        return frag
    if r.basis == "temporal":
        concept = "@date"
        explicit = False
        if _UPDATE_VERB.search(ql):
            concept, explicit = "updated", True      # the verb binds WHICH date
        frag["order"] = {"concept": concept, "dir": r.direction or "desc", "explicit": explicit}
    elif r.basis == "metric":
        m = re.search(r"\bby\s+(?:their\s+|the\s+|its\s+)?([a-z][a-z ]{2,30}?)(?:\s+(?:in|for|with|and|that|which|descending|ascending|order)\b|[?.,]|$)", ql)
        # "top 5 … by price" / "cheapest": the ranking word fixes the direction explicitly
        frag["order"] = {"concept": (m.group(1).strip() if m else "@measure"), "dir": r.direction or "desc",
                         "explicit": bool(r.ranked)}
    elif r.sort_requested:
        m = re.search(r"\b(?:sort(?:ed)?|order(?:ed)?)\s+(?:them\s+)?by\s+(?:their\s+|the\s+|its\s+)?(?:respective\s+)?([a-z][a-z ]{2,30}?)(?:\s+(?:in|for|with|and|descending|ascending|order)\b|[?.,]|$)", ql)
        if m:
            d = "asc" if re.search(r"\b(ascending|alphabetical|a to z|a-z)\b", ql) else \
                "desc" if re.search(r"\b(descending|reverse)\b", ql) else "asc"
            frag["order"] = {"concept": m.group(1).strip(), "dir": d}
    return frag or None


def p_temporal(q: str, vocab=None) -> Optional[Dict[str, Any]]:
    """an explicit window → time.window; the concept stays None (grounding picks the card's
    business date unless a verb binds another date)."""
    from query.temporal_parser import run_temporal_parser
    res = run_temporal_parser(q)
    tf = res.temporal_filter
    if not tf or not (tf.start or tf.end):
        return None
    # 'recent' / 'recently' / 'latest' name no period — they are an ORDER, not a window
    vague = re.compile(r"^\s*(most\s+)?(recent|recently|latest|lately|newest|to date|ever|so far)\s*$", re.I)
    exprs = [e for e in (res.raw_expressions or []) if str(e).strip()]
    if exprs and all(vague.match(str(e)) for e in exprs):
        return None
    ql = q.lower()
    concept = "updated" if _UPDATE_VERB.search(ql) else "created" if re.search(r"\bcreated\b", ql) else None
    return {"time": {"concept": concept, "window": {"from": tf.start, "to": tf.end}}}


def p_aggregate(q: str, vocab=None) -> Optional[Dict[str, Any]]:
    """aggregation operator + grouping phrase ('per/by/each <dim>')."""
    from veda.planning import aggregate_operator, grouped_mode, grouped_count_mode, superlative_mode
    ql = q.lower()
    op = aggregate_operator(q)
    frag: Dict[str, Any] = {}
    if op:
        agg = {"AVG": "avg", "SUM": "sum", "MIN": "min", "MAX": "max", "COUNT": "count"}[op]
        # 'highest/lowest' are ranking words far more often than MAX/MIN over the set
        if agg in ("min", "max") and not re.search(r"\b(maximum|minimum|max|min)\b", ql):
            agg = None
        if agg == "count" and re.search(r"\b(how many|number of|count)\b", ql) and \
                re.search(r"\b(distinct|different|unique)\b", ql):
            agg = "count_distinct"
        if agg:
            frag["aggregation"] = agg
    grp = grouped_mode(q) or grouped_count_mode(q)
    if grp and not superlative_mode(q):
        m = re.search(r"\b(?:per|for each|each|by|grouped by|broken down by)\s+([a-z][a-z ]{2,30}?)(?:\s+(?:in|for|with|and|that|which|descending|ascending|order)\b|[?.,]|$)", ql)
        if m:
            frag["group_by"] = [m.group(1).strip()]
    elif re.search(r"\bgroup(?:ed)?\s+(?:\w+\s+){0,6}?(?:by|based on)\s+", ql):
        m = re.search(r"\b(?:by|based on)\s+(?:their\s+|the\s+|its\s+)?([a-z][a-z ]{2,30}?)(?:\s+(?:in|for|with|and|descending|ascending|order)\b|[?.,]|$)", ql)
        if m:
            frag["group_by"] = [m.group(1).strip()]
            if "aggregation" not in frag:
                frag["aggregation"] = "count"
    return frag or None


def p_numeric(q: str, vocab=None) -> Optional[Dict[str, Any]]:
    """numeric comparisons → filters on the measure phrase nearest BEFORE the comparator
    ('priced above 10,000' → price > 10000; 'fall between 100 and 50,000' → @measure)."""
    from query.numeric_filter import parse_numeric_predicates
    preds = parse_numeric_predicates(q)
    if not preds:
        return None
    ql = q.lower()
    measure_words = set()
    if vocab is not None:
        for ent in (vocab.measure_glossary or {}).values():
            for p in ent.get("phrases") or []:
                measure_words.add(p.lower())
    out = []
    for p in preds:
        i = ql.find(p.phrase.lower())
        before = ql[:max(0, i)].split()[-4:]
        concept = "@measure"
        for n in range(len(before), 0, -1):
            cand = " ".join(before[-n:]).strip(" ,.?")
            if cand in measure_words or re.sub(r"(ed|d)$", "", cand) in measure_words:
                concept = cand
                break
        if p.op == "BETWEEN":
            out.append({"concept": concept, "op": "between", "value": [p.low, p.high]})
        else:
            out.append({"concept": concept, "op": p.op, "value": p.low})
    return {"filters": out}


def p_values(q: str, vocab=None) -> Optional[Dict[str, Any]]:
    """value-glossary phrases present in the question → '=' filters carrying the STORED
    value ("on the market" → status = APPROVED). Longest phrase wins per column; a phrase
    that maps to values of several columns emits one fragment per column (grounding keeps
    the one on the chosen entity)."""
    if vocab is None:
        return None
    ql = " " + re.sub(r"[^a-z0-9 ]+", " ", q.lower()) + " "
    ql = re.sub(r"\s+", " ", ql)
    try:
        from veda.understanding.frame_grounding import name_hits
        named_tables = {h["table"] for h in name_hits(vocab, q)}
    except Exception:
        named_tables = set()
    out = []
    for key, vmap in (vocab.value_glossary or {}).items():
        t, _, col = key.partition(".")
        best = None
        found = []
        for v, phrases in vmap.items():
            for ph in phrases or []:
                core = re.sub(r"\s+", " ", ph.lower()).strip()
                try:
                    from ingestion.vocabulary import phrase_is_grammar
                    if phrase_is_grammar(core):
                        continue
                except Exception:
                    pass
                p = " " + core + " "
                # a one-word phrase ('up', 'live') is too common to identify a value on its
                # own unless its table is one the question names
                if " " not in core and (len(core) < 5 or t not in named_tables):
                    continue
                if p in ql:
                    if best is None or len(p) > best[0]:
                        best = (len(p), v, ph)
                    if v not in [x[0] for x in found]:
                        found.append((v, ph))
        if len(found) > 1:
            # several values of one column named ("debited or credited") → IN over them
            out.append({"concept": col.replace("_", " "), "op": "in", "value": [v for v, _ in found],
                        "_table": t, "_phrase": " / ".join(ph for _, ph in found)})
        elif best:
            out.append({"concept": col.replace("_", " "), "op": "=", "value": best[1],
                        "_table": t, "_phrase": best[2]})
    return {"filters": out} if out else None


def p_existence(q: str, vocab=None) -> Optional[Dict[str, Any]]:
    from veda.planning import existence_mode
    m = existence_mode(q)
    return {"_existence": m} if m else None


PRODUCERS: List[Tuple[str, Callable]] = [
    ("ranking", p_ranking),
    ("temporal", p_temporal),
    ("aggregate", p_aggregate),
    ("numeric", p_numeric),
    ("values", p_values),
    ("existence", p_existence),
]


def run_producers(q: str, vocab=None) -> Dict[str, Dict[str, Any]]:
    off = _disabled()
    out: Dict[str, Dict[str, Any]] = {}
    for name, fn in PRODUCERS:
        if name in off:
            continue
        try:
            frag = fn(q, vocab)
        except Exception:
            frag = None
        if frag:
            out[name] = frag
    return out


# ── merge ────────────────────────────────────────────────────────────────────────────
def _same(a, b) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()


def merge(frame: Frame, fragments: Dict[str, Dict[str, Any]]) -> Frame:
    """Fill / confirm / record-conflict, per slot. Never overwrites a filled SLM slot."""
    fr = frame
    alts: Dict[str, List[Any]] = fr.provenance.setdefault("_alternatives", {})  # type: ignore[assignment]
    for name, frag in fragments.items():
        tag = f"producer:{name}"
        if "_existence" in frag:
            fr.provenance["existence"] = frag["_existence"]
        if "limit" in frag:
            if fr.limit is None:
                fr.limit = frag["limit"]
                fr.provenance["limit"] = tag
            elif fr.limit == frag["limit"]:
                fr.provenance["limit"] = f"slm+{tag}"
            else:
                fr.provenance["conflict:limit"] = tag
                alts.setdefault("limit", []).append(frag["limit"])
        if "order" in frag:
            o = frag["order"]
            if fr.order is None or (o.get("explicit") and not (
                    _same(fr.order.concept, o["concept"]) and fr.order.dir == o["dir"])):
                if fr.order is not None:
                    # the question's own "sort by X" / verb-bound date overrides the SLM's
                    # reading; the SLM's stays as the alternative grounding may fall back to
                    alts.setdefault("order", []).append({"concept": fr.order.concept, "dir": fr.order.dir})
                    fr.provenance["conflict:order"] = "slm"
                fr.order = FrameOrder(concept=o["concept"], dir=o["dir"])
                fr.provenance["order"] = tag
            elif fr.order.dir == o["dir"] and (_same(fr.order.concept, o["concept"]) or o["concept"].startswith("@")):
                fr.provenance["order"] = f"slm+{tag}"
            else:
                fr.provenance["conflict:order"] = tag
                alts.setdefault("order", []).append(o)
        if "time" in frag:
            t = frag["time"]
            if fr.time is None or fr.time.window is None:
                concept = (fr.time.concept if fr.time else None) or t.get("concept")
                fr.time = FrameTime(concept=concept, window=t.get("window"))
                fr.provenance["time"] = tag
            elif fr.time.window != t.get("window"):
                # the parser's ISO window is EXACT; the SLM's is a guess at date arithmetic
                # ("last month" came back as 2023-10 on 2026-09-25) — the parser wins, the
                # SLM's stays only as a recorded alternative
                alts.setdefault("time", []).append({"concept": fr.time.concept, "window": fr.time.window})
                fr.time = FrameTime(concept=fr.time.concept or t.get("concept"), window=t.get("window"))
                fr.provenance["time"] = tag
                fr.provenance["conflict:time"] = "slm"
            else:
                fr.provenance["time"] = f"slm+{tag}"
        if "time" in frag:
            _dated = [x for x in fr.filters if x.op in (">", ">=", "<", "<=", "between")
                      and re.search(r"\b(date|dated|time|day|month|year|joined|created|when)\b", x.concept.lower())
                      and x.__dict__.get("_producer") != "numeric"]
            for x in _dated:
                fr.filters.remove(x)
                fr.provenance.setdefault("dropped_filters", []).append(f"{x.concept} {x.op} {x.value} (time window owns it)")
        if "aggregation" in frag:
            if fr.aggregation == "none":
                fr.aggregation = frag["aggregation"]
                fr.provenance["aggregation"] = tag
            elif fr.aggregation == frag["aggregation"]:
                fr.provenance["aggregation"] = f"slm+{tag}"
            else:
                fr.provenance["conflict:aggregation"] = tag
                alts.setdefault("aggregation", []).append(frag["aggregation"])
        if "group_by" in frag:
            if not fr.group_by:
                fr.group_by = list(frag["group_by"])
                fr.provenance["group_by"] = tag
            elif not any(_same(a, b) for a in fr.group_by for b in frag["group_by"]):
                alts.setdefault("group_by", []).append(frag["group_by"])
                fr.provenance["conflict:group_by"] = tag
        for f in frag.get("filters") or []:
            existing = [x for x in fr.filters if _same(x.concept, f["concept"]) or
                        (f.get("_table") and _same(x.value, f["value"]))]
            if name == "numeric":
                cmp_ops = (">", ">=", "<", "<=", "between")
                cmp = [x for x in fr.filters if x.op in cmp_ops]

                def _nums(v):
                    try:
                        return sorted(float(z) for z in (v if isinstance(v, (list, tuple)) else [v]))
                    except (TypeError, ValueError):
                        return None
                want = _nums(f["value"])
                same_vals = [x for x in cmp if _nums(x.value) == want]
                if len(same_vals) > 1:
                    # the SLM restated one range on two concepts ('payment date' and 'paid
                    # amount'); keep the one that is not a date phrase
                    keep = [x for x in same_vals if not re.search(r"\b(date|time|day|month|year)\b", x.concept)] or same_vals[:1]
                    for x in same_vals:
                        if x not in keep[:1]:
                            fr.filters.remove(x)
                    same_vals = keep[:1]
                existing = same_vals or cmp[:1]
            if not existing:
                ff = FrameFilter(concept=f["concept"], op=f["op"], value=f["value"])
                if f.get("_table"):
                    ff.__dict__["_table"] = f["_table"]
                    ff.__dict__["_phrase"] = f.get("_phrase")
                ff.__dict__["_producer"] = name
                fr.filters.append(ff)
                fr.provenance[f"filter:{f['concept']}"] = tag
            else:
                x = existing[0]
                if name == "numeric":
                    # the literal numbers/op are the parser's; the concept stays the SLM's
                    # unless the SLM had none — or put a money range on a DATE phrase
                    x.op, x.value = f["op"], f["value"]
                    if (x.concept.startswith("@") and not f["concept"].startswith("@")) or \
                            re.search(r"\b(date|time|day|month|year)\b", x.concept):
                        x.concept = f["concept"]
                elif name == "values":
                    # the stored value is the glossary's; keep the SLM's concept wording
                    x.__dict__["_table"] = f.get("_table")
                    x.__dict__["_phrase"] = f.get("_phrase")
                    x.__dict__["_glossary_value"] = f["value"]
                    if f["op"] == "in":
                        # the question names several values of this column; the SLM kept one
                        x.op, x.value = "in", list(f["value"])
                x.__dict__["_producer"] = name
                fr.provenance[f"filter:{x.concept}"] = f"slm+{tag}"
    return fr


def vote(frames: List[Frame]) -> Tuple[Frame, List[str]]:
    """Self-consistency: slot-wise majority over samples; a slot without a majority is
    `uncertain` (it keeps the first sample's value, and becomes the clarify if grounding
    cannot settle it)."""
    if not frames:
        return Frame(), []
    views = [slot_values(f) for f in frames]
    base = frames[0]
    uncertain = []
    for slot in views[0]:
        vals = [v[slot] for v in views]
        best = max(set(map(repr, vals)), key=lambda r: sum(1 for v in vals if repr(v) == r))
        n = sum(1 for v in vals if repr(v) == best)
        if n * 2 <= len(vals):
            uncertain.append(slot)
            continue
        winner = frames[[repr(v[slot]) for v in views].index(best)]
        setattr(base, slot, getattr(winner, slot))
    base.uncertain = uncertain
    return base, uncertain
