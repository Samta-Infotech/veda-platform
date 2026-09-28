"""veda.understanding.frame_grounding — Stage 3: every frame slot → a real artifact, or a
typed clarify naming the slot and its candidates. Nothing is silently dropped.

Entity (the authority order):
  NAME       the question itself names the entity: a card's business name / plural /
             alias, contiguous or with a short gap ("properties we put up for sale" →
             "properties for sale"). The most SPECIFIC name wins (a longer phrase that
             contains another's span). A NAME-grounded entity OVERRIDES the router.
  COVERAGE   several named entities (or a named one that cannot host the frame's slots):
             the entity on which every column slot grounds wins when it is FK-linked to the
             named one ("properties on the market priced above 10,000" → sale listing, a
             child of property that carries both status and price). Unlinked ties → clarify.
  MODEL      the extractor's pick, when the question names nothing but the pick is a card.
  SYNTHETIC  nearest synthetic question's table (sim ≥ 0.80), advisory.
  RETRIEVAL  the router's primary, advisory.
Cross-source: the same name in ≥ 2 sources with no cross-source FK → clarify listing them.

Measure → measure glossary / key measures, type-checked (MONETARY for money words).
Filter  → column by glossary / domain / name; value by exact domain → value glossary
          (mapping carried as a note) → sampled lookup; unmapped → clarify with the column's
          real values.
Time    → the card's business date unless a verb binds another (updated / created / paid).
Order   → @date / @measure / @id / a date verb / a measure phrase / a (parent's) name column.
Group   → CATEGORY / IDENTIFIER / FLAG columns (or a parent's display column via FK).

A `DecisionBackend` seam (Stage 3.7) sits beside the entity and measure picks: the rules
decide; a model backend, when configured, runs in SHADOW and is logged, and may flip a
slot only when FRAME_DECISION_FLIP_ENABLED and its confidence ≥ the threshold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from veda.understanding.frame import Frame, FrameFilter, FrameOrder
from veda.understanding.vocabulary import ScopeVocab, norm_phrase, nearest_examples

NAME, COVERAGE, MODEL, SYNTHETIC, RETRIEVAL, SESSION = (
    "NAME", "NAME+COVERAGE", "MODEL", "SYNTHETIC", "RETRIEVAL", "SESSION")
AUTHORITATIVE = frozenset({NAME, COVERAGE})


# ── results ──────────────────────────────────────────────────────────────────────────
@dataclass
class GJoin:
    table: str                 # the joined (parent) table
    fk_column: str             # anchor-side FK column
    pk_column: str = "id"
    kind: str = "LEFT"


@dataclass
class GFilter:
    table: str
    column: str
    op: str
    value: Any = None
    concept: str = ""
    grounding: str = "domain"          # domain | glossary | numeric | temporal | null_check | sampled
    note: Optional[str] = None         # user-facing mapping ("'on the market' → status APPROVED")
    stype: Optional[str] = None


@dataclass
class GroundedFrame:
    frame: Frame
    anchor: str
    anchor_method: str
    source_id: Optional[str] = None
    joins: List[GJoin] = field(default_factory=list)
    measure: Optional[Tuple[str, Optional[str]]] = None      # (aggregation, column|None)
    filters: List[GFilter] = field(default_factory=list)
    group_by: List[Tuple[str, str]] = field(default_factory=list)     # (table, column)
    order: Optional[Tuple[str, str, str]] = None                      # (table, column, dir)
    limit: Optional[int] = None
    distinct: bool = False
    time: Optional[Dict[str, Any]] = None                            # {table, column, start, end}
    projection: List[Tuple[str, str]] = field(default_factory=list)   # (table, column)
    notes: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    @property
    def authoritative(self) -> bool:
        return self.anchor_method in AUTHORITATIVE


@dataclass
class FrameClarify:
    slot: str
    message: str
    candidates: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


# ── decision backend seam (Stage 3.7) ────────────────────────────────────────────────
class DecisionBackend:
    name = "rules"

    def choice(self, concept: str, candidates: List[str], context: Dict[str, Any]) -> Tuple[Optional[str], float]:
        return None, 0.0

    def score(self, question: str, card: Dict[str, Any]) -> float:
        return 0.0


class RulesBackend(DecisionBackend):
    """The rules ARE the decision; this backend only reports them (agreement baseline)."""
    name = "rules"


_BACKEND: Dict[str, DecisionBackend] = {}


def decision_backend() -> DecisionBackend:
    try:
        import config
        name = str(getattr(config, "FRAME_DECISION_BACKEND", "rules") or "rules")
    except Exception:
        name = "rules"
    if name not in _BACKEND:
        b: DecisionBackend = RulesBackend()
        if name != "rules":
            # A trained head plugs in here (module path "pkg.mod:Class"). None ships in this
            # repo yet — see the report: no labelled set big enough to train one exists.
            try:
                mod, _, cls = name.partition(":")
                import importlib
                b = getattr(importlib.import_module(mod), cls)()
            except Exception:
                b = RulesBackend()
        _BACKEND[name] = b
    return _BACKEND[name]


def _shadow(slot: str, concept: str, candidates: List[str], chosen: Optional[str], ctx, ev) -> Optional[str]:
    """Run the model backend beside the rules; log agreement; flip only when allowed."""
    b = decision_backend()
    if b.name == "rules" or not candidates:
        return chosen
    try:
        pick, conf = b.choice(concept, candidates, ctx)
    except Exception:
        return chosen
    ev.setdefault("decision_shadow", []).append(
        {"slot": slot, "rules": chosen, "model": pick, "conf": round(float(conf or 0), 3),
         "agree": pick == chosen})
    try:
        import config
        if (getattr(config, "FRAME_DECISION_FLIP_ENABLED", False) and pick in candidates
                and conf >= float(getattr(config, "FRAME_DECISION_FLIP_THRESHOLD", 0.9))):
            return pick
    except Exception:
        pass
    return chosen


# ── schema helpers ───────────────────────────────────────────────────────────────────
def _cols(sm, table) -> Dict[str, dict]:
    cols = (sm or {}).get("columns", {}) or {}
    pre = table + "."
    return {k[len(pre):]: (v or {}) for k, v in cols.items() if k.startswith(pre)}


def _st(meta) -> str:
    return str((meta or {}).get("semantic_type") or "").upper()


def _toks(s: str) -> List[str]:
    return [w for w in norm_phrase(s).split() if len(w) > 1]


_STOPW = {"the", "a", "an", "of", "their", "its", "our", "by", "respective", "current",
          "currently", "each", "per", "for", "in", "on", "with", "and", "record", "records",
          "entry", "entries", "data", "value", "values", "configuration", "configurations",
          "setting", "settings", "detail", "details", "info", "information", "type"}


def _content(s: str) -> List[str]:
    return [w for w in _toks(s) if w not in _STOPW]


def _col_toks(c: str) -> set:
    return {norm_phrase(p) for p in c.lower().split("_") if p}


def _bare(table: str) -> str:
    return table.split(".", 1)[1] if table.startswith("src") and "." in table else table


# ── the name matcher (NAME evidence) ─────────────────────────────────────────────────
def name_hits(vocab: ScopeVocab, question: str, max_gap: int = 6) -> List[Dict[str, Any]]:
    """Every card phrase the question contains — contiguous, or in order with ≤ max_gap
    extra words between its tokens. [{table, phrase, kind, start, end, ntok, gap}]."""
    qt = norm_phrase(question).split()
    out = []
    for phrase, lst in vocab.name_index().items():
        pt = phrase.split()
        if not pt:
            continue
        for s in range(len(qt)):
            if qt[s] != pt[0]:
                continue
            i, j, gap = s, 1, 0
            ok = True
            while j < len(pt):
                i += 1
                while i < len(qt) and qt[i] != pt[j]:
                    gap += 1
                    i += 1
                    if gap > max_gap:
                        break
                if i >= len(qt) or gap > max_gap:
                    ok = False
                    break
                j += 1
            if ok:
                for t, kind in lst:
                    out.append({"table": t, "phrase": phrase, "kind": kind, "start": s,
                                "end": i, "ntok": len(pt), "gap": gap})
                break
    # most specific per table: most tokens, then smallest gap
    best: Dict[str, Dict[str, Any]] = {}
    for h in out:
        b = best.get(h["table"])
        if b is None or (h["ntok"], -h["gap"]) > (b["ntok"], -b["gap"]):
            best[h["table"]] = h
    hits = list(best.values())
    # a hit whose span is strictly inside a longer hit's span (for another table) is
    # subsumed — "properties" inside "properties … for sale"
    keep = []
    for h in hits:
        if any(o is not h and o["ntok"] > h["ntok"] and o["start"] <= h["start"] and h["end"] <= o["end"]
               for o in hits):
            continue
        keep.append(h)
    for h in keep:
        # STRONG = the card's own business name / plural, a multi-word alias, or a one-word
        # alias the user wrote as a PLURAL ("payments" — rows); a singular one-word alias
        # ("sale" in "for sale", "file" in "on file") is a weak, often incidental match
        qword = qt[h["start"]] if h["start"] < len(qt) else ""
        raw = _norm_words(question)
        rw = raw[h["start"]] if h["start"] < len(raw) else ""
        h["strong"] = (h["kind"] in ("business_name", "plural") or h["ntok"] >= 2
                       or (rw.endswith("s") and rw != qword))
    # a compound "property payments": the modifier precedes the HEAD noun — the head names
    # the rows (adjacent hits, the earlier one ends right before the later one starts)
    heads = []
    for h in keep:
        if any(o is not h and o["start"] == h["end"] + 1 for o in keep):
            h["modifier"] = True
        heads.append(h)
    if any(h.get("strong") and not h.get("modifier") for h in heads):
        heads = [h for h in heads if h.get("strong") or h.get("modifier")]
    return sorted(heads, key=lambda h: (bool(h.get("modifier")), -h["ntok"], h["gap"], h["start"]))


def _norm_words(s: str) -> List[str]:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s or "").lower())).strip().split()


# ── column grounding on one table ────────────────────────────────────────────────────
_UPDATED = re.compile(r"\b(updat\w*|modif\w*|chang\w*|edit\w*)\b")
_CREATED = re.compile(r"\b(creat\w*|added|registered|put up|listed|joined|signed up)\b")
_PAID = re.compile(r"\b(paid|payment date|settled)\b")
_MONEY = re.compile(r"\b(price\w*|cost\w*|expensive|cheap\w*|amount\w*|value|fee\w*|paid|rent|money|spend\w*)\b")
_SIZE = re.compile(r"\b(small\w*|larg\w*|big\w*|biggest|high\w*|low\w*|most|least)\b")
_NAMEW = re.compile(r"\b(name|names|alphabetical\w*|title|label)\b")


def _date_col(vocab, sm, table, concept: Optional[str]) -> Optional[str]:
    cols = _cols(sm, table)
    temporal = [c for c, m in cols.items() if _st(m) == "TEMPORAL"]
    card = vocab.cards.get(table) or {}
    c = (concept or "").lower()
    if c in ("", "@date", "date", "time", "when"):
        return card.get("business_date_column") or next((x for x in ("created_at",) if x in temporal), None) \
            or (temporal[0] if temporal else None)
    if _UPDATED.search(c):
        return next((x for x in temporal if re.match(r"(updated|modified|last_modified)", x)), None)
    if _CREATED.search(c):
        return next((x for x in temporal if x.startswith("created")), None)
    if _PAID.search(c):
        hit = next((x for x in temporal if "pay" in x or "paid" in x), None)
        if hit:
            return hit
    ct = set(_content(c)) - {"date", "time", "at", "on"}
    if ct:
        hit = [x for x in temporal if ct & _col_toks(x)]
        if len(hit) == 1:
            return hit[0]
    if re.search(r"\b(date|dated|time|recent\w*|latest|oldest|newest|earliest|when)\b", c):
        # "payment date" of a ledger entry with no payment_date column is the date the
        # entry itself happened: the card's business date
        return card.get("business_date_column") or (temporal[0] if temporal else None)
    return None


def _measure_col(vocab, sm, table, concept: Optional[str], *, money_hint: bool = False
                 ) -> Tuple[Optional[str], List[str]]:
    """(column, candidates). Measure glossary phrases first, then name tokens, then the
    card's key measures; type-checked (money words → MONETARY)."""
    cols = _cols(sm, table)
    card = vocab.cards.get(table) or {}
    c = (concept or "").lower().strip()
    meas = [x for x, m in cols.items() if _st(m) in ("METRIC", "MONETARY")]
    if c in ("", "@measure"):
        km = [m for m in (card.get("key_measures") or []) if m in cols]
        return (km[0] if km else (meas[0] if len(meas) == 1 else None)), km or meas
    if re.search(r"\b(date|dated|time|day|month|year|when|at|on)\b", c) and not _MONEY.search(c):
        return None, []                      # a date phrase is never an amount
    cn = norm_phrase(c)
    exact_col = c.replace(" ", "_")
    if exact_col in meas:
        return exact_col, [exact_col]          # "rating" → rating, not max_rating
    gl = [(k.split(".", 1)[1], v) for k, v in vocab.measure_glossary.items() if k.rsplit(".", 1)[0] == table]
    exact = [col for col, ent in gl if cn in {norm_phrase(p) for p in ent.get("phrases") or []}]
    if len(exact) == 1:
        return exact[0], exact
    ct = set(_content(c))
    tok = [x for x in meas if ct and ct <= _col_toks(x)] or [x for x in meas if ct & _col_toks(x)]
    if len(tok) == 1:
        return tok[0], tok
    if exact:
        return None, exact
    if (money_hint or _MONEY.search(c) or _SIZE.search(c)):
        km = [m for m in (card.get("key_measures") or []) if m in cols]
        money = [m for m in km if _st(cols[m]) == "MONETARY"] or \
            [x for x in meas if _st(cols[x]) == "MONETARY"]
        if money:
            return money[0], money
    return (tok[0] if len(tok) == 1 else None), tok


def _plain_col(vocab, sm, table, concept: str, kinds=None) -> Tuple[Optional[str], List[str]]:
    cols = _cols(sm, table)
    c = (concept or "").lower().strip()
    if not c:
        return None, []
    cn = c.replace(" ", "_")
    if cn in cols and (not kinds or _st(cols[cn]) in kinds):
        return cn, [cn]
    ct = set(_content(c))
    if not ct:
        return None, []
    ok = [x for x in cols if (not kinds or _st(cols[x]) in kinds)]
    full = [x for x in ok if ct <= _col_toks(x)]
    if len(full) == 1:
        return full[0], full
    part = [x for x in ok if ct & _col_toks(x)]
    # FK columns name their entity: "currency" → currency_id
    if len(part) > 1:
        idc = [x for x in part if x.endswith("_id") and (_col_toks(x) - {"id"}) <= ct]
        if len(idc) == 1:
            return idc[0], idc
    if len(part) == 1:
        return part[0], part
    # aliases/business_role text
    al = [x for x in ok if ct <= set(_toks(" ".join(map(str, (cols[x].get("aliases") or []))) + " " +
                                             str(cols[x].get("business_role") or "")))]
    if len(al) == 1:
        return al[0], al
    return None, full or part


def _name_col(vocab, sm, table) -> Optional[str]:
    card = vocab.cards.get(table) or {}
    d = card.get("display_column")
    cols = _cols(sm, table)
    if d and d in cols:
        return d
    for pref in ("name", "title", "label", "project_name", "full_name", "first_name"):
        if pref in cols:
            return pref
    names = [c for c in cols if "name" in c.split("_")]
    return names[0] if names else None


def _parents(vocab, table) -> List[str]:
    return list((vocab.cards.get(table) or {}).get("parent_entities") or [])


def _fk_to(sm, graph_edges, table, parent) -> Optional[str]:
    for e in graph_edges:
        if e.get("source_table") == _bare(table) and e.get("target_table") == _bare(parent) \
                and e.get("target_column") == "id" and not e.get("polymorphic") \
                and e.get("cardinality") in ("N:1", "1:1"):
            return e.get("source_column")
    # naming convention fallback: <parent entity>_id on the child
    cols = _cols(sm, table)
    ptail = _bare(parent).split("_")[-1]
    for c in cols:
        if c == f"{ptail}_id":
            return c
    return None


def _edges(vocab, table) -> List[dict]:
    sid = vocab.source_of.get(table)
    try:
        from ingestion.vocabulary import _graph
        from veda_core.context import try_current
        ctx = try_current()
        g = _graph(sid, ctx.tenant if ctx else "default")
        return list(g.get("edges") or [])
    except Exception:
        return []


# ── value domains at query time ──────────────────────────────────────────────────────
def column_domain(vocab, sm, table, column, *, live: bool = True) -> List[str]:
    vals = list((vocab.value_glossary.get(f"{table}.{column}") or {}).keys())
    meta = _cols(sm, table).get(column) or {}
    for v in meta.get("sample_values") or []:
        if str(v) not in vals:
            vals.append(str(v))
    if vals or not live:
        return vals
    try:
        from veda.understanding.frame_probes import distinct_values
        return distinct_values(_bare(table), column, limit=41) or []
    except Exception:
        return []


def ground_value(vocab, sm, table, column, value, question: str) -> Tuple[Optional[Any], Optional[str], str]:
    """(stored value | None, note, method)."""
    if value is None:
        return None, None, "none"
    sval = str(value).strip()
    dom = column_domain(vocab, sm, table, column)
    low = {d.lower(): d for d in dom}
    if sval.lower() in low:
        return low[sval.lower()], None, "domain"
    gl = vocab.value_glossary.get(f"{table}.{column}") or {}
    sn = norm_phrase(sval)
    for v, phrases in gl.items():
        if sn and sn in {norm_phrase(p) for p in phrases or []}:
            return v, f"'{sval}' read as {column.replace('_', ' ')} {v}", "glossary"
    # the sampled value store (value arbiter's lookup), anchor-scoped
    try:
        from query.resolution import typed_value_lookup
        for (t, c, _st2, raw) in typed_value_lookup()(sval.lower()) or []:
            if t == _bare(table) and c == column:
                return raw, None, "sampled"
    except Exception:
        pass
    return None, None, "unmapped"


# ── the entity decision ──────────────────────────────────────────────────────────────
def _slot_concepts(fr: Frame) -> List[Tuple[str, str]]:
    out = []
    if fr.measure:
        out.append(("measure", fr.measure))
    for f in fr.filters:
        out.append(("filter", f.concept))
    for g in fr.group_by:
        out.append(("group", g))
    if fr.order:
        out.append(("order", fr.order.concept))
    return out


def _grounds_on(vocab, sm, table, kind, concept, fr: Frame) -> bool:
    c = (concept or "").lower()
    if kind == "order":
        if c.startswith("@") or _date_col(vocab, sm, table, c):
            return c != "@measure" or bool(_measure_col(vocab, sm, table, c)[0])
        if _measure_col(vocab, sm, table, c)[0]:
            return True
        if _NAMEW.search(c):
            return True       # a name column on it or on a parent (joined)
        return bool(_plain_col(vocab, sm, table, c)[0])
    if kind == "measure":
        return bool(_measure_col(vocab, sm, table, c)[0])
    if kind == "group":
        return bool(_plain_col(vocab, sm, table, c)[0]) or _NAMEW.search(c) is not None
    if kind == "filter":
        f = next((x for x in fr.filters if x.concept == concept), None)
        if f is not None and f.__dict__.get("_table") == table:
            return True
        if f is not None and f.op in (">", ">=", "<", "<=", "between"):
            return bool(_measure_col(vocab, sm, table, c)[0]) or bool(_date_col(vocab, sm, table, c))
        return bool(_plain_col(vocab, sm, table, c)[0])
    return False


def _coverage(vocab, sm, table, fr: Frame) -> Tuple[int, int]:
    sl = _slot_concepts(fr)
    return sum(1 for k, c in sl if _grounds_on(vocab, sm, table, k, c, fr)), len(sl)


def _linked(vocab, a, b) -> bool:
    return b in _parents(vocab, a) or a in _parents(vocab, b)


def choose_entity(fr: Frame, vocab: ScopeVocab, sm, question: str,
                  router_primary: Optional[str] = None, ev: Optional[Dict[str, Any]] = None
                  ) -> Tuple[Optional[str], str, Optional[FrameClarify]]:
    ev = ev if ev is not None else {}
    hits = name_hits(vocab, question)
    # a name used INSIDE a slot's concept ("ordered by PROPERTY name", "grouped by
    # property type") refers to an attribute path, not the entity whose rows are listed
    slot_texts = [" " + norm_phrase(c) + " " for _k, c in _slot_concepts(fr) if c and not str(c).startswith("@")]
    qtoks = norm_phrase(question).split()

    def _attr(h):
        # the user's own words continue the name into an attribute the frame uses as a slot
        # ("ordered by PROPERTY NAME"), not "payments made" next to a slot 'payment date'
        nxt = qtoks[h["end"] + 1] if h["end"] + 1 < len(qtoks) else None
        return bool(nxt) and any(f" {h['phrase']} {nxt} " in st for st in slot_texts)
    attr_hits = [h for h in hits if _attr(h)]
    if attr_hits:
        ev["attribute_hits"] = [(h["table"], h["phrase"]) for h in attr_hits]
    hits = [h for h in hits if h not in attr_hits]
    named = [h["table"] for h in hits if not h.get("modifier")] or [h["table"] for h in hits]
    ev["name_hits"] = [(h["table"], h["phrase"], h["gap"]) for h in hits]
    model_pick = fr.provenance.get("entity_table") if isinstance(fr.provenance.get("entity_table"), str) else None
    ev["model_pick"] = model_pick
    # tables a producer tied a filter to (value glossary): evidence of the answering entity
    filter_tables = [f.__dict__.get("_table") for f in fr.filters if f.__dict__.get("_table")]
    cands = list(dict.fromkeys(named + ([model_pick] if model_pick else []) + filter_tables))
    cov = {t: _coverage(vocab, sm, t, fr) for t in cands}
    ev["coverage"] = {t: f"{a}/{b}" for t, (a, b) in cov.items()}
    full = [t for t in cands if cov[t][0] == cov[t][1]]

    def _pick(tables):
        # secondaries named by the model are the OTHER tables; a named table the model
        # picked is the entity; else the most specific name hit
        if model_pick in tables:
            return model_pick
        for h in hits:
            if h["table"] in tables:
                return h["table"]
        return tables[0]

    # cross-source: the SAME phrase names tables in ≥ 2 sources with no link between them
    if len(hits) >= 2:
        top = hits[0]
        same = [h for h in hits if h["phrase"] == top["phrase"]]
        srcs = {vocab.source_of.get(h["table"]) for h in same}
        if len(srcs) >= 2 and not any(_linked(vocab, a["table"], b["table"]) for a in same for b in same if a is not b):
            opts = [f"{h['table']} (source {vocab.source_of.get(h['table'])})" for h in same]
            return None, NAME, FrameClarify(
                "entity", f"'{top['phrase']}' exists in more than one data source ({', '.join(opts)}). "
                          f"Which one do you mean?", [h["table"] for h in same], ev)

    if named:
        named_full = [t for t in named if t in full]
        if named_full:
            if len(named_full) == 1:
                return named_full[0], NAME, None
            # several named tables can host the frame: model pick, then specificity; two
            # UNLINKED named entities with no preference → clarify
            p = _pick(named_full)
            if model_pick in named_full or all(_linked(vocab, p, o) for o in named_full if o != p):
                return p, NAME, None
            # sibling tiebreak: the card whose own description / "not to be confused"
            # phrase shares the most words with the question (a clear margin, else clarify)
            qw = set(_content(question))

            def _dscore(t):
                c = vocab.cards.get(t) or {}
                txt = " ".join([str(c.get("one_row_is") or "")] +
                               [str(ph).split(";")[0] for ph in (c.get("distinguishes_from") or {}).values()])
                return len(qw & set(_content(txt)))
            ranked = sorted(named_full, key=lambda t: -_dscore(t))
            if len(ranked) >= 2 and _dscore(ranked[0]) >= _dscore(ranked[1]) + 1:
                ev["sibling_tiebreak"] = {t: _dscore(t) for t in ranked}
                return ranked[0], NAME, None
            return None, NAME, FrameClarify(
                "entity", "This question names more than one kind of record: " +
                ", ".join(f"{vocab.cards[t]['plural']}" for t in named_full) + ". Which one should the answer list?",
                named_full, ev)
        # a named entity cannot host the slots — a LINKED table that can (and that the
        # question/extractor also points at) takes it
        linked_full = [t for t in full if any(_linked(vocab, t, n) for n in named)]
        if linked_full:
            return _pick(linked_full), COVERAGE, None
        # nothing hosts every slot: stay on the most specific name; the unhosted slot will
        # clarify on its own (with that entity's real columns)
        return _pick(named), NAME, None
    if model_pick:
        return model_pick, MODEL, None
    ex = nearest_examples(vocab, question, 1)
    if ex and float(ex[0].get("_sim") or 0) >= 0.80 and ex[0].get("table") in vocab.cards:
        return ex[0]["table"], SYNTHETIC, None
    # the router's primary may be handed in lazily (frame_path._RouterPrimary): asked for
    # only here, when nothing in the question or the frame named the entity
    if callable(router_primary):
        router_primary = router_primary()
    if router_primary and router_primary in vocab.cards:
        return router_primary, RETRIEVAL, None
    return None, RETRIEVAL, None


_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$")


def _as_date(v) -> Optional[str]:
    """An ISO date/datetime string, or None. The temporal parser (a producer) supplies
    exact ISO windows; the SLM's free text ('recent', 'last quarter') is not a date."""
    if v is None:
        return None
    s = str(v).strip()
    if _ISO.match(s):
        return s
    if re.fullmatch(r"\d{4}", s):
        return f"{s}-01-01"
    return None


_GROUP_W = re.compile(r"\b(per|each|every|group|grouped|grouping|breakdown|break down|broken down|distribution|split|by each|for each|categori[sz]e)\b")
_COUNT_W = re.compile(r"\b(how many|number of|count|counts|how much|total number|tally)\b")
_MINMAX_W = re.compile(r"\b(minimum|maximum|max|min|lowest value|highest value|smallest value|largest value)\b")
# a superlative inside a breakdown ("highest monthly fee PER CATEGORY") is the per-group
# extreme — a row ranking cannot be expressed per group, so it licenses MIN / MAX there
_EXTREME_W = re.compile(r"\b(highest|lowest|largest|smallest|biggest|greatest|most expensive|cheapest|priciest|top|bottom)\b")
_SUMAVG_W = {"sum": re.compile(r"\b(total|sum|overall|combined|altogether|how much)\b"),
             "avg": re.compile(r"\b(average|avg|mean|typical)\b")}


_MEASURE_PHRASES_CACHE: Dict[str, List[str]] = {}


def license_aggregation(fr: Frame, question: str) -> None:
    """An aggregation must be LICENSED by the question's wording. 'the cheapest listings'
    / 'the smallest payments' are rankings of rows, not MIN(); 'a breakdown of entries' is
    not COUNT(*). An unlicensed min/max becomes an ORDER on the measure; an unlicensed
    count/sum/avg becomes a plain list. Recorded in provenance."""
    ql = question.lower()
    # words that are part of a MEASURE's name ("tax total", "max rating", "min budget") do
    # not license an aggregation — strip every measure phrase / column name first
    # …but only where the frame USES that phrase as a column (a filter or the sort key), so
    # "the total amount of all quotes" still licenses SUM
    used = [f.concept.lower() for f in fr.filters if f.op in (">", ">=", "<", "<=", "between")]
    if fr.order:
        used.append(fr.order.concept.lower())
    for phr in sorted(set(_MEASURE_PHRASES_CACHE.get("all", [])) | set(used), key=len, reverse=True):
        if " " in phr and phr in ql and any(phr in u or u in phr for u in used):
            ql = ql.replace(phr, " ")
    if fr.group_by and not _GROUP_W.search(ql):
        # "which properties have the most expensive records" / "an alphabetical list" name
        # no breakdown — a GROUP BY the wording never asked for changes the answer's grain
        fr.provenance["group_by_unlicensed"] = list(fr.group_by)
        fr.group_by = []
        if fr.aggregation in ("count", "count_distinct", "sum", "avg") and not _COUNT_W.search(ql) \
                and not any(r.search(ql) for r in _SUMAVG_W.values()):
            fr.aggregation = "none"
    agg = fr.aggregation
    if agg in ("min", "max") and not _MINMAX_W.search(ql) \
            and not (fr.group_by and _EXTREME_W.search(ql)):
        if fr.order is None:
            fr.order = FrameOrder(concept=fr.measure or "@measure", dir="asc" if agg == "min" else "desc")
        fr.provenance["aggregation_unlicensed"] = agg
        fr.aggregation = "none"
    elif agg in ("count", "count_distinct") and not _COUNT_W.search(ql) and not fr.group_by:
        fr.provenance["aggregation_unlicensed"] = agg
        fr.aggregation = "none"
    elif agg in _SUMAVG_W and not _SUMAVG_W[agg].search(ql) and not fr.group_by:
        fr.provenance["aggregation_unlicensed"] = agg
        fr.aggregation = "none"


# ── ground ───────────────────────────────────────────────────────────────────────────
def ground_frame(fr: Frame, vocab: ScopeVocab, sm, question: str,
                 router_primary: Optional[str] = None):
    """Frame → GroundedFrame | FrameClarify | None (None = no entity at all → degrade)."""
    ev: Dict[str, Any] = {}
    _MEASURE_PHRASES_CACHE["all"] = sorted({p.lower() for ent in vocab.measure_glossary.values()
                                            for p in (ent.get("phrases") or [])} |
                                           {k.split(".", 1)[1].replace("_", " ") for k in vocab.measure_glossary})
    license_aggregation(fr, question)
    anchor, method, clar = choose_entity(fr, vocab, sm, question, router_primary, ev)
    if clar is not None:
        return clar
    if not anchor:
        return None
    anchor = _shadow("entity", fr.entity or "", list(vocab.cards), anchor, {"question": question}, ev) or anchor
    # "a list of our properties and the payments associated with them": a LIST whose
    # question also names a CHILD of the chosen entity is a list of the child's rows (the
    # finer grain), with the parent joined for display.
    reanchored_parent = None
    if fr.aggregation == "none" and not fr.group_by:
        # only a STRONG child mention counts: its business name / plural, or a multi-word
        # alias — a one-word alias ('sale' → sale transactions) is too weak to move the grain
        _kinds = {h["table"]: h for h in name_hits(vocab, question)}
        named_children = [h[0] for h in ev.get("name_hits") or []
                          if h[0] != anchor and anchor in _parents(vocab, h[0])
                          and _kinds.get(h[0], {}).get("strong")]
        if named_children:
            reanchored_parent, anchor = anchor, named_children[0]
            ev["reanchored"] = {"from": reanchored_parent, "to": anchor}
            method = NAME
    card = vocab.cards.get(anchor) or {}
    cols = _cols(sm, anchor)
    if not cols:
        return None
    gf = GroundedFrame(frame=fr, anchor=anchor, anchor_method=method,
                       source_id=vocab.source_of.get(anchor), evidence=ev,
                       confidence=float(fr.confidence or 0.0))
    edges = _edges(vocab, anchor)
    unresolved: List[FrameClarify] = []

    def _join_parent(parent) -> Optional[GJoin]:
        fk = _fk_to(sm, edges, anchor, parent)
        if not fk:
            return None
        for j in gf.joins:
            if j.table == parent:
                return j
        j = GJoin(table=parent, fk_column=fk)
        gf.joins.append(j)
        return j

    def _named_parent(concept) -> Optional[str]:
        """a parent entity the concept names ('property name' → assets_asset)."""
        ct = " " + norm_phrase(concept) + " "
        for p in _parents(vocab, anchor):
            pc = vocab.cards.get(p) or {}
            for ph in [pc.get("business_name"), pc.get("plural"), *(pc.get("aliases") or [])]:
                if ph and f" {norm_phrase(ph)} " in ct:
                    return p
        return None

    if reanchored_parent:
        _join_parent(reanchored_parent)
    # ── aggregation + measure ──
    agg = fr.aggregation
    if agg in ("sum", "avg", "min", "max"):
        col, cands = _measure_col(vocab, sm, anchor, fr.measure, money_hint=bool(_MONEY.search(question.lower())))
        col = _shadow("measure", fr.measure or "", cands, col, {"question": question}, ev)
        if not col:
            unresolved.append(FrameClarify(
                "measure", (f"Which amount should I {agg} for {card.get('plural') or anchor}? "
                            + (f"Options: {', '.join(c.replace('_', ' ') for c in cands)}." if cands else
                               "I couldn't find a numeric column for that.")),
                cands))
        else:
            gf.measure = (agg, col)
    elif agg in ("count", "count_distinct"):
        dcol = None
        if agg == "count_distinct" or fr.distinct:
            dcol, _ = _plain_col(vocab, sm, anchor, fr.measure or "")
        if dcol is None:
            # "how many amenity CATEGORIES are there": the counted noun, minus the entity's own
            # name, is a column of the entity → COUNT(DISTINCT col), never COUNT(*)
            m = re.search(r"\b(?:how many|number of|count of|count the)\s+(?:distinct\s+|different\s+|unique\s+)?([a-z ]+?)\s+(?:are|is|do|does|exist|there|in|of|have|we)\b",
                          question.lower() + " ")
            if m:
                ent_words = set()
                for ph in [card.get("business_name"), card.get("plural"), *(card.get("aliases") or [])]:
                    ent_words |= set(norm_phrase(ph or "").split())
                rest = " ".join(w for w in norm_phrase(m.group(1)).split() if w not in ent_words)
                if rest:
                    c2, _ = _plain_col(vocab, sm, anchor, rest, kinds={"CATEGORY", "FLAG"})
                    dcol = c2
                    if dcol:
                        ev["counted_noun"] = {"phrase": m.group(1), "column": dcol}
        gf.measure = ("count", dcol)
        gf.distinct = bool(dcol)

    # ── filters ──
    for f in fr.filters:
        c = f.concept or ""
        prod_table = f.__dict__.get("_table")
        op = f.op
        if op in (">", ">=", "<", "<=", "between"):
            # numeric (or a date range)
            col, cands = _measure_col(vocab, sm, anchor, c, money_hint=bool(_MONEY.search(c)))
            dcol = None if col else _date_col(vocab, sm, anchor, c) if re.search(r"date|time|when|day|month|year", c) else None
            if col:
                val = f.value
                try:
                    if op == "between":
                        vv = list(val) if isinstance(val, (list, tuple)) else [val]
                        lo, hi = float(vv[0]), float(vv[-1])
                        val = [min(lo, hi), max(lo, hi)]
                    else:
                        val = float(val)
                except (TypeError, ValueError, IndexError):
                    unresolved.append(FrameClarify("filter", f"I couldn't read '{f.value}' as a number for {c}.", []))
                    continue
                gf.filters.append(GFilter(anchor, col, op, val, c, "numeric", stype=_st(cols.get(col))))
            elif dcol:
                vals = f.value if isinstance(f.value, (list, tuple)) else [f.value]
                if all(_as_date(v) for v in vals):
                    val = [_as_date(v) for v in vals] if op == "between" else _as_date(vals[0])
                    gf.filters.append(GFilter(anchor, dcol, op, val, c, "temporal", stype="TEMPORAL"))
                else:
                    # 'recent' / 'lately' is not a period: read as newest-first, and say so
                    gf.notes.append(f"'{f.value}' has no fixed period — showing the newest first")
                    if fr.order is None:
                        fr.order = FrameOrder(concept=dcol, dir="desc")
            else:
                unresolved.append(FrameClarify(
                    "filter", (f"Which amount does '{c}' refer to for {card.get('plural') or anchor}? "
                               + (f"Options: {', '.join(x.replace('_', ' ') for x in cands)}." if cands else
                                  "None of its columns is numeric.")), cands))
            continue
        if op in ("is_null", "is_not_null"):
            _first = (re.findall(r"[a-z]+", c.lower()) or [""])[0]
            if not re.search(rf"\b(with|without|has|have|having|missing|lacking|no)\s+(?:an?\s+|any\s+|no\s+)?{re.escape(_first)}",
                             question.lower()):
                # "which properties HAVE the MOST EXPENSIVE records" is a superlative, not an
                # existence test — a null check needs existence wording right before its concept
                ev.setdefault("dropped", []).append(f"unlicensed null-check on '{c}'")
                continue
            if norm_phrase(c) in vocab.name_index():
                # "…have payment transactions" is EXISTENCE of related rows — not a null
                # check on some column that happens to share the word; not compiled here
                ev.setdefault("dropped", []).append(f"null-check on entity '{c}'")
                continue
            col, cands = _plain_col(vocab, sm, anchor, c)
            if col:
                gf.filters.append(GFilter(anchor, col, "IS NULL" if op == "is_null" else "IS NOT NULL",
                                          None, c, "null_check", stype=_st(cols.get(col))))
            else:
                unresolved.append(FrameClarify("filter", f"I couldn't find '{c}' on {card.get('plural') or anchor}.", cands))
            continue
        if f.__dict__.get("_inherited") and prod_table in (anchor, _bare(anchor)):
            # a DEPENDENT part's rows: the values are the parent part's own verified result
            # ("…and for those, …"), not words to ground against a domain
            icol = c.replace(" ", "_")
            vals = [v for v in (f.value if isinstance(f.value, list) else [f.value]) if v not in (None, "")]
            if icol in cols and vals:
                gf.filters.append(GFilter(anchor, icol, "IN", vals, c, "inherited",
                                          f"the {len(vals)} rows of the previous part",
                                          stype=_st(cols.get(icol))))
                continue
        # categorical (= / != / in)
        if f.value in (None, "", []) or (isinstance(f.value, list) and not any(v not in (None, "") for v in f.value)):
            # "per city" echoed as `city IN ()` — a filter with no value constrains nothing
            ev.setdefault("dropped", []).append(f"empty-valued filter on '{c}'")
            continue
        col = None
        if prod_table == anchor and f.__dict__.get("_producer") == "values":
            col = c.replace(" ", "_") if c.replace(" ", "_") in cols else None
        if col is None:
            col, cands = _plain_col(vocab, sm, anchor, c, kinds={"CATEGORY", "FLAG", "IDENTIFIER", "FREE_TEXT"})
        else:
            cands = [col]
        if col is None:
            # the value may identify the column: which of the anchor's category columns has it?
            vals = f.value if isinstance(f.value, list) else [f.value]
            owners = []
            for cc, m in cols.items():
                if _st(m) in ("CATEGORY", "FLAG"):
                    if all(ground_value(vocab, sm, anchor, cc, v, question)[0] is not None for v in vals if v is not None):
                        owners.append(cc)
            if len(owners) == 1:
                col = owners[0]
        if col is None and prod_table and prod_table != anchor and not _linked(vocab, anchor, prod_table):
            # a glossary phrase of an UNRELATED table ('linked accounts' inside the entity's own
            # name "razorpay linked accounts") qualifies nothing here — ignore it
            ev.setdefault("dropped", []).append(f"glossary phrase of unrelated {prod_table}: '{f.__dict__.get('_phrase')}'")
            continue
        if col is None and f.__dict__.get("_glossary_value") is None and prod_table and prod_table != anchor:
            # a glossary filter for a LINKED table other than the anchor: not this entity's qualifier
            unresolved.append(FrameClarify(
                "filter", f"'{f.__dict__.get('_phrase') or f.value}' describes "
                          f"{(vocab.cards.get(prod_table) or {}).get('plural') or prod_table}, not "
                          f"{card.get('plural') or anchor}.", [prod_table]))
            continue
        if col is None:
            life = card.get("lifecycle_column")
            dom = column_domain(vocab, sm, anchor, life) if life else []
            unresolved.append(FrameClarify(
                "filter", (f"I couldn't match '{c}' to a field of {card.get('plural') or anchor}."
                           + (f" Its {life.replace('_', ' ')} values are {', '.join(dom[:12])}." if dom else "")),
                cands))
            continue
        vals = f.value if isinstance(f.value, list) else [f.value]
        # the user may NAME the value right next to the column word — "have an ACTIVE
        # status", "status is ACTIVE". That word is the value asked for, whatever the
        # extractor normalised it to; it must ground on its own, or this is a clarify.
        _cw = re.escape(col.replace("_", " "))
        _m = (re.search(rf"\b(?:an?|with|has|have|having)\s+([a-z]+)\s+{_cw}\b", question.lower())
              or re.search(rf"\b{_cw}\s+(?:is|of|=|equal to)\s+([a-z]+)\b", question.lower()))
        if _m and _m.group(1) not in ("the", "a", "an", "any", "some", "current", "their", "its", "no"):
            _said = _m.group(1)
            if ground_value(vocab, sm, anchor, col, _said, question)[0] is None:
                vals = [_said]
                ev["explicit_value"] = _said
        grounded_vals, notes = [], []
        bad = None
        for v in vals:
            gv, note, how = ground_value(vocab, sm, anchor, col, v, question)
            if gv is None and f.__dict__.get("_glossary_value") is not None:
                # the extractor's wording didn't ground but the glossary matched a phrase in
                # the question for this column — unless the user NAMED this value next to
                # the column word ("an ACTIVE status"): then their word must ground itself
                sv = str(v or "").lower().strip()
                named_explicitly = bool(sv) and re.search(
                    rf"\b{re.escape(sv)}\s+{re.escape(col.replace('_', ' '))}|\b{re.escape(col.replace('_', ' '))}\s+(?:is\s+|of\s+|=\s*)?{re.escape(sv)}\b",
                    question.lower())
                if not named_explicitly:
                    gv, how = f.__dict__["_glossary_value"], "glossary"
                    note = f"'{f.__dict__.get('_phrase')}' read as {col.replace('_', ' ')} {gv}"
            if gv is None:
                bad = v
                break
            grounded_vals.append(gv)
            if note:
                notes.append(note)
        if bad is not None:
            dom = column_domain(vocab, sm, anchor, col)
            unresolved.append(FrameClarify(
                "filter", (f"There is no {col.replace('_', ' ')} '{bad}' for {card.get('plural') or anchor}."
                           + (f" The {col.replace('_', ' ')} values are {', '.join(dom[:12])}." if dom else "")
                           + " Which one did you mean?"), dom[:12]))
            continue
        # a categorical filter must be LICENSED by the question: the stored value, a glossary
        # phrase for it, or the user's own word must be in the text (the SLM restating
        # 'properties for sale' as status = APPROVED is not the user asking for it)
        _qlow = " " + norm_phrase(question) + " "
        _gl = vocab.value_glossary.get(f"{anchor}.{col}") or {}

        def _licensed(gv, raw):
            words = [str(gv), str(raw or "")] + list(_gl.get(gv) or [])
            if any(w and f" {norm_phrase(w)} " in _qlow for w in words):
                return True
            if _st(cols.get(col)) == "FLAG" or str(gv).lower() in ("true", "false", "t", "f", "yes", "no"):
                # a yes/no column is named by its own word: "publicly visible" → is_public
                ctoks = [t for t in col.lower().split("_") if t not in ("is", "has", "can", "was", "flag") and len(t) > 2]
                return bool(ctoks) and all(re.search(rf"\b{re.escape(t)}", question.lower()) for t in ctoks)
            return False
        if f.__dict__.get("_producer") != "values" and not all(
                _licensed(gv, rv) for gv, rv in zip(grounded_vals, vals)):
            ev.setdefault("filters_unlicensed", []).append(f"{col} {op} {grounded_vals}")
            continue
        if f.__dict__.get("_phrase") and not notes:
            _ph = f.__dict__["_phrase"]
            if not any(f" {norm_phrase(str(g))} " in _qlow for g in grounded_vals):
                notes.append(f"'{_ph}' read as {col.replace('_', ' ')} {', '.join(map(str, grounded_vals))}")
        gop = "IN" if len(grounded_vals) > 1 or op == "in" else ("!=" if op == "!=" else "=")
        dom_low = {d.lower() for d in column_domain(vocab, sm, anchor, col, live=False)}
        if gop == "IN" and dom_low and {str(v).lower() for v in grounded_vals} >= dom_low:
            # "whether they were debited OR credited" names every value: no narrowing —
            # the column is shown, not filtered
            gf.notes.append(f"showing every {col.replace('_', ' ')} ({', '.join(map(str, grounded_vals))})")
            ev.setdefault("vacuous_filters", []).append(col)
            continue
        gf.filters.append(GFilter(anchor, col, gop, grounded_vals if gop == "IN" else grounded_vals[0],
                                  c, "glossary" if notes else "domain", "; ".join(notes) or None,
                                  stype=_st(cols.get(col))))
        gf.notes.extend(notes)

    # ── time window ──
    if fr.time and fr.time.window:
        w = fr.time.window
        fs, ts = _as_date(w.get("from")), _as_date(w.get("to"))
        if (w.get("from") and not fs) or (w.get("to") and not ts):
            gf.notes.append("the period named has no fixed dates — no date window applied")
            fr.time.window = None
        else:
            fr.time.window = {"from": fs, "to": ts}
    if fr.time and fr.time.window and (fr.time.window.get("from") or fr.time.window.get("to")):
        tcol = _date_col(vocab, sm, anchor, fr.time.concept or "@date")
        if not tcol:
            unresolved.append(FrameClarify("time", f"{card.get('plural') or anchor} have no date to apply that period to.", []))
        else:
            gf.time = {"table": anchor, "column": tcol, "start": fr.time.window.get("from"),
                       "end": fr.time.window.get("to")}

    # ── group by ──
    for g in fr.group_by:
        col, cands = _plain_col(vocab, sm, anchor, g, kinds={"CATEGORY", "FLAG", "IDENTIFIER", "FREE_TEXT", "TEMPORAL"})
        if col:
            gf.group_by.append((anchor, col))
            continue
        p = _named_parent(g)
        if p and _join_parent(p):
            nc = _name_col(vocab, sm, p)
            if nc:
                gf.group_by.append((p, nc))
                continue
        unresolved.append(FrameClarify(
            "group_by", f"I couldn't find '{g}' to group {card.get('plural') or anchor} by.", cands))
    if gf.group_by and gf.measure is None:
        gf.measure = ("count", None)

    # ── order ──
    o = fr.order
    if o is None and gf.group_by and re.search(r"\b(descending|ascending)\b", question.lower()):
        gt, gc = gf.group_by[0]
        gf.order = (gt, gc, "desc" if "descending" in question.lower() else "asc")
    def _ground_order(o):
            oc = (o.concept or "").lower()
            target = None
            if oc == "@id":
                target = (anchor, "id") if "id" in cols else None
            elif oc == "@measure":
                mc, _ = _measure_col(vocab, sm, anchor, "@measure")
                target = (anchor, mc) if mc else None
            if target is None and not oc.startswith("@") and _NAMEW.search(oc):
                p = _named_parent(oc)
                if p and _join_parent(p):
                    nc = _name_col(vocab, sm, p)
                    target = (p, nc) if nc else None
                elif not p:
                    _rp = (ev.get("reanchored") or {}).get("from")
                    if _rp and _join_parent(_rp) and _name_col(vocab, sm, _rp):
                        # "an alphabetical list of our PROPERTIES and their payments": the rows
                        # moved to payments, but the name being sorted is the property's
                        target = (_rp, _name_col(vocab, sm, _rp))
                    nc = None if target else _name_col(vocab, sm, anchor)
                    if nc:
                        target = (anchor, nc)
                    else:
                        # the anchor has no name column: a named/secondary parent's
                        for s in list(fr.secondaries) + [h[0] for h in ev.get("name_hits") or []]:
                            pt = next((t for t, c in vocab.cards.items() if norm_phrase(c.get("business_name") or "") == norm_phrase(s)), s)
                            if pt in _parents(vocab, anchor) and _join_parent(pt):
                                nc = _name_col(vocab, sm, pt)
                                if nc:
                                    target = (pt, nc)
                                    break
            if target is None:
                dc = _date_col(vocab, sm, anchor, oc) if (oc.startswith("@date") or re.search(
                    r"date|time|recent|latest|oldest|newest|earliest|updat|modif|creat|when|dated", oc)) else None
                if dc:
                    target = (anchor, dc)
            if target is None:
                mc, _ = _measure_col(vocab, sm, anchor, oc)
                if mc:
                    target = (anchor, mc)
            if target is None and gf.group_by:
                # "grouped by currency in descending order": the order concept is the group key
                for gt, gc in gf.group_by:
                    if set(_content(oc)) & _col_toks(gc):
                        target = (gt, gc)
            if target is None:
                pc, _ = _plain_col(vocab, sm, anchor, oc)
                if pc:
                    target = (anchor, pc)
            if target is not None and target[0] == anchor and re.match(r"(created|updated|modified|last_modified)(_|$)", target[1]):
                # an AUDIT timestamp answers "latest / oldest / most recently dated" only when
                # the question uses the verb that binds it ("updated", "added"); otherwise the
                # entity's own business date is what the user means
                verb = (_UPDATED if target[1].startswith(("updated", "modified", "last_modified")) else _CREATED)
                bd = (vocab.cards.get(anchor) or {}).get("business_date_column")
                if bd and bd != target[1] and bd in cols and not verb.search(question.lower()):
                    ev["audit_date_replaced"] = {"from": target[1], "to": bd}
                    target = (anchor, bd)
            return target

    if o is not None and not gf.order:
        # the SLM's order first; a producer's conflicting order is an ALTERNATIVE, taken
        # when the SLM's does not ground (conflicts are settled by grounding, not rank)
        alts = [o] + [FrameOrder(concept=a["concept"], dir=a["dir"]) for a in
                      ((fr.provenance.get("_alternatives") or {}).get("order") or [])
                      if isinstance(a, dict)]
        target, used = None, None
        for cand in alts:
            target = _ground_order(cand)
            if target is not None:
                used = cand
                break
        if target is None:
            unresolved.append(FrameClarify(
                "order", f"I couldn't tell what to sort {card.get('plural') or anchor} by ('{o.concept}').", []))
        else:
            gf.order = (target[0], target[1], used.dir)
            if used is not o:
                ev["order_from_alternative"] = used.concept
    gf.limit = fr.limit
    if fr.limit is not None and fr.provenance.get("limit") == "slm" and not re.search(
            rf"\b{fr.limit}\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|twenty|fifty|hundred)\b",
            question.lower()):
        # a row count the user never wrote (the SLM's '1000') is not the question's
        gf.limit = None
        ev["limit_unlicensed"] = fr.limit
    alt_lim = (fr.provenance.get("_alternatives") or {}).get("limit") or []
    if fr.limit is not None and alt_lim and not re.search(rf"\b{fr.limit}\b", question):
        # the SLM's row count is not a number the user wrote; the parser's is
        gf.limit = int(alt_lim[0])
    if fr.distinct and not gf.measure:
        gf.distinct = True

    if unresolved:
        first = unresolved[0]
        first.evidence = {**ev, "all": [u.message for u in unresolved], "anchor": anchor,
                          "anchor_method": method}
        return first

    # ── projection: what a reader needs to check the answer ──
    proj: List[Tuple[str, str]] = []

    def _add(t, c):
        if c and (t, c) not in proj:
            proj.append((t, c))
    if not gf.measure and not gf.group_by:
        _add(anchor, "id" if "id" in cols else None)
        _add(anchor, card.get("display_column") if card.get("display_column") in cols else None)
        _add(anchor, card.get("business_date_column") if card.get("business_date_column") in cols else None)
        _add(anchor, card.get("lifecycle_column") if card.get("lifecycle_column") in cols else None)
        for m in (card.get("key_measures") or [])[:2]:
            _add(anchor, m if m in cols else None)
        for f in gf.filters:
            _add(f.table, f.column)
        if gf.order:
            _add(gf.order[0], gf.order[1])
        if gf.time:
            _add(anchor, gf.time["column"])
        for j in gf.joins:
            _add(j.table, _name_col(vocab, sm, j.table))
    gf.projection = proj
    return gf
