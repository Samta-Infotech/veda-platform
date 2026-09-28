"""veda.understanding.compound — one message, N intents, N sources (front-door decomposition).

    message → extract_intents (one SLM call) → ground_intents (per frame, per source)
            → plan_compound (single path, or N parts) → [veda_hybrid runs each part on its
            own source] → compose_reply

Everything here is PURE: no SLM, no DB. The extractor lives in frame_extractor; execution
(narrowing the request scope, running the part's lane) lives in veda_hybrid.

Grounding a frame to a SOURCE (not yet to a table — the per-source frame path does that
once the part runs on its source):
  * a `rag` frame → the document source whose document card matches (named title, else the
    best section/topic overlap); its "filters" are the topic phrases;
  * a data frame → the source whose cards NAME its entity: exactly one source → that
    source; several unlinked sources → a clarify for THIS frame only; none → the frame
    degrades alone (the part runs through the existing chain);
  * a `dependent` frame → inherits its parent's grounded entity and source, and takes the
    parent result's top_values as candidate filter values (attached at run time).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from veda.understanding.frame import Frame, Intents
from veda.understanding.vocabulary import (ScopeVocab, norm_phrase, doc_cards_of,
                                           source_kinds_of)

GROUNDED, CLARIFY, DEGRADE = "grounded", "clarify", "degrade"


@dataclass
class IntentGrounding:
    index: int
    part: str
    kind: Optional[str] = None                 # sql | tabular | rag
    source_id: Optional[str] = None
    entity: Optional[str] = None               # scope card key, or a document title
    entity_name: Optional[str] = None          # business name (never an identifier)
    method: str = ""                           # NAME | MODEL | DOCUMENT | TOPIC | INHERITED
    outcome: str = GROUNDED                    # grounded | clarify | degrade
    message: Optional[str] = None              # the clarify text (business names only)
    candidates: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    depends_on: Optional[int] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


# ── document matching ────────────────────────────────────────────────────────────────
def doc_score(card: Dict[str, Any], text: str, frame_topics: List[str]) -> Tuple[int, List[str]]:
    """How strongly a message part points at a document: every topic phrase of the card
    the part contains (weighted by its length in words), plus the frame's own topic
    phrases that match one of the card's."""
    t = " " + norm_phrase(text) + " "
    hits, score = [], 0
    for ph in card.get("topics") or []:
        n = norm_phrase(ph)
        if n and f" {n} " in t:
            hits.append(ph)
            score += len(n.split())
    for ft in frame_topics or []:
        fn = norm_phrase(ft)
        if not fn:
            continue
        for ph in card.get("topics") or []:
            n = norm_phrase(ph)
            if n and (f" {fn} " in f" {n} " or f" {n} " in f" {fn} "):
                if ph not in hits:
                    hits.append(ph)
                score += 1
                break
    return score, hits


def best_document(vocab: ScopeVocab, text: str, frame_topics: List[str],
                  named: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], int, List[str]]:
    cards = doc_cards_of(vocab)
    if named:
        for c in cards:
            if norm_phrase(c.get("title") or "") == norm_phrase(named):
                s, h = doc_score(c, text, frame_topics)
                return c, max(s, 1), h
    best, bs, bh = None, 0, []
    for c in cards:
        s, h = doc_score(c, text, frame_topics)
        if s > bs:
            best, bs, bh = c, s, h
    return best, bs, bh


# ── data matching ────────────────────────────────────────────────────────────────────
def _tables_named_by(vocab: ScopeVocab, concept: Optional[str]) -> List[str]:
    """Every card whose business name / plural / alias IS this concept (any source)."""
    if not concept:
        return []
    key = norm_phrase(re.sub(r"\s*\([^)]*\)$", "", concept))
    return [t for t, _k in vocab.name_index().get(key, [])]


def _linked(vocab: ScopeVocab, a: str, b: str) -> bool:
    pa = (vocab.cards.get(a) or {}).get("parent_entities") or []
    pb = (vocab.cards.get(b) or {}).get("parent_entities") or []
    return b in pa or a in pb


def _card_words(vocab: ScopeVocab, t: str) -> set:
    """Everything a card says it carries, as normalised words: its measures, dimensions,
    lifecycle / display / date columns, measure-glossary phrases, value-glossary columns
    and values, and its own table name."""
    c = vocab.cards.get(t) or {}
    bits: List[str] = []
    for k in ("key_measures", "key_dimensions"):
        bits += [str(x) for x in (c.get(k) or [])]
    for k in ("lifecycle_column", "display_column", "business_date_column"):
        if c.get(k):
            bits.append(str(c[k]))
    for key, ent in vocab.measure_glossary.items():
        if key.rsplit(".", 1)[0] == t:
            bits.append(key.rsplit(".", 1)[1])
            bits += [str(p) for p in (ent or {}).get("phrases") or []]
    for key, m in vocab.value_glossary.items():
        if key.rsplit(".", 1)[0] == t:
            bits.append(key.rsplit(".", 1)[1])
            bits += [str(v) for v in (m or {})]
    bits.append(str(c.get("_bare_table") or t))
    words = set()
    for b in bits:
        words |= set(norm_phrase(b.replace("_", " ")).split())
    return {w for w in words if len(w) > 2}


def _coverage_score(vocab: ScopeVocab, t: str, fr: Frame, part: str) -> int:
    """How many of the part's slot concepts / content words this card carries (its own
    name words excluded — both candidates share those)."""
    c = vocab.cards.get(t) or {}
    own = set()
    for ph in [c.get("business_name"), c.get("plural"), *(c.get("aliases") or [])]:
        own |= set(norm_phrase(ph or "").split())
    concepts = [fr.measure or ""] + [f.concept for f in fr.filters] + list(fr.group_by) + \
        ([fr.order.concept] if fr.order else []) + [part]
    words = set()
    for x in concepts:
        words |= set(norm_phrase(str(x or "")).split())
    words = {w for w in words if len(w) > 2} - own
    return len(words & _card_words(vocab, t))


def _src_name(sid: Optional[str], source_names: Dict[str, str]) -> str:
    return source_names.get(str(sid)) or f"data source {sid}"


def ground_one(fr: Frame, i: int, vocab: ScopeVocab, *, prior: Optional[IntentGrounding] = None,
               source_names: Optional[Dict[str, str]] = None) -> IntentGrounding:
    from veda.understanding.frame_grounding import name_hits
    source_names = source_names or {}
    kinds = source_kinds_of(vocab)
    part = fr.part or ""
    g = IntentGrounding(index=i, part=part, topics=list(fr.topics), depends_on=fr.depends_on)

    # ── dependent: inherit the parent's grounding ──
    if fr.depends_on is not None and prior is not None:
        g.kind, g.source_id, g.entity = prior.kind, prior.source_id, prior.entity
        g.entity_name, g.method = prior.entity_name, "INHERITED"
        g.outcome = prior.outcome if prior.outcome != CLARIFY else DEGRADE
        g.evidence["parent"] = prior.index
        return g

    doc_sources = [s for s, k in kinds.items() if k == "rag"]
    dcard, dscore, dhits = best_document(vocab, part, fr.topics,
                                         named=fr.provenance.get("document"))
    g.evidence["doc"] = {"title": (dcard or {}).get("title"), "score": dscore, "hits": dhits[:6]}

    hits = name_hits(vocab, part) if part else []
    strong = [h for h in hits if h.get("strong") and not h.get("modifier")]
    g.evidence["name_hits"] = [(h["table"], h["phrase"]) for h in hits]
    model_tables = _tables_named_by(vocab, fr.entity)
    mt = fr.provenance.get("entity_table")
    if isinstance(mt, str) and mt in vocab.cards and mt not in model_tables:
        model_tables.insert(0, mt)
    g.evidence["model_tables"] = model_tables

    def _as_rag(method):
        g.kind = "rag"
        g.method = method
        if dcard is not None:
            g.source_id = str(dcard.get("source_id"))
            g.entity = dcard.get("title")
            g.entity_name = dcard.get("title")
            if not g.topics:
                g.topics = [h for h in dhits[:3]]
        elif len(doc_sources) == 1:
            g.source_id = doc_sources[0]
        else:
            g.outcome = DEGRADE
            g.evidence["reason"] = "no_document_source"
        return g

    # ── a rag frame ──
    wants_rag = fr.kind == "rag" or bool(fr.provenance.get("document"))
    if wants_rag and doc_sources:
        # the model calling a DATA question a document one: the part names a record kind
        # outright and touches no document section
        if strong and dscore == 0:
            g.evidence["rag_overridden"] = [h["phrase"] for h in strong]
        else:
            return _as_rag("DOCUMENT" if fr.provenance.get("document") else "TOPIC")

    # ── a data frame: which source's cards NAME its entity ──
    named_tables: List[str] = []
    phrase = None
    if hits:
        ordered = strong or hits
        # the model's entity, when the part names it, wins; else the most specific name
        top = next((h for h in ordered if h["table"] in model_tables), ordered[0])
        phrase = top["phrase"]
        named_tables = [t for t, _k in vocab.name_index().get(phrase, [])] or [top["table"]]
        if model_tables and top["table"] in model_tables:
            named_tables = [top["table"]] + [t for t in named_tables if t != top["table"]
                                             and vocab.source_of.get(t) != vocab.source_of.get(top["table"])]
        g.method = "NAME"
    elif model_tables:
        # the part names no card; the extractor's pick stands only while the part does
        # not point at a document instead
        if dscore >= 2 and doc_sources:
            return _as_rag("TOPIC")
        named_tables, phrase = model_tables, fr.entity
        g.method = "MODEL"
    else:
        if dscore >= 1 and doc_sources:
            return _as_rag("TOPIC")
        g.outcome = DEGRADE
        g.evidence["reason"] = "no_entity"
        return g

    by_src: Dict[str, List[str]] = {}
    for t in named_tables:
        by_src.setdefault(str(vocab.source_of.get(t)), []).append(t)
    if len(by_src) > 1:
        firsts = [ts[0] for ts in by_src.values()]
        # COVERAGE: the same name in two sources, but only one of them carries what the
        # part asks about ("amenities" + "monthly fee" → the catalog that has a fee)
        cov = {t: _coverage_score(vocab, t, fr, part) for t in firsts}
        g.evidence["coverage"] = cov
        ranked = sorted(firsts, key=lambda t: -cov[t])
        if cov[ranked[0]] >= cov[ranked[1]] + 1:
            by_src = {str(vocab.source_of.get(ranked[0])): [ranked[0]]}
            g.method = "NAME+COVERAGE"
        elif not any(_linked(vocab, a, b) for a in firsts for b in firsts if a != b):
            g.outcome = CLARIFY
            g.candidates = firsts
            opts = [f"{(vocab.cards.get(t) or {}).get('plural') or phrase} in "
                    f"{_src_name(vocab.source_of.get(t), source_names)}" for t in firsts]
            g.message = (f"'{phrase}' could mean {' or '.join(opts)}. Which one do you mean?")
            return g
    sid, tables = next(iter(by_src.items()))
    g.source_id = sid
    g.entity = tables[0]
    g.entity_name = (vocab.cards.get(tables[0]) or {}).get("business_name")
    g.kind = kinds.get(sid) or "sql"
    if g.kind == "rag":             # a card in a document source cannot happen; be safe
        g.kind = "sql"
    return g


def ground_intents(its: Intents, vocab: ScopeVocab,
                   source_names: Optional[Dict[str, str]] = None) -> List[IntentGrounding]:
    out: List[IntentGrounding] = []
    for i, fr in enumerate(its.intents):
        prior = out[fr.depends_on] if fr.depends_on is not None and fr.depends_on < len(out) else None
        g = ground_one(fr, i, vocab, prior=prior, source_names=source_names)
        fr.kind = g.kind
        fr.source_hint = g.source_id
        out.append(g)
    return out


def plan_compound(its: Intents, groundings: List[IntentGrounding], segments: List[str]) -> str:
    """'single' | 'compound'. More than one intent is a compound message — UNLESS the
    segmenter found no clause boundary AND every intent lands on the same source with
    the same lane: then the extractor split ONE question into its clauses ('properties
    and the payments made for them'), which is a join the single path answers."""
    if len(its.intents) <= 1:
        return "single"
    if len(segments) <= 1:
        keys = {(g.source_id, g.kind) for g in groundings if g.outcome == GROUNDED}
        if len(keys) <= 1 and all(g.outcome != CLARIFY for g in groundings):
            return "single"
    return "compound"


# ── composing the reply ──────────────────────────────────────────────────────────────
_DOC_NEGATIVE = re.compile(r"\b(do(?:es)? not|don't|doesn't) (contain|include|mention|provide|have)\b|"
                           r"\bno information\b|\bnot (?:mentioned|covered|found) in the (?:documents?|context|passages?)\b",
                           re.I)


def part_label(i: int, part: str) -> str:
    p = str(part or "").strip().rstrip("?.!, ")
    return f"{i + 1}. {p[:1].upper() + p[1:]}?"


def compose_reply(parts: List[Dict[str, Any]], summary_line: Optional[str] = None) -> str:
    """The reply in PART ORDER: for each part its label, then its answer sentence (or its
    clarify / refusal / timeout note) and its citation; then one summary line.

    `parts`: [{part, outcome, answer, citations, lane}] — outcome ∈ answered | clarify |
    refused | timeout | error. A part routed to the DATA never carries a document-negative
    sentence ("the documents do not contain …") — that sentence is about a different part."""
    blocks = []
    for i, p in enumerate(parts):
        ans = str(p.get("answer") or "").strip()
        oc = p.get("outcome")
        if p.get("lane") in ("sql", "tabular") and _DOC_NEGATIVE.search(ans):
            ans = ""
        if oc == "timeout":
            ans = "This part took too long to answer, so I stopped it — please ask it on its own."
        elif not ans:
            ans = {"clarify": "I need a detail to answer this part.",
                   "refused": "I couldn't answer this part from the data.",
                   "error": "Something went wrong answering this part."}.get(oc, "No answer.")
        cites = [c for c in (p.get("citations") or []) if c][:3]
        if cites:
            ans = strip_sources(ans) or ans
        line = f"**{part_label(i, p.get('part'))}**\n{ans}"
        if cites:
            line += f"\n_Source: {', '.join(cites)}_"
        blocks.append(line)
    if summary_line:
        blocks.append(summary_line.strip())
    return "\n\n".join(blocks)


def _numbers(text: str) -> set:
    """Every figure in a text, normalised ("1,200." → "1200") so sentence punctuation and
    thousands separators never make a copied number look invented."""
    out = set()
    for n in re.findall(r"\d[\d,]*(?:\.\d+)?", str(text or "")):
        n = n.replace(",", "").rstrip(".")
        if n:
            out.add(n)
    return out


def strip_sources(answer: str) -> str:
    """A document answer's trailing 'Sources: …' line (the citation is shown once, below)."""
    return re.sub(r"\n?\s*_?Sources?:[^\n]*_?\s*$", "", str(answer or "").strip(), flags=re.I).strip()


def summary_is_safe(summary: str, parts: List[Dict[str, Any]]) -> bool:
    """A composed summary line may not invent figures and may not claim a document lacks
    something a DATA part answered."""
    if not summary:
        return False
    if not _numbers(summary) <= _numbers(" ".join(str(p.get("answer") or "") for p in parts)):
        return False
    if _DOC_NEGATIVE.search(summary) and any(p.get("lane") in ("sql", "tabular") for p in parts):
        return False
    return True


def fallback_summary(parts: List[Dict[str, Any]]) -> str:
    n = len(parts)
    ok = sum(1 for p in parts if p.get("outcome") == "answered")
    if ok == n:
        return f"All {n} parts of your question are answered above."
    left = [str(i + 1) for i, p in enumerate(parts) if p.get("outcome") != "answered"]
    return (f"{ok} of {n} parts answered; part{'s' if len(left) > 1 else ''} "
            f"{', '.join(left)} need{'' if len(left) > 1 else 's'} a follow-up.")
