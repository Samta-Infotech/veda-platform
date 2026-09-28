"""veda.understanding.frame_extractor — question → Frame (one constrained SLM call).

Input handed to the SLM:
  * the scope's entity cards as CONCEPTS (≤ FRAME_MAX_CONCEPTS): business name, plural,
    what one row is, the lifecycle column's values, key measures and dates as phrases —
    never table or column identifiers;
  * the FRAME_FEWSHOT_K nearest synthetic / trace examples (by BGE-M3 cosine), each with
    its expected frame;
  * the session prior (last turn's entity) when there is one.

Output: a closed-vocabulary frame decoded under a JSON schema (`call_slm(json_schema=…)`,
Ollama `format`), entity closed over the concept names shown; temperature 0, num_predict ≤
FRAME_EXTRACTOR_NUM_PREDICT, num_ctx SLM_NUM_CTX, timeout FRAME_EXTRACTOR_TIMEOUT.
Optional self-consistency (FRAME_SELF_CONSISTENCY): N samples at T=0.2, slot-wise vote.

Then the deterministic producers (veda.understanding.producers) fill / confirm / flag
conflicts slot by slot. Any failure → None (the caller degrades to the existing chain).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from veda.understanding.frame import Frame, json_schema, normalise
from veda.understanding import producers as P
from veda.understanding.vocabulary import ScopeVocab, nearest_examples, norm_phrase

#: The aggregation slot's rule — IDENTICAL wording on both the single-frame prompt
#: (`_SYS`) and the compound intents prompt (`_INTENTS_SYS`), factored once so the two
#: extractors can never drift (§10.3 — the compound path silently dropped filters
#: because its prompt only summarised these rules instead of stating them).
_AGG_RULE = (
    "- aggregation: none | count | count_distinct | sum | avg | min | max. A request to LIST, "
    "SHOW or RANK rows is 'none' — 'cheapest', 'most expensive', 'latest', 'oldest', 'top 5' "
    "are ORDER, not aggregation.\n"
)

#: filters / group_by / order / limit / time / distinct — shared verbatim between `_SYS`
#: and `_INTENTS_SYS` (see `_AGG_RULE`). This is the block whose absence from the
#: compound prompt caused "vendors in Kochi" to become a bare `count group_by city`
#: with the WHERE dropped: the model was never told the filters slot's shape, its op
#: vocabulary, or that "recent"/"latest"/"oldest" are ORDER, not a filter.
_SLOT_RULES = (
    "- filters: conditions on rows [{concept, op, value}]. concept = the attribute phrase "
    "(e.g. 'status', 'price', 'amount'); value = the user's word or number. op one of "
    "= != > >= < <= between in is_null is_not_null. 'between 100 and 50,000' → op between, "
    "value [100, 50000]. Never turn 'recent', 'latest', 'oldest' into a filter.\n"
    "- group_by: attributes after 'per', 'for each', 'grouped by', 'breakdown by'.\n"
    "- order: {concept, dir}. 'most recent/latest/newest' → the date, desc; 'oldest' → the "
    "date, asc; 'cheapest/smallest' → the amount/price, asc; 'most expensive/largest' → desc; "
    "'alphabetical' → the name, asc; 'reverse alphabetical' → the name, desc; 'most recently "
    "modified/updated' → concept 'updated', desc. 'by date' after a superlative is ORDER.\n"
    "- limit: an explicit number of rows ('top 5', 'last 10'), else null.\n"
    "- time: {concept, window{from,to}} only for an explicit period ('last 30 days', 'in "
    "2024'); 'recent' alone is NOT a window. null otherwise.\n"
    "- distinct: true only for 'distinct/unique/different'.\n"
)

_SYS = (
    "You convert a business question into a JSON FRAME that states what the question MEANS. "
    "You never write SQL and never name tables or columns: use the business CONCEPTS listed "
    "and the user's own words.\n"
    "Slots:\n"
    "- entity: the concept whose ROWS the answer lists or counts — pick one of the listed "
    "concept names (its aliases count as that concept). null if none fits.\n"
    "- secondaries: other listed concepts the question also needs (e.g. 'properties and their "
    "payments' → entity the payments concept, secondaries ['property']).\n"
    "- measure: the numeric thing summed/averaged (a phrase), else null.\n"
    + _AGG_RULE
    + _SLOT_RULES
    + "- confidence: 0..1, how sure you are of the entity.\n"
    "Output only the JSON object."
)


_DETAILED = 10           # concepts shown with full detail; the rest compactly (num_ctx budget)


def _line(vocab: ScopeVocab, t: str, name: str, detailed: bool) -> str:
    c = vocab.cards[t]
    pl = c.get("plural")
    al = [a for a in (c.get("aliases") or []) if a not in (name, pl)]
    if not detailed:
        return f"- {name} ({pl}{'; ' + ', '.join(al[:3]) if al[:3] else ''})"
    bits = [f"- {name} (plural: {pl}" + (f"; also called: {', '.join(al[:6])}" if al else "") + ")"]
    if c.get("one_row_is"):
        bits.append(f"— one row is {c['one_row_is'][:90]}")
    life = c.get("lifecycle_column")
    if life:
        vals = [v for v in (vocab.value_glossary.get(f"{t}.{life}") or {})][:5]
        if vals:
            bits.append(f"| {life.replace('_', ' ')}: {', '.join(vals)}")
    ms = [m.replace("_", " ") for m in (c.get("key_measures") or [])[:3]]
    if ms:
        bits.append(f"| measures: {', '.join(ms)}")
    if c.get("business_date_column"):
        bits.append(f"| date: {c['business_date_column'].replace('_', ' ')}")
    dist = [ph for o, ph in (c.get("distinguishes_from") or {}).items() if o in vocab.cards][:2]
    if dist:
        bits.append("| not to be confused: " + " / ".join(dist))
    return " ".join(bits)


def _concepts(vocab: ScopeVocab, query: str, k: int) -> Tuple[List[str], List[str], List[str], List[str]]:
    """(static_lines, question_lines, entity names, their scope tables).

    The STATIC block — the scope's cards by importance, identical for every question — is
    what makes the prompt cheap: Ollama reuses the evaluated prefix across calls, so only
    the few-shots and the question are evaluated per call (measured: a 2.3k-token prompt
    took 16 s to evaluate on the local 7B). Cards the question NAMES but the static block
    does not reach go in a short per-question block after it, with full detail."""
    static = [t for t in sorted(vocab.cards, key=lambda t: (-vocab.cards[t].get("importance", 0), t))
              if vocab.cards[t].get("table_type") != "BRIDGE"][:k]
    qn = " " + norm_phrase(query) + " "
    hit_tables = []
    for phrase, lst in vocab.name_index().items():
        if phrase and f" {phrase} " in qn:
            for t, _kind in lst:
                if t not in hit_tables:
                    hit_tables.append(t)
    names, tables, static_lines, q_lines = [], [], [], []

    def _name(t):
        n = vocab.cards[t].get("business_name") or t
        return n if n not in names else f"{n} ({t.split('_')[0]})"
    for i, t in enumerate(static):
        n = _name(t)
        names.append(n)
        tables.append(t)
        static_lines.append(_line(vocab, t, n, i < _DETAILED))
    for t in hit_tables:
        if t in tables:
            continue
        n = _name(t)
        names.append(n)
        tables.append(t)
        q_lines.append(_line(vocab, t, n, True))
    return static_lines, q_lines, names, tables


def _fewshot(vocab: ScopeVocab, query: str, k: int) -> List[str]:
    out = []
    for ex in nearest_examples(vocab, query, k):
        fr = {k: v for k, v in (ex.get("frame") or {}).items()
              if k != "confidence" and v not in (None, [], {}, False, "none")}
        out.append(f"Q: {ex['question']}\nFRAME: {json.dumps(fr, separators=(',', ':'))}")
    return out


def _session_prior() -> Optional[str]:
    try:
        from veda_core.context import current_conversation_context
    except Exception:
        try:
            from context import current_conversation_context      # type: ignore
        except Exception:
            return None
    try:
        cc = current_conversation_context() or {}
    except Exception:
        return None
    t = cc.get("entity_table")
    return t or None


def _call(user: str, system: str, schema: Dict[str, Any], *, temperature: float, seed: int,
          num_predict: int, timeout: int) -> Optional[Dict[str, Any]]:
    try:
        from slm import call_slm
        import config
        raw = call_slm(user, system=system, purpose="frame_extract", temperature=temperature,
                       seed=seed, num_predict=num_predict,
                       num_ctx=getattr(config, "SLM_NUM_CTX", 4096), timeout=timeout,
                       json_schema=schema)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {str(e)[:160]}"}
    try:
        obj = json.loads(raw)
    except Exception:
        # constrained decoding should make this impossible; count it if it happens
        s = (raw or "").strip()
        i, j = s.find("{"), s.rfind("}")
        try:
            obj = json.loads(s[i:j + 1]) if i >= 0 < j else None
        except Exception:
            obj = None
        if obj is None:
            return {"_error": "unparseable", "_raw": s[:200]}
    return obj if isinstance(obj, dict) else {"_error": "not_an_object"}


# ══ compound messages: message → {intents: [Frame…], relation} ════════════════════════
_INTENTS_SYS = (
    "A user message may contain SEVERAL independent questions, each about a different "
    "thing (a database, a spreadsheet, a policy document). Split the message into its "
    "questions and give ONE frame per question, in message order. You never write SQL "
    "and never name tables or columns: use the CONCEPTS and DOCUMENTS listed and the "
    "user's own words.\n"
    "For each intent:\n"
    "- part: the words of the message this intent answers, copied as a standalone question.\n"
    "- kind: 'rag' when the answer is in a DOCUMENT (policies, rules, entitlements, how-to, "
    "'can I', 'do I need', 'how many days of leave'); 'sql' when it is in the DATA "
    "(listing, counting, ranking or totalling records).\n"
    "- entity: for sql, the concept whose ROWS the answer lists or counts (one of the "
    "listed concept names); for rag, the document title.\n"
    "- topics (rag only): 1-3 short topic phrases from the document's sections.\n"
    "- measure: the numeric thing summed/averaged (a phrase), else null.\n"
    + _AGG_RULE
    + _SLOT_RULES
    + "The filters/group_by/order/limit/time/distinct rules above apply to sql intents "
    "only — a rag intent leaves them empty/null and uses topics instead.\n"
    "- depends_on: the index of an EARLIER intent whose RESULT this one uses ('…and for "
    "those, …'), else null.\n"
    "relation: 'dependent' only when some intent has depends_on; else 'independent'.\n"
    "A single question is ONE intent — do not split one question into its clauses "
    "('properties and their payments' is one question). Output only the JSON object."
)

_INTERROG = (r"what|which|who|whom|whose|when|where|why|how|do|does|did|is|are|was|were|"
             r"can|could|should|shall|will|would|may|list|show|give|tell|name|find")
# a boundary is a clause break followed by a SECOND interrogative (or an explicit joiner)
_SEG_BOUNDARY = re.compile(
    rf"(?:\?\s+(?=\S)|;\s*(?:and\s+|also\s+|plus\s+)?|,\s*(?:and\s+|or\s+|also\s+|plus\s+)?(?=(?:{_INTERROG})\b)"
    rf"|,?\s+and also\s+|,\s*plus\s+|,?\s+additionally,?\s+)", re.I)
_ANAPH = re.compile(r"\b(those|them|these|their|its|it|that one|such)\b", re.I)


def segment(message: str) -> List[str]:
    """Deterministic candidate parts of a message: split at `;`, a mid-message `?`,
    ' and also ', ', plus ', ' additionally ', or a comma followed by a SECOND
    interrogative ('…, what are…', '…, and which…', '…, can I…'). Never inside
    parentheses. A PRODUCER, not a gate: the extractor's list wins; segments only
    confirm its parts or add one it missed."""
    s = str(message or "").strip()
    if not s:
        return []
    # mask parenthesised spans so a boundary inside them is never taken
    masked, depth = [], 0
    for ch in s:
        if ch == "(":
            depth += 1
        masked.append("\0" if depth and ch in ",;?" else ch)
        if ch == ")" and depth:
            depth -= 1
    m = "".join(masked)
    cuts = [0]
    for mt in _SEG_BOUNDARY.finditer(m):
        cuts.append(mt.start())
        cuts.append(mt.end())
    cuts.append(len(s))
    out = []
    for i in range(0, len(cuts) - 1, 2):
        seg = s[cuts[i]:cuts[i + 1]].strip(" ,;?.").strip()
        seg = re.sub(r"^(and|also|plus|or)\s+", "", seg, flags=re.I).strip()
        if seg:
            out.append(seg)
    # a fragment too short to be a question on its own belongs to its neighbour
    merged: List[str] = []
    for seg in out:
        if merged and len(_content_words(seg)) < 2:
            merged[-1] = f"{merged[-1]}, {seg}"
        else:
            merged.append(seg)
    return merged


def _content_words(s: str) -> List[str]:
    stop = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", "by", "is",
            "are", "was", "were", "do", "does", "did", "i", "me", "my", "we", "our", "you",
            "can", "what", "which", "how", "who", "there", "it", "its", "be", "any", "all"}
    return [w for w in norm_phrase(s).split() if w not in stop and len(w) > 1]


def _overlap(a: str, b: str) -> float:
    """share of b's content words that a contains"""
    bw = set(_content_words(b))
    if not bw:
        return 0.0
    return len(set(_content_words(a)) & bw) / len(bw)


def reconcile(frames: List[Frame], segments: List[str], message: str) -> Tuple[List[Frame], Dict[str, Any]]:
    """The extractor's list wins; segments confirm or add. Returns (frames in message
    order, diagnostics)."""
    diag: Dict[str, Any] = {"segments": list(segments), "added": [], "filled_part": []}
    frames = [f for f in frames if f is not None]
    for i, f in enumerate(frames):
        if not f.part:
            # a frame with no part text: the segment it overlaps most, else the message
            best = max(segments, key=lambda s: _overlap(s, " ".join(
                [f.entity or ""] + [x.concept for x in f.filters] + list(f.topics))), default=None)
            f.part = best or message
            diag["filled_part"].append(i)
    covered = set()
    for j, seg in enumerate(segments):
        for f in frames:
            if _overlap(f.part or "", seg) >= 0.5 or _overlap(seg, f.part or "") >= 0.6:
                covered.add(j)
                break
    for j, seg in enumerate(segments):
        if j in covered or len(segments) < 2:
            continue
        if _ANAPH.search(seg) or len(_content_words(seg)) < 3:
            continue          # continues the part before it; not a question of its own
        nf = Frame(part=seg)
        nf.provenance["added_by"] = "segment"
        frames.append(nf)
        diag["added"].append(seg)

    low = message.lower()

    def pos(f):
        p = (f.part or "").lower()
        i = low.find(p[:40]) if p else -1
        if i >= 0:
            return i
        # position of the best-overlapping segment
        best = max(range(len(segments)), key=lambda k: _overlap(segments[k], p), default=None)
        return low.find(segments[best].lower()[:40]) if best is not None else 10 ** 6
    order = sorted(range(len(frames)), key=lambda k: (pos(frames[k]), k))
    remap = {old: new for new, old in enumerate(order)}
    out = [frames[k] for k in order]
    for pos_f, f in enumerate(out):
        if f.depends_on is not None:
            f.depends_on = remap.get(f.depends_on)
            if f.depends_on is not None and f.depends_on >= pos_f:
                f.depends_on = None
    return out, diag


def _scope_concepts(vocab: ScopeVocab, query: str, k: int) -> Tuple[List[str], Dict[str, str]]:
    """Concept lines for the WHOLE scope: every card of a small source (a spreadsheet's two
    tables), the rest of the budget to the big sources by importance, plus any card the
    message names. Compact lines (name, plural, aliases) — the intent call is budgeted."""
    by_src: Dict[str, List[str]] = {}
    for t in sorted(vocab.cards, key=lambda t: (-vocab.cards[t].get("importance", 0), t)):
        if vocab.cards[t].get("table_type") == "BRIDGE":
            continue
        by_src.setdefault(vocab.source_of.get(t, "?"), []).append(t)
    chosen: List[str] = []
    small = [s for s, ts in by_src.items() if len(ts) <= 10]
    for s in small:
        chosen += by_src[s]
    big = [s for s in by_src if s not in small]
    room = max(0, k - len(chosen))
    per = (room // len(big)) if big else 0
    for s in big:
        chosen += by_src[s][:per]
    qn = " " + norm_phrase(query) + " "
    for phrase, lst in vocab.name_index().items():
        if phrase and f" {phrase} " in qn:
            for t, _kind in lst:
                if t not in chosen:
                    chosen.append(t)
    lines, names = [], {}
    for t in chosen:
        c = vocab.cards[t]
        n = c.get("business_name") or t
        if n in names:
            n = f"{n} ({t.split('_')[0]})"
        names[n] = t
        pl = c.get("plural")
        al = [a for a in (c.get("aliases") or []) if a not in (n, pl)][:3]
        lines.append(f"- {n} ({pl}{'; ' + ', '.join(al) if al else ''})")
    return lines, names


def _doc_lines(doc_cards: List[Dict[str, Any]], query: str, max_secs: int = 40) -> List[str]:
    out = []
    qn = set(_content_words(query))
    for d in doc_cards:
        secs = list(d.get("sections") or [])
        # the sections the message touches first, then the rest in document order
        hit = [s for s in secs if set(_content_words(s)) & qn]
        rest = [s for s in secs if s not in hit]
        shown = (hit + rest)[:max_secs]
        out.append(f"- {d.get('title')}" + (f" — sections: {'; '.join(x.lower() for x in shown)}" if shown else ""))
    return out


def extract_intents(query: str, vocab: ScopeVocab, *, doc_cards: Optional[List[Dict[str, Any]]] = None,
                    stats: Optional[Dict[str, Any]] = None):
    """Message → Intents (1..5 frames + relation), or None on failure. One constrained SLM
    call; the deterministic segmenter confirms / adds parts afterwards. Each frame's
    entity is a CONCEPT NAME (or a document title for a rag frame); `provenance
    ['entity_table']` records the card it names when it is a card."""
    import config
    from veda.understanding.frame import Intents, intents_json_schema, normalise_intents
    st = stats if stats is not None else {}
    doc_cards = list(doc_cards or [])
    k = int(getattr(config, "FRAME_MAX_CONCEPTS", 40))
    lines, shown = _scope_concepts(vocab, query, k)
    titles = [d.get("title") for d in doc_cards if d.get("title")]
    st["concepts_shown"] = len(lines)
    st["documents_shown"] = len(titles)
    parts = ["CONCEPTS (records in the data):", *lines, ""]
    if doc_cards:
        parts += ["DOCUMENTS (policies and text):", *_doc_lines(doc_cards, query), ""]
    try:
        from ingestion.vocabulary import compound_examples
        exs = compound_examples(vocab.cards, doc_cards, vocab.source_of,
                                vg=vocab.value_glossary, mg=vocab.measure_glossary)
    except Exception:
        exs = []
    st["fewshot"] = len(exs)
    if exs:
        parts.append("EXAMPLES:")
        for ex in exs:
            its = [{kk: vv for kk, vv in it.items() if vv not in (None, [], {}, "none")}
                   for it in ex["intents"]]
            parts.append(f"Q: {ex['question']}\nINTENTS: " + json.dumps(
                {"intents": its, "relation": ex["relation"]}, separators=(",", ":")))
        parts.append("")
    parts += [f"MESSAGE: {query}", "INTENTS:"]
    schema = intents_json_schema(entity_enum=list(shown) or None, doc_enum=titles or None)
    npred = int(getattr(config, "FRAME_INTENTS_NUM_PREDICT", 700))
    tmo = int(getattr(config, "FRAME_INTENTS_TIMEOUT", 40))
    t0 = time.time()
    obj = _call("\n".join(parts), _INTENTS_SYS, schema, temperature=0.0, seed=0,
                num_predict=npred, timeout=tmo)
    st["slm_ms"] = round((time.time() - t0) * 1000.0, 1)
    segs = segment(query)
    st["segments"] = segs
    if not obj or obj.get("_error"):
        st["error"] = (obj or {}).get("_error", "none")
        return None
    its = normalise_intents(obj)
    for fr in its.intents:
        if fr.entity and fr.entity in shown:
            fr.provenance["entity_table"] = shown[fr.entity]
        elif fr.entity and fr.entity in titles:
            fr.provenance["document"] = fr.entity
            fr.kind = "rag"
        fr.provenance.update({s: "slm" for s in ("entity", "measure", "aggregation", "order", "limit")
                              if getattr(fr, s, None) not in (None, "none", [])})
    frames, diag = reconcile(its.intents, segs, query)
    st["reconcile"] = diag
    st["n_intents"] = len(frames)
    rel = its.relation if any(f.depends_on is not None for f in frames) else "independent"
    return Intents(intents=frames[:5], relation=rel)


def extract_frame(query: str, vocab: ScopeVocab, *, use_producers: bool = True,
                  stats: Optional[Dict[str, Any]] = None,
                  observations: Optional[List[str]] = None) -> Optional[Frame]:
    """Question → Frame (entity as a CONCEPT NAME from the cards), or None on failure.
    `stats` (optional dict) receives timing/parse diagnostics for the trace and the eval."""
    import config
    st = stats if stats is not None else {}
    k = int(getattr(config, "FRAME_MAX_CONCEPTS", 60))
    lines, q_lines, names, tables = _concepts(vocab, query, k)
    shown = dict(zip(names, tables))            # the enum is over the names the model saw
    st["concepts_shown"] = len(names)

    parts = ["CONCEPTS (business entities in scope):", *lines, ""]
    if q_lines:
        parts += ["MORE CONCEPTS named in this question:", *q_lines, ""]
    fs = _fewshot(vocab, query, int(getattr(config, "FRAME_FEWSHOT_K", 5)))
    st["fewshot"] = len(fs)
    if fs:
        parts += ["EXAMPLES:", *fs, ""]
    prior = _session_prior()
    if prior and prior in vocab.cards:
        parts.append(f"The previous question in this conversation was about: "
                     f"{vocab.cards[prior].get('plural') or vocab.cards[prior].get('business_name')}.")
    if observations:
        # Stage 4 revision round: what the data said about the previous frame — evidence
        # for a re-extraction of the FRAME, never an instruction to write SQL
        parts += ["OBSERVATIONS FROM THE DATA about a previous reading of this question:",
                  *[f"- {o}" for o in observations], ""]
    parts += [f"QUESTION: {query}", "FRAME:"]
    user = "\n".join(parts)
    schema = json_schema(entity_enum=names or None)
    npred = int(getattr(config, "FRAME_EXTRACTOR_NUM_PREDICT", 200))
    tmo = int(getattr(config, "FRAME_EXTRACTOR_TIMEOUT", 20))
    t0 = time.time()
    samples: List[Frame] = []
    n = int(getattr(config, "FRAME_SELF_CONSISTENCY_N", 3)) if getattr(config, "FRAME_SELF_CONSISTENCY", False) else 1
    errors = []
    for i in range(max(1, n)):
        obj = _call(user, _SYS, schema, temperature=(0.0 if n == 1 else 0.2), seed=i,
                    num_predict=npred, timeout=tmo)
        if not obj or obj.get("_error"):
            errors.append((obj or {}).get("_error", "none"))
            continue
        samples.append(normalise(obj))
    st["slm_ms"] = round((time.time() - t0) * 1000.0, 1)
    st["samples"] = len(samples)
    st["errors"] = errors
    if not samples:
        st["parse_failure"] = True
        return None
    if len(samples) > 1:
        fr, unc = P.vote(samples)
        st["uncertain"] = unc
    else:
        fr = samples[0]
    fr.provenance.update({s: "slm" for s in ("entity", "measure", "aggregation", "order", "limit")
                          if getattr(fr, s, None) not in (None, "none", [])})
    # entity: a shown NAME → keep the name, remember its table for grounding
    if fr.entity and fr.entity in shown:
        fr.provenance["entity_table"] = shown[fr.entity]
    if use_producers:
        frags = P.run_producers(query, vocab)
        st["producers"] = sorted(frags)
        fr = P.merge(fr, frags)
    return fr
