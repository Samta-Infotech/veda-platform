"""chatbot.memory.delta — DETERMINISTIC follow-up delta detection (M4 / Checkpoint C.3).

WHY THIS EXISTS
---------------
Before this module every follow-up turn cost an SLM classification call, because the only
thing that could say what "of those, only the Kochi ones" meant was a model. Measured on
this deployment (scripts/eval_sessions.py, script s4) that made a follow-up cost 2 SLM
calls — one to classify, one to summarise — against a target of 1.

But almost every real follow-up is a closed-class construction over a KNOWN set of slots.
"only the X ones", "by category", "top 3", "go back" are grammar, not judgment, once you
know which dimensions and values the previous result actually had — and the previous turn
already told us exactly that (FrameEntry.drill_options / result.top_values, straight off
the engine's one post-execution analysis pass). So the rules run first and the SLM runs
only on what the rules genuinely cannot decide.

WHAT MAKES THIS SAFE
--------------------
The rules never invent a value. A value is only accepted when it matches something the
PREVIOUS RESULT actually contained (`top_values`), or the message names a dimension the
previous result actually had (`drill_options.dimensions`). Anything else returns
AMBIGUOUS, which is the caller's signal to fall through to the SLM classifier — this
module's failure mode is "ask the model", never "guess a filter".

Pure functions, no I/O, no LLM, no veda_core import (chatbot/ runs in the api tier — the
same boundary frame.py documents).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# the closed operation set — mirrors the plan's vocabulary
OP_ADD_FILTER = "add_filter"
OP_CHANGE_GROUP = "change_group"
OP_CHANGE_MEASURE = "change_measure"
OP_CHANGE_ORDER = "change_order"
OP_DRILL_UP = "drill_up"
OP_SWITCH_FRAME = "switch_frame"
OP_NEW_TOPIC = "new_topic"
OP_AMBIGUOUS = "ambiguous"

OPS = (OP_ADD_FILTER, OP_CHANGE_GROUP, OP_CHANGE_MEASURE, OP_CHANGE_ORDER,
       OP_DRILL_UP, OP_SWITCH_FRAME, OP_NEW_TOPIC, OP_AMBIGUOUS)

# ── surface cues ─────────────────────────────────────────────────────────────
# Anaphora: the message refers back to the previous result rather than naming a subject.
# Its PRESENCE is what makes a bare fragment a follow-up at all.
_ANAPHORA = re.compile(
    r"\b(of (those|them|these)|those|them|these|that|it|its|their|the same|"
    r"the above|from (that|those))\b", re.I)

_DRILL_UP = re.compile(
    r"^\s*(go\s+back|back|previous|undo|revert|overall|"
    r"the\s+whole\s+thing|before\s+that)\b", re.I)

_GROUP = re.compile(r"\b(by|per|broken\s+down\s+by|grouped?\s+by|group\s+by)\b", re.I)
_INSTEAD = re.compile(r"\binstead\b", re.I)

_ORDER = re.compile(
    r"\b(top|bottom|highest|lowest|largest|smallest|best|worst|first|last|"
    r"most|least|cheapest|biggest|longest|shortest|oldest|newest)\b", re.I)
_LIMIT_N = re.compile(r"\b(?:top|bottom|first|last)\s+(\d{1,4})\b", re.I)

_ONLY = re.compile(r"\b(only|just|filter(?:ed)?\s+(?:to|by)|where|with|in|for)\b", re.I)

# A message that OPENS with one of these is continuing the previous turn, whatever else
# it contains. Distinct from _ONLY (which matches mid-sentence prepositions like "in"
# that appear in ordinary standalone questions) — this is anchored at the start.
_FOLLOWUP_LEAD = re.compile(
    r"^\s*(only|just|of\s+(those|them|these)|and|but|what\s+about|how\s+about|"
    r"also|now|then|excluding|without)\b", re.I)

_MEASURE_WORDS = {
    "count": "count", "how many": "count", "number": "count",
    "total": "sum", "sum": "sum",
    "average": "avg", "avg": "avg", "mean": "avg",
    "max": "max", "maximum": "max", "highest": "max",
    "min": "min", "minimum": "min", "lowest": "min",
}

# Words that never identify a dimension or a value on their own.
_STOP = {
    "the", "a", "an", "of", "those", "them", "these", "that", "this", "it", "its",
    "their", "and", "or", "but", "to", "in", "on", "for", "with", "by", "per", "from",
    "only", "just", "show", "me", "give", "list", "what", "which", "how", "many", "much",
    "is", "are", "was", "were", "do", "does", "did", "have", "has", "had", "can", "could",
    "instead", "also", "now", "then", "about", "into", "out", "up", "down", "back",
    "break", "broken", "group", "grouped", "top", "bottom", "first", "last", "same",
    "there", "here", "one", "ones", "any", "all", "some", "each", "every",
}


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _stem(t: str) -> str:
    """Crude English plural strip, enough to match a user's plural against a stored
    singular value. Users type "repairs"; the column holds "Repair". Matching on raw
    substrings missed every such pair, so a real value-narrowing turn ("how many of
    those are repairs") looked like no value at all and fell through to the SLM."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("es") and not t.endswith(("ses", "zes")):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _stems(s: str) -> List[str]:
    return [_stem(t) for t in _norm(s).split() if t]


def _phrase_in(needle: str, haystack_stems: List[str]) -> bool:
    """Is `needle` (a value or dimension name) present in the message as a contiguous
    run of stemmed tokens? Stem-to-stem on both sides, so singular/plural and simple
    inflection differences match without a wordlist."""
    ns = [_stem(t) for t in _norm(needle).split() if t]
    if not ns:
        return False
    n = len(ns)
    return any(haystack_stems[i:i + n] == ns for i in range(len(haystack_stems) - n + 1))


def _tokens(s: str) -> List[str]:
    return [t for t in _norm(s).split() if t and t not in _STOP]


def _match_dimension(message: str, dimensions: List[str]) -> Optional[str]:
    """A dimension named in the message, matched on its normalised token set.

    Longest match wins so "property type" beats "type". Returns None unless exactly one
    dimension matches — an ambiguous mention is not a decision this layer may make.
    """
    ms = _stems(message)
    hits = []
    for d in dimensions or []:
        dn = _norm(d)
        if not dn:
            continue
        if _phrase_in(dn, ms):
            hits.append((len(dn), d))
            continue
        # also match the bare column name of a qualified/underscored dimension
        alt = _norm(str(d).replace("_", " ").split(".")[-1])
        if alt and alt != dn and _phrase_in(alt, ms):
            hits.append((len(alt), d))
    if not hits:
        return None
    hits.sort(reverse=True)
    if len(hits) > 1 and hits[0][0] == hits[1][0]:
        return None                      # two equally-good readings — not ours to pick
    return hits[0][1]


def _match_value(message: str, top_values: Dict[str, List[str]]) -> Optional[Tuple[str, str]]:
    """(column, value) for a literal the message names that the PREVIOUS RESULT actually
    contained. This is the anti-hallucination rule of this module: a filter value is only
    ever a value we have already seen in the data, never a noun lifted out of the message.
    Longest value wins ("New York" over "New"); an equal-length tie returns None."""
    ms = _stems(message)
    hits = []
    for col, vals in (top_values or {}).items():
        for v in vals or []:
            vn = _norm(v)
            if len(vn) < 2:
                continue
            if _phrase_in(vn, ms):
                hits.append((len(vn), col, v))
    if not hits:
        return None
    hits.sort(reverse=True)
    if len(hits) > 1 and hits[0][0] == hits[1][0] and hits[0][1] != hits[1][1]:
        return None
    return hits[0][1], hits[0][2]


def _match_measure(message: str, measures: List[str]) -> Optional[Tuple[str, Optional[str]]]:
    """(aggregation, column) when the message asks for a different measure."""
    low = f" {_norm(message)} "
    agg = None
    for word, a in _MEASURE_WORDS.items():
        if f" {_norm(word)} " in low:
            agg = a
            break
    if agg is None:
        return None
    col = _match_dimension(message, measures or [])
    return agg, col


def _frame_reference(message: str, stack_questions: List[str]) -> Optional[int]:
    """The index of an EARLIER stack entry the message points back to ("the Mumbai one",
    "go to the vendors question"). Scored by how many of that turn's own content words
    the message repeats; a single shared word is not enough to move the cursor."""
    toks = set(_tokens(message))
    if not toks:
        return None
    best, best_score = None, 0
    for i, q in enumerate(stack_questions):
        overlap = len(toks & set(_tokens(q)))
        if overlap > best_score:
            best, best_score = i, overlap
    return best if best_score >= 2 else None


def detect(message: str, entry: Optional[Dict[str, Any]],
           stack: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Classify one follow-up against the CURRENT stack entry, deterministically.

    Returns {op, slot, concept, value, target_frame_index, confidence, rule} — the same
    shape the SLM classifier returns, so the caller handles both identically. `op` is
    OP_AMBIGUOUS when the rules cannot decide; that is the caller's cue to spend the SLM
    call, and it is the ONLY path on which one is spent.
    """
    out = {"op": OP_AMBIGUOUS, "slot": None, "concept": None, "value": None,
           "target_frame_index": None, "confidence": 0.0, "rule": None}
    msg = str(message or "").strip()
    if not msg:
        return out

    # No previous answered turn → nothing to be a delta OF.
    if not entry:
        return {**out, "op": OP_NEW_TOPIC, "confidence": 1.0, "rule": "no_frame"}

    opts = entry.get("drill_options") or {}
    dims = list(opts.get("dimensions") or [])
    measures = list(opts.get("measures") or [])
    top_values = (entry.get("result") or {}).get("top_values") or {}

    # 1. drill_up — a pure navigation trigger, no data content of its own.
    if _DRILL_UP.match(msg):
        return {**out, "op": OP_DRILL_UP, "confidence": 1.0, "rule": "drill_up_phrase"}

    # 2. switch_frame — names an EARLIER turn strongly enough to move the cursor.
    stack_qs = [str(e.get("question") or "") for e in (stack or [])]
    if len(stack_qs) > 1:
        idx = _frame_reference(msg, stack_qs[:-1])
        if idx is not None and _ANAPHORA.search(msg) is None and len(_tokens(msg)) <= 8:
            return {**out, "op": OP_SWITCH_FRAME, "target_frame_index": idx,
                    "confidence": 0.8, "rule": "frame_reference"}

    has_anaphora = bool(_ANAPHORA.search(msg))
    n_tokens = len(_tokens(msg))

    # 3. change_order / limit — "top 3", "show the top 3 by amount".
    m_lim = _LIMIT_N.search(msg)
    if m_lim and (has_anaphora or n_tokens <= 6):
        try:
            limit = int(m_lim.group(1))
        except ValueError:
            limit = None
        if limit:
            return {**out, "op": OP_CHANGE_ORDER, "slot": "limit", "value": limit,
                    "concept": _match_dimension(msg, dims + measures),
                    "confidence": 0.95, "rule": "limit_n"}

    # 4. change_group — "by category", "broken down by city", "group by city instead".
    if _GROUP.search(msg):
        dim = _match_dimension(msg, dims)
        if dim:
            return {**out, "op": OP_CHANGE_GROUP, "slot": "group_keys", "concept": dim,
                    "confidence": 0.95 if (has_anaphora or _INSTEAD.search(msg) or n_tokens <= 6) else 0.75,
                    "rule": "group_by_dimension"}

    # 5. add_filter — a value the PREVIOUS RESULT actually contained.
    #    Checked BEFORE the measure rule: "how many of those are repairs" contains the
    #    measure word "how many", but the measure is unchanged (it was already a count)
    #    and the actual delta is the new filter. Ordering measure first classified every
    #    such turn as change_measure and silently dropped the narrowing.
    val = _match_value(msg, top_values)
    if val:
        col, v = val
        if has_anaphora or _ONLY.search(msg) or n_tokens <= 5:
            return {**out, "op": OP_ADD_FILTER, "slot": "filters", "concept": col,
                    "value": v, "confidence": 0.95, "rule": "value_in_top_values"}

    # An UNACCOUNTED literal vetoes a confident in-scope edit. "how many of those are in
    # Mumbai" matches the measure rule below on "how many" — but "Mumbai" is a value this
    # result does not contain, and answering the measure change while dropping the filter
    # produced the exact wrong answer this layer exists to prevent (source 4's maintenance
    # table has no city at all, and the turn reported "8 maintenances are recorded in
    # Mumbai"). If the message carries a content word we cannot account for against this
    # result's own dimensions, measures or values, it is not an in-scope edit — it goes to
    # the classifier, and the caller widens the scope so routing can move sources.
    _accounted = set()
    for n in list(dims) + list(measures) + list(_MEASURE_WORDS):
        _accounted.update(_stem(t) for t in _norm(n).split())
    for _vals in top_values.values():
        for _v in (_vals or []):
            _accounted.update(_stem(t) for t in _norm(_v).split())
    _unaccounted = [t for t in _tokens(msg) if _stem(t) not in _accounted]
    if _unaccounted:
        return {**out, "op": OP_AMBIGUOUS, "confidence": 0.0, "continuation": True,
                "rule": f"unaccounted_literal:{_unaccounted[0]}"}

    # 6. change_measure — "what is the average rent of those".
    #    Requires a back-reference, OR a measure column this result actually has. Without
    #    that second condition a standalone question that merely starts with "how many"
    #    ("how many properties are there" — a NEW topic on a different entity) was read as
    #    a measure change on the previous frame.
    mm = _match_measure(msg, measures)
    if mm:
        agg, col = mm
        if has_anaphora:
            return {**out, "op": OP_CHANGE_MEASURE, "slot": "measure", "concept": col,
                    "value": agg, "confidence": 0.9 if col else 0.8, "rule": "measure_word"}
        if col and n_tokens <= 7:
            return {**out, "op": OP_CHANGE_MEASURE, "slot": "measure", "concept": col,
                    "value": agg, "confidence": 0.85, "rule": "measure_word_named_column"}

    # 7. A self-contained sentence with no back-reference is a new topic.
    #    Measured on RAW word count, not content tokens: "how many properties are there"
    #    is an unmistakably standalone question, but four of its five words are stopwords,
    #    so a content-token threshold saw length 1 and returned ambiguous — spending an
    #    SLM call on the easiest classification in the set. Rules 1-6 have already claimed
    #    every follow-up shape that mentions a known dimension, value or measure, so what
    #    reaches here with no anaphora and a full sentence's worth of words is a new topic.
    # A SHORT superlative ("which is the most expensive", "who has the highest rating")
    # is a question about the set already on screen, not a new subject — it names no
    # entity of its own. Letting rule 7 call it a new topic re-routed it from scratch and
    # dropped every filter the user had built up. It is not confidently resolvable either
    # (which measure?), so it falls through to the classifier rather than being guessed.
    _short_superlative = bool(_ORDER.search(msg)) and len(_norm(msg).split()) <= 6

    # An ELLIPTICAL question names no subject at all — "how many are there", "which
    # ones", "and the average". Every one of its words is a function word, so there is
    # nothing for a new topic to be ABOUT: it can only be asking about the set already on
    # screen. Rule 7 counted raw words and called these new topics, which reset the scope
    # and lost the thread ("how many are there" on script s5, turn 2).
    _elliptical = not _tokens(msg)

    if (not has_anaphora and not _FOLLOWUP_LEAD.match(msg) and not _short_superlative
            and not _elliptical and len(_norm(msg).split()) >= 4):
        return {**out, "op": OP_NEW_TOPIC, "confidence": 0.7, "rule": "standalone_sentence"}

    # Opens as a continuation but names nothing we can ground — e.g. "only the Bangalore
    # ones" where Bangalore is not among this result's values. This is NOT a new topic
    # (calling it one silently drops the user's narrowing and re-routes from scratch) and
    # it is not a filter we may invent. It is exactly what the SLM classifier is for.
    if _FOLLOWUP_LEAD.match(msg) or _short_superlative or _elliptical:
        # `continuation` says: we KNOW this message continues the previous turn (it opens
        # as a continuation, or it is a bare superlative/elliptical question naming no
        # subject of its own), but not what it does to the IR. The caller uses it to veto
        # a `new_topic` classification — keeping the scope and the frame — while still
        # letting the SLM decide the operation.
        return {**out, "op": OP_AMBIGUOUS, "confidence": 0.0,
                "continuation": True, "rule": "unresolved_continuation"}

    # Anything else: a back-reference we could not resolve to a slot. Ask the model.
    return out


def has_back_reference(message: str) -> bool:
    """Does this message point back at the previous result ("them", "those", "that one")?

    Objective surface evidence, used to VETO a `new_topic` classification. A message
    containing a back-reference pronoun cannot be a new topic — there is nothing for the
    pronoun to refer to in a new topic — yet the classifier returns one often enough to
    matter: on script s4 it called both "which vendors handled them" and "who has the
    highest rating" new topics, which reset the session scope to every source and made
    both turns fail against a deployment that cannot answer unpinned multi-source
    questions at all. Keeping the scope costs nothing when the classifier is right (the
    engine re-grounds either way); losing it costs the whole turn when it is wrong.
    """
    return bool(_ANAPHORA.search(str(message or "")))


def sources_named(message: str, entities_by_source: Dict[str, List[str]],
                  exclude: Optional[List[int]] = None) -> List[int]:
    """Source ids whose OWN entities this message names, excluding `exclude`.

    The question a session-sticky scope cannot answer on its own: has the user moved to
    subject matter that lives somewhere else? "which properties do they belong to" asked
    of a maintenance session names `properties`, which is source 2's entity, not source
    4's — and answering it from source 4 alone produces a confident answer to a different
    question (measured: source 4's maintenance table has no city column, yet a Mumbai
    follow-up reported "8 maintenances are recorded in Mumbai").

    Entity names come from each source's routing card via the api tier
    (apps/query/scope.py::_routing_card_entities) — this module reads no files and
    imports no veda_core. Stem-matched, so "properties" matches the entity "property".
    """
    ms = _stems(message)
    excl = {int(s) for s in (exclude or [])}
    hits = []
    for sid, names in (entities_by_source or {}).items():
        try:
            sid_i = int(sid)
        except (TypeError, ValueError):
            continue
        if sid_i in excl:
            continue
        for n in (names or []):
            nn = _norm(n)
            # single-token entity names must be real words, not ids/codes
            if len(nn) < 4:
                continue
            if _phrase_in(nn, ms) and sid_i not in hits:
                hits.append(sid_i)
                break
    return hits


def continues_thread(delta: Optional[Dict[str, Any]], message: str) -> bool:
    """True when this message demonstrably continues the previous turn — either it
    carries a back-reference, or the rule layer flagged it as an unresolvable
    continuation. Used to veto a `new_topic` classification (see has_back_reference)."""
    return bool((delta or {}).get("continuation")) or has_back_reference(message)


def is_confident(delta: Dict[str, Any], threshold: float = 0.75) -> bool:
    """True when the rule layer resolved the turn well enough to skip the SLM call."""
    return (delta.get("op") not in (None, OP_AMBIGUOUS)
            and float(delta.get("confidence") or 0.0) >= threshold)
