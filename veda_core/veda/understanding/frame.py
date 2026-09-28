"""veda.understanding.frame — the closed-vocabulary QUESTION FRAME (meaning-first pass).

A frame is what a question MEANS, settled once, before any SQL exists:

    {entity, secondaries[], measure, aggregation, filters[{concept, op, value}],
     group_by[], order{concept, dir}|null, limit|null,
     time{concept, window{from,to}|null}|null, distinct, confidence}

Every string in a RAW frame is a CONCEPT (a business phrase from the entity cards /
glossaries, or the user's own words) — never an identifier. Grounding
(veda.understanding.frame_grounding) turns each slot into a real table/column/value or a
typed clarify naming that slot; the compiler (veda.understanding.frame_compiler) turns a
fully grounded frame into a QueryIR and SQL through the existing deterministic builders.

Ordering, window and grouping are SEPARATE slots:
    "most recent"            → order {concept: <business date>, dir: desc}
    "last 30 days"           → time.window
    "by city" (a breakdown)  → group_by ["city"]
    "by date" after a superlative ("oldest … by date") → order, not group_by

Nothing here touches the DB or the model: pure data, the JSON schema handed to the
constrained decoder, and a normaliser that clamps anything off-schema.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

FRAME_VERSION = 1

AGGREGATIONS = ("none", "count", "count_distinct", "sum", "avg", "min", "max")
OPS = ("=", "!=", ">", ">=", "<", "<=", "between", "in", "is_null", "is_not_null")
DIRS = ("asc", "desc")
SLOT_NAMES = ("entity", "measure", "aggregation", "filters", "group_by", "order", "limit",
              "time", "distinct")


@dataclass
class FrameFilter:
    concept: str
    op: str = "="
    value: Any = None

    def key(self):
        return (self.concept.strip().lower(), self.op, _hashable(self.value))


@dataclass
class FrameOrder:
    concept: str
    dir: str = "desc"


@dataclass
class FrameTime:
    concept: Optional[str] = None                   # which date ("updated", "created", None = business date)
    window: Optional[Dict[str, Optional[str]]] = None   # {"from": iso|None, "to": iso|None}


@dataclass
class Frame:
    entity: Optional[str] = None
    secondaries: List[str] = field(default_factory=list)
    measure: Optional[str] = None
    aggregation: str = "none"
    filters: List[FrameFilter] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    order: Optional[FrameOrder] = None
    limit: Optional[int] = None
    time: Optional[FrameTime] = None
    distinct: bool = False
    confidence: float = 0.0
    # provenance: slot → "slm" | "producer:<name>" | "slm+producer:<name>" | "session"
    provenance: Dict[str, str] = field(default_factory=dict)
    # slots on which self-consistency samples disagreed (Stage 2.4)
    uncertain: List[str] = field(default_factory=list)
    version: int = FRAME_VERSION
    # ── compound messages (front-door decomposition) ──
    # A frame is ONE intent of a message. These stay at their defaults on the single
    # path; the intent-list extractor fills them.
    part: Optional[str] = None           # the part of the message this frame answers
    kind: Optional[str] = None           # sql | rag | tabular (set by grounding)
    source_hint: Optional[str] = None    # the source whose cards the entity grounded in
    topics: List[str] = field(default_factory=list)   # rag: the topic phrases
    depends_on: Optional[int] = None     # index of the frame whose result this one uses

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Frame":
        return normalise(d)

    def is_empty(self) -> bool:
        return not (self.entity or self.measure or self.filters or self.group_by
                    or self.order or self.limit or self.time)


def _hashable(v):
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    return v


# ── the JSON schema handed to the constrained decoder (Ollama `format`) ──────────────
def json_schema(entity_enum: Optional[List[str]] = None) -> Dict[str, Any]:
    """The frame's JSON schema. `entity_enum`, when given, closes the entity slot over the
    scope's business names (the decoder cannot emit an entity that is not a card) — `null`
    stays legal so "no entity named" remains expressible."""
    ent: Dict[str, Any] = {"type": ["string", "null"]}
    if entity_enum:
        ent = {"anyOf": [{"type": "null"}, {"type": "string", "enum": list(entity_enum)}]}
    scalar = {"type": ["string", "number", "boolean", "null"]}
    value = {"anyOf": [scalar, {"type": "array", "items": scalar, "maxItems": 12}]}
    return {
        "type": "object",
        "properties": {
            "entity": ent,
            "secondaries": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
            "measure": {"type": ["string", "null"]},
            "aggregation": {"type": "string", "enum": list(AGGREGATIONS)},
            "filters": {"type": "array", "maxItems": 6, "items": {
                "type": "object",
                "properties": {"concept": {"type": "string"},
                               "op": {"type": "string", "enum": list(OPS)},
                               "value": value},
                "required": ["concept", "op", "value"]}},
            "group_by": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            "order": {"anyOf": [{"type": "null"}, {
                "type": "object",
                "properties": {"concept": {"type": "string"},
                               "dir": {"type": "string", "enum": list(DIRS)}},
                "required": ["concept", "dir"]}]},
            "limit": {"type": ["integer", "null"]},
            "time": {"anyOf": [{"type": "null"}, {
                "type": "object",
                "properties": {"concept": {"type": ["string", "null"]},
                               "window": {"anyOf": [{"type": "null"}, {
                                   "type": "object",
                                   "properties": {"from": {"type": ["string", "null"]},
                                                  "to": {"type": ["string", "null"]}},
                                   "required": ["from", "to"]}]}},
                "required": ["concept", "window"]}]},
            "distinct": {"type": "boolean"},
            "confidence": {"type": "number"},
        },
        "required": ["entity", "measure", "aggregation", "filters", "group_by", "order",
                     "limit", "time", "distinct", "confidence"],
    }


# ── normaliser: anything → a well-formed Frame (off-schema values clamped, never raised) ─
def _s(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def normalise(obj: Any) -> Frame:
    """Coerce a decoded object into a Frame. Unknown enums fall back to their neutral
    value (aggregation 'none', op '=', dir 'desc'); malformed sub-objects are dropped."""
    if isinstance(obj, Frame):
        return copy.deepcopy(obj)
    d = obj if isinstance(obj, dict) else {}
    agg = str(d.get("aggregation") or "none").strip().lower()
    if agg not in AGGREGATIONS:
        agg = {"distinct_count": "count_distinct", "average": "avg", "mean": "avg",
               "total": "sum", "maximum": "max", "minimum": "min", "list": "none",
               "rank": "none"}.get(agg, "none")
    filters: List[FrameFilter] = []
    for f in (d.get("filters") or []):
        if not isinstance(f, dict) or not _s(f.get("concept")):
            continue
        op = str(f.get("op") or "=").strip().lower()
        op = {"==": "=", "<>": "!=", "not in": "!=", "is null": "is_null",
              "is not null": "is_not_null", "not_null": "is_not_null"}.get(op, op)
        if op not in OPS:
            op = "="
        filters.append(FrameFilter(concept=_s(f["concept"]), op=op, value=f.get("value")))
    order = None
    o = d.get("order")
    if isinstance(o, dict) and _s(o.get("concept")):
        dr = str(o.get("dir") or "desc").strip().lower()
        order = FrameOrder(concept=_s(o["concept"]), dir=dr if dr in DIRS else "desc")
    time = None
    t = d.get("time")
    if isinstance(t, dict) and (_s(t.get("concept")) or isinstance(t.get("window"), dict)):
        w = t.get("window")
        win = None
        if isinstance(w, dict) and (_s(w.get("from")) or _s(w.get("to"))):
            win = {"from": _s(w.get("from")), "to": _s(w.get("to"))}
        time = FrameTime(concept=_s(t.get("concept")), window=win)
        if time.window is None and time.concept is None:
            time = None
    lim = d.get("limit")
    try:
        lim = int(lim) if lim not in (None, "", False) else None
        if lim is not None and lim <= 0:
            lim = None
    except (TypeError, ValueError):
        lim = None
    try:
        conf = max(0.0, min(1.0, float(d.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        conf = 0.0
    return Frame(
        entity=_s(d.get("entity")),
        secondaries=[x for x in (_s(v) for v in (d.get("secondaries") or [])) if x][:4],
        measure=_s(d.get("measure")),
        aggregation=agg,
        filters=filters,
        group_by=[x for x in (_s(v) for v in (d.get("group_by") or [])) if x][:3],
        order=order,
        limit=lim,
        time=time,
        distinct=bool(d.get("distinct")),
        confidence=conf,
        provenance=dict(d.get("provenance") or {}),
        uncertain=list(d.get("uncertain") or []),
        part=_s(d.get("part")),
        kind=(str(d.get("kind")).strip().lower() if str(d.get("kind") or "").strip().lower() in KINDS else None),
        source_hint=_s(d.get("source_hint")),
        topics=[x for x in (_s(v) for v in (d.get("topics") or [])) if x][:6],
        depends_on=_int_or_none(d.get("depends_on")),
    )


def _int_or_none(v) -> Optional[int]:
    try:
        return int(v) if v not in (None, "", False) and not isinstance(v, bool) else None
    except (TypeError, ValueError):
        return None


# ── compound messages: one frame per intent ──────────────────────────────────────────
KINDS = ("sql", "rag", "tabular")
RELATIONS = ("independent", "dependent")
MAX_INTENTS = 5


@dataclass
class Intents:
    """What a MESSAGE means: one frame per independent question in it. `dependent` only
    when a later part uses an earlier part's RESULT ("…and for those, …"); then that
    frame's `depends_on` names the earlier one."""
    intents: List[Frame] = field(default_factory=list)
    relation: str = "independent"

    def __len__(self) -> int:
        return len(self.intents)


def intents_json_schema(entity_enum: Optional[List[str]] = None,
                        doc_enum: Optional[List[str]] = None) -> Dict[str, Any]:
    """{intents: [Frame + part/kind/topics/depends_on], relation} — 1..MAX_INTENTS.
    The entity slot closes over the table concepts AND the document titles in scope, so a
    frame can name a document (that frame is a `rag` frame)."""
    names = list(dict.fromkeys(list(entity_enum or []) + list(doc_enum or [])))
    item = json_schema(entity_enum=names or None)
    item = copy.deepcopy(item)
    item["properties"]["part"] = {"type": "string"}
    item["properties"]["kind"] = {"type": "string", "enum": ["sql", "rag"]}
    item["properties"]["topics"] = {"type": "array", "items": {"type": "string"}, "maxItems": 4}
    item["properties"]["depends_on"] = {"type": ["integer", "null"]}
    # the compound call is budgeted — the frame slots a part does not use may be omitted
    item["required"] = ["part", "kind", "entity", "aggregation"]
    return {
        "type": "object",
        "properties": {
            "intents": {"type": "array", "items": item, "minItems": 1, "maxItems": MAX_INTENTS},
            "relation": {"type": "string", "enum": list(RELATIONS)},
        },
        "required": ["intents", "relation"],
    }


def normalise_intents(obj: Any) -> Intents:
    """Coerce a decoded {intents, relation} into Intents. Never raises; an off-schema
    object yields zero intents (the caller degrades)."""
    d = obj if isinstance(obj, dict) else {}
    raw = d.get("intents")
    if not isinstance(raw, list):
        raw = [d] if d.get("entity") or d.get("part") else []
    frames = [normalise(x) for x in raw if isinstance(x, dict)][:MAX_INTENTS]
    rel = str(d.get("relation") or "independent").strip().lower()
    if rel not in RELATIONS:
        rel = "independent"
    for i, fr in enumerate(frames):
        # a dependency must point BACKWARDS at a real earlier frame
        if fr.depends_on is not None and not (0 <= fr.depends_on < i):
            fr.depends_on = None
    if rel == "dependent" and not any(fr.depends_on is not None for fr in frames):
        # "dependent" with no frame naming its parent: ONLY the second part is wired to the
        # first (conservative — a guessed chain would run later parts on rows they may not
        # be about; they stay independent)
        for i, fr in enumerate(frames[1:], start=1):
            fr.depends_on = i - 1
            break
    if rel == "independent":
        for fr in frames:
            fr.depends_on = None
    return Intents(intents=frames, relation=rel)


def slot_values(fr: Frame) -> Dict[str, Any]:
    """A comparable, case-folded view of each slot — used by self-consistency voting, the
    producer merge and the slot-accuracy eval."""
    def low(x):
        return x.strip().lower() if isinstance(x, str) else x
    return {
        "entity": low(fr.entity),
        "measure": low(fr.measure),
        "aggregation": fr.aggregation,
        "filters": tuple(sorted((low(f.concept), f.op, str(_hashable(f.value)).lower())
                                for f in fr.filters)),
        "group_by": tuple(sorted(low(g) for g in fr.group_by)),
        "order": (low(fr.order.concept), fr.order.dir) if fr.order else None,
        "limit": fr.limit,
        "time": ((low(fr.time.concept), _hashable(fr.time.window)) if fr.time else None),
        "distinct": fr.distinct,
    }
