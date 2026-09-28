"""veda.understanding.continuity — the continuity lane: a chat follow-up applied to the
PREVIOUS turn's query structurally, instead of re-entering the engine as a first turn.

WHY. Every follow-up already arrives with the previous turn's state (conversation_context:
entity_table, filters, group_by, measures, aggregation, order_by, limit, source_id) and the
chat tier's deterministic delta. Re-entering the pipeline as a first turn threw both away and
re-derived the question from the user's fragment — "of those, which are for sale" was judged
by the lexical qualifier gate against words that name nothing on their own
(qualifier_dropped), "break that down by property type" re-anchored and clarified.

WHAT. One entry point, `run_continuity`, hooked at veda/pipeline.py run_query stage 0:

  1. the prior query is RECONSTRUCTED from the context as a GroundedFrame (anchor =
     entity_table, anchor_method CONTEXT) — the frame compiler's own input;
  2. the delta is applied to it by op, grounding ONLY the slot that changed;
  3. the result is compiled by the frame compiler (a complete IR, head `continuity.<op>`)
     and rides the pipeline's fast-path slot through the frame lane — the normal firewall
     (structural checks; no lexical qualifier gate), execution and NL summary.

Never re-routes, never selects a fresh anchor. A slot that does not ground → a typed clarify
in business words (no table / column identifiers). A delta that names something outside the
prior entity (another entity, a value only another table holds) → the lane DECLINES and the
existing path answers, with the reason in the trace.

PRE-APPLIED vs LANE-APPLIED deltas (the double-application rule). The chat tier
(chatbot/nodes.py::context_resolve_node) already mutates its frame for some deltas before it
builds the context:

  * drill_up      — the drill stack is popped and the frame rebuilt from what remains;
  * switch_frame  — the frame moves onto the referenced stack entry;
  * replace / remove (classifier) and deterministic shape deltas — apply_context_delta.

For those it sends `delta.applied = True`: the context ALREADY IS the new state and the lane
compiles it as-is. Every other op (add_filter, remove_filter, change_group, change_measure,
change_order from the rule layer, and `ambiguous`) arrives with `applied = False` and is
applied here, once. Lane ops are written idempotently (a filter already present is not added
twice; group / measure / order are replaced, not appended), so a stale flag cannot double a
predicate.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from veda.understanding.frame import Frame
from veda.understanding.frame_grounding import (
    GFilter, GJoin, GroundedFrame, _bare, _cols, _st, _plain_col, _measure_col, _name_col,
    _col_toks, _content, _edges, column_domain, name_hits, _COUNT_W, _SUMAVG_W, _MINMAX_W)
from veda.understanding.vocabulary import norm_phrase

CONTEXT = "CONTEXT"

# ops the lane applies; switch_frame / drill_up / replace only ever arrive pre-applied
LANE_OPS = ("add_filter", "remove_filter", "change_group", "change_measure", "change_order")
WIRE_OPS = LANE_OPS + ("drill_up", "switch_frame", "replace", "ambiguous", "new_topic", "compare")

_GROUP_TYPES = {"CATEGORY", "FLAG", "TEMPORAL"}
_FILTER_TYPES = {"CATEGORY", "FLAG"}
_MEASURE_TYPES = {"METRIC", "MONETARY"}
_MAX_CANDIDATES = 5

# explain's operator words (veda/business_explain.py::_OP_WORD) → the compiler's operators
_OP_FROM_WORD = {"equals": "=", "=": "=", "not equals": "!=", "!=": "!=",
                 "greater than": ">", ">": ">", "greater than or equal to": ">=", ">=": ">=",
                 "less than": "<", "<": "<", "less than or equal to": "<=", "<=": "<=",
                 "is one of": "IN", "in": "IN", "is": "IS NULL", "is not": "IS NOT NULL"}

# aggregation words → the IR's aggregation names (explain writes "average", the chat "avg")
_AGG_NORM = {"count": "count", "sum": "sum", "total": "sum", "avg": "avg", "average": "avg",
             "mean": "avg", "min": "min", "minimum": "min", "max": "max", "maximum": "max"}


@dataclass
class ContinuityResult:
    kind: str                                   # sql | clarify | decline
    reason: str = ""
    frame_result: Any = None                    # FramePathResult for kind="sql"
    message: Optional[str] = None               # kind="clarify"
    candidates: List[str] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)


class _Clarify(Exception):
    def __init__(self, slot: str, message: str, candidates: Optional[List[str]] = None):
        super().__init__(message)
        self.slot, self.message, self.candidates = slot, message, list(candidates or [])[:_MAX_CANDIDATES]


class _Decline(Exception):
    pass


# ── entry point ──────────────────────────────────────────────────────────────────────
def is_active(conv: Optional[Dict[str, Any]]) -> bool:
    """The lane runs for a chat follow-up that carries remembered state and is not a new
    topic. Everything else (first turns, non-chat callers) never reaches it."""
    if not isinstance(conv, dict) or not conv.get("entity_table"):
        return False
    op = (conv.get("delta") or {}).get("op")
    return op != "new_topic"


def run_continuity(query: str, sm, conv: Dict[str, Any], *, vocab=None,
                   slm: Optional[Callable[..., str]] = None) -> ContinuityResult:
    """Apply this turn's delta to the prior query and compile it. See the module docstring.
    `vocab` / `slm` are injectable for tests; production resolves the scope vocabulary and
    uses slm.call_slm."""
    tr: Dict[str, Any] = {}
    delta = dict(conv.get("delta") or {})
    tr["op"] = delta.get("op")
    tr["applied"] = bool(delta.get("applied"))
    if not delta or not delta.get("op"):
        return _decline("no_delta", tr)
    if delta.get("op") not in WIRE_OPS:
        return _decline(f"unknown_op:{delta.get('op')}", tr)
    if delta["op"] in ("new_topic", "compare"):
        # a comparison carries both sides in its own text; a new topic carries no state
        return _decline(f"op:{delta['op']}", tr)
    if delta["op"] == "switch_frame" and not delta.get("applied"):
        # only exists as a chat-side stack move; unapplied, the context is stale
        return _decline("not_applied:switch_frame", tr)
    if delta["op"] == "replace" and not delta.get("applied"):
        # The classifier said "replace <field> with <value>" but the chat could not apply it
        # (measured 2026-09-27: "only Mumbai" after a per-city count — the prior had no city
        # filter to replace). With a value it is a filter on that field — add_filter replaces
        # a same-column filter, so it is the same edit; without one, the one constrained call
        # places it. Declining here sent the turn to a 5-call agent re-plan that lost the
        # grouping.
        delta = ({**delta, "op": "add_filter"} if delta.get("value") not in (None, "")
                 else {**delta, "op": "ambiguous"})
        tr["op"] = "replace→" + delta["op"]
    # An agent-planned prior is NOT a reason to decline: the context carries the same
    # entity / filters / grouping, so the lane edits it like any other prior. Only when the
    # lane itself declines (prior not reconstructable, other entity, …) does the existing
    # path — frame_path._agent_follow_up — take the turn (measured 2026-09-27: declining on
    # agent_plan routed every follow-up of an agent-planned first turn to a 4–5 call re-plan).
    try:
        if vocab is None:
            from veda.understanding.vocabulary import scope_vocab
            vocab = scope_vocab(sm)
        anchor = _scope_key(sm, conv.get("entity_table"), conv.get("source_id"))
        if anchor is None:
            return _decline("entity_not_in_scope", tr)
        prior = reconstruct(conv, anchor, sm, vocab)
        tr["prior_ir_hash"] = _ir_hash(prior)
        # the delta must be about the prior entity: a message naming a different entity is a
        # topic change the router owns (decline, never re-route here)
        other = _names_other_entity(query, anchor, vocab)
        if other and delta.get("op") not in ("drill_up", "switch_frame"):
            return _decline(f"names_other_entity:{_bare(other)}", tr)
        gf = prior
        op = delta["op"]
        if op == "ambiguous" and not delta.get("applied"):
            # a value the entity already holds is a filter — no model call needed
            # (live 2026-09-27: "only Mumbai" sent as ambiguous; the one call read it as
            # remove_filter)
            det = _value_in_message(query, prior, sm, vocab)
            delta = det if det is not None else _slm_delta(query, prior, sm, vocab, slm, tr)
            if delta is None:
                return _decline("ambiguous_unresolved", tr)
            op = delta["op"]
            tr["op"] = "ambiguous→" + op
        if not delta.get("applied") and op in LANE_OPS:
            gf = apply_delta(prior, delta, query, sm, vocab, tr)
        elif not delta.get("applied") and op in ("drill_up",):
            gf = _drop_filter(prior, None, None, tr)          # the chat could not pop: one level
        head_op = op if op in WIRE_OPS else "ambiguous"
        return _compile(gf, sm, vocab, head_op, conv, tr)
    except _Clarify as c:
        tr["clarify"] = {"slot": c.slot, "candidates": c.candidates}
        return ContinuityResult("clarify", f"unresolved:{c.slot}", message=c.message,
                                candidates=c.candidates, trace=tr)
    except _Decline as d:
        return _decline(str(d), tr)
    except Exception as e:                                     # never fail a turn here
        tr["exception"] = f"{type(e).__name__}: {str(e)[:200]}"
        return _decline("exception", tr)


def _decline(reason: str, tr: Dict[str, Any]) -> ContinuityResult:
    tr["declined"] = reason
    return ContinuityResult("decline", reason, trace=tr)


# ── 1. reconstruct the prior query ───────────────────────────────────────────────────
def _scope_key(sm, table: Optional[str], source_id) -> Optional[str]:
    tabs = (sm or {}).get("tables") or {}
    if not table:
        return None
    if table in tabs:
        return table
    q = f"src{source_id}.{table}" if source_id is not None else None
    if q and q in tabs:
        return q
    return None


def _parents(sm, vocab, anchor) -> List[Tuple[str, str]]:
    """[(fk_column, parent scope table)] — the anchor's DECLARED N:1 / 1:1 edges (the
    relationship graph the frame path joins over); data-inferred edges are not joins."""
    out: List[Tuple[str, str]] = []
    cols = _cols(sm, anchor)
    tabs = (sm or {}).get("tables") or {}
    pre = anchor.split(".", 1)[0] + "." if anchor.startswith("src") and "." in anchor else ""
    for e in _edges(vocab, anchor):
        if (e.get("source_table") == _bare(anchor) and e.get("cardinality") in ("N:1", "1:1")
                and not e.get("polymorphic") and e.get("target_column") == "id"
                and e.get("discovery") in ("declared_fk", None)
                and e.get("source_column") in cols):
            p = pre + str(e.get("target_table"))
            if p in tabs and (e["source_column"], p) not in out:
                out.append((e["source_column"], p))
    return out


def _join(gf: GroundedFrame, sm, vocab, parent: str) -> GJoin:
    for j in gf.joins:
        if j.table == parent:
            return j
    for fk, p in _parents(sm, vocab, gf.anchor):
        if p == parent:
            j = GJoin(table=parent, fk_column=fk)
            gf.joins.append(j)
            return j
    raise _Decline(f"no_join:{_bare(parent)}")


def _split_qualified(gf: GroundedFrame, sm, vocab, column: str):
    """("parent_table", "col") when `column` is remembered qualified ("assets_assettype.name")
    and that table is the anchor or one of its N:1 parents; else (None, column)."""
    if "." not in column:
        return None, column
    t, c = column.rsplit(".", 1)
    for cand in [gf.anchor] + [p for _fk, p in _parents(sm, vocab, gf.anchor)]:
        if _bare(cand) == _bare(t) and c in _cols(sm, cand):
            return cand, c
    return None, column


def _owner_table(gf: GroundedFrame, sm, vocab, column: str) -> Optional[str]:
    """The anchor when it has `column`, else the one N:1 parent that does."""
    if column in _cols(sm, gf.anchor):
        return gf.anchor
    hits = [p for _fk, p in _parents(sm, vocab, gf.anchor) if column in _cols(sm, p)]
    if len(hits) > 1:
        # a parent whose DISPLAY column this is wins ("name" of the asset type)
        disp = [p for p in hits if _name_col(vocab, sm, p) == column]
        hits = disp if len(disp) == 1 else hits
    return hits[0] if len(hits) == 1 else None


def _grounding_for(sm, table, column, op) -> str:
    st = _st(_cols(sm, table).get(column))
    if op in ("IS NULL", "IS NOT NULL"):
        return "null_check"
    if st == "TEMPORAL":
        return "temporal"
    if st in _MEASURE_TYPES and op in (">", ">=", "<", "<=", "=", "!="):
        return "numeric"
    return "context"


def reconstruct(conv: Dict[str, Any], anchor: str, sm, vocab) -> GroundedFrame:
    """The context → a complete GroundedFrame. Any remembered slot that cannot be placed
    on the anchor or one of its N:1 parents makes the lane decline: compiling a prior that
    silently lost a filter would answer a broader question than the conversation holds."""
    gf = GroundedFrame(frame=Frame(), anchor=anchor, anchor_method=CONTEXT,
                       source_id=(str(conv["source_id"]) if conv.get("source_id") is not None
                                  else vocab.source_of.get(anchor)),
                       confidence=1.0)
    for f in conv.get("filters") or []:
        col = str(f.get("column") or "")
        op = _OP_FROM_WORD.get(str(f.get("operator") or "equals").lower().strip())
        if not col or op is None:
            raise _Decline(f"prior_unreconstructable:filter:{col or '?'}")
        tbl = _owner_table(gf, sm, vocab, col)
        if tbl is None:
            raise _Decline(f"prior_unreconstructable:filter:{col}")
        if tbl != anchor:
            _join(gf, sm, vocab, tbl)
        val: Any = f.get("value")
        if op == "IN":
            val = [v.strip(" '\"") for v in str(val).strip("[]()").split(",") if v.strip(" '\"")]
        grounding = _grounding_for(sm, tbl, col, op)
        if grounding == "numeric":
            try:
                val = float(val)
            except (TypeError, ValueError):
                grounding = "context"
        if op in ("IS NULL", "IS NOT NULL"):
            val = None
        gf.filters.append(GFilter(tbl, col, op, val, col.replace("_", " "), grounding,
                                  stype=_st(_cols(sm, tbl).get(col))))
    for g in conv.get("group_by") or []:
        tbl, col = _split_qualified(gf, sm, vocab, str(g))
        tbl = tbl or _owner_table(gf, sm, vocab, col)
        if tbl is None:
            raise _Decline(f"prior_unreconstructable:group_by:{col}")
        if tbl != anchor:
            _join(gf, sm, vocab, tbl)
        gf.group_by.append((tbl, col))
    agg = _AGG_NORM.get(str(conv.get("aggregation") or "").lower().strip())
    mcol = next((m for m in (conv.get("measures") or []) if m in _cols(sm, anchor)), None)
    if agg and agg != "count":
        if not mcol:
            raise _Decline("prior_unreconstructable:measure")
        gf.measure = (agg, mcol)
    elif agg == "count" or gf.group_by:
        gf.measure = ("count", None)
    order_by = [str(o) for o in (conv.get("order_by") or [])]
    if order_by:
        gf.order = _prior_order(gf, sm, vocab, order_by[0])
    try:
        lim = int(conv["limit"]) if conv.get("limit") is not None else None
    except (TypeError, ValueError):
        lim = None
    if gf.measure and not gf.group_by:
        lim, gf.order = None, None             # a single total has no rows to order or cap
    gf.limit = lim
    return gf


def _prior_order(gf: GroundedFrame, sm, vocab, field_: str) -> Optional[Tuple[str, str, str]]:
    """The remembered ORDER BY field → (table, column, dir). The context carries no
    direction; a remembered ranking is a top-N, which the harvest records highest-first.
    A grouped answer's measure (its column, or the aggregate's result alias) orders by
    the aggregate — the compiler maps it to the alias."""
    if gf.measure and field_ in {"count", gf.measure[1] or "", f"{gf.measure[0]}_{gf.measure[1]}"}:
        return (gf.anchor, gf.measure[1] or "count", "desc")
    tbl = _owner_table(gf, sm, vocab, field_)
    if tbl is None:
        return None                               # an alias we cannot place: default order
    if tbl != gf.anchor:
        _join(gf, sm, vocab, tbl)
    return (tbl, field_, "desc")


def _ir_hash(gf: GroundedFrame) -> str:
    from veda.understanding.frame_compiler import to_ir
    try:
        d = to_ir(gf).to_dict()
        d.pop("head", None)
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]
    except Exception:
        return ""


# ── 2. apply the delta ───────────────────────────────────────────────────────────────
def _copy(gf: GroundedFrame) -> GroundedFrame:
    import copy
    return copy.deepcopy(gf)


def apply_delta(prior: GroundedFrame, delta: Dict[str, Any], query: str, sm, vocab,
                tr: Dict[str, Any]) -> GroundedFrame:
    gf = _copy(prior)
    op = delta.get("op")
    concept = (str(delta["concept"]).strip() if delta.get("concept") not in (None, "") else None)
    value = delta.get("value")
    if op == "add_filter":
        return _add_filter(gf, concept, value, query, sm, vocab, tr)
    if op == "remove_filter":
        return _drop_filter(gf, concept, value, tr)
    if op == "change_group":
        return _change_group(gf, concept, query, sm, vocab, tr)
    if op == "change_measure":
        return _change_measure(gf, concept, value, query, sm, vocab, tr)
    if op == "change_order":
        return _change_order(gf, concept, value, query, sm, vocab, tr)
    return gf


def _plural(vocab, table) -> str:
    c = vocab.cards.get(table) or {}
    return str(c.get("plural") or c.get("business_name") or _human(_bare(table)))


def _human(col: str) -> str:
    """A column's business wording: underscores → spaces, a trailing id dropped."""
    w = [x for x in str(col).split("_") if x]
    if len(w) > 1 and w[-1] == "id":
        w = w[:-1]
    return " ".join(w)


def _dimension_names(gf: GroundedFrame, sm, vocab) -> List[str]:
    """What the entity can be broken down by, in business words (≤ 5)."""
    card = vocab.cards.get(gf.anchor) or {}
    cols = _cols(sm, gf.anchor)
    out: List[str] = []
    for c in list(card.get("key_dimensions") or []) + [c for c, m in cols.items() if _st(m) in _GROUP_TYPES]:
        if c in cols and _st(cols[c]) in _GROUP_TYPES and _human(c) not in out:
            out.append(_human(c))
    for _fk, p in _parents(sm, vocab, gf.anchor):
        n = (vocab.cards.get(p) or {}).get("business_name")
        if n and n not in out:
            out.insert(min(len(out), 2), str(n))
    return out[:_MAX_CANDIDATES]


def _measure_names(gf: GroundedFrame, sm, vocab) -> List[str]:
    cols = _cols(sm, gf.anchor)
    card = vocab.cards.get(gf.anchor) or {}
    ms = [m for m in (card.get("key_measures") or []) if m in cols] + \
         [c for c, m in cols.items() if _st(m) in _MEASURE_TYPES]
    return list(dict.fromkeys(_human(m) for m in ms))[:_MAX_CANDIDATES]


# ── add_filter ──
def _sampled_owners(value: str) -> List[Tuple[str, str, Any]]:
    """[(bare table, column, stored value)] — the sampled value store (the value arbiter's
    lookup, query/resolution.typed_value_lookup)."""
    try:
        from query.resolution import typed_value_lookup
        return [(t, c, raw) for (t, c, _st2, raw) in (typed_value_lookup()(str(value).lower()) or [])]
    except Exception:
        return []


def _embedding_owners(value: str, columns: List[Tuple[str, str]]) -> List[Tuple[str, str, Any, float]]:
    """Last resort: the per-value embedding index (ingestion/value_embedder.py,
    entity_value_embeddings) restricted to `columns` [(scope table, column)].
    [(table, column, stored display, cosine)] at cosine ≥ 0.80."""
    try:
        import numpy as np
        from ingestion.value_embedder import load_value_embeddings
        from ingestion import m3_encoder
        nodes, _norms, displays, _cls, mat = load_value_embeddings()
        if not nodes:
            return []
        want = {}
        for t, c in columns:
            want[f"{_bare(t)}.{c}"] = (t, c)
        idx = [i for i, n in enumerate(nodes) if any(str(n).endswith(k) for k in want)]
        if not idx:
            return []
        q = np.asarray(m3_encoder.encode_dense([str(value)]), dtype="float32")[0]
        q = q / (np.linalg.norm(q) or 1.0)
        out = []
        for i in idx:
            cos = float(mat[i] @ q)
            if cos >= 0.80:
                k = next(k for k in want if str(nodes[i]).endswith(k))
                out.append((want[k][0], want[k][1], displays[i], cos))
        return sorted(out, key=lambda r: -r[3])[:3]
    except Exception:
        return []


def _live_domain(vocab, sm, table, column) -> List[str]:
    return column_domain(vocab, sm, table, column, live=True)


def _filter_columns(gf: GroundedFrame, sm, vocab) -> List[Tuple[str, str]]:
    """Where a new value may live: the anchor's categorical columns, then each N:1
    parent's display / categorical columns (the compiler joins parents N:1 only)."""
    out = [(gf.anchor, c) for c, m in _cols(sm, gf.anchor).items() if _st(m) in _FILTER_TYPES]
    for _fk, p in _parents(sm, vocab, gf.anchor):
        out += [(p, c) for c, m in _cols(sm, p).items() if _st(m) in _FILTER_TYPES]
    return out


def _concept_column(gf: GroundedFrame, concept: Optional[str], sm, vocab,
                    kinds=None) -> Optional[Tuple[str, str]]:
    """The column a delta's concept names — a raw column of the anchor/parents (the chat's
    rule layer names result columns), else the frame grounding's own column matcher."""
    if not concept:
        return None
    c = concept.strip()
    for t in [gf.anchor] + [p for _fk, p in _parents(sm, vocab, gf.anchor)]:
        meta = _cols(sm, t).get(c)
        if meta is not None and (not kinds or _st(meta) in kinds):
            return (t, c)
    col, _ = _plain_col(vocab, sm, gf.anchor, c, kinds=kinds)
    return (gf.anchor, col) if col else None


def ground_new_value(gf: GroundedFrame, concept: Optional[str], value, query: str, sm, vocab
                     ) -> Tuple[Optional[Tuple[str, str, Any]], str, List[str]]:
    """((table, column, stored value) | None, method, candidates). The column is the
    delta's named one when it has one; otherwise the value's owner among the anchor's (and
    its N:1 parents') columns. Order: domain / glossary → sampled store → live domain of
    the named column → embedding match."""
    from veda.understanding.frame_grounding import ground_value
    sval = str(value).strip()
    named = _concept_column(gf, concept, sm, vocab)
    cols = [named] if named else _filter_columns(gf, sm, vocab)
    # 1. the sampled / glossary domain the semantic model already holds (no DB)
    hits = []
    for t, c in cols:
        dom = {d.lower(): d for d in column_domain(vocab, sm, t, c, live=False)}
        if sval.lower() in dom:
            hits.append((t, c, dom[sval.lower()], "domain"))
            continue
        gl = vocab.value_glossary.get(f"{t}.{c}") or {}
        sn = norm_phrase(sval)
        for v, phrases in gl.items():
            if sn and sn in {norm_phrase(p) for p in phrases or []}:
                hits.append((t, c, v, "glossary"))
                break
    # 2. the sampled value store
    if not hits:
        allowed = {(_bare(t), c): (t, c) for t, c in cols}
        for bt, c, raw in _sampled_owners(sval):
            if (bt, c) in allowed:
                t, _c = allowed[(bt, c)]
                if not any(h[0] == t and h[1] == c for h in hits):
                    hits.append((t, c, raw, "sampled"))
    # 3. the named column's live domain (one probe, only when the column is named)
    if not hits and named:
        gv, _note, how = ground_value(vocab, sm, named[0], named[1], sval, query)
        if gv is not None:
            hits.append((named[0], named[1], gv, how))
    # 4. embedding match
    if not hits:
        for t, c, disp, _cos in _embedding_owners(sval, cols):
            hits.append((t, c, disp, "embedding"))
            break
    if len(hits) == 1 or (hits and len({(h[0], h[1]) for h in hits}) == 1):
        t, c, v, how = hits[0]
        return (t, c, v), how, []
    if hits:
        return None, "ambiguous", [f"{_human(h[1])} {h[2]}" for h in hits]
    cands: List[str] = []
    if named:
        dom = column_domain(vocab, sm, named[0], named[1], live=False) or \
            (_live_domain(vocab, sm, named[0], named[1]) or [])
        cands = difflib.get_close_matches(sval, dom, n=_MAX_CANDIDATES, cutoff=0.5) or dom[:_MAX_CANDIDATES]
    return None, "unmapped", cands


def _value_in_message(query: str, gf: GroundedFrame, sm, vocab) -> Optional[Dict[str, Any]]:
    """{op: add_filter, concept, value} when exactly ONE word span of the message (1–3 words,
    longest first) is a stored value of one of the entity's filterable columns — found in
    the semantic model's domain, the value glossary or the sampled store, never a live probe
    or a model call. Anything else (no span, or spans on different columns) → None, and the
    one constrained call places the message. The data decides, not a word list."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9&'\-]*", query or "")
    cols = _filter_columns(gf, sm, vocab)
    if not words or not cols:
        return None
    allowed = {(_bare(t), c): (t, c) for t, c in cols}
    found: Dict[Tuple[str, str], Tuple[str, Any]] = {}
    used: set = set()
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            if any(k in used for k in range(i, i + n)):
                continue
            span = " ".join(words[i:i + n])
            if len(span) < 3:
                continue
            hit = None
            for t, c in cols:
                dom = {d.lower(): d for d in column_domain(vocab, sm, t, c, live=False)}
                if span.lower() in dom:
                    hit = ((t, c), dom[span.lower()])
                    break
            if hit is None:
                for bt, c, raw in _sampled_owners(span):
                    if (bt, c) in allowed:
                        hit = (allowed[(bt, c)], raw)
                        break
            if hit is not None:
                found.setdefault(hit[0], hit[1])
                used.update(range(i, i + n))
    if len(found) != 1:
        return None
    (t, c), v = next(iter(found.items()))
    return {"op": "add_filter", "concept": c, "value": v, "applied": False,
            "method": "message_value"}


def _value_elsewhere(value, anchor: str, gf: GroundedFrame, sm, vocab) -> Optional[str]:
    """A table outside the anchor and its parents that holds `value` (sampled store)."""
    local = {_bare(gf.anchor)} | {_bare(p) for _fk, p in _parents(sm, vocab, anchor)}
    for bt, _c, _raw in _sampled_owners(str(value)):
        if bt not in local:
            return bt
    return None


def _add_filter(gf, concept, value, query, sm, vocab, tr) -> GroundedFrame:
    if value in (None, "", []):
        raise _Clarify("filter", f"Which value should I narrow the {_plural(vocab, gf.anchor)} to?",
                       _dimension_names(gf, sm, vocab))
    hit, how, cands = ground_new_value(gf, concept, value, query, sm, vocab)
    if hit is None:
        if how == "unmapped" and _value_elsewhere(value, gf.anchor, gf, sm, vocab):
            # the value belongs to some other entity: that is a different question
            raise _Decline("value_outside_entity")
        what = _plural(vocab, gf.anchor)
        if how == "ambiguous":
            raise _Clarify("filter", f"'{value}' matches more than one field of the {what}. "
                                     f"Which did you mean: {', '.join(cands)}?", cands)
        named = _concept_column(gf, concept, sm, vocab)
        field_words = f"{_human(named[1])} " if named else ""
        msg = f"I couldn't find the {field_words}'{value}' among the {what}."
        if cands:
            msg += f" Did you mean one of: {', '.join(map(str, cands))}?"
        raise _Clarify("filter", msg, cands)
    t, c, v = hit
    if t != gf.anchor:
        _join(gf, sm, vocab, t)
    if any(f.table == t and f.column == c and f.op == "=" and str(f.value).lower() == str(v).lower()
           for f in gf.filters):
        tr.update(slot="filter", grounded_to=f"{_bare(t)}.{c}", method="already_applied")
        return gf
    # one value per column: a new value on a column already narrowed REPLACES it
    # ("only Pune" after Mumbai) — the conjunction of two values of one column is empty
    replaced = [f for f in gf.filters if f.table == t and f.column == c and f.op in ("=", "IN")]
    gf.filters = [f for f in gf.filters if f not in replaced]
    gf.filters.append(GFilter(t, c, "=", v, concept or _human(c), "domain" if how != "glossary" else "glossary",
                              stype=_st(_cols(sm, t).get(c))))
    tr.update(slot="filter", grounded_to=f"{_bare(t)}.{c}", method=how,
              **({"replaced": [str(f.value) for f in replaced]} if replaced else {}))
    return gf


# ── remove_filter / drill_up ──
def _drop_filter(gf, concept, value, tr) -> GroundedFrame:
    gf = _copy(gf)
    if not gf.filters:
        raise _Clarify("filter", "There is no narrowing left to remove from this answer.", [])
    target = None
    for f in reversed(gf.filters):
        if (concept and (f.column == concept or _human(f.column) == norm_phrase(concept)
                         or norm_phrase(concept) in norm_phrase(_human(f.column)))) \
                or (value is not None and str(f.value).lower() == str(value).lower()):
            target = f
            break
    target = target or gf.filters[-1]
    gf.filters = [f for f in gf.filters if f is not target]
    tr.update(slot="filter", grounded_to=f"{_bare(target.table)}.{target.column}", method="removed")
    return gf


# ── change_group ──
def _concept_variants(concept: str, anchor: str, vocab) -> List[str]:
    """'property type' on an entity whose aliases include 'asset' also reads 'asset type':
    the entity's own name inside a concept is interchangeable with its other names."""
    card = vocab.cards.get(anchor) or {}
    names = [norm_phrase(n) for n in [card.get("business_name"), card.get("plural"),
                                      *(card.get("aliases") or [])] if n]
    cn = norm_phrase(concept)
    out = [concept]
    for n in names:
        if n and f" {n} " in f" {cn} ":
            for m in names:
                if m != n and len(m.split()) == 1:
                    v = f" {cn} ".replace(f" {n} ", f" {m} ").strip()
                    if v not in out:
                        out.append(v)
            rest = f" {cn} ".replace(f" {n} ", " ").strip()
            if rest and rest not in out:
                out.append(rest)
    return out


def _exact_column(gf: GroundedFrame, concept: str, sm, vocab, kinds) -> Optional[Tuple[str, str]]:
    c = concept.strip()
    for t in [gf.anchor] + [p for _fk, p in _parents(sm, vocab, gf.anchor)]:
        meta = _cols(sm, t).get(c)
        if meta is not None and (not kinds or _st(meta) in kinds):
            return (t, c)
    return None


def ground_dimension(gf: GroundedFrame, concept: str, sm, vocab) -> Optional[Tuple[str, str]]:
    """A grouping concept → (table, column): a CATEGORY / FLAG / TEMPORAL column of the
    anchor, or an N:1 parent's display column (reached by the parent's name, or by the
    anchor's FK column that names it).

    The entity's own name inside the concept is not a column word: "property type" on
    properties is the asset type (a parent), never corner_property — so the fuzzy column
    matcher only ever sees the concept with the entity's names taken out."""
    hit = _exact_column(gf, concept, sm, vocab, _GROUP_TYPES)
    if hit:
        return hit
    parents = _parents(sm, vocab, gf.anchor)
    variants = _concept_variants(concept, gf.anchor, vocab)
    for v in variants:
        vt = set(_content(v))
        # a parent named by the concept ('asset type', its plural or an alias)
        for fk, p in parents:
            pc = vocab.cards.get(p) or {}
            names = {norm_phrase(n) for n in [pc.get("business_name"), pc.get("plural"),
                                               *(pc.get("aliases") or [])] if n}
            if norm_phrase(v) in names:
                nc = _name_col(vocab, sm, p)
                if nc:
                    return (p, nc)
        # the anchor's FK column the concept names ('asset type' → asset_type_id → parent)
        for fk, p in parents:
            if vt and vt == (_col_toks(fk) - {"id"}):
                nc = _name_col(vocab, sm, p)
                if nc:
                    return (p, nc)
    own = _entity_words(gf.anchor, vocab)
    for v in variants:
        if set(norm_phrase(v).split()) & own:
            continue
        col, _ = _plain_col(vocab, sm, gf.anchor, v, kinds=_GROUP_TYPES)
        if col:
            return (gf.anchor, col)
    return None


def _entity_words(anchor: str, vocab) -> set:
    card = vocab.cards.get(anchor) or {}
    out = set()
    for n in [card.get("business_name"), card.get("plural"), *(card.get("aliases") or [])]:
        if n and len(norm_phrase(n).split()) == 1:
            out.add(norm_phrase(n))
    return out


def _change_group(gf, concept, query, sm, vocab, tr) -> GroundedFrame:
    if not concept:
        raise _Clarify("group_by", f"What should I break the {_plural(vocab, gf.anchor)} down by?",
                       _dimension_names(gf, sm, vocab))
    hit = ground_dimension(gf, concept, sm, vocab)
    if hit is None:
        raise _Clarify("group_by", (f"I couldn't find '{concept}' to break the "
                                    f"{_plural(vocab, gf.anchor)} down by. You can break them "
                                    f"down by: {', '.join(_dimension_names(gf, sm, vocab))}."),
                       _dimension_names(gf, sm, vocab))
    t, c = hit
    if t != gf.anchor:
        _join(gf, sm, vocab, t)
    gf.group_by = [(t, c)]
    if gf.measure is None:
        gf.measure = ("count", None)
    # the previous ranking / page belonged to the previous grouping
    gf.order, gf.limit = None, None
    _prune_joins(gf)
    tr.update(slot="group_by", grounded_to=f"{_bare(t)}.{c}",
              method="parent_display" if t != gf.anchor else "column")
    return gf


def _prune_joins(gf: GroundedFrame) -> None:
    used = {f.table for f in gf.filters} | {t for t, _c in gf.group_by} | \
        ({gf.order[0]} if gf.order else set())
    gf.joins = [j for j in gf.joins if j.table in used]


# ── change_measure ──
def _strip_agg_words(text: str) -> str:
    t = " " + str(text or "").lower() + " "
    for rx in [_COUNT_W, _MINMAX_W, *_SUMAVG_W.values()]:
        t = rx.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def _agg_of(value, query: str) -> Optional[str]:
    a = _AGG_NORM.get(str(value or "").lower().strip())
    if a:
        return a
    ql = str(query or "").lower()
    for k, rx in _SUMAVG_W.items():
        if rx.search(ql):
            return k
    if _COUNT_W.search(ql):
        return "count"
    return None


def _glossary_measures(text: str, vocab, sm) -> List[Tuple[int, str, str]]:
    """[(phrase length, scope table, column)] for every measure-glossary phrase or measure
    column name the text contains — longest first."""
    tn = f" {norm_phrase(text)} "
    out = []
    for key, ent in (vocab.measure_glossary or {}).items():
        t, _, c = key.rpartition(".")
        for ph in list(ent.get("phrases") or []) + [c.replace("_", " ")]:
            pn = norm_phrase(ph)
            if pn and f" {pn} " in tn:
                out.append((len(pn.split()), t, c))
    return sorted(set(out), key=lambda r: -r[0])


def _change_measure(gf, concept, value, query, sm, vocab, tr) -> GroundedFrame:
    agg = _agg_of(value, query)
    what = _plural(vocab, gf.anchor)
    if agg is None:
        raise _Clarify("measure", f"How should I summarise the {what}: a count, a total or an average?",
                       ["count", "total", "average"])
    if agg == "count":
        gf.measure = ("count", None)
        gf.distinct = False
        tr.update(slot="measure", grounded_to="count", method="aggregation")
        return _shape_after_measure(gf)
    col = None
    if concept:
        hit = _concept_column(gf, concept, sm, vocab, kinds=_MEASURE_TYPES)
        col = hit[1] if hit and hit[0] == gf.anchor else None
        if col is None:
            col, _ = _measure_col(vocab, sm, gf.anchor, concept)
    phrases = _glossary_measures(_strip_agg_words(query), vocab, sm) if col is None else []
    if col is None and phrases:
        best = phrases[0][0]
        # among equally long matches: the column the phrase names outright, then an
        # entity's KEY measure, then the shorter (more specific) column name
        top = sorted({(t, c) for n, t, c in phrases if n == best},
                     key=lambda tc: (norm_phrase(tc[1].replace("_", " ")) not in f" {norm_phrase(query)} ",
                                     tc[1] not in ((vocab.cards.get(tc[0]) or {}).get("key_measures") or []),
                                     len(tc[1].split("_")), tc))
        mine = [c for t, c in top if t == gf.anchor]
        if len(set(mine)) == 1:
            col = mine[0]
        elif not mine:
            # the measure the user named lives on another entity: say which, in business words
            others = list(dict.fromkeys(f"{_human(c)} of {_plural(vocab, t)}" for t, c in top))[:3]
            raise _Clarify("measure", (f"The {what} themselves don't record that — it is recorded as "
                                       f"the {' or the '.join(others)}. Ask about those directly, or "
                                       f"summarise the {what} by: "
                                       f"{', '.join(_measure_names(gf, sm, vocab)[:3]) or 'count'}."),
                           (others + _measure_names(gf, sm, vocab))[:_MAX_CANDIDATES])
    if col is None:
        raise _Clarify("measure", (f"Which amount should I {'total' if agg == 'sum' else 'average' if agg == 'avg' else agg} "
                                   f"for the {what}? Options: {', '.join(_measure_names(gf, sm, vocab)) or 'none'}."),
                       _measure_names(gf, sm, vocab))
    gf.measure = (agg, col)
    gf.distinct = False
    tr.update(slot="measure", grounded_to=f"{agg}({_bare(gf.anchor)}.{col})", method="measure")
    return _shape_after_measure(gf)


def _shape_after_measure(gf: GroundedFrame) -> GroundedFrame:
    if not gf.group_by:
        gf.order, gf.limit, gf.projection = None, None, []
    elif gf.order and gf.order[1] not in {c for _t, c in gf.group_by}:
        gf.order = None                        # the old measure's ranking
    _prune_joins(gf)
    return gf


# ── change_order ──
def _change_order(gf, concept, value, query, sm, vocab, tr) -> GroundedFrame:
    from veda.understanding.producers import p_ranking
    rk = p_ranking(query) or {}
    try:
        n = int(value) if value is not None else rk.get("limit")
    except (TypeError, ValueError):
        n = rk.get("limit")
    direction = ((rk.get("order") or {}).get("dir") or "desc").lower()
    what = _plural(vocab, gf.anchor)
    target = None
    if concept:
        g = next(((t, c) for t, c in gf.group_by if c == concept or _human(c) == norm_phrase(concept)), None)
        if g:
            target = g
        elif gf.measure and concept in ("count", gf.measure[1] or "", f"{gf.measure[0]}_{gf.measure[1]}"):
            target = (gf.anchor, gf.measure[1] or "count")
        else:
            mc = _concept_column(gf, concept, sm, vocab, kinds=_MEASURE_TYPES)
            target = mc if mc and mc[0] == gf.anchor else None
            if target and gf.group_by:
                # ranking groups by a measure they don't aggregate is not expressible
                gf.measure = (gf.measure[0] if gf.measure and gf.measure[0] != "count" else "sum", target[1])
    if target is None:
        if gf.measure and gf.group_by:
            target = (gf.anchor, gf.measure[1] or "count")
        elif gf.measure and not gf.group_by:
            raise _Clarify("order", (f"This answer is a single figure for the {what}, so there is "
                                     f"nothing to rank. Break it down first — for example by: "
                                     f"{', '.join(_dimension_names(gf, sm, vocab))}."),
                           _dimension_names(gf, sm, vocab))
        else:
            mc, cands = _measure_col(vocab, sm, gf.anchor, "@measure")
            if not mc:
                raise _Clarify("order", f"Rank the {what} by what? Options: "
                                        f"{', '.join(_measure_names(gf, sm, vocab)) or 'none'}.",
                               _measure_names(gf, sm, vocab))
            target = (gf.anchor, mc)
    gf.order = (target[0], target[1], direction)
    gf.limit = n if n else gf.limit
    on_measure = bool(gf.measure) and target[1] in ("count", gf.measure[1])
    tr.update(slot="order", method="ranking",
              grounded_to=(f"{gf.measure[0]}({_bare(target[0])}.{gf.measure[1]})" if on_measure and gf.measure[1]
                           else "count" if on_measure else f"{_bare(target[0])}.{target[1]}"),
              limit=gf.limit, direction=direction)
    return gf


# ── ambiguous: ONE constrained SLM call for the op ───────────────────────────────────
def _compact(gf: GroundedFrame, vocab) -> Dict[str, Any]:
    return {"entity": _plural(vocab, gf.anchor),
            "filters": [f"{_human(f.column)} {f.op} {f.value}" for f in gf.filters],
            "group_by": [_human(c) for _t, c in gf.group_by],
            "measure": (f"{gf.measure[0]} of {_human(gf.measure[1])}" if gf.measure and gf.measure[1]
                        else (gf.measure[0] if gf.measure else "list")),
            "order": (f"{_human(gf.order[1])} {gf.order[2]}" if gf.order else None),
            "limit": gf.limit}


def delta_schema() -> Dict[str, Any]:
    return {"type": "object",
            "properties": {"op": {"type": "string", "enum": list(LANE_OPS) + ["none"]},
                           "slot": {"type": ["string", "null"]},
                           "concept": {"type": ["string", "null"]},
                           "value": {"type": ["string", "number", "null"]}},
            "required": ["op", "concept", "value"]}


_SYSTEM = ("You map ONE follow-up message onto ONE edit of the previous query. Reply with JSON "
           "only. op: add_filter (narrow to a value; value = the value as the user wrote it, "
           "concept = the field if named), remove_filter (drop a narrowing), change_group "
           "(break down by something else; concept = what to group by), change_measure (count / "
           "total / average of something; value = count|sum|avg|min|max, concept = the amount), "
           "change_order (top / bottom N; value = N, concept = what to rank by), none (the message "
           "is not an edit of the previous query). Use the user's own words for concept and value.")


def _slm_delta(query: str, prior: GroundedFrame, sm, vocab, slm, tr) -> Optional[Dict[str, Any]]:
    user = json.dumps({"previous_query": _compact(prior, vocab),
                       "can_group_by": _dimension_names(prior, sm, vocab),
                       "measures": _measure_names(prior, sm, vocab),
                       "message": query}, ensure_ascii=False)
    try:
        if slm is None:
            from slm import call_slm as slm
        import config
        raw = slm(user, system=_SYSTEM, purpose="continuity_delta", temperature=0.0, seed=7,
                  num_predict=80, timeout=int(getattr(config, "CONTINUITY_SLM_TIMEOUT_S", 30)),
                  json_schema=delta_schema())
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        tr["slm"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        return None
    if not isinstance(obj, dict) or obj.get("op") not in LANE_OPS:
        tr["slm"] = {"op": (obj or {}).get("op") if isinstance(obj, dict) else None}
        return None
    tr["slm"] = {"op": obj.get("op"), "concept": obj.get("concept"), "value": obj.get("value")}
    return {"op": obj["op"], "slot": obj.get("slot"), "concept": obj.get("concept"),
            "value": obj.get("value"), "confidence": 0.6, "applied": False}


# ── scope ─────────────────────────────────────────────────────────────────────────────
def _names_other_entity(query: str, anchor: str, vocab) -> Optional[str]:
    """An entity the message names by its business name / plural (a strong hit) that is
    neither the anchor nor one of its parents' — the delta is about something else."""
    try:
        hits = name_hits(vocab, query)
    except Exception:
        return None
    parents = set((vocab.cards.get(anchor) or {}).get("parent_entities") or [])
    named = [h["table"] for h in hits if h.get("strong") and not h.get("modifier")]
    if any(t == anchor for t in named):
        return None
    for t in named:
        if t != anchor and _bare(t) not in {_bare(p) for p in parents}:
            return t
    return None


# ── 3. compile ────────────────────────────────────────────────────────────────────────
def _compile(gf: GroundedFrame, sm, vocab, op: str, conv, tr) -> ContinuityResult:
    from veda.understanding.frame_compiler import compile_frame, Declined
    from veda.understanding.frame_path import FramePathResult
    try:
        from veda_core.context import try_current
        ctx = try_current()
        scope = [str(s) for s in (ctx.source_ids or (ctx.source_id,))] if ctx else []
    except Exception:
        scope = []
    if not gf.measure and not gf.group_by and not gf.projection:
        gf.projection = _list_projection(gf, sm, vocab)
    comp = compile_frame(gf, sm, source_scope=scope)
    if isinstance(comp, Declined):
        tr["compile"] = {"declined": comp.slot}
        raise _Clarify(comp.slot.split(":")[0], (f"I can't express that change on the "
                                                 f"{_plural(vocab, gf.anchor)} as asked. Could you "
                                                 f"rephrase which part should change?"), [])
    comp.ir.head = f"continuity.{op}"
    comp.ir.grounding_method["anchor"] = CONTEXT
    tr["sql"] = comp.sql
    tr["head"] = comp.ir.head
    tr["ir_hash"] = _ir_hash(gf)
    res = FramePathResult("sql", f"continuity:{op}", sql=comp.sql, ir=comp.ir, anchor=gf.anchor,
                          tables=comp.tables, columns=comp.columns, frame=gf.frame, grounded=gf,
                          notes=list(gf.notes), trace={"continuity": dict(tr)},
                          source_id=(str(gf.source_id) if gf.source_id is not None else None))
    return ContinuityResult("sql", f"compiled:{op}", frame_result=res, trace=tr)


def _list_projection(gf: GroundedFrame, sm, vocab) -> List[Tuple[str, str]]:
    """What a list answer shows — the frame grounding's own projection rule."""
    cols = _cols(sm, gf.anchor)
    card = vocab.cards.get(gf.anchor) or {}
    proj: List[Tuple[str, str]] = []

    def _add(t, c):
        if c and (t, c) not in proj:
            proj.append((t, c))
    _add(gf.anchor, "id" if "id" in cols else None)
    for k in ("display_column", "business_date_column", "lifecycle_column"):
        _add(gf.anchor, card.get(k) if card.get(k) in cols else None)
    for m in (card.get("key_measures") or [])[:2]:
        _add(gf.anchor, m if m in cols else None)
    for f in gf.filters:
        _add(f.table, f.column)
    if gf.order:
        _add(gf.order[0], gf.order[1])
    return proj
