"""chatbot.memory.context — the typed context handed to VEDA Core alongside the user's
own, unmodified words.

WHY THIS EXISTS. The conversation layer has always held structured state (QueryFrame),
and the engine has always taken a single natural-language string. So the last step before
the boundary flattened the state into English and glued it onto the user's message:

    user typed    : only the debit ones
    engine received: only the debit ones (for Single Financial Transactions (accounts_generalledger))

`Single Financial Transactions` is the engine's own display label for
`accounts_generalledger`, carried in the frame as `entity_display`. Downstream,
`veda/validation.py::qualifier_completeness` gates on "every content token THE USER
NAMED", and had no way to tell which tokens the user named — so it refused on `financial`
(which substring-matches the real column `financial_year_id`) even though the user's own
word, `debit`, is a real value with 476 rows behind it.

That is not a wording bug, and frame.py's own comments record why no wording can fix it:
naming the shape made the engine spend 54s asking whether "measuring" was a column;
naming the entity causes the refusal above; naming NEITHER made a bare "go back"
re-resolve to a different, similarly-named table. Three constraints, no string satisfies
all of them — because a string is the wrong carrier.

WHAT IS AND IS NOT IN HERE.

  · `user_message` is the user's text, byte-for-byte. Nothing in this module edits it.
  · Filters travel as VALUES ONLY. A frame's `field` is the engine's humanised LABEL
    ("Location"), and the table has no such column (it is `city_name`) — measured
    2026-09-22, echoing either the label OR the real column name produced a WRONG table,
    while the bare value produced the right one. The engine is the side that knows the
    schema; this layer does not pretend to.
  · `entity_display` is NEVER included. It is presentation metadata and is precisely what
    contaminated the query.

This is a VIEW over QueryFrame, not a second state. It is built after
apply_context_delta, so it is the canonical state for this turn — never the old state and
the new one together.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any, Dict, List, Optional

# Bounds for what crosses the process boundary. The engine must not be handed an
# unbounded structure because a conversation grew long.
_MAX_FILTER_VALUES = 20
_MAX_LIST = 20
_MAX_VALUE_LEN = 200


def _clean_strs(values: Any, cap: int = _MAX_LIST) -> List[str]:
    out: List[str] = []
    for v in (values or []):
        if v is None:
            continue
        s = str(v).strip()
        if s and len(s) <= _MAX_VALUE_LEN and s not in out:
            out.append(s)
        if len(out) >= cap:
            break
    return out


@dataclass(frozen=True)
class ConversationContext:
    """What VEDA Core genuinely needs to know about the conversation — and nothing else."""

    user_message: str                                   # immutable, exactly as typed
    entity_table: Optional[str] = None                  # raw table name, never the label
    source_id: Optional[int] = None
    filters: List[Dict[str, Any]] = _dc_field(default_factory=list)   # {column, operator, value}
    filter_values: List[str] = _dc_field(default_factory=list)        # values only, for display
    group_by: List[str] = _dc_field(default_factory=list)
    measures: List[str] = _dc_field(default_factory=list)
    order_by: List[str] = _dc_field(default_factory=list)
    limit: Optional[int] = None
    aggregation: str = ""          # which aggregate the previous turn computed
    route: str = ""
    drill_depth: int = 0
    # the planner agent's remembered plan (draft form), its tool log and the question it
    # answered — only when the previous turn was agent-planned (veda/agent/)
    agent_plan: Optional[Dict[str, Any]] = None
    agent_log: List[Dict[str, Any]] = _dc_field(default_factory=list)
    agent_question: str = ""
    # THIS turn's delta on the previous query, as the chat tier decided it (wire_delta) —
    # the engine's continuity lane applies it structurally (veda/understanding/continuity.py)
    delta: Optional[Dict[str, Any]] = None

    # ── construction ────────────────────────────────────────────────────────────────
    @classmethod
    def from_frame(cls, frame: Optional[Dict[str, Any]], user_message: str,
                   *, carry_state: bool = True,
                   delta: Optional[Dict[str, Any]] = None) -> "ConversationContext":
        """Build from the frame AS IT STANDS AFTER the delta was applied.

        `carry_state=False` is the new-topic case: the message is self-contained, so no
        remembered state travels with it — the same rule the previous rendering used when
        it returned the message unchanged for `new_topic`/`ambiguous`.
        """
        if not frame or not carry_state:
            return cls(user_message=user_message)

        raw_filters = frame.get("filters") or []
        values = _clean_strs(
            [f.get("value") for f in raw_filters
             if isinstance(f, dict) and f.get("value") is not None],
            cap=_MAX_FILTER_VALUES,
        )
        # STRUCTURED filters — only those that know their own COLUMN. A filter harvested
        # before `column` existed (the frame lives 7 days, so those are still in flight)
        # carries only the humanised label, and a label is not a column: it would have to
        # be re-grounded from the value alone, which is exactly the guess that put an
        # is_gated filter onto all_day_access. Such a filter is dropped rather than guessed.
        filters: List[Dict[str, Any]] = []
        for f in raw_filters:
            if not isinstance(f, dict):
                continue
            col, val = f.get("column"), f.get("value")
            if not col or val is None or len(filters) >= _MAX_FILTER_VALUES:
                continue
            filters.append({"column": str(col)[:200],
                            "operator": str(f.get("operator") or "equals")[:32],
                            "value": str(val)[:_MAX_VALUE_LEN]})
        # The frame writes orderings as {"field": ..., "desc": ...} (harvest_frame reads
        # analytics["orderings"]); "column" was this module's own guess at the key and
        # silently produced an empty list for every real frame.
        order_by = _clean_strs(
            [(o.get("field") or o.get("column") if isinstance(o, dict) else o)
             for o in (frame.get("order_by") or [])]
        )
        limit = frame.get("limit")
        try:
            limit = int(limit) if limit is not None else None
        except (TypeError, ValueError):
            limit = None
        src = frame.get("source_id")
        try:
            src = int(src) if src is not None else None
        except (TypeError, ValueError):
            src = None

        _ag = None
        try:
            _stack = list(frame.get("stack") or [])
            _cur = frame.get("cursor", -1)
            _top = (_stack[_cur] if isinstance(_cur, int) and -len(_stack) <= _cur < len(_stack)
                    else (_stack[-1] if _stack else None))
            _ag = (_top or {}).get("agent") if isinstance(_top, dict) else None
        except Exception:
            _ag = None
        return cls(
            delta=dict(delta) if delta else None,
            agent_plan=(_ag or {}).get("plan") if isinstance(_ag, dict) else None,
            agent_log=list((_ag or {}).get("log") or []) if isinstance(_ag, dict) else [],
            agent_question=str((_ag or {}).get("question") or "") if isinstance(_ag, dict) else "",
            user_message=user_message,
            entity_table=(frame.get("entity") or None),
            source_id=src,
            filters=filters,
            filter_values=values,
            group_by=_clean_strs(frame.get("group_by")),
            measures=_clean_strs(frame.get("measures")),
            order_by=order_by,
            limit=limit,
            aggregation=str(frame.get("aggregation") or ""),
            route=str(frame.get("route") or ""),
            drill_depth=len(frame.get("drill_path") or []),
        )

    # ── transport ───────────────────────────────────────────────────────────────────
    def to_payload(self) -> Dict[str, Any]:
        """JSON-safe dict for `flags["conversation_context"]`.

        Empty collections and None are omitted: an absent key and an empty one mean the
        same thing to the engine, and a smaller payload is a smaller thing to get wrong.
        """
        payload: Dict[str, Any] = {"user_message": self.user_message}
        if self.entity_table:
            payload["entity_table"] = self.entity_table
        if self.source_id is not None:
            payload["source_id"] = self.source_id
        if self.filters:
            payload["filters"] = [dict(f) for f in self.filters]
        for key, val in (("filter_values", self.filter_values),
                         ("group_by", self.group_by),
                         ("measures", self.measures),
                         ("order_by", self.order_by)):
            if val:
                payload[key] = list(val)
        if self.limit is not None:
            payload["limit"] = self.limit
        if self.aggregation:
            payload["aggregation"] = self.aggregation
        if self.route:
            payload["route"] = self.route
        if self.drill_depth:
            payload["drill_depth"] = self.drill_depth
        if self.delta and self.entity_table:
            payload["delta"] = dict(self.delta)
        if self.agent_plan:
            payload["agent_plan"] = dict(self.agent_plan)
            payload["agent_log"] = list(self.agent_log)
            payload["agent_question"] = self.agent_question
        return payload

    def is_empty(self) -> bool:
        """True when nothing but the message travels — i.e. there is no context to send."""
        return not (self.entity_table or self.filters or self.filter_values or self.group_by
                    or self.measures or self.order_by or self.limit is not None)


# ── the delta on the wire ────────────────────────────────────────────────────────────
#: ops the engine's continuity lane understands (veda/understanding/continuity.py::WIRE_OPS)
WIRE_OPS = ("add_filter", "remove_filter", "change_group", "change_measure", "change_order",
            "drill_up", "switch_frame", "replace", "ambiguous", "new_topic", "compare")

# the classifier's delta_type vocabulary → a wire op, for turns the rule layer did not place
_SHAPE_SLOTS = ("limit", "order_by", "group_by", "measures")
_FROM_DELTA_TYPE = {"drill_up": "drill_up", "replace": "replace", "remove": "remove_filter",
                    "new_topic": "new_topic", "compare": "compare"}


def wire_delta(rule_delta: Optional[Dict[str, Any]], delta_type: Optional[str],
               delta_field: str = "", delta_value: str = "", *, chat_applied: bool = False,
               comparison: bool = False) -> Dict[str, Any]:
    """THIS turn's delta, for the engine: {op, slot, concept, value, confidence, applied}.

    `applied` is the double-application rule. context_resolve_node MUTATES the frame for
    some deltas before the context is built from it — drill_up pops the drill stack,
    switch_frame moves onto another stack entry, replace / remove (and the deterministic
    shape deltas) go through apply_context_delta. For those the context already IS the new
    state: `applied=True`, and the engine compiles it as-is. Every other op — the rule
    layer's add_filter / change_group / change_measure / change_order, or `ambiguous` when
    nothing placed the message — arrives `applied=False` and the engine applies it once.

    The rule layer's own op wins when it was confident (it is grounded in the previous
    result's values and dimensions); otherwise the classifier's delta_type is mapped, and a
    refine / drill_down the classifier could not break into slots is `ambiguous` — the
    engine spends its one constrained call on exactly that.
    """
    rd = rule_delta or {}
    rule_ok = (rd.get("op") not in (None, "ambiguous")
               and float(rd.get("confidence") or 0.0) >= 0.75)
    if comparison or delta_type == "compare":
        op = "compare"
    elif delta_type in ("drill_up", "new_topic"):
        op = delta_type
    elif delta_type in ("replace", "remove") and chat_applied:
        # a remembered FILTER removed, or any slot replaced / a shape slot (limit, order,
        # grouping) removed: the frame already holds the result either way
        op = ("remove_filter" if delta_type == "remove" and delta_field not in _SHAPE_SLOTS
              else "replace")
    elif rule_ok:
        op = rd["op"]
    else:
        op = _FROM_DELTA_TYPE.get(str(delta_type or ""), "ambiguous")
    out: Dict[str, Any] = {"op": op}
    if op == rd.get("op") and rule_ok:
        slot, concept, value = rd.get("slot"), rd.get("concept"), rd.get("value")
        conf = float(rd.get("confidence") or 0.0)
    else:
        slot, concept, value = None, (delta_field or None), (delta_value or None)
        conf = float(rd.get("confidence") or 0.0) if op == rd.get("op") else 0.0
    for k, v in (("slot", slot), ("concept", concept), ("value", value)):
        if v not in (None, ""):
            out[k] = v if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)[:_MAX_VALUE_LEN]
    out["confidence"] = round(conf, 3)
    out["applied"] = bool(op in ("drill_up", "switch_frame")
                          or (op in ("replace", "remove_filter") and chat_applied))
    return out
