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

import re
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


def _known_operation(op: Optional[str]) -> str:
    """The operation if it is one of the closed set, else "" — never a free string."""
    from chatbot.prompts.delta_types import DELTA_TYPES
    op = str(op or "").strip().lower()
    return op if op in DELTA_TYPES else ""


def _terms_in(terms: Optional[List[str]], message: str) -> List[str]:
    """The resolved terms that really are words of the message — nothing else travels."""
    words = {w.lower() for w in re.findall(r"[A-Za-z0-9]+", message or "")}
    out: List[str] = []
    for t in terms or []:
        t = str(t or "").strip()
        if t and t.lower() in words and t.lower() not in (o.lower() for o in out):
            out.append(t[:32])
    return out[:8]


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
    # WHAT this turn does to the remembered state — the turn's delta, from the closed set
    # in chatbot/prompts/delta_types.py. The engine needs it for exactly one decision it
    # cannot otherwise make safely: whether the remembered shape is being REPLAYED. A
    # drill-up replays the base question, whose own grouping words ("distribution ... by
    # facing") look like a request for a NEW grouping; inferring the difference from the
    # text meant matching column names against words, which broke the moment the planner
    # grouped by a column the user never said (corner_property; measured 2026-09-25,
    # depth 2 -> 1 came back as 1000 raw rows). Said once, by the layer that knows.
    operation: str = ""
    # Words of `user_message` this layer RESOLVED against memory — the "second" / "one" of
    # "show the second one", which now travel as an id filter. They name a row the user
    # saw, not data, so the engine's qualifier gate must not demand them in the SQL
    # (measured 2026-09-25: "I couldn't map 'second' to any column or value"). Only words
    # that actually occur in the message are ever sent.
    resolved_terms: List[str] = _dc_field(default_factory=list)

    # ── construction ────────────────────────────────────────────────────────────────
    @classmethod
    def from_frame(cls, frame: Optional[Dict[str, Any]], user_message: str,
                   *, carry_state: bool = True,
                   operation: Optional[str] = None,
                   resolved_terms: Optional[List[str]] = None) -> "ConversationContext":
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

        return cls(
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
            operation=_known_operation(operation),
            resolved_terms=_terms_in(resolved_terms, user_message),
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
        if self.operation:
            payload["operation"] = self.operation
        if self.resolved_terms:
            payload["resolved_terms"] = list(self.resolved_terms)
        return payload

    def is_empty(self) -> bool:
        """True when nothing but the message travels — i.e. there is no context to send."""
        return not (self.entity_table or self.filters or self.filter_values or self.group_by
                    or self.measures or self.order_by or self.limit is not None)
