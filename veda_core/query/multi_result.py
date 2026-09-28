# =============================================================================
# query/multi_result.py
# VEDA — compound-query result envelope.
#
# A single user utterance may carry MORE THAN ONE independent question
# ("how many incidents are open AND list the active users"). Those need
# DIFFERENT SQL — they don't recompose into one query. The front door splits
# such an utterance into independent sub-queries (query.slm_layer.run_decomposer),
# runs EACH through the existing single-query pipeline UNCHANGED, and collects
# the per-sub results here.
#
# The single-query case is just a MultiResult with ONE item — so every caller
# branches on MultiResult ALWAYS and the pipeline downstream of the front door
# stays single-intent-dumb (it never learns about compound queries).
# =============================================================================

from dataclasses import dataclass, field
from typing import List, Optional, Any

# Closed enum — a sub-query either answered, was refused (expressible but not
# safely answerable: ungrounded value, out-of-scope shape, …), or errored
# (head/infra failure). Anything outside this set is a programming bug.
STATUS_OK = "ok"
STATUS_REFUSED = "refused"
STATUS_ERROR = "error"
_STATUSES = frozenset({STATUS_OK, STATUS_REFUSED, STATUS_ERROR})

# Typed outcome of one PART of a compound message. Maps onto the closed status enum
# (clarify → refused, timeout → error) so nothing that reads `status` changes.
OUTCOME_ANSWERED, OUTCOME_CLARIFY, OUTCOME_REFUSED = "answered", "clarify", "refused"
OUTCOME_TIMEOUT, OUTCOME_ERROR = "timeout", "error"
OUTCOME_STATUS = {OUTCOME_ANSWERED: STATUS_OK, OUTCOME_CLARIFY: STATUS_REFUSED,
                  OUTCOME_REFUSED: STATUS_REFUSED, OUTCOME_TIMEOUT: STATUS_ERROR,
                  OUTCOME_ERROR: STATUS_ERROR}


@dataclass
class SubResult:
    sub_query: str                       # the standalone question this block answers
    status: str                          # ok | refused | error  (see _STATUSES)
    route: str                           # deterministic | rag | hybrid | nosql | none
    result: Optional[Any] = None         # the existing head result, untouched
    refuse_reason: Optional[str] = None  # populated when status != ok
    # ── one PART of a compound message (front-door decomposition) ──
    # `status` stays in the closed enum above so every existing consumer keeps working;
    # `outcome` is the typed part result the compound reply and the chat tier read.
    part: Optional[str] = None           # the part text this item answers (its label)
    outcome: Optional[str] = None        # answered | clarify | refused | timeout | error
    lane: Optional[str] = None           # sql | tabular | rag
    source_id: Optional[str] = None      # the source this part ran on
    depends_on: Optional[int] = None     # index of the part whose result this one used
    elapsed_ms: Optional[float] = None

    def __post_init__(self):
        if self.status not in _STATUSES:
            self.status = STATUS_ERROR


@dataclass
class MultiResult:
    items: List[SubResult] = field(default_factory=list)  # order preserved (query order)
    # One summary over several per-source answers (independent multi-source APPEND merge,
    # 2026-09-18): composed from the items' own answers, never new facts. None when the
    # result is single-source or the merge was not APPEND. Items stay one-per-source.
    summary: Optional[str] = None
    trace_id: Optional[str] = None       # the ONE query-trace correlation id (observability);
                                         # set by run_hybrid_query, surfaced to the API caller
                                         # so a client can grep the full trace by this id
    # True when the items are the PARTS of one compound message (one intent each, labelled
    # with its part text) rather than one question answered by several sources.
    compound: bool = False
    relation: Optional[str] = None       # independent | dependent (compound only)

    @property
    def is_compound(self) -> bool:
        return len(self.items) > 1

    @property
    def ok(self) -> bool:
        """True only if EVERY sub-query answered. Partial success is not success —
        callers that need all-or-nothing semantics check this; the UI still renders
        each item with its own status regardless."""
        return bool(self.items) and all(i.status == STATUS_OK for i in self.items)

    @classmethod
    def single(cls, sub_query: str, status: str, route: str,
               result: Any = None, refuse_reason: Optional[str] = None) -> "MultiResult":
        return cls(items=[SubResult(sub_query=sub_query, status=status, route=route,
                                    result=result, refuse_reason=refuse_reason)])
