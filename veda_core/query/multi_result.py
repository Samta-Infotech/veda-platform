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


@dataclass
class SubResult:
    sub_query: str                       # the standalone question this block answers
    status: str                          # ok | refused | error  (see _STATUSES)
    route: str                           # deterministic | rag | hybrid | nosql | none
    result: Optional[Any] = None         # the existing head result, untouched
    refuse_reason: Optional[str] = None  # populated when status != ok

    def __post_init__(self):
        if self.status not in _STATUSES:
            self.status = STATUS_ERROR


@dataclass
class MultiResult:
    items: List[SubResult] = field(default_factory=list)  # order preserved (query order)

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
