"""query/execution_request.py — Canonical Execution Request (Phase B1, foundation only).

Phase B1 of the Canonical Execution Request / Adapter Contract
(docs/architecture/VEDA_CANONICAL_EXECUTION_REQUEST_AUDIT.md §5/§9). Per the audit's evidence, the
de facto execution contract already exists as agents.py::BaseSourceAgent.execute()'s scattered
keyword arguments (query/source_id/source_ids/sm/cols/evidence/execution_context/on_event) — this
module NAMES that contract as one typed object, it does not invent a new one. No SQL text,
connection, or schema crosses this boundary today (confirmed absent, not merely omitted) — those
stay internal to the wrapped pipeline (run_query/generation.py/execute_sql), unchanged.

``sm``/``cols`` are kept on the request (matching current behavior exactly) but are explicitly
relational/tabular-only fields — FileSystemAgent/NoSqlAgent already accept and ignore them via the
existing kwargs signature, and ExecutionRequest preserves that same optionality rather than
inventing a stricter contract the current agents don't have.

Like source_capabilities.py (Phase A1) and source_adapters.py (Phase A2), this module has NO call
sites anywhere else in the codebase yet — not even source_adapters.py's execute_request() forces
it into dispatch(). Importing it changes nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from query.routing_contracts import ExecutionContext


@dataclass(frozen=True)
class ExecutionRequest:
    """The canonical, source-agnostic execution payload. Universal fields (query/source_id/
    source_ids/execution_context/on_event) are required by every source kind; ``sm``/``cols`` are
    relational/tabular-only and optional, exactly mirroring how _SqlAgent uses them today and how
    FileSystemAgent/NoSqlAgent already ignore them. Frozen — a request must not be mutated after
    construction; build a new one instead."""

    query: str
    source_id: str = ""
    source_ids: List[str] = field(default_factory=list)
    execution_context: Optional[ExecutionContext] = None
    on_event: Optional[Callable] = None

    # Relational/tabular-only — not required by document/nosql sources. See module docstring.
    sm: Optional[dict] = None
    cols: Optional[list] = None

    def __post_init__(self):
        if not self.query:
            raise ValueError("ExecutionRequest.query must be non-empty")
