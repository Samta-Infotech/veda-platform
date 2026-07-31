"""veda.understanding.schema — the TYPED CONTRACT for the understanding layer.

Two typed objects, versioned:

  RawIntent      — the LLM's output. CONCEPTS ONLY (user-facing words). It may name
                   an entity/measure/dimension as a *phrase*; it must NEVER be trusted
                   to name a real table/column. Untrusted until grounded.

  GroundedIntent — the deterministic grounding's output. Every field is a REAL,
                   schema-validated artifact (table/column names that exist). This is
                   the only thing allowed to flow downstream to the planner.

  Refusal        — emitted when a required concept cannot be grounded (the
                   anti-hallucination firewall) or the query is impossible/ambiguous.

Keeping these separate + typed is the enterprise contract: the trust boundary is the
type boundary. A RawIntent can never be mistaken for a GroundedIntent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

# Closed vocabulary — source-agnostic analytical intents.
INTENTS = frozenset({
    "list", "count", "sum", "avg", "max", "min", "rank", "compare", "refuse", "clarify",
})


# ── LLM output (untrusted concepts) ──────────────────────────────────────────
@dataclass
class RawIntent:
    """Exactly what the LLM returned, normalized + shape-validated but NOT grounded.
    Every string here is a CONCEPT/phrase, not a verified schema artifact."""
    intent: str                              # one of INTENTS
    grain: Optional[str] = None              # the entity the answer is PER (the subject)
    measure: Optional[str] = None            # what is aggregated/ranked-by, or None
    dimensions: List[str] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)   # [{concept, value}]
    entities: List[str] = field(default_factory=list)
    confidence: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)   # verbatim LLM json, for trace

    def is_valid_shape(self) -> bool:
        return self.intent in INTENTS


# ── grounded, validated output (trusted) ─────────────────────────────────────
@dataclass
class GroundedMeasure:
    kind: str                 # "count" | "sum" | "avg" | "max" | "min"
    table: Optional[str] = None   # real table the measure lives on / is counted from
    column: Optional[str] = None  # real column (None for COUNT(*))
    concept: str = ""             # the originating concept (trace)


@dataclass
class GroundedFilter:
    table: str                # real table
    column: str               # real column (existence-validated)
    value: Any = None         # value is still value-grounded downstream (L6a)
    concept: str = ""


@dataclass
class GroundedIntent:
    intent: str
    anchor: Optional[str]                 # real grain/anchor table (validated)
    secondaries: List[str] = field(default_factory=list)   # other real tables
    measure: Optional[GroundedMeasure] = None
    dimensions: List[GroundedFilter] = field(default_factory=list)   # real dim columns
    filters: List[GroundedFilter] = field(default_factory=list)
    confidence: float = 0.0
    schema_version: int = SCHEMA_VERSION
    evidence: Dict[str, Any] = field(default_factory=dict)   # grounding provenance/trace

    @property
    def tables(self) -> List[str]:
        return [t for t in ([self.anchor] + list(self.secondaries)) if t]


@dataclass
class Refusal:
    """The firewall output — a concept couldn't be grounded, or the query is
    impossible/ambiguous. Carries WHY (for the user + trace). Never a guess."""
    reason: str                            # machine tag: "ungrounded" | "ambiguous" | "impossible"
    message: str                           # human-facing
    unresolved: List[str] = field(default_factory=list)   # concepts that failed to ground
    evidence: Dict[str, Any] = field(default_factory=dict)
