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
        """True when the intent is a member of the closed INTENTS vocabulary."""
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
    value: Any = None         # grounded value: a sampled categorical value (normalised) or a number
    concept: str = ""
    # M2 (2026-09-16): a filter is a TYPED predicate, not a bare (column, value) pair.
    op: str = "="             # "=" | "!=" | ">" | ">=" | "<" | "<="
    numeric: bool = False     # True → value is a number compared on a numeric column
    semantic_type: Optional[str] = None   # the column's L2 semantic_type at grounding time


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
    # M2 (2026-09-16): grounded time window on the anchor's TEMPORAL column, or None.
    time: Optional[Dict[str, Any]] = None          # {"column", "start", "end"}
    # "how many amenity categories" → COUNT(DISTINCT category): the dimension column
    # the count is OVER (validated on the anchor), or None for a plain row count.
    distinct_column: Optional[str] = None
    # HOW the anchor was grounded (grounding.GROUND_*): exact_name | glossary |
    # table_vocabulary | name_tokens | retrieval. Pre-M3 item 1 (2026-09-16).
    anchor_method: str = "retrieval"

    @property
    def reentry_eligible(self) -> bool:
        """May this intent take FIRST position on re-entry? Only when the anchor was
        grounded by the user's own naming (exact / glossary / vocabulary / name tokens).
        A retrieval-top-table grounding is a ranked guess: candidate-only, always."""
        from veda.understanding.grounding import REENTRY_METHODS
        return self.anchor is not None and self.anchor_method in REENTRY_METHODS

    @property
    def fully_grounded(self) -> bool:
        """Every REQUIRED slot resolved to a real artifact: the anchor, and every dimension /
        filter / time window the question carried (those refuse in ground() when they
        can't). The LLM's free-form `entities` list is informational — a phantom entry it
        echoes (e.g. 'assets_property') must not demote a correctly grounded intent."""
        return self.anchor is not None

    @property
    def tables(self) -> List[str]:
        """All real tables in play: the anchor followed by the secondaries (drops None)."""
        return [t for t in ([self.anchor] + list(self.secondaries)) if t]


@dataclass
class Refusal:
    """The firewall output — a concept couldn't be grounded, or the query is
    impossible/ambiguous. Carries WHY (for the user + trace). Never a guess."""
    reason: str                            # machine tag: "ungrounded" | "ambiguous" | "impossible"
    message: str                           # human-facing
    unresolved: List[str] = field(default_factory=list)   # concepts that failed to ground
    evidence: Dict[str, Any] = field(default_factory=dict)
