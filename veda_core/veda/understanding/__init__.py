"""veda.understanding — enterprise query-understanding layer (flag-gated, default OFF).

Source-agnostic, zero-hallucination. Pipeline:

    understand_query(query, sm, ...) →
        1. EXTRACT (LLM)      : query → RawIntent (concepts only; never columns)
        2. GROUND (deterministic) : concepts → real, validated schema artifacts
        3. FIREWALL           : anything ungroundable → REFUSE/CLARIFY (never guess)
     → GroundedIntent | Refusal | None

Returning None means "degrade" — the caller falls back to the EXISTING pipeline
unchanged (SLM down, low confidence, parse failure). Nothing here ever reaches SQL
without deterministic validation, and the flag defaults OFF so production is
byte-identical until explicitly enabled.
"""
from veda.understanding.schema import (          # noqa: F401
    RawIntent, GroundedIntent, Refusal, GroundedMeasure, GroundedFilter,
    INTENTS, SCHEMA_VERSION,
)
from veda.understanding.orchestrator import understand_query  # noqa: F401

__all__ = [
    "understand_query", "RawIntent", "GroundedIntent", "Refusal",
    "GroundedMeasure", "GroundedFilter", "INTENTS", "SCHEMA_VERSION",
]
