"""query/cross_source_guard.py — grounding guard for cross-source synthesized answers (routing Phase 4.5).

The highest hallucination risk in multi-source answering is SYNTHESIS: an LLM blending several
sources' results into one narrative can assert a number that appears in NEITHER source. VEDA already
has a single-answer numeric guard (`NL_SUMMARY_NUMERIC_GUARD` → `result_explainer._answer_numbers_grounded`);
this reuses that exact machinery, extended so the "allowed" numbers are pooled across ALL merged
source parts. A synthesized answer whose every stated number traces to some source's data passes;
one that invents a value is flagged so the caller can fall back (present the grounded per-source
parts instead of the synthesis).
"""
from __future__ import annotations

from typing import Optional, Tuple

# Reuse the platform's numeric-grounding primitive rather than reimplementing tolerance/magnitude
# parsing (crore/lakh/%/currency handling all live there).
from query.result_explainer import _answer_numbers_grounded


def _facts_from_merge(merge_result) -> dict:
    """Build a facts payload (recursively walked by the numeric guard) from the merged source parts:
    every source's rows + its own answer text are legitimate ground for a synthesized claim."""
    parts = getattr(merge_result, "parts", None)
    if parts is None and isinstance(merge_result, dict):
        parts = merge_result.get("parts")
    return {"parts": list(parts or [])}


def answer_grounded_in_sources(answer: str, merge_result) -> bool:
    """True when every number the synthesized ``answer`` states is traceable to some merged source's
    data (within the platform's standard ±2% / small-count tolerance)."""
    if not answer:
        return True
    return _answer_numbers_grounded(answer, _facts_from_merge(merge_result), None)


def guard_cross_source_answer(answer: str, merge_result) -> Tuple[bool, Optional[str]]:
    """(ok, reason). ok=False means the synthesis asserted an ungrounded number and must NOT be
    shown as-is — the caller should present the per-source parts (with provenance) instead."""
    if answer_grounded_in_sources(answer, merge_result):
        return True, None
    return False, "cross-source synthesis stated a number not present in any source's data"
