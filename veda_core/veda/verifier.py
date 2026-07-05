"""Phase 4 / L5 — Verifier (ARCHITECTURE_HYBRID.md §2 L5).

Substrate-grounded ambiguity detection — NOT a free-form LLM judge. It reuses the anchor
scorer's already-computed alternatives: if a second entity sits within the confidence margin
of the chosen anchor, the subject is ambiguous and the answer's trust must drop / clarify.

This is the cheap, deterministic catch for the S01-class "confident wrong table" failure:
the signal already exists in the scorer's alternatives — the verifier just reads it.

Pure logic. No DB, no Ollama.
"""
from typing import Any, Dict, List, Optional

# Default margin: a runner-up within this of the top is "too close" → ambiguous subject.
DEFAULT_MARGIN = 0.06


def verify_anchor(chosen_anchor: str,
                  alternatives: List[Dict[str, Any]],
                  margin: float = DEFAULT_MARGIN) -> Dict[str, Any]:
    """alternatives: [{"table": str, "score": float}, ...] from the anchor scorer.
    Returns {ambiguous, runner_up, gap, confidence_penalty, reason}."""
    scored = sorted([a for a in (alternatives or []) if "score" in a],
                    key=lambda a: a["score"], reverse=True)
    if len(scored) < 2:
        return {"ambiguous": False, "runner_up": None, "gap": None,
                "confidence_penalty": 0.0, "reason": "no competing candidate"}

    top, second = scored[0], scored[1]
    gap = round(float(top["score"]) - float(second["score"]), 4)
    ambiguous = gap < margin

    # If the chosen anchor isn't even the top-scored candidate, that's a stronger signal.
    chosen_is_top = (top.get("table", "").lower() == (chosen_anchor or "").lower())
    if not chosen_is_top:
        ambiguous = True

    penalty = 0.0
    if ambiguous:
        # closer the runner-up, larger the penalty (capped)
        penalty = round(min(0.5, (margin - gap) + 0.1) if gap < margin else 0.2, 3)

    reason = ("subject ambiguous: " + (f"chosen '{chosen_anchor}' is not top candidate "
              f"'{top.get('table')}'" if not chosen_is_top
              else f"runner-up '{second.get('table')}' within {margin} (gap {gap})")) \
        if ambiguous else f"clear winner (gap {gap} ≥ {margin})"

    return {"ambiguous": ambiguous, "runner_up": second.get("table"),
            "gap": gap, "confidence_penalty": penalty, "reason": reason}


def verify(ir: Dict[str, Any],
           alternatives: List[Dict[str, Any]],
           margin: float = DEFAULT_MARGIN) -> Dict[str, Any]:
    """Run the verifier over a (consensus) IR. Returns the IR with adjusted confidence plus
    a verdict {ambiguous, should_clarify, reason}. Does not mutate the input IR."""
    v = verify_anchor(ir.get("anchor", ""), alternatives, margin)
    out = dict(ir)
    if v["ambiguous"]:
        out["confidence"] = round(max(0.0, float(ir.get("confidence", 0.0)) - v["confidence_penalty"]), 3)
    return {
        "ir": out,
        "ambiguous": v["ambiguous"],
        "should_clarify": v["ambiguous"],
        "runner_up": v["runner_up"],
        "reason": v["reason"],
    }
