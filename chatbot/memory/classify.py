"""chatbot.memory.classify — the ONE constrained-output SLM call this package
makes (§8/§23 of docs/MEMORY_ARCHITECTURE.md).

Reuses chatbot/llm.py::call_slm — the same backend-agnostic (Ollama|vLLM)
transport every other chatbot/ call already uses (external-API-friendly:
whatever OLLAMA_URL/VLLM_URL is configured, this rides along, same as
classify_node's own call). Deterministic decoding (temperature 0) — same
reproducibility rationale as veda_core/veda/generation.py::generate_sql()'s
"SQL generation must be DETERMINISTIC" comment; classification needs the same
guarantee for the same reason (run-to-run drift here would make memory
non-reproducible, which is worse than not having it).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from ..llm import call_slm
from ..prompts.delta_classify import (
    DELTA_TYPES,
    build_delta_classify_system_prompt,
    build_delta_classify_user_prompt,
)

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_delta_response(raw: Optional[str], message: str) -> Tuple[str, list]:
    """Parse + vocabulary-gate + confidence-gate a raw SLM response containing
    `delta_type`/`slot_candidates` JSON fields, regardless of which prompt
    produced it. Extracted (latency fix — see chatbot/prompts/supervisor.py's
    module docstring) so classify_node's MERGED call (action + delta_type in
    ONE round-trip, when a frame exists) and classify_delta()'s standalone
    fallback call share the exact same parsing/gating logic — no duplicated,
    divergence-prone copy of the vocabulary/confidence gates.

    Fails closed to "ambiguous" on any parse failure — never guesses a
    continuation type it isn't sure of (refuse-over-guess, same posture as
    chatbot/nodes.py::classify_node's own error handling)."""
    if not raw:
        return "ambiguous", []

    match = _JSON_RE.search(raw)
    if not match:
        return "ambiguous", []
    try:
        parsed = json.loads(match.group())
    except Exception:
        return "ambiguous", []

    delta_type = parsed.get("delta_type")
    if delta_type not in DELTA_TYPES:
        return "ambiguous", []

    slots = parsed.get("slot_candidates") or []
    if not isinstance(slots, list):
        slots = []

    # Vocabulary gate (§12 barrier 2): a slot candidate the model claims came
    # from the message must actually be a substring of it — this is enforced
    # again, independent of the prompt's own instruction, so a model that
    # ignores the instruction still can't inject an unrelated value.
    msg_lower = message.lower()
    grounded_slots = [s for s in slots if isinstance(s, str) and s.lower() in msg_lower]

    # Audit fix (H3): slot_candidates were computed and vocabulary-gated but
    # then discarded entirely — real SLM cost for zero effect. Give them an
    # actual role: "refine"/"drill_down"/"compare" all imply the model found
    # a NEW concrete value to apply. If the model proposed candidates but
    # EVERY one of them failed the grounding check (i.e. it claimed a value
    # that isn't actually in the user's own message), that's a real signal
    # the classification itself is an unsupported guess, not a grounded
    # continuation — downgrade to "ambiguous" rather than trust it.
    # "drill_up"/"new_topic" are exempt: they legitimately add no new value.
    # A `delta_type` with slots=[] (model proposed nothing at all) is NOT
    # downgraded here — that's a softer signal (e.g. a legitimate
    # regroup/comparison with no new literal value) and downgrading it too
    # would make ordinary refinements needlessly ask for clarification.
    if delta_type in ("refine", "drill_down", "compare") and slots and not grounded_slots:
        logger.info(
            "parse_delta_response: delta_type=%s proposed slot_candidates=%r but NONE were "
            "grounded in the message — downgrading to ambiguous rather than trusting "
            "an unsupported classification", delta_type, slots,
        )
        return "ambiguous", []

    return delta_type, grounded_slots


def classify_delta(
    frame: Optional[Dict[str, Any]], message: str, episodic: Optional[list] = None,
) -> Tuple[str, list]:
    """Standalone fallback call (chatbot/nodes.py::context_resolve_node uses
    this ONLY when classify_node's own merged call — see
    chatbot/prompts/supervisor.py — didn't already produce a valid
    delta_type this turn, e.g. its SLM call failed/timed out). Returns
    (delta_type, slot_candidates); fails closed to "ambiguous" on any
    SLM error/timeout/unparseable output.

    `episodic`: the short capped Redis buffer (audit fix H1 — previously
    computed/stored but never actually passed into this prompt), used purely
    to help resolve "it"/"that"/"tell me more"-style references; never a
    source of new filter/entity facts (see build_delta_classify_user_prompt).
    """
    if not frame or not frame.get("entity"):
        # No prior frame to continue from — deterministically new_topic, no
        # SLM call needed at all (mirrors classify_node's own instant
        # deterministic fast paths for greetings/thanks/bye/date questions).
        return "new_topic", []

    try:
        raw = call_slm(
            build_delta_classify_system_prompt(),
            build_delta_classify_user_prompt(frame, message, episodic),
            temperature=0,
            max_tokens=80,
        )
    except Exception:
        # call_slm itself returns None on failure rather than raising, but this
        # guards against any future change to that contract — refuse-over-guess.
        logger.warning("classify_delta: call_slm failed, defaulting to ambiguous", exc_info=True)
        return "ambiguous", []

    return parse_delta_response(raw, message)
