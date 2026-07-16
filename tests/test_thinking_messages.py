"""Tests for apps/chat/thinking_messages.py — the business-friendly "thinking"
message mapping (UX Phase 1). Pure dict/function, no Django settings needed.

Covers: every real phase string emitted anywhere in the pipeline
(chatbot/nodes.py, veda_core/veda_hybrid.py, veda_core/veda/pipeline.py — as
of 2026-07) has a friendly mapping; unknown/future phases degrade gracefully
to their original message instead of disappearing; no forbidden internal
terminology (SQL, table names, tier2, RAG, NoSQL, supervisor/routing/schema-
linking as literal jargon) ever leaks into a mapped message."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps.chat.thinking_messages import THINKING_PHASE_MESSAGES, business_friendly_message

# Every phase string actually emitted in production as of 2026-07 (see
# chatbot/nodes.py's classify_node/context_resolve_node/call_engine_node,
# veda_core/veda_hybrid.py's _emit calls, veda_core/veda/pipeline.py's _tick
# calls, plus apps/chat/services.py's own local visualization_prep) — kept as
# an explicit list here (not re-derived from the mapping itself) so a phase
# silently dropped from THINKING_PHASE_MESSAGES fails this test, rather than
# the test trivially passing against whatever's left.
_REAL_PHASES = {
    "supervisor_classify", "supervisor_followup",
    "classify", "route", "sql_probe", "decompose", "sub_query",
    "tier2", "rag", "hybrid", "nosql", "answer",
    "schema_linking", "sql_planning", "output",
    "visualization_prep",
    # veda_core/query/rag_layer.py + veda_hybrid.py::_run_nosql sub-steps
    # (2026-07-16) — see thinking_messages.py's own comments for why these
    # were added (previously silent black boxes around the SLM/query-
    # building calls inside the rag/hybrid/nosql paths).
    "rag_retrieve", "rag_synthesize", "hybrid_retrieve", "hybrid_synthesize",
    "nosql_build",
}

_FORBIDDEN_TERMS = re.compile(
    r"\b(sql|tier-?2|rag|nosql|supervisor|routing|schema.linking|"
    r"database|table|query engine)\b",
    re.IGNORECASE,
)


def test_every_real_phase_has_a_friendly_mapping():
    missing = _REAL_PHASES - THINKING_PHASE_MESSAGES.keys()
    assert not missing, f"phases emitted in production but missing a friendly message: {missing}"


def test_no_forbidden_technical_terms_in_any_mapped_message():
    leaky = {phase: msg for phase, msg in THINKING_PHASE_MESSAGES.items()
            if _FORBIDDEN_TERMS.search(msg)}
    assert not leaky, f"business-friendly messages leaking internal terminology: {leaky}"


def test_unknown_phase_falls_back_to_original_message():
    """A phase not yet in the mapping (e.g. a future pipeline addition) must
    degrade to showing its own original message, never crash or go blank."""
    assert business_friendly_message("some_brand_new_phase", "raw internal text") == "raw internal text"


def test_known_phase_returns_mapped_message_not_fallback():
    assert business_friendly_message("route", "Routed to sql engine") == THINKING_PHASE_MESSAGES["route"]
    assert business_friendly_message("route", "Routed to sql engine") != "Routed to sql engine"


def test_answer_phase_collapses_every_engine_path_to_one_message():
    """"answer" is the terminal phase for every engine path (deterministic,
    tier2, rag, hybrid, nosql) with several different raw messages — all of
    them must collapse to the SAME friendly message (the point of a phase-
    keyed mapping, not a message-keyed one)."""
    raw_variants = [
        "Deterministic SQL answered the query",
        "Tier-2 SQL answered the query",
        "SQL query executed",
        "SQL query could not be answered",
        "Synthesized answer from retrieved documents",
        "Fused SQL and document results into an answer",
        "NoSQL query executed",
    ]
    mapped = {business_friendly_message("answer", raw) for raw in raw_variants}
    assert mapped == {THINKING_PHASE_MESSAGES["answer"]}


def test_mapping_values_are_all_non_empty_strings():
    for phase, msg in THINKING_PHASE_MESSAGES.items():
        assert isinstance(msg, str) and msg.strip(), f"phase {phase!r} has an empty/invalid message"
