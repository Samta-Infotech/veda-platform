"""chatbot.state — the state object passed through every LangGraph node."""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict


class Turn(TypedDict):
    role: str          # "user" | "assistant"
    content: str


# Audit fix (H4): `history` used a plain operator.add reducer — every node's
# returned "history" list was APPENDED forever, with nothing ever capping the
# checkpoint's own stored size (only the PROMPT read side, history[-6:] in
# chatbot/prompts/supervisor.py, was bounded — the underlying Redis-persisted
# checkpoint kept growing per turn for the life of a session). This was the
# original problem docs/MEMORY_ARCHITECTURE.md set out to fix and it was
# never actually addressed by the rest of that work. Custom reducer: still
# appends (so _turn_delta()'s per-turn [user, assistant] pair is preserved
# exactly as before), but caps the STORED list itself — same effect as
# operator.add from every node's point of view, bounded in the checkpoint.
_HISTORY_MAX_TURNS = 10          # keep last 10 turns = 20 [user, assistant] entries


def _capped_append(existing: List["Turn"], new: List["Turn"]) -> List["Turn"]:
    combined = (existing or []) + (new or [])
    return combined[-(_HISTORY_MAX_TURNS * 2):]


class ChatState(TypedDict, total=False):
    # ── input ────────────────────────────────────────────────────────────────
    message: str                      # this turn's raw user message
    # Annotated + _capped_append = a REDUCER: each node's returned "history"
    # list is APPENDED to (not overwritten on top of) whatever the
    # checkpointer already has for this thread_id/session_id, THEN trimmed to
    # the last _HISTORY_MAX_TURNS turns (audit fix H4 — see _capped_append
    # above). Terminal nodes append this turn's [user, assistant] pair;
    # classify/context_resolve only READ it (never return it), so they never
    # trigger an append. Callers no longer need to thread conversation
    # history through manually — the checkpointer + this reducer accumulate
    # it (boundedly) automatically per session.
    history: Annotated[List[Turn], _capped_append]
    session_id: str
    tenant: str
    source_id: Optional[int]           # forwarded to InferenceClient.stream_hybrid_query
    request_id: str                    # forwarded as X-Request-Id (tracing across api->inference)

    # ── supervisor decision ─────────────────────────────────────────────────
    action: str                        # "smalltalk" | "answer" | "clarify" | "followup"
    resolved_query: str                # the query actually sent to the engine
                                       # (== message, or message merged with prior context)

    # ── engine result ───────────────────────────────────────────────────────
    engine_result: Dict[str, Any]      # raw run_hybrid_query() result (dict form)
    status: str                        # "answered" | "refuse" | "clarify" | "no_table" |
                                       # "ungrounded" | "qualifier_dropped" | "ir_mismatch" |
                                       # "error" | "unavailable" (transport/infra failure —
                                       # distinct from a reachable engine's own refusal)

    # ── output ───────────────────────────────────────────────────────────────
    reply_text: str
    needs_clarification: bool
    clarification_question: Optional[str]
    sql: Optional[str]
    rows: Optional[list]
    engine_unavailable: bool           # True only when status == "unavailable"

    # ── structured analytical memory (chatbot/memory/, docs/MEMORY_ARCHITECTURE.md) ──
    # Loaded by memory_read_node from Redis (chatbot/memory/store.py::MemoryStore),
    # consumed/mutated by context_resolve_node (pre-execution) and written by
    # memory_write_node (post-execution, evidence-only — see chatbot/memory/frame.py).
    # NOT part of the checkpoint's growth path: unlike `history` above, these
    # are small, bounded structures re-read/re-written from their OWN Redis
    # keys each turn, capped independent of this checkpoint's own size.
    frame: Dict[str, Any]              # chatbot.memory.frame.QueryFrame
    drill_stack: List[Dict[str, Any]]  # chatbot.memory.frame.DrillLevel list
    delta_type: str                    # "new_topic"|"refine"|"drill_down"|"drill_up"|"compare"|"ambiguous"
    # Audit fix (H1): the short capped Redis episodic buffer (MemoryStore's
    # ":episodic" key), loaded by memory_read_node and passed to
    # classify_delta() for reference-resolution ("it"/"that"/"tell me more")
    # ONLY — never re-parsed back into the QueryFrame itself (frame updates
    # come exclusively from harvest_frame(), i.e. executed/validated
    # evidence). Previously written every turn but never read anywhere —
    # dead code; now actually consumed.
    episodic: List[Dict[str, str]]
