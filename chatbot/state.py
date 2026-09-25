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
    source_ids: Optional[List[int]]    # validated multi-source scope (P5), primary first —
                                       # forwarded alongside source_id so scoped chat turns
                                       # retrieve/federate exactly like /api/v1/query
    request_id: str                    # forwarded as X-Request-Id (tracing across api->inference)
    data_scope: Optional[Dict[str, Any]]  # Gate 1 (User Story 3, Task 15) — the api tier's
                                       # precomputed RBAC data-scope payload (a plain
                                       # JSON-safe dict, see apps.access_management.services.
                                       # serialize_data_scope), forwarded to call_engine_node
                                       # as the X-Veda-Data-Scope header. None = no narrowing.
    # The api tier's precomputed grounding facts (apps/chat/services.py), one bool each.
    # DECLARED, or LangGraph drops them from the graph input without a word: measured
    # 2026-09-25, message_mentions_data never reached classify_node at all — the
    # "a follow-up in a session that has answered nothing, naming nothing in the data"
    # guard it feeds had been dead since the value became a precomputed bool.
    # Who is asking — the key for the only memory that crosses sessions (per-user session
    # summaries, chatbot/memory/topics.py). Set by apps/chat/services.py; None for callers
    # with no user (CLI), which simply have no cross-session memory. DECLARED — see above.
    memory_in: Optional[Dict[str, Any]]  # chatbot/telemetry.py::memory_summary — what memory
                                         # held when THIS turn started (the decision record)
    user_id: Optional[str]
    previous_topics: List[Dict[str, Any]]  # topics from the user's OTHER sessions, already
                                           # filtered to this turn's authorised sources
    message_mentions_data: Optional[bool]
    message_names_only_values: Optional[bool]  # only data VALUES, no table/column word
    data_vocabulary: Optional[List[str]]  # words the scoped sources actually contain
                                       # (apps.query.data_vocabulary) — used ONLY to skip
                                       # the engine for a message that names none of them
    source_profiles: Optional[Dict[str, Any]]  # per-source routing metadata
                                       # (source_type/is_canonical/domain_tags/description, from
                                       # apps.query.scope.source_profiles_for), forwarded to
                                       # call_engine_node the same way /api/v1/query does. WITHOUT
                                       # it the engine cannot tell a datalake/document source from a
                                       # relational one: veda_hybrid._is_datalake_source returns
                                       # False, the datalake-isolated semantic model is never
                                       # loaded, and the SQL planner is handed the PRIMARY source's
                                       # schema instead — 178 homzhub tables that do not contain
                                       # `vendors` at all, so a vendor question could only pick
                                       # among irrelevant tables (measured: worklists_quote vs
                                       # list_of_values_listofvalue at 0.1503 vs 0.1502, then an
                                       # "ambiguous subject" clarify). {} / None = same as before.
    # The structured conversation state for THIS turn, built by context_resolve_node from
    # the frame AFTER the delta was applied (chatbot/memory/context.py). Carried to the
    # engine in `flags`, alongside — never inside — the user's own words.
    #
    # IT MUST BE DECLARED HERE. LangGraph's StateGraph is typed by this TypedDict and
    # silently DROPS any key a node returns that the schema does not name. This field was
    # lost once (2026-09-23) and the symptom did not point at it at all: the whole
    # conversation boundary went quiet — context_resolve_node still classified the delta
    # correctly and still passed the user's words through untouched, but the context never
    # reached call_engine_node, so every follow-up arrived at the engine as a bare fragment
    # ("only the Nagpur ones"), was routed to RAG, and came back "The provided context does
    # not contain information specific to Nagpur", which then overwrote the frame with a
    # document entity and poisoned the rest of the conversation.
    conversation_context: Optional[Dict[str, Any]]

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
    delta_field: Optional[str]         # replace/remove only — which remembered filter the
                                       # delta acts on; Python binds it, the model only names it
    delta_value: Optional[str]         # the ONE grounded word from the user's own message
    # What the conversation layer actually carried into THIS turn, for the user to see.
    # Built by context_resolve_node/clarify_reply_node from facts that already exist —
    # the frame it merged and the delta it applied — and never re-derived or inferred.
    # None on any turn that used no context at all (a first question, smalltalk, recall),
    # which is exactly when there is nothing honest to show.
    context_used: Optional[Dict[str, Any]]
    # chatbot.memory.context.ConversationContext payload, sent to the engine as
    # flags.conversation_context. MUST stay declared here: StateGraph silently drops any key
    # a node returns that this TypedDict does not declare, and losing this one turned every
    # follow-up into a bare fragment (see SESSION_HANDOFF_2026-09-24.md §1).
    conversation_context: Optional[Dict[str, Any]]
    comparison: Dict[str, Any]         # chatbot.memory.frame.build_comparison — both sides of an
                                       # active comparison; SESSION-level, read by memory_read_node
    pending_clarification: Dict[str, Any]  # {question, original_query, missing, turn_index} —
                                       # set when the engine asks, consumed by the next turn
    memory_reset: bool                 # set by memory_read_node on a "start over" match;
                                       # classify_node ends the turn on it (nodes.py::reset_node)
    result_reference: Optional[Dict[str, Any]]  # chatbot/memory/reference.py — identities of the
                                       # rows the user is looking at, read from Redis each turn.
                                       # DECLARED or LangGraph silently drops it (see the
                                       # conversation_context note: that exact trap disabled
                                       # the whole memory boundary once).
    # chatbot/memory/topics.py — the bounded index of earlier topics (≤5, most recent first),
    # read from Redis by memory_read_node every turn ALREADY filtered to this turn's
    # authorised sources, and written only by memory_write_node on an answered turn.
    # DECLARED or LangGraph silently drops it (the conversation_context trap above).
    topic_index: Optional[List[Dict[str, Any]]]
    # THIS turn's return-to-topic decision, set by classify_node only:
    #   {"kind": "restore", "topic": <index entry>}  — the message names ONE remembered
    #                                                   non-current topic; replay it
    #   {"kind": "ambiguous", "candidates": [...]}   — it names several; ask which
    # None on every other turn. Reset to None by memory_read_node at the start of EVERY
    # turn, so a route that skips classify's final return can never see a stale one.
    topic_restore: Optional[Dict[str, Any]]
    # The source whose memory memory_read_node discarded THIS turn because it is no
    # longer in the caller's scope (None otherwise). Lets a follow-up that points back
    # at that memory be told the truth instead of being answered as smalltalk.
    memory_revoked_source: Optional[Any]
    # Earlier answered results the user can point back at by name ("the 1st one from
    # the price list") — chatbot/memory/reference.py::remember_result. Filtered to this
    # turn's authorised sources by memory_read_node.
    result_history: Optional[List[Dict[str, Any]]]
    last_result: Dict[str, Any]        # the ANSWERED result the user is currently looking at,
                                       # kept across turns so a presentation-only follow-up can
                                       # redraw it — engine_result is cleared every turn
    recall_kind: Optional[str]         # "query"|"sql"|"table"|"filters"|"rows"|"trail" — set only by
                                       # a question ABOUT the conversation (nodes.py::recall_node),
                                       # answered from the QueryFrame without the engine
    viz_override: Optional[str]        # "pie"|"bar"|"line"|"table"|"chart"|"csv" — set only by a
                                       # presentation-only follow-up (nodes.py::represent_node),
                                       # which re-renders the previous result without the engine
    # Audit fix (H1): the short capped Redis episodic buffer (MemoryStore's
    # ":episodic" key), loaded by memory_read_node and passed to
    # classify_delta() for reference-resolution ("it"/"that"/"tell me more")
    # ONLY — never re-parsed back into the QueryFrame itself (frame updates
    # come exclusively from harvest_frame(), i.e. executed/validated
    # evidence). Previously written every turn but never read anywhere —
    # dead code; now actually consumed.
    episodic: List[Dict[str, str]]
