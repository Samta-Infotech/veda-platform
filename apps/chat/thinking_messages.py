"""apps.chat.thinking_messages — business-friendly display text for the chat
"thinking" progress stream (UX-only, Phase 1).

Single source of truth mapping internal pipeline phase identifiers (emitted,
UNCHANGED, by chatbot/nodes.py's classify_node/context_resolve_node and
forwarded verbatim from veda_core/veda_hybrid.py + veda_core/veda/pipeline.py
via the inference-tier SSE stream) to what the end user actually sees while a
query runs. Purely a display-layer translation:

- No phase is renamed — internal phase strings, logging, and tracing
  (ExplainTrace, `tr.set`/`tr.check`) are completely untouched.
- No pipeline/routing/retrieval/SQL logic changes — this module is consulted
  ONLY at the point a "thinking" SSE event is about to leave the api tier for
  the frontend (ConversationQueryService._run_streamed's on_event,
  apps/chat/services.py) — see that call site.
- No AI/SLM/LLM call, no dynamic generation, no added latency — a plain dict
  lookup.

Never expose in these messages: SQL, table/database names, "supervisor",
"routing", "schema linking", "tier2", "RAG"/"NoSQL", or any other internal
component name — see each phase's own source (file:line) for what it
actually maps from.
"""
from __future__ import annotations

# phase -> business-friendly message. One entry per DISTINCT phase string
# emitted anywhere in the pipeline (chatbot/nodes.py + veda_core/veda_hybrid.py
# + veda_core/veda/pipeline.py) — several internal phases carry more than one
# possible raw message (e.g. "answer" is the terminal phase for every engine
# path — deterministic/tier2/RAG/hybrid/nosql — and "sql_planning" covers 8
# distinct planning actions); those intentionally collapse to ONE friendly
# message per phase, matching the UX goal of describing "what the assistant
# is doing", not which internal action fired.
THINKING_PHASE_MESSAGES: dict[str, str] = {
    # chatbot/nodes.py — node-local phases, precede the engine call
    "supervisor_classify": "🧠 Understanding your question...",
    "supervisor_followup": "📖 Understanding the context of your request...",

    # veda_core/veda_hybrid.py
    "classify": "🧭 Analyzing your question...",
    "route": "🎯 Determining the best way to answer your question...",
    "sql_probe": "⚡ Checking for a fast answer...",
    "decompose": "🧩 Breaking down your question...",
    "sub_query": "🔄 Answering part of your question...",
    "tier2": "🔍 Digging deeper to find your answer...",
    "rag": "📄 Searching through your documents...",
    "hybrid": "🔗 Combining information from multiple sources...",
    "nosql": "🔍 Searching your records...",
    "answer": "📝 Preparing your answer...",

    # veda_core/veda/pipeline.py
    "schema_linking": "🗂 Finding the required business information...",
    "sql_planning": "📋 Preparing the analysis...",
    "output": "✅ Finalizing the results...",

    # apps/chat/services.py — api-tier-local, NOT forwarded from veda_core/
    # chatbot (there's no pipeline phase for chart-building; it happens
    # synchronously inside _build_reply_events). Only emitted when a chart is
    # actually about to be shown (see that call site) — a text/table-only
    # answer never shows this, so it's never a message with nothing behind it.
    "visualization_prep": "📊 Creating a visual summary...",
}


def business_friendly_message(phase: str, fallback: str) -> str:
    """The text to actually show the user for `phase`. Falls back to the
    original internal message when a phase isn't in the mapping yet — a
    newly-added internal phase (future pipeline change) degrades to showing
    its raw message instead of silently disappearing, until it's added here.
    """
    return THINKING_PHASE_MESSAGES.get(phase, fallback)
