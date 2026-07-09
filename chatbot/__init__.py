"""chatbot — standalone Supervisor/Planner LangGraph over the VEDA engine.

Separate from ``apps/chat`` (the existing Django API for the frontend) — this
package is built and tested standalone first. It does NOT touch ``apps/chat``,
``veda_core``, or the ``inference`` service. Once tested, ``apps/chat``'s
``ConversationQueryService`` will call into ``chatbot.run_chat_turn`` instead
of hitting the engine directly.
"""
