"""chatbot — Supervisor/Planner LangGraph over the VEDA engine.

``apps/chat``'s ``ConversationQueryService`` (apps/chat/services.py) calls
into ``chatbot.run_chat_turn`` for every turn. This package itself still never
imports ``veda_core`` directly — it reaches the engine over HTTP via
``apps.query.inference_client.InferenceClient`` (chatbot/nodes.py::
call_engine_node), the same client every other api-tier caller uses.
"""
