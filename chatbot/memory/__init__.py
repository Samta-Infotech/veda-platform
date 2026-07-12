"""chatbot.memory — structured analytical memory (see docs/MEMORY_ARCHITECTURE.md).

Replaces free-text conversation replay with a small, schema-grounded
QueryFrame + DrillStack, persisted in Redis with O(1) size regardless of
conversation length. Every field in a stored frame traces back to an
ACTUALLY EXECUTED, already-validated engine result (`frame.harvest_frame`) —
never to an LLM's free-text guess. The only LLM call in this package
(`classify.classify_delta`) is a closed-set classifier whose output is
re-validated deterministically before anything is merged (`frame.merge_frame`).

Public entry points used by chatbot/nodes.py:
    from chatbot.memory.store import MemoryStore
    from chatbot.memory import frame as memory_frame
    from chatbot.memory.classify import classify_delta
"""
from __future__ import annotations
