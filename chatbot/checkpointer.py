"""chatbot.checkpointer — LangGraph state persistence.

NOW: RedisSaver (langgraph-checkpoint-redis). This needs the RediSearch module
(FT.* commands), which the project's plain Homebrew Redis (redis-cache on
:6379) does NOT have — so this points at a separate `redis-stack-server`
Docker container on :6380 instead (started once for local dev:
`docker run -d --name veda-redis-stack -p 6380:6379 redis/redis-stack-server`).
Survives across separate `python chat_cli.py` invocations (unlike MemorySaver,
which only lives for one process's lifetime) — same `thread_id`/session_id
picks its conversation back up even in a brand-new process.

LATER (when wired into apps/chat, if we want durability beyond Redis's
eviction/restart semantics, or to avoid running a second Redis just for this):
swap get_checkpointer() to a PostgresSaver against the Django `veda` DB
instead. Every other module (graph.py, run.py) is unaffected by this swap —
they only call get_checkpointer(), never construct a saver directly.
"""
from __future__ import annotations

import logging
import os
import threading

from langgraph.checkpoint.redis import RedisSaver

logger = logging.getLogger(__name__)

_CHECKPOINTER = None
_LOCK = threading.Lock()

# redis-stack-server (RediSearch-capable), separate from the project's
# plain redis-cache/redis-broker instance on :6379.
CHECKPOINTER_REDIS_URL = os.environ.get("CHATBOT_CHECKPOINTER_REDIS_URL", "redis://localhost:6380/0")


def get_checkpointer() -> RedisSaver:
    """Returns the process-wide RedisSaver, building + `.setup()`-ing it on
    first call. Raises (does not swallow) if Redis is unreachable — a broken
    checkpointer means no conversation can be tracked at all, so failing loud
    and immediately at startup is correct here, unlike call_slm's
    per-request soft-fail."""
    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        with _LOCK:
            if _CHECKPOINTER is None:
                logger.info("get_checkpointer: connecting to %s", CHECKPOINTER_REDIS_URL)
                try:
                    saver = RedisSaver(redis_url=CHECKPOINTER_REDIS_URL)
                    saver.setup()   # idempotent: creates the redis search indices on first run
                except Exception:
                    logger.exception(
                        "get_checkpointer: could not initialize RedisSaver at %s "
                        "(is redis-stack-server running? see module docstring)",
                        CHECKPOINTER_REDIS_URL,
                    )
                    raise
                _CHECKPOINTER = saver
    return _CHECKPOINTER
