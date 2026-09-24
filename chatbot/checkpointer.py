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

# Checkpoints used to be written with NO expiry at all. Measured on this machine
# (2026-09-16): 7,281 `checkpoint:*` + 38,484 `checkpoint_write:*` keys, 172 MB,
# every one of them TTL -1 — the set only ever grows, one conversation's worth per
# turn, forever. The instance runs `maxmemory 0` with `noeviction`, so nothing
# reclaims them: Redis grows until the host is out of RAM and then fails EVERY
# write at once, taking all live conversations down together rather than
# degrading. A bounded window is the fix — a conversation older than this is no
# longer resumable anyway, and the chat TRANSCRIPT is not lost with it (that lives
# in Postgres, apps/chat/models.py::ChatMessage); only the graph's replay state is.
# Sliding, not absolute: refresh_on_read keeps an ACTIVE conversation alive
# indefinitely, the same way chatbot/memory/store.py's own 4h idle window does.
_DEFAULT_CHECKPOINT_TTL_MINUTES = 7 * 24 * 60          # 7d idle


def _ttl_minutes() -> float:
    """Parsed defensively: an unreadable value falls back to the default rather than
    raising at IMPORT time, which took the whole process down before any handler could
    log why. A misconfigured window is an operational problem; a container that will not
    start is an outage."""
    raw = os.environ.get("CHATBOT_CHECKPOINT_TTL_MINUTES")
    if raw is None or not str(raw).strip():
        return _DEFAULT_CHECKPOINT_TTL_MINUTES
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("CHATBOT_CHECKPOINT_TTL_MINUTES=%r is not a number — using the "
                       "%d-minute default", raw, _DEFAULT_CHECKPOINT_TTL_MINUTES)
        return _DEFAULT_CHECKPOINT_TTL_MINUTES


_CHECKPOINT_TTL_MINUTES = _ttl_minutes()


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
                    # A NEGATIVE window means "no expiry" — the previous, unbounded
                    # behaviour, kept as an escape hatch. It is implemented by omitting
                    # the ttl config entirely, NOT by passing -1: langgraph-checkpoint-
                    # redis computes `int(default_ttl * 60)` and hands it straight to
                    # `EXPIRE`, and Redis DELETES a key given a non-positive TTL. Passing
                    # -1 would therefore have made every conversation amnesiac from its
                    # first turn, while looking like it had done nothing.
                    _ttl = ({"default_ttl": _CHECKPOINT_TTL_MINUTES, "refresh_on_read": True}
                            if _CHECKPOINT_TTL_MINUTES > 0 else None)
                    saver = RedisSaver(redis_url=CHECKPOINTER_REDIS_URL, ttl=_ttl)
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
