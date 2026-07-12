"""chatbot.memory.store — Redis I/O for the structured analytical memory
(§4 of docs/MEMORY_ARCHITECTURE.md).

Deliberately a SEPARATE Redis connection/keyspace from chatbot/checkpointer.py's
RedisSaver — that one persists LangGraph's own execution-checkpoint state
(an engine concern); this one persists analytical memory (a domain concern).
Both may point at the same physical redis-stack instance (they do, by
default) without coupling: different key prefixes, independent TTLs, and this
store never touches LangGraph's own checkpoint keys.

Keys (per docs/MEMORY_ARCHITECTURE.md §4):
    veda:mem:{tenant}:{session_id}:frame     STRING (JSON) — current QueryFrame
    veda:mem:{tenant}:{session_id}:stack     LIST (JSON per element) — DrillStack
    veda:mem:{tenant}:{session_id}:episodic  LIST (JSON per element) — capped, short
All TTL'd with a sliding idle window, refreshed on every read AND write — an
expired/missing frame is treated identically to "no memory yet" (turn 1),
never as corruption (see chatbot/nodes.py::memory_read_node).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import redis

logger = logging.getLogger(__name__)

_MEMORY_REDIS_URL = os.environ.get(
    "CHATBOT_MEMORY_REDIS_URL",
    os.environ.get("CHATBOT_CHECKPOINTER_REDIS_URL", "redis://localhost:6380/0"),
)
_TTL_SECS = int(os.environ.get("VEDA_MEMORY_TTL_SECS", str(4 * 3600)))   # sliding 4h idle window
_EPISODIC_MAX = 3
_STACK_MAX = 10

_CLIENT = None
_LOCK = threading.Lock()


def _client() -> "redis.Redis":
    global _CLIENT
    if _CLIENT is None:
        with _LOCK:
            if _CLIENT is None:
                _CLIENT = redis.Redis.from_url(_MEMORY_REDIS_URL, decode_responses=True)
    return _CLIENT


def _k(tenant: str, session_id: str, suffix: str) -> str:
    return f"veda:mem:{tenant}:{session_id}:{suffix}"


class MemoryStore:
    """Stateless facade — every method is a plain Redis call. Any transport
    error degrades to "no memory" (returns None/[]) rather than raising —
    memory is a latency/quality optimization, never a hard dependency the
    whole turn should fail on (mirrors call_slm()'s own per-request soft-fail
    philosophy, not chatbot/checkpointer.py's fail-loud-at-startup one, since
    THIS store is a per-turn optimization, not the graph's own execution
    state)."""

    @staticmethod
    def read_frame(tenant: str, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            raw = _client().get(_k(tenant, session_id, "frame"))
            if not raw:
                return None
            _client().expire(_k(tenant, session_id, "frame"), _TTL_SECS)
            return json.loads(raw)
        except Exception:
            logger.warning("MemoryStore.read_frame failed for session=%s", session_id, exc_info=True)
            return None

    @staticmethod
    def write_frame(tenant: str, session_id: str, frame: Dict[str, Any],
                     expected_version: Optional[int] = None) -> bool:
        """Optimistic-lock write: WATCH the key, verify the stored version
        still matches `expected_version` before committing. FIXED (audit C3):
        a version conflict now ABORTS the write (returns False) instead of
        proceeding anyway — the old code detected the conflict, logged it,
        and clobbered the fresher value with a merge computed against stale
        data, which made the "lock" purely cosmetic. Aborting means the
        caller's frame (built from a prev_frame that's no longer current)
        is discarded rather than silently overwriting newer evidence; the
        turn's own reply to the user is completely unaffected either way —
        only the NEXT turn's memory read is (correctly) whatever the
        winning concurrent writer left behind."""
        key = _k(tenant, session_id, "frame")
        try:
            c = _client()
            if expected_version is not None:
                with c.pipeline() as pipe:
                    while True:
                        try:
                            pipe.watch(key)
                            current_raw = pipe.get(key)
                            current_version = json.loads(current_raw).get("version") if current_raw else 0
                            if current_version != expected_version:
                                pipe.unwatch()
                                logger.warning(
                                    "MemoryStore.write_frame version conflict session=%s "
                                    "expected=%s actual=%s — ABORTING this write (last-committed "
                                    "frame wins; this turn's own reply to the user is unaffected)",
                                    session_id, expected_version, current_version,
                                )
                                return False
                            pipe.multi()
                            pipe.set(key, json.dumps(frame, default=str), ex=_TTL_SECS)
                            pipe.execute()
                            break
                        except redis.WatchError:
                            continue
            else:
                c.set(key, json.dumps(frame, default=str), ex=_TTL_SECS)
            return True
        except Exception:
            logger.warning("MemoryStore.write_frame failed for session=%s", session_id, exc_info=True)
            return False

    @staticmethod
    def read_stack(tenant: str, session_id: str) -> List[Dict[str, Any]]:
        """FIXED (audit C1): write_stack() pushes `reversed(stack)` via LPUSH,
        and LPUSH itself reverses on insertion — net effect, LRANGE(0,-1)
        ALREADY returns the list in the correct oldest-first order (verified
        by hand-tracing the push sequence). The previous code applied an
        extra `reversed()` here on top of that, silently re-inverting the
        stack to newest-first on every read — which made pop_drill() (which
        assumes oldest-first, dropping the LAST/most-specific element) strip
        the OLDEST/outermost drill level instead on "go back." No reversal
        needed on the read side at all."""
        try:
            key = _k(tenant, session_id, "stack")
            raw = _client().lrange(key, 0, _STACK_MAX - 1)
            if raw:
                _client().expire(key, _TTL_SECS)
            return [json.loads(x) for x in raw]
        except Exception:
            logger.warning("MemoryStore.read_stack failed for session=%s", session_id, exc_info=True)
            return []

    @staticmethod
    def write_stack(tenant: str, session_id: str, stack: List[Dict[str, Any]]) -> None:
        try:
            key = _k(tenant, session_id, "stack")
            c = _client()
            with c.pipeline() as pipe:
                pipe.delete(key)
                if stack:
                    # `stack` is oldest-first (index -1 = newest/most specific level, the
                    # convention push_drill/pop_drill use). LPUSH inserts at the head, so
                    # pushing in REVERSED order here means the LAST command queued
                    # (LPUSH(stack[0])) ends up executed last and lands at the head —
                    # i.e. the Redis-side list, read head-to-tail via LRANGE(0,-1), comes
                    # back out in the SAME oldest-first order as `stack` itself. No
                    # reversal needed on the read side (see read_stack — this used to be
                    # reversed there too, a confirmed bug: audit C1).
                    for level in reversed(stack[-_STACK_MAX:]):
                        pipe.lpush(key, json.dumps(level, default=str))
                    pipe.ltrim(key, 0, _STACK_MAX - 1)
                    pipe.expire(key, _TTL_SECS)
                pipe.execute()
        except Exception:
            logger.warning("MemoryStore.write_stack failed for session=%s", session_id, exc_info=True)

    @staticmethod
    def read_episodic(tenant: str, session_id: str) -> List[Dict[str, str]]:
        """FIXED (found while wiring this buffer into classify_delta — same
        class of bug as audit C1): push_episodic_turn() LPUSHes the
        assistant message then the user message (in that order) for EACH
        turn, so the stored Redis list is newest-turn-first with the
        [user, assistant] pair's own internal order intact — e.g.
        [user_N, assistant_N, user_{N-1}, assistant_{N-1}, ...]. The
        previous code reversed that ENTIRE flat list element-by-element,
        which correctly flips turn order to oldest-first but ALSO flips
        each turn's internal order to [assistant, user] — backwards. Fix:
        reverse by PAIR (chunks of 2), preserving each turn's own
        [user, assistant] order while still returning oldest-turn-first
        overall."""
        try:
            key = _k(tenant, session_id, "episodic")
            raw = _client().lrange(key, 0, _EPISODIC_MAX * 2 - 1)   # user+assistant per turn
            if raw:
                _client().expire(key, _TTL_SECS)
            pairs = [raw[i:i + 2] for i in range(0, len(raw), 2)]
            ordered = [item for pair in reversed(pairs) for item in pair]
            return [json.loads(x) for x in ordered]
        except Exception:
            logger.warning("MemoryStore.read_episodic failed for session=%s", session_id, exc_info=True)
            return []

    @staticmethod
    def push_episodic_turn(tenant: str, session_id: str, user_message: str, assistant_gist: str) -> None:
        """Pushes the [user, assistant] pair for this turn and trims to the
        cap. `assistant_gist` MUST be a short templated string (e.g.
        "answered: active_users=5,857"), never the full markdown/table reply
        — see docs/MEMORY_ARCHITECTURE.md §4/§17."""
        try:
            key = _k(tenant, session_id, "episodic")
            c = _client()
            with c.pipeline() as pipe:
                pipe.lpush(key, json.dumps({"role": "assistant", "content": assistant_gist}, default=str))
                pipe.lpush(key, json.dumps({"role": "user", "content": user_message}, default=str))
                pipe.ltrim(key, 0, _EPISODIC_MAX * 2 - 1)
                pipe.expire(key, _TTL_SECS)
                pipe.execute()
        except Exception:
            logger.warning("MemoryStore.push_episodic_turn failed for session=%s", session_id, exc_info=True)

    @staticmethod
    def reset(tenant: str, session_id: str) -> None:
        """Explicit wipe — used when a hard "start over" is detected (deterministic
        fast path, mirrors chatbot/nodes.py's _GREETING_RE-style instant matches)."""
        try:
            c = _client()
            c.delete(_k(tenant, session_id, "frame"),
                     _k(tenant, session_id, "stack"),
                     _k(tenant, session_id, "episodic"))
        except Exception:
            logger.warning("MemoryStore.reset failed for session=%s", session_id, exc_info=True)
