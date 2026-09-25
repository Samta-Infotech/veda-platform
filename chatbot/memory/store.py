"""chatbot.memory.store — Redis I/O for the structured analytical memory
(§4 of docs/MEMORY_ARCHITECTURE.md).

Deliberately a SEPARATE Redis connection/keyspace from chatbot/checkpointer.py's
RedisSaver — that one persists LangGraph's own execution-checkpoint state
(an engine concern); this one persists analytical memory (a domain concern).
Both may point at the same physical redis-stack instance (they do, by
default) without coupling: different key prefixes, independent TTLs, and this
store never touches LangGraph's own checkpoint keys.

Keys (per docs/MEMORY_ARCHITECTURE.md §4):
    veda:mem:{tenant}:{session}:src:{source}:frame  STRING (JSON) — that source's QueryFrame
    veda:mem:{tenant}:{session}:src:{source}:stack  LIST  (JSON per element) — its DrillStack
    veda:mem:{tenant}:{session}:src:{source}:ref    STRING (JSON) — the previous answer's row
                                                    identities (chatbot/memory/reference.py)
    veda:mem:{tenant}:{session}:active              STRING — id of the last answered source
    veda:mem:{tenant}:{session}:sources             SET    — every source this session has used
    veda:mem:{tenant}:{session}:episodic            LIST  (JSON per element) — capped, short
    veda:mem:{tenant}:{session}:topics              STRING (JSON list) — the bounded topic
                                                    index, one snapshot per (source, entity),
                                                    most recent first (chatbot/memory/topics.py)
    veda:mem:{tenant}:{session}:frame|stack         the PRE-2026-09-18 unscoped keys, still read

FRAME AND STACK ARE PER SOURCE; EPISODIC IS NOT. A frame is evidence harvested from one
source's executed SQL, so one session working across two sources holds two of them — with
a single session-wide key the newer source's frame overwrote the older one and the first
topic was simply gone (live-tested, 2026-09-17). The episodic buffer is the CONVERSATION,
which has one thread regardless of how many sources answered it, so it stays session-wide.

`source_id=None` on any method means "the unscoped key", which is exactly what every
caller used before source scoping existed — a deployment that never sets a source, and
the CLI, behave byte-identically to before. A scoped read falls back to the unscoped key
when the scoped one is missing, so a session that was mid-conversation across the deploy
keeps its memory; the legacy key is never written again and ages out on its own TTL.

All TTL'd with a sliding idle window, refreshed on every read AND write — an
expired/missing frame is treated identically to "no memory yet" (turn 1),
never as corruption (see chatbot/nodes.py::memory_read_node).
All TTL'd with a sliding idle window, refreshed on every read AND write — an
expired/missing frame is treated identically to "no memory yet" (turn 1),
never as corruption (see chatbot/nodes.py::memory_read_node).
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import redis

logger = logging.getLogger(__name__)

_MEMORY_REDIS_URL = os.environ.get(
    "CHATBOT_MEMORY_REDIS_URL",
    os.environ.get("CHATBOT_CHECKPOINTER_REDIS_URL", "redis://localhost:6380/0"),
)
# Sliding IDLE window, refreshed on every read and write — an active conversation
# never expires, however long it runs. Raised from 4h to 7 days (2026-09-17) to match
# chatbot/checkpointer.py's own window: at 4h a user who asked something in the morning
# and came back after lunch said "only the ones in Nagpur" to a frame that had already
# expired, and the follow-up reached the engine with no context at all. The checkpoint
# (the conversation the frame describes) survived 7 days, so the frame was the shorter
# of the two and the one that broke the turn.
_TTL_SECS = int(os.environ.get("VEDA_MEMORY_TTL_SECS", str(7 * 24 * 3600)))
_EPISODIC_MAX = 3
_STACK_MAX = 10

# Per-session turn lock. One chat's turns are causally ordered by definition — turn N+1
# is a follow-up to turn N — so running two of them at once is not a workload to support,
# it is a race to prevent. Measured 2026-09-18, 4 concurrent turns on one session:
# history kept 2 of the 8 entries and the frame reached version 1 instead of 4, silently.
# Six turns of conversation were simply gone, with nothing anywhere saying so.
#
# Serialization, not optimistic retry: LangGraph's checkpoint carries no version to retry
# against, so a retry could only ever repair the frame (which already has an optimistic
# lock and already aborts correctly) and never the history the checkpointer lost.
#
# LEASE must outlive the slowest realistic turn — a federated engine call runs into the
# minutes — because a lease that expires mid-turn hands the lock to a waiter and puts
# both turns back in the race. WAIT is bounded so a dead holder cannot wedge a chat
# forever: past it the turn proceeds anyway, which is the pre-lock behaviour and is
# strictly better than refusing the user's message.
_LOCK_LEASE_SECS = int(os.environ.get("VEDA_TURN_LOCK_LEASE_SECS", "600"))
_LOCK_WAIT_SECS = float(os.environ.get("VEDA_TURN_LOCK_WAIT_SECS", "120"))
_LOCK_POLL_SECS = 0.05

# Release only a lock we still own: after a lease expiry the key may already belong to
# the next turn, and a blind DELETE would hand a third turn into the race with it.
_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
end
return 0
"""

_CLIENT = None
_LOCK = threading.Lock()


def _client() -> "redis.Redis":
    global _CLIENT
    if _CLIENT is None:
        with _LOCK:
            if _CLIENT is None:
                _CLIENT = redis.Redis.from_url(_MEMORY_REDIS_URL, decode_responses=True)
    return _CLIENT


def _k(tenant: str, session_id: str, suffix: str, source_id: Optional[Any] = None) -> str:
    """Key for one memory slot. `source_id=None` returns the pre-source-scoping key,
    which is what the unscoped callers (CLI, any deployment with no source set) still
    use and what the backward-compatible read path falls back to."""
    base = f"veda:mem:{tenant}:{session_id}"
    if source_id is None:
        return f"{base}:{suffix}"
    return f"{base}:src:{source_id}:{suffix}"


def _user_sessions_key(tenant: str, user_id: Any) -> str:
    """Per USER, tenant-scoped: summaries of the user's recent sessions."""
    return f"veda:mem:{tenant}:user:{user_id}:sessions"


def _comparison_key(tenant: str, session_id: str) -> str:
    """SESSION level, deliberately not under a source. A comparison can span two
    sources; filing it under one would let that side silently own the other — the same
    mistake the single session-wide frame made before source scoping."""
    return f"veda:mem:{tenant}:{session_id}:comparison"


def _active_key(tenant: str, session_id: str) -> str:
    return f"veda:mem:{tenant}:{session_id}:active"


def _topics_key(tenant: str, session_id: str) -> str:
    """SESSION level, like the comparison: the topic index is the list of places this
    conversation has been, across every source it has used. Each entry records its own
    source, so a read can drop what the caller may no longer see."""
    return f"veda:mem:{tenant}:{session_id}:topics"


def _sources_key(tenant: str, session_id: str) -> str:
    """A SET of every source this session has written a frame for. Kept so `reset` can
    delete them all without SCANning the keyspace, and so a caller can ask which topics
    a session is holding without knowing the scope up front."""
    return f"veda:mem:{tenant}:{session_id}:sources"


@contextlib.contextmanager
def session_turn_lock(tenant: str, session_id: str,
                      wait_secs: Optional[float] = None,
                      lease_secs: Optional[int] = None):
    """Serialize the turns of ONE chat session. See _LOCK_LEASE_SECS for why.

    Degrades to no locking on any Redis failure — an unreachable Redis must not make the
    chat endpoint unavailable, and running unserialized is exactly what every turn did
    before this existed. Yields True when the lock was actually held, False when the turn
    is proceeding without it, so the caller can say which happened.
    """
    if not session_id:
        yield False
        return
    key = f"veda:lock:turn:{tenant}:{session_id}"
    token = uuid.uuid4().hex
    wait = _LOCK_WAIT_SECS if wait_secs is None else wait_secs
    lease = _LOCK_LEASE_SECS if lease_secs is None else lease_secs
    acquired = False
    client = None
    try:
        client = _client()
        deadline = time.monotonic() + wait
        delay = _LOCK_POLL_SECS
        while True:
            if client.set(key, token, nx=True, px=int(lease * 1000)):
                acquired = True
                break
            if time.monotonic() >= deadline:
                logger.warning(
                    "session_turn_lock: session=%s still busy after %.0fs — running this "
                    "turn WITHOUT the lock rather than refusing the user's message; its "
                    "history/frame writes may be lost to the concurrent turn", session_id, wait)
                break
            time.sleep(delay)
            delay = min(delay * 1.5, 0.5)      # back off, but stay responsive
    except Exception:
        logger.warning("session_turn_lock: unavailable for session=%s — proceeding "
                       "unserialized", session_id, exc_info=True)
    try:
        yield acquired
    finally:
        if acquired and client is not None:
            try:
                client.eval(_RELEASE_LUA, 1, key, token)
            except Exception:
                logger.warning("session_turn_lock: release failed for session=%s (the "
                               "lease expires on its own)", session_id, exc_info=True)


def _authorised_episodic(entries: List[Dict[str, Any]],
                         authorised_source_ids: Optional[Any]) -> List[Dict[str, Any]]:
    """Drop episodic entries whose source the caller is no longer entitled to.

    The frame has had this check since source pinning (nodes.py::_frame_still_authorised);
    the episodic buffer did not, because it is session-wide and carried no source at all.
    That left a real seam: an entry's gist can hold a value read out of the data
    ("answered: active_users=5,857"), and it survived a revoked grant for as long as the
    buffer lived — seven days.

    Same fail-open rules as the frame guard, for the same reasons:
      · entry with NO source_id  -> kept. Written before this field existed (the buffer
        lives 7 days, so those entries are still in flight), or by a path that carries
        no source. Nothing to compare is not a denial.
      · caller with NO resolved scope (None) -> everything kept. The CLI and other
        non-HTTP callers resolve no scope; this is not an empty scope.
      · caller with an EMPTY but resolved scope -> everything scoped is dropped. An
        empty grant list is a decision, not an absence — the last grant withdrawn is
        the likeliest revocation shape.

    Pairs are NOT re-paired here: read_episodic returns a flat [user, assistant, ...]
    list and both halves of a turn carry the same source_id, so a dropped turn loses
    both halves together.
    """
    if authorised_source_ids is None:
        return entries
    try:
        allowed = {str(s) for s in authorised_source_ids}
    except TypeError:
        logger.warning("_authorised_episodic: unreadable scope %r — keeping nothing "
                       "scoped", authorised_source_ids)
        allowed = set()
    kept, dropped = [], 0
    for e in entries:
        src = e.get("source_id")
        if src is None or str(src) in allowed:
            kept.append(e)
        else:
            dropped += 1
    if dropped:
        logger.warning("_authorised_episodic: dropped %d episodic entr%s from sources "
                       "outside the caller's scope %r", dropped,
                       "y" if dropped == 1 else "ies", sorted(allowed))
    return kept


class MemoryStore:
    """Stateless facade — every method is a plain Redis call. Any transport
    error degrades to "no memory" (returns None/[]) rather than raising —
    memory is a latency/quality optimization, never a hard dependency the
    whole turn should fail on (mirrors call_slm()'s own per-request soft-fail
    philosophy, not chatbot/checkpointer.py's fail-loud-at-startup one, since
    THIS store is a per-turn optimization, not the graph's own execution
    state)."""

    @staticmethod
    def active_source(tenant: str, session_id: str) -> Optional[str]:
        """The source whose frame this session last wrote — i.e. the topic the user was
        on. Read when a turn names no source of its own; a turn that DOES name one wants
        that source's frame, not this one."""
        try:
            return _client().get(_active_key(tenant, session_id)) or None
        except Exception:
            logger.warning("MemoryStore.active_source failed for session=%s", session_id,
                           exc_info=True)
            return None

    @staticmethod
    def known_sources(tenant: str, session_id: str) -> List[str]:
        """Every source this session holds a frame for."""
        try:
            return sorted(_client().smembers(_sources_key(tenant, session_id)) or [])
        except Exception:
            logger.warning("MemoryStore.known_sources failed for session=%s", session_id,
                           exc_info=True)
            return []

    @staticmethod
    def read_frame(tenant: str, session_id: str,
                    source_id: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """This source's frame. With no source_id, the session's ACTIVE source's frame —
        so a caller that does not track sources still reads the topic the user is on.

        Falls back to the pre-source-scoping key when the scoped one holds nothing, so a
        conversation that was live across the deploy keeps its context. The fallback is
        read-only: the next write lands on the scoped key and the legacy one expires.
        """
        try:
            c = _client()
            if source_id is None:
                source_id = MemoryStore.active_source(tenant, session_id)
            for candidate in ([source_id, None] if source_id is not None else [None]):
                key = _k(tenant, session_id, "frame", candidate)
                raw = c.get(key)
                if not raw:
                    continue
                frame = json.loads(raw)
                if candidate is None and source_id is not None:
                    # The legacy key is session-wide, so it may hold a DIFFERENT source's
                    # topic than the one being asked for. Handing that back would
                    # reintroduce exactly the cross-source bleed this scoping removes, so
                    # the fallback only answers when it does not contradict the request.
                    remembered = frame.get("source_id")
                    if remembered is not None and str(remembered) != str(source_id):
                        continue
                c.expire(key, _TTL_SECS)
                return frame
            return None
        except Exception:
            logger.warning("MemoryStore.read_frame failed for session=%s", session_id, exc_info=True)
            return None

    @staticmethod
    def write_frame(tenant: str, session_id: str, frame: Dict[str, Any],
                     expected_version: Optional[int] = None,
                     source_id: Optional[Any] = None) -> bool:
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
        key = _k(tenant, session_id, "frame", source_id)
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
            if source_id is not None:
                # Written AFTER the frame, and never inside the optimistic-lock
                # transaction: a pointer to a frame that did not commit would send the
                # next turn to an empty key. An aborted write returns above, before here.
                with c.pipeline() as pipe:
                    pipe.set(_active_key(tenant, session_id), str(source_id), ex=_TTL_SECS)
                    pipe.sadd(_sources_key(tenant, session_id), str(source_id))
                    pipe.expire(_sources_key(tenant, session_id), _TTL_SECS)
                    pipe.execute()
            return True
        except Exception:
            logger.warning("MemoryStore.write_frame failed for session=%s", session_id, exc_info=True)
            return False

    @staticmethod
    def read_stack(tenant: str, session_id: str,
                    source_id: Optional[Any] = None) -> List[Dict[str, Any]]:
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
            c = _client()
            if source_id is None:
                source_id = MemoryStore.active_source(tenant, session_id)
            # Same legacy fallback as read_frame, and for the same reason: the stack
            # describes the frame, so the two must come from the same generation of key.
            for candidate in ([source_id, None] if source_id is not None else [None]):
                key = _k(tenant, session_id, "stack", candidate)
                raw = c.lrange(key, 0, _STACK_MAX - 1)
                if raw:
                    c.expire(key, _TTL_SECS)
                    return [json.loads(x) for x in raw]
            return []
        except Exception:
            logger.warning("MemoryStore.read_stack failed for session=%s", session_id, exc_info=True)
            return []

    @staticmethod
    def write_stack(tenant: str, session_id: str, stack: List[Dict[str, Any]],
                     source_id: Optional[Any] = None) -> None:
        try:
            key = _k(tenant, session_id, "stack", source_id)
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
    def read_reference(tenant: str, session_id: str,
                       source_id: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """The previous answer's row identities (chatbot/memory/reference.py), or None.
        Per source, like the frame it belongs to. Unreadable memory is "no memory"."""
        try:
            key = _k(tenant, session_id, "ref", source_id)
            raw = _client().get(key)
            if not raw:
                return None
            _client().expire(key, _TTL_SECS)
            out = json.loads(raw)
            return out if isinstance(out, dict) else None
        except Exception:
            logger.warning("MemoryStore.read_reference failed for session=%s",
                           session_id, exc_info=True)
            return None

    @staticmethod
    def write_reference(tenant: str, session_id: str, reference: Optional[Dict[str, Any]],
                        source_id: Optional[Any] = None) -> None:
        """Write, or clear with None. An answered turn whose rows are not referable CLEARS
        the slot: an ordinal always means a row of the answer the user is looking at now,
        never of one before it."""
        try:
            key = _k(tenant, session_id, "ref", source_id)
            if not reference:
                _client().delete(key)
                return
            _client().set(key, json.dumps(reference, default=str), ex=_TTL_SECS)
        except Exception:
            logger.warning("MemoryStore.write_reference failed for session=%s",
                           session_id, exc_info=True)

    @staticmethod
    def read_topics(tenant: str, session_id: str) -> List[Dict[str, Any]]:
        """The WHOLE topic index, most recent first (chatbot/memory/topics.py).

        Deliberately unfiltered: memory_read_node filters it to the current turn's grants
        with the frame's own guard (nodes.py::_frame_still_authorised) before anything
        else sees it, while memory_write_node and the per-source reset below re-write the
        list — and writing back a filtered view would silently delete the topics of
        sources that are merely outside THIS turn's scope rather than revoked. Unreadable
        memory is "no memory"."""
        try:
            key = _topics_key(tenant, session_id)
            raw = _client().get(key)
            if not raw:
                return []
            _client().expire(key, _TTL_SECS)
            out = json.loads(raw)
            return [e for e in out if isinstance(e, dict)] if isinstance(out, list) else []
        except Exception:
            logger.warning("MemoryStore.read_topics failed for session=%s", session_id,
                           exc_info=True)
            return []

    @staticmethod
    def write_topics(tenant: str, session_id: str,
                     topics: Optional[List[Dict[str, Any]]]) -> None:
        """Write the whole index, or clear it with an empty list/None."""
        try:
            key = _topics_key(tenant, session_id)
            if not topics:
                _client().delete(key)
                return
            _client().set(key, json.dumps(topics, default=str), ex=_TTL_SECS)
        except Exception:
            logger.warning("MemoryStore.write_topics failed for session=%s", session_id,
                           exc_info=True)

    @staticmethod
    def read_user_sessions(tenant: str, user_id: Optional[Any]) -> List[Dict[str, Any]]:
        """The user's recent session summaries (chatbot/memory/topics.py::session_summary),
        most recent first. Per USER, not per session — the one memory that crosses chats.
        Unreadable memory is "no memory"."""
        if user_id in (None, ""):
            return []
        try:
            key = _user_sessions_key(tenant, user_id)
            raw = _client().get(key)
            if not raw:
                return []
            _client().expire(key, _TTL_SECS)
            out = json.loads(raw)
            return [s for s in out if isinstance(s, dict)] if isinstance(out, list) else []
        except Exception:
            logger.warning("MemoryStore.read_user_sessions failed for user=%s", user_id,
                           exc_info=True)
            return []

    @staticmethod
    def write_user_sessions(tenant: str, user_id: Optional[Any],
                            sessions: List[Dict[str, Any]]) -> None:
        if user_id in (None, ""):
            return
        try:
            key = _user_sessions_key(tenant, user_id)
            if not sessions:
                _client().delete(key)
                return
            _client().set(key, json.dumps(sessions, default=str), ex=_TTL_SECS)
        except Exception:
            logger.warning("MemoryStore.write_user_sessions failed for user=%s", user_id,
                           exc_info=True)

    @staticmethod
    def read_comparison(tenant: str, session_id: str) -> Optional[Dict[str, Any]]:
        """The active comparison, or None. Unreadable memory means "no memory", never a
        failed turn — the same degradation every read here uses."""
        try:
            raw = _client().get(_comparison_key(tenant, session_id))
            if not raw:
                return None
            _client().expire(_comparison_key(tenant, session_id), _TTL_SECS)
            out = json.loads(raw)
            return out if isinstance(out, dict) else None
        except Exception:
            logger.warning("MemoryStore.read_comparison failed for session=%s",
                           session_id, exc_info=True)
            return None

    @staticmethod
    def write_comparison(tenant: str, session_id: str,
                         comparison: Optional[Dict[str, Any]]) -> None:
        """Write or clear the session's comparison. Passing None clears it, so a caller
        that has decided the comparison is stale does not need a second method."""
        try:
            key = _comparison_key(tenant, session_id)
            if not comparison:
                _client().delete(key)
                return
            _client().set(key, json.dumps(comparison, default=str), ex=_TTL_SECS)
        except Exception:
            logger.warning("MemoryStore.write_comparison failed for session=%s",
                           session_id, exc_info=True)

    @staticmethod
    def read_episodic(tenant: str, session_id: str,
                      authorised_source_ids: Optional[Any] = None) -> List[Dict[str, str]]:
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
            entries = [json.loads(x) for x in ordered]
            return _authorised_episodic(entries, authorised_source_ids)
        except Exception:
            logger.warning("MemoryStore.read_episodic failed for session=%s", session_id, exc_info=True)
            return []

    @staticmethod
    def push_episodic_turn(tenant: str, session_id: str, user_message: str,
                           assistant_gist: str, source_id: Optional[Any] = None) -> None:
        """Pushes the [user, assistant] pair for this turn and trims to the
        cap. `assistant_gist` MUST be a short templated string (e.g.
        "answered: active_users=5,857"), never the full markdown/table reply
        — see docs/MEMORY_ARCHITECTURE.md §4/§17."""
        try:
            key = _k(tenant, session_id, "episodic")
            c = _client()
            with c.pipeline() as pipe:
                # source_id travels WITH the entry. The buffer stays session-wide — the
                # conversation is one thread and that is deliberate — but each entry now
                # records which source produced it, so a read can drop what the caller
                # is no longer entitled to without destroying the thread.
                _src = None if source_id is None else str(source_id)
                pipe.lpush(key, json.dumps({"role": "assistant", "content": assistant_gist,
                                            "source_id": _src}, default=str))
                pipe.lpush(key, json.dumps({"role": "user", "content": user_message,
                                            "source_id": _src}, default=str))
                pipe.ltrim(key, 0, _EPISODIC_MAX * 2 - 1)
                pipe.expire(key, _TTL_SECS)
                pipe.execute()
        except Exception:
            logger.warning("MemoryStore.push_episodic_turn failed for session=%s", session_id, exc_info=True)

    @staticmethod
    def reset(tenant: str, session_id: str, source_id: Optional[Any] = None) -> None:
        """Explicit wipe — used when a hard "start over" is detected (deterministic
        fast path, mirrors chatbot/nodes.py's _GREETING_RE-style instant matches), and
        when a remembered source turns out to be outside this turn's authorised scope.

        `source_id` wipes ONE source's frame and stack, leaving the rest of the session
        intact. That is what an access revocation calls for: losing the grant on one
        source is not a reason to throw away the user's work on another, and the whole
        point of scoping the keys is that this is now expressible. The conversation
        buffer is session-wide, so a scoped reset does not touch it.

        With no `source_id` everything goes — every source's frame and stack (the
        `sources` SET is what makes that possible without SCANning), the pointers, the
        episodic buffer, and the pre-source-scoping keys.
        """
        try:
            c = _client()
            if source_id is not None:
                c.delete(_k(tenant, session_id, "frame", source_id),
                         _k(tenant, session_id, "stack", source_id),
                         _k(tenant, session_id, "ref", source_id))
                c.srem(_sources_key(tenant, session_id), str(source_id))
                if (c.get(_active_key(tenant, session_id)) or "") == str(source_id):
                    c.delete(_active_key(tenant, session_id))
                # That source's topics go with its frame; other sources' topics stay —
                # the same precision the per-source frame/stack delete above has.
                from .topics import without_source
                _topics = MemoryStore.read_topics(tenant, session_id)
                _kept = without_source(_topics, source_id)
                if len(_kept) != len(_topics):
                    MemoryStore.write_topics(tenant, session_id, _kept)
                return
            keys = [_k(tenant, session_id, "frame"),
                    _k(tenant, session_id, "stack"),
                    _k(tenant, session_id, "ref"),
                    _k(tenant, session_id, "episodic"),
                    _active_key(tenant, session_id),
                    _sources_key(tenant, session_id),
                    # Session-level, so it is cleared by the whole-session reset only —
                    # a per-source reset must not drop a comparison whose other side
                    # lives in a source the user did not reset.
                    _comparison_key(tenant, session_id),
                    # Session-level too: "start over" forgets every earlier topic.
                    _topics_key(tenant, session_id)]
            for known in MemoryStore.known_sources(tenant, session_id):
                keys.append(_k(tenant, session_id, "frame", known))
                keys.append(_k(tenant, session_id, "stack", known))
                keys.append(_k(tenant, session_id, "ref", known))
            c.delete(*keys)
        except Exception:
            logger.warning("MemoryStore.reset failed for session=%s", session_id, exc_info=True)
