"""An in-memory stand-in for the slice of redis-py that chatbot/memory/store.py uses.

fakeredis is not installed in the images, and the memory tests that need Redis otherwise
run against the live redis-stack (tests/test_memory_source_scope.py). This lets a test
drive the REAL MemoryStore code — key layout, optimistic lock, active pointer — with no
server: install it with `install(monkeypatch)`, which swaps store._client() for a fresh
instance and returns it.

Covers exactly: get/set(ex, nx, px)/expire/delete, lpush/lrange/ltrim, sadd/smembers/srem,
eval (the lock-release script only), and pipeline() with watch/unwatch/multi/execute.
"""
from __future__ import annotations

from typing import Any, Dict, List


class WatchError(Exception):
    pass


class FakeRedis:
    def __init__(self):
        self.kv: Dict[str, Any] = {}

    # ── strings ──────────────────────────────────────────────────────────────
    def get(self, key):
        v = self.kv.get(key)
        return v if isinstance(v, str) else None

    def set(self, key, value, ex=None, nx=False, px=None):
        if nx and key in self.kv:
            return None
        self.kv[key] = str(value)
        return True

    def expire(self, key, secs):
        return key in self.kv

    def delete(self, *keys):
        n = 0
        for k in keys:
            n += 1 if self.kv.pop(k, None) is not None else 0
        return n

    # ── lists ────────────────────────────────────────────────────────────────
    def lpush(self, key, *values):
        lst: List[str] = self.kv.setdefault(key, [])
        for v in values:
            lst.insert(0, str(v))
        return len(lst)

    def lrange(self, key, start, end):
        lst = self.kv.get(key) or []
        end = len(lst) - 1 if end == -1 else end
        return list(lst[start:end + 1])

    def ltrim(self, key, start, end):
        if key in self.kv:
            self.kv[key] = self.lrange(key, start, end)
        return True

    # ── sets ─────────────────────────────────────────────────────────────────
    def sadd(self, key, *values):
        s = self.kv.setdefault(key, set())
        s.update(str(v) for v in values)
        return len(values)

    def smembers(self, key):
        return set(self.kv.get(key) or set())

    def srem(self, key, *values):
        s = self.kv.get(key) or set()
        for v in values:
            s.discard(str(v))
        return len(values)

    # ── scripting (session_turn_lock's compare-and-delete only) ──────────────
    def eval(self, script, numkeys, key, token):
        if self.get(key) == token:
            return self.delete(key)
        return 0

    def ping(self):
        return True

    def pipeline(self):
        return _Pipeline(self)


class _Pipeline:
    """Immediate reads, queued writes after multi(); a pipeline that never calls multi()
    queues every call (redis-py's transaction=True default behaves the same for writes)."""

    def __init__(self, r: FakeRedis):
        self.r = r
        self.ops: List[tuple] = []
        self.watching = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.ops = []
        return False

    def watch(self, *keys):
        self.watching = True

    def unwatch(self):
        self.watching = False

    def multi(self):
        self.watching = False

    def __getattr__(self, name):
        target = getattr(self.r, name)
        if self.watching and name in ("get", "lrange", "smembers"):
            return target

        def queued(*a, **k):
            self.ops.append((target, a, k))
            return self
        return queued

    def execute(self):
        out = [fn(*a, **k) for fn, a, k in self.ops]
        self.ops = []
        return out


def install(monkeypatch) -> FakeRedis:
    """Point chatbot.memory.store at a fresh in-memory Redis for this test."""
    from chatbot.memory import store
    fake = FakeRedis()
    monkeypatch.setattr(store, "_CLIENT", fake)
    monkeypatch.setattr(store, "_client", lambda: fake)
    return fake
