"""Tests for chatbot/checkpointer.py's checkpoint retention window.

Checkpoints were written with no expiry at all, on an instance running
`maxmemory 0` / `noeviction` — measured 2026-09-16: 45,765 checkpoint keys,
172 MB, every one TTL -1. Nothing reclaimed them, so the set only grew; the
failure mode is not a slow degradation but every live conversation breaking at
once when the host runs out of RAM. These tests pin the bounded window (and the
escape hatch back to unbounded), with no Redis needed — the saver is faked.

Run: ``pytest tests/test_chatbot_checkpointer_ttl.py``
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _reload_with(monkeypatch, env: dict):
    """Re-import the module under a given env, with RedisSaver faked out so no
    Redis (and no network) is touched — the assertion is about what the module
    ASKS the saver for, which is exactly the part that regressed."""
    import chatbot.checkpointer as C

    captured = {}

    class _FakeSaver:
        def __init__(self, redis_url=None, **kwargs):
            captured["redis_url"] = redis_url
            captured.update(kwargs)

        def setup(self):
            captured["setup_called"] = True

    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    module = importlib.reload(C)
    monkeypatch.setattr(module, "RedisSaver", _FakeSaver)
    module._CHECKPOINTER = None
    module.get_checkpointer()
    return module, captured


def test_checkpoints_get_a_bounded_default_ttl(monkeypatch):
    _, captured = _reload_with(monkeypatch, {"CHATBOT_CHECKPOINT_TTL_MINUTES": None})
    assert captured["ttl"]["default_ttl"] == 7 * 24 * 60      # 7 days, in minutes
    assert captured["setup_called"] is True


def test_the_window_slides_so_an_active_conversation_never_expires(monkeypatch):
    """Absolute expiry would cut off a conversation someone is still using."""
    _, captured = _reload_with(monkeypatch, {"CHATBOT_CHECKPOINT_TTL_MINUTES": None})
    assert captured["ttl"]["refresh_on_read"] is True


def test_the_window_is_env_overridable(monkeypatch):
    _, captured = _reload_with(monkeypatch, {"CHATBOT_CHECKPOINT_TTL_MINUTES": "90"})
    assert captured["ttl"]["default_ttl"] == 90


def test_a_negative_window_omits_the_ttl_config_entirely(monkeypatch):
    """The escape hatch back to unbounded checkpoints.

    It must be implemented by NOT passing a ttl, never by passing -1. An independent
    review traced langgraph-checkpoint-redis: it computes `int(default_ttl * 60)` and
    hands the result straight to EXPIRE, and Redis DELETES a key given a non-positive
    TTL. Passing -1 would have made every conversation amnesiac from its first turn
    while looking like it had done nothing — and the previous version of this test
    asserted only that the config dict carried -1, so it would have passed on that."""
    _, captured = _reload_with(monkeypatch, {"CHATBOT_CHECKPOINT_TTL_MINUTES": "-1"})
    assert captured["ttl"] is None


def test_a_positive_window_still_configures_expiry(monkeypatch):
    _, captured = _reload_with(monkeypatch, {"CHATBOT_CHECKPOINT_TTL_MINUTES": "90"})
    assert captured["ttl"] == {"default_ttl": 90, "refresh_on_read": True}


def test_the_checkpoint_never_expires_before_the_frame_it_describes(monkeypatch):
    """The QueryFrame describes a conversation the graph replays from the checkpoint.
    A checkpoint that expired FIRST would leave the frame referring to something no
    longer there. Both windows slide on every turn, so equal is fine — shorter is not."""
    from chatbot.memory import store as S
    module, _ = _reload_with(monkeypatch, {"CHATBOT_CHECKPOINT_TTL_MINUTES": None})
    assert module._CHECKPOINT_TTL_MINUTES * 60 >= S._TTL_SECS
