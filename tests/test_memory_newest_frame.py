"""Memory is READ under the same key it is WRITTEN — the frame's source, not source_ids[0].

memory_write_node files a frame under the source that ANSWERED (explain.sources).
memory_read_node used to read only the turn's nominal source, source_ids[0], so in a
multi-source scope a turn answered by any other source left a frame the next turn could
not see. These drive the two real nodes against the real MemoryStore code (in-memory
Redis, tests/tools/fake_redis.py).

Run: pytest tests/test_memory_newest_frame.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import pytest                                         # noqa: E402

from chatbot import nodes                             # noqa: E402
from chatbot.memory.store import MemoryStore          # noqa: E402
from tools import fake_redis                          # noqa: E402

T, S = "t", "sess"


@pytest.fixture(autouse=True)
def _redis(monkeypatch):
    return fake_redis.install(monkeypatch)


def _answered(source, table="vendors", city=None):
    return {
        "status": "answered", "table": table, "source_id": source,
        "sql": f"SELECT COUNT(*) FROM {table}", "cols": ["n"], "rows": [[4]],
        "explain": {"data_used": {"datasets": [table.title()]},
                    "filters": {"applied": ([{"field": "City", "column": "city",
                                              "operator": "equals", "value": city}]
                                            if city else [])},
                    "operations": [{"type": "count", "summary": "Count"}],
                    "sources": [{"id": str(source)}]},
        "analytics": {"column_stats": [], "available_dimensions": ["city"]},
    }


def _write(message, result, *, scope, frame=None, memory_source_id=None):
    state = {"tenant": T, "session_id": S, "message": message, "status": "answered",
             "engine_result": result, "source_id": scope[0], "source_ids": scope,
             "frame": frame or {}, "drill_stack": [], "delta_type": "new_topic",
             "memory_source_id": memory_source_id}
    return nodes.memory_write_node(state)


def _read(scope):
    return nodes.memory_read_node({"tenant": T, "session_id": S, "message": "and then",
                                   "source_id": scope[0], "source_ids": scope})


def test_frame_answered_by_the_non_primary_source_is_read_back():
    scope = [2, 3]
    _write("how many vendors are there", _answered(3), scope=scope)
    assert MemoryStore.read_frame(T, S, 2) is None           # nothing under the primary

    out = _read(scope)
    assert out["frame"].get("entity") == "vendors", "turn 2 did not see turn 1's frame"
    assert str(out["frame"]["source_id"]) == "3"
    assert out["memory_source_id"] == "3"


def test_the_newest_frame_wins_when_several_sources_hold_one():
    scope = [2, 3]
    _write("how many vendors are there", _answered(3, "vendors"), scope=scope)
    _write("how many assets are there", _answered(2, "assets"), scope=scope)
    assert _read(scope)["frame"]["entity"] == "assets"

    _write("vendors again", _answered(3, "vendors", city="Pune"), scope=scope)
    out = _read(scope)
    assert out["frame"]["entity"] == "vendors"
    assert out["frame"]["filters"][0]["value"] == "Pune"


def test_unstamped_legacy_frames_fall_back_to_the_active_source(_redis):
    """Frames written before `written_at` existed carry no stamp. Among those the source
    that answered last (the active pointer) wins — what a single read returned before."""
    import json
    for sid, ent in ((2, "assets"), (3, "vendors")):
        _redis.set(f"veda:mem:{T}:{S}:src:{sid}:frame",
                   json.dumps({"version": 1, "entity": ent, "source_id": sid, "filters": []}))
    _redis.set(f"veda:mem:{T}:{S}:active", "3")
    assert _read([2, 3])["frame"]["entity"] == "vendors"
    _redis.set(f"veda:mem:{T}:{S}:active", "2")
    assert _read([2, 3])["frame"]["entity"] == "assets"


def test_a_write_to_another_sources_key_is_not_aborted_by_the_version_check():
    """prev_frame's version describes the key it was read from. A turn answered by a
    different source writes another key, whose stored version is unrelated — checking
    against it discarded the write, so that source never got a frame."""
    scope = [2, 3]
    for _ in range(3):                                     # source 2's key: version 3
        prev = _read(scope)["frame"]
        _write("assets", _answered(2, "assets"), scope=scope, frame=prev,
               memory_source_id="2")
    _write("vendors", _answered(3, "vendors"), scope=scope)   # source 3's key: version 1

    prev = _read(scope)                                    # newest: source 3, version 1
    assert prev["frame"]["entity"] == "vendors" and prev["memory_source_id"] == "3"
    _write("assets again", _answered(2, "assets", city="Kochi"), scope=scope,
           frame=prev["frame"], memory_source_id=prev["memory_source_id"])
    back = MemoryStore.read_frame(T, S, 2)
    assert back["filters"] and back["filters"][0]["value"] == "Kochi", \
        "the write to source 2's key was aborted by source 3's frame version"


def test_revoked_source_is_still_discarded():
    """_frame_still_authorised keeps working on whichever frame the read chose."""
    _write("how many vendors are there", _answered(3), scope=[2, 3])
    out = _read([2])                       # source 3 no longer in scope
    assert out["frame"] == {} or out["frame"].get("entity") is None
