"""Per-source analytical memory — one QueryFrame per source, not one per session.

A frame is evidence harvested from ONE source's executed SQL. With a single
session-wide key, a session that touched two sources kept only the newer frame and the
earlier topic was gone with no signal (live-tested 2026-09-17). These tests pin the
scoping, its backward compatibility, and the narrower blast radius it gives the RBAC
revocation guard.

Runs against the real redis-stack (the same instance the store talks to by default) —
there is no fakeredis in this environment, and the behaviour under test IS the key
layout, which a mock would only restate. Every test uses a unique session id and cleans
up after itself; the whole module skips when Redis is unreachable.

Run: ``pytest tests/test_memory_source_scope.py``
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from chatbot.memory import store as S
from chatbot.memory.store import MemoryStore

try:
    S._client().ping()
except Exception:                                    # noqa: BLE001 — environment, not a failure
    pytest.skip("redis-stack not reachable from here", allow_module_level=True)

TENANT = "test-scope"


def _frame(source_id, city):
    return {"version": 1, "entity": "assets_asset", "entity_display": "Assets",
            "source_id": source_id,
            "filters": [{"field": "City", "operator": "equals", "value": city,
                         "source": "executed_sql"}]}


@pytest.fixture
def session():
    name = f"scope-{uuid.uuid4().hex[:10]}"
    yield name
    MemoryStore.reset(TENANT, name)


# ---------------------------------------------------------------------------
# the scoping itself
# ---------------------------------------------------------------------------

def test_two_sources_keep_two_separate_frames(session):
    MemoryStore.write_frame(TENANT, session, _frame(2, "Pune"), source_id=2)
    MemoryStore.write_frame(TENANT, session, _frame(3, "Mumbai"), source_id=3)

    assert MemoryStore.read_frame(TENANT, session, 2)["filters"][0]["value"] == "Pune"
    assert MemoryStore.read_frame(TENANT, session, 3)["filters"][0]["value"] == "Mumbai"


def test_answering_on_one_source_does_not_erase_the_other(session):
    """Scenario A: work on source 2, switch to source 3, come back to source 2."""
    MemoryStore.write_frame(TENANT, session, _frame(2, "Pune"), source_id=2)
    MemoryStore.write_stack(TENANT, session, [{"dimension": "City", "value": "Pune"}],
                            source_id=2)
    MemoryStore.write_frame(TENANT, session, _frame(3, "Mumbai"), source_id=3)

    back = MemoryStore.read_frame(TENANT, session, 2)
    assert back is not None, "source 2's topic was erased by working on source 3"
    assert back["filters"][0]["value"] == "Pune"
    assert MemoryStore.read_stack(TENANT, session, 2) == [{"dimension": "City",
                                                           "value": "Pune"}]


def test_the_active_pointer_follows_the_last_write(session):
    MemoryStore.write_frame(TENANT, session, _frame(2, "Pune"), source_id=2)
    assert MemoryStore.active_source(TENANT, session) == "2"
    MemoryStore.write_frame(TENANT, session, _frame(3, "Mumbai"), source_id=3)
    assert MemoryStore.active_source(TENANT, session) == "3"


def test_a_read_with_no_source_follows_the_active_pointer(session):
    """What a single-source deployment and the CLI get: the topic last answered."""
    MemoryStore.write_frame(TENANT, session, _frame(2, "Pune"), source_id=2)
    MemoryStore.write_frame(TENANT, session, _frame(3, "Mumbai"), source_id=3)
    assert MemoryStore.read_frame(TENANT, session)["source_id"] == 3


def test_known_sources_lists_every_topic_the_session_holds(session):
    MemoryStore.write_frame(TENANT, session, _frame(2, "Pune"), source_id=2)
    MemoryStore.write_frame(TENANT, session, _frame(4, "Delhi"), source_id=4)
    assert MemoryStore.known_sources(TENANT, session) == ["2", "4"]


# ---------------------------------------------------------------------------
# backward compatibility
# ---------------------------------------------------------------------------

def test_an_unscoped_write_and_read_behave_exactly_as_before(session):
    MemoryStore.write_frame(TENANT, session, _frame(None, "Pune"))
    assert MemoryStore.read_frame(TENANT, session)["filters"][0]["value"] == "Pune"
    assert MemoryStore.active_source(TENANT, session) is None


def test_a_pre_deploy_frame_is_still_found_by_a_scoped_read(session):
    """A conversation that was live across the deploy keeps its context: the scoped key
    is empty, so the read falls back to the key the old code wrote."""
    MemoryStore.write_frame(TENANT, session, _frame(2, "Pune"))      # legacy, unscoped
    assert MemoryStore.read_frame(TENANT, session, 2)["filters"][0]["value"] == "Pune"


def test_a_pre_deploy_frame_from_another_source_is_not_handed_over(session):
    """The legacy key is session-wide, so it may hold a different source's topic.
    Returning that would reintroduce exactly the cross-source bleed this removes."""
    MemoryStore.write_frame(TENANT, session, _frame(2, "Pune"))      # legacy, source 2
    assert MemoryStore.read_frame(TENANT, session, 3) is None


def test_a_scoped_frame_wins_over_a_stale_legacy_one(session):
    MemoryStore.write_frame(TENANT, session, _frame(2, "OLD"))       # legacy
    MemoryStore.write_frame(TENANT, session, _frame(2, "Pune"), source_id=2)
    assert MemoryStore.read_frame(TENANT, session, 2)["filters"][0]["value"] == "Pune"


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_a_scoped_reset_leaves_the_other_source_alone(session):
    """Scenario C: the grant on source 2 is revoked. Source 3's work survives."""
    MemoryStore.write_frame(TENANT, session, _frame(2, "Pune"), source_id=2)
    MemoryStore.write_frame(TENANT, session, _frame(3, "Mumbai"), source_id=3)
    MemoryStore.push_episodic_turn(TENANT, session, "how many assets", "answered: 128")

    MemoryStore.reset(TENANT, session, source_id=2)

    assert MemoryStore.read_frame(TENANT, session, 2) is None
    assert MemoryStore.read_frame(TENANT, session, 3)["filters"][0]["value"] == "Mumbai"
    assert MemoryStore.known_sources(TENANT, session) == ["3"]
    # the conversation buffer is session-wide and is NOT a scoped reset's business
    assert MemoryStore.read_episodic(TENANT, session)


def test_a_scoped_reset_clears_the_active_pointer_only_when_it_pointed_there(session):
    MemoryStore.write_frame(TENANT, session, _frame(2, "Pune"), source_id=2)
    MemoryStore.write_frame(TENANT, session, _frame(3, "Mumbai"), source_id=3)
    MemoryStore.reset(TENANT, session, source_id=2)
    assert MemoryStore.active_source(TENANT, session) == "3"
    MemoryStore.reset(TENANT, session, source_id=3)
    assert MemoryStore.active_source(TENANT, session) is None


def test_a_full_reset_wipes_every_source(session):
    MemoryStore.write_frame(TENANT, session, _frame(2, "Pune"), source_id=2)
    MemoryStore.write_frame(TENANT, session, _frame(3, "Mumbai"), source_id=3)
    MemoryStore.write_frame(TENANT, session, _frame(None, "Legacy"))
    MemoryStore.write_stack(TENANT, session, [{"dimension": "City", "value": "Pune"}],
                            source_id=2)
    MemoryStore.push_episodic_turn(TENANT, session, "q", "answered: 1")

    MemoryStore.reset(TENANT, session)

    assert MemoryStore.read_frame(TENANT, session, 2) is None
    assert MemoryStore.read_frame(TENANT, session, 3) is None
    assert MemoryStore.read_frame(TENANT, session) is None
    assert MemoryStore.read_stack(TENANT, session, 2) == []
    assert MemoryStore.read_episodic(TENANT, session) == []
    assert MemoryStore.known_sources(TENANT, session) == []
    assert MemoryStore.active_source(TENANT, session) is None


# ---------------------------------------------------------------------------
# the optimistic lock still works per source
# ---------------------------------------------------------------------------

def test_the_version_lock_is_per_source(session):
    MemoryStore.write_frame(TENANT, session, {**_frame(2, "Pune"), "version": 1},
                            source_id=2)
    # a writer holding version 1 wins; a second holding the same stale version loses
    assert MemoryStore.write_frame(TENANT, session, {**_frame(2, "Nagpur"), "version": 2},
                                   expected_version=1, source_id=2) is True
    assert MemoryStore.write_frame(TENANT, session, {**_frame(2, "Delhi"), "version": 2},
                                   expected_version=1, source_id=2) is False
    assert MemoryStore.read_frame(TENANT, session, 2)["filters"][0]["value"] == "Nagpur"
    # and a conflict on source 2 says nothing about source 3
    assert MemoryStore.write_frame(TENANT, session, _frame(3, "Mumbai"), source_id=3) is True


def test_an_aborted_write_does_not_move_the_active_pointer(session):
    MemoryStore.write_frame(TENANT, session, {**_frame(2, "Pune"), "version": 1},
                            source_id=2)
    MemoryStore.write_frame(TENANT, session, _frame(3, "Mumbai"), source_id=3)
    assert MemoryStore.active_source(TENANT, session) == "3"
    assert MemoryStore.write_frame(TENANT, session, {**_frame(2, "Delhi"), "version": 9},
                                   expected_version=7, source_id=2) is False
    assert MemoryStore.active_source(TENANT, session) == "3"
