"""Tests for query.capability_observation — Capability Planning shadow observation (Phase C1).
Pure, no DB, no SLM. Every test proves OBSERVE-ONLY: no candidate mutation, no filtering."""
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORE = os.path.abspath(os.path.join(ROOT, "veda_core"))
sys.path.insert(0, CORE)

import query.capability_observation as CO  # noqa: E402
from query.query_requirements import QueryRequirements  # noqa: E402
from query.routing_contracts import CandidateSource  # noqa: E402


def _c(sid, typ):
    return CandidateSource(source_id=sid, source_type=typ)


# ── comparison scenarios (requirement C in the task) ─────────────────────────────────────────
def test_aggregation_required_source_supports():
    obs = CO.observe_candidate_capabilities(_c("5", "relational"),
                                             QueryRequirements(requires_aggregation=True, aggregate_type="count"))
    assert obs.compatible is True and obs.incompatibilities == []


def test_aggregation_required_source_lacks():
    obs = CO.observe_candidate_capabilities(_c("9", "document"),
                                             QueryRequirements(requires_aggregation=True, aggregate_type="count"))
    assert obs.compatible is False
    assert any("AGGREGATION" in msg for msg in obs.incompatibilities)


def test_temporal_requirement_never_flagged_no_matching_capability_exists():
    """Documented gap: no TEMPORAL capability exists in the Phase A model, so requires_temporal
    must never be silently treated as incompatible against ANY source kind."""
    for kind in ("relational", "datalake", "document", "nosql"):
        obs = CO.observe_candidate_capabilities(
            _c("1", kind), QueryRequirements(requires_temporal=True, temporal_requirement="last 6 months"))
        assert obs.compatible is True and obs.incompatibilities == []


def test_fully_compatible_no_requirements():
    obs = CO.observe_candidate_capabilities(_c("1", "document"), QueryRequirements())
    assert obs.compatible is True and obs.incompatibilities == []


def test_missing_capability_document_source_for_aggregation_with_type():
    obs = CO.observe_candidate_capabilities(_c("1", "document"),
                                             QueryRequirements(requires_aggregation=True, aggregate_type="sum"))
    assert obs.compatible is False
    assert obs.capabilities.source_kind == "document"
    assert obs.requirements.aggregate_type == "sum"


def test_unknown_source_kind_gets_empty_capabilities_and_is_flagged_incompatible():
    obs = CO.observe_candidate_capabilities(_c("1", "future_kind"),
                                             QueryRequirements(requires_aggregation=True))
    assert obs.compatible is False   # empty capability set can't satisfy any positive requirement


# ── no-mutation proof ──────────────────────────────────────────────────────────────────────────
def test_observe_does_not_mutate_candidate():
    c = _c("5", "relational")
    before = (c.source_id, c.source_type, c.presence_tier, c.is_canonical)
    CO.observe_candidate_capabilities(c, QueryRequirements(requires_aggregation=True))
    after = (c.source_id, c.source_type, c.presence_tier, c.is_canonical)
    assert before == after


def test_shadow_run_does_not_reorder_or_filter_or_mutate_candidates(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "CAPABILITY_PLANNING_SHADOW_ENABLED", True)
    candidates = [_c("5", "relational"), _c("9", "document"), _c("7", "datalake")]
    before_ids = [c.source_id for c in candidates]
    before_len = len(candidates)
    CO.run_capability_planning_shadow("how many projects", candidates)
    assert [c.source_id for c in candidates] == before_ids   # order + membership unchanged
    assert len(candidates) == before_len                      # nothing added/removed


# ── flag behavior ───────────────────────────────────────────────────────────────────────────────
def test_shadow_run_flag_off_never_logs(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "CAPABILITY_PLANNING_SHADOW_ENABLED", False)
    called = {"n": 0}
    monkeypatch.setattr(CO, "_log_shadow_observation", lambda *a, **kw: called.__setitem__("n", called["n"] + 1))
    CO.run_capability_planning_shadow("how many projects", [_c("5", "relational")])
    assert called["n"] == 0


def test_shadow_run_flag_on_logs_once(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "CAPABILITY_PLANNING_SHADOW_ENABLED", True)
    called = {"n": 0, "requirements": None, "observations": None}

    def fake_log(query, requirements, observations):
        called["n"] += 1
        called["requirements"] = requirements
        called["observations"] = observations

    monkeypatch.setattr(CO, "_log_shadow_observation", fake_log)
    candidates = [_c("5", "relational"), _c("9", "document")]
    CO.run_capability_planning_shadow("how many projects", candidates)
    assert called["n"] == 1
    assert called["requirements"].requires_aggregation is True
    assert len(called["observations"]) == 2


def test_shadow_run_swallows_exceptions_and_never_raises(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "CAPABILITY_PLANNING_SHADOW_ENABLED", True)

    def boom(*a, **kw):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(CO, "observe_candidate_capabilities", boom)
    # must not raise, even though the flag is on and the internal comparison is broken
    CO.run_capability_planning_shadow("how many projects", [_c("5", "relational")])
