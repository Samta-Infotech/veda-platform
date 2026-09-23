"""Tests for query.capability_filter — Phase C2 narrow aggregation capability filtering.
Pure, no DB, no SLM. Every test proves scope/safety: aggregation-only, no mutation, order
preserved, never returns an empty list."""
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORE = os.path.abspath(os.path.join(ROOT, "veda_core"))
sys.path.insert(0, CORE)

import query.capability_filter as CF  # noqa: E402
from query.routing_contracts import CandidateSource  # noqa: E402


def _c(sid, typ):
    return CandidateSource(source_id=sid, source_type=typ)


# ── 1. Flag OFF → exact original candidates unchanged ────────────────────────────────────────
def test_flag_off_returns_same_object_unchanged(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "CAPABILITY_FILTERING_ENABLED", False)
    candidates = [_c("5", "relational"), _c("9", "document")]
    result = CF.filter_candidates_by_aggregation_capability("how many projects", candidates)
    assert result is candidates   # identity, not just equality


# ── 2. Aggregation query + incompatible candidate → removed ──────────────────────────────────
def test_aggregation_query_removes_incompatible_candidate(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "CAPABILITY_FILTERING_ENABLED", True)
    candidates = [_c("5", "relational"), _c("9", "document")]
    result = CF.filter_candidates_by_aggregation_capability("how many projects do we have", candidates)
    assert [c.source_id for c in result] == ["5"]


# ── 3. Aggregation query + compatible candidates → retained ───────────────────────────────────
def test_aggregation_query_retains_compatible_candidates(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "CAPABILITY_FILTERING_ENABLED", True)
    candidates = [_c("5", "relational"), _c("7", "datalake")]
    result = CF.filter_candidates_by_aggregation_capability("how many projects", candidates)
    assert [c.source_id for c in result] == ["5", "7"]


# ── 4. Non-aggregation query → nothing filtered ────────────────────────────────────────────────
def test_non_aggregation_query_filters_nothing(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "CAPABILITY_FILTERING_ENABLED", True)
    candidates = [_c("5", "relational"), _c("9", "document")]
    result = CF.filter_candidates_by_aggregation_capability("list all vendors", candidates)
    assert result is candidates   # no requirement -> no-op, same object


# ── 5. Candidate order preserved ──────────────────────────────────────────────────────────────
def test_order_preserved_when_filtering(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "CAPABILITY_FILTERING_ENABLED", True)
    candidates = [_c("9", "document"), _c("7", "datalake"), _c("3", "document"), _c("5", "relational")]
    result = CF.filter_candidates_by_aggregation_capability("how many projects", candidates)
    assert [c.source_id for c in result] == ["7", "5"]   # original relative order kept


# ── 6. Original candidate list not mutated ─────────────────────────────────────────────────────
def test_original_list_not_mutated(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "CAPABILITY_FILTERING_ENABLED", True)
    candidates = [_c("5", "relational"), _c("9", "document")]
    original_ids = [c.source_id for c in candidates]
    CF.filter_candidates_by_aggregation_capability("how many projects", candidates)
    assert [c.source_id for c in candidates] == original_ids   # untouched after the call
    assert len(candidates) == 2


# ── 7. All candidates incompatible → safe fallback to original candidates ─────────────────────
def test_all_incompatible_falls_back_to_original(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "CAPABILITY_FILTERING_ENABLED", True)
    candidates = [_c("9", "document"), _c("3", "document")]
    result = CF.filter_candidates_by_aggregation_capability("how many projects", candidates)
    assert result is candidates   # never empty — falls back, same object
    assert len(result) == 2


def test_fallback_logs_a_warning(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "CAPABILITY_FILTERING_ENABLED", True)
    called = {"n": 0}
    monkeypatch.setattr(CF, "_log_fallback", lambda *a, **kw: called.__setitem__("n", called["n"] + 1))
    candidates = [_c("9", "document")]
    CF.filter_candidates_by_aggregation_capability("how many projects", candidates)
    assert called["n"] == 1


# ── 8. Flag ON genuinely changes the candidate list where expected ────────────────────────────
def test_flag_on_genuinely_changes_list_when_incompatible_present(monkeypatch):
    import config as _cfg
    candidates = [_c("5", "relational"), _c("9", "document")]

    monkeypatch.setattr(_cfg, "CAPABILITY_FILTERING_ENABLED", False)
    off = CF.filter_candidates_by_aggregation_capability("how many projects", candidates)
    monkeypatch.setattr(_cfg, "CAPABILITY_FILTERING_ENABLED", True)
    on = CF.filter_candidates_by_aggregation_capability("how many projects", candidates)

    assert off is candidates
    assert [c.source_id for c in on] == ["5"]
    assert on is not candidates


# ── Exception safety ───────────────────────────────────────────────────────────────────────────
def test_exception_in_requirements_falls_back_to_original(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "CAPABILITY_FILTERING_ENABLED", True)
    monkeypatch.setattr(CF, "derive_requirements", lambda q: (_ for _ in ()).throw(RuntimeError("boom")))
    candidates = [_c("5", "relational")]
    result = CF.filter_candidates_by_aggregation_capability("how many projects", candidates)
    assert result is candidates
