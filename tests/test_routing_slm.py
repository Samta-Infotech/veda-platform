"""Tests for query.routing_slm — bounded SLM ambiguity resolver + validator (routing Phase 3.4/3.5).

SLM call is injected, so no live model. Run: `pytest tests/test_routing_slm.py`.
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORE = os.path.abspath(os.path.join(ROOT, "veda_core"))
sys.path.insert(0, CORE)

from query.routing_contracts import CandidateSource  # noqa: E402
from query.routing_slm import resolve_ambiguity, validate_slm_decision  # noqa: E402


def _cands():
    return [
        CandidateSource(source_id="5", source_type="relational", domain_tags=["finance"],
                        evidence_summary={"columns": ["revenue"]}),
        CandidateSource(source_id="7", source_type="datalake", domain_tags=["product"],
                        evidence_summary={"columns": ["events"]}),
    ]


IDS = {"5", "7"}


# ── validator (pure) ───────────────────────────────────────────────────────────
def test_validator_accepts_valid():
    # new contract (decision / selected_source_ids / NONE)
    assert validate_slm_decision({"decision": "SINGLE", "selected_source_ids": ["5"]}, IDS)[0]
    assert validate_slm_decision({"decision": "MULTI", "selected_source_ids": ["5", "7"]}, IDS)[0]
    assert validate_slm_decision({"decision": "NONE", "selected_source_ids": []}, IDS)[0]
    # backward-compatible old keys (mode / source_ids) still accepted for SINGLE/MULTI
    assert validate_slm_decision({"mode": "SINGLE", "source_ids": ["5"]}, IDS)[0]


def test_validator_rejects_bad():
    assert not validate_slm_decision({"decision": "SINGLE", "selected_source_ids": ["9"]}, IDS)[0]  # non-candidate
    assert not validate_slm_decision({"decision": "MULTI", "selected_source_ids": ["5"]}, IDS)[0]   # count
    assert not validate_slm_decision({"decision": "SINGLE", "selected_source_ids": ["5", "7"]}, IDS)[0]
    assert not validate_slm_decision({"decision": "NONE", "selected_source_ids": ["5"]}, IDS)[0]     # NONE must be empty
    assert not validate_slm_decision({"decision": "BOGUS"}, IDS)[0]
    assert not validate_slm_decision("not json", IDS)[0]


def test_none_becomes_no_match():
    from query.routing_slm import resolve_boundary
    d = resolve_boundary("book a flight", _cands(),
                         slm_call=lambda s, u: '{"decision":"NONE","selected_source_ids":[],"reason":"out of scope"}')
    assert d.status == "NO_MATCH" and d.decision_method == "slm"


# ── resolve_ambiguity (injected SLM) ────────────────────────────────────────────
def test_valid_single():
    d = resolve_ambiguity("rev?", _cands(),
                          slm_call=lambda s, u: '{"mode":"SINGLE","source_ids":["5"],"reason":"finance"}')
    assert d.status == "ROUTED" and d.mode == "SINGLE" and d.source_ids == ["5"]
    assert d.decision_method == "slm" and d.validation_status == "passed"


def test_valid_multi_with_fenced_json():
    d = resolve_ambiguity("rev?", _cands(),
                          slm_call=lambda s, u: 'ok:\n```json\n{"mode":"MULTI","source_ids":["5","7"]}\n```')
    assert d.mode == "MULTI" and set(d.source_ids) == {"5", "7"}


def test_hallucinated_source_becomes_clarify():
    d = resolve_ambiguity("rev?", _cands(),
                          slm_call=lambda s, u: '{"mode":"SINGLE","source_ids":["crm_99"]}')
    assert d.status == "CLARIFICATION_REQUIRED" and d.reason_code == "INVALID_SLM_DECISION"


def test_explicit_clarify():
    d = resolve_ambiguity("rev?", _cands(),
                          slm_call=lambda s, u: '{"mode":"CLARIFY","reason":"ambiguous"}')
    assert d.status == "CLARIFICATION_REQUIRED"


def test_slm_error_becomes_clarify_not_crash():
    def boom(s, u):
        raise RuntimeError("slm down")
    d = resolve_ambiguity("rev?", _cands(), slm_call=boom)
    assert d.status == "CLARIFICATION_REQUIRED"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except Exception:
            failed += 1; print("FAIL", fn.__name__); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
