import traceback
"""Tests for confident-SINGLE SLM skip (flag-gated, default OFF).

When exactly ONE source is STRONG (the rest WEAK/NONE), the decision boundary should NOT escalate to the
bounded SLM just because a WEAK runner-up's raw top_score is close — trust the deterministic SINGLE.
Genuine competition (>=2 STRONG) still reaches the SLM. Run: `python tests/test_confident_single_skip.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

import config as _config  # noqa: E402
import query.source_coordinator as SC  # noqa: E402
from query.routing_contracts import CandidateSource, RoutingDecision, STATUS_ROUTED, MODE_SINGLE  # noqa: E402


def _cand(sid, score, tier):
    return CandidateSource(source_id=sid, source_type="", presence_tier=tier, top_score=score)


def _single_decision(sid):
    return RoutingDecision(status=STATUS_ROUTED, mode=MODE_SINGLE, source_ids=[sid],
                           reason_code="SINGLE_CANDIDATE")


# one STRONG (homzhub 0.523) + a WEAK doc whose chunk makes top_score close (0.453).
_ONE_STRONG = [_cand("2", 0.523, "STRONG"), _cand("3", 0.453, "WEAK")]
# two genuinely STRONG sources.
_TWO_STRONG = [_cand("2", 0.520, "STRONG"), _cand("4", 0.500, "STRONG")]


def test_flag_off_close_runner_still_boundaries():
    # OFF: the close WEAK runner-up sends the confident single to the SLM (existing behaviour).
    _config.CONFIDENT_SINGLE_SKIP_SLM_ENABLED = False
    at_boundary, _ = SC._decision_boundary(_ONE_STRONG, _single_decision("2"), ambiguous=False)
    assert at_boundary is True


def test_flag_on_one_strong_skips_slm():
    _config.CONFIDENT_SINGLE_SKIP_SLM_ENABLED = True
    try:
        at_boundary, _ = SC._decision_boundary(_ONE_STRONG, _single_decision("2"), ambiguous=False)
        assert at_boundary is False           # confident SINGLE → no SLM
    finally:
        _config.CONFIDENT_SINGLE_SKIP_SLM_ENABLED = False


def test_flag_on_two_strong_still_boundaries():
    # genuine competition (>=2 STRONG) must STILL reach the SLM even with the flag on.
    _config.CONFIDENT_SINGLE_SKIP_SLM_ENABLED = True
    try:
        at_boundary, _ = SC._decision_boundary(_TWO_STRONG, _single_decision("2"), ambiguous=False)
        assert at_boundary is True
    finally:
        _config.CONFIDENT_SINGLE_SKIP_SLM_ENABLED = False


def test_flag_on_ambiguous_still_boundaries():
    # an AMBIGUOUS decision always reaches the SLM regardless of the flag.
    _config.CONFIDENT_SINGLE_SKIP_SLM_ENABLED = True
    try:
        at_boundary, _ = SC._decision_boundary(_ONE_STRONG, _single_decision("2"), ambiguous=True)
        assert at_boundary is True
    finally:
        _config.CONFIDENT_SINGLE_SKIP_SLM_ENABLED = False


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print("PASS", name)
        except Exception:
            failed += 1; print("FAIL", name); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
