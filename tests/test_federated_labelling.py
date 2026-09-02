"""Tests for federated failure labelling + strict-mode surfacing (routing gaps #2/#3).

`_labelled_failure` is pure; the strict-mode surfacing is checked through _run_coordinator with an
injected run_federated. Run: `python tests/test_federated_labelling.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

from query.federated_route import _labelled_failure  # noqa: E402


class _C:
    def __init__(self, sid):
        self.source_id = sid


# ── _labelled_failure (pure) ────────────────────────────────────────────────────
def test_transient_failure_is_retryable():
    p = _labelled_failure({"status": "refused_federated", "reason": "connection timed out"},
                          [_C("5"), _C("7")])
    assert p["failure_class"] == "transient" and p["retryable"] is True
    assert p["sources"] == ["5", "7"] and p["partial"] is False and p["required_sources_all"] is True


def test_permanent_failure_not_retryable():
    p = _labelled_failure({"status": "exec_error_federated",
                           "reason": "referenced column does not exist", "sources": ["5", "7"]}, [])
    assert p["failure_class"] == "permanent" and p["retryable"] is False


def test_ok_payload_untouched():
    p = _labelled_failure({"status": "ok", "result": {}}, [])
    assert "failure_class" not in p and "retryable" not in p


# ── strict-mode surfacing via the coordinator ────────────────────────────────────
def _coord_setup():
    import config
    import veda_hybrid
    import query.source_coordinator as SC
    from context import RequestContext, set_context
    from query.routing_contracts import RoutingDecision, STATUS_ROUTED, MODE_MULTI, RC_RELATIONSHIP_EDGE, CandidateSource
    config.MULTISOURCE_ROUTING_ENABLED = True
    config.MULTISOURCE_ROUTING_SHADOW = False
    set_context(RequestContext(source_id=5, tenant="default", source_ids=(5, 7)))
    SC.plan_route = lambda q, sids, **k: RoutingDecision(
        status=STATUS_ROUTED, mode=MODE_MULTI, source_ids=["5", "7"], reason_code=RC_RELATIONSHIP_EDGE,
        candidate_sources=[CandidateSource("5", source_type="relational"),
                           CandidateSource("7", source_type="datalake")])
    return veda_hybrid


def test_genuine_join_failure_is_surfaced_not_silent():
    import query.federated_route as FR
    vh = _coord_setup()
    FR.run_federated = lambda q, tenant, source_ids, verbose=False: {
        "status": "refused_federated", "reason": "referenced column does not exist",
        "sources": ["5", "7"], "failure_class": "permanent", "retryable": False}
    mr = vh._run_coordinator("join query")
    # surfaced as a refusal/error — NOT None (which would silently fall back to a single source)
    assert mr is not None and mr.items[0].status in ("refused", "error")


def test_genuine_join_success_flows():
    import query.federated_route as FR
    vh = _coord_setup()
    FR.run_federated = lambda q, tenant, source_ids, verbose=False: {
        "status": "ok", "result": {"columns": ["rev"], "rows": [[100]]},
        "answer": "combined", "sources": ["5", "7"], "provenance": []}
    mr = vh._run_coordinator("join query")
    assert mr is not None


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
