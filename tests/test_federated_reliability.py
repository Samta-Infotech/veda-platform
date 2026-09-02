import query.federated_route as fr
import query.source_coordinator as sc
import traceback
"""Tests for federated-route bounded transient-retry (query/reliability.py, flag-gated).

The federated path already LABELS a failure `retryable` (federated_route._labelled_failure) but,
unlike the single/independent agent paths, never RETRIED it. `execute_federated_reliably` closes that
asymmetry: it re-runs run_federated ONLY on a transient failure the payload itself labels, bounded and
flag-gated (default-OFF → single pass-through). Run: `python tests/test_federated_reliability.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

import config as _config  # noqa: E402
from query import reliability as R  # noqa: E402


def _runner(payloads):
    """Return a run() that yields the given payloads in order, then repeats the last."""
    seq = list(payloads)
    state = {"i": 0}

    def run():
        i = min(state["i"], len(seq) - 1)
        state["i"] += 1
        return seq[i]
    return run, state


# ---- federated_transient (label reader) ----------------------------------------------------------

def test_transient_label_true():
    assert R.federated_transient(
        {"status": "exec_error_federated", "retryable": True}) is True


def test_permanent_label_false():
    assert R.federated_transient(
        {"status": "refused_federated", "retryable": False}) is False


def test_ok_and_none_never_transient():
    assert R.federated_transient({"status": "ok"}) is False
    assert R.federated_transient(None) is False


def test_missing_label_falls_back_to_classify():
    # No `retryable` key → classify the reason string. "timeout" → transient.
    assert R.federated_transient(
        {"status": "exec_error_federated", "reason": "connection timed out"}) is True
    assert R.federated_transient(
        {"status": "exec_error_federated", "reason": "syntax error near FROM"}) is False


# ---- execute_federated_reliably (bounded retry) --------------------------------------------------

def test_flag_off_is_passthrough():
    _config.FEDERATED_TRANSIENT_RETRY_ENABLED = False
    run, state = _runner([{"status": "exec_error_federated", "retryable": True},
                          {"status": "ok"}])
    res = R.execute_federated_reliably(run, enabled=None)  # None → read config (OFF)
    assert res.get("status") == "exec_error_federated"     # no retry
    assert state["i"] == 1                                 # ran exactly once


def test_transient_retries_until_ok():
    try:
        _config.FEDERATED_TRANSIENT_RETRY_ENABLED = True
        _config.FEDERATED_MAX_RETRIES = 2
        run, state = _runner([{"status": "exec_error_federated", "retryable": True},
                              {"status": "ok"}])
        res = R.execute_federated_reliably(run)
        assert res.get("status") == "ok"
        assert res.get("retry_attempts") == 1
        assert state["i"] == 2                             # initial + 1 retry
    finally:
        _config.FEDERATED_TRANSIENT_RETRY_ENABLED = False
        _config.FEDERATED_MAX_RETRIES = 1


def test_permanent_not_retried():
    try:
        _config.FEDERATED_TRANSIENT_RETRY_ENABLED = True
        _config.FEDERATED_MAX_RETRIES = 3
        run, state = _runner([{"status": "refused_federated", "retryable": False},
                              {"status": "ok"}])
        res = R.execute_federated_reliably(run)
        assert res.get("status") == "refused_federated"    # permanent → no retry
        assert state["i"] == 1
    finally:
        _config.FEDERATED_TRANSIENT_RETRY_ENABLED = False
        _config.FEDERATED_MAX_RETRIES = 1


def test_retry_exhausted_returns_last_failure():
    try:
        _config.FEDERATED_TRANSIENT_RETRY_ENABLED = True
        _config.FEDERATED_MAX_RETRIES = 2
        run, state = _runner([{"status": "exec_error_federated", "retryable": True}])  # always fails
        res = R.execute_federated_reliably(run)
        assert res.get("status") == "exec_error_federated"  # clean surface of the failure
        assert res.get("retry_attempts") == 2               # bounded — stopped at max
        assert state["i"] == 3                              # initial + 2 retries
    finally:
        _config.FEDERATED_TRANSIENT_RETRY_ENABLED = False
        _config.FEDERATED_MAX_RETRIES = 1


def test_none_passthrough_not_federated():
    try:
        _config.FEDERATED_TRANSIENT_RETRY_ENABLED = True
        _config.FEDERATED_MAX_RETRIES = 2
        run, state = _runner([None])                        # 'not federated — use normal path'
        assert R.execute_federated_reliably(run) is None
        assert state["i"] == 1                              # never retried
    finally:
        _config.FEDERATED_TRANSIENT_RETRY_ENABLED = False
        _config.FEDERATED_MAX_RETRIES = 1


# ---- end-to-end wiring at the REAL dispatch site (fault-injection) -------------------------------

def test_delegate_retries_end_to_end():
    """The coordinator's real _federated_delegate must actually retry a transient federated failure
    (not just the unit helper). Inject a transient-then-ok run_federated and assert recovery."""
    calls = {"n": 0}

    def fake(query, tenant, source_ids):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"status": "exec_error_federated", "retryable": True, "reason": "connection reset"}
        return {"status": "ok", "result": {"rows": [[1]], "columns": ["c"]}}

    orig = fr.run_federated
    fr.run_federated = fake
    _config.FEDERATED_TRANSIENT_RETRY_ENABLED = True
    _config.FEDERATED_MAX_RETRIES = 2
    try:
        res = sc._federated_delegate("q", "default", ["2", "4"])
        assert res.get("status") == "ok"
        assert calls["n"] == 2                              # initial + 1 retry, end-to-end
    finally:
        fr.run_federated = orig
        _config.FEDERATED_TRANSIENT_RETRY_ENABLED = False
        _config.FEDERATED_MAX_RETRIES = 1


def test_delegate_flag_off_no_retry_end_to_end():
    """Flag OFF → the real dispatch site is a pure pass-through (byte-identical to before)."""
    calls = {"n": 0}

    def fake(query, tenant, source_ids):
        calls["n"] += 1
        return {"status": "exec_error_federated", "retryable": True, "reason": "connection reset"}

    orig = fr.run_federated
    fr.run_federated = fake
    _config.FEDERATED_TRANSIENT_RETRY_ENABLED = False
    try:
        res = sc._federated_delegate("q", "default", ["2", "4"])
        assert res.get("status") == "exec_error_federated"
        assert calls["n"] == 1                              # never retried when OFF
    finally:
        fr.run_federated = orig


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
