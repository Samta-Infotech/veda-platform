"""Tests for the unified cross-source OPERATION classifier (routing Phase 2, flag-gated).

The SLM makes only a bounded, closed-set CHOICE; its label is validated deterministically against the
structural context (relationship / documents / data-source count). An out-of-set, unparseable, or
structurally-infeasible choice degrades to UNSUPPORTED — never guessed into execution. Dispatch of a
classified operation reuses the existing deterministic planners and, crucially, NEVER falls through to
free-form SQL. Run: `python tests/test_operation_classifier.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

from query import operation_classifier as OC  # noqa: E402
from query.operation_classifier import (  # noqa: E402
    classify_operation, OperationContext, FEDERATED_OPS, ALL_OPS,
    OP_SEMI_JOIN, OP_AGG_AFTER_JOIN, OP_SET_INTERSECTION, OP_DOC_TO_STRUCTURED,
    OP_LOOKUP_ENRICH, OP_UNSUPPORTED,
)


def _fake_slm(reply):
    """Install a stub call_slm that returns `reply` verbatim; returns a restore fn."""
    orig = OC.call_slm
    OC.call_slm = lambda *a, **k: reply
    return lambda: setattr(OC, "call_slm", orig)


_REL = OperationContext(has_relationship=True, has_documents=False, data_source_count=2)
_NO_REL = OperationContext(has_relationship=False, has_documents=False, data_source_count=2)
_DOCS = OperationContext(has_relationship=False, has_documents=True, data_source_count=1)


# ── classification + deterministic validation ────────────────────────────────────────────────────

def test_valid_semi_join_with_relationship():
    restore = _fake_slm('{"operation": "SEMI_JOIN", "reason": "X that also have Y"}')
    try:
        d = classify_operation("which assets have maintenance tickets", _REL)
        assert d.operation == OP_SEMI_JOIN and d.valid is True and d.method == "slm"
    finally:
        restore()


def test_infeasible_op_without_relationship_downgrades():
    # SLM picks SEMI_JOIN but no cross-source relationship exists → UNSUPPORTED (feasibility gate).
    restore = _fake_slm('{"operation": "SEMI_JOIN", "reason": "..."}')
    try:
        d = classify_operation("which assets have maintenance tickets", _NO_REL)
        assert d.operation == OP_UNSUPPORTED and d.valid is False and d.method == "deterministic"
    finally:
        restore()


def test_aggregate_after_join_valid():
    restore = _fake_slm('{"operation": "AGGREGATE_AFTER_JOIN", "reason": "count per city"}')
    try:
        d = classify_operation("maintenance cost per city", _REL)
        assert d.operation == OP_AGG_AFTER_JOIN and d.valid is True
    finally:
        restore()


def test_explicit_unsupported_is_valid_outcome():
    restore = _fake_slm('{"operation": "UNSUPPORTED", "reason": "no single op fits"}')
    try:
        d = classify_operation("something weird across sources", _REL)
        assert d.operation == OP_UNSUPPORTED  # UNSUPPORTED is always a legitimate, safe outcome
    finally:
        restore()


def test_out_of_set_operation_downgrades():
    # DOC_TO_STRUCTURED is NOT in FEDERATED_OPS → offering only FEDERATED_OPS, it must be rejected.
    restore = _fake_slm('{"operation": "DOC_TO_STRUCTURED", "reason": "..."}')
    try:
        d = classify_operation("q", _REL, allowed_ops=FEDERATED_OPS)
        assert d.operation == OP_UNSUPPORTED and d.valid is False
    finally:
        restore()


def test_invalid_json_downgrades():
    restore = _fake_slm("not json at all")
    try:
        d = classify_operation("q", _REL)
        assert d.operation == OP_UNSUPPORTED and d.valid is False
    finally:
        restore()


def test_fenced_json_is_parsed():
    restore = _fake_slm('```json\n{"operation": "SET_INTERSECTION", "reason": "both"}\n```')
    try:
        d = classify_operation("which products are on both lists", _REL)
        assert d.operation == OP_SET_INTERSECTION and d.valid is True
    finally:
        restore()


def test_slm_exception_is_safe():
    orig = OC.call_slm
    def boom(*a, **k):
        raise RuntimeError("slm down")
    OC.call_slm = boom
    try:
        d = classify_operation("q", _REL)
        assert d.operation == OP_UNSUPPORTED and d.valid is False   # degrades safely, never raises
    finally:
        OC.call_slm = orig


def test_federated_ops_excludes_doc_to_structured():
    assert OP_DOC_TO_STRUCTURED not in FEDERATED_OPS
    assert {OP_SEMI_JOIN, OP_AGG_AFTER_JOIN, OP_SET_INTERSECTION, OP_LOOKUP_ENRICH} <= FEDERATED_OPS
    assert OP_UNSUPPORTED in ALL_OPS


# ── expressiveness guard (negation / count-threshold → UNSUPPORTED) ───────────────────────────────

def test_negation_refused_for_semi_join():
    # "assets that do NOT have tickets" needs an anti-join — no supported op expresses it → UNSUPPORTED.
    restore = _fake_slm('{"operation": "SEMI_JOIN", "reason": "membership"}')
    try:
        d = classify_operation("assets that do not have any tickets", _REL)
        assert d.operation == OP_UNSUPPORTED and d.valid is False and d.method == "deterministic"
    finally:
        restore()


def test_count_threshold_refused_for_semi_join():
    # "more than 5 tickets" needs HAVING — semi-join is existence-only → UNSUPPORTED (not silently dropped).
    restore = _fake_slm('{"operation": "SEMI_JOIN", "reason": "membership"}')
    try:
        d = classify_operation("assets with more than 5 tickets", _REL)
        assert d.operation == OP_UNSUPPORTED and d.valid is False
    finally:
        restore()


def test_without_keyword_refused():
    restore = _fake_slm('{"operation": "SET_INTERSECTION", "reason": "both"}')
    try:
        d = classify_operation("cities with assets but without any tickets", _REL)
        assert d.operation == OP_UNSUPPORTED and d.valid is False
    finally:
        restore()


def test_plain_query_not_over_refused():
    # No negation / threshold → the guard stays silent; a clean SEMI_JOIN survives.
    restore = _fake_slm('{"operation": "SEMI_JOIN", "reason": "membership"}')
    try:
        d = classify_operation("which assets have maintenance tickets", _REL)
        assert d.operation == OP_SEMI_JOIN and d.valid is True
    finally:
        restore()


# ── dispatch NEVER reaches free-form SQL ──────────────────────────────────────────────────────────

def test_dispatch_unsupported_refuses_never_free_form():
    import query.federated_route as fr
    payload = fr._dispatch_classified_operation(
        OP_UNSUPPORTED, query="q", by_source={}, hints=[], kinds={}, rel_sid=None,
        schema_text="", join_text="", cols=[], chunks=[], tenant="default")
    assert isinstance(payload, dict) and payload.get("status") == "refused_federated"
    assert payload.get("operation") == OP_UNSUPPORTED


def test_dispatch_lookup_enrich_uses_structured_plan():
    # LOOKUP_ENRICH routes to the structured-plan planner (reused). When it can't build a plan, dispatch
    # REFUSES — never free-form SQL. Stub the planner so no real SLM is called.
    import query.federated_route as fr
    orig = fr._generate_structured_plan
    fr._generate_structured_plan = lambda *a, **k: None
    try:
        payload = fr._dispatch_classified_operation(
            OP_LOOKUP_ENRICH, query="show vendors with asset count", by_source={"2": {}}, hints=[],
            kinds={}, rel_sid=None, schema_text="", join_text="", cols=[], chunks=[], tenant="default")
        assert payload.get("status") == "refused_federated" and payload.get("operation") == OP_LOOKUP_ENRICH
    finally:
        fr._generate_structured_plan = orig


def test_dispatch_semi_join_planner_none_refuses():
    # When the deterministic planner can't build a plan, dispatch REFUSES — it does not fall to SQL.
    import query.federated_route as fr
    import query.semi_join_planner as sj
    orig = sj.plan_semi_join
    sj.plan_semi_join = lambda *a, **k: None
    try:
        payload = fr._dispatch_classified_operation(
            OP_SEMI_JOIN, query="q", by_source={"2": {}}, hints=[], kinds={}, rel_sid=None,
            schema_text="", join_text="", cols=[], chunks=[], tenant="default")
        assert payload.get("status") == "refused_federated" and payload.get("operation") == OP_SEMI_JOIN
    finally:
        sj.plan_semi_join = orig


if __name__ == "__main__":
    import traceback
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print("PASS", name)
        except Exception:
            failed += 1; print("FAIL", name); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
