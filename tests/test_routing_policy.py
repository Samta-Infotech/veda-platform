"""Tests for query.routing_policy — deterministic-first source routing (routing Phase 3.2). Pure, no DB."""
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORE = os.path.abspath(os.path.join(ROOT, "veda_core"))
sys.path.insert(0, CORE)

from query.routing_contracts import CandidateSource  # noqa: E402
from query.routing_policy import decide  # noqa: E402


def _c(sid, tier="STRONG", canon=False, tags=None, typ="relational"):
    return CandidateSource(source_id=sid, source_type=typ, presence_tier=tier,
                           is_canonical=canon, domain_tags=tags or [])


def test_no_candidates_is_no_match():
    d, amb = decide([_c("1", "NONE")])
    assert d.status == "NO_MATCH" and d.reason_code == "NO_EVIDENCE" and not amb


def test_single_candidate():
    d, amb = decide([_c("5")])
    assert d.mode == "SINGLE" and d.source_ids == ["5"] and not amb


def test_two_with_edge_is_multi():
    d, amb = decide([_c("5"), _c("7")], {frozenset({"5", "7"})})
    assert d.mode == "MULTI" and set(d.source_ids) == {"5", "7"}
    assert d.reason_code == "RELATIONSHIP_EDGE" and d.relationship_basis and not amb


def test_same_domain_canonical_tiebreak():
    d, amb = decide([_c("5", canon=True, tags=["finance"]), _c("7", tags=["finance"])])
    assert d.mode == "SINGLE" and d.source_ids == ["5"]
    assert d.reason_code == "CANONICAL_SELECTED" and d.canonical_basis["source_id"] == "5" and not amb


def test_unrelated_multi_is_ambiguous():
    d, amb = decide([_c("5", tags=["finance"]), _c("7", tags=["product"])])
    assert d.mode == "NONE" and d.reason_code == "AMBIGUOUS_SOURCE_SELECTION" and amb


def test_strong_preferred_over_weak():
    d, amb = decide([_c("5", "STRONG"), _c("7", "WEAK")])
    assert d.mode == "SINGLE" and d.source_ids == ["5"]


def test_weak_only_still_routes():
    d, amb = decide([_c("5", "WEAK"), _c("7", "WEAK")], {frozenset({"5", "7"})})
    assert d.mode == "MULTI"


def test_two_canonical_same_domain_is_ambiguous():
    # a misconfiguration (two canonicals for one domain) must not be silently guessed
    d, amb = decide([_c("5", canon=True, tags=["finance"]), _c("7", canon=True, tags=["finance"])])
    assert amb and d.reason_code == "AMBIGUOUS_SOURCE_SELECTION"


def test_canonical_without_shared_domain_stays_ambiguous():
    # canonical only breaks ties within the SAME domain; unrelated domains stay ambiguous
    d, amb = decide([_c("5", canon=True, tags=["finance"]), _c("7", tags=["hr"])])
    assert amb


def test_edge_beats_canonical():
    # a genuine join relationship wins over a same-domain canonical pick
    d, amb = decide([_c("5", canon=True, tags=["finance"]), _c("7", tags=["finance"])],
                    {frozenset({"5", "7"})})
    assert d.mode == "MULTI" and not amb


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
